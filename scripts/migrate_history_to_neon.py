"""One-time, repeatable SQLite -> Neon history migration.

Run from the project root after setting DATABASE_URL in .env:
    python scripts/migrate_history_to_neon.py
The source SQLite database is never modified. A consistent SQLite backup is created
beside it before any rows are copied.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import psycopg2
from psycopg2.extras import RealDictCursor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "data" / "history.db"
TABLES = {
    "runs": ("id", "timestamp_utc", "input_filename", "image_sha256", "width", "height",
             "settings_sha256", "mode", "selected_classes", "threshold", "rickshaw_mode"),
    "model_results": ("run_id", "model", "weights_sha256", "total_count",
                      "counts_by_class", "inference_ms"),
    "reliability": ("run_id", "count_range_min", "count_range_max", "agreement_count",
                    "agreement_box", "flag_status", "flag_source", "predicted_error"),
    "corrections": ("id", "run_id", "timestamp_utc", "reviewer", "corrected_counts", "note"),
}


def main() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    args = parser.parse_args()
    source = args.source.resolve()
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        parser.error("DATABASE_URL is missing. Add it to the project .env file.")
    if not source.is_file():
        parser.error(f"SQLite database not found: {source}")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup = source.with_name(f"{source.stem}.pre-neon-{stamp}{source.suffix}")
    with sqlite3.connect(source) as live, sqlite3.connect(backup) as snapshot:
        live.backup(snapshot)
    print(f"Safety backup created: {backup}")

    with sqlite3.connect(source) as sqlite:
        sqlite.row_factory = sqlite3.Row
        available = {r[0] for r in sqlite.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        required = set(TABLES) | {"schema_version"}
        missing = required - available
        if missing:
            parser.error("SQLite schema is missing expected tables: " + ", ".join(sorted(missing)))
        source_counts = {table: sqlite.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                         for table in TABLES}
        rows = {table: [tuple(row[column] for column in columns)
                        for row in sqlite.execute(f"SELECT {','.join(columns)} FROM {table}")]
                for table, columns in TABLES.items()}

    from core import postgres_history
    postgres_history.migrate()
    inserted = {}
    with psycopg2.connect(database_url) as pg:
        with pg.cursor() as cur:
            for table in ("runs", "model_results", "reliability", "corrections"):
                columns = TABLES[table]
                placeholders = ",".join(["%s"] * len(columns))
                names = ",".join(columns)
                conflict = "ON CONFLICT DO NOTHING"
                cur.executemany(
                    f"INSERT INTO {table} ({names}) VALUES ({placeholders}) {conflict}",
                    rows[table],
                )
                inserted[table] = cur.rowcount
            cur.execute("SELECT setval(pg_get_serial_sequence('corrections','id'), "
                        "GREATEST(COALESCE((SELECT MAX(id) FROM corrections), 1), 1), "
                        "EXISTS(SELECT 1 FROM corrections))")
            cur.execute("SELECT COUNT(*) FROM runs")
            target_runs = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM model_results")
            target_models = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM reliability")
            target_reliability = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM corrections")
            target_corrections = cur.fetchone()[0]

    print("\nMigration finished. Source SQLite was left unchanged.")
    print("Rows in source:", source_counts)
    print("Rows newly inserted (conflicts skipped):", inserted)
    print("Rows now in Neon:", {"runs": target_runs, "model_results": target_models,
          "reliability": target_reliability, "corrections": target_corrections})
    print("Compare source and Neon totals before deploying. Keep the backup and original DB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
