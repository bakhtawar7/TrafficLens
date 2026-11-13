"""Local SQLite storage for image runs and user-reported corrections."""
from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import config

LATEST_SCHEMA_VERSION = 1


def _database_path(path: Path | None) -> Path:
    return config.HISTORY_DB if path is None else Path(path)


class HistoryStoreError(RuntimeError):
    """A readable error for history database failures."""


@dataclass(frozen=True)
class RunWriteResult:
    run_id: str
    inserted: bool


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=1.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 1000")
    return connection


def migrate(path: Path | None = None) -> int:
    """Apply ordered schema migrations and return the current version."""
    path = _database_path(path)
    try:
        with _connect(path) as connection:
            integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise HistoryStoreError(f"History database integrity check failed: {integrity}")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS schema_version "
                "(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            applied = {
                int(row["version"])
                for row in connection.execute("SELECT version FROM schema_version")
            }
            if applied and max(applied) > LATEST_SCHEMA_VERSION:
                raise HistoryStoreError(
                    f"History database schema {max(applied)} is newer than supported "
                    f"schema {LATEST_SCHEMA_VERSION}."
                )
            if 1 not in applied:
                connection.executescript(
                    """
                    CREATE TABLE runs (
                        id TEXT PRIMARY KEY,
                        timestamp_utc TEXT NOT NULL,
                        input_filename TEXT NOT NULL,
                        image_sha256 TEXT NOT NULL,
                        width INTEGER NOT NULL CHECK (width > 0),
                        height INTEGER NOT NULL CHECK (height > 0),
                        settings_sha256 TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        selected_classes TEXT NOT NULL,
                        threshold REAL NOT NULL,
                        rickshaw_mode INTEGER NOT NULL CHECK (rickshaw_mode IN (0, 1)),
                        UNIQUE (
                            image_sha256, settings_sha256, selected_classes,
                            threshold, rickshaw_mode, mode
                        )
                    );
                    CREATE TABLE model_results (
                        run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                        model TEXT NOT NULL,
                        weights_sha256 TEXT,
                        total_count INTEGER NOT NULL CHECK (total_count >= 0),
                        counts_by_class TEXT NOT NULL,
                        inference_ms REAL NOT NULL CHECK (inference_ms >= 0),
                        PRIMARY KEY (run_id, model)
                    );
                    CREATE TABLE reliability (
                        run_id TEXT PRIMARY KEY REFERENCES runs(id) ON DELETE CASCADE,
                        count_range_min INTEGER,
                        count_range_max INTEGER,
                        agreement_count REAL,
                        agreement_box REAL,
                        flag_status TEXT NOT NULL,
                        flag_source TEXT,
                        predicted_error REAL
                    );
                    CREATE TABLE corrections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                        timestamp_utc TEXT NOT NULL,
                        reviewer TEXT NOT NULL,
                        corrected_counts TEXT NOT NULL,
                        note TEXT NOT NULL DEFAULT ''
                    );
                    CREATE INDEX idx_runs_image_sha256 ON runs(image_sha256);
                    CREATE INDEX idx_runs_timestamp_utc ON runs(timestamp_utc);
                    """
                )
                connection.execute(
                    "INSERT INTO schema_version(version, applied_at) VALUES (?, ?)",
                    (1, _utc_now()),
                )
            return max(applied | {1})
    except HistoryStoreError:
        raise
    except sqlite3.DatabaseError as error:
        raise HistoryStoreError(
            "The local history database is unavailable or corrupt. History was not saved; "
            "close other database users and check data/history.db."
        ) from error
    except OSError as error:
        raise HistoryStoreError(
            "The local history database cannot be opened for writing. Check folder "
            "permissions and available disk space."
        ) from error


def record_run(
    run: dict[str, Any],
    model_results: list[dict[str, Any]],
    reliability_record: dict[str, Any] | None = None,
    path: Path | None = None,
) -> RunWriteResult:
    """Insert a run once for its distinct image/settings/control tuple."""
    path = _database_path(path)
    run_id = str(uuid.uuid4())
    try:
        migrate(path)
        with _connect(path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO runs (
                    id, timestamp_utc, input_filename, image_sha256, width, height,
                    settings_sha256, mode, selected_classes, threshold, rickshaw_mode
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    run.get("timestamp_utc", _utc_now()),
                    run["input_filename"],
                    run["image_sha256"],
                    int(run["width"]),
                    int(run["height"]),
                    run["settings_sha256"],
                    run["mode"],
                    json.dumps(run["selected_classes"], separators=(",", ":")),
                    float(run["threshold"]),
                    int(bool(run["rickshaw_mode"])),
                ),
            )
            inserted = cursor.rowcount == 1
            if not inserted:
                existing = connection.execute(
                    """
                    SELECT id FROM runs
                    WHERE image_sha256 = ? AND settings_sha256 = ?
                        AND selected_classes = ? AND threshold = ?
                        AND rickshaw_mode = ? AND mode = ?
                    """,
                    (
                        run["image_sha256"],
                        run["settings_sha256"],
                        json.dumps(run["selected_classes"], separators=(",", ":")),
                        float(run["threshold"]),
                        int(bool(run["rickshaw_mode"])),
                        run["mode"],
                    ),
                ).fetchone()
                if existing is None:
                    raise HistoryStoreError("The duplicate history run could not be located.")
                run_id = str(existing["id"])

            for result in model_results:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO model_results (
                        run_id, model, weights_sha256, total_count,
                        counts_by_class, inference_ms
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        result["model"],
                        result.get("weights_sha256"),
                        int(result["total_count"]),
                        json.dumps(result["counts_by_class"], separators=(",", ":")),
                        float(result["inference_ms"]),
                    ),
                )
            if reliability_record is not None:
                connection.execute(
                    """
                    INSERT INTO reliability (
                        run_id, count_range_min, count_range_max, agreement_count,
                        agreement_box, flag_status, flag_source, predicted_error
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        count_range_min = excluded.count_range_min,
                        count_range_max = excluded.count_range_max,
                        agreement_count = excluded.agreement_count,
                        agreement_box = excluded.agreement_box,
                        flag_status = excluded.flag_status,
                        flag_source = excluded.flag_source,
                        predicted_error = excluded.predicted_error
                    """,
                    (
                        run_id,
                        reliability_record.get("count_range_min"),
                        reliability_record.get("count_range_max"),
                        reliability_record.get("agreement_count"),
                        reliability_record.get("agreement_box"),
                        reliability_record.get("flag_status", "not validated"),
                        reliability_record.get("flag_source"),
                        reliability_record.get("predicted_error"),
                    ),
                )
            connection.commit()
        return RunWriteResult(run_id, inserted)
    except HistoryStoreError:
        raise
    except (sqlite3.DatabaseError, OSError, KeyError, TypeError, ValueError) as error:
        raise HistoryStoreError(
            "The run could not be saved to local history. Check database permissions, "
            "disk space, and the run details; the image count remains available."
        ) from error


def list_runs(
    path: Path | None = None,
    input_filter: str = "",
    model_filter: str = "",
) -> list[dict[str, Any]]:
    """Return matching runs, newest first, with their result and reliability rows."""
    path = _database_path(path)
    try:
        migrate(path)
        with _connect(path) as connection:
            query = """
                SELECT r.*, l.count_range_min, l.count_range_max, l.agreement_count,
                    l.agreement_box, l.flag_status, l.flag_source, l.predicted_error
                FROM runs r LEFT JOIN reliability l ON l.run_id = r.id
                WHERE r.input_filename LIKE ?
                    AND (? = '' OR EXISTS (
                        SELECT 1 FROM model_results m
                        WHERE m.run_id = r.id AND m.model LIKE ?
                    ))
                ORDER BY r.timestamp_utc DESC, r.id DESC
            """
            rows = connection.execute(
                query, (f"%{input_filter}%", model_filter, f"%{model_filter}%")
            ).fetchall()
            return [dict(row) for row in rows]
    except HistoryStoreError:
        raise
    except sqlite3.DatabaseError as error:
        raise HistoryStoreError("History could not be read from data/history.db.") from error


def get_run(run_id: str, path: Path | None = None) -> dict[str, Any] | None:
    """Load a run and its models, signals, and correction history."""
    path = _database_path(path)
    rows = list_runs(path)
    run = next((row for row in rows if row["id"] == run_id), None)
    if run is None:
        return None
    try:
        with _connect(path) as connection:
            run["model_results"] = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM model_results WHERE run_id = ? ORDER BY model",
                    (run_id,),
                )
            ]
            run["corrections"] = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM corrections WHERE run_id = ? ORDER BY id",
                    (run_id,),
                )
            ]
        return run
    except sqlite3.DatabaseError as error:
        raise HistoryStoreError("The selected history run could not be loaded.") from error


def delete_run(run_id: str, path: Path | None = None) -> None:
    """Delete one run; dependent rows are removed by foreign-key cascade."""
    path = _database_path(path)
    try:
        migrate(path)
        with _connect(path) as connection:
            connection.execute("DELETE FROM runs WHERE id = ?", (run_id,))
    except (sqlite3.DatabaseError, OSError) as error:
        raise HistoryStoreError("The selected history run could not be deleted.") from error


def clear_history(path: Path | None = None) -> None:
    """Delete all runs and dependent rows while preserving the database schema."""
    path = _database_path(path)
    try:
        migrate(path)
        with _connect(path) as connection:
            connection.execute("DELETE FROM runs")
    except (sqlite3.DatabaseError, OSError) as error:
        raise HistoryStoreError("History could not be cleared from the local database.") from error


def export_rows(path: Path | None = None) -> list[dict[str, Any]]:
    """Flatten the run/model history into CSV-ready rows."""
    path = _database_path(path)
    try:
        migrate(path)
        with _connect(path) as connection:
            rows = connection.execute(
                """
                SELECT r.*, m.model, m.weights_sha256, m.total_count,
                    m.counts_by_class, m.inference_ms, l.count_range_min,
                    l.count_range_max, l.agreement_count, l.agreement_box,
                    l.flag_status, l.flag_source, l.predicted_error
                FROM runs r
                LEFT JOIN model_results m ON m.run_id = r.id
                LEFT JOIN reliability l ON l.run_id = r.id
                ORDER BY r.timestamp_utc, r.id, m.model
                """
            ).fetchall()
            return [dict(row) for row in rows]
    except sqlite3.DatabaseError as error:
        raise HistoryStoreError("History could not be exported from the local database.") from error
