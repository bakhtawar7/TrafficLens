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


def _decode_json(value: str | None, fallback: Any) -> Any:
    if value is None:
        return fallback
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise HistoryStoreError("History contains invalid JSON data and cannot be reviewed.") from error


def review_queue(
    path: Path | None = None,
    filename_filter: str = "",
    queue_filter: str = "All",
) -> list[dict[str, Any]]:
    """Return reviewable runs, flagged first, with model outputs kept out of the list."""
    path = _database_path(path)
    try:
        migrate(path)
        with _connect(path) as connection:
            runs = [
                dict(row)
                for row in connection.execute(
                    """
                    SELECT r.id, r.timestamp_utc, r.input_filename, r.selected_classes,
                        r.threshold, r.rickshaw_mode, l.flag_status, l.flag_source,
                        l.count_range_min, l.count_range_max, l.agreement_count,
                        l.agreement_box
                    FROM runs r LEFT JOIN reliability l ON l.run_id = r.id
                    WHERE r.input_filename LIKE ?
                    """,
                    (f"%{filename_filter}%",),
                )
            ]
            for run in runs:
                run["model_results"] = [
                    dict(row)
                    for row in connection.execute(
                        """
                        SELECT model, total_count, counts_by_class
                        FROM model_results WHERE run_id = ? ORDER BY model
                        """,
                        (run["id"],),
                    )
                ]
                latest = connection.execute(
                    """
                    SELECT id, timestamp_utc, reviewer, corrected_counts, note
                    FROM corrections WHERE run_id = ? ORDER BY id DESC LIMIT 1
                    """,
                    (run["id"],),
                ).fetchone()
                run["latest_correction"] = (
                    None if latest is None else {
                        **dict(latest),
                        "corrected_counts": _decode_json(latest["corrected_counts"], {}),
                    }
                )
                run["selected_classes"] = _decode_json(run["selected_classes"], [])
                for result in run["model_results"]:
                    result["counts_by_class"] = _decode_json(result["counts_by_class"], {})

        for run in runs:
            status = str(run.get("flag_status") or "not validated").casefold()
            run["queue_group"] = (
                "flagged" if status in {"flagged", "yes", "needs review"} else "other"
            )
            totals = [int(result["total_count"]) for result in run["model_results"]]
            run["disagreement_span"] = (
                max(totals) - min(totals) if len(totals) > 1 else 0
            )
            run["review_order_label"] = (
                "by model disagreement"
                if status in {"not validated", "unvalidated", ""}
                else "flagged first"
            )
        runs.sort(
            key=lambda run: (
                run["queue_group"] != "flagged",
                -run["disagreement_span"]
                if run["review_order_label"] == "by model disagreement"
                else 0,
                run["timestamp_utc"],
                run["id"],
            )
        )
        if queue_filter == "Unreviewed":
            runs = [run for run in runs if run["latest_correction"] is None]
        elif queue_filter == "Reviewed":
            runs = [run for run in runs if run["latest_correction"] is not None]
        return runs
    except HistoryStoreError:
        raise
    except sqlite3.DatabaseError as error:
        raise HistoryStoreError("The human review queue could not be read.") from error


def save_correction(
    run_id: str,
    corrected_counts: dict[str, int],
    reviewer: str,
    note: str = "",
    path: Path | None = None,
) -> int:
    """Append one complete user correction; never update an earlier correction."""
    path = _database_path(path)
    expected = set(config.LABEL_CLASS_NAMES)
    if set(corrected_counts) != expected:
        raise ValueError("A correction must provide all seven project vehicle categories.")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in corrected_counts.values()
    ):
        raise ValueError("Corrected counts must be non-negative whole numbers.")
    if not reviewer.strip():
        raise ValueError("Enter a reviewer name or anonymous reviewer label.")
    try:
        migrate(path)
        with _connect(path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO corrections (
                    run_id, timestamp_utc, reviewer, corrected_counts, note
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    _utc_now(),
                    reviewer.strip(),
                    json.dumps(corrected_counts, separators=(",", ":"), sort_keys=True),
                    note.strip(),
                ),
            )
            return int(cursor.lastrowid)
    except sqlite3.IntegrityError as error:
        raise HistoryStoreError("The selected run does not exist; correction was not saved.") from error
    except (sqlite3.DatabaseError, OSError) as error:
        raise HistoryStoreError("The correction could not be appended to local history.") from error


def correction_comparison(run: dict[str, Any], correction: dict[str, Any]) -> dict[str, Any]:
    """Compare stored model counts with one correction over its selected categories."""
    selected = [
        name for name in run["selected_classes"]
        if name in config.LABEL_CLASS_NAMES
    ]
    reference = sum(int(correction["corrected_counts"].get(name, 0)) for name in selected)
    models = []
    low, high = run.get("count_range_min"), run.get("count_range_max")
    for model in run.get("model_results", []):
        prediction = int(model["total_count"])
        models.append({
            "model": model["model"],
            "predicted_total": prediction,
            "corrected_total": reference,
            "signed_error": prediction - reference,
            "inside_stated_range": (
                None if low is None or high is None
                else int(low) <= reference <= int(high)
            ),
        })
    return {
        "selected_classes": selected,
        "corrected_total": reference,
        "models": models,
    }


def correction_aggregates(path: Path | None = None) -> dict[str, Any]:
    """Summarize absolute pre-review model-total errors against latest corrections."""
    queue = review_queue(path=path, queue_filter="Reviewed")
    groups: dict[str, dict[str, Any]] = {
        "flagged": {"runs": set(), "absolute_errors": []},
        "unflagged": {"runs": set(), "absolute_errors": []},
    }
    for run in queue:
        correction = run["latest_correction"]
        if correction is None:
            continue
        comparison = correction_comparison(run, correction)
        status = str(run.get("flag_status") or "not validated").casefold()
        group_name = (
            "flagged" if status in {"flagged", "yes", "needs review"} else "unflagged"
        )
        group = groups[group_name]
        group["runs"].add(run["id"])
        group["absolute_errors"].extend(
            abs(model["signed_error"]) for model in comparison["models"]
        )
    return {
        group_name: {
            "reviewed_runs": len(values["runs"]),
            "mean_absolute_error_before_correction": (
                sum(values["absolute_errors"]) / len(values["absolute_errors"])
                if values["absolute_errors"] else None
            ),
            "model_run_comparisons": len(values["absolute_errors"]),
        }
        for group_name, values in groups.items()
    }


def correction_export_rows(path: Path | None = None) -> list[dict[str, Any]]:
    """Export every appended human correction, one row per correction."""
    path = _database_path(path)
    try:
        migrate(path)
        with _connect(path) as connection:
            rows = connection.execute(
                """
                SELECT c.id AS correction_id, c.run_id, c.timestamp_utc,
                    c.reviewer, c.corrected_counts, c.note, r.input_filename,
                    r.image_sha256, r.mode, r.rickshaw_mode, r.selected_classes
                FROM corrections c JOIN runs r ON r.id = c.run_id
                ORDER BY c.id
                """
            ).fetchall()
        return [
            {
                **dict(row),
                "corrected_counts": _decode_json(row["corrected_counts"], {}),
                "selected_classes": _decode_json(row["selected_classes"], []),
            }
            for row in rows
        ]
    except (sqlite3.DatabaseError, HistoryStoreError) as error:
        raise HistoryStoreError("User corrections could not be exported.") from error


def blind_review_rows(queue: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Make safe pre-reveal queue summaries without model counts or count range."""
    return [
        {
            "Run ID": run["id"],
            "Timestamp (UTC)": run["timestamp_utc"],
            "Filename": run["input_filename"],
            "Queue group": run["queue_group"],
            "Order": run["review_order_label"],
            "Review status": (
                "reviewed" if run["latest_correction"] is not None else "not reviewed"
            ),
        }
        for run in queue
    ]
