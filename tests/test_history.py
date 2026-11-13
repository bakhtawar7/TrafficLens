"""Temporary-database tests for run history storage."""
from __future__ import annotations

import sqlite3

import pytest

from core import history


def _run(filename: str = "road.jpg", image_hash: str = "image-hash") -> dict:
    return {
        "input_filename": filename,
        "image_sha256": image_hash,
        "width": 640,
        "height": 480,
        "settings_sha256": "settings-hash",
        "mode": "standard",
        "selected_classes": ["car", "truck"],
        "threshold": 0.25,
        "rickshaw_mode": False,
    }


def _models() -> list[dict]:
    return [
        {
            "model": "YOLOv8n",
            "weights_sha256": "weights-hash",
            "total_count": 2,
            "counts_by_class": {"car": 1, "truck": 1},
            "inference_ms": 4.5,
        }
    ]


def test_insert_and_deduplicate_run_without_storing_image_bytes(tmp_path):
    database = tmp_path / "history.db"
    first = history.record_run(_run(), _models(), path=database)
    duplicate = history.record_run(
        _run(),
        [
            {
                "model": "YOLOv8s",
                "weights_sha256": "weights-s",
                "total_count": 3,
                "counts_by_class": {"car": 3, "truck": 0},
                "inference_ms": 8.0,
            }
        ],
        {"flag_status": "not validated"},
        path=database,
    )

    assert first.inserted
    assert duplicate.inserted is False
    assert duplicate.run_id == first.run_id
    run = history.get_run(first.run_id, path=database)
    assert run is not None
    assert {row["model"] for row in run["model_results"]} == {"YOLOv8n", "YOLOv8s"}
    assert run["flag_status"] == "not validated"
    with sqlite3.connect(database) as connection:
        tables = {row[0] for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
    assert "image_bytes" not in tables
    assert "images" not in tables


def test_foreign_keys_are_enabled_and_enforced(tmp_path):
    database = tmp_path / "history.db"
    history.migrate(database)
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "INSERT INTO model_results "
                "(run_id, model, total_count, counts_by_class, inference_ms) "
                "VALUES ('missing', 'YOLOv8n', 0, '{}', 0)"
            )


def test_migration_is_repeatable_and_records_schema_version(tmp_path):
    database = tmp_path / "history.db"
    assert history.migrate(database) == history.LATEST_SCHEMA_VERSION
    assert history.migrate(database) == history.LATEST_SCHEMA_VERSION
    with sqlite3.connect(database) as connection:
        versions = connection.execute(
            "SELECT version FROM schema_version ORDER BY version"
        ).fetchall()
    assert versions == [(history.LATEST_SCHEMA_VERSION,)]


def test_corrupt_database_is_reported_without_replacement(tmp_path):
    database = tmp_path / "corrupt.db"
    original = b"not a sqlite database"
    database.write_bytes(original)

    with pytest.raises(history.HistoryStoreError, match="unavailable or corrupt"):
        history.migrate(database)

    assert database.read_bytes() == original


def test_clear_history_removes_runs_and_dependent_rows(tmp_path):
    database = tmp_path / "history.db"
    result = history.record_run(
        _run(),
        _models(),
        {"flag_status": "not validated"},
        path=database,
    )
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "INSERT INTO corrections(run_id, timestamp_utc, reviewer, corrected_counts) "
            "VALUES (?, '2026-01-01T00:00:00Z', 'reviewer', '{}')",
            (result.run_id,),
        )

    history.clear_history(database)

    assert history.list_runs(database) == []
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT COUNT(*) FROM model_results").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM reliability").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM corrections").fetchone()[0] == 0
