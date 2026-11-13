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


def _queued_run(
    path,
    image_hash,
    timestamp,
    status,
    counts,
    corrected_car=None,
):
    row = _run(image_hash=image_hash)
    row["timestamp_utc"] = timestamp
    results = [
        {
            "model": model,
            "weights_sha256": None,
            "total_count": total,
            "counts_by_class": {"car": total, "truck": 0},
            "inference_ms": 1.0,
        }
        for model, total in counts
    ]
    written = history.record_run(
        row,
        results,
        {
            "flag_status": status,
            "count_range_min": min(total for _, total in counts),
            "count_range_max": max(total for _, total in counts),
        },
        path=path,
    )
    if corrected_car is not None:
        history.save_correction(
            written.run_id,
            {
                name: corrected_car if name == "car" else 0
                for name in history.config.LABEL_CLASS_NAMES
            },
            "anonymous",
            path=path,
        )
    return written.run_id


def test_review_queue_orders_flagged_then_unvalidated_by_disagreement_and_is_blind(tmp_path):
    database = tmp_path / "history.db"
    unvalidated_small = _queued_run(
        database, "image-small", "2026-01-01T00:00:00Z", "not validated",
        [("A", 3), ("B", 4)],
    )
    unvalidated_large = _queued_run(
        database, "image-large", "2026-01-02T00:00:00Z", "not validated",
        [("A", 1), ("B", 8)],
    )
    flagged = _queued_run(
        database, "image-flagged", "2026-01-03T00:00:00Z", "flagged",
        [("A", 4), ("B", 5)],
    )

    queue = history.review_queue(database)
    assert [run["id"] for run in queue] == [
        flagged,
        unvalidated_large,
        unvalidated_small,
    ]
    assert queue[1]["review_order_label"] == "by model disagreement"
    safe_rows = history.blind_review_rows(queue)
    assert not any("model_results" in row or "counts_by_class" in row for row in safe_rows)
    assert not any("count_range_min" in row for row in safe_rows)


def test_corrections_are_append_only_latest_is_shown_and_labels_are_untouched(
    tmp_path, monkeypatch
):
    database = tmp_path / "history.db"
    labels = tmp_path / "LABELS.csv"
    labels.write_text("item_id,car\nitem1,7\n", encoding="utf-8")
    before = labels.read_bytes()
    monkeypatch.setattr(history.config, "LABELS_CSV", labels)
    run_id = _queued_run(
        database, "append-only", "2026-01-01T00:00:00Z", "not flagged",
        [("YOLOv8n", 4)],
    )
    first_counts = {name: 0 for name in history.config.LABEL_CLASS_NAMES}
    first_counts["car"] = 2
    second_counts = dict(first_counts)
    second_counts["car"] = 3

    history.save_correction(run_id, first_counts, "reviewer", "first", database)
    history.save_correction(run_id, second_counts, "reviewer", "second", database)

    run = history.get_run(run_id, database)
    assert run is not None
    assert len(run["corrections"]) == 2
    latest = history.review_queue(database)[0]["latest_correction"]
    assert latest["corrected_counts"] == second_counts
    assert labels.read_bytes() == before


def test_correction_comparison_and_aggregate_error_math(tmp_path):
    database = tmp_path / "history.db"
    flagged_id = _queued_run(
        database, "flagged-math", "2026-01-01T00:00:00Z", "flagged",
        [("YOLOv8n", 3), ("YOLOv8s", 4)], corrected_car=3,
    )
    unflagged_id = _queued_run(
        database, "unflagged-math", "2026-01-02T00:00:00Z", "not flagged",
        [("YOLOv8n", 5)], corrected_car=2,
    )

    queue = {run["id"]: run for run in history.review_queue(database)}
    comparison = history.correction_comparison(
        queue[flagged_id], queue[flagged_id]["latest_correction"]
    )
    assert [row["signed_error"] for row in comparison["models"]] == [0, 1]
    assert all(row["inside_stated_range"] for row in comparison["models"])
    aggregates = history.correction_aggregates(database)
    assert aggregates["flagged"]["reviewed_runs"] == 1
    assert aggregates["flagged"]["mean_absolute_error_before_correction"] == 0.5
    assert aggregates["unflagged"]["reviewed_runs"] == 1
    assert aggregates["unflagged"]["mean_absolute_error_before_correction"] == 3.0
    assert history.correction_export_rows(database)[0]["corrected_counts"]["car"] == 3
    assert unflagged_id in queue
