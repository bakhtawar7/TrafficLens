"""JSON and CSV export for a counting run.

Both formats carry the same scalar fields so a downstream reviewer can reconstruct the
run. JSON keeps the full detection list; CSV is one summary row per run with one count
column per class. Export file names follow ``trafficlens_<stem>_<UTC timestamp>``.
"""
from __future__ import annotations

import csv
import io
import json
import os
from datetime import datetime, timezone
from typing import Any

from .counting import CountResult

# Scalar fields shared by JSON and CSV, in a stable order.
SCALAR_FIELDS: tuple[str, ...] = (
    "input_filename",
    "timestamp_utc",
    "model",
    "confidence_threshold",
    "image_width",
    "image_height",
    "inference_ms",
    "total_count",
)


def _utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string with a Z suffix."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _compact_timestamp(iso_timestamp: str) -> str:
    """Turn an ISO timestamp into a filesystem safe compact stamp for filenames."""
    return iso_timestamp.replace("-", "").replace(":", "").replace("T", "").rstrip("Z") + "Z"


def build_payload(result: CountResult, input_filename: str, timestamp_utc: str | None = None) -> dict[str, Any]:
    """Assemble the full export payload from a :class:`CountResult`.

    Args:
        result: The counting outcome to describe.
        input_filename: Name of the source image, used verbatim in the payload.
        timestamp_utc: ISO timestamp override, mainly for deterministic tests.

    Returns:
        A JSON serialisable dictionary with every Task 3 field.
    """
    stamp = timestamp_utc or _utc_now_iso()
    return {
        "input_filename": input_filename,
        "timestamp_utc": stamp,
        "model": result.model_name,
        "confidence_threshold": round(result.confidence_threshold, 4),
        "selected_classes": list(result.selected_classes),
        "image_width": result.image_width,
        "image_height": result.image_height,
        "inference_ms": round(result.inference_ms, 2),
        "total_count": result.total_count,
        "counts_by_class": dict(result.counts_by_class),
        "detections": [
            {
                "class": d.class_name,
                "confidence": round(d.confidence, 4),
                "bbox_xyxy": [round(v, 2) for v in d.xyxy],
            }
            for d in result.detections
        ],
    }


def to_json_bytes(payload: dict[str, Any]) -> bytes:
    """Serialise an export payload to UTF-8 encoded JSON bytes."""
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def to_csv_bytes(payload: dict[str, Any]) -> bytes:
    """Serialise one export payload to a single data row CSV in UTF-8 bytes.

    The header carries the scalar fields, the selected class list, and one count column
    per class, so consecutive runs can be appended to the same file.
    """
    counts = payload["counts_by_class"]
    buffer = io.StringIO(newline="")
    fieldnames = list(SCALAR_FIELDS) + ["selected_classes"] + list(counts.keys())
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    row: dict[str, Any] = {key: payload[key] for key in SCALAR_FIELDS}
    row["selected_classes"] = ";".join(payload["selected_classes"])
    row.update(counts)
    writer.writerow(row)
    return buffer.getvalue().encode("utf-8")


def export_filename(input_filename: str, timestamp_utc: str, extension: str) -> str:
    """Build a ``trafficlens_<stem>_<UTC stamp>.<extension>`` export name."""
    stem = os.path.splitext(os.path.basename(input_filename))[0] or "image"
    safe_stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)
    return f"trafficlens_{safe_stem}_{_compact_timestamp(timestamp_utc)}.{extension.lstrip('.')}"


def make_json_filename(input_filename: str, timestamp_utc: str) -> str:
    """Export filename for the JSON artefact."""
    return export_filename(input_filename, timestamp_utc, "json")


def make_csv_filename(input_filename: str, timestamp_utc: str) -> str:
    """Export filename for the CSV artefact."""
    return export_filename(input_filename, timestamp_utc, "csv")
