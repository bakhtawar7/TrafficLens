"""Load and fingerprint the benchmark settings file.

The settings file (default ``configs/default_settings.json``) records the
per-model defaults a run was produced under: confidence threshold, input size,
sliced mode and tile settings, and NMS IoU where the model uses NMS. The
benchmark and the evaluator embed the file's SHA-256 plus those fields into
``results/summary.json`` so every reported statistic can be traced back to the
exact settings that produced it. Loading only reads; it never selects or
changes a setting.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from core import config


def resolve_settings_path(path: Path | str | None = None) -> Path:
    """The explicit path when given, otherwise the pre-registered defaults file."""
    return config.DEFAULT_SETTINGS_JSON if path is None else Path(path)


def load_settings(path: Path | str | None = None) -> dict[str, Any]:
    """Parse the settings file.

    Raises:
        FileNotFoundError: when the file is missing, so a run can never silently
            proceed without a settings provenance.
    """
    resolved = resolve_settings_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"settings file missing: {resolved}")
    return json.loads(resolved.read_text(encoding="utf-8"))


def settings_sha256(path: Path | str | None = None) -> str:
    """SHA-256 hex digest of the settings file bytes."""
    resolved = resolve_settings_path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"settings file missing: {resolved}")
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def settings_record(path: Path | str | None = None) -> dict[str, Any]:
    """Provenance block for ``results/summary.json``.

    Carries the file name, its SHA-256, and the per-model settings (which include
    ``input_size`` and the ``uses_nms``/``nms_iou`` fields) plus the sliced tile
    defaults, so the summary states the full configuration without the reader
    having to locate the file.
    """
    resolved = resolve_settings_path(path)
    settings = load_settings(resolved)
    return {
        "settings_file": resolved.name,
        "settings_sha256": settings_sha256(resolved),
        "models": settings.get("models", {}),
        "sliced_tiles": settings.get("sliced_tiles", {}),
    }