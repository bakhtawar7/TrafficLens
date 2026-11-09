"""Leakage guard for the development subset.

Tuning code may only ever see the seeded development images. This module is the
read path for that work: :func:`development_ids` returns only the development
subset ids recorded in ``configs/testset_plan.json``, and
:func:`read_development_image` raises :class:`DevelopmentOnlyError` when any
code asks it to read a non-development image. The guard never selects a setting
and never opens an evaluation image.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core import config, dataset


class DevelopmentOnlyError(RuntimeError):
    """Raised when tuning code asks for anything outside the development subset."""


def development_ids(plan_path: Path | None = None) -> set[int]:
    """The item numbers of the seeded development subset, and nothing else.

    Raises:
        FileNotFoundError: when the plan file is missing; the subset is defined
            only there, so guessing is never an option.
        ValueError: when the plan lists an id that is not a TrafficLens item.
    """
    path = config.TESTSET_PLAN_JSON if plan_path is None else Path(plan_path)
    if not path.exists():
        raise FileNotFoundError(f"development subset needs the seeded plan: {path}")
    plan = json.loads(path.read_text(encoding="utf-8"))
    ids: set[int] = set()
    for value in plan.get("development_subset", []):
        number = dataset.parse_item_number(value)
        if number is None:
            raise ValueError(f"development subset holds an unparseable item id: {value!r}")
        ids.add(number)
    return ids


def assert_development(item_id: object, plan_path: Path | None = None) -> int:
    """Return the item number when the id belongs to the development subset.

    Accepts any id spelling ``dataset.parse_item_number`` understands
    (``tl0005``, ``tl005``, ``TL5``).

    Raises:
        DevelopmentOnlyError: when the id is unparseable or not in the subset.
    """
    number = dataset.parse_item_number(item_id)
    if number is None or number not in development_ids(plan_path):
        raise DevelopmentOnlyError(
            f"item {item_id!r} is not in the development subset; "
            "tuning code may only read development images"
        )
    return number


def read_development_image(path: Path | str, plan_path: Path | None = None) -> np.ndarray:
    """Decode a development image, refusing every other image before reading it.

    The item id is checked against the seeded plan first, so an evaluation
    image is rejected without its bytes ever being opened.

    Raises:
        DevelopmentOnlyError: when the file name is not a development item.
        FileNotFoundError: when the development image file is missing.
        ValueError: when the file exists but cannot be decoded.
    """
    resolved = Path(path)
    number = dataset.parse_item_number(resolved.stem.split("_")[0])
    if number is None or number not in development_ids(plan_path):
        raise DevelopmentOnlyError(
            f"{resolved.name} is not a development image; "
            "tuning code may only read development images"
        )
    if not resolved.exists():
        raise FileNotFoundError(f"development image missing: {resolved}")
    image = dataset.load_image(resolved)
    if image is None:
        raise ValueError(f"could not decode development image: {resolved}")
    return image


def development_image_paths(
    images_dir: Path | None = None, plan_path: Path | None = None
) -> list[Path]:
    """Sorted paths of the development images present in the folder.

    Only files whose item id is in the seeded subset are returned; every other
    test set image is filtered out before any caller can open it.

    Raises:
        FileNotFoundError: when the images folder is missing.
    """
    directory = config.TESTSET_IMAGES_DIR if images_dir is None else Path(images_dir)
    if not directory.exists():
        raise FileNotFoundError(f"images folder missing: {directory}")
    ids = development_ids(plan_path)
    paths: list[Path] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in config.ALLOWED_EXTENSIONS:
            continue
        number = dataset.parse_item_number(path.stem.split("_")[0])
        if number is not None and number in ids:
            paths.append(path)
    return paths