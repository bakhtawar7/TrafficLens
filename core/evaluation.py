"""Evaluation of cached model counts against the labeller's own counts.

The evaluator joins three artefacts: ``LABELS.csv`` (manual counts per class,
first and second labelling passes), ``SOURCES.csv`` (condition tags per item),
and ``results/raw_detections.jsonl`` (cached raw detections at the lowest sweep
threshold, from which any higher threshold is derived by confidence filtering).
Item ids are read through :func:`core.dataset.parse_item_number`, which accepts the
legacy zero padded spellings and returns the item number, so ``tl001`` and ``tl0001``
both refer to item 1. Written output always uses the canonical four digit form
defined by ``core.dataset.ITEM_ID_PATTERN``.

Rows whose ``notes`` cell still carries the auto derived draft wording
(:data:`UNVERIFIED_NOTE_MARKERS`) are refused: they never enter a label load, so no
error statistic, agreement figure, or labeller comparison can be computed from counts
that no labeller has verified. The refusal is reported in the summary and printed.

Ground truth has two targets, always reported separately:

* covered total: car + bus + truck + motorcycle, the classes COCO models can
  honestly claim;
* all-vehicles total: additionally rickshaw and loader_rickshaw. Models cannot
  detect those, so this target bakes in the rickshaw gap; it bounds how large
  the honest error floor is before any model fault.

Errors compare model counts to ground truth per item. The development subset
recorded in ``configs/testset_plan.json`` is excluded from every statistic and
from the written table. Confidence intervals are percentile bootstraps over
items (``config.SEED``, 2000 resamples by default), deterministic across runs.

Definitions fixed by this module (docs/hypotheses.md does not spell them out):

* ``Count Range``: ``min-max`` across the three standard model counts;
* ``Agreement``: mean of the three pairwise class-aware box agreements
  (:func:`core.reliability.box_agreement`) between standard detections;
* ``Actual Error``: absolute difference between the covered ground truth and
  the median of the three standard model counts;
* labeller agreement: mean absolute difference between first and second pass
  covered totals over items labelled twice.
"""
from __future__ import annotations

import csv
import json
import random
import statistics
from pathlib import Path
from typing import Any

from . import config, dataset
from .detector import Detection
from .reliability import box_agreement

#: LABELS.csv class columns that make up the covered ground truth target.
COVERED_CLASSES: tuple[str, ...] = tuple(config.TARGET_CLASS_NAMES)
#: Additional vehicle classes only the labeller can count (no COCO class).
LOCAL_VEHICLE_CLASSES: tuple[str, ...] = ("rickshaw", "loader_rickshaw")

#: Note wording that marks a LABELS.csv row as an unverified machine draft rather than a
#: labeller's count. Matched case insensitively as a substring: both producers write this
#: sentence from ``core.dataset.UNVERIFIED_DRAFT_NOTE``, and a test asserts that wording is
#: refused here, so rewording the note without rewording these markers fails the suite
#: instead of quietly allowing an unverified count to be scored.
UNVERIFIED_NOTE_MARKERS: tuple[str, ...] = ("auto derived", "requires labeller verification")

#: Exact column order of results/evaluation_table.csv.
TABLE_COLUMNS: tuple[str, ...] = (
    "Item", "Condition", "Ground Truth", "YOLOv8n", "YOLOv8s", "RT-DETR-L",
    "Sliced Count", "Count Range", "Agreement", "Actual Error",
)

_MODEL_DISPLAY: dict[str, str] = {key: spec.display_name for key, spec in config.MODEL_REGISTRY.items()}


def normalise_item_id(value: str) -> int | None:
    """Extract the item number from ids like ``tl001`` or ``tl0001``; None if unlike.

    Thin alias for :func:`core.dataset.parse_item_number`, kept so the join used by every
    loader in this module has one name.
    """
    return dataset.parse_item_number(value)


def is_unverified_note(notes: object) -> bool:
    """True when a notes cell says the counts were auto derived or still need checking."""
    text = str(notes or "").lower()
    return any(marker in text for marker in UNVERIFIED_NOTE_MARKERS)


def _int_or_none(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(str(value).strip()))
    except ValueError:
        return None


def covered_total(counts: dict[str, int]) -> int:
    """Car + bus + truck + motorcycle."""
    return sum(counts.get(name, 0) for name in COVERED_CLASSES)


def all_vehicles_total(counts: dict[str, int]) -> int:
    """Covered total plus rickshaw and loader rickshaw."""
    return covered_total(counts) + sum(counts.get(name, 0) for name in LOCAL_VEHICLE_CLASSES)


def load_labels(
    path: Path, refuse_unverified: bool = True
) -> dict[int, dict[str, dict[str, int] | None]]:
    """Read LABELS.csv into per-item first and second pass class counts.

    Rows without any count values are ignored. When ``refuse_unverified`` is true, a row
    whose notes still say the counts were auto derived is ignored as well, because an
    unverified draft is not ground truth and must not feed any statistic. A blank
    labeller pass is treated as the first pass. Later rows for the same item and pass
    overwrite earlier ones.
    """
    items: dict[int, dict[str, dict[str, int] | None]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            item = normalise_item_id(row.get("item_id", ""))
            if item is None:
                continue
            if refuse_unverified and is_unverified_note(row.get("notes", "")):
                continue
            counts = {name: _int_or_none(row.get(name)) for name in config.LABEL_CLASS_NAMES}
            if all(value is None for value in counts.values()):
                continue  # no counts at all: an unfilled scaffold row
            clean = {name: (value if value is not None else 0) for name, value in counts.items()}
            passes = _int_or_none(row.get("labeller_pass"))
            slot = "pass2" if passes == 2 else "pass1"
            items.setdefault(item, {"pass1": None, "pass2": None})[slot] = clean
    return items


def unverified_label_rows(path: Path) -> list[str]:
    """Return the canonical ids of LABELS.csv rows whose notes mark them unverified.

    These are the rows :func:`load_labels` refuses. The list is reported so the refusal is
    visible in the output instead of looking like an empty test set.
    """
    if not Path(path).exists():
        return []
    found: list[str] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            number = normalise_item_id(row.get("item_id", ""))
            if number is not None and is_unverified_note(row.get("notes", "")):
                found.append(dataset.format_item_id(number))
    return sorted(set(found))


def load_conditions(path: Path) -> dict[int, str]:
    """Read the semicolon condition tags of every item from SOURCES.csv."""
    conditions: dict[int, str] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            item = normalise_item_id(row.get("item_id", ""))
            if item is not None:
                conditions[item] = (row.get("condition_tags") or "").strip()
    return conditions


def load_development_ids(path: Path) -> set[int]:
    """Item numbers in the seeded development subset, excluded from all outputs."""
    if not path.exists():
        return set()
    plan = json.loads(path.read_text(encoding="utf-8"))
    return {n for n in (normalise_item_id(v) for v in plan.get("development_subset", [])) if n is not None}


def load_detection_cache(
    path: Path, confidence: float = config.DEFAULT_CONFIDENCE
) -> dict[tuple[int, str, str], list[Detection]]:
    """Read the benchmark JSONL into (item, model key, mode) detection lists.

    Entries were produced at the lowest sweep threshold; detections below
    ``confidence`` are filtered out here, which reproduces a direct run at
    that threshold (see ``core.reliability``). Detections of classes outside
    the covered vehicle classes are dropped.
    """
    cache: dict[tuple[int, str, str], list[Detection]] = {}
    if not path.exists():
        return cache
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            item = normalise_item_id(str(entry.get("image", "")).split("_")[0])
            if item is None:
                continue
            detections = [
                Detection(int(d["class_id"]), str(d["class_name"]), float(d["confidence"]),
                          tuple(float(v) for v in d["xyxy"]))
                for d in entry.get("detections", [])
                if d["class_name"] in COVERED_CLASSES and float(d["confidence"]) >= confidence
            ]
            cache[(item, str(entry["model"]), str(entry["mode"]))] = detections
    return cache


def bootstrap_ci(values: list[float], samples: int = 2000, seed: int = config.SEED) -> tuple[float | None, float | None]:
    """Percentile bootstrap 95% CI of the mean of ``values``; (None, None) if empty."""
    if not values:
        return (None, None)
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean([values[i] for i in (rng.randrange(len(values)) for _ in values)])
        for _ in range(samples)
    )
    lower = means[int(0.025 * samples)]
    upper = means[min(int(0.975 * samples), samples - 1)]
    return (round(lower, 4), round(upper, 4))


def _mean_abs_error(pairs: list[tuple[int, int]]) -> float:
    """Mean absolute error over (prediction, ground truth) pairs."""
    return round(statistics.fmean(abs(p - g) for p, g in pairs), 4)


def build_rows(
    labels: dict[int, dict[str, dict[str, int] | None]],
    conditions: dict[int, str],
    cache: dict[tuple[int, str, str], list[Detection]],
    excluded: set[int],
    sliced_model_key: str = config.DEFAULT_MODEL_NAME,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Assemble one table row per evaluable item plus the ids with missing data.

    An item is evaluable when it has first-pass counts, is not in ``excluded``,
    and the cache holds standard entries for every registry model plus a sliced
    entry for ``sliced_model_key``.
    """
    model_keys = list(config.MODEL_REGISTRY)
    rows: list[dict[str, Any]] = []
    missing: list[int] = []
    for item in sorted(labels):
        entry = labels[item]
        first = entry.get("pass1")
        if first is None or item in excluded:
            continue
        needed = [(item, key, "standard") for key in model_keys] + [(item, sliced_model_key, "sliced")]
        if any(key not in cache for key in needed):
            missing.append(item)
            continue
        standard = {key: cache[(item, key, "standard")] for key in model_keys}
        counts = {key: len(standard[key]) for key in model_keys}
        sliced_count = len(cache[(item, sliced_model_key, "sliced")])
        gt_covered = covered_total(first)
        values = [counts[key] for key in model_keys]
        pairwise = [
            box_agreement(standard[a], standard[b])
            for i, a in enumerate(model_keys)
            for b in model_keys[i + 1:]
        ]
        agreement = round(statistics.fmean(pairwise), 4) if pairwise else None
        actual_error = abs(gt_covered - round(statistics.median(values)))
        rows.append({
            "Item": dataset.format_item_id(item),
            "Condition": conditions.get(item, ""),
            "Ground Truth": gt_covered,
            **{_MODEL_DISPLAY[key]: counts[key] for key in model_keys},
            "Sliced Count": sliced_count,
            "Count Range": f"{min(values)}-{max(values)}",
            "Agreement": agreement,
            "Actual Error": actual_error,
        })
    return rows, missing


def summarise(
    labels: dict[int, dict[str, dict[str, int] | None]],
    conditions: dict[int, str],
    rows: list[dict[str, Any]],
    missing: list[int],
    excluded: set[int],
    confidence: float,
    sliced_model_key: str,
    bootstrap_samples: int,
    refused_unverified: list[str] | None = None,
) -> dict[str, Any]:
    """Compute every statistic that goes into ``results/summary.json``.

    ``refused_unverified`` is the list of canonical ids whose label rows were rejected for
    carrying unverified draft notes. It is recorded so the artefact states, without the
    reader having to infer it, which rows were deliberately not scored.
    """
    refused = sorted(set(refused_unverified or []))
    evaluated = {row["Item"]: row for row in rows}
    labelled_non_dev = [
        item for item, data in labels.items()
        if data.get("pass1") is not None and item not in excluded
    ]

    model_mae: dict[str, Any] = {}
    for key, spec in config.MODEL_REGISTRY.items():
        covered_pairs, all_pairs = [], []
        for row in rows:
            first = labels[normalise_item_id(row["Item"])]["pass1"]
            covered_pairs.append((row[_MODEL_DISPLAY[key]], covered_total(first)))
            all_pairs.append((row[_MODEL_DISPLAY[key]], all_vehicles_total(first)))
        covered_errors = [abs(p - g) for p, g in covered_pairs]
        all_errors = [abs(p - g) for p, g in all_pairs]
        model_mae[spec.display_name] = {
            "covered": {"mae": round(statistics.fmean(covered_errors), 4) if covered_errors else None,
                        "ci95": bootstrap_ci(covered_errors, bootstrap_samples), "n": len(covered_errors)},
            "all_vehicles": {"mae": round(statistics.fmean(all_errors), 4) if all_errors else None,
                             "ci95": bootstrap_ci(all_errors, bootstrap_samples), "n": len(all_errors)},
        }

    condition_mae: dict[str, Any] = {}
    for row in rows:
        for tag in (row["Condition"].split(";") if row["Condition"] else []):
            tag = tag.strip()
            if not tag:
                continue
            first = labels[normalise_item_id(row["Item"])]["pass1"]
            gt = covered_total(first)
            bucket = condition_mae.setdefault(tag, {})
            for key, spec in config.MODEL_REGISTRY.items():
                bucket.setdefault(spec.display_name, []).append(abs(row[_MODEL_DISPLAY[key]] - gt))
    condition_summary = {
        tag: {
            name: {"mae": round(statistics.fmean(errors), 4),
                   "ci95": bootstrap_ci(errors, bootstrap_samples), "n": len(errors)}
            for name, errors in models.items()
        }
        for tag, models in sorted(condition_mae.items())
    }

    both = [
        (item, data) for item, data in labels.items()
        if data.get("pass1") is not None and data.get("pass2") is not None and item not in excluded
    ]
    covered_differences = [
        abs(covered_total(d["pass1"]) - covered_total(d["pass2"])) for _, d in both
    ]
    all_differences = [
        abs(all_vehicles_total(d["pass1"]) - all_vehicles_total(d["pass2"])) for _, d in both
    ]
    labeller = {
        "items_both_passes": len(both),
        "covered_mean_abs_difference": round(statistics.fmean(covered_differences), 4) if covered_differences else None,
        "covered_ci95": bootstrap_ci(covered_differences, bootstrap_samples),
        "all_vehicles_mean_abs_difference": round(statistics.fmean(all_differences), 4) if all_differences else None,
        "identical_fraction": round(sum(1 for d in covered_differences if d == 0) / len(covered_differences), 4) if covered_differences else None,
    }

    warning = None
    if refused and not rows:
        warning = (
            f"no error statistic computed: all {len(refused)} labelled row(s) were refused "
            "because their notes record the counts as auto derived and unverified"
        )
    elif len(labelled_non_dev) < 30:
        warning = (
            f"only {len(labelled_non_dev)} labelled items outside the development subset "
            "(fewer than 30): every statistic is provisional"
        )

    return {
        "settings": {
            "confidence": confidence,
            "sliced_model": sliced_model_key,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": config.SEED,
        },
        "items": {
            "labelled_first_pass_total": sum(1 for d in labels.values() if d.get("pass1") is not None),
            "development_excluded": len(excluded),
            "labelled_outside_development": len(labelled_non_dev),
            "evaluated": len(rows),
            "missing_detections": [dataset.format_item_id(i) for i in missing],
            "present_in_table": sorted(evaluated),
        },
        "warning": warning,
        "refusals": {
            "unverified_label_rows": {
                "count": len(refused),
                "items": refused,
                "note_markers": list(UNVERIFIED_NOTE_MARKERS),
                "statement": (
                    "error statistics were not computed from these rows because their notes "
                    "record the counts as auto derived and awaiting labeller verification"
                ),
            },
        },
        "model_mae": model_mae,
        "condition_mae": condition_summary,
        "labeller_passes": labeller,
    }


def evaluate(
    labels_csv: Path = config.LABELS_CSV,
    sources_csv: Path = config.SOURCES_CSV,
    plan_json: Path = config.TESTSET_PLAN_JSON,
    detections_jsonl: Path = config.RAW_DETECTIONS_JSONL,
    table_out: Path = config.EVALUATION_TABLE_CSV,
    summary_out: Path = config.EVALUATION_SUMMARY_JSON,
    confidence: float = config.DEFAULT_CONFIDENCE,
    sliced_model_key: str = config.DEFAULT_MODEL_NAME,
    bootstrap_samples: int = 2000,
    write: bool = True,
) -> dict[str, Any]:
    """Run the full evaluation and return (and by default write) the summary.

    The development subset is excluded everywhere. Rows whose notes still describe the
    counts as an unverified auto derived draft are refused before any statistic runs, and
    the refusal is printed. Raises ValueError when ``sliced_model_key`` is not a registry
    key.
    """
    if sliced_model_key not in config.MODEL_REGISTRY:
        raise ValueError(f"unknown sliced model key: {sliced_model_key}")
    labels_path = Path(labels_csv)
    refused = unverified_label_rows(labels_path)
    labels = load_labels(labels_path)
    conditions = load_conditions(Path(sources_csv))
    excluded = load_development_ids(Path(plan_json))
    cache = load_detection_cache(Path(detections_jsonl), confidence)
    rows, missing = build_rows(labels, conditions, cache, excluded, sliced_model_key)
    summary = summarise(labels, conditions, rows, missing, excluded, confidence,
                        sliced_model_key, bootstrap_samples, refused_unverified=refused)
    if refused:
        print(
            f"REFUSED {len(refused)} LABELS row(s) whose notes record the counts as auto "
            f"derived and requiring labeller verification: {', '.join(refused)}"
        )
        print(f"         {len(rows)} row(s) were scored.")
    if write:
        table_path = Path(table_out)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        with table_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(TABLE_COLUMNS))
            writer.writeheader()
            writer.writerows(rows)
        Path(summary_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if summary["warning"]:
            print(f"WARNING: {summary['warning']}")
    return summary
