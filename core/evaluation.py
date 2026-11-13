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
from the written table. Confidence intervals are scene-level (cluster) percentile
bootstraps: whole scenes from the ``scene`` column of SOURCES.csv are resampled
with replacement, because images drawn from one scene are not independent. The
seed (``config.SEED``) and resample count (2000 by default) are unchanged, so the
intervals stay deterministic across runs.

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

from . import config, dataset, settings
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

#: Exact column order of results/evaluation_table.csv. The first ten are the
#: original Week 4 columns; the rest are the count-audit additions (scene, the
#: all-vehicles ground truth, one signed error per model per target, and the
#: exploratory-condition flag). Existing columns keep their names and positions.
TABLE_COLUMNS: tuple[str, ...] = (
    "Item", "Condition", "Ground Truth", "YOLOv8n", "YOLOv8s", "RT-DETR-L",
    "Sliced Count", "Count Range", "Agreement", "Actual Error",
    "Scene", "All Vehicles Ground Truth",
    "YOLOv8n Covered Signed Error", "YOLOv8s Covered Signed Error", "RT-DETR-L Covered Signed Error",
    "YOLOv8n All Vehicles Signed Error", "YOLOv8s All Vehicles Signed Error", "RT-DETR-L All Vehicles Signed Error",
    "Exploratory Conditions",
)

#: A condition tag carried by fewer than this many evaluation images is reported as
#: exploratory: the sample is too small for the figure to carry inferential weight.
EXPLORATORY_MIN_IMAGES: int = 15

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


def load_adjudications(path: Path) -> dict[int, dict[str, dict[str, int]]]:
    """Load corrected and preserved original counts from explicitly adjudicated rows.

    An adjudicated row stores corrected counts in the regular class columns and
    the pre-adjudication values in ``original_<class>`` columns. Requiring both
    values prevents adjudication from silently discarding the submitted label.
    """
    records: dict[int, dict[str, dict[str, int]]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("adjudicated", "")).strip().lower() != "yes":
                continue
            item = normalise_item_id(row.get("item_id", ""))
            if item is None:
                continue
            pass_number = _int_or_none(row.get("labeller_pass"))
            if pass_number not in (None, 1):
                raise ValueError(
                    f"adjudicated row for {dataset.format_item_id(item)} must be pass 1"
                )
            if item in records:
                raise ValueError(f"multiple adjudicated rows found for {dataset.format_item_id(item)}")
            original = {
                name: _int_or_none(row.get(f"original_{name}"))
                for name in config.LABEL_CLASS_NAMES
            }
            corrected = {name: _int_or_none(row.get(name)) for name in config.LABEL_CLASS_NAMES}
            if any(value is None for value in original.values()):
                raise ValueError(
                    f"adjudicated row for {dataset.format_item_id(item)} must preserve "
                    "every category in original_<category> columns"
                )
            if any(value is None for value in corrected.values()):
                raise ValueError(
                    f"adjudicated row for {dataset.format_item_id(item)} must contain "
                    "every corrected category count"
                )
            records[item] = {
                "original": {name: int(value) for name, value in original.items()},
                "corrected": {name: int(value) for name, value in corrected.items()},
            }
    return records


def restore_original_labels(
    labels: dict[int, dict[str, dict[str, int] | None]],
    adjudications: dict[int, dict[str, dict[str, int]]],
) -> dict[int, dict[str, dict[str, int] | None]]:
    """Copy loaded labels and restore original values for an unadjudicated comparison."""
    original_labels = {
        item: dict(passes)
        for item, passes in labels.items()
    }
    for item, record in adjudications.items():
        if item not in original_labels or original_labels[item].get("pass1") is None:
            raise ValueError(
                f"adjudicated row for {dataset.format_item_id(item)} has no first-pass label"
            )
        original_labels[item]["pass1"] = dict(record["original"])
    return original_labels


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


def load_scenes(path: Path) -> dict[int, str]:
    """Read the scene group of every item from the ``scene`` column of SOURCES.csv.

    The scene is the cluster unit for the scene-level bootstrap: images copied from one
    burst share a scene label and are not statistically independent. Items whose row has
    no scene (or a SOURCES file written before the column existed) simply stay out of the
    mapping and are later treated as their own single-image cluster.
    """
    scenes: dict[int, str] = {}
    if not Path(path).exists():
        return scenes
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            item = normalise_item_id(row.get("item_id", ""))
            scene = (row.get("scene") or "").strip()
            if item is not None and scene:
                scenes[item] = scene
    return scenes


def load_development_ids(path: Path) -> set[int]:
    """Item numbers in the seeded development subset, excluded from all outputs."""
    if not path.exists():
        return set()
    plan = json.loads(path.read_text(encoding="utf-8"))
    return {n for n in (normalise_item_id(v) for v in plan.get("development_subset", [])) if n is not None}


def load_detection_cache(
    path: Path,
    confidence: float | dict[str, float] = config.DEFAULT_CONFIDENCE,
    settings_sha256: str | None = None,
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
            if settings_sha256 is not None and entry.get("settings_sha256") != settings_sha256:
                continue
            item = normalise_item_id(str(entry.get("image", "")).split("_")[0])
            if item is None:
                continue
            model_key = str(entry["model"])
            model_confidence = (
                confidence.get(model_key, config.DEFAULT_CONFIDENCE)
                if isinstance(confidence, dict)
                else confidence
            )
            detections = [
                Detection(int(d["class_id"]), str(d["class_name"]), float(d["confidence"]),
                          tuple(float(v) for v in d["xyxy"]))
                for d in entry.get("detections", [])
                if d["class_name"] in COVERED_CLASSES
                and float(d["confidence"]) >= model_confidence
            ]
            cache[(item, model_key, str(entry["mode"]))] = detections
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


def cluster_bootstrap_ci(
    values: list[float], clusters: list[str], samples: int = 2000, seed: int = config.SEED
) -> tuple[float | None, float | None]:
    """Scene-level (cluster) percentile bootstrap 95% CI of the mean of ``values``.

    ``values[i]`` belongs to the cluster named by ``clusters[i]`` (the item's scene). A
    resample draws whole clusters with replacement -- exactly as many clusters as the input
    contains -- and the resampled mean is taken over every value in the drawn clusters,
    counting a cluster's contribution once per time it is picked. Images from one scene are
    correlated, so resampling scenes rather than single pixels keeps that dependence in the
    interval and stops the CI from being falsely tight. With no scenes to group by, every
    item is its own cluster and the result matches :func:`bootstrap_ci` up to RNG order.
    Empty input yields ``(None, None)``.
    """
    if not values:
        return (None, None)
    groups: dict[str, list[float]] = {}
    for value, cluster in zip(values, clusters):
        groups.setdefault(cluster, []).append(float(value))
    keys = list(groups)  # first-appearance order, so the draw is deterministic
    count = len(keys)
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean([v for key in (keys[rng.randrange(count)] for _ in keys) for v in groups[key]])
        for _ in range(samples)
    )
    lower = means[int(0.025 * samples)]
    upper = means[min(int(0.975 * samples), samples - 1)]
    return (round(lower, 4), round(upper, 4))


def labeller_pass_statistics(
    labels: dict[int, dict[str, dict[str, int] | None]],
    scenes: dict[int, str] | None = None,
    excluded: set[int] | None = None,
    samples: int = 2000,
) -> dict[str, Any]:
    """Compare pass 2 with pass 1 by category and covered total.

    Differences are signed (pass 2 minus pass 1); the reported bootstrap
    intervals are scene-cluster intervals over the signed and absolute means.
    """
    scenes = scenes or {}
    excluded = excluded or set()
    both = [
        (item, data)
        for item, data in labels.items()
        if data.get("pass1") is not None
        and data.get("pass2") is not None
        and item not in excluded
    ]
    category_differences: dict[str, list[int]] = {
        name: [] for name in config.LABEL_CLASS_NAMES
    }
    covered_differences: list[int] = []
    clusters: list[str] = []
    item_differences: list[dict[str, Any]] = []
    for item, data in sorted(both):
        first = data["pass1"]
        second = data["pass2"]
        differences = {
            name: second.get(name, 0) - first.get(name, 0)
            for name in config.LABEL_CLASS_NAMES
        }
        covered_difference = covered_total(second) - covered_total(first)
        for name, difference in differences.items():
            category_differences[name].append(difference)
        covered_differences.append(covered_difference)
        scene = scenes.get(item, f"#{dataset.format_item_id(item)}")
        clusters.append(scene)
        item_differences.append({
            "item_id": dataset.format_item_id(item),
            "scene": scene,
            "categories": differences,
            "covered_total_difference": covered_difference,
        })

    def describe(differences: list[int], cluster_ids: list[str]) -> dict[str, Any]:
        absolute = [abs(value) for value in differences]
        return {
            "n": len(differences),
            "mean_difference": round(statistics.fmean(differences), 4) if differences else None,
            "mean_difference_ci95": cluster_bootstrap_ci(
                differences, cluster_ids, samples
            ),
            "mean_absolute_difference": round(statistics.fmean(absolute), 4) if absolute else None,
            "mean_absolute_difference_ci95": cluster_bootstrap_ci(
                absolute, cluster_ids, samples
            ),
            "exact_match_fraction": (
                round(sum(value == 0 for value in differences) / len(differences), 4)
                if differences else None
            ),
        }

    per_category = {
        name: describe(values, clusters)
        for name, values in category_differences.items()
    }
    return {
        "items_both_passes": len(both),
        "per_category": per_category,
        "covered_total": describe(covered_differences, clusters),
        "items": item_differences,
    }


def error_stats(
    pairs: list[tuple[int, int]],
    clusters: list[str],
    samples: int = 2000,
    seed: int = config.SEED,
) -> dict[str, Any]:
    """Full signed-error profile for one (prediction, ground-truth) pairing.

    Signed error is ``prediction - ground_truth``, so a negative value is an undercount and
    a positive value an overcount. The returned mapping keeps the historic ``mae``, ``ci95``
    and ``n`` keys (the interval is now the scene-level cluster bootstrap of the absolute
    error) and adds the count-audit measures:

    * ``bias``: mean signed error (systematic under- or over-counting);
    * ``undercounted`` / ``overcounted`` / ``exact``: how many images fall on each side;
    * ``mean_relative_error``: mean of ``|prediction - ground_truth| / ground_truth`` over
      images whose ground truth is positive. Images with a ground truth of zero are not
      divided by zero: they are excluded from this mean and reported separately as
      ``zero_true_items`` (a zero-truth image is still scored under/over/exact above, so the
      absolute bias is not lost). ``relative_error_items`` is the count the mean used.

    An empty input returns a zero-count profile with the point estimates set to ``None``.
    """
    n = len(pairs)
    if n == 0:
        return {
            "mae": None, "ci95": (None, None), "n": 0, "bias": None,
            "undercounted": 0, "overcounted": 0, "exact": 0,
            "mean_relative_error": None, "relative_error_items": 0, "zero_true_items": 0,
        }
    signed = [prediction - truth for prediction, truth in pairs]
    absolute = [abs(s) for s in signed]
    relative = [abs(p - g) / g for p, g in pairs if g > 0]
    return {
        "mae": round(statistics.fmean(absolute), 4),
        "ci95": cluster_bootstrap_ci(absolute, clusters, samples, seed),
        "n": n,
        "bias": round(statistics.fmean(signed), 4),
        "undercounted": sum(1 for s in signed if s < 0),
        "overcounted": sum(1 for s in signed if s > 0),
        "exact": sum(1 for s in signed if s == 0),
        "mean_relative_error": round(statistics.fmean(relative), 4) if relative else None,
        "relative_error_items": len(relative),
        "zero_true_items": sum(1 for _, g in pairs if g == 0),
    }


def _mean_abs_error(pairs: list[tuple[int, int]]) -> float:
    """Mean absolute error over (prediction, ground truth) pairs."""
    return round(statistics.fmean(abs(p - g) for p, g in pairs), 4)


def _row_tags(row: dict[str, Any]) -> list[str]:
    """The cleaned condition tags of one table row."""
    return [tag.strip() for tag in str(row.get("Condition", "")).split(";") if tag.strip()]


def condition_image_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """How many evaluation images carry each condition tag (an item joins every tag it has)."""
    counts: dict[str, int] = {}
    for row in rows:
        for tag in _row_tags(row):
            counts[tag] = counts.get(tag, 0) + 1
    return counts


def condition_scene_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """How many distinct scenes each condition tag's evaluation images come from."""
    scenes: dict[str, set[str]] = {}
    for row in rows:
        scene = str(row.get("Scene", "")).strip() or f"#{row.get('Item', '')}"
        for tag in _row_tags(row):
            scenes.setdefault(tag, set()).add(scene)
    return {tag: len(members) for tag, members in scenes.items()}


def exploratory_tags(counts: dict[str, int], threshold: int = EXPLORATORY_MIN_IMAGES) -> set[str]:
    """Condition tags backed by fewer than ``threshold`` evaluation images."""
    return {tag for tag, count in counts.items() if count < threshold}


def annotate_exploratory(
    rows: list[dict[str, Any]], threshold: int = EXPLORATORY_MIN_IMAGES
) -> dict[str, int]:
    """Fill each row's ``Exploratory Conditions`` cell and return the per-tag image counts.

    A row lists the subset of its own tags that are exploratory, so the per-item table shows
    at a glance which conditions a given figure sits inside of.
    """
    counts = condition_image_counts(rows)
    exploratory = exploratory_tags(counts, threshold)
    for row in rows:
        row["Exploratory Conditions"] = ";".join(tag for tag in _row_tags(row) if tag in exploratory)
    return counts


def build_rows(
    labels: dict[int, dict[str, dict[str, int] | None]],
    conditions: dict[int, str],
    cache: dict[tuple[int, str, str], list[Detection]],
    excluded: set[int],
    sliced_model_key: str = config.DEFAULT_MODEL_NAME,
    scenes: dict[int, str] | None = None,
) -> tuple[list[dict[str, Any]], list[int]]:
    """Assemble one table row per evaluable item plus the ids with missing data.

    An item is evaluable when it has first-pass counts, is not in ``excluded``,
    and the cache holds standard entries for every registry model plus a sliced
    entry for ``sliced_model_key``.

    ``scenes`` maps an item number to its scene label; it feeds the ``Scene`` column and
    the scene-level bootstrap. Every model's signed error is written on both the covered
    and the all-vehicles target (``prediction - ground truth``), and the ``Exploratory
    Conditions`` cell is left blank here for :func:`annotate_exploratory` to fill once all
    rows exist.
    """
    scenes = scenes or {}
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
        gt_all = all_vehicles_total(first)
        values = [counts[key] for key in model_keys]
        pairwise = [
            box_agreement(standard[a], standard[b])
            for i, a in enumerate(model_keys)
            for b in model_keys[i + 1:]
        ]
        agreement = round(statistics.fmean(pairwise), 4) if pairwise else None
        actual_error = abs(gt_covered - round(statistics.median(values)))
        row: dict[str, Any] = {
            "Item": dataset.format_item_id(item),
            "Condition": conditions.get(item, ""),
            "Ground Truth": gt_covered,
            **{_MODEL_DISPLAY[key]: counts[key] for key in model_keys},
            "Sliced Count": sliced_count,
            "Count Range": f"{min(values)}-{max(values)}",
            "Agreement": agreement,
            "Actual Error": actual_error,
            "Scene": scenes.get(item, ""),
            "All Vehicles Ground Truth": gt_all,
            "Exploratory Conditions": "",
        }
        # Signed error per model on each target: prediction minus ground truth.
        for key in model_keys:
            display = _MODEL_DISPLAY[key]
            row[f"{display} Covered Signed Error"] = counts[key] - gt_covered
        for key in model_keys:
            display = _MODEL_DISPLAY[key]
            row[f"{display} All Vehicles Signed Error"] = counts[key] - gt_all
        rows.append({column: row[column] for column in TABLE_COLUMNS})
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
    scenes: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Compute every statistic that goes into ``results/summary.json``.

    ``refused_unverified`` is the list of canonical ids whose label rows were rejected for
    carrying unverified draft notes. It is recorded so the artefact states, without the
    reader having to infer it, which rows were deliberately not scored.

    ``scenes`` maps an item number to its scene label and, together with the ``Scene``
    column already on each row, supplies the cluster unit for the scene-level bootstrap.
    Every error interval produced here is a cluster bootstrap, so ``scenes`` (or the row
    scene) is what keeps correlated images from narrowing an interval unfairly.
    """
    scenes = scenes or {}
    refused = sorted(set(refused_unverified or []))
    evaluated = {row["Item"]: row for row in rows}
    labelled_non_dev = [
        item for item, data in labels.items()
        if data.get("pass1") is not None and item not in excluded
    ]
    scene_by_item = {
        normalise_item_id(row["Item"]): str(row.get("Scene", "")).strip() or f"#{row['Item']}"
        for row in rows
    }

    def cluster_of(number: int, canonical: str) -> str:
        """The scene (cluster) label for an item, falling back to a unique per-item token."""
        return scene_by_item.get(number) or scenes.get(number) or f"#{canonical}"

    model_mae: dict[str, Any] = {}
    for key, spec in config.MODEL_REGISTRY.items():
        display = _MODEL_DISPLAY[key]
        covered_pairs, all_pairs, clusters = [], [], []
        for row in rows:
            number = normalise_item_id(row["Item"])
            first = labels[number]["pass1"]
            prediction = row[display]
            covered_pairs.append((prediction, covered_total(first)))
            all_pairs.append((prediction, all_vehicles_total(first)))
            clusters.append(cluster_of(number, row["Item"]))
        model_mae[spec.display_name] = {
            "covered": error_stats(covered_pairs, clusters, bootstrap_samples),
            "all_vehicles": error_stats(all_pairs, clusters, bootstrap_samples),
        }

    tag_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for tag in _row_tags(row):
            tag_rows.setdefault(tag, []).append(row)
    image_counts = condition_image_counts(rows)
    scene_counts = condition_scene_counts(rows)
    condition_mae: dict[str, Any] = {}
    condition_detail: dict[str, Any] = {}
    for tag, trows in sorted(tag_rows.items()):
        per_model: dict[str, Any] = {}
        for key, spec in config.MODEL_REGISTRY.items():
            display = _MODEL_DISPLAY[key]
            pairs, clusters = [], []
            for row in trows:
                number = normalise_item_id(row["Item"])
                first = labels[number]["pass1"]
                pairs.append((row[display], covered_total(first)))
                clusters.append(cluster_of(number, row["Item"]))
            per_model[spec.display_name] = error_stats(pairs, clusters, bootstrap_samples)
        condition_mae[tag] = per_model
        n_images = image_counts.get(tag, 0)
        condition_detail[tag] = {
            "n_images": n_images,
            "scenes": scene_counts.get(tag, 0),
            "exploratory": n_images < EXPLORATORY_MIN_IMAGES,
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
    labeller_clusters = [cluster_of(item, dataset.format_item_id(item)) for item, _ in both]
    labeller = {
        "items_both_passes": len(both),
        "covered_mean_abs_difference": round(statistics.fmean(covered_differences), 4) if covered_differences else None,
        "covered_ci95": cluster_bootstrap_ci(covered_differences, labeller_clusters, bootstrap_samples),
        "all_vehicles_mean_abs_difference": round(statistics.fmean(all_differences), 4) if all_differences else None,
        "identical_fraction": round(sum(1 for d in covered_differences if d == 0) / len(covered_differences), 4) if covered_differences else None,
        "difference_detail": labeller_pass_statistics(
            labels, scenes=scenes, excluded=excluded, samples=bootstrap_samples
        ),
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
            "bootstrap_method": "scene-level cluster",
            "exploratory_min_images": EXPLORATORY_MIN_IMAGES,
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
        "condition_mae": condition_mae,
        "condition_detail": condition_detail,
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
    settings_json: Path | None = None,
) -> dict[str, Any]:
    """Run the full evaluation and return (and by default write) the summary.

    The development subset is excluded everywhere. Rows whose notes still describe the
    counts as an unverified auto derived draft are refused before any statistic runs, and
    the refusal is printed. Raises ValueError when ``sliced_model_key`` is not a registry
    key. ``settings_json`` names the settings file the run was produced under (default
    ``configs/default_settings.json``); its SHA-256, input sizes, and NMS fields are
    recorded in the summary so the statistics stay traceable to their configuration.
    """
    if sliced_model_key not in config.MODEL_REGISTRY:
        raise ValueError(f"unknown sliced model key: {sliced_model_key}")
    labels_path = Path(labels_csv)
    refused = unverified_label_rows(labels_path)
    labels = load_labels(labels_path)
    conditions = load_conditions(Path(sources_csv))
    scenes = load_scenes(Path(sources_csv))
    excluded = load_development_ids(Path(plan_json))
    selected_settings = settings.settings_record(settings_json)
    model_confidence = {
        key: float(record.get("confidence", confidence))
        for key, record in selected_settings["models"].items()
    }
    cache = load_detection_cache(
        Path(detections_jsonl),
        model_confidence if settings_json is not None else confidence,
        selected_settings["settings_sha256"] if settings_json is not None else None,
    )
    adjudications = load_adjudications(labels_path)
    rows, missing = build_rows(labels, conditions, cache, excluded, sliced_model_key, scenes)
    # Flag exploratory conditions on the rows before anything is written or summarised.
    annotate_exploratory(rows)
    summary = summarise(labels, conditions, rows, missing, excluded, confidence,
                        sliced_model_key, bootstrap_samples, refused_unverified=refused,
                        scenes=scenes)
    summary["settings"].update(selected_settings)
    summary["settings"]["model_confidence"] = model_confidence
    evaluated_adjudications = {
        item: record
        for item, record in adjudications.items()
        if item not in excluded and labels.get(item, {}).get("pass1") is not None
    }
    inconsistent_adjudications = [
        item for item, record in evaluated_adjudications.items()
        if labels[item]["pass1"] != record["corrected"]
    ]
    if inconsistent_adjudications:
        formatted = ", ".join(
            dataset.format_item_id(item) for item in sorted(inconsistent_adjudications)
        )
        raise ValueError(
            "adjudicated corrected counts must match the first-pass category columns: "
            + formatted
        )
    labels_without_adjudications = restore_original_labels(labels, evaluated_adjudications)
    if evaluated_adjudications:
        rows_without_adjudications, missing_without_adjudications = build_rows(
            labels_without_adjudications, conditions, cache, excluded, sliced_model_key, scenes
        )
        annotate_exploratory(rows_without_adjudications)
        summary_without_adjudications = summarise(
            labels_without_adjudications,
            conditions,
            rows_without_adjudications,
            missing_without_adjudications,
            excluded,
            confidence,
            sliced_model_key,
            bootstrap_samples,
            scenes=scenes,
        )
    else:
        summary_without_adjudications = summary
    comparable_fields = ("items", "model_mae", "condition_mae", "condition_detail", "labeller_passes")
    summary["adjudication"] = {
        "adjudicated_rows": len(adjudications),
        "evaluation_adjudicated_rows": len(evaluated_adjudications),
        "preserved_values": [
            {
                "item_id": dataset.format_item_id(item),
                "original": record["original"],
                "corrected": record["corrected"],
            }
            for item, record in sorted(adjudications.items())
        ],
        "with_adjudicated": {key: summary[key] for key in comparable_fields},
        "without_adjudicated": {
            key: summary_without_adjudications[key] for key in comparable_fields
        },
    }
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
