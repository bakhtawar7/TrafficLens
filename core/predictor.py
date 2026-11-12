"""Scene-grouped cross-validation for predicting audited counting error.

The predictor consumes inference-only feature rows and evaluator-produced audited
errors. All preprocessing and model fitting is performed inside each training
fold; scenes are kept intact in validation and bootstrap resampling.
"""
from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from . import config, dataset, evaluation, features

RIDGE_ALPHA = 1.0
BOOSTING_ITERATIONS = 10
BOOSTING_DEPTH = 2
BOOSTING_LEARNING_RATE = 0.05
MAX_FOLDS = 5
BOOTSTRAP_SAMPLES = 2000
PRIMARY_FLAG_FRACTION = 0.25
H1_FLAG_FRACTION = 0.20

MODEL_NAMES = ("ridge", "gradient_boosted", "mean_confidence", "training_mean")
CONFIDENCE_COLUMNS = tuple(
    f"{key}_mean_confidence" for key in config.MODEL_REGISTRY
)


@dataclass(frozen=True)
class PredictorRow:
    """One image's inference features and evaluator-verified target."""

    item_id: str
    scene: str
    target: float
    values: dict[str, float | None]


def _number(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def load_predictor_rows(
    features_csv: Path,
    evaluation_csv: Path,
    sources_csv: Path,
    development_ids: set[int] | None = None,
) -> list[PredictorRow]:
    """Join inference features to evaluator ``Actual Error`` and scene groups."""
    excluded = development_ids or set()
    feature_rows: dict[int, dict[str, float | None]] = {}
    with Path(features_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            number = evaluation.normalise_item_id(row.get("item_id", ""))
            if number is None:
                continue
            feature_rows[number] = {
                column: _number(row.get(column))
                for column in features.TABLE_COLUMNS[2:]
            }

    scenes = evaluation.load_scenes(Path(sources_csv))
    result: list[PredictorRow] = []
    with Path(evaluation_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            number = evaluation.normalise_item_id(row.get("Item", ""))
            target = _number(row.get("Actual Error"))
            if number is None or number in excluded or target is None or number not in feature_rows:
                continue
            item_id = dataset.format_item_id(number)
            scene = str(scenes.get(number) or row.get("Scene") or f"#{item_id}").strip()
            result.append(PredictorRow(item_id, scene or f"#{item_id}", target, feature_rows[number]))
    result.sort(key=lambda item: item.item_id)
    if len({row.item_id for row in result}) != len(result):
        raise ValueError("evaluator table contains duplicate item ids")
    return result


def scene_group_folds(
    scenes: list[str], n_splits: int = MAX_FOLDS, seed: int = config.SEED
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Create deterministic size-balanced folds without splitting any scene."""
    groups: dict[str, list[int]] = {}
    for index, scene in enumerate(scenes):
        groups.setdefault(scene or f"#{index}", []).append(index)
    if len(groups) < 2:
        raise ValueError("scene-grouped validation requires at least two distinct scenes")
    fold_count = min(max(2, n_splits), len(groups))
    rng = random.Random(seed)
    shuffled = list(groups.items())
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda pair: len(pair[1]), reverse=True)
    buckets: list[list[int]] = [[] for _ in range(fold_count)]
    for scene, indices in shuffled:
        target_fold = min(range(fold_count), key=lambda i: (len(buckets[i]), i))
        buckets[target_fold].extend(indices)
    all_indices = set(range(len(scenes)))
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for bucket in buckets:
        test = np.array(sorted(bucket), dtype=int)
        train = np.array(sorted(all_indices.difference(bucket)), dtype=int)
        folds.append((train, test))
    return folds


def _matrix(rows: list[PredictorRow]) -> tuple[np.ndarray, list[str]]:
    columns = list(features.TABLE_COLUMNS[2:])
    values = np.array([
        [np.nan if row.values.get(column) is None else float(row.values[column]) for column in columns]
        for row in rows
    ], dtype=float)
    return values, columns


def _impute_and_scale(
    train: np.ndarray, other: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    medians = np.zeros(train.shape[1], dtype=float)
    for column in range(train.shape[1]):
        present = train[:, column][np.isfinite(train[:, column])]
        medians[column] = float(np.median(present)) if len(present) else 0.0
    train_filled = np.where(np.isfinite(train), train, medians)
    other_filled = np.where(np.isfinite(other), other, medians)
    means = train_filled.mean(axis=0)
    scales = train_filled.std(axis=0)
    scales[scales == 0] = 1.0
    return (train_filled - means) / scales, (other_filled - means) / scales


def _ridge_predict(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, alpha: float = RIDGE_ALPHA
) -> np.ndarray:
    x_train, x_test = _impute_and_scale(train_x, test_x)
    y_mean = float(train_y.mean())
    centered = train_y - y_mean
    weights = np.linalg.solve(
        x_train.T @ x_train + alpha * np.eye(x_train.shape[1]), x_train.T @ centered
    )
    return np.maximum(0.0, x_test @ weights + y_mean)


def _best_split(
    x: np.ndarray, residuals: np.ndarray, min_leaf: int = 2
) -> tuple[int, float] | None:
    best: tuple[int, float] | None = None
    best_loss = float(np.sum((residuals - residuals.mean()) ** 2))
    for column in range(x.shape[1]):
        unique = np.unique(x[:, column])
        if len(unique) < 2:
            continue
        if len(unique) > 17:
            thresholds = np.unique(np.quantile(unique, np.linspace(0.1, 0.9, 9)))
        else:
            thresholds = (unique[:-1] + unique[1:]) / 2
        for threshold in thresholds:
            left = x[:, column] <= threshold
            if left.sum() < min_leaf or (~left).sum() < min_leaf:
                continue
            left_mean = float(residuals[left].mean())
            right_mean = float(residuals[~left].mean())
            loss = float(np.sum((residuals[left] - left_mean) ** 2)
                         + np.sum((residuals[~left] - right_mean) ** 2))
            if loss < best_loss - 1e-12:
                best_loss = loss
                best = (column, float(threshold))
    return best


@dataclass(frozen=True)
class _Tree:
    value: float | None = None
    feature: int | None = None
    threshold: float | None = None
    left: _Tree | None = None
    right: _Tree | None = None


def _fit_tree(x: np.ndarray, residuals: np.ndarray, depth: int) -> _Tree:
    if depth <= 0 or len(residuals) < 4:
        return _Tree(value=float(residuals.mean()))
    split = _best_split(x, residuals)
    if split is None:
        return _Tree(value=float(residuals.mean()))
    column, threshold = split
    left_mask = x[:, column] <= threshold
    if left_mask.sum() < 2 or (~left_mask).sum() < 2:
        return _Tree(value=float(residuals.mean()))
    return _Tree(
        feature=column,
        threshold=threshold,
        left=_fit_tree(x[left_mask], residuals[left_mask], depth - 1),
        right=_fit_tree(x[~left_mask], residuals[~left_mask], depth - 1),
    )


def _tree_values(tree: _Tree, x: np.ndarray) -> np.ndarray:
    if tree.value is not None:
        return np.full(len(x), tree.value, dtype=float)
    assert tree.feature is not None and tree.threshold is not None
    assert tree.left is not None and tree.right is not None
    left_mask = x[:, tree.feature] <= tree.threshold
    values = np.empty(len(x), dtype=float)
    values[left_mask] = _tree_values(tree.left, x[left_mask])
    values[~left_mask] = _tree_values(tree.right, x[~left_mask])
    return values


def _fit_boosted(
    x: np.ndarray, y: np.ndarray, iterations: int = BOOSTING_ITERATIONS
) -> tuple[float, list[_Tree]]:
    prediction = float(y.mean())
    trees: list[_Tree] = []
    for _ in range(iterations):
        residuals = y - prediction
        tree = _fit_tree(x, residuals, BOOSTING_DEPTH)
        if tree.value is not None and abs(tree.value) < 1e-12:
            break
        trees.append(tree)
        prediction += BOOSTING_LEARNING_RATE * _tree_values(tree, x)
    return float(y.mean()), trees


def _boosted_predict(
    train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray
) -> np.ndarray:
    x_train, x_test = _impute_and_scale(train_x, test_x)
    baseline, trees = _fit_boosted(x_train, train_y)
    prediction = np.full(len(test_x), baseline, dtype=float)
    for tree in trees:
        prediction += BOOSTING_LEARNING_RATE * _tree_values(tree, x_test)
    return np.maximum(0.0, prediction)


def _confidence_matrix(x: np.ndarray, columns: list[str]) -> np.ndarray:
    selected = [columns.index(column) for column in CONFIDENCE_COLUMNS if column in columns]
    if not selected:
        raise ValueError("features table has no per-model mean confidence columns")
    confidence = x[:, selected]
    valid = np.isfinite(confidence)
    counts = valid.sum(axis=1)
    means = np.divide(
        np.where(valid, confidence, 0.0).sum(axis=1),
        counts,
        out=np.full(len(confidence), np.nan, dtype=float),
        where=counts > 0,
    )
    return means.reshape(-1, 1)


def _cv_predictions(
    x: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    columns: list[str],
) -> dict[str, np.ndarray]:
    predictions = {name: np.full(len(y), np.nan, dtype=float) for name in MODEL_NAMES}
    confidence = _confidence_matrix(x, columns)
    for train, test in folds:
        train_y = y[train]
        predictions["ridge"][test] = _ridge_predict(x[train], train_y, x[test])
        predictions["gradient_boosted"][test] = _boosted_predict(x[train], train_y, x[test])
        predictions["mean_confidence"][test] = _ridge_predict(
            confidence[train], train_y, confidence[test]
        )
        predictions["training_mean"][test] = max(0.0, float(train_y.mean()))
    if any(not np.all(np.isfinite(value)) for value in predictions.values()):
        raise RuntimeError("cross-validation failed to produce a finite out-of-fold prediction")
    return predictions


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    """Spearman rank correlation with average ranks for ties."""
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    ranked_x = _rankdata(x)
    ranked_y = _rankdata(y)
    if np.std(ranked_x) == 0 or np.std(ranked_y) == 0:
        return None
    return float(np.corrcoef(ranked_x, ranked_y)[0, 1])


def _cluster_draws(scenes: list[str], rng: random.Random) -> list[int]:
    members: dict[str, list[int]] = {}
    for index, scene in enumerate(scenes):
        members.setdefault(scene or f"#{index}", []).append(index)
    keys = list(members)
    selected: list[int] = []
    for _ in keys:
        selected.extend(members[rng.choice(keys)])
    return selected


def _cluster_ci(
    size: int, scenes: list[str], statistic: Callable[[np.ndarray], float | None],
    samples: int, seed: int,
) -> dict[str, float | None]:
    point = statistic(np.arange(size, dtype=int))
    if point is None or samples <= 0 or len(set(scenes)) < 2:
        return {"estimate": point, "low": None, "high": None}
    rng = random.Random(seed)
    replicates: list[float] = []
    for _ in range(samples):
        indices = np.asarray(_cluster_draws(scenes, rng), dtype=int)
        value = statistic(indices)
        if value is not None and math.isfinite(value):
            replicates.append(value)
    if not replicates:
        return {"estimate": point, "low": None, "high": None}
    return {
        "estimate": point,
        "low": float(np.quantile(replicates, 0.025)),
        "high": float(np.quantile(replicates, 0.975)),
    }


def _ci(
    values: list[float], scenes: list[str], statistic: Callable[[np.ndarray], float | None],
    samples: int, seed: int,
) -> dict[str, float | None]:
    return _cluster_ci(len(values), scenes, statistic, samples, seed)


def _mae_ci(
    actual: np.ndarray, predicted: np.ndarray, scenes: list[str], samples: int, seed: int
) -> dict[str, float | None]:
    return _ci(
        list(range(len(actual))), scenes,
        lambda indices: float(np.mean(np.abs(actual[indices] - predicted[indices]))),
        samples, seed,
    )


def _spearman_ci(
    actual: np.ndarray, predicted: np.ndarray, scenes: list[str], samples: int, seed: int
) -> dict[str, float | None]:
    return _ci(
        list(range(len(actual))), scenes,
        lambda indices: spearman(predicted[indices], actual[indices]),
        samples, seed,
    )


def _top_indices(values: np.ndarray, fraction: float, item_ids: list[str]) -> set[int]:
    count = max(1, math.ceil(len(values) * fraction))
    return set(sorted(range(len(values)), key=lambda i: (-values[i], item_ids[i]))[:count])


def _flag_metrics(
    actual: np.ndarray, predicted: np.ndarray, item_ids: list[str], fraction: float
) -> dict[str, float]:
    flagged = _top_indices(predicted, fraction, item_ids)
    true_worst = _top_indices(actual, fraction, item_ids)
    true_positive = len(flagged & true_worst)
    unflagged = set(range(len(actual))) - flagged
    return {
        "cutoff_fraction": fraction,
        "flagged_count": len(flagged),
        "precision": true_positive / len(flagged),
        "recall": true_positive / len(true_worst),
        "mean_true_error_flagged": float(np.mean([actual[index] for index in flagged])),
        "mean_true_error_unflagged": (
            float(np.mean([actual[index] for index in unflagged])) if unflagged else 0.0
        ),
    }


def _flag_ci(
    actual: np.ndarray,
    predicted: np.ndarray,
    scenes: list[str],
    item_ids: list[str],
    fraction: float,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float | None]]:
    names = (
        "precision", "recall", "mean_true_error_flagged", "mean_true_error_unflagged",
    )
    result: dict[str, dict[str, float | None]] = {}
    for offset, name in enumerate(names):
        result[name] = _ci(
            list(range(len(actual))), scenes,
            lambda indices, metric=name: _flag_metrics(
                actual[indices], predicted[indices],
                [item_ids[index] for index in indices], fraction,
            )[metric],
            samples, seed + offset,
        )
    return result


def evaluate_rows(
    rows: list[PredictorRow],
    n_splits: int = MAX_FOLDS,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
    seed: int = config.SEED,
) -> dict[str, Any]:
    """Fit out-of-fold predictions, scene-bootstrap metrics, flags, and H1/H2 decisions."""
    if len(rows) < 2:
        raise ValueError("at least two labelled evaluation images are required")
    scenes = [row.scene for row in rows]
    folds = scene_group_folds(scenes, n_splits, seed)
    x, columns = _matrix(rows)
    actual = np.asarray([row.target for row in rows], dtype=float)
    item_ids = [row.item_id for row in rows]
    predictions = _cv_predictions(x, actual, folds, columns)

    model_results: dict[str, Any] = {}
    for model_name, predicted in predictions.items():
        mae = _mae_ci(actual, predicted, scenes, bootstrap_samples, seed)
        rho = _spearman_ci(actual, predicted, scenes, bootstrap_samples, seed + 1)
        flags = _flag_metrics(actual, predicted, item_ids, PRIMARY_FLAG_FRACTION)
        flag_intervals = _flag_ci(
            actual, predicted, scenes, item_ids, PRIMARY_FLAG_FRACTION,
            bootstrap_samples, seed + 2,
        )
        model_results[model_name] = {
            "mae": mae,
            "spearman_predicted_vs_actual": rho,
            "needs_review": {"primary_rule": flags, "scene_bootstrap_95ci": flag_intervals},
        }

    comparisons: dict[str, Any] = {}
    for model_name in ("ridge", "gradient_boosted"):
        comparisons[model_name] = {}
        for baseline in ("mean_confidence", "training_mean"):
            comparisons[model_name][baseline] = _cluster_ci(
                len(actual), scenes,
                lambda indices, model=model_name, base=baseline: float(
                    np.mean(np.abs(actual[indices] - predictions[base][indices]))
                    - np.mean(np.abs(actual[indices] - predictions[model][indices]))
                ),
                bootstrap_samples, seed + 3,
            )

    confidence = predictions["mean_confidence"]
    ridge = predictions["ridge"]
    h1_rho_difference = _cluster_ci(
        len(actual), scenes,
        lambda indices: (
            None if spearman(ridge[indices], actual[indices]) is None
            or spearman(confidence[indices], actual[indices]) is None
            else float(spearman(ridge[indices], actual[indices])
                       - spearman(confidence[indices], actual[indices]))
        ),
        bootstrap_samples, seed + 5,
    )
    ridge_worst20 = _flag_metrics(actual, ridge, item_ids, H1_FLAG_FRACTION)
    confidence_worst20 = _flag_metrics(actual, confidence, item_ids, H1_FLAG_FRACTION)
    h1_margin_met = (
        h1_rho_difference["estimate"] is not None
        and h1_rho_difference["estimate"] >= 0.1
        and h1_rho_difference["low"] is not None
        and h1_rho_difference["low"] > 0
    )
    h1_capture_met = ridge_worst20["recall"] > confidence_worst20["recall"]

    cross = np.asarray([row.values.get("cross_family_count_disagreement", np.nan) for row in rows], dtype=float)
    within = np.asarray([row.values.get("within_family_count_disagreement", np.nan) for row in rows], dtype=float)
    valid_h2 = np.isfinite(cross) & np.isfinite(within)
    if int(valid_h2.sum()) >= 2:
        cross_rho = _spearman_ci(actual[valid_h2], cross[valid_h2],
                                 [scene for scene, valid in zip(scenes, valid_h2) if valid],
                                 bootstrap_samples, seed + 6)
        within_rho = _spearman_ci(actual[valid_h2], within[valid_h2],
                                  [scene for scene, valid in zip(scenes, valid_h2) if valid],
                                  bootstrap_samples, seed + 7)
        valid_scenes = [scene for scene, valid in zip(scenes, valid_h2) if valid]
        h2_actual = actual[valid_h2]
        h2_cross = cross[valid_h2]
        h2_within = within[valid_h2]
        h2_difference = _cluster_ci(
            len(h2_actual), valid_scenes,
            lambda indices: (
                None if spearman(h2_cross[indices], h2_actual[indices]) is None
                or spearman(h2_within[indices], h2_actual[indices]) is None
                else float(spearman(h2_cross[indices], h2_actual[indices])
                           - spearman(h2_within[indices], h2_actual[indices]))
            ),
            bootstrap_samples, seed + 8,
        )
    else:
        cross_rho = within_rho = h2_difference = {"estimate": None, "low": None, "high": None}

    assignments: list[dict[str, Any]] = []
    for fold_number, (train, test) in enumerate(folds, 1):
        assignments.append({
            "fold": fold_number,
            "train_scenes": sorted({scenes[index] for index in train}),
            "validation_scenes": sorted({scenes[index] for index in test}),
        })

    return {
        "sample_size": len(rows),
        "exploratory": len(rows) < 60,
        "validation": {
            "method": "deterministic, size-balanced scene-grouped cross-validation",
            "folds": len(folds),
            "hyperparameters": {
                "ridge_alpha": RIDGE_ALPHA,
                "boosting_iterations": BOOSTING_ITERATIONS,
                "boosting_max_depth": BOOSTING_DEPTH,
                "boosting_learning_rate": BOOSTING_LEARNING_RATE,
                "tuning": "none; fixed before validation",
                "preprocessing": "training-fold median imputation and standardisation",
            },
            "assignments": assignments,
        },
        "models": model_results,
        "paired_mae_improvement_vs_baselines": comparisons,
        "hypotheses": {
            "H1": {
                "primary_model": "ridge",
                "ridge_minus_mean_confidence_spearman": h1_rho_difference,
                "ridge_worst_20_percent_capture": ridge_worst20,
                "mean_confidence_worst_20_percent_capture": confidence_worst20,
                "reject": not (h1_margin_met and h1_capture_met),
                "rule": "support only if Spearman improvement is >=0.1 with a scene-bootstrap interval excluding zero and the learned model captures more worst-20% images at the same budget",
            },
            "H2": {
                "correlation": "Spearman on held-out joined items",
                "cross_family_disagreement_vs_error": cross_rho,
                "within_family_disagreement_vs_error": within_rho,
                "cross_minus_within_spearman": h2_difference,
                "reject": not (
                    h2_difference["estimate"] is not None
                    and h2_difference["estimate"] > 0
                    and h2_difference["low"] is not None
                    and h2_difference["low"] > 0
                ),
                "rule": "support only if cross-family correlation is higher and the scene-bootstrap interval for the difference excludes zero",
            },
        },
    }


def permutation_skill_test(
    rows: list[PredictorRow],
    permutations: int = 99,
    seed: int = config.SEED,
    n_splits: int = MAX_FOLDS,
) -> dict[str, float | int]:
    """One-sided permutation test of ridge MAE skill over the confidence baseline."""
    scenes = [row.scene for row in rows]
    folds = scene_group_folds(scenes, n_splits, seed)
    x, columns = _matrix(rows)
    actual = np.asarray([row.target for row in rows], dtype=float)
    observed_predictions = _cv_predictions(x, actual, folds, columns)

    def skill(target: np.ndarray, predicted: dict[str, np.ndarray]) -> float:
        return float(
            np.mean(np.abs(target - predicted["mean_confidence"]))
            - np.mean(np.abs(target - predicted["ridge"]))
        )

    observed = skill(actual, observed_predictions)
    rng = np.random.default_rng(seed)
    null_skills: list[float] = []
    for _ in range(permutations):
        shuffled = rng.permutation(actual)
        null_predictions = _cv_predictions(x, shuffled, folds, columns)
        null_skills.append(skill(shuffled, null_predictions))
    p_value = (1 + sum(value >= observed for value in null_skills)) / (permutations + 1)
    return {
        "observed_mae_improvement": observed,
        "permutations": permutations,
        "one_sided_p_value": p_value,
    }


def audited_label_readiness(
    sources_csv: Path, labels_csv: Path, plan_json: Path
) -> dict[str, Any]:
    """Identify exact missing audited first-pass labels for evaluation and development."""
    source_ids: set[int] = set()
    with Path(sources_csv).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            item = evaluation.normalise_item_id(row.get("item_id", ""))
            if item is not None:
                source_ids.add(item)
    excluded = evaluation.load_development_ids(Path(plan_json))
    labels = evaluation.load_labels(Path(labels_csv))
    audited = {item for item, passes in labels.items() if passes.get("pass1") is not None}
    evaluation_ids = source_ids - excluded
    development_missing = excluded - audited
    return {
        "evaluation_total": len(evaluation_ids),
        "evaluation_audited": len(evaluation_ids & audited),
        "evaluation_audited_ids": [
            dataset.format_item_id(item) for item in sorted(evaluation_ids & audited)
        ],
        "evaluation_missing_ids": [
            dataset.format_item_id(item) for item in sorted(evaluation_ids - audited)
        ],
        "development_total": len(excluded),
        "development_missing_ids": [
            dataset.format_item_id(item) for item in sorted(development_missing)
        ],
    }
