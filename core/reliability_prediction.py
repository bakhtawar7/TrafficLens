"""Load validated needs-review evidence and score an uploaded image safely."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from . import config, features

VERDICTS = {"supported", "not supported", "inconclusive"}
UNVALIDATED_MESSAGE = (
    "Reliability prediction not yet validated. The signals above are measured but "
    "have not been tested against audited counts."
)
_SUMMARY_H1 = re.compile(
    r"\|\s*H1\s*\|\s*(supported|not supported|inconclusive)\s*\|",
    re.IGNORECASE,
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _summary_verdict(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _SUMMARY_H1.search(text)
    if not match:
        return None
    verdict = match.group(1).lower()
    return verdict if verdict in VERDICTS else None


def _report_verdict(report: dict[str, Any]) -> str | None:
    hypotheses = report.get("hypotheses")
    h1 = hypotheses.get("H1") if isinstance(hypotheses, dict) else None
    if isinstance(h1, dict):
        verdict = str(h1.get("verdict", "")).strip().lower()
        if verdict in VERDICTS:
            return verdict
    return None


def load_flag_configuration(
    predictor_path: Path | str | None = None,
    summary_path: Path | str | None = None,
) -> dict[str, Any] | None:
    """Return a usable stored flag configuration, or ``None`` if not validated.

    The final Week 3 summary verdict takes precedence over a predictor-report
    verdict. Old reports without a deployment model remain unvalidated rather
    than fabricating a per-image score from aggregate validation metrics.
    """
    predictor_file = Path(predictor_path or (config.RESULTS_DIR / "predictor_report.json"))
    summary_file = Path(summary_path or (config.RESULTS_DIR / "week3_summary.md"))
    report = _read_json(predictor_file) if predictor_file.is_file() else {}
    verdict = _summary_verdict(summary_file) if summary_file.is_file() else None
    verdict = verdict or _report_verdict(report)
    if verdict is None:
        return None

    deployment = report.get("deployment")
    if not isinstance(deployment, dict):
        return None
    if verdict == "supported":
        source = "ridge"
    else:
        policy = report.get("flag_policy")
        source = str(policy.get("fallback_baseline", "")).strip() if isinstance(policy, dict) else ""
        if source not in {"mean_confidence", "training_mean"}:
            return None
    model_data = deployment.get("models")
    model = model_data.get(source) if isinstance(model_data, dict) else None
    metrics_data = report.get("models")
    metrics_model = metrics_data.get(source) if isinstance(metrics_data, dict) else None
    needs_review = metrics_model.get("needs_review") if isinstance(metrics_model, dict) else None
    primary_rule = needs_review.get("primary_rule") if isinstance(needs_review, dict) else None
    intervals = needs_review.get("scene_bootstrap_95ci") if isinstance(needs_review, dict) else None
    if not isinstance(model, dict) or not isinstance(primary_rule, dict) or not isinstance(intervals, dict):
        return None
    try:
        threshold = float(model["flag_threshold"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(threshold):
        return None

    readiness = report.get("audited_label_readiness")
    audited = readiness.get("evaluation_audited") if isinstance(readiness, dict) else None
    if audited is None:
        audited = report.get("sample_size")
    try:
        audited_images = int(audited)
    except (TypeError, ValueError):
        return None
    if audited_images <= 0:
        return None

    return {
        "verdict": verdict,
        "source": source,
        "model": model,
        "threshold": threshold,
        "audited_images": audited_images,
        "exploratory": bool(report.get("exploratory", audited_images < 60)),
        "precision": primary_rule.get("precision"),
        "recall": primary_rule.get("recall"),
        "intervals": intervals,
        "flag_fraction": float(deployment.get("flag_fraction", 0.25)),
    }


def score_features(
    feature_row: dict[str, Any], configuration: dict[str, Any]
) -> dict[str, Any] | None:
    """Apply a stored linear deployment model and return score and contributions."""
    model = configuration.get("model")
    if not isinstance(model, dict):
        return None
    try:
        columns = list(model["feature_columns"])
        medians = [float(value) for value in model["imputation_medians"]]
        means = [float(value) for value in model["means"]]
        scales = [float(value) for value in model["scales"]]
        coefficients = [float(value) for value in model["coefficients"]]
        intercept = float(model["intercept"])
    except (KeyError, TypeError, ValueError):
        return None
    lengths = {len(columns), len(medians), len(means), len(scales), len(coefficients)}
    if len(lengths) != 1 or any(scale <= 0 or not math.isfinite(scale) for scale in scales):
        return None

    contributions: list[tuple[str, float]] = []
    prediction = intercept
    for column, median, mean, scale, coefficient in zip(
        columns, medians, means, scales, coefficients
    ):
        try:
            value = float(feature_row.get(column))
        except (TypeError, ValueError):
            value = median
        if not math.isfinite(value):
            value = median
        contribution = ((value - mean) / scale) * coefficient
        prediction += contribution
        contributions.append((column, contribution))
    prediction = max(0.0, prediction)
    top_features = [
        {"feature": name, "contribution": round(value, 6)}
        for name, value in sorted(contributions, key=lambda pair: abs(pair[1]), reverse=True)[:3]
    ]
    return {
        "predicted_error": prediction,
        "flagged": prediction >= float(configuration["threshold"]),
        "top_features": top_features,
    }


def format_interval(metrics: dict[str, Any], name: str) -> str:
    """Format a stored lower/upper interval without converting it to accuracy."""
    interval = metrics.get(name)
    if not isinstance(interval, dict):
        return "not available"
    low, high = interval.get("low"), interval.get("high")
    if low is None or high is None:
        return "not available"
    try:
        return f"{float(low):.3f} to {float(high):.3f}"
    except (TypeError, ValueError):
        return "not available"


def feature_names() -> tuple[str, ...]:
    """Names of the features used by the shared image feature computation."""
    return features.TABLE_COLUMNS[2:]
