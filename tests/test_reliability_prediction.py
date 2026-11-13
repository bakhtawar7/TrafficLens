from __future__ import annotations

import json

from core.reliability_prediction import load_flag_configuration, score_features


def _report(verdict: str) -> dict:
    model = {
        "feature_columns": ["signal"],
        "imputation_medians": [0.0],
        "means": [0.0],
        "scales": [1.0],
        "coefficients": [2.0],
        "intercept": 1.0,
        "flag_threshold": 2.0,
    }
    stats = {
        "needs_review": {
            "primary_rule": {"precision": 0.6, "recall": 0.7},
            "scene_bootstrap_95ci": {
                "precision": {"low": 0.4, "high": 0.8},
                "recall": {"low": 0.5, "high": 0.9},
            },
        }
    }
    return {
        "sample_size": 40,
        "exploratory": False,
        "hypotheses": {"H1": {"verdict": verdict}},
        "flag_policy": {"fallback_baseline": "mean_confidence"},
        "deployment": {
            "flag_fraction": 0.25,
            "models": {
                "ridge": model,
                "mean_confidence": model,
            },
        },
        "models": {"ridge": stats, "mean_confidence": stats},
    }


def test_missing_result_files_are_unvalidated(tmp_path):
    assert load_flag_configuration(
        tmp_path / "missing.json", tmp_path / "missing.md"
    ) is None


def test_week_three_summary_verdict_overrides_report_verdict(tmp_path):
    report_path = tmp_path / "predictor_report.json"
    report_path.write_text(json.dumps(_report("supported")), encoding="utf-8")
    summary_path = tmp_path / "week3_summary.md"
    summary_path.write_text(
        "| Hypothesis | Status |\n|---|---|\n| H1 | inconclusive |\n",
        encoding="utf-8",
    )

    configuration = load_flag_configuration(report_path, summary_path)

    assert configuration is not None
    assert configuration["verdict"] == "inconclusive"
    assert configuration["source"] == "mean_confidence"


def test_deployment_prediction_returns_error_flag_and_top_contributors(tmp_path):
    report_path = tmp_path / "predictor_report.json"
    report_path.write_text(json.dumps(_report("supported")), encoding="utf-8")
    configuration = load_flag_configuration(report_path, tmp_path / "absent.md")
    assert configuration is not None

    prediction = score_features({"signal": 1.0}, configuration)

    assert prediction is not None
    assert prediction["predicted_error"] == 3.0
    assert prediction["flagged"] is True
    assert prediction["top_features"] == [{"feature": "signal", "contribution": 2.0}]
