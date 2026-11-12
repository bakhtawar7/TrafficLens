"""Run scene-grouped cross-validation for audited counting-error prediction.

This command requires a sufficiently labelled evaluation set and a fully labelled
development subset. It uses only ``features.csv`` and evaluator-produced audited
errors as model inputs/targets; labels are opened only for the explicit readiness
gate. It never runs inference or evaluates development items.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config, evaluation, predictor  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict audited counting error with scene-grouped CV.")
    parser.add_argument("--features", default=str(config.FEATURES_CSV))
    parser.add_argument("--evaluation", default=str(config.EVALUATION_TABLE_CSV))
    parser.add_argument("--sources", default=str(config.SOURCES_CSV))
    parser.add_argument("--labels", default=str(config.LABELS_CSV))
    parser.add_argument("--plan", default=str(config.TESTSET_PLAN_JSON))
    parser.add_argument("--output", default=str(config.RESULTS_DIR / "predictor_report.json"))
    parser.add_argument("--bootstrap-samples", type=int, default=predictor.BOOTSTRAP_SAMPLES)
    parser.add_argument("--permutations", type=int, default=49)
    args = parser.parse_args(argv)

    required = {
        "features": Path(args.features),
        "evaluation": Path(args.evaluation),
        "sources": Path(args.sources),
        "labels": Path(args.labels),
        "plan": Path(args.plan),
    }
    missing_files = [f"{name}: {path}" for name, path in required.items() if not path.is_file()]
    if missing_files:
        print("ERROR: required input files are missing:")
        for item in missing_files:
            print(f"  {item}")
        return 1

    readiness = predictor.audited_label_readiness(
        required["sources"], required["labels"], required["plan"]
    )
    print(
        f"Audited first-pass labels: {readiness['evaluation_audited']}/"
        f"{readiness['evaluation_total']} evaluation images; "
        f"{len(readiness['development_missing_ids'])} development images missing."
    )
    blocked = False
    if readiness["evaluation_audited"] < 30:
        blocked = True
        print("ERROR: at least 30 evaluation images need audited first-pass labels.")
        print(f"  Missing evaluation ids: {', '.join(readiness['evaluation_missing_ids']) or 'none'}")
    if readiness["development_missing_ids"]:
        blocked = True
        print("ERROR: every development image must have an audited first-pass label.")
        print(f"  Missing development ids: {', '.join(readiness['development_missing_ids'])}")
    if blocked:
        return 1

    development_ids = evaluation.load_development_ids(required["plan"])
    rows = predictor.load_predictor_rows(
        required["features"], required["evaluation"], required["sources"], development_ids
    )
    joined_ids = {row.item_id for row in rows}
    missing_results = sorted(set(readiness["evaluation_audited_ids"]) - joined_ids)
    if missing_results:
        print("ERROR: audited evaluation images are missing features or evaluator errors:")
        print(f"  Missing ids: {', '.join(missing_results)}")
        return 1
    if len(rows) < 30:
        print(
            "ERROR: only "
            f"{len(rows)} evaluation images have both inference features and evaluator errors; "
            "at least 30 are required."
        )
        return 1
    if len({row.scene for row in rows}) < 2:
        print("ERROR: scene-grouped validation requires at least two evaluation scenes.")
        return 1

    report = predictor.evaluate_rows(
        rows, bootstrap_samples=args.bootstrap_samples, seed=config.SEED
    )
    report["audited_label_readiness"] = readiness
    if args.permutations > 0:
        report["permutation_test_ridge_vs_mean_confidence"] = predictor.permutation_skill_test(
            rows, permutations=args.permutations, seed=config.SEED
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    status = "EXPLORATORY (<60 images)" if report["exploratory"] else "not exploratory by sample-size rule"
    print(f"Sample size: {report['sample_size']} evaluation images ({status}).")
    print(f"Scene-grouped CV and aggregate metrics written to {output}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
