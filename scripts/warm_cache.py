"""Warm every model against every approved local demo sample.

Run after placing provenance-cleared images in data/samples/. This script may download
pretrained weights through the detector loader. It never changes frozen settings,
evaluation labels, cached detections, or benchmark results. Fine-tuned weights must
already exist locally; their source licence must be confirmed before use.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from core import config, detector, validation


def sample_images() -> list[Path]:
    """Return supported images under data/samples, excluding generated/hidden folders."""
    return sorted(
        path for path in config.SAMPLES_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in config.ALLOWED_EXTENSIONS
        and not any(part.startswith(".") for part in path.relative_to(config.SAMPLES_DIR).parts)
    )


def main() -> int:
    samples = sample_images()
    if not samples:
        print("ERROR: no demo images found in data/samples/. Add approved, licensed samples first.")
        return 1

    models = []
    for key in config.PRETRAINED_MODEL_KEYS:
        models.append((key, detector.build_detector(key), None))

    fine_key = config.FINETUNED_MODEL_KEY
    fine_weights = config.FINETUNED_MODEL_WEIGHTS
    if not fine_weights.is_file():
        print(f"ERROR: fine-tuned checkpoint is missing: {fine_weights}")
        print("The fine-tuned model is required for this warm-cache run; no model was silently skipped.")
        return 1
    fine_settings = json.loads(config.FINETUNED_SETTINGS_JSON.read_text(encoding="utf-8"))
    models.append((fine_key, detector.build_detector(fine_key), fine_settings))

    failures = []
    for key, model, fine in models:
        try:
            model.warmup()
        except Exception as error:
            failures.append((key, "warmup", str(error)))
            continue
        for sample in samples:
            try:
                data = sample.read_bytes()
                validation.validate_image_input(sample.name, data)
                image = validation.decode_image(data)
                if fine is None:
                    detections = model.detect(image, config.DEFAULT_CONFIDENCE)
                else:
                    detections = model.detect(
                        image,
                        float(fine["confidence"]),
                        input_size=int(fine["input_size"]),
                        nms_iou=float(fine["nms_iou"]),
                    )
                print(f"OK {sample.relative_to(ROOT)} | {key} | detections={len(detections)}")
            except Exception as error:
                failures.append((key, str(sample), f"{type(error).__name__}: {error}"))
                print(f"FAIL {sample.name} | {key} | {type(error).__name__}: {error}")

    if failures:
        print("\nWarm-cache failures:")
        for failure in failures:
            print(" - " + " | ".join(failure))
        return 1
    print(f"Warm-cache complete: {len(samples)} samples x {len(models)} models.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
