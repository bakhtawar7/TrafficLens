"""Run every registry detector on one image and print counts plus timing.

Each model is built from ``config.MODEL_REGISTRY``, then ``warmup()`` performs the
weights load (recorded separately as ``load_seconds``) and one dummy frame, so the
reported inference time covers prediction only and never includes loading. Counts are
limited to the four COCO classes TrafficLens can honestly report: car, motorcycle,
bus, and truck. Rickshaws have no COCO class and are never claimed as detected.

The first run downloads the pretrained weights through ultralytics; no training or
fine tuning happens anywhere.

Usage::

    python scripts/run_detectors.py                       # first committed test set image
    python scripts/run_detectors.py --image path/to.jpg --confidence 0.4
    python scripts/run_detectors.py --models yolov8n,rtdetr-l
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config, counting, dataset  # noqa: E402
from core.detector import build_detector  # noqa: E402


def default_image() -> Path | None:
    """Return the first committed test set image, or None when the folder is empty."""
    if not config.TESTSET_IMAGES_DIR.exists():
        return None
    candidates = sorted(
        path for path in config.TESTSET_IMAGES_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in config.ALLOWED_EXTENSIONS
    )
    return candidates[0] if candidates else None


def main(argv: list[str] | None = None) -> int:
    """Parse arguments, run every requested detector, and print counts and timing."""
    parser = argparse.ArgumentParser(description="Compare registry detectors on one image.")
    parser.add_argument("--image", default=None, help="Image file; defaults to the first test set image.")
    parser.add_argument("--confidence", type=float, default=config.DEFAULT_CONFIDENCE,
                        help="Confidence threshold applied to every model.")
    parser.add_argument("--models", default=",".join(config.PRETRAINED_MODEL_KEYS),
                        help="Comma separated registry keys (default: all).")
    args = parser.parse_args(argv)

    if args.image:
        image_path = Path(args.image).expanduser()
    else:
        image_path = default_image()
        if image_path is None:
            print("ERROR: no image given and the test set folder is empty.")
            return 1
    if not image_path.exists():
        print(f"ERROR: image not found: {image_path}")
        return 1
    image = dataset.load_image(image_path)
    if image is None:
        print(f"ERROR: image could not be decoded: {image_path}")
        return 1

    keys = [key.strip() for key in args.models.split(",") if key.strip()]
    unknown = [key for key in keys if key not in config.MODEL_REGISTRY]
    if unknown:
        print(f"ERROR: models not in registry: {', '.join(unknown)}")
        print(f"Known keys: {', '.join(config.PRETRAINED_MODEL_KEYS)}")
        return 1

    print(f"Image: {image_path}  ({image.shape[1]}x{image.shape[0]})")
    print(f"Confidence: {args.confidence}")
    print(f"Classes counted: {', '.join(config.TARGET_CLASS_NAMES)} "
          "(rickshaws have no COCO class and are never claimed)\n")

    failures = 0
    for key in keys:
        detector = build_detector(key)
        started = time.perf_counter()
        detector.warmup()
        warmup_seconds = time.perf_counter() - started
        result = counting.count_vehicles(image, detector, confidence_threshold=args.confidence)
        load_seconds = detector.load_seconds
        if load_seconds is None:
            failures += 1
            print(f"{detector.name}: ERROR: warmup did not record a load time.")
            continue
        counts = " ".join(f"{name}={result.counts_by_class[name]}" for name in config.TARGET_CLASS_NAMES)
        print(f"{detector.name} ({detector.family} family, {detector.weights})")
        print(f"  model load  : {load_seconds:.2f} s (excluded from inference timing)")
        print(f"  warmup total: {warmup_seconds:.2f} s (load plus one dummy frame)")
        print(f"  inference   : {result.inference_ms:.1f} ms")
        print(f"  counts      : {counts} (total {result.total_count})\n")

    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
