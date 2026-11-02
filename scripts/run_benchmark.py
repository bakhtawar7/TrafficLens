"""Benchmark every registry model over the test set and cache raw detections.

Each model runs in two modes: ``standard`` (one full-frame pass) and ``sliced``
(the same detector inside :class:`core.detector.SlicedDetector` with the tile
defaults from ``config``). Detections are stored at the *lowest* threshold of
``config.THRESHOLD_SWEEP`` so every higher threshold can be derived later by
confidence filtering (see ``core.reliability``); nothing is re-run for that.

One JSON line per (image, model, mode) goes to ``results/raw_detections.jsonl``
with the SHA-256 of the image file, load time, inference time measured after
warmup, and the raw boxes. Entries whose (image hash, model, mode) triple is
already in the file are skipped, so reruns only do the missing work.

Usage::

    python scripts/run_benchmark.py --dev        # development subset only
    python scripts/run_benchmark.py              # all test set images
    python scripts/run_benchmark.py --models yolov8n,rtdetr-l
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import config, dataset  # noqa: E402
from core.detector import SlicedDetector, build_detector  # noqa: E402

MODES: tuple[str, ...] = ("standard", "sliced")
DEFAULT_OUTPUT: Path = config.RESULTS_DIR / "raw_detections.jsonl"


def image_file_hash(path: Path) -> str:
    """SHA-256 hex digest of the image file bytes; the cache identity."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def list_images(images_dir: Path, dev_only: bool) -> list[Path]:
    """Sorted test set images; ``--dev`` restricts to the development subset.

    Raises:
        FileNotFoundError: when the images folder or the seeded plan file is
            missing (the plan defines the development subset).
    """
    if not images_dir.exists():
        raise FileNotFoundError(f"images folder missing: {images_dir}")
    images = sorted(
        path for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in config.ALLOWED_EXTENSIONS
    )
    if not dev_only:
        return images
    if not config.TESTSET_PLAN_JSON.exists():
        raise FileNotFoundError(
            f"development subset needs the seeded plan: {config.TESTSET_PLAN_JSON}"
        )
    plan = json.loads(config.TESTSET_PLAN_JSON.read_text(encoding="utf-8"))
    dev_ids = set(plan.get("development_subset", []))
    return [path for path in images if path.stem.split("_")[0] in dev_ids]


def load_cached_keys(output_path: Path) -> set[tuple[str, str, str]]:
    """Read existing JSONL and collect (image_hash, model, mode) already stored."""
    keys: set[tuple[str, str, str]] = set()
    if not output_path.exists():
        return keys
    with output_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                keys.add((entry["image_hash"], entry["model"], entry["mode"]))
            except (json.JSONDecodeError, KeyError):
                print(f"WARNING: ignoring malformed cached line: {line[:80]}")
    return keys


def serialise_detections(detections) -> list[dict]:  # noqa: ANN001 - list[Detection]
    """Convert Detection objects into plain JSON dictionaries."""
    return [
        {
            "class_id": detection.class_id,
            "class_name": detection.class_name,
            "confidence": detection.confidence,
            "xyxy": list(detection.xyxy),
        }
        for detection in detections
    ]


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark over the requested images and append uncached entries."""
    parser = argparse.ArgumentParser(description="Cache raw detections for every model and mode.")
    parser.add_argument("--dev", action="store_true",
                        help="run only the seeded development subset from configs/testset_plan.json")
    parser.add_argument("--images-dir", default=str(config.TESTSET_IMAGES_DIR),
                        help="folder of test set images")
    parser.add_argument("--models", default=",".join(config.MODEL_REGISTRY),
                        help="comma separated registry keys (default: all)")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="JSONL cache file (appended, never rewritten)")
    args = parser.parse_args(argv)

    keys = [key.strip() for key in args.models.split(",") if key.strip()]
    unknown = [key for key in keys if key not in config.MODEL_REGISTRY]
    if unknown:
        print(f"ERROR: models not in registry: {', '.join(unknown)}")
        print(f"Known keys: {', '.join(config.MODEL_REGISTRY)}")
        return 1

    try:
        images = list_images(Path(args.images_dir), args.dev)
    except FileNotFoundError as error:
        print(f"ERROR: {error}")
        return 1
    if not images:
        print("ERROR: no images selected.")
        return 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cached = load_cached_keys(output_path)
    threshold = min(config.THRESHOLD_SWEEP)

    print(f"Images: {len(images)} ({'development subset' if args.dev else 'full test set folder'})")
    print(f"Models: {', '.join(keys)} | Modes: {', '.join(MODES)}")
    print(f"Raw detections stored at threshold {threshold}")
    print(f"Cache: {len(cached)} entries already in {output_path}\n")

    written = 0
    skipped = 0
    digests = {path: image_file_hash(path) for path in images}
    started_run = time.perf_counter()
    with output_path.open("a", encoding="utf-8") as handle:
        for key in keys:
            pending = [
                path for path in images
                if any((digests[path], key, mode) not in cached for mode in MODES)
            ]
            if not pending:
                skipped += len(MODES) * len(images)
                print(f"{config.MODEL_REGISTRY[key].display_name}: all "
                      f"{len(MODES) * len(images)} entries cached; model not loaded\n")
                continue
            detector = build_detector(key)
            warmup_start = time.perf_counter()
            detector.warmup()
            warmup_seconds = time.perf_counter() - warmup_start
            sliced = SlicedDetector(detector)
            print(f"{detector.name}: load {detector.load_seconds:.2f} s, "
                  f"warmup {warmup_seconds:.2f} s")

            for path in images:
                digest = digests[path]
                todo = [mode for mode in MODES if (digest, key, mode) not in cached]
                skipped += len(MODES) - len(todo)
                if not todo:
                    continue
                image = dataset.load_image(path)
                if image is None:
                    print(f"ERROR: could not decode {path.name}; model work skipped")
                    return 1
                height, width = image.shape[:2]
                for mode in todo:
                    runner = detector if mode == "standard" else sliced
                    start = time.perf_counter()
                    detections = runner.detect(image, threshold)
                    inference_ms = (time.perf_counter() - start) * 1000.0
                    entry = {
                        "image": path.name,
                        "image_hash": digest,
                        "model": key,
                        "model_name": detector.name,
                        "mode": mode,
                        "threshold": threshold,
                        "load_seconds": detector.load_seconds,
                        "inference_ms": inference_ms,
                        "width": width,
                        "height": height,
                        "detections": serialise_detections(detections),
                    }
                    handle.write(json.dumps(entry) + "\n")
                    handle.flush()
                    written += 1
                    print(f"  {path.name} {mode:>8}: {len(detections):>3} detections, "
                          f"{inference_ms:8.1f} ms")
            print()

    elapsed = time.perf_counter() - started_run
    print(f"done: {written} entries written, {skipped} skipped as cached, "
          f"{elapsed:.1f} s of wall-clock work")
    return 0


if __name__ == "__main__":
    sys.exit(main())
