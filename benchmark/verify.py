"""Verify that the pilot benchmark package has no images and checks its hashes."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}


def main() -> int:
    image_files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if image_files:
        print("ERROR: image files are not permitted in this metadata package:")
        for path in image_files:
            print(f"  {path.relative_to(ROOT)}")
        return 1

    checksum_file = ROOT / "SHA256SUMS"
    if not checksum_file.is_file():
        print("PENDING: SHA256SUMS has not been generated. Run python benchmark/generate_checksums.py.")
        return 2

    failures = []
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            failures.append(f"malformed checksum line: {line}")
            continue
        target = (ROOT / relative).resolve()
        if ROOT.resolve() not in target.parents or not target.is_file():
            failures.append(f"missing or unsafe path: {relative}")
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            failures.append(f"checksum mismatch: {relative}")
    if failures:
        print("ERROR: checksum verification failed:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("OK: no images found; all listed package checksums match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
