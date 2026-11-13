"""Rebuild the TrafficLens test set from the original dataset archive.

This script re-applies the three-stage pipeline (clean, tag, build) to a local copy
of the Dhaka-AI archive. It is provided so that a reviewer who has access to the
original dataset can verify that the benchmark images are identical to those produced
by the published seeded selection.

Requirements
------------
* The Dhaka-AI annotated archive must be available at the path given by --source.
* The virtual environment must have all packages from requirements.txt installed.
* Run from the project root.

Usage
-----
::

    python benchmark/rebuild_testset.py --source C:/path/to/archive [--dry-run]

The ``--dry-run`` flag logs what would happen without writing any file.

Verification
------------
After running, compare the SHA-256 checksums of the rebuilt images against
``benchmark/checksums.json``::

    python benchmark/verify.py

Any hash mismatch means the archive copy differs from the one used to build the
published benchmark.

Seed
----
The selection uses seed 20261028 (stored in ``configs/testset_plan.json`` and in
``core/config.py``). The seed is applied to Python's ``random`` module via
``random.seed(seed)`` in ``core/curation.py`` before every draw. All inputs
(cleaned image list, frozen tag cut-offs, scene groups) are deterministic given
the same archive, so repeating this script with the same archive must yield
the same 75 items.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Path to the Dhaka-AI archive directory.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_dir():
        print(f"ERROR: --source {source} is not a directory.", file=sys.stderr)
        sys.exit(1)

    commands = [
        [sys.executable, str(ROOT / "scripts" / "clean_dataset.py"), "--source", str(source)],
        [sys.executable, str(ROOT / "scripts" / "tag_dataset.py"), "--source", str(source)],
        [sys.executable, str(ROOT / "scripts" / "build_testset.py"), "--source", str(source), "--apply"],
        [sys.executable, str(ROOT / "scripts" / "check_testset.py")],
    ]

    for cmd in commands:
        print(f"Running: {' '.join(cmd)}")
        if not args.dry_run:
            result = subprocess.run(cmd, cwd=ROOT)
            if result.returncode != 0:
                print(f"Command failed with exit code {result.returncode}.", file=sys.stderr)
                sys.exit(result.returncode)
        else:
            print("  (dry run; not executed)")

    if not args.dry_run:
        print("\nRebuild complete. Run `python benchmark/verify.py` to check checksums.")
    else:
        print("\nDry run complete. Remove --dry-run to execute.")


if __name__ == "__main__":
    main()
