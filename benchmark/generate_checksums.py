"""Generate SHA256SUMS for all released benchmark package files."""
from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "SHA256SUMS"
EXCLUDED = {"SHA256SUMS", "__pycache__"}


def main() -> None:
    files = sorted(
        path for path in ROOT.rglob("*")
        if path.is_file()
        and path.name not in EXCLUDED
        and "__pycache__" not in path.parts
    )
    lines = [
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(ROOT).as_posix()}"
        for path in files
    ]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(lines)} checksums to {OUTPUT.name}")


if __name__ == "__main__":
    main()
