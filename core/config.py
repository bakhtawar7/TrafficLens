"""Central configuration for TrafficLens.

Defines the target vehicle classes, counting defaults, file handling limits, and
on disk paths. Every other module reads its constants from here so that Week 2 can
adjust behaviour from a single place.
"""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
TESTSET_DIR: Path = DATA_DIR / "testset"
TESTSET_IMAGES_DIR: Path = TESTSET_DIR / "images"
TESTSET_CLIPS_DIR: Path = TESTSET_DIR / "clips"
SAMPLES_DIR: Path = DATA_DIR / "samples"
SOURCES_CSV: Path = TESTSET_DIR / "SOURCES.csv"
LABELS_CSV: Path = TESTSET_DIR / "LABELS.csv"

# Artifacts produced by the test set construction scripts.
TESTSET_INVENTORY_CSV: Path = TESTSET_DIR / "inventory.csv"
TESTSET_SELECTION_CSV: Path = TESTSET_DIR / "selection_log.csv"
TESTSET_REVIEW_DIR: Path = TESTSET_DIR / "review"
TESTSET_REVIEW_CHECKLIST_CSV: Path = TESTSET_REVIEW_DIR / "REVIEW.csv"

# ---------------------------------------------------------------------------
# Target classes (COCO ids). Rickshaws have no COCO class and are never claimed.
# ---------------------------------------------------------------------------
TARGET_CLASS_MAP: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

#: The class names offered in the UI, in display order.
TARGET_CLASS_NAMES: tuple[str, ...] = tuple(TARGET_CLASS_MAP.values())

# ---------------------------------------------------------------------------
# Counting defaults
# ---------------------------------------------------------------------------
DEFAULT_CONFIDENCE: float = 0.25
MIN_CONFIDENCE: float = 0.05
MAX_CONFIDENCE: float = 0.95
DEFAULT_MODEL_NAME: str = "yolov8n"
DEFAULT_WEIGHTS_FILE: str = "yolov8n.pt"

# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
ALLOWED_EXTENSIONS: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")
MAX_UPLOAD_BYTES: int = 15 * 1024 * 1024  # 15 MB default, configurable at call time

# ---------------------------------------------------------------------------
# Annotation colours (RGB) keyed by class name. A stable colour per class keeps
# output consistent between runs.
# ---------------------------------------------------------------------------
CLASS_COLORS_RGB: dict[str, tuple[int, int, int]] = {
    "car": (46, 204, 113),
    "motorcycle": (241, 196, 15),
    "bus": (52, 152, 219),
    "truck": (231, 76, 60),
}
DEFAULT_COLOR_RGB: tuple[int, int, int] = (149, 165, 166)
