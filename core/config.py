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

# Review artefacts proposed by dataset inspection, kept separate from the code mapping.
CONFIGS_DIR: Path = PROJECT_ROOT / "configs"
CLASS_MAPPING_JSON: Path = CONFIGS_DIR / "class_mapping.json"
TAG_CUTOFFS_JSON: Path = CONFIGS_DIR / "tag_cutoffs.json"

# Generated curation outputs, never written back into the source dataset.
RESULTS_DIR: Path = PROJECT_ROOT / "results"
CURATION_LOG_CSV: Path = RESULTS_DIR / "curation_log.csv"
AUDIT_OVERLAYS_DIR: Path = RESULTS_DIR / "audit_overlays"
AUDIT_SHEET_CSV: Path = RESULTS_DIR / "audit_sheet.csv"

# ---------------------------------------------------------------------------
# Dataset provenance and stratified test set sampling
# ---------------------------------------------------------------------------
#: Master seed for every reproducible draw (sampling, second pass, development subset).
SEED: int = 20261028
#: Short identifier recorded in SOURCES.csv and used in the copied file names.
DATASET_NAME: str = "dhaka_ai"
#: The source label written into the SOURCES.csv source column.
DATASET_SOURCE: str = "Dhaka-AI annotated archive (user supplied)"
#: Licence note; must be replaced with a real grant before publication.
DATASET_LICENCE: str = "UNVERIFIED: supplied by operator, confirm usage rights before publication"

# Target size. 60 is not reachable at three per apparent scene because this archive
# holds only about fifteen real scene groups across its splits; 45 is the honest maximum
# that still satisfies every per-tag and rickshaw quota under the three-per-scene cap.
TESTSET_TARGET: int = 45
TESTSET_MIN_PER_TAG: int = 8
TESTSET_MIN_RICKSHAW: int = 15
TESTSET_MAX_PER_SCENE: int = 3
#: Frame block width used to define a finer burst scene when a filename carries no place
#: name, so consecutive captures group into runs rather than one undifferentiated bucket.
TESTSET_SCENE_BURST: int = 25
DEV_SUBSET_SIZE: int = 10
SECOND_PASS_FRACTION: float = 0.2

#: Machine readable record of the sampling run: seed, targets, and chosen subsets.
TESTSET_PLAN_JSON: Path = CONFIGS_DIR / "testset_plan.json"
ANNOTATION_COUNTS_CSV: Path = TESTSET_DIR / "annotation_counts.csv"

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
