"""Central configuration for TrafficLens.

Defines the target vehicle classes, counting defaults, file handling limits, and
on disk paths. Every other module reads its constants from here so that Week 2 can
adjust behaviour from a single place.
"""
from __future__ import annotations

from dataclasses import dataclass
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
#: Pre-registered record of the current per-model defaults; the benchmark and the
#: evaluator hash this file into results/summary.json so runs stay auditable.
DEFAULT_SETTINGS_JSON: Path = CONFIGS_DIR / "default_settings.json"

# Generated curation outputs, never written back into the source dataset.
RESULTS_DIR: Path = PROJECT_ROOT / "results"
CURATION_LOG_CSV: Path = RESULTS_DIR / "curation_log.csv"
AUDIT_OVERLAYS_DIR: Path = RESULTS_DIR / "audit_overlays"
AUDIT_SHEET_CSV: Path = RESULTS_DIR / "audit_sheet.csv"
# Benchmark and evaluation artefacts.
RAW_DETECTIONS_JSONL: Path = RESULTS_DIR / "raw_detections.jsonl"
EVALUATION_TABLE_CSV: Path = RESULTS_DIR / "evaluation_table.csv"
EVALUATION_SUMMARY_JSON: Path = RESULTS_DIR / "summary.json"
#: Label free reliability features, one row per cached image (core.features).
FEATURES_CSV: Path = RESULTS_DIR / "features.csv"
# Timestamped copies of data/testset/LABELS.csv taken by the manual labelling app
# before a session's first write, so a bad save can always be rolled back.
LABEL_BACKUPS_DIR: Path = RESULTS_DIR / "label_backups"

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

#: Count columns of data/testset/LABELS.csv, in file order.
LABEL_CLASS_NAMES: tuple[str, ...] = (
    "car", "bus", "truck", "motorcycle", "rickshaw", "loader_rickshaw", "other",
)

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

# Default sweep for core.reliability: the detector runs once at the lowest
# value and the higher thresholds are derived by confidence filtering.
THRESHOLD_SWEEP: tuple[float, ...] = (0.15, 0.25, 0.35, 0.50)

# IoU at which two same-class boxes count as the same vehicle for the
# agreement measures in core.reliability.
BOX_IOU_THRESHOLD: float = 0.5

# ---------------------------------------------------------------------------
# Model registry. Every entry names a pretrained ultralytics weights file that
# ultralytics downloads on first use. TrafficLens never trains or fine tunes.
# The stable key identifies the model; display_name is for reports and the UI;
# family selects the wrapper class (yolo: YOLO, detr: RTDETR).
# ---------------------------------------------------------------------------
MODEL_FAMILIES: tuple[str, ...] = ("yolo", "detr")


@dataclass(frozen=True)
class ModelSpec:
    """One registered pretrained model."""

    display_name: str
    family: str
    weights_file: str


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "yolov8n": ModelSpec("YOLOv8n", "yolo", "yolov8n.pt"),
    "yolov8s": ModelSpec("YOLOv8s", "yolo", "yolov8s.pt"),
    "rtdetr-l": ModelSpec("RT-DETR-L", "detr", "rtdetr-l.pt"),
}

# ---------------------------------------------------------------------------
# Sliced (tiled) inference defaults. SlicedDetector cuts an image into
# overlapping tiles and merges the duplicate detections with class-aware
# non-max suppression; the values below are its constructor defaults.
# ---------------------------------------------------------------------------
#: Edge length in pixels of each square tile.
SLICE_TILE_SIZE: int = 640
#: Fraction of a tile edge shared with the neighbouring tiles, in [0, 1).
SLICE_OVERLAP_RATIO: float = 0.2
#: IoU above which two same-class boxes are treated as one object.
SLICE_IOU_THRESHOLD: float = 0.5

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
