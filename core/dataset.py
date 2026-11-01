"""Test set construction logic for TrafficLens.

The Week 4 benchmark needs images with trustworthy ground truth counts. A raw annotated
archive is not directly usable: annotations use their own label vocabulary, some files are
broken, several frames are near duplicates, and the condition tags required by
``SOURCES.csv`` are not recorded anywhere. This module turns those concerns into small
pure functions that a script can drive over a directory and a test can exercise with
synthetic input.

The ground truth rules implemented here come from ``docs/labelling_protocol.md``:
the seven TrafficLens categories, the twelve pixel minimum size rule, and the fixed
condition tag set from ``docs/test_set_guide.md``.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ElementTree
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------

#: The seven ground truth categories recorded in ``LABELS.csv``, in column order.
LABEL_CATEGORIES: tuple[str, ...] = (
    "car",
    "bus",
    "truck",
    "motorcycle",
    "rickshaw",
    "loader_rickshaw",
    "other",
)

#: The four categories a COCO pretrained detector can actually report. Rickshaws and
#: loaders are recorded as ground truth so the gap can be measured, never claimed.
DETECTABLE_CATEGORIES: tuple[str, ...] = ("car", "motorcycle", "bus", "truck")

#: Condition tags accepted by ``scripts/check_testset.py``.
CONDITION_TAGS: tuple[str, ...] = ("day", "night", "glare", "dense", "sparse", "distant")

#: Mapping from an archive label to a TrafficLens category. Anything absent is mapped to
#: ``other`` and recorded in the notes so a reviewer can overrule the decision.
LABEL_TO_CATEGORY: dict[str, str] = {
    "car": "car",
    "minivan": "car",
    "suv": "car",
    "van": "car",
    "taxi": "car",
    "jeep": "car",
    "bus": "bus",
    "minibus": "bus",
    "microbus": "bus",
    "coach": "bus",
    "truck": "truck",
    "lorry": "truck",
    "pickup": "truck",
    "garbagevan": "truck",
    "tanker": "truck",
    "motorbike": "motorcycle",
    "motorcycle": "motorcycle",
    "motor-bike": "motorcycle",
    "scooter": "motorcycle",
    "rickshaw": "rickshaw",
    "auto rickshaw": "rickshaw",
    "autorickshaw": "rickshaw",
    "cycle rickshaw": "rickshaw",
    "three wheelers (cng)": "rickshaw",
    "three wheeler": "rickshaw",
    "cng": "rickshaw",
    "loader rickshaw": "loader_rickshaw",
    "easy bike": "loader_rickshaw",
    "human hauler": "loader_rickshaw",
    "wheelbarrow": "loader_rickshaw",
    "bicycle": "other",
    "ambulance": "other",
    "policecar": "other",
    "army vehicle": "other",
    "tractor": "other",
}

# ---------------------------------------------------------------------------
# Quality and tagging thresholds
# ---------------------------------------------------------------------------

#: Longest visible side, in pixels, below which a vehicle is background detail.
MIN_VEHICLE_PIXELS: int = 12

#: Longest visible side, in pixels, below which a vehicle counts as distant traffic.
DISTANT_VEHICLE_PIXELS: int = 32

#: Mean luma at or above which the scene is treated as daylight.
DAY_LUMA: float = 78.0

#: Mean luma at or below which the scene is treated as night.
NIGHT_LUMA: float = 52.0

#: A shaded street can measure dark while the sky above it is bright, so the top band of
#: the frame is checked before a scene is called night.
SKY_BAND_DIVISOR: int = 5

#: Mean luma of the top band at or above which the scene is daylight regardless of ground.
DAY_SKY_LUMA: float = 110.0

#: Luma at or above which a pixel counts as well lit. How much of a frame receives light
#: separates a shaded day street from a real night better than the mean brightness does,
#: because night scenes stay dark overall while carrying a few bright light sources.
WELL_LIT_LUMA: float = 100.0

#: Fraction of well lit pixels at or above which the scene is daylight regardless of mean.
DAY_LIT_FRACTION: float = 0.20

#: Fraction of pixels blown to near white at which the scene is tagged ``glare``.
GLARE_HIGHLIGHT_FRACTION: float = 0.012

#: Vehicle count at or above which the scene is tagged ``dense``.
DENSE_COUNT: int = 14

#: Vehicle count at or below which the scene is tagged ``sparse``.
SPARSE_COUNT: int = 5

#: Fraction of surviving boxes that must be distant to tag the item ``distant``.
DISTANT_FRACTION: float = 0.45

#: Blur score (variance of the Laplacian) under which an image is too soft to label.
MIN_SHARPNESS: float = 45.0

#: Shortest edge under which an image is rejected as too small to count reliably.
MIN_EDGE_PIXELS: int = 360

#: Hamming distance under which two perceptual hashes are treated as the same frame.
DUPLICATE_HASH_DISTANCE: int = 8

# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VocObject:
    """One annotated object in PASCAL VOC form."""

    label: str
    xyxy: tuple[float, float, float, float]
    truncated: bool = False
    difficult: bool = False

    @property
    def longest_side(self) -> float:
        """Length of the longer box edge in pixels."""
        x_min, y_min, x_max, y_max = self.xyxy
        return max(abs(x_max - x_min), abs(y_max - y_min))

    @property
    def area(self) -> float:
        """Box area in square pixels."""
        x_min, y_min, x_max, y_max = self.xyxy
        return max(0.0, x_max - x_min) * max(0.0, y_max - y_min)


@dataclass(frozen=True)
class VocAnnotation:
    """A parsed PASCAL VOC XML annotation."""

    filename: str
    width: int
    height: int
    objects: tuple[VocObject, ...] = ()


def normalise_label(name: str) -> str:
    """Lower case an archive label and collapse separators so variants compare equal."""
    cleaned = re.sub(r"[\s_\-]+", " ", (name or "").strip().lower())
    return re.sub(r"\s+", " ", cleaned)


def map_label(name: str) -> tuple[str, bool]:
    """Map an archive label onto a TrafficLens category.

    Args:
        name: Raw label as written in the annotation.

    Returns:
        A tuple of ``(category, known)`` where ``known`` is False when the label was not
        in :data:`LABEL_TO_CATEGORY` and the category fell back to ``other``.
    """
    key = normalise_label(name)
    category = LABEL_TO_CATEGORY.get(key)
    if category is None:
        return "other", False
    return category, True


def _child_text(node: ElementTree.Element, tag: str, default: str = "") -> str:
    """Read the text of a direct child element, tolerating absence."""
    child = node.find(tag)
    if child is None or child.text is None:
        return default
    return child.text.strip()


def _to_int(value: str, default: int = 0) -> int:
    """Parse an integer, returning ``default`` when the text is not numeric."""
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_voc_annotation(path: Path) -> tuple[VocAnnotation | None, str]:
    """Parse one VOC XML file.

    Args:
        path: Location of the ``.xml`` annotation.

    Returns:
        ``(annotation, error)``. Exactly one of the two is populated: ``annotation`` is
        ``None`` with a human readable reason when the file cannot be trusted.
    """
    try:
        raw = path.read_bytes().decode("utf-8", errors="ignore")
    except OSError as exc:
        return None, f"could not read {path.name}: {exc}"

    if "<annotation" not in raw:
        return None, f"{path.name} is not a PASCAL VOC annotation"
    if "<!DOCTYPE" in raw.upper() or "<!ENTITY" in raw.upper():
        return None, f"{path.name} declares a DTD or entity, which is refused"

    try:
        root = ElementTree.fromstring(raw)
    except ElementTree.ParseError as exc:
        return None, f"{path.name} XML is malformed ({exc})"

    objects: list[VocObject] = []
    for node in root.findall("object"):
        name = _child_text(node, "name")
        box = node.find("bndbox")
        if not name or box is None:
            continue
        xyxy = (
            float(_to_int(_child_text(box, "xmin"))),
            float(_to_int(_child_text(box, "ymin"))),
            float(_to_int(_child_text(box, "xmax"))),
            float(_to_int(_child_text(box, "ymax"))),
        )
        if xyxy[2] <= xyxy[0] or xyxy[3] <= xyxy[1]:
            continue
        objects.append(
            VocObject(
                label=name,
                xyxy=xyxy,
                truncated=_child_text(node, "truncated") == "1",
                difficult=_child_text(node, "difficult") == "1",
            )
        )

    size = root.find("size")
    width = _to_int(_child_text(size, "width")) if size is not None else 0
    height = _to_int(_child_text(size, "height")) if size is not None else 0
    return (
        VocAnnotation(
            filename=_child_text(root, "filename"),
            width=width,
            height=height,
            objects=tuple(objects),
        ),
        "",
    )


def scale_objects(objects: Iterable[VocObject], from_size: tuple[int, int], to_size: tuple[int, int]) -> tuple[VocObject, ...]:
    """Rescale boxes when the recorded annotation size differs from the image on disk.

    Args:
        objects: Boxes expressed against ``from_size``.
        from_size: ``(width, height)`` the annotation claims.
        to_size: ``(width, height)`` of the decoded image.

    Returns:
        The rescaled boxes. Input is returned unchanged when either size is unusable.
    """
    src_w, src_h = from_size
    dst_w, dst_h = to_size
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return tuple(objects)
    if (src_w, src_h) == (dst_w, dst_h):
        return tuple(objects)
    sx = dst_w / src_w
    sy = dst_h / src_h
    scaled: list[VocObject] = []
    for obj in objects:
        x_min, y_min, x_max, y_max = obj.xyxy
        scaled.append(
            VocObject(
                label=obj.label,
                xyxy=(x_min * sx, y_min * sy, x_max * sx, y_max * sy),
                truncated=obj.truncated,
                difficult=obj.difficult,
            )
        )
    return tuple(scaled)


# ---------------------------------------------------------------------------
# Ground truth counting
# ---------------------------------------------------------------------------


def count_by_category(objects: Iterable[VocObject], min_pixels: int = MIN_VEHICLE_PIXELS) -> dict[str, object]:
    """Apply the labelling protocol to a set of annotated objects.

    Objects smaller than ``min_pixels`` on their longest side are dropped as background
    detail, exactly as ``docs/labelling_protocol.md`` requires.

    Returns:
        A dictionary with ``counts`` (per category, zeros included), ``kept`` (objects that
        survived the size rule), ``dropped_small``, and ``unmapped`` (archive labels that
        fell back to ``other``).
    """
    counts = {category: 0 for category in LABEL_CATEGORIES}
    dropped_small = 0
    kept = 0
    unmapped: set[str] = set()
    for obj in objects:
        if obj.longest_side < min_pixels:
            dropped_small += 1
            continue
        kept += 1
        category, known = map_label(obj.label)
        counts[category] += 1
        if not known:
            unmapped.add(normalise_label(obj.label))
    return {
        "counts": counts,
        "kept": kept,
        "dropped_small": dropped_small,
        "unmapped": sorted(unmapped),
    }


def surviving_objects(objects: Iterable[VocObject], min_pixels: int = MIN_VEHICLE_PIXELS) -> tuple[VocObject, ...]:
    """Return only the objects large enough to be counted under the size rule."""
    return tuple(obj for obj in objects if obj.longest_side >= min_pixels)


def total_vehicles(counts: dict[str, int]) -> int:
    """Sum the per category ground truth counts."""
    return int(sum(counts.get(category, 0) for category in LABEL_CATEGORIES))


# ---------------------------------------------------------------------------
# Image measurements
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImageMetrics:
    """Automatic measurements used to suggest condition tags and reject bad files."""

    width: int
    height: int
    mean_luma: float
    highlight_fraction: float
    sharpness: float
    sky_luma: float = 0.0
    lit_fraction: float = 0.0

    @property
    def short_edge(self) -> int:
        """Shorter image edge in pixels."""
        return min(self.width, self.height)


def load_image(path: Path) -> np.ndarray | None:
    """Decode an image from disk, returning ``None`` when it cannot be read.

    Windows paths with unusual characters are handled by reading the bytes directly.
    """
    try:
        buffer = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if buffer.size == 0:
        return None
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    return None if image is None else image


def measure_image(image: np.ndarray) -> ImageMetrics:
    """Compute luma, blown highlights, sky brightness, lit area, and sharpness for one image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    band = max(1, gray.shape[0] // SKY_BAND_DIVISOR)
    return ImageMetrics(
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        mean_luma=float(gray.mean()),
        highlight_fraction=float(np.mean(gray >= 245)),
        sharpness=float(laplacian.var()),
        sky_luma=float(gray[:band, :].mean()),
        lit_fraction=float(np.mean(gray >= WELL_LIT_LUMA)),
    )


def perceptual_hash(image: np.ndarray, size: int = 16) -> str:
    """Return a compact difference style hash used to spot near duplicate frames.

    The hash is derived from horizontal gradient signs on a downscaled grayscale image,
    which survives re-encoding but changes when the scene content changes.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    packed = np.packbits(bits.astype(np.uint8))
    return packed.tobytes().hex()


def hash_distance(left: str, right: str) -> int:
    """Hamming distance between two hashes produced by :func:`perceptual_hash`."""
    try:
        a = np.unpackbits(np.frombuffer(bytes.fromhex(left), dtype=np.uint8))
        b = np.unpackbits(np.frombuffer(bytes.fromhex(right), dtype=np.uint8))
    except ValueError:
        return 8 * max(len(left), len(right)) // 2
    if a.shape != b.shape:
        return int(np.inf)
    return int(np.count_nonzero(a != b))


def stable_seed(text: str) -> int:
    """Deterministic 31 bit seed derived from a filename, used for reproducible draws."""
    return zlib.crc32(text.encode("utf-8")) & 0x7FFFFFFF


def objects_to_json(objects: Iterable[VocObject]) -> str:
    """Serialise annotated objects so an inventory CSV can carry the boxes forward."""
    return json.dumps(
        [
            {
                "label": obj.label,
                "xyxy": [round(float(value), 2) for value in obj.xyxy],
                "truncated": int(obj.truncated),
            }
            for obj in objects
        ]
    )


def objects_from_json(raw: object) -> tuple[VocObject, ...]:
    """Rebuild objects written by :func:`objects_to_json`; empty on any problem."""
    if not isinstance(raw, str) or not raw:
        return ()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ()
    objects: list[VocObject] = []
    for entry in payload:
        try:
            box = tuple(float(value) for value in entry["xyxy"])
            objects.append(
                VocObject(
                    label=str(entry["label"]),
                    xyxy=(box[0], box[1], box[2], box[3]),
                    truncated=bool(int(entry.get("truncated", 0))),
                )
            )
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return tuple(objects)


# ---------------------------------------------------------------------------
# Condition tagging
# ---------------------------------------------------------------------------


def suggest_condition_tags(metrics: ImageMetrics, counts: dict[str, int], objects: Iterable[VocObject]) -> list[str]:
    """Suggest ``SOURCES.csv`` condition tags from measurements and ground truth boxes.

    The suggestions are deliberately conservative. Lighting is read from three signals:
    overall luma, the brightness of the top band of the frame, and the share of the frame
    that is well lit. A shaded daytime street fails the overall luma test yet passes the
    other two, so it is no longer mistaken for night, while a night frame keeps a small
    lit share because its brightness comes from a few light sources rather than the scene.
    Density comes from the vehicle count and distance from the share of small boxes. A
    reviewer confirms or corrects every tag in the review pass.
    """
    tags: list[str] = []

    daylight = (
        metrics.mean_luma >= DAY_LUMA
        or metrics.lit_fraction >= DAY_LIT_FRACTION
        or metrics.sky_luma >= DAY_SKY_LUMA
    )
    if daylight:
        tags.append("day")
    elif metrics.mean_luma <= NIGHT_LUMA:
        tags.append("night")
    else:
        tags.append("day" if metrics.mean_luma > (NIGHT_LUMA + DAY_LUMA) / 2 else "night")

    if metrics.highlight_fraction >= GLARE_HIGHLIGHT_FRACTION:
        tags.append("glare")

    total = total_vehicles(counts)
    if total >= DENSE_COUNT:
        tags.append("dense")
    elif total <= SPARSE_COUNT:
        tags.append("sparse")

    sides = [obj.longest_side for obj in objects]
    if sides:
        distant_share = sum(1 for side in sides if side < DISTANT_VEHICLE_PIXELS) / len(sides)
        if distant_share >= DISTANT_FRACTION and total > 0:
            tags.append("distant")

    return tags


def merge_tags(tags: Iterable[str]) -> str:
    """Join tags in the canonical order using the semicolon separator."""
    order = {tag: index for index, tag in enumerate(CONDITION_TAGS)}
    unique = sorted({tag.strip().lower() for tag in tags if tag.strip()}, key=lambda tag: order.get(tag, 99))
    return ";".join(unique)


def parse_tag_quota(text: str) -> tuple[str, int]:
    """Parse a ``tag=count`` selection quota such as ``night=12``.

    Raises:
        ValueError: the expression is malformed, the tag is outside the fixed set, or the
            count is not a positive whole number.
    """
    tag, separator, raw_count = (text or "").partition("=")
    tag = tag.strip().lower()
    if not separator or not tag:
        raise ValueError(f"'{text}' is not a tag=count pair")
    if tag not in CONDITION_TAGS:
        raise ValueError(f"'{tag}' is not one of: {', '.join(CONDITION_TAGS)}")
    try:
        count = int(raw_count)
    except ValueError as exc:
        raise ValueError(f"'{raw_count}' is not a whole number") from exc
    if count < 1:
        raise ValueError("a tag quota must be at least 1")
    return tag, count


def split_tags(raw: object) -> tuple[list[str], list[str]]:
    """Split a tag cell into valid tags plus anything outside the fixed tag set.

    Both ``;`` and ``-`` are accepted as separators because contact sheet captions and
    file names use the hyphenated form.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return [], []
    parts = [part.strip().lower() for part in re.split(r"[;|\-]", str(raw)) if part.strip()]
    valid = [part for part in parts if part in CONDITION_TAGS]
    invalid = [part for part in parts if part not in CONDITION_TAGS]
    return valid, invalid


def tags_to_slug(tags: Iterable[str]) -> str:
    """Render a tag list as the hyphenated file name component."""
    joined = merge_tags(tags)
    return joined.replace(";", "-") if joined else "mixed"


# ---------------------------------------------------------------------------
# Area inference
# ---------------------------------------------------------------------------

#: Place names recognised inside archive filenames. Anything else is grouped by the
#: collection batch so that picks stay spread across sources.
KNOWN_AREAS: tuple[str, ...] = (
    "motijheel",
    "shahbag",
    "mirpur",
    "sobhanbag",
    "farmgate",
    "airport",
    "highway",
    "overbridge",
    "feni",
    "dhaka",
    "chattogram",
    "sylhet",
    "khulna",
    "rajshahi",
    "narayanganj",
    "gazipur",
    "tangail",
)

_SLUG_SPLIT = re.compile(r"[^a-z0-9]+")


def infer_area(original_filename: str, source_folder: str = "") -> tuple[str, str]:
    """Guess a coarse location for an archive image.

    Args:
        original_filename: Name recorded in the annotation or on disk.
        source_folder: Folder the item was found in, used as the batch name.

    Returns:
        ``(area, origin)`` where ``origin`` is ``filename`` when a recognised place name
        was present and ``batch`` when only the collection folder could be used.
    """
    haystack = f"{original_filename} {source_folder}".lower()
    for area in KNOWN_AREAS:
        if area in haystack:
            return area, "filename"
    for part in _SLUG_SPLIT.split(original_filename.lower()):
        if len(part) >= 4 and not part.isdigit() and part not in {"jpeg", "jpg", "png", "maxresdefault", "copy"}:
            return part, "batch"
    return "unknown", "batch"


def make_item_filename(item_id: str, area: str, tags: Iterable[str], suffix: str = ".jpg") -> str:
    """Build ``<item_id>_<area>_<conditions><suffix>`` from the naming convention."""
    return f"{item_id}_{area}_{tags_to_slug(tags)}{suffix}"


_DATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?<!\d)(20\d{2})[-_.](\d{2})[-_.](\d{2})(?!\d)"),
    re.compile(r"(?<!\d)(\d{2})(\d{2})(\d{2})(?!\d)"),
)


def extract_date(original_filename: str) -> str:
    """Best effort ISO capture date recovered from an archive file name.

    Two shapes are recognised: ``2015-08-27`` and a six digit ``ddmmyy`` block. Anything
    else returns an empty string so ``SOURCES.csv`` stays honest rather than invented.
    """
    match = _DATE_PATTERNS[0].search(original_filename or "")
    if match:
        year, month, day = (int(part) for part in match.groups())
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"{year:04d}-{month:02d}-{day:02d}"
    match = _DATE_PATTERNS[1].search(original_filename or "")
    if match:
        day, month, year = (int(part) for part in match.groups())
        if 1 <= month <= 12 and 1 <= day <= 31:
            return f"20{year:02d}-{month:02d}-{day:02d}"
    return ""


# ---------------------------------------------------------------------------
# Candidate screening and selection
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """One archive image with everything needed to decide whether it joins the test set."""

    image_path: str
    xml_path: str
    original_filename: str
    source_split: str
    width: int = 0
    height: int = 0
    area: str = "unknown"
    area_origin: str = "batch"
    counts: dict[str, int] = field(default_factory=lambda: {c: 0 for c in LABEL_CATEGORIES})
    kept_objects: int = 0
    dropped_small: int = 0
    unmapped_labels: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    mean_luma: float = 0.0
    sky_luma: float = 0.0
    lit_fraction: float = 0.0
    highlight_fraction: float = 0.0
    sharpness: float = 0.0
    phash: str = ""
    size_mismatch: bool = False
    parse_error: str = ""
    reject_reasons: list[str] = field(default_factory=list)
    objects: tuple[VocObject, ...] = ()

    @property
    def total(self) -> int:
        """Total vehicles in the ground truth count."""
        return total_vehicles(self.counts)

    @property
    def detectable_total(self) -> int:
        """Vehicles that a COCO pretrained detector could report."""
        return sum(self.counts.get(category, 0) for category in DETECTABLE_CATEGORIES)

    @property
    def categories_present(self) -> int:
        """Number of distinct categories with at least one vehicle."""
        return sum(1 for category in LABEL_CATEGORIES if self.counts.get(category, 0) > 0)

    @property
    def key(self) -> str:
        """Stable identity for logging, based on the archive file name."""
        return Path(self.image_path).name


def screen_candidate(candidate: Candidate) -> list[str]:
    """Return the reasons a candidate is unfit for the benchmark, empty when fit.

    Rejection is about the item, not the scene: a hard scene can still be valuable, but a
    broken annotation, a tiny frame, or an empty street cannot be used as ground truth.
    """
    reasons: list[str] = []
    if candidate.parse_error:
        reasons.append(candidate.parse_error)
        return reasons
    if not candidate.image_path:
        reasons.append("no image file on disk")
    if candidate.width == 0 or candidate.height == 0:
        reasons.append("image could not be decoded")
    else:
        if candidate.size_mismatch:
            reasons.append("annotation size disagrees with the image")
        if min(candidate.width, candidate.height) < MIN_EDGE_PIXELS:
            reasons.append(f"short edge below {MIN_EDGE_PIXELS}px")
        if candidate.sharpness < MIN_SHARPNESS:
            reasons.append(f"too blurry (sharpness {candidate.sharpness:.0f})")
    if candidate.kept_objects == 0:
        reasons.append("no annotated vehicle survives the size rule")
    if candidate.total < 2:
        reasons.append("fewer than two vehicles to count")
    if candidate.total > 80:
        reasons.append("too heavily annotated to verify by eye")
    return reasons


def candidate_score(candidate: Candidate) -> float:
    """Rank candidates by how much information a well formed scene carries."""
    balance = min(candidate.categories_present, 4) / 4.0
    detectable = min(candidate.detectable_total, 20) / 20.0
    spread = 1.0 - abs(candidate.total - 12) / 40.0
    clarity = min(candidate.sharpness, 400.0) / 400.0
    return 0.35 * balance + 0.30 * detectable + 0.20 * max(spread, 0.0) + 0.15 * clarity


def select_candidates(
    candidates: Iterable[Candidate],
    target_size: int = 55,
    max_per_area: int = 8,
    max_per_tag_signature: int = 6,
    duplicate_distance: int = DUPLICATE_HASH_DISTANCE,
    tag_quotas: dict[str, int] | None = None,
) -> tuple[list[Candidate], list[Candidate]]:
    """Pick a diverse subset of screened candidates.

    The pass is deterministic: candidates are ranked by :func:`candidate_score`, ties are
    broken by name, and a candidate is skipped when its area bucket, tag signature, or
    near duplicate frame is already full. A second fill pass relaxes the diversity caps so
    the target size is still reached on thin archives.

    Args:
        candidates: Screened candidates, fit and unfit mixed; unfit ones are reported back.
        target_size: How many items to accept.
        max_per_area: Cap on items sharing one coarse area.
        max_per_tag_signature: Cap on items sharing one identical tag combination.
        duplicate_distance: Hamming distance under which a frame is a near duplicate.
        tag_quotas: Minimum number of accepted items carrying each named condition tag.
            Reserved items are chosen before the general pass, because an abundant
            condition otherwise crowds a rare one out of the ranking.

    Returns:
        ``(accepted, rejected)`` where ``rejected`` carries the reasons in
        ``reject_reasons``.
    """
    fit: list[Candidate] = []
    rejected: list[Candidate] = []
    for candidate in candidates:
        # Screening is recomputed rather than read back from reject_reasons so that calling
        # this function twice on the same objects gives the same answer.
        reasons = screen_candidate(candidate)
        if reasons:
            candidate.reject_reasons = reasons
            rejected.append(candidate)
        else:
            candidate.reject_reasons = []
            fit.append(candidate)

    ordered = sorted(fit, key=lambda item: (-candidate_score(item), item.key))
    accepted: list[Candidate] = []
    accepted_hashes: list[str] = []
    area_counts: dict[str, int] = {}
    signature_counts: dict[str, int] = {}

    def try_accept(candidate: Candidate, strict: bool) -> bool:
        """Add one candidate unless it duplicates a frame or breaks an enforced cap."""
        if len(accepted) >= target_size:
            return False
        if any(hash_distance(candidate.phash, seen) < duplicate_distance for seen in accepted_hashes):
            return False
        signature = merge_tags(candidate.tags) or "day"
        if strict and area_counts.get(candidate.area, 0) >= max_per_area:
            return False
        if strict and signature_counts.get(signature, 0) >= max_per_tag_signature:
            return False
        accepted.append(candidate)
        accepted_hashes.append(candidate.phash)
        area_counts[candidate.area] = area_counts.get(candidate.area, 0) + 1
        signature_counts[signature] = signature_counts.get(signature, 0) + 1
        return True

    def fill(pool: list[Candidate], limit: int, strict: bool) -> None:
        """Accept from ``pool`` in ranked order until ``limit`` items are held."""
        for candidate in pool:
            if len(accepted) >= limit:
                return
            try_accept(candidate, strict)

    def holding(tag: str) -> int:
        """Count accepted items that carry a condition tag."""
        return sum(1 for candidate in accepted if tag in candidate.tags)

    for tag, quota in (tag_quotas or {}).items():
        pool = [candidate for candidate in ordered if tag in candidate.tags]
        limit = min(target_size, len(accepted) + max(0, int(quota) - holding(tag)))
        fill(pool, limit, strict=True)
        fill(pool, limit, strict=False)

    fill(ordered, target_size, strict=True)
    fill(ordered, target_size, strict=False)

    chosen_keys = {candidate.key for candidate in accepted}
    for candidate in fit:
        if candidate.key not in chosen_keys:
            candidate.reject_reasons = ["not selected: diversity budget already filled"]
            rejected.append(candidate)
    return accepted, rejected
