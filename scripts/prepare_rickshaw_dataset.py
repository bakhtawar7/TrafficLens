"""Prepare an isolated one-class YOLO dataset from the supplied VOC archive."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import dataset

DEFAULT_ARCHIVE = Path(r"C:\Users\bakht\Downloads\archive.zip")
DEFAULT_OUTPUT = ROOT / "runs" / "rickshaw_finetune" / "dataset"
TARGET_CATEGORIES = {"rickshaw", "loader_rickshaw"}


@dataclass(frozen=True)
class ArchiveImage:
    path: str
    split_root: str
    source_group: str
    perceptual_hash: str


@dataclass(frozen=True)
class ParsedImage:
    archive_path: str
    annotation_path: str
    source_group: str
    split: str
    width: int
    height: int
    labels: tuple[str, ...]


def source_group(filename: str) -> str:
    """Collapse numbered contributor filenames into a common source group."""
    stem = PurePosixPath(filename.replace("\\", "/")).stem.lower()
    stem = re.sub(r"\s*\(\d+\)$", "", stem)
    stem = re.sub(r"[\s_-]*\d+$", "", stem)
    return stem.rstrip(" _-") or "unknown"


def parse_voc_labels(
    xml_bytes: bytes, image_width: int, image_height: int
) -> tuple[list[str], str | None]:
    """Return normalized one-class labels, or a reason to reject the annotation."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return [], "annotation is not valid XML"
    labels: list[str] = []
    for object_node in root.findall(".//object"):
        box_node = object_node.find("bndbox")
        name = (object_node.findtext("name") or "").strip().lower()
        if box_node is None:
            return [], "object is missing a bounding box"
        try:
            x_min, y_min, x_max, y_max = (
                float(box_node.findtext(key))
                for key in ("xmin", "ymin", "xmax", "ymax")
            )
        except (TypeError, ValueError):
            return [], "bounding box coordinates are invalid"
        if (
            not all(math.isfinite(value) for value in (x_min, y_min, x_max, y_max))
            or x_min < 0
            or y_min < 0
            or x_max > image_width
            or y_max > image_height
            or x_max <= x_min
            or y_max <= y_min
        ):
            return [], "bounding box is outside the decoded image or has no area"
        category = dataset.LABEL_TO_CATEGORY.get(name)
        if category not in TARGET_CATEGORIES:
            continue
        center_x = ((x_min + x_max) / 2) / image_width
        center_y = ((y_min + y_max) / 2) / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height
        labels.append(
            f"0 {center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}"
        )
    return labels, None


def assign_duplicate_safe_splits(images: list[ArchiveImage]) -> dict[str, str]:
    """Keep perceptual duplicates together, reserve Pias for validation, and seal test."""
    parent = list(range(len(images)))
    size = [1] * len(images)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if size[left_root] < size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        size[left_root] += size[right_root]

    hashes = [int(image.perceptual_hash, 16) for image in images]
    for index, value in enumerate(hashes):
        for earlier in range(index):
            if (value ^ hashes[earlier]).bit_count() < dataset.DUPLICATE_HASH_DISTANCE:
                union(index, earlier)

    components: dict[int, list[ArchiveImage]] = {}
    for index, image in enumerate(images):
        components.setdefault(find(index), []).append(image)

    assignments: dict[str, str] = {}
    for component in components.values():
        has_external_holdout = any(image.split_root != "train" for image in component)
        has_validation_source = any(
            image.split_root == "train" and image.source_group == "pias"
            for image in component
        )
        split = (
            "excluded_holdout"
            if has_external_holdout
            else "val"
            if has_validation_source
            else "train"
        )
        for image in component:
            if image.split_root == "train":
                assignments[image.path] = split
    return assignments


def _archive_images(
    archive: zipfile.ZipFile,
) -> tuple[list[ArchiveImage], dict[str, str]]:
    entries: list[ArchiveImage] = []
    image_names: dict[str, str] = {}
    for info in archive.infolist():
        if info.is_dir():
            continue
        image_path = PurePosixPath(info.filename)
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        split_root = image_path.parts[0]
        if split_root not in {"train", "test", "test round 2"}:
            continue
        raw = archive.read(info.filename)
        image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            continue
        annotation_path = str(image_path.with_suffix(".xml"))
        source_filename = image_path.name
        try:
            xml = archive.read(annotation_path)
            source_filename = ET.fromstring(xml).findtext("filename") or source_filename
        except (KeyError, ET.ParseError):
            pass
        entries.append(
            ArchiveImage(
                path=info.filename,
                split_root=split_root,
                source_group=source_group(source_filename),
                perceptual_hash=dataset.perceptual_hash(image),
            )
        )
        image_names[info.filename] = info.filename
    return entries, image_names


def _archive_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_dataset(archive_path: Path, output: Path) -> dict[str, object]:
    """Extract validated one-class images into a fresh, ignored output directory."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive_path}")
    if output.exists():
        raise FileExistsError(
            f"Output already exists; choose a fresh path rather than overwriting: {output}"
        )

    with zipfile.ZipFile(archive_path) as archive:
        archive_images, _ = _archive_images(archive)
        assignments = assign_duplicate_safe_splits(archive_images)
        member_names = {info.filename.lower(): info.filename for info in archive.infolist()}
        train_images = [
            image for image in archive_images if image.split_root == "train"
        ]
        parsed: list[ParsedImage] = []
        rejected: list[tuple[str, str, str]] = []
        for image in train_images:
            split = assignments.get(image.path, "excluded_holdout")
            if split == "excluded_holdout":
                rejected.append((image.path, "", "perceptual duplicate of sealed test image"))
                continue
            image_path = PurePosixPath(image.path)
            xml_path = str(image_path.with_suffix(".xml"))
            actual_xml = member_names.get(xml_path.lower())
            if actual_xml is None:
                rejected.append((image.path, xml_path, "matching annotation not found"))
                continue
            raw_image = archive.read(image.path)
            decoded = cv2.imdecode(
                np.frombuffer(raw_image, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if decoded is None:
                rejected.append((image.path, actual_xml, "image could not be decoded"))
                continue
            height, width = decoded.shape[:2]
            labels, error = parse_voc_labels(archive.read(actual_xml), width, height)
            if error:
                rejected.append((image.path, actual_xml, error))
                continue
            parsed.append(
                ParsedImage(
                    archive_path=image.path,
                    annotation_path=actual_xml,
                    source_group=image.source_group,
                    split=split,
                    width=width,
                    height=height,
                    labels=tuple(labels),
                )
            )

        output.mkdir(parents=True)
        for split in ("train", "val"):
            (output / "images" / split).mkdir(parents=True)
            (output / "labels" / split).mkdir(parents=True)
        manifest_path = output / "manifest.csv"
        manifest_fields = (
            "archive_image", "archive_annotation", "source_group", "split",
            "width", "height", "rickshaw_instances",
        )
        with (
            manifest_path.open("w", encoding="utf-8", newline="") as manifest_file,
            (output / "rejected.csv").open("w", encoding="utf-8", newline="") as rejected_file,
        ):
            manifest_writer = csv.DictWriter(manifest_file, fieldnames=manifest_fields)
            rejected_writer = csv.writer(rejected_file)
            manifest_writer.writeheader()
            rejected_writer.writerow(("archive_image", "archive_annotation", "reason"))
            for row in rejected:
                rejected_writer.writerow(row)
            counts = {split: {"images": 0, "positive_images": 0, "instances": 0} for split in ("train", "val")}
            for item in parsed:
                basename = PurePosixPath(item.archive_path).name
                digest = hashlib.sha256(item.archive_path.encode("utf-8")).hexdigest()[:10]
                destination_stem = f"{Path(basename).stem}_{digest}"
                image_destination = (
                    output / "images" / item.split
                    / f"{destination_stem}{Path(basename).suffix.lower()}"
                )
                label_name = f"{destination_stem}.txt"
                label_destination = output / "labels" / item.split / label_name
                image_destination.write_bytes(archive.read(item.archive_path))
                label_destination.write_text(
                    "\n".join(item.labels) + ("\n" if item.labels else ""),
                    encoding="utf-8",
                )
                stats = counts[item.split]
                stats["images"] += 1
                stats["positive_images"] += bool(item.labels)
                stats["instances"] += len(item.labels)
                manifest_writer.writerow({
                    "archive_image": item.archive_path,
                    "archive_annotation": item.annotation_path,
                    "source_group": item.source_group,
                    "split": item.split,
                    "width": item.width,
                    "height": item.height,
                    "rickshaw_instances": len(item.labels),
                })

    yaml_path = output / "dataset.yaml"
    yaml_path.write_text(
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: rickshaw\n",
        encoding="utf-8",
    )
    summary: dict[str, object] = {
        "archive": str(archive_path.resolve()),
        "archive_sha256": _archive_digest(archive_path),
        "target_class": "rickshaw",
        "included_source_categories": sorted(
            name for name, category in dataset.LABEL_TO_CATEGORY.items()
            if category in TARGET_CATEGORIES
        ),
        "split_method": (
            "Pias source group and perceptual-duplicate-connected images reserved for "
            "validation; archive test folders and their duplicate-connected images excluded "
            f"using Hamming distance < {dataset.DUPLICATE_HASH_DISTANCE}"
        ),
        "counts": counts,
        "rejected_images": len(rejected),
        "dataset_yaml": str(yaml_path.resolve()),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = prepare_dataset(args.archive, args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
