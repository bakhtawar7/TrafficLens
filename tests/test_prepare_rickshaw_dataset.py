"""Synthetic tests for the isolated rickshaw dataset preparation helpers."""
from __future__ import annotations

import xml.etree.ElementTree as ET

from core import dataset
from scripts import prepare_rickshaw_dataset as prepare


def _voc_xml(name: str, bounds: tuple[int, int, int, int]) -> bytes:
    root = ET.Element("annotation")
    obj = ET.SubElement(root, "object")
    ET.SubElement(obj, "name").text = name
    box = ET.SubElement(obj, "bndbox")
    for key, value in zip(("xmin", "ymin", "xmax", "ymax"), bounds):
        ET.SubElement(box, key).text = str(value)
    return ET.tostring(root)


def test_source_group_normalizes_numbered_contributor_names() -> None:
    assert prepare.source_group("Pias (758).jpg") == "pias"
    assert prepare.source_group("Navid_01.jpg") == "navid"
    assert prepare.source_group("Numan_  (1).jpg") == "numan"


def test_voc_aliases_combine_rickshaw_and_loader_categories() -> None:
    xml = ET.fromstring(_voc_xml("human hauler", (10, 20, 50, 80)))
    bicycle = ET.SubElement(xml, "object")
    ET.SubElement(bicycle, "name").text = "bicycle"
    box = ET.SubElement(bicycle, "bndbox")
    for key, value in zip(("xmin", "ymin", "xmax", "ymax"), (1, 1, 20, 20)):
        ET.SubElement(box, key).text = str(value)

    labels, error = prepare.parse_voc_labels(ET.tostring(xml), 100, 100)

    assert error is None
    assert labels == ["0 0.30000000 0.50000000 0.40000000 0.60000000"]


def test_invalid_or_out_of_bounds_boxes_reject_the_annotation() -> None:
    labels, error = prepare.parse_voc_labels(
        _voc_xml("rickshaw", (0, 0, 120, 10)), 100, 100
    )
    assert labels == []
    assert error == "bounding box is outside the decoded image or has no area"

    labels, error = prepare.parse_voc_labels(b"not xml", 100, 100)
    assert labels == []
    assert error == "annotation is not valid XML"


def test_validation_and_test_duplicate_components_do_not_leak() -> None:
    threshold = dataset.DUPLICATE_HASH_DISTANCE
    images = [
        prepare.ArchiveImage("train/pias_a.jpg", "train", "pias", "00" * 32),
        prepare.ArchiveImage(
            "train/numan_duplicate.jpg", "train", "numan",
            f"{(1 << (threshold - 1)):064x}",
        ),
        prepare.ArchiveImage(
            "train/other.jpg", "train", "other",
            f"{sum(1 << bit for bit in range(20, 20 + threshold)):064x}",
        ),
        prepare.ArchiveImage(
            "train/external_match.jpg", "train", "external",
            f"{sum(1 << bit for bit in range(40, 40 + threshold)):064x}",
        ),
        prepare.ArchiveImage(
            "test/sealed.jpg", "test", "sealed",
            f"{sum(1 << bit for bit in range(40, 40 + threshold)):064x}",
        ),
    ]

    splits = prepare.assign_duplicate_safe_splits(images)

    assert splits["train/pias_a.jpg"] == "val"
    assert splits["train/numan_duplicate.jpg"] == "val"
    assert splits["train/other.jpg"] == "train"
    assert splits["train/external_match.jpg"] == "excluded_holdout"
