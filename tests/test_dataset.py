"""Tests for archive parsing, ground truth mapping, tagging, and test set selection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core import dataset
from core.dataset import Candidate, VocObject


def write_voc(path: Path, objects: list[tuple[str, tuple[float, float, float, float]]], width: int = 640, height: int = 480, filename: str = "frame.jpg") -> Path:
    """Write a minimal PASCAL VOC XML file for tests."""
    entries = "\n".join(
        f"""  <object>
    <name>{name}</name>
    <truncated>0</truncated>
    <difficult>0</difficult>
    <bndbox><xmin>{box[0]}</xmin><ymin>{box[1]}</ymin><xmax>{box[2]}</xmax><ymax>{box[3]}</ymax></bndbox>
  </object>"""
        for name, box in objects
    )
    path.write_text(
        f"""<annotation>
  <filename>{filename}</filename>
  <size><width>{width}</width><height>{height}</height><depth>3</depth></size>
{entries}
</annotation>""",
        encoding="utf-8",
    )
    return path


def make_candidate(**overrides) -> Candidate:
    """Build a screenable candidate with sensible defaults for tests."""
    base = dict(
        image_path="C:/archive/frame.jpg",
        xml_path="C:/archive/frame.xml",
        original_filename="frame.jpg",
        source_split="train",
        width=800,
        height=600,
        area="motijheel",
        sharpness=220.0,
        kept_objects=6,
        counts={category: 0 for category in dataset.LABEL_CATEGORIES} | {"car": 4, "bus": 2},
        tags=["day", "sparse"],
        phash="abcd" * 16,
    )
    base.update(overrides)
    return Candidate(**base)


def test_map_label_covers_archive_vocabulary():
    assert dataset.map_label("Three Wheelers (CNG)")[0] == "rickshaw"
    assert dataset.map_label("minibus")[0] == "bus"
    assert dataset.map_label("pickup")[0] == "truck"
    assert dataset.map_label("motorbike")[0] == "motorcycle"
    assert dataset.map_label("ambulance")[0] == "other"
    assert dataset.map_label("van")[0] == "car"


def test_map_label_unknown_marks_unmapped():
    category, known = dataset.map_label("space craft")
    assert (category, known) == ("other", False)


def test_parse_voc_reads_objects_and_size(tmp_path):
    path = write_voc(tmp_path / "a.xml", [("car", (10, 10, 110, 90)), ("bus", (0, 0, 400, 300))])
    annotation, error = dataset.parse_voc_annotation(path)
    assert error == ""
    assert annotation is not None
    assert annotation.width == 640 and len(annotation.objects) == 2
    assert annotation.objects[0].longest_side == pytest.approx(100.0)


def test_parse_voc_rejects_non_voc_and_dtd(tmp_path):
    broken = tmp_path / "broken.xml"
    broken.write_text("0.1 0.2 0.3 0.4", encoding="utf-8")
    annotation, error = dataset.parse_voc_annotation(broken)
    assert annotation is None and "not a PASCAL VOC annotation" in error

    dtd = tmp_path / "dtd.xml"
    dtd.write_text("<!DOCTYPE annotation [<!ENTITY x 'y'>]><annotation/>", encoding="utf-8")
    annotation, error = dataset.parse_voc_annotation(dtd)
    assert annotation is None and "DTD" in error


def test_count_by_category_applies_size_rule():
    objects = [
        VocObject("car", (0, 0, 40, 30)),
        VocObject("car", (0, 0, 11, 4)),  # below the 12px rule
        VocObject("rickshaw", (0, 0, 20, 20)),
        VocObject("spaceship", (0, 0, 60, 60)),
    ]
    result = dataset.count_by_category(objects)
    assert result["counts"]["car"] == 1
    assert result["counts"]["rickshaw"] == 1
    assert result["counts"]["other"] == 1
    assert result["dropped_small"] == 1
    assert result["unmapped"] == ["spaceship"]


def test_scale_objects_when_declaration_differs():
    objects = [VocObject("car", (10, 10, 110, 90))]
    scaled = dataset.scale_objects(objects, (640, 480), (1280, 960))
    assert scaled[0].xyxy == pytest.approx((20.0, 20.0, 220.0, 180.0))
    assert dataset.scale_objects(objects, (640, 480), (640, 480))[0] is objects[0]


def synthetic_image(brightness: int = 160, seed: int = 1) -> np.ndarray:
    """Return a textured BGR image with a controllable mean brightness."""
    rng = np.random.RandomState(seed)
    base = np.full((80, 120, 3), brightness, dtype=np.int16)
    noise = rng.randint(-25, 25, base.shape)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def test_measure_image_and_hash_shape():
    image = synthetic_image()
    metrics = dataset.measure_image(image)
    assert (metrics.width, metrics.height) == (120, 80)
    digest = dataset.perceptual_hash(image)
    assert len(digest) == 64
    assert dataset.hash_distance(digest, digest) == 0
    assert dataset.hash_distance(digest, dataset.perceptual_hash(synthetic_image(seed=7))) > 4


def shaded_daytime_image() -> np.ndarray:
    """A dark street under a bright sky, the classic false night scene."""
    image = synthetic_image(brightness=35, seed=4)
    image[: image.shape[0] // 5, :, :] = 235
    return image


def test_sky_band_stops_a_shaded_day_street_being_called_night():
    shaded = dataset.suggest_condition_tags(
        dataset.measure_image(shaded_daytime_image()),
        {category: 0 for category in dataset.LABEL_CATEGORIES} | {"car": 6},
        [VocObject("car", (0, 0, 60, 50)) for _ in range(6)],
    )
    assert "day" in shaded and "night" not in shaded

    dark = dataset.suggest_condition_tags(
        dataset.measure_image(synthetic_image(35, seed=4)),
        {category: 0 for category in dataset.LABEL_CATEGORIES} | {"car": 6},
        [VocObject("car", (0, 0, 60, 50)) for _ in range(6)],
    )
    assert "night" in dark


def lit_night_image() -> np.ndarray:
    """A real night frame: dark overall, carrying a small patch of bright light sources."""
    image = synthetic_image(brightness=30, seed=11)
    image[30:44, 10:60, :] = 250
    return image


def test_headlights_do_not_turn_a_night_frame_into_day():
    metrics = dataset.measure_image(lit_night_image())
    assert metrics.lit_fraction < dataset.DAY_LIT_FRACTION
    tags = dataset.suggest_condition_tags(
        metrics,
        {category: 0 for category in dataset.LABEL_CATEGORIES} | {"car": 6},
        [VocObject("car", (0, 0, 60, 50)) for _ in range(6)],
    )
    assert "night" in tags and "day" not in tags


def test_measure_image_reports_the_well_lit_area_fraction():
    assert dataset.measure_image(synthetic_image(160)).lit_fraction == pytest.approx(1.0)
    assert dataset.measure_image(synthetic_image(30)).lit_fraction == pytest.approx(0.0)


def test_condition_tags_for_day_night_dense_and_distant():
    objects = [VocObject("car", (0, 0, 20, 20)) for _ in range(20)]
    counts = {category: 0 for category in dataset.LABEL_CATEGORIES} | {"car": 20}
    day = dataset.suggest_condition_tags(dataset.measure_image(synthetic_image(180)), counts, objects)
    assert "day" in day and "dense" in day

    night = dataset.suggest_condition_tags(dataset.measure_image(synthetic_image(30)), counts, objects)
    assert "night" in night

    tiny = [VocObject("car", (0, 0, 200, 160)) for _ in range(2)]
    sparse_counts = {category: 0 for category in dataset.LABEL_CATEGORIES} | {"car": 2}
    tags = dataset.suggest_condition_tags(dataset.measure_image(synthetic_image(200)), sparse_counts, tiny)
    assert "sparse" in tags and "distant" not in tags


def test_merge_and_split_tags_round_trip():
    merged = dataset.merge_tags(["dense", "day", "day", "night"])
    assert merged == "day;night;dense"
    valid, invalid = dataset.split_tags("day-dense-rain")
    assert valid == ["day", "dense"] and invalid == ["rain"]


def test_infer_area_prefers_place_over_batch():
    assert dataset.infer_area("02_Motijheel_280714_0005.jpg", "train") == ("motijheel", "filename")
    area, origin = dataset.infer_area("Dipto_101.jpg", "Final Train Dataset")
    assert (area, origin) == ("dipto", "batch")


def test_extract_date_reads_both_shapes():
    assert dataset.extract_date("shot_2015-08-27_final.jpg") == "2015-08-27"
    assert dataset.extract_date("02_Motijheel_280714_0005.jpg") == "2014-07-28"
    assert dataset.extract_date("Dipto_101.jpg") == ""


def test_item_filename_follows_naming_convention():
    assert dataset.make_item_filename("tl001", "mirpur", ["day", "dense"]) == "tl001_mirpur_day-dense.jpg"


def test_screen_candidate_reports_reasons():
    reasons = dataset.screen_candidate(make_candidate(sharpness=10.0, kept_objects=0))
    assert any("blurry" in reason for reason in reasons)
    assert "no annotated vehicle survives the size rule" in reasons
    assert dataset.screen_candidate(make_candidate()) == []


def test_screen_candidate_rejects_unreadable_items():
    assert dataset.screen_candidate(make_candidate(parse_error="image could not be decoded")) == [
        "image could not be decoded"
    ]


def test_objects_json_round_trip():
    objects = (VocObject("car", (1.0, 2.0, 30.0, 40.0), truncated=True),)
    restored = dataset.objects_from_json(dataset.objects_to_json(objects))
    assert restored[0].label == "car"
    assert restored[0].xyxy == pytest.approx((1.0, 2.0, 30.0, 40.0))
    assert restored[0].truncated is True
    assert dataset.objects_from_json(json.dumps([{"label": "car"}])) == ()


def distinct_hash(index: int) -> str:
    """A perceptual hash that is far from every other test hash, built from scene noise."""
    return dataset.perceptual_hash(synthetic_image(seed=index))


def test_select_candidates_deduplicates_and_caps_areas():
    pool = []
    for index in range(30):
        area = "motijheel" if index < 20 else f"area{index}"
        tags = ["day", "dense"] if index < 25 else ["night", "sparse"]
        pool.append(
            make_candidate(
                image_path=f"C:/archive/{index}.jpg",
                original_filename=f"{index}.jpg",
                area=area,
                tags=tags,
                phash=distinct_hash(index),
            )
        )
    accepted, rejected = dataset.select_candidates(pool, target_size=10, max_per_area=3, max_per_tag_signature=3)
    assert len(accepted) == 10
    areas = {candidate.area for candidate in accepted}
    assert len(areas) > 1
    assert all(candidate.reject_reasons for candidate in rejected)


def test_select_candidates_falls_back_when_caps_too_tight():
    pool = [
        make_candidate(
            image_path=f"C:/archive/{index}.jpg",
            original_filename=f"{index}.jpg",
            phash=distinct_hash(index),
        )
        for index in range(6)
    ]
    accepted, rejected = dataset.select_candidates(pool, target_size=5, max_per_area=1)
    assert len(accepted) == 5
    assert len(rejected) == 1


def test_select_candidates_treats_near_duplicates_as_one_frame():
    twin = make_candidate(image_path="C:/archive/twin.jpg", original_filename="twin.jpg", phash=distinct_hash(0))
    first = make_candidate(image_path="C:/archive/first.jpg", original_filename="first.jpg", phash=distinct_hash(0))
    other = make_candidate(image_path="C:/archive/other.jpg", original_filename="other.jpg", phash=distinct_hash(9))
    accepted, _rejected = dataset.select_candidates([first, twin, other], target_size=3, max_per_area=9, max_per_tag_signature=9)
    assert [candidate.key for candidate in accepted] == ["first.jpg", "other.jpg"]


def test_load_image_returns_none_for_missing_file(tmp_path):
    assert dataset.load_image(tmp_path / "nope.jpg") is None


def test_tag_quotas_reserve_rare_conditions():
    pool = [
        make_candidate(
            image_path=f"C:/archive/day_{index}.jpg",
            original_filename=f"day_{index}.jpg",
            phash=distinct_hash(index),
        )
        for index in range(30)
    ] + [
        make_candidate(
            image_path=f"C:/archive/night_{index}.jpg",
            original_filename=f"night_{index}.jpg",
            tags=["night", "sparse"],
            phash=distinct_hash(100 + index),
        )
        for index in range(6)
    ]

    plain, _ = dataset.select_candidates(pool, target_size=10)
    plain_night = sum(1 for candidate in plain if "night" in candidate.tags)
    assert plain_night < 4

    reserved, _ = dataset.select_candidates(pool, target_size=10, tag_quotas={"night": 4})
    assert sum(1 for candidate in reserved if "night" in candidate.tags) == 4
    assert len(reserved) == 10


def test_select_candidates_is_repeatable_on_the_same_objects():
    pool = [
        make_candidate(
            image_path=f"C:/archive/frame_{index}.jpg",
            original_filename=f"frame_{index}.jpg",
            tags=["night", "dense"] if index % 3 == 0 else ["day", "dense"],
            phash=distinct_hash(index),
        )
        for index in range(12)
    ]
    first, _ = dataset.select_candidates(pool, target_size=6, tag_quotas={"night": 2})
    second, _ = dataset.select_candidates(pool, target_size=6, tag_quotas={"night": 2})
    assert [candidate.key for candidate in first] == [candidate.key for candidate in second]


def test_tag_quota_beyond_the_available_pool_is_partial():
    pool = [
        make_candidate(
            image_path=f"C:/archive/night_{index}.jpg",
            original_filename=f"night_{index}.jpg",
            tags=["night", "dense"],
            phash=distinct_hash(index),
        )
        for index in range(3)
    ]
    accepted, _ = dataset.select_candidates(pool, target_size=3, tag_quotas={"night": 9})
    assert len(accepted) == 3


@pytest.mark.parametrize(
    ("text", "expected"),
    [("night=12", ("night", 12)), ("Dense=3", ("dense", 3))],
)
def test_parse_tag_quota_accepts_valid_expressions(text, expected):
    assert dataset.parse_tag_quota(text) == expected


@pytest.mark.parametrize("text", ["night", "rain=4", "night=many", "night=0", "=4"])
def test_parse_tag_quota_rejects_bad_expressions(text):
    with pytest.raises(ValueError):
        dataset.parse_tag_quota(text)
