"""Synthetic coverage for documented image input and counting edge cases."""
from __future__ import annotations

from io import BytesIO

import cv2
import numpy as np
import pytest
from PIL import Image

from core import counting, validation
from core.config import MAX_UPLOAD_BYTES, TARGET_CLASS_NAMES
from core.detector import Detection
from core.validation import (
    CorruptImageError,
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)


def _encoded_image(
    extension: str = ".jpg",
    shape: tuple[int, int, int] = (32, 48, 3),
    *,
    grayscale: bool = False,
    alpha: bool = False,
) -> bytes:
    if grayscale:
        image = np.full(shape[:2], 127, dtype=np.uint8)
    elif alpha:
        image = np.full((*shape[:2], 4), (32, 96, 160, 128), dtype=np.uint8)
    else:
        image = np.full(shape, 127, dtype=np.uint8)
    ok, encoded = cv2.imencode(extension, image)
    assert ok
    return encoded.tobytes()


def _exif_rotated_jpeg() -> bytes:
    image = Image.new("RGB", (48, 32), (100, 120, 140))
    exif = image.getexif()
    exif[274] = 6
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif)
    return buffer.getvalue()


def _cmyk_jpeg() -> bytes:
    image = Image.new("CMYK", (12, 8), (0, 64, 128, 16))
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


def _animated_gif() -> bytes:
    first = Image.new("RGB", (8, 8), (10, 20, 30))
    second = Image.new("RGB", (8, 8), (30, 20, 10))
    buffer = BytesIO()
    first.save(
        buffer,
        format="GIF",
        save_all=True,
        append_images=[second],
        duration=100,
        loop=0,
    )
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("filename", "extension"),
    [("image.jpg", ".jpg"), ("image.png", ".png"), ("image.webp", ".webp")],
)
def test_valid_jpg_png_and_webp_decode_to_bgr(filename: str, extension: str) -> None:
    data = _encoded_image(extension)
    validation.validate_image_input(filename, data)
    image = validation.decode_image(data)
    assert image.shape == (32, 48, 3)
    assert image.dtype == np.uint8


def test_grayscale_png_decodes_as_three_channel_bgr() -> None:
    image = validation.decode_image(_encoded_image(".png", grayscale=True))
    assert image.shape == (32, 48, 3)
    assert image.dtype == np.uint8
    assert np.all(image[:, :, 0] == image[:, :, 1])


def test_png_with_alpha_decodes_as_three_channel_bgr() -> None:
    image = validation.decode_image(_encoded_image(".png", alpha=True))
    assert image.shape == (32, 48, 3)
    assert image.dtype == np.uint8


def test_jpeg_exif_rotation_is_applied_during_decode() -> None:
    image = validation.decode_image(_exif_rotated_jpeg())
    assert image.shape == (48, 32, 3)


def test_cmyk_jpeg_decodes_to_bgr() -> None:
    image = validation.decode_image(_cmyk_jpeg())
    assert image.shape == (8, 12, 3)
    assert image.dtype == np.uint8


def test_animated_gif_is_rejected_with_supported_type_list() -> None:
    with pytest.raises(UnsupportedFileTypeError, match="Accepted types"):
        validation.validate_image_input("animation.gif", _animated_gif())


def test_unicode_filename_is_validated_without_path_assumptions() -> None:
    data = _encoded_image(".png")
    validation.validate_image_input("交通_دھاکہ_🚲.png", data)


def test_corrupt_image_is_rejected_after_extension_validation() -> None:
    data = b"\xff\xd8\x00not a complete JPEG"
    validation.validate_image_input("corrupt.jpg", data)
    with pytest.raises(CorruptImageError):
        validation.decode_image(data)


def test_empty_file_is_rejected() -> None:
    with pytest.raises(EmptyFileError):
        validation.validate_image_input("empty.jpg", b"")


def test_text_renamed_as_jpg_is_rejected_as_corrupt() -> None:
    data = b"This is text, not an image."
    validation.validate_image_input("notes.jpg", data)
    with pytest.raises(CorruptImageError):
        validation.decode_image(data)


def test_unsupported_image_format_is_rejected_even_with_image_bytes() -> None:
    with pytest.raises(UnsupportedFileTypeError):
        validation.validate_image_input("image.bmp", _encoded_image(".png"))


def test_file_over_configured_size_limit_is_rejected() -> None:
    data = b"x" * (MAX_UPLOAD_BYTES + 1)
    with pytest.raises(FileTooLargeError):
        validation.validate_image_input("oversized.jpg", data)


def test_very_small_image_is_decoded_and_counted(stub_factory) -> None:
    image = validation.decode_image(_encoded_image(".png", (1, 1, 3)))
    result = counting.count_vehicles(image, stub_factory([]))
    assert (result.image_width, result.image_height) == (1, 1)
    assert result.total_count == 0


def test_very_large_image_is_decoded_and_counted(stub_factory) -> None:
    image = validation.decode_image(_encoded_image(".jpg", (3000, 4000, 3)))
    result = counting.count_vehicles(image, stub_factory([]))
    assert (result.image_width, result.image_height) == (4000, 3000)
    assert result.total_count == 0


def _scene_image(vehicle_count: int) -> np.ndarray:
    image = np.zeros((160, 320, 3), dtype=np.uint8)
    for index in range(vehicle_count):
        x = (index % 8) * 38 + 2
        y = (index // 8) * 40 + 4
        cv2.rectangle(image, (x, y), (x + 24, y + 20), (40, 120, 220), -1)
    return image


def _vehicle_detections(count: int) -> list[Detection]:
    return [
        Detection(2, "car", 0.9, (float(index), 0.0, float(index + 10), 10.0))
        for index in range(count)
    ]


@pytest.mark.parametrize(("vehicle_count", "expected"), [(0, 0), (1, 1), (20, 20)])
def test_scene_vehicle_count_cases_use_stub_detections(
    stub_factory, vehicle_count: int, expected: int
) -> None:
    result = counting.count_vehicles(
        _scene_image(vehicle_count),
        stub_factory(_vehicle_detections(vehicle_count)),
    )
    assert result.total_count == expected


def test_non_target_dominated_scene_excludes_non_target_classes(stub_factory) -> None:
    detections = [
        Detection(0, "person", 0.99, (0.0, 0.0, 20.0, 30.0)),
        Detection(1, "bicycle", 0.95, (25.0, 0.0, 45.0, 20.0)),
        Detection(2, "car", 0.9, (50.0, 0.0, 80.0, 20.0)),
    ]
    result = counting.count_vehicles(
        _scene_image(1), stub_factory(detections), list(TARGET_CLASS_NAMES)
    )
    assert result.total_count == 1
    assert [d.class_name for d in result.detections] == ["car"]
