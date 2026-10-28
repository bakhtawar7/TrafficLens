"""Tests for each input validation failure type and a valid decode."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from core import validation
from core.validation import (
    CorruptImageError,
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    validate_image_input,
    decode_image,
)


def _valid_png_bytes() -> bytes:
    image = np.full((40, 60, 3), 128, dtype=np.uint8)
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def test_empty_file_raises():
    with pytest.raises(EmptyFileError):
        validate_image_input("a.jpg", b"")


def test_oversized_file_raises():
    data = b"x" * (100 + 1)
    with pytest.raises(FileTooLargeError):
        validate_image_input("a.jpg", data, max_bytes=100)


def test_unsupported_type_raises():
    with pytest.raises(UnsupportedFileTypeError):
        validate_image_input("notes.txt", b"hello world")


def test_uppercase_extension_is_accepted():
    validate_image_input("PHOTO.JPG", b"content")  # should not raise


def test_corrupt_image_raises_on_decode():
    with pytest.raises(CorruptImageError):
        decode_image(b"not a real image at all")


def test_valid_image_decodes_to_bgr():
    image = decode_image(_valid_png_bytes())
    assert image.shape == (40, 60, 3)
    assert image.dtype == np.uint8


def test_error_messages_are_human_readable():
    with pytest.raises(FileTooLargeError) as exc:
        validate_image_input("a.jpg", b"x" * 2000, max_bytes=1000)
    assert "MB" in str(exc.value)
