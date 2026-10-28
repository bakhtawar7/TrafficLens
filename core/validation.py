"""Input validation and error types for TrafficLens.

The UI must never surface a raw traceback. Every rejected upload is mapped to a
specific error carrying a short human readable message. Decoding failures are turned
into the same error family so callers can catch a single base type.
"""
from __future__ import annotations

import os

import cv2
import numpy as np

from . import config


class TrafficLensInputError(ValueError):
    """Base class for every rejected input. Carries a user safe message."""

    message: str = "The input could not be accepted."

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.message


class EmptyFileError(TrafficLensInputError):
    """Raised when the uploaded file has zero bytes."""

    message = "The uploaded file is empty. Please choose an image with content."


class FileTooLargeError(TrafficLensInputError):
    """Raised when the uploaded file exceeds the configured size limit."""

    def __init__(self, size_bytes: int, max_bytes: int) -> None:
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes
        limit_mb = max_bytes / (1024 * 1024)
        actual_mb = size_bytes / (1024 * 1024)
        self.message = (
            f"The uploaded file is {actual_mb:.1f} MB, which exceeds the "
            f"{limit_mb:.0f} MB limit. Please provide a smaller image."
        )
        super().__init__(self.message)


class UnsupportedFileTypeError(TrafficLensInputError):
    """Raised when the file extension is not an accepted image format."""

    def __init__(self, extension: str) -> None:
        self.extension = extension
        allowed = ", ".join(sorted(config.ALLOWED_EXTENSIONS))
        self.message = (
            f"File type '{extension or 'unknown'}' is not supported. "
            f"Accepted types are: {allowed}."
        )
        super().__init__(self.message)


class CorruptImageError(TrafficLensInputError):
    """Raised when a file passes the checks but cannot be decoded as an image."""

    message = (
        "The image could not be read. The file may be corrupt or truncated. "
        "Please try a different image."
    )


def validate_image_input(
    filename: str,
    data: bytes,
    max_bytes: int = config.MAX_UPLOAD_BYTES,
) -> None:
    """Validate an uploaded image, raising a specific error on any problem.

    Args:
        filename: Original name of the uploaded file, used to read the extension.
        data: Raw bytes of the file.
        max_bytes: Maximum allowed size in bytes.

    Raises:
        EmptyFileError: ``data`` is zero bytes.
        FileTooLargeError: ``data`` exceeds ``max_bytes``.
        UnsupportedFileTypeError: extension is not in the allowed set.
    """
    if data is None or len(data) == 0:
        raise EmptyFileError()
    if len(data) > max_bytes:
        raise FileTooLargeError(len(data), max_bytes)
    extension = os.path.splitext(filename or "")[1].lower()
    if extension not in config.ALLOWED_EXTENSIONS:
        raise UnsupportedFileTypeError(extension)


def decode_image(data: bytes) -> np.ndarray:
    """Decode image bytes into a BGR ``numpy`` array.

    Args:
        data: Raw image bytes.

    Returns:
        The decoded image as a ``(H, W, 3)`` uint8 BGR array.

    Raises:
        CorruptImageError: the bytes are not a decodable image.
    """
    buffer = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
    if image is None:
        raise CorruptImageError()
    return image
