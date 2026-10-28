"""Bounding box and label rendering.

Boxes are drawn with supervision using a stable colour per class. Labels are rendered
with a solid coloured swatch behind contrasting text so they stay readable on both dark
and bright images. Empty detections return an untouched copy.
"""
from __future__ import annotations

import cv2
import numpy as np
import supervision as sv

from . import config
from .detector import Detection

# Ordered class names define the colour palette indices.
_PALETTE_NAMES: list[str] = list(config.CLASS_COLORS_RGB.keys())
_PALETTE_RGB: list[tuple[int, int, int]] = [
    config.CLASS_COLORS_RGB[name] for name in _PALETTE_NAMES
] + [config.DEFAULT_COLOR_RGB]


def _color_index(class_name: str) -> int:
    """Palette index for a class name, falling back to the default colour."""
    if class_name in _PALETTE_NAMES:
        return _PALETTE_NAMES.index(class_name)
    return len(_PALETTE_NAMES)


def _rgb_to_bgr(color: tuple[int, int, int]) -> tuple[int, int, int]:
    """Convert an RGB tuple to the BGR order OpenCV expects."""
    r, g, b = color
    return (b, g, r)


def _contrasting_text_color(bg_rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    """Return black or white BGR text colour that contrasts with a background."""
    r, g, b = bg_rgb
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def annotate(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw boxes and readable labels for ``detections`` onto a copy of ``image_bgr``.

    Args:
        image_bgr: ``(H, W, 3)`` uint8 BGR source image.
        detections: Detections to render. Empty input yields an unchanged copy.

    Returns:
        A new BGR image with annotations drawn on it.
    """
    annotated = image_bgr.copy()
    if not detections:
        return annotated

    xyxy = np.array([d.xyxy for d in detections], dtype=np.float32)
    confidence = np.array([d.confidence for d in detections], dtype=np.float32)
    class_ids = np.array([d.class_id for d in detections], dtype=np.int32)
    color_lookup = np.array([_color_index(d.class_name) for d in detections], dtype=np.int32)

    sv_detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_ids,
        data={"labels": [d.class_name for d in detections]},
    )

    palette = sv.ColorPalette(
        colors=[sv.Color(r=r, g=g, b=b) for (r, g, b) in _PALETTE_RGB]
    )
    box_annotator = sv.BoxAnnotator(
        color=palette, color_lookup=sv.ColorLookup.INDEX, thickness=2
    )
    annotated = box_annotator.annotate(
        scene=annotated.copy(),
        detections=sv_detections,
        custom_color_lookup=color_lookup,
    )

    for detection in detections:
        _draw_label(annotated, detection)
    return annotated


def _draw_label(image: np.ndarray, detection: Detection) -> None:
    """Draw one readable label with a coloured background onto ``image`` in place."""
    x_min, y_min = int(detection.xyxy[0]), int(detection.xyxy[1])
    text = f"{detection.class_name} {detection.confidence:.2f}"
    color_rgb = (
        config.CLASS_COLORS_RGB[detection.class_name]
        if detection.class_name in config.CLASS_COLORS_RGB
        else config.DEFAULT_COLOR_RGB
    )
    background = _rgb_to_bgr(color_rgb)
    text_color = _contrasting_text_color(color_rgb)

    scale, thickness = 0.5, 1
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    pad = 3
    top = max(y_min - text_h - baseline - 2 * pad, 0)
    cv2.rectangle(
        image,
        (x_min, top),
        (x_min + text_w + 2 * pad, top + text_h + baseline + 2 * pad),
        background,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x_min + pad, top + text_h + pad),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )
