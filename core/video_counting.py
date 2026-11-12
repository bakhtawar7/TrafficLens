"""Provenance-gated vehicle tracking and virtual-line counts for video."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Protocol
from urllib.parse import urlparse

import cv2
import numpy as np
import supervision as sv

from . import config, settings
from .detector import Detection, Detector


class VideoProcessingError(ValueError):
    """A video cannot be counted with the requested inputs or provenance."""


@dataclass(frozen=True)
class VideoCountResult:
    """Summary of a completed video pass."""

    inbound: int
    outbound: int
    crossings: tuple[dict[str, int | str], ...]
    frames_processed: int
    width: int
    height: int
    fps: float
    confidence: float
    model: str


class _Tracker(Protocol):
    def update(self, detections: sv.Detections) -> sv.Detections:
        """Associate the current frame's detections with tracks."""
        ...


def verify_video_provenance(
    filename: str, provenance_csv: Path = config.VIDEO_PROVENANCE_CSV
) -> dict[str, str]:
    """Require a complete provenance record for the uploaded video's filename."""
    path = Path(provenance_csv)
    if not path.is_file():
        raise VideoProcessingError(
            f"No video provenance file found at {path}. Add a row to PROVENANCE.csv first."
        )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    matches = [
        row for row in rows
        if (row.get("filename") or "").strip().casefold() == Path(filename).name.casefold()
    ]
    if len(matches) != 1:
        raise VideoProcessingError(
            f"No unique provenance row for {Path(filename).name}. Add its source URL, "
            "author, licence text, and download date to PROVENANCE.csv."
        )

    row = matches[0]
    required = ("source_url", "author", "licence_text", "download_date")
    missing = [field for field in required if not (row.get(field) or "").strip()]
    if missing:
        raise VideoProcessingError(
            f"Video provenance for {Path(filename).name} is incomplete: "
            f"{', '.join(missing)}."
        )
    parsed_url = urlparse(row["source_url"].strip())
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        raise VideoProcessingError("Video provenance source_url must be an HTTP(S) URL.")
    try:
        date.fromisoformat(row["download_date"].strip())
    except ValueError as error:
        raise VideoProcessingError(
            "Video provenance download_date must use YYYY-MM-DD format."
        ) from error
    return {field: (row.get(field) or "").strip() for field in ("filename", *required)}


def _create_bytetrack(frame_rate: float, confidence: float) -> _Tracker:
    """Build Roboflow's ByteTrack implementation for Supervision detections."""
    try:
        from trackers import ByteTrackTracker
    except ImportError as error:
        raise VideoProcessingError(
            "ByteTrack is unavailable. Install the project requirements, including "
            "the Roboflow trackers package, before processing video."
        ) from error

    return ByteTrackTracker(
        frame_rate=frame_rate,
        track_activation_threshold=confidence,
        high_conf_det_threshold=max(confidence, 0.6),
        minimum_consecutive_frames=2,
    )


def _to_supervision_detections(detections: list[Detection]) -> sv.Detections:
    vehicles = [d for d in detections if d.class_name in config.TARGET_CLASS_NAMES]
    if not vehicles:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=np.float32),
            confidence=np.empty((0,), dtype=np.float32),
            class_id=np.empty((0,), dtype=np.int32),
        )
    return sv.Detections(
        xyxy=np.asarray([d.xyxy for d in vehicles], dtype=np.float32),
        confidence=np.asarray([d.confidence for d in vehicles], dtype=np.float32),
        class_id=np.asarray([d.class_id for d in vehicles], dtype=np.int32),
    )


def _signed_side(
    point: tuple[float, float],
    start: tuple[int, int],
    end: tuple[int, int],
) -> int:
    value = (end[0] - start[0]) * (point[1] - start[1]) - (
        end[1] - start[1]
    ) * (point[0] - start[0])
    return 0 if math.isclose(value, 0.0, abs_tol=1e-6) else (1 if value > 0 else -1)


def _crosses_line_segment(
    previous: tuple[float, float],
    current: tuple[float, float],
    start: tuple[int, int],
    end: tuple[int, int],
) -> bool:
    """Return whether the track motion intersects the finite virtual line."""
    motion = (current[0] - previous[0], current[1] - previous[1])
    line = (end[0] - start[0], end[1] - start[1])
    denominator = motion[0] * line[1] - motion[1] * line[0]
    if math.isclose(denominator, 0.0, abs_tol=1e-9):
        return False
    offset = (start[0] - previous[0], start[1] - previous[1])
    motion_fraction = (offset[0] * line[1] - offset[1] * line[0]) / denominator
    line_fraction = (offset[0] * motion[1] - offset[1] * motion[0]) / denominator
    return (
        -1e-9 <= motion_fraction <= 1.0 + 1e-9
        and -1e-9 <= line_fraction <= 1.0 + 1e-9
    )


def _line_direction(previous_side: int, current_side: int) -> str:
    return "inbound" if previous_side < current_side else "outbound"


def _draw_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    line_start: tuple[int, int],
    line_end: tuple[int, int],
    inbound: int,
    outbound: int,
) -> None:
    cv2.line(frame, line_start, line_end, (0, 255, 255), 2)
    tracker_ids = detections.tracker_id
    for index, box in enumerate(detections.xyxy):
        x0, y0, x1, y1 = (int(round(float(value))) for value in box)
        track_id = int(tracker_ids[index]) if tracker_ids is not None else -1
        cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 220, 60), 2)
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x0, max(16, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (60, 220, 60),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        f"Inbound: {inbound}  Outbound: {outbound}",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def process_video(
    input_path: Path,
    annotated_output_path: Path,
    crossings_csv_path: Path,
    detector: Detector,
    line_start: tuple[int, int],
    line_end: tuple[int, int],
    *,
    settings_path: Path = config.FINAL_SETTINGS_JSON,
    provenance_path: Path = config.VIDEO_PROVENANCE_CSV,
    tracker_factory: Callable[[float, float], _Tracker] | None = None,
) -> VideoCountResult:
    """Track a provenance-cleared video and write an annotated video and crossing CSV.

    Inbound means movement from the negative to the positive side of the directed
    line (``line_start`` to ``line_end``); outbound is the reverse. Frame numbers
    in the CSV are zero-based. A vehicle's bottom-centre anchor is used for crossing.
    """
    video_path = Path(input_path)
    output_path = Path(annotated_output_path)
    crossings_path = Path(crossings_csv_path)
    verify_video_provenance(video_path.name, provenance_path)

    settings_record = settings.settings_record(settings_path)
    model_key = config.DEFAULT_MODEL_NAME
    model_settings = settings_record["models"].get(model_key)
    if not model_settings:
        raise VideoProcessingError(
            f"Frozen settings do not include the video model {model_key}."
        )
    confidence = float(model_settings["confidence"])
    input_size = model_settings.get("input_size")
    nms_iou = model_settings.get("nms_iou") if model_settings.get("uses_nms") else None

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise VideoProcessingError(f"Could not open video file {video_path.name}.")

    writer: cv2.VideoWriter | None = None
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise VideoProcessingError("Video has invalid frame rate or dimensions.")
        if line_start == line_end:
            raise VideoProcessingError("The virtual line endpoints must be different.")
        for x, y in (line_start, line_end):
            if not (0 <= x < width and 0 <= y < height):
                raise VideoProcessingError(
                    f"Line coordinates must be inside the {width}x{height} video frame."
                )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        crossings_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(
            *("mp4v" if output_path.suffix.lower() == ".mp4" else "MJPG")
        )
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise VideoProcessingError(
                f"Could not create annotated output video at {output_path}."
            )

        tracker = (tracker_factory or _create_bytetrack)(fps, confidence)
        track_sides: dict[int, tuple[int, tuple[float, float]]] = {}
        crossings: list[dict[str, int | str]] = []
        inbound = 0
        outbound = 0
        frame_number = 0

        while True:
            success, frame = capture.read()
            if not success:
                break
            detections = detector.detect(
                frame,
                confidence,
                input_size=input_size,
                nms_iou=nms_iou,
            )
            tracked = tracker.update(_to_supervision_detections(detections))
            tracker_ids = tracked.tracker_id
            if tracker_ids is not None:
                for index, track_value in enumerate(tracker_ids):
                    track_id = int(track_value)
                    if track_id < 0:
                        continue
                    x0, y0, x1, y1 = tracked.xyxy[index]
                    anchor = (float((x0 + x1) / 2), float(y1))
                    side = _signed_side(anchor, line_start, line_end)
                    if side == 0:
                        continue
                    previous = track_sides.get(track_id)
                    if previous is not None and previous[0] != side:
                        if _crosses_line_segment(
                            previous[1], anchor, line_start, line_end
                        ):
                            direction = _line_direction(previous[0], side)
                            crossings.append(
                                {
                                    "track_id": track_id,
                                    "frame": frame_number,
                                    "direction": direction,
                                }
                            )
                            if direction == "inbound":
                                inbound += 1
                            else:
                                outbound += 1
                    track_sides[track_id] = (side, anchor)

            _draw_frame(frame, tracked, line_start, line_end, inbound, outbound)
            writer.write(frame)
            frame_number += 1

        if frame_number == 0:
            raise VideoProcessingError("Video contains no readable frames.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    with crossings_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ("track_id", "frame", "direction")
        csv_writer = csv.DictWriter(handle, fieldnames=fields)
        csv_writer.writeheader()
        csv_writer.writerows(crossings)

    return VideoCountResult(
        inbound=inbound,
        outbound=outbound,
        crossings=tuple(crossings),
        frames_processed=frame_number,
        width=width,
        height=height,
        fps=fps,
        confidence=confidence,
        model=detector.name,
    )
