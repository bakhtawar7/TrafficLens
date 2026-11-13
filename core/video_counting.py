"""Vehicle tracking and virtual-line counts for video.

Two entry points:
- ``process_video_upload``: for files uploaded through the UI.  No provenance row
  is required.  Used only by the interactive tab.
- ``process_video``: for sample clips stored in ``data/samples/video/``.  A
  complete PROVENANCE.csv row is still required for those files.

Both share the same counting kernel (``_run_counting_loop``).
"""
from __future__ import annotations

import csv
import hashlib
import math
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Callable, Iterator, Protocol
from urllib.parse import urlparse

import cv2
import numpy as np
import supervision as sv

from . import config, settings
from .detector import Detection, Detector


# ---------------------------------------------------------------------------
# Public error type
# ---------------------------------------------------------------------------

class VideoProcessingError(ValueError):
    """A human-readable error for video pipeline failures."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VideoCountResult:
    """Summary of one completed (or stopped) video pass."""

    inbound: int
    outbound: int
    crossings: tuple[dict, ...]
    frames_processed: int
    width: int
    height: int
    fps: float
    confidence: float
    model: str
    # Per-class and track statistics
    inbound_by_class: dict[str, int] = field(default_factory=dict)
    outbound_by_class: dict[str, int] = field(default_factory=dict)
    track_count: int = 0
    fragment_count: int = 0
    partial: bool = False
    source_note: str = ""


# ---------------------------------------------------------------------------
# Tracker protocol
# ---------------------------------------------------------------------------

class _Tracker(Protocol):
    def update(self, detections: sv.Detections) -> sv.Detections:
        ...


# ---------------------------------------------------------------------------
# Provenance gate (kept for sample clips)
# ---------------------------------------------------------------------------

def verify_video_provenance(
    filename: str, provenance_csv: Path = config.VIDEO_PROVENANCE_CSV
) -> dict[str, str]:
    """Require a complete provenance record for a sample clip filename."""
    path = Path(provenance_csv)
    if not path.is_file():
        raise VideoProcessingError(
            f"No video provenance file found at {path}. "
            "Add a row to data/samples/video/PROVENANCE.csv first."
        )
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    matches = [
        row for row in rows
        if (row.get("filename") or "").strip().casefold() == Path(filename).name.casefold()
    ]
    if len(matches) != 1:
        raise VideoProcessingError(
            f"No unique provenance row for {Path(filename).name}. "
            "Add its source URL, author, licence text, and download date to PROVENANCE.csv."
        )

    row = matches[0]
    required = ("source_url", "author", "licence_text", "download_date")
    missing = [f for f in required if not (row.get(f) or "").strip()]
    if missing:
        raise VideoProcessingError(
            f"Video provenance for {Path(filename).name} is incomplete: "
            f"{', '.join(missing)}."
        )
    parsed_url = urlparse(row["source_url"].strip())
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname:
        raise VideoProcessingError(
            "Video provenance source_url must be an HTTP(S) URL."
        )
    try:
        date.fromisoformat(row["download_date"].strip())
    except ValueError as error:
        raise VideoProcessingError(
            "Video provenance download_date must use YYYY-MM-DD format."
        ) from error
    return {f: (row.get(f) or "").strip() for f in ("filename", *required)}


def list_sample_clips(
    provenance_csv: Path = config.VIDEO_PROVENANCE_CSV,
) -> list[dict[str, str]]:
    """Return rows from PROVENANCE.csv whose clip file exists in VIDEO_SAMPLES_DIR."""
    if not provenance_csv.is_file():
        return []
    with provenance_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = []
    for row in rows:
        filename = (row.get("filename") or "").strip()
        if not filename:
            continue
        clip_path = config.VIDEO_SAMPLES_DIR / filename
        if clip_path.is_file():
            result.append(
                {
                    "filename": filename,
                    "path": str(clip_path),
                    "author": (row.get("author") or "").strip(),
                    "licence_text": (row.get("licence_text") or "").strip(),
                    "source_url": (row.get("source_url") or "").strip(),
                }
            )
    return result


# ---------------------------------------------------------------------------
# Video validation
# ---------------------------------------------------------------------------

def validate_video_file(
    path: Path,
    *,
    max_bytes: int = config.MAX_VIDEO_UPLOAD_BYTES,
    max_seconds: float = config.VIDEO_DEFAULT_MAX_SECONDS,
    default_fps: float = config.VIDEO_DEFAULT_FPS,
) -> dict:
    """Open ``path`` with OpenCV and return metadata, raising ``VideoProcessingError`` on any problem.

    Returns a dict with keys: width, height, fps, assumed_fps, frame_count, duration_s.
    ``assumed_fps`` is True when the container reported 0/NaN and the default was used.
    """
    size = path.stat().st_size
    if size == 0:
        raise VideoProcessingError(
            "The uploaded file is empty (0 bytes). "
            "Check that the upload completed and try again."
        )
    if size > max_bytes:
        limit_mb = max_bytes // (1024 * 1024)
        raise VideoProcessingError(
            f"The video is {size // (1024*1024)} MB, which exceeds the {limit_mb} MB limit. "
            "Trim the clip before uploading. Example ffmpeg command:\n\n"
            "    ffmpeg -i input.mp4 -ss 0 -t 60 -c copy trimmed.mp4"
        )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoProcessingError(
            f"Could not open the video file '{path.name}'. "
            "The file may be corrupt or use a codec OpenCV cannot decode. "
            "If the file uses HEVC (H.265), convert it first:\n\n"
            "    ffmpeg -i input.mp4 -vcodec libx264 -acodec aac converted.mp4"
        )
    try:
        raw_fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Try reading a first frame to catch zero-frame or unreadable files
        ok, _ = cap.read()
    finally:
        cap.release()

    if not ok or width <= 0 or height <= 0:
        raise VideoProcessingError(
            f"'{path.name}' has no readable frames or invalid dimensions. "
            "The file may be corrupt or truncated."
        )

    assumed_fps = False
    if not math.isfinite(raw_fps) or raw_fps <= 0:
        fps = default_fps
        assumed_fps = True
    else:
        fps = raw_fps

    # frame_count from the container header is often unreliable; use it only as a hint
    if frame_count_prop > 0:
        duration_s = frame_count_prop / fps
    else:
        duration_s = None  # will be measured during processing

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "assumed_fps": assumed_fps,
        "frame_count": frame_count_prop if frame_count_prop > 0 else None,
        "duration_s": duration_s,
    }


def file_content_hash(path: Path) -> str:
    """Return the SHA-256 of ``path``'s bytes."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# ByteTrack factory
# ---------------------------------------------------------------------------

def _create_bytetrack(frame_rate: float, confidence: float) -> _Tracker:
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


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Frame drawing
# ---------------------------------------------------------------------------

def _draw_frame(
    frame: np.ndarray,
    detections: sv.Detections,
    line_start: tuple[int, int],
    line_end: tuple[int, int],
    inbound: int,
    outbound: int,
    inbound_label: str = "Inbound",
    outbound_label: str = "Outbound",
) -> None:
    cv2.line(frame, line_start, line_end, (0, 255, 255), 2)
    tracker_ids = detections.tracker_id
    for index, box in enumerate(detections.xyxy):
        x0, y0, x1, y1 = (int(round(float(v))) for v in box)
        track_id = int(tracker_ids[index]) if tracker_ids is not None else -1
        cv2.rectangle(frame, (x0, y0), (x1, y1), (60, 220, 60), 2)
        cv2.putText(
            frame, f"ID {track_id}",
            (x0, max(16, y0 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 220, 60), 1, cv2.LINE_AA,
        )
    cv2.putText(
        frame,
        f"{inbound_label}: {inbound}  {outbound_label}: {outbound}",
        (12, 26),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Line overlay helper (for preview)
# ---------------------------------------------------------------------------

def draw_line_on_frame(
    frame: np.ndarray,
    line_start: tuple[int, int],
    line_end: tuple[int, int],
    inbound_label: str = "Inbound",
    outbound_label: str = "Outbound",
) -> np.ndarray:
    """Return a copy of ``frame`` with the counting line drawn on it."""
    out = frame.copy()
    cv2.line(out, line_start, line_end, (0, 255, 255), 2)
    # Label the two sides near the midpoint
    mid_x = (line_start[0] + line_end[0]) // 2
    mid_y = (line_start[1] + line_end[1]) // 2
    cv2.putText(
        out, f"← {outbound_label} | {inbound_label} →",
        (max(0, mid_x - 80), max(16, mid_y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA,
    )
    return out


# ---------------------------------------------------------------------------
# H.264 encoding via ffmpeg (best-effort)
# ---------------------------------------------------------------------------

def encode_h264(raw_path: Path, out_path: Path) -> bool:
    """Re-encode ``raw_path`` to H.264 MP4 using ffmpeg.  Returns True on success."""
    if shutil.which("ffmpeg") is None:
        return False
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(raw_path),
                "-vcodec", "libx264", "-preset", "fast",
                "-crf", "23", "-movflags", "+faststart",
                "-an", str(out_path),
            ],
            capture_output=True, timeout=300,
        )
        return result.returncode == 0 and out_path.stat().st_size > 0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# View check: sample three frames
# ---------------------------------------------------------------------------

def sample_frame_detections(
    video_path: Path,
    detector: Detector,
    confidence: float,
    input_size: int | None,
    nms_iou: float | None,
    *,
    n_samples: int = 3,
    start_s: float = 0.0,
    end_s: float | None = None,
) -> list[int]:
    """Detect on ``n_samples`` evenly-spaced frames; return per-frame vehicle counts."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or config.VIDEO_DEFAULT_FPS
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        start_frame = int(start_s * fps)
        end_frame = int(end_s * fps) if end_s is not None and end_s > 0 else total
        end_frame = min(end_frame, total) if total > 0 else end_frame
        span = max(end_frame - start_frame, 1)
        sample_frames = [
            start_frame + int(i * span / max(n_samples - 1, 1))
            for i in range(n_samples)
        ]
        counts = []
        for frame_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
            ok, frame = cap.read()
            if not ok:
                counts.append(0)
                continue
            dets = detector.detect(frame, confidence, input_size=input_size, nms_iou=nms_iou)
            vehicle_dets = [d for d in dets if d.class_name in config.TARGET_CLASS_NAMES]
            counts.append(len(vehicle_dets))
        return counts
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Core counting loop (shared by both entry points)
# ---------------------------------------------------------------------------

def _run_counting_loop(
    input_path: Path,
    annotated_output_path: Path,
    crossings_csv_path: Path,
    detector: Detector,
    line_start: tuple[int, int],
    line_end: tuple[int, int],
    confidence: float,
    input_size: int | None,
    nms_iou: float | None,
    fps: float,
    *,
    tracker_factory: Callable[[float, float], _Tracker] | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
    frame_stride: int = 1,
    inbound_label: str = "Inbound",
    outbound_label: str = "Outbound",
    stop_flag: list[bool] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> VideoCountResult:
    """Core frame loop.  Shared by upload and sample-clip entry points."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise VideoProcessingError(
            f"Could not open '{input_path.name}'. The file may have been removed or is locked."
        )

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_prop = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if line_start == line_end:
        cap.release()
        raise VideoProcessingError("The virtual line start and end points must be different.")
    for x, y in (line_start, line_end):
        if not (0 <= x < width and 0 <= y < height):
            cap.release()
            raise VideoProcessingError(
                f"Line coordinates ({x},{y}) must be inside the {width}×{height} frame."
            )

    output_path = Path(annotated_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    crossings_path = Path(crossings_csv_path)
    crossings_path.parent.mkdir(parents=True, exist_ok=True)

    # Write raw mp4v first; re-encode to H.264 afterwards if ffmpeg available
    raw_output = output_path.with_suffix(".raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_output), fourcc, fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise VideoProcessingError(
            f"Could not create output video at '{raw_output}'. "
            "Check that the output directory is writable."
        )

    tracker = (tracker_factory or _create_bytetrack)(fps, confidence)
    track_sides: dict[int, tuple[int, tuple[float, float]]] = {}
    track_first_seen: dict[int, int] = {}
    crossings: list[dict] = []
    inbound = 0
    outbound = 0
    inbound_by_class: dict[str, int] = {}
    outbound_by_class: dict[str, int] = {}
    fragment_count = 0
    frames_processed = 0
    partial = False

    # Class id → name mapping from registry detections
    class_id_to_name: dict[int, str] = {
        v: k for k, v in {
            "car": 2, "motorcycle": 3, "bus": 5, "truck": 7
        }.items()
    }

    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
    actual_end = end_frame if end_frame is not None else (total_frames_prop or 10**9)

    try:
        while True:
            if stop_flag is not None and stop_flag[0]:
                partial = True
                break
            pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            if pos >= actual_end:
                break
            # frame stride: skip frames between reads
            if frame_stride > 1 and frames_processed > 0:
                for _ in range(frame_stride - 1):
                    cap.grab()

            success, frame = cap.read()
            if not success:
                break

            dets = detector.detect(frame, confidence, input_size=input_size, nms_iou=nms_iou)
            sv_dets = _to_supervision_detections(dets)
            tracked = tracker.update(sv_dets)

            # Map class_id back to name for per-class tracking
            det_class_map: dict[int, str] = {}
            for d in dets:
                if d.class_name in config.TARGET_CLASS_NAMES:
                    det_class_map[d.class_id] = d.class_name

            tids = tracked.tracker_id
            if tids is not None:
                for idx, track_val in enumerate(tids):
                    track_id = int(track_val)
                    if track_id < 0:
                        continue
                    x0, y0, x1, y1 = tracked.xyxy[idx]
                    anchor = (float((x0 + x1) / 2), float(y1))
                    side = _signed_side(anchor, line_start, line_end)
                    if side == 0:
                        continue
                    prev = track_sides.get(track_id)
                    if track_id not in track_first_seen:
                        track_first_seen[track_id] = frames_processed
                    elif track_id in track_first_seen and prev is not None:
                        # Detect re-appearance after a gap → fragment
                        if abs(frames_processed - prev[2]) > fps * 2:  # type: ignore[index]
                            fragment_count += 1
                    if prev is not None and prev[0] != side:
                        if _crosses_line_segment(prev[1], anchor, line_start, line_end):
                            direction = _line_direction(prev[0], side)
                            # Determine class name from class_id in supervised det
                            class_id = int(tracked.class_id[idx]) if tracked.class_id is not None else -1
                            class_name = det_class_map.get(class_id, "unknown")
                            crossings.append({
                                "track_id": track_id,
                                "frame": frames_processed + start_frame,
                                "time_s": round((frames_processed + start_frame) / fps, 3),
                                "direction": direction,
                                "class": class_name,
                            })
                            if direction == "inbound":
                                inbound += 1
                                inbound_by_class[class_name] = inbound_by_class.get(class_name, 0) + 1
                            else:
                                outbound += 1
                                outbound_by_class[class_name] = outbound_by_class.get(class_name, 0) + 1
                    track_sides[track_id] = (side, anchor, frames_processed)  # type: ignore[assignment]

            _draw_frame(
                frame, tracked, line_start, line_end, inbound, outbound,
                inbound_label, outbound_label,
            )
            writer.write(frame)
            frames_processed += 1

            if progress_callback is not None:
                total_est = (actual_end - start_frame) // max(frame_stride, 1)
                progress_callback(frames_processed, total_est)

        if frames_processed == 0:
            raise VideoProcessingError(
                f"'{input_path.name}' contains no readable frames in the selected range."
            )
    finally:
        cap.release()
        writer.release()

    # Try H.264 re-encode; fall back to raw mp4v
    h264_ok = encode_h264(raw_output, output_path)
    if h264_ok:
        raw_output.unlink(missing_ok=True)
        encoding_used = "h264"
    else:
        raw_output.rename(output_path)
        encoding_used = "mp4v"

    # Write crossing CSV with extended columns
    with crossings_path.open("w", encoding="utf-8", newline="") as handle:
        fields = ("track_id", "frame", "time_s", "direction", "class")
        w = csv.DictWriter(handle, fieldnames=fields)
        w.writeheader()
        w.writerows(crossings)

    result = VideoCountResult(
        inbound=inbound,
        outbound=outbound,
        crossings=tuple(crossings),
        frames_processed=frames_processed,
        width=width,
        height=height,
        fps=fps,
        confidence=confidence,
        model=detector.name,
        inbound_by_class=inbound_by_class,
        outbound_by_class=outbound_by_class,
        track_count=len(track_first_seen),
        fragment_count=fragment_count,
        partial=partial,
    )
    # Attach encoding metadata as a non-frozen attribute via object.__setattr__
    object.__setattr__(result, "_encoding_used", encoding_used)
    return result


# ---------------------------------------------------------------------------
# Public entry point 1: uploaded file (no provenance required)
# ---------------------------------------------------------------------------

def process_video_upload(
    input_path: Path,
    annotated_output_path: Path,
    crossings_csv_path: Path,
    detector: Detector,
    line_start: tuple[int, int],
    line_end: tuple[int, int],
    *,
    settings_path: Path = config.FINAL_SETTINGS_JSON,
    tracker_factory: Callable[[float, float], _Tracker] | None = None,
    start_s: float = 0.0,
    end_s: float | None = None,
    frame_stride: int = 1,
    inbound_label: str = "Inbound",
    outbound_label: str = "Outbound",
    stop_flag: list[bool] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    source_note: str = "",
) -> VideoCountResult:
    """Process an uploaded video file.  No provenance row is required."""
    settings_record = settings.settings_record(settings_path)
    model_key = config.DEFAULT_MODEL_NAME
    model_settings = settings_record["models"].get(model_key)
    if not model_settings:
        raise VideoProcessingError(
            f"Frozen settings do not include model '{model_key}'."
        )
    confidence = float(model_settings["confidence"])
    input_size = model_settings.get("input_size")
    nms_iou = model_settings.get("nms_iou") if model_settings.get("uses_nms") else None

    meta = validate_video_file(input_path)
    fps = meta["fps"]
    start_frame = int(start_s * fps)
    end_frame = int(end_s * fps) if end_s is not None else None

    result = _run_counting_loop(
        input_path, annotated_output_path, crossings_csv_path,
        detector, line_start, line_end,
        confidence, input_size, nms_iou, fps,
        tracker_factory=tracker_factory,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_stride=frame_stride,
        inbound_label=inbound_label,
        outbound_label=outbound_label,
        stop_flag=stop_flag,
        progress_callback=progress_callback,
    )
    # Attach source note
    object.__setattr__(result, "source_note", source_note)
    return result


# ---------------------------------------------------------------------------
# Public entry point 2: sample clip (provenance required)
# ---------------------------------------------------------------------------

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
    """Process a sample clip; a provenance row is required.

    This function keeps the original signature used by tests and scripts.
    """
    verify_video_provenance(Path(input_path).name, provenance_path)

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

    cap_check = cv2.VideoCapture(str(input_path))
    if not cap_check.isOpened():
        cap_check.release()
        raise VideoProcessingError(f"Could not open video file {Path(input_path).name}.")
    fps_raw = cap_check.get(cv2.CAP_PROP_FPS)
    cap_check.release()

    if not math.isfinite(fps_raw) or fps_raw <= 0:
        raise VideoProcessingError("Video has invalid frame rate or dimensions.")

    return _run_counting_loop(
        input_path, annotated_output_path, crossings_csv_path,
        detector, line_start, line_end,
        confidence, input_size, nms_iou, fps_raw,
        tracker_factory=tracker_factory,
    )
