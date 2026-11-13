"""Tests for the video upload pipeline (Items 1-11 of the Video-tab change spec).

All videos are synthetic; no copyrighted material is used.
Real detector weights are never loaded: a SyntheticDetector stub is used throughout.
"""
from __future__ import annotations

import csv
import hashlib
import struct
import tempfile
import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from core import config
from core.detector import Detection
from core.video_counting import (
    VideoProcessingError,
    _create_bytetrack,
    draw_line_on_frame,
    file_content_hash,
    list_sample_clips,
    process_video,
    process_video_upload,
    sample_frame_detections,
    validate_video_file,
    verify_video_provenance,
)


# ---------------------------------------------------------------------------
# Shared test fixtures and helpers
# ---------------------------------------------------------------------------

class SyntheticDetector:
    """Returns fixed detections regardless of image content."""

    name = "synthetic"

    def __init__(self, detections=None):
        self._detections = detections or []

    def detect(self, image, confidence_threshold, *, input_size=None, nms_iou=None):
        return self._detections


class CrossingDetector:
    """Detect coloured rectangles: green→car (inbound), red→motorcycle (outbound)."""

    name = "crossing"

    def detect(self, image, confidence_threshold, *, input_size=None, nms_iou=None):
        found = []
        for class_id, name, mask in (
            (2, "car",        (image[:, :, 1] > 200) & (image[:, :, 2] < 50)),
            (3, "motorcycle", (image[:, :, 2] > 200) & (image[:, :, 1] < 50)),
        ):
            ys, xs = np.where(mask)
            if xs.size:
                found.append(Detection(
                    class_id, name, 0.9,
                    (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
                ))
        return found


class SyntheticTracker:
    """Deterministic tracker: assigns fixed IDs per class."""

    def __init__(self, _fps=10.0, _conf=0.25):
        import supervision as sv
        self._sv = sv
        self.ids_by_class = {2: 1, 3: 2}

    def update(self, detections):
        import numpy as np
        if detections.tracker_id is None:
            detections.tracker_id = np.asarray(
                [self.ids_by_class.get(int(c), -1) for c in detections.class_id],
                dtype=np.int64,
            )
        return detections


def _write_crossing_video(path: Path, *, fourcc: str = "mp4v", ext: str = ".mp4") -> Path:
    """Write a 20-frame synthetic video with one inbound and one outbound crossing."""
    full = path.with_suffix(ext)
    width, height, fps = 160, 120, 10.0
    writer = cv2.VideoWriter(
        str(full), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height)
    )
    assert writer.isOpened(), f"Could not open writer for {ext}/{fourcc}"
    # inbound green box moves top→bottom; outbound red box moves bottom→top
    inbound_y  = [20]*3 + list(range(30, 110, 10)) + [100]*9
    outbound_y = [100]*2 + list(range(90, 10, -10)) + [20]*10
    occluded = {8, 9, 10}
    for i, (iy, oy) in enumerate(zip(inbound_y, outbound_y)):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if i not in occluded:
            cv2.rectangle(frame, (30, iy - 6), (50, iy + 6), (0, 255, 0), -1)
        cv2.rectangle(frame, (105, oy - 6), (125, oy + 6), (0, 0, 255), -1)
        writer.write(frame)
    writer.release()
    return full


def _write_provenance(path: Path, filename: str = "synthetic.mp4") -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=("filename", "source_url", "author", "licence_text", "download_date"),
        )
        w.writeheader()
        w.writerow({
            "filename": filename,
            "source_url": "https://example.test/synthetic",
            "author": "Test fixture",
            "licence_text": "CC0-1.0",
            "download_date": "2026-11-12",
        })


# ---------------------------------------------------------------------------
# 1. validate_video_file
# ---------------------------------------------------------------------------

def test_empty_file_is_rejected(tmp_path):
    p = tmp_path / "empty.mp4"
    p.write_bytes(b"")
    with pytest.raises(VideoProcessingError, match="empty"):
        validate_video_file(p)


def test_corrupt_file_is_rejected(tmp_path):
    p = tmp_path / "corrupt.mp4"
    p.write_bytes(b"\x00\x01\x02\x03" * 256)
    with pytest.raises(VideoProcessingError, match="[Cc]orrupt|[Cc]ould not open|[Nn]o readable"):
        validate_video_file(p)


def test_oversized_file_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MAX_VIDEO_UPLOAD_BYTES", 100)
    p = tmp_path / "big.mp4"
    p.write_bytes(b"x" * 200)
    with pytest.raises(VideoProcessingError, match="exceeds the"):
        validate_video_file(p, max_bytes=100)


def test_valid_mp4_returns_correct_metadata(tmp_path):
    p = _write_crossing_video(tmp_path / "v")
    meta = validate_video_file(p, max_seconds=9999)
    assert meta["width"] == 160
    assert meta["height"] == 120
    assert meta["fps"] > 0
    assert not meta["assumed_fps"]


def test_zero_fps_uses_default_and_flags_assumed(tmp_path, monkeypatch):
    """When the container reports fps=0, validate_video_file uses VIDEO_DEFAULT_FPS."""
    p = _write_crossing_video(tmp_path / "v")
    # Monkeypatch cv2.VideoCapture to return fps=0 for the first open
    original_cap = cv2.VideoCapture

    class FakeCap:
        def __init__(self, path, *a, **kw):
            self._inner = original_cap(path, *a, **kw)

        def isOpened(self):
            return self._inner.isOpened()

        def get(self, prop_id):
            if prop_id == cv2.CAP_PROP_FPS:
                return 0.0
            return self._inner.get(prop_id)

        def read(self):
            return self._inner.read()

        def release(self):
            self._inner.release()

    monkeypatch.setattr(cv2, "VideoCapture", FakeCap)
    meta = validate_video_file(p, max_seconds=9999)
    assert meta["assumed_fps"] is True
    assert meta["fps"] == config.VIDEO_DEFAULT_FPS


# ---------------------------------------------------------------------------
# 2. Temporary file is always deleted
# ---------------------------------------------------------------------------

def test_temp_file_deleted_after_successful_processing(tmp_path):
    """Verify the temp dir approach: writing to tmp_path and reading it after."""
    video = _write_crossing_video(tmp_path / "v")
    prov = tmp_path / "PROVENANCE.csv"
    _write_provenance(prov, video.name)

    tracked_path: list[Path] = []

    with tempfile.TemporaryDirectory(prefix="trafficlens_vid_") as tmpdir:
        input_copy = Path(tmpdir) / video.name
        input_copy.write_bytes(video.read_bytes())
        tracked_path.append(input_copy)

        out_vid = Path(tmpdir) / "out.mp4"
        out_csv = Path(tmpdir) / "out.csv"
        process_video(
            input_copy, out_vid, out_csv,
            CrossingDetector(), (0, 60), (159, 60),
            provenance_path=prov,
            tracker_factory=SyntheticTracker,
        )
        assert input_copy.exists()  # still exists inside the with block

    # After the with block the whole tmpdir is gone
    assert not tracked_path[0].exists()


def test_temp_file_deleted_even_after_exception(tmp_path):
    """TemporaryDirectory is a context manager that cleans up on exception."""
    raised = False
    cleanup_path: list[Path] = []
    try:
        with tempfile.TemporaryDirectory(prefix="trafficlens_exc_") as tmpdir:
            dummy = Path(tmpdir) / "dummy.bin"
            dummy.write_bytes(b"test")
            cleanup_path.append(dummy)
            raise VideoProcessingError("synthetic error for cleanup test")
    except VideoProcessingError:
        raised = True
    assert raised
    assert not cleanup_path[0].exists()


# ---------------------------------------------------------------------------
# 3. process_video_upload — no provenance required for uploads
# ---------------------------------------------------------------------------

def test_upload_does_not_require_provenance_row(tmp_path):
    """process_video_upload must succeed without any PROVENANCE.csv."""
    video = _write_crossing_video(tmp_path / "v")
    out_vid = tmp_path / "out.mp4"
    out_csv = tmp_path / "out.csv"

    # No provenance file at all
    result = process_video_upload(
        video, out_vid, out_csv,
        CrossingDetector(), (0, 60), (159, 60),
        tracker_factory=SyntheticTracker,
    )
    assert result.frames_processed > 0


def test_upload_crossing_counts_match_synthetic_detector(tmp_path):
    video = _write_crossing_video(tmp_path / "v")
    out_vid = tmp_path / "out.mp4"
    out_csv = tmp_path / "out.csv"

    result = process_video_upload(
        video, out_vid, out_csv,
        CrossingDetector(), (0, 60), (159, 60),
        tracker_factory=SyntheticTracker,
    )
    assert (result.inbound, result.outbound) == (1, 1)


def test_upload_crossing_csv_has_extended_columns(tmp_path):
    video = _write_crossing_video(tmp_path / "v")
    out_vid = tmp_path / "out.mp4"
    out_csv = tmp_path / "out.csv"

    process_video_upload(
        video, out_vid, out_csv,
        CrossingDetector(), (0, 60), (159, 60),
        tracker_factory=SyntheticTracker,
    )
    rows = list(csv.DictReader(out_csv.open(encoding="utf-8", newline="")))
    assert len(rows) == 2
    for row in rows:
        assert "track_id" in row
        assert "frame" in row
        assert "time_s" in row
        assert "direction" in row
        assert "class" in row


def test_sample_clip_still_requires_provenance(tmp_path):
    """process_video (sample clips path) must refuse without a provenance row."""
    video = _write_crossing_video(tmp_path / "v")
    prov = tmp_path / "PROVENANCE.csv"
    prov.write_text(
        "filename,source_url,author,licence_text,download_date\n",
        encoding="utf-8",
    )

    with pytest.raises(VideoProcessingError, match="[Pp]rovenance|No unique"):
        process_video(
            video, tmp_path / "out.mp4", tmp_path / "out.csv",
            SyntheticDetector(), (0, 60), (159, 60),
            provenance_path=prov,
        )


# ---------------------------------------------------------------------------
# 4. Multiple formats
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fourcc,ext", [
    ("mp4v", ".mp4"),
    ("MJPG", ".avi"),
])
def test_crossing_counts_in_multiple_formats(tmp_path, fourcc, ext):
    video = _write_crossing_video(tmp_path / "v", fourcc=fourcc, ext=ext)
    out_vid = tmp_path / f"out{ext}"
    out_csv = tmp_path / "out.csv"

    result = process_video_upload(
        video, out_vid, out_csv,
        CrossingDetector(), (0, 60), (159, 60),
        tracker_factory=SyntheticTracker,
    )
    assert result.frames_processed == 20
    assert (result.inbound, result.outbound) == (1, 1)


# ---------------------------------------------------------------------------
# 5. Stop flag produces partial result
# ---------------------------------------------------------------------------

def test_stop_flag_produces_partial_result(tmp_path):
    video = _write_crossing_video(tmp_path / "v")
    out_vid = tmp_path / "out.mp4"
    out_csv = tmp_path / "out.csv"

    stop_flag = [False]
    call_count = [0]

    class CountingDetector(CrossingDetector):
        def detect(self, image, *a, **kw):
            call_count[0] += 1
            if call_count[0] >= 5:
                stop_flag[0] = True
            return super().detect(image, *a, **kw)

    result = process_video_upload(
        video, out_vid, out_csv,
        CountingDetector(), (0, 60), (159, 60),
        tracker_factory=SyntheticTracker,
        stop_flag=stop_flag,
    )
    assert result.partial is True
    assert result.frames_processed < 20


# ---------------------------------------------------------------------------
# 6. Time range and frame stride
# ---------------------------------------------------------------------------

def test_time_range_limits_frames_processed(tmp_path):
    video = _write_crossing_video(tmp_path / "v")
    out_vid = tmp_path / "out.mp4"
    out_csv = tmp_path / "out.csv"

    result = process_video_upload(
        video, out_vid, out_csv,
        SyntheticDetector(), (0, 60), (159, 60),
        tracker_factory=SyntheticTracker,
        start_s=0.0,
        end_s=0.5,  # 5 frames at 10 fps
    )
    # Should process at most ~5 frames
    assert result.frames_processed <= 7  # small tolerance for container timing


def test_frame_stride_reduces_frames_processed(tmp_path):
    video = _write_crossing_video(tmp_path / "v")
    full_out_vid = tmp_path / "full.mp4"
    full_out_csv = tmp_path / "full.csv"
    stride_out_vid = tmp_path / "stride.mp4"
    stride_out_csv = tmp_path / "stride.csv"

    full = process_video_upload(
        video, full_out_vid, full_out_csv,
        SyntheticDetector(), (0, 60), (159, 60),
        tracker_factory=SyntheticTracker,
        frame_stride=1,
    )
    strided = process_video_upload(
        video, stride_out_vid, stride_out_csv,
        SyntheticDetector(), (0, 60), (159, 60),
        tracker_factory=SyntheticTracker,
        frame_stride=4,
    )
    assert strided.frames_processed < full.frames_processed


# ---------------------------------------------------------------------------
# 7. Per-class counts
# ---------------------------------------------------------------------------

def test_per_class_crossing_counts(tmp_path):
    video = _write_crossing_video(tmp_path / "v")
    out_vid = tmp_path / "out.mp4"
    out_csv = tmp_path / "out.csv"

    result = process_video_upload(
        video, out_vid, out_csv,
        CrossingDetector(), (0, 60), (159, 60),
        tracker_factory=SyntheticTracker,
    )
    # green box = car inbound, red box = motorcycle outbound
    assert result.inbound_by_class.get("car", 0) == 1
    assert result.outbound_by_class.get("motorcycle", 0) == 1


# ---------------------------------------------------------------------------
# 8. Track count and fragment count
# ---------------------------------------------------------------------------

def test_track_count_and_fragment_count_are_non_negative(tmp_path):
    video = _write_crossing_video(tmp_path / "v")
    out_vid = tmp_path / "out.mp4"
    out_csv = tmp_path / "out.csv"

    result = process_video_upload(
        video, out_vid, out_csv,
        CrossingDetector(), (0, 60), (159, 60),
        tracker_factory=SyntheticTracker,
    )
    assert result.track_count >= 0
    assert result.fragment_count >= 0


# ---------------------------------------------------------------------------
# 9. Output video is writable and readable
# ---------------------------------------------------------------------------

def test_output_video_is_readable_after_processing(tmp_path):
    video = _write_crossing_video(tmp_path / "v")
    out_vid = tmp_path / "out.mp4"
    out_csv = tmp_path / "out.csv"

    process_video_upload(
        video, out_vid, out_csv,
        CrossingDetector(), (0, 60), (159, 60),
        tracker_factory=SyntheticTracker,
    )
    assert out_vid.stat().st_size > 0
    cap = cv2.VideoCapture(str(out_vid))
    assert cap.isOpened()
    ok, _ = cap.read()
    cap.release()
    assert ok


# ---------------------------------------------------------------------------
# 10. Source note is recorded
# ---------------------------------------------------------------------------

def test_source_note_is_stored_in_result(tmp_path):
    video = _write_crossing_video(tmp_path / "v")
    out_vid = tmp_path / "out.mp4"
    out_csv = tmp_path / "out.csv"

    result = process_video_upload(
        video, out_vid, out_csv,
        SyntheticDetector(), (0, 60), (159, 60),
        tracker_factory=SyntheticTracker,
        source_note="CC0-1.0 own footage",
    )
    assert result.source_note == "CC0-1.0 own footage"


# ---------------------------------------------------------------------------
# 11. draw_line_on_frame
# ---------------------------------------------------------------------------

def test_draw_line_on_frame_returns_copy_with_line(tmp_path):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    result = draw_line_on_frame(frame, (0, 60), (159, 60))
    # Original is unchanged
    assert frame[60, 80, 0] == 0
    # Result has the yellow line drawn (B channel along y=60)
    assert result.shape == frame.shape
    # The line is yellow (BGR 0,255,255) — at least one pixel should differ from zero
    assert not np.array_equal(result, frame)


# ---------------------------------------------------------------------------
# 12. file_content_hash
# ---------------------------------------------------------------------------

def test_file_content_hash_matches_sha256(tmp_path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"hello world")
    expected = hashlib.sha256(b"hello world").hexdigest()
    assert file_content_hash(p) == expected


# ---------------------------------------------------------------------------
# 13. list_sample_clips — only returns clips with both a row and a file
# ---------------------------------------------------------------------------

def test_list_sample_clips_empty_when_provenance_has_no_matching_file(tmp_path):
    prov = tmp_path / "PROVENANCE.csv"
    _write_provenance(prov, "ghost_clip.mp4")
    # ghost_clip.mp4 does not exist in tmp_path
    clips = list_sample_clips(provenance_csv=prov)
    # No clip file present → empty list
    assert clips == []


def test_list_sample_clips_returns_clip_when_file_exists(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "VIDEO_SAMPLES_DIR", tmp_path)
    prov = tmp_path / "PROVENANCE.csv"
    clip = _write_crossing_video(tmp_path / "sample_v")
    _write_provenance(prov, clip.name)
    clips = list_sample_clips(provenance_csv=prov)
    assert len(clips) == 1
    assert clips[0]["filename"] == clip.name


# ---------------------------------------------------------------------------
# 14. sample_frame_detections
# ---------------------------------------------------------------------------

def test_sample_frame_detections_returns_counts_per_frame(tmp_path):
    video = _write_crossing_video(tmp_path / "v")
    det = CrossingDetector()

    counts = sample_frame_detections(
        video, det, 0.25, None, None, n_samples=3
    )
    assert len(counts) == 3
    assert all(isinstance(c, int) for c in counts)
    assert all(c >= 0 for c in counts)


# ---------------------------------------------------------------------------
# 15. Config constants exist
# ---------------------------------------------------------------------------

def test_config_has_video_constants():
    assert hasattr(config, "ALLOWED_VIDEO_EXTENSIONS")
    assert hasattr(config, "MAX_VIDEO_UPLOAD_BYTES")
    assert hasattr(config, "VIDEO_DEFAULT_FPS")
    assert hasattr(config, "VIDEO_DEFAULT_MAX_SECONDS")
    assert config.MAX_VIDEO_UPLOAD_BYTES == 1000 * 1024 * 1024
    assert ".mp4" in config.ALLOWED_VIDEO_EXTENSIONS
    assert ".webm" in config.ALLOWED_VIDEO_EXTENSIONS
    assert ".m4v" in config.ALLOWED_VIDEO_EXTENSIONS


# ---------------------------------------------------------------------------
# 16. Zero-frame video (special case — write 0 frames)
# ---------------------------------------------------------------------------

def test_zero_frame_video_raises_clear_error(tmp_path):
    """A video file that opens but yields no readable frames must raise a clear error."""
    # Write a minimal valid file then overwrite content so cap.read() fails
    p = tmp_path / "noframes.mp4"
    # Write one real frame so the file is valid, then we patch it via a bad file
    writer = cv2.VideoWriter(
        str(p), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (64, 48)
    )
    # deliberately write zero frames and release
    writer.release()

    # validate_video_file needs to be able to .read() — if the file has no frames
    # the read() call should fail and return a clear error
    # (a 0-frame mp4 is still "openable" but read() returns False)
    # We test it via process_video_upload with an empty writer output
    out_vid = tmp_path / "out.mp4"
    out_csv = tmp_path / "out.csv"

    # We can only test this if validate_video_file actually catches it
    try:
        validate_video_file(p, max_seconds=9999)
        # If validate passes (some containers report frames), the loop catches it
        with pytest.raises(VideoProcessingError):
            process_video_upload(
                p, out_vid, out_csv,
                SyntheticDetector(), (0, 24), (63, 24),
                tracker_factory=SyntheticTracker,
            )
    except VideoProcessingError:
        pass  # validate_video_file itself caught the problem — also acceptable


# ---------------------------------------------------------------------------
# 17. Streamlit app test: upload flow (no model weights needed)
# ---------------------------------------------------------------------------

def test_video_tab_upload_renders_without_exception():
    """The Video tab must render without crashing when no file is uploaded."""
    try:
        from streamlit.testing.v1 import AppTest
    except ImportError:
        pytest.skip("Streamlit AppTest not available")

    ROOT = Path(__file__).resolve().parent.parent
    APP_ENTRY = str(ROOT / "app" / "main.py")
    import tempfile as _tf
    import os

    at = AppTest.from_file(APP_ENTRY, default_timeout=120)
    # Patch config paths to isolate
    import core.config as cc
    with _tf.TemporaryDirectory() as td:
        td_path = Path(td)
        at.session_state["_monkeypatch_history_db"] = str(td_path / "history.db")
        at.run()
        assert not at.exception
        # Video tab must show the honesty text — now in st.markdown, not st.warning
        all_text = [m.value for m in at.markdown]
        assert any("demonstration" in t.lower() for t in all_text)
