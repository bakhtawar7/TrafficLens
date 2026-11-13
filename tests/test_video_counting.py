"""Synthetic-only tests for provenance-gated video counting."""
from __future__ import annotations

import csv
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
import supervision as sv

from core.detector import Detection
from core.video_counting import (
    VideoProcessingError,
    _create_bytetrack,
    process_video,
    verify_video_provenance,
)


class SyntheticDetector:
    """Detect the coloured rectangles drawn into the synthetic test video."""

    name = "synthetic"

    def __init__(self):
        self.settings_used = []

    def detect(self, image, confidence_threshold, *, input_size=None, nms_iou=None):
        self.settings_used.append((confidence_threshold, input_size, nms_iou))
        found = []
        for class_id, class_name, mask in (
            (2, "car", (image[:, :, 1] > 200) & (image[:, :, 2] < 50)),
            (3, "motorcycle", (image[:, :, 2] > 200) & (image[:, :, 1] < 50)),
        ):
            ys, xs = np.where(mask)
            if xs.size:
                found.append(
                    Detection(
                        class_id,
                        class_name,
                        0.9,
                        (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())),
                    )
                )
        return found


class SyntheticTracker:
    """Keep the synthetic rectangles' known identities through brief occlusion."""

    def __init__(self, _frame_rate: float, _confidence: float):
        self.ids_by_class = {2: 17, 3: 23}

    def update(self, detections: sv.Detections) -> sv.Detections:
        detections.tracker_id = np.asarray(
            [self.ids_by_class[int(class_id)] for class_id in detections.class_id],
            dtype=np.int64,
        )
        return detections


def _write_synthetic_video(path: Path) -> None:
    width, height, fps = 160, 120, 10.0
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    assert writer.isOpened()
    inbound_y = [20, 20, 20, 30, 40, 50, 60, 70, 80, 90, 100, 100] + [100] * 8
    outbound_y = [100, 100, 90, 80, 70, 60, 50, 40, 30, 20, 20, 20] + [20] * 8
    occluded_frames = {8, 9, 10}
    for frame_number, (in_y, out_y) in enumerate(zip(inbound_y, outbound_y)):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if frame_number not in occluded_frames:
            cv2.rectangle(frame, (30, in_y - 6), (50, in_y + 6), (0, 255, 0), -1)
        cv2.rectangle(frame, (105, out_y - 6), (125, out_y + 6), (0, 0, 255), -1)
        writer.write(frame)
    writer.release()


def _write_provenance(path: Path, filename: str = "synthetic.avi") -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("filename", "source_url", "author", "licence_text", "download_date"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "filename": filename,
                "source_url": "https://example.test/synthetic",
                "author": "Synthetic test fixture",
                "licence_text": "CC0-1.0",
                "download_date": "2026-11-12",
            }
        )


def test_synthetic_rectangles_cross_both_directions_through_occlusion(tmp_path):
    video_path = tmp_path / "synthetic.avi"
    provenance_path = tmp_path / "PROVENANCE.csv"
    annotated_path = tmp_path / "annotated.avi"
    crossings_path = tmp_path / "crossings.csv"
    _write_synthetic_video(video_path)
    _write_provenance(provenance_path)
    detector = SyntheticDetector()

    result = process_video(
        video_path,
        annotated_path,
        crossings_path,
        detector,
        (0, 60),
        (159, 60),
        provenance_path=provenance_path,
        tracker_factory=SyntheticTracker,
    )

    assert (result.inbound, result.outbound) == (1, 1)
    assert result.frames_processed == 20
    assert set(detector.settings_used) == {(0.25, 640, 0.7)}
    assert annotated_path.stat().st_size > 0
    assert [
        (row["track_id"], row["direction"])
        for row in result.crossings
    ] == [(17, "inbound"), (23, "outbound")]
    with crossings_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    # CSV now includes time_s and class columns in addition to the original three
    assert "track_id" in rows[0]
    assert "frame" in rows[0]
    assert "direction" in rows[0]
    assert "time_s" in rows[0]
    assert "class" in rows[0]
    assert [row["direction"] for row in rows] == ["inbound", "outbound"]

    capture = cv2.VideoCapture(str(annotated_path))
    assert capture.isOpened()
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == result.frames_processed
    capture.release()


def test_missing_provenance_refuses_before_detector_or_output(tmp_path):
    video_path = tmp_path / "unlicensed.avi"
    video_path.write_bytes(b"synthetic placeholder")
    provenance_path = tmp_path / "PROVENANCE.csv"
    provenance_path.write_text(
        "filename,source_url,author,licence_text,download_date\n", encoding="utf-8"
    )
    annotated_path = tmp_path / "annotated.avi"
    crossings_path = tmp_path / "crossings.csv"

    class DetectorMustNotRun(SyntheticDetector):
        def detect(self, *args, **kwargs):
            raise AssertionError("Detector ran before provenance was checked.")

    with pytest.raises(VideoProcessingError, match="No unique provenance row"):
        process_video(
            video_path,
            annotated_path,
            crossings_path,
            DetectorMustNotRun(),
            (0, 60),
            (159, 60),
            provenance_path=provenance_path,
            tracker_factory=SyntheticTracker,
        )
    assert not annotated_path.exists()
    assert not crossings_path.exists()


def test_incomplete_or_invalid_provenance_is_rejected(tmp_path):
    path = tmp_path / "PROVENANCE.csv"
    path.write_text(
        "filename,source_url,author,licence_text,download_date\n"
        "synthetic.avi,not-a-url,,,not-a-date\n",
        encoding="utf-8",
    )
    with pytest.raises(VideoProcessingError, match="incomplete"):
        verify_video_provenance("synthetic.avi", path)

    path.write_text(
        "filename,source_url,author,licence_text,download_date\n"
        "synthetic.avi,not-a-url,Author,CC0-1.0,2026-11-12\n",
        encoding="utf-8",
    )
    with pytest.raises(VideoProcessingError, match="HTTP"):
        verify_video_provenance("synthetic.avi", path)

    path.write_text(
        "filename,source_url,author,licence_text,download_date\n"
        "synthetic.avi,https://example.test/video,Author,CC0-1.0,not-a-date\n",
        encoding="utf-8",
    )
    with pytest.raises(VideoProcessingError, match="YYYY-MM-DD"):
        verify_video_provenance("synthetic.avi", path)


def test_bytetrack_factory_uses_supervision_compatible_tracker(monkeypatch):
    captured = {}

    class FakeByteTrackTracker:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "trackers",
        types.SimpleNamespace(ByteTrackTracker=FakeByteTrackTracker),
    )
    tracker = _create_bytetrack(25.0, 0.25)
    assert isinstance(tracker, FakeByteTrackTracker)
    assert captured == {
        "frame_rate": 25.0,
        "track_activation_threshold": 0.25,
        "high_conf_det_threshold": 0.6,
        "minimum_consecutive_frames": 2,
    }


def test_real_bytetrack_handles_synthetic_crossing_clip_when_installed(tmp_path):
    pytest.importorskip("trackers")
    from trackers import ByteTrackTracker

    video_path = tmp_path / "synthetic.avi"
    provenance_path = tmp_path / "PROVENANCE.csv"
    _write_synthetic_video(video_path)
    _write_provenance(provenance_path)

    def tracker_factory(fps: float, confidence: float):
        return ByteTrackTracker(
            frame_rate=fps,
            track_activation_threshold=confidence,
            high_conf_det_threshold=max(confidence, 0.6),
            minimum_consecutive_frames=1,
            minimum_iou_threshold=0.0,
        )

    result = process_video(
        video_path,
        tmp_path / "annotated.avi",
        tmp_path / "crossings.csv",
        SyntheticDetector(),
        (0, 60),
        (159, 60),
        provenance_path=provenance_path,
        tracker_factory=tracker_factory,
    )
    assert (result.inbound, result.outbound) == (1, 1)
