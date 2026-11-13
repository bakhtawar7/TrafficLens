"""Video tab: upload, validate, preview, count, and download.

Provenance is NOT required for uploaded files; the user accepts responsibility
via the notice shown in the uploader.  Provenance IS still required for sample
clips stored in data/samples/video/.

No result bytes are stored in the history database; only the filename, content
hash, and optional source note are recorded.
"""
from __future__ import annotations

import hashlib
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from core import config, settings
from core.video_counting import (
    VideoProcessingError,
    draw_line_on_frame,
    list_sample_clips,
    process_video,
    process_video_upload,
    sample_frame_detections,
    validate_video_file,
    verify_video_provenance,
)

# ---------------------------------------------------------------------------
# Accepted extensions (shown to Streamlit's file_uploader)
# ---------------------------------------------------------------------------
_VIDEO_EXTS = [e.lstrip(".") for e in config.ALLOWED_VIDEO_EXTENSIONS]

# ---------------------------------------------------------------------------
# Session-state keys
# ---------------------------------------------------------------------------
_KEY_STOP = "video_stop_flag"
_KEY_CACHE = "video_result_cache"   # dict keyed by cache_key → (result, video_bytes, csv_bytes)
_KEY_UPLOAD_HASH = "video_last_hash"


# ---------------------------------------------------------------------------
# Detector (cached at process level)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the YOLOv8n model for video")
def _get_video_detector():
    from core.detector import build_detector
    return build_detector(config.DEFAULT_MODEL_NAME)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _settings_hash() -> str:
    try:
        return hashlib.sha256(config.FINAL_SETTINGS_JSON.read_bytes()).hexdigest()
    except OSError:
        return "unavailable"


def _cache_key(
    content_hash: str,
    line_start: tuple[int, int],
    line_end: tuple[int, int],
    start_s: float,
    end_s: float,
    stride: int,
    inbound_label: str,
    outbound_label: str,
) -> str:
    return hashlib.sha256(
        f"{content_hash}|{_settings_hash()}|{line_start}|{line_end}"
        f"|{start_s:.3f}|{end_s:.3f}|{stride}|{inbound_label}|{outbound_label}"
        .encode()
    ).hexdigest()


def _first_readable_frame(path: Path) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def _render_preview(frame: np.ndarray, line_start, line_end, inbound_label, outbound_label) -> None:
    overlay = draw_line_on_frame(frame, line_start, line_end, inbound_label, outbound_label)
    st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="First frame with counting line", width="stretch")


def _render_results(result, video_bytes: bytes, csv_bytes: bytes, encoding_used: str) -> None:
    """Display counts and download buttons for a completed or partial result."""
    partial_note = " (partial — processing was stopped early)" if result.partial else ""
    st.success(f"Processed {result.frames_processed} frames at {result.fps:.1f} fps.{partial_note}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Inbound", result.inbound)
    col2.metric("Outbound", result.outbound)
    col3.metric("Tracks", result.track_count)
    col4.metric("ID fragments", result.fragment_count,
                help="Apparent track re-identifications. Each fragment may cause a double count or miss.")

    if result.inbound_by_class or result.outbound_by_class:
        all_classes = sorted(
            set(result.inbound_by_class) | set(result.outbound_by_class)
        )
        import pandas as pd
        rows = [
            {
                "Class": cls,
                "Inbound": result.inbound_by_class.get(cls, 0),
                "Outbound": result.outbound_by_class.get(cls, 0),
            }
            for cls in all_classes
        ]
        st.markdown("**Crossings by class**")
        st.dataframe(pd.DataFrame(rows), hide_index=True)

    enc_note = (
        "Annotated video encoded as H.264 MP4 (browser-playable)."
        if encoding_used == "h264"
        else "ffmpeg not available — annotated video encoded as mp4v. "
             "It may not play in all browsers; use the download button. "
             "To get H.264: install ffmpeg and reprocess."
    )
    st.caption(enc_note)

    st.video(video_bytes)

    dl1, dl2 = st.columns(2)
    dl1.download_button(
        "Download annotated video",
        video_bytes,
        file_name="trafficlens_annotated.mp4",
        mime="video/mp4",
        key="video_dl_annotated",
    )
    dl2.download_button(
        "Download crossing CSV",
        csv_bytes,
        file_name="trafficlens_crossings.csv",
        mime="text/csv",
        key="video_dl_csv",
    )


# ---------------------------------------------------------------------------
# Line configuration panel
# ---------------------------------------------------------------------------

def _line_controls(width: int, height: int) -> tuple[tuple[int, int], tuple[int, int], str, str]:
    """Return (line_start, line_end, inbound_label, outbound_label)."""
    st.markdown("**Counting line**")
    st.caption(
        "Place the line so vehicles cross it perpendicularly. "
        "'Inbound' is movement from the negative to the positive side of the directed "
        "line (start → end); reversing the endpoints swaps the labels."
    )

    preset = st.selectbox(
        "Quick preset",
        ["Custom", "Horizontal mid-frame", "Vertical mid-frame"],
        key="video_line_preset",
        help="Choose a preset to fill in the coordinates, then adjust as needed.",
    )
    if preset == "Horizontal mid-frame":
        default_x1, default_y1 = 0, height // 2
        default_x2, default_y2 = width - 1, height // 2
    elif preset == "Vertical mid-frame":
        default_x1, default_y1 = width // 2, 0
        default_x2, default_y2 = width // 2, height - 1
    else:
        default_x1, default_y1 = 0, height // 2
        default_x2, default_y2 = width - 1, height // 2

    c1, c2, c3, c4 = st.columns(4)
    x1 = int(c1.number_input("Start X", 0, width - 1, default_x1, key="video_x1"))
    y1 = int(c2.number_input("Start Y", 0, height - 1, default_y1, key="video_y1"))
    x2 = int(c3.number_input("End X", 0, width - 1, default_x2, key="video_x2"))
    y2 = int(c4.number_input("End Y", 0, height - 1, default_y2, key="video_y2"))

    d1, d2 = st.columns(2)
    inbound_label = d1.text_input(
        "Name for inbound direction",
        value="Inbound",
        key="video_inbound_label",
        help="Movement from the start-side to the end-side of the line.",
    )
    outbound_label = d2.text_input(
        "Name for outbound direction",
        value="Outbound",
        key="video_outbound_label",
    )
    return (x1, y1), (x2, y2), inbound_label or "Inbound", outbound_label or "Outbound"


# ---------------------------------------------------------------------------
# Processing controls panel
# ---------------------------------------------------------------------------

def _processing_controls(duration_s: float | None) -> tuple[float, float, int]:
    """Return (start_s, end_s, frame_stride)."""
    st.markdown("**Processing range and speed**")
    max_s = duration_s if duration_s and duration_s > 0 else 600.0

    allow_longer = st.checkbox(
        f"Process beyond the default {config.VIDEO_DEFAULT_MAX_SECONDS:.0f} s limit",
        key="video_allow_longer",
        help="Processing time scales linearly with clip length. Long clips can take several minutes.",
    )
    if allow_longer:
        st.warning(
            f"Extended processing is enabled. Clips longer than "
            f"{config.VIDEO_DEFAULT_MAX_SECONDS:.0f} s may take several minutes "
            "depending on hardware."
        )
    effective_max = max_s if allow_longer else min(max_s, config.VIDEO_DEFAULT_MAX_SECONDS)

    c1, c2 = st.columns(2)
    start_s = float(c1.number_input(
        "Start (seconds)", 0.0, float(effective_max), 0.0, 1.0, key="video_start_s",
        help="Skip to this point before processing begins.",
    ))
    end_s = float(c2.number_input(
        "End (seconds)", start_s, float(effective_max), float(effective_max), 1.0,
        key="video_end_s",
        help="Stop processing at this point.",
    ))

    stride = int(st.slider(
        "Frame stride",
        min_value=1, max_value=10, value=1, step=1,
        key="video_stride",
        help=(
            "Process every Nth frame. Stride 1 = every frame (slowest, most accurate). "
            "Stride 2 = every other frame (2× faster, may miss fast crossings)."
        ),
    ))
    if stride > 1:
        st.caption(
            f"Stride {stride}: approximately {1/stride:.0%} of frames will be processed. "
            "Fast-moving vehicles may be missed."
        )
    return start_s, end_s, stride


# ---------------------------------------------------------------------------
# View check
# ---------------------------------------------------------------------------

def _view_check(input_path: Path, confidence: float, input_size, nms_iou,
                start_s: float, end_s: float) -> None:
    if not st.button("Run view check (3 sample frames)", key="video_view_check",
                     help="Detect vehicles on 3 frames to confirm the angle suits these models."):
        return
    with st.spinner("Sampling 3 frames…"):
        try:
            counts = sample_frame_detections(
                input_path, _get_video_detector(), confidence, input_size, nms_iou,
                start_s=start_s, end_s=end_s if end_s > start_s else None,
            )
        except Exception as exc:
            st.warning(f"View check failed: {exc}")
            return
    st.write(f"Vehicles found on 3 sample frames: {counts}")
    if all(c == 0 for c in counts):
        st.warning(
            "No vehicles detected on any sample frame. "
            "These models are trained on street-level footage; "
            "top-down or aerial views, very dark scenes, or heavily occluded traffic "
            "may produce zero detections. You can still proceed."
        )
    else:
        st.success(f"View check passed — vehicles visible in at least one sample frame.")


# ---------------------------------------------------------------------------
# Main pipeline for a path-on-disk video
# ---------------------------------------------------------------------------

def _process_and_show(
    input_path: Path,
    content_hash: str,
    source_note: str,
    meta: dict,
    line_start: tuple[int, int],
    line_end: tuple[int, int],
    start_s: float,
    end_s: float,
    stride: int,
    inbound_label: str,
    outbound_label: str,
    for_sample: bool = False,
    sample_provenance: dict | None = None,
) -> None:
    ck = _cache_key(content_hash, line_start, line_end, start_s, end_s, stride,
                    inbound_label, outbound_label)

    cache: dict = st.session_state.get(_KEY_CACHE, {})
    if ck in cache:
        result, video_bytes, csv_bytes = cache[ck]
        st.info("Showing cached result — settings and line unchanged.")
        enc = getattr(result, "_encoding_used", "mp4v")
        _render_results(result, video_bytes, csv_bytes, enc)
        return

    # --- confidence / size / nms from settings ---
    try:
        srec = settings.settings_record(config.FINAL_SETTINGS_JSON)
        msettings = srec["models"].get(config.DEFAULT_MODEL_NAME, {})
        confidence = float(msettings.get("confidence", config.DEFAULT_CONFIDENCE))
        input_size = msettings.get("input_size")
        nms_iou = msettings.get("nms_iou") if msettings.get("uses_nms") else None
    except Exception:
        confidence = config.DEFAULT_CONFIDENCE
        input_size = None
        nms_iou = None

    _view_check(input_path, confidence, input_size, nms_iou, start_s, end_s)

    if not st.button("Process video", type="primary", key="video_process_btn"):
        return

    # Initialise stop flag in session state
    st.session_state[_KEY_STOP] = [False]

    stop_col, _ = st.columns([1, 4])
    stop_btn = stop_col.button("Stop", key="video_stop_btn",
                               help="End processing cleanly; partial results will be shown.")
    if stop_btn:
        st.session_state[_KEY_STOP][0] = True

    progress_bar = st.progress(0, text="Starting…")
    fps_display = st.empty()
    t_start = time.perf_counter()
    frames_done = [0]

    def _progress(done: int, total: int) -> None:
        frames_done[0] = done
        frac = min(done / max(total, 1), 1.0)
        elapsed = time.perf_counter() - t_start
        fps_now = done / max(elapsed, 1e-6)
        remaining = (total - done) / max(fps_now, 1e-6) if total > done else 0
        progress_bar.progress(frac, text=f"{done}/{total} frames")
        fps_display.caption(
            f"{fps_now:.1f} frames/s — "
            f"~{remaining:.0f} s remaining"
        )

    with tempfile.TemporaryDirectory(prefix="trafficlens_out_") as tmp_out:
        out_video = Path(tmp_out) / "annotated.mp4"
        out_csv = Path(tmp_out) / "crossings.csv"

        try:
            with st.spinner("Processing…"):
                if for_sample:
                    result = process_video(
                        input_path, out_video, out_csv,
                        _get_video_detector(), line_start, line_end,
                        provenance_path=config.VIDEO_PROVENANCE_CSV,
                    )
                else:
                    result = process_video_upload(
                        input_path, out_video, out_csv,
                        _get_video_detector(), line_start, line_end,
                        start_s=start_s,
                        end_s=end_s if end_s > start_s else None,
                        frame_stride=stride,
                        inbound_label=inbound_label,
                        outbound_label=outbound_label,
                        stop_flag=st.session_state.get(_KEY_STOP),
                        progress_callback=_progress,
                        source_note=source_note,
                    )
        except VideoProcessingError as exc:
            progress_bar.empty()
            fps_display.empty()
            st.error(str(exc))
            return
        except Exception as exc:
            progress_bar.empty()
            fps_display.empty()
            st.error(
                f"An unexpected error occurred during processing: {exc}. "
                "Check the video file and try again."
            )
            return

        progress_bar.progress(1.0, text="Done")
        fps_display.empty()

        video_bytes = out_video.read_bytes()
        csv_bytes = out_csv.read_bytes()

    # Cache result
    cache[ck] = (result, video_bytes, csv_bytes)
    st.session_state[_KEY_CACHE] = cache

    enc = getattr(result, "_encoding_used", "mp4v")
    _render_results(result, video_bytes, csv_bytes, enc)


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render() -> None:
    """Render the full Video tab."""
    st.markdown("### Video vehicle counting")
    st.markdown(
        "**Video results are a demonstration, not an evaluation, and carry no accuracy claim.** "
        "Counts depend on the detector, the viewing angle, and the tracker, "
        "and identity switches or fragments can cause double counts or misses."
    )
    st.caption(
        "Results come from a pilot on Dhaka imagery. "
        "The fine-tuned mode is not used for video. "
        "No percentage is implied."
    )

    # ------------------------------------------------------------------ #
    # Sample clips section                                                 #
    # ------------------------------------------------------------------ #
    sample_clips = list_sample_clips()
    if sample_clips:
        st.markdown("#### Sample clips")
        st.caption(
            "These clips have a provenance row in data/samples/video/PROVENANCE.csv."
        )
        options = ["— select a sample —"] + [c["filename"] for c in sample_clips]
        chosen = st.selectbox(
            "Sample clips",
            options,
            key="video_sample_select",
            help="Select a clip to use it; provenance is shown below.",
        )
        if chosen != "— select a sample —":
            clip = next(c for c in sample_clips if c["filename"] == chosen)
            st.caption(
                f"**Attribution:** {clip['author']} | "
                f"**Licence:** {clip['licence_text']} | "
                f"**Source:** {clip['source_url']}"
            )
            clip_path = Path(clip["path"])
            try:
                meta = validate_video_file(
                    clip_path,
                    max_bytes=config.MAX_VIDEO_UPLOAD_BYTES,
                    max_seconds=9999,
                )
            except VideoProcessingError as exc:
                st.error(str(exc))
            else:
                _render_video_meta(meta)
                frame = _first_readable_frame(clip_path)
                if frame is not None:
                    line_start, line_end, inbound_lbl, outbound_lbl = _line_controls(
                        meta["width"], meta["height"]
                    )
                    _render_preview(frame, line_start, line_end, inbound_lbl, outbound_lbl)
                    start_s, end_s, stride = _processing_controls(meta["duration_s"])
                    _process_and_show(
                        clip_path,
                        content_hash=hashlib.sha256(clip_path.read_bytes()).hexdigest(),
                        source_note=clip["licence_text"],
                        meta=meta,
                        line_start=line_start,
                        line_end=line_end,
                        start_s=start_s,
                        end_s=end_s,
                        stride=stride,
                        inbound_label=inbound_lbl,
                        outbound_label=outbound_lbl,
                        for_sample=True,
                    )
            st.divider()

    # ------------------------------------------------------------------ #
    # Upload section                                                       #
    # ------------------------------------------------------------------ #
    st.markdown("#### Upload a video")
    st.info(
        "Your video is processed on this computer and is not kept after processing. "
        "You are responsible for having the right to use it."
    )

    limit_mb = config.MAX_VIDEO_UPLOAD_BYTES // (1024 * 1024)
    uploaded = st.file_uploader(
        "Upload a video file",
        type=_VIDEO_EXTS,
        key="video_upload",
        help=f"Accepted: {', '.join(e.upper() for e in _VIDEO_EXTS)}. Up to {limit_mb} MB.",
    )

    if uploaded is None:
        st.info("Upload a video or select a sample clip above to configure a counting line.")
        return

    source_note = st.text_input(
        "Source or licence note (optional)",
        key="video_source_note",
        help="Stored in the run record and exports. For example: 'Own footage, CC0' or a URL.",
    )

    # Detect when a new file is uploaded → clear old cache
    file_data = uploaded.getvalue()
    content_hash = _file_hash(file_data)
    if st.session_state.get(_KEY_UPLOAD_HASH) != content_hash:
        st.session_state[_KEY_UPLOAD_HASH] = content_hash
        st.session_state.pop(_KEY_CACHE, None)

    # Write to temp file for validation and processing
    with tempfile.TemporaryDirectory(prefix="trafficlens_vid_") as tmpdir:
        input_path = Path(tmpdir) / Path(uploaded.name).name
        input_path.write_bytes(file_data)
        del file_data  # release memory

        try:
            meta = validate_video_file(
                input_path,
                max_bytes=config.MAX_VIDEO_UPLOAD_BYTES,
                max_seconds=9999,
            )
        except VideoProcessingError as exc:
            st.error(str(exc))
            return

        _render_video_meta(meta)

        frame = _first_readable_frame(input_path)
        if frame is None:
            st.error(
                f"Could not read the first frame of '{uploaded.name}'. "
                "The file may be corrupt or use a codec OpenCV cannot decode."
            )
            return

        line_start, line_end, inbound_lbl, outbound_lbl = _line_controls(
            meta["width"], meta["height"]
        )
        _render_preview(frame, line_start, line_end, inbound_lbl, outbound_lbl)

        start_s, end_s, stride = _processing_controls(meta["duration_s"])

        _process_and_show(
            input_path,
            content_hash=content_hash,
            source_note=source_note,
            meta=meta,
            line_start=line_start,
            line_end=line_end,
            start_s=start_s,
            end_s=end_s,
            stride=stride,
            inbound_label=inbound_lbl,
            outbound_label=outbound_lbl,
            for_sample=False,
        )


def _render_video_meta(meta: dict) -> None:
    fps_note = " (assumed — container did not report a valid rate)" if meta["assumed_fps"] else ""
    duration_note = (
        f"{meta['duration_s']:.1f} s"
        if meta["duration_s"] is not None
        else "unknown"
    )
    st.caption(
        f"Resolution: {meta['width']} × {meta['height']} px | "
        f"Frame rate: {meta['fps']:.2f} fps{fps_note} | "
        f"Duration: {duration_note}"
        + (
            f" | Frames (container header): {meta['frame_count']}"
            if meta["frame_count"] is not None
            else ""
        )
    )
