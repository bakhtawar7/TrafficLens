# Image input and counting test cases

These cases exercise the upload validation and image decode path, plus controlled
count aggregation on synthetic scenes. Vehicle-count scenarios use a stub detector
with known outputs: they verify input plumbing and class filtering, **not** detector
accuracy or visual recognition. No real dataset images or model weights are used.

| ID | Input | Expected behaviour | Actual result | Status |
|---|---|---|---|---|
| IMG-01 | Valid JPG bytes named `image.jpg` | Accept and decode to a three-channel `uint8` BGR image. | Decoded to `(32, 48, 3)` BGR. (`test_valid_jpg_png_and_webp_decode_to_bgr`) | PASS |
| IMG-02 | Valid PNG bytes named `image.png` | Accept and decode to a three-channel `uint8` BGR image. | Decoded to `(32, 48, 3)` BGR. (`test_valid_jpg_png_and_webp_decode_to_bgr`) | PASS |
| IMG-03 | Valid WEBP bytes named `image.webp` | Accept and decode to a three-channel `uint8` BGR image. | Decoded to `(32, 48, 3)` BGR. (`test_valid_jpg_png_and_webp_decode_to_bgr`) | PASS |
| IMG-04 | Grayscale PNG | Accept and expand to three-channel BGR. | Decoded to `(32, 48, 3)`; grayscale channels agree. (`test_grayscale_png_decodes_as_three_channel_bgr`) | PASS |
| IMG-05 | PNG with alpha channel | Accept and decode to three-channel BGR, dropping alpha for detector input. | Decoded to `(32, 48, 3)` BGR. (`test_png_with_alpha_decodes_as_three_channel_bgr`) | PASS |
| IMG-06 | JPEG with EXIF orientation 6 (rotate 90 degrees clockwise) | Apply EXIF orientation during decode; swap width and height. | Decoded shape is `(48, 32, 3)` from an encoded `(32, 48)` image. (`test_jpeg_exif_rotation_is_applied_during_decode`) | PASS |
| IMG-07 | Corrupt JPEG-like bytes with `.jpg` extension | Pass extension validation, then report a corrupt-image decode error. | `CorruptImageError` raised. (`test_corrupt_image_is_rejected_after_extension_validation`) | PASS |
| IMG-08 | Empty `.jpg` file | Reject before decode with the empty-file error. | `EmptyFileError` raised. (`test_empty_file_is_rejected`) | PASS |
| IMG-09 | Plain text bytes renamed to `.jpg` | Reject at decode; extension alone must not make text an image. | `CorruptImageError` raised. (`test_text_renamed_as_jpg_is_rejected`) | PASS |
| IMG-10 | PNG image bytes named with unsupported `.bmp` extension | Reject the unsupported extension. | `UnsupportedFileTypeError` raised. (`test_unsupported_image_format_is_rejected_even_with_image_bytes`) | PASS |
| IMG-11 | File of `MAX_UPLOAD_BYTES + 1` bytes | Reject before decoding because it exceeds the configured limit. | `FileTooLargeError` raised. (`test_file_over_configured_size_limit_is_rejected`) | PASS |
| IMG-12 | Very small 1 × 1 PNG | Decode and pass dimensions through count aggregation without crashing. | Count result dimensions are 1 × 1; total is zero with the empty stub detector. (`test_very_small_image_is_decoded_and_counted`) | PASS |
| IMG-13 | Large 4000 × 3000 JPEG | Decode and pass dimensions through count aggregation without crashing. | Count result dimensions are 4000 × 3000; total is zero with the empty stub detector. (`test_very_large_image_is_decoded_and_counted`) | PASS |
| IMG-14 | Blank synthetic scene, stub returns no vehicle detections | Return zero vehicles. | Total is zero. (`test_scene_vehicle_count_cases_use_stub_detections[0-0]`) | PASS |
| IMG-15 | Synthetic scene with one vehicle, stub returns one car detection | Return one vehicle. | Total is one. (`test_scene_vehicle_count_cases_use_stub_detections[1-1]`) | PASS |
| IMG-16 | Crowded synthetic scene, stub returns 20 car detections | Aggregate all 20 vehicle detections. | Total is 20. (`test_scene_vehicle_count_cases_use_stub_detections[20-20]`) | PASS |
| IMG-17 | Synthetic scene dominated by person/bicycle detections and one car | Exclude non-target classes; retain the car. | Total is one and only the car remains. (`test_non_target_dominated_scene_excludes_non_target_classes`) | PASS |

The valid, invalid, and edge cases are automated in
[`tests/test_image_cases.py`](../tests/test_image_cases.py); the limitations-panel
content is checked in [`tests/test_app.py`](../tests/test_app.py). All listed cases
passed in the focused run. See the project test command and full-suite outcome in the
change summary for the latest verification.

---

## Video upload and counting test cases

These cases exercise the video upload pipeline, validation, and crossing-count logic.
No real model weights are used; a `CrossingDetector` stub detects coloured rectangles,
and a `SyntheticTracker` assigns fixed IDs. No copyrighted video is used.
Video results are a demonstration; they are not part of any evaluation.

| ID | Input | Expected behaviour | Actual result | Status |
|---|---|---|---|---|
| VID-01 | Empty file (0 bytes) | `VideoProcessingError` with "empty" message. | Raised. (`test_empty_file_is_rejected`) | PASS |
| VID-02 | Corrupt bytes (not a video) | `VideoProcessingError` with codec/corrupt message. | Raised. (`test_corrupt_file_is_rejected`) | PASS |
| VID-03 | File exceeding `MAX_VIDEO_UPLOAD_BYTES` | `VideoProcessingError` with size message and ffmpeg trim hint. | Raised. (`test_oversized_file_is_rejected`) | PASS |
| VID-04 | Valid 160×120 synthetic MP4, 10 fps, 20 frames | Returns correct width, height, fps; `assumed_fps` False. | Correct metadata returned. (`test_valid_mp4_returns_correct_metadata`) | PASS |
| VID-05 | Container reports fps = 0 | Uses `VIDEO_DEFAULT_FPS`; sets `assumed_fps = True`. | Default used; flag set. (`test_zero_fps_uses_default_and_flags_assumed`) | PASS |
| VID-06 | Uploaded file processed successfully | Temp directory is deleted after processing. | Directory absent after `with` block. (`test_temp_file_deleted_after_successful_processing`) | PASS |
| VID-07 | Exception during processing | Temp directory is still deleted. | Directory absent after exception. (`test_temp_file_deleted_even_after_exception`) | PASS |
| VID-08 | Uploaded MP4, no provenance row present | `process_video_upload` succeeds without any PROVENANCE.csv. | No error; frames processed. (`test_upload_does_not_require_provenance_row`) | PASS |
| VID-09 | Synthetic video, one inbound green box, one outbound red box | `(inbound, outbound) == (1, 1)`. | Correct. (`test_upload_crossing_counts_match_synthetic_detector`) | PASS |
| VID-10 | Crossing CSV columns | CSV has track_id, frame, time_s, direction, class. | All columns present. (`test_upload_crossing_csv_has_extended_columns`) | PASS |
| VID-11 | Sample clip path with no provenance row | `process_video` raises `VideoProcessingError`. | Raised. (`test_sample_clip_still_requires_provenance`) | PASS |
| VID-12 | Synthetic AVI (MJPG) crossing video | Correct crossing counts. | `(1, 1)`. (`test_crossing_counts_in_multiple_formats[MJPG-.avi]`) | PASS |
| VID-13 | Stop flag set after 5 frames | `result.partial == True`; `frames_processed < 20`. | Partial result returned. (`test_stop_flag_produces_partial_result`) | PASS |
| VID-14 | `end_s = 0.5` on a 20-frame 10-fps clip | At most ~7 frames processed. | Correct. (`test_time_range_limits_frames_processed`) | PASS |
| VID-15 | `frame_stride = 4` versus stride 1 | Strided result processes fewer frames. | Fewer frames. (`test_frame_stride_reduces_frames_processed`) | PASS |
| VID-16 | Per-class crossing counts | `inbound_by_class["car"] == 1`, `outbound_by_class["motorcycle"] == 1`. | Correct. (`test_per_class_crossing_counts`) | PASS |
| VID-17 | Output annotated video | File is non-empty; cv2 can open and read a frame. | Readable. (`test_output_video_is_readable_after_processing`) | PASS |
| VID-18 | `source_note` passed to `process_video_upload` | `result.source_note` matches. | Correct. (`test_source_note_is_stored_in_result`) | PASS |
| VID-19 | `draw_line_on_frame` | Returns a copy with the line drawn; original unchanged. | Correct. (`test_draw_line_on_frame_returns_copy_with_line`) | PASS |
| VID-20 | `list_sample_clips` with no matching file | Returns empty list. | Correct. (`test_list_sample_clips_empty_when_provenance_has_no_matching_file`) | PASS |
| VID-21 | `list_sample_clips` with matching file | Returns one entry with correct filename. | Correct. (`test_list_sample_clips_returns_clip_when_file_exists`) | PASS |
| VID-22 | `sample_frame_detections` on 3 frames | Returns list of 3 non-negative integers. | Correct. (`test_sample_frame_detections_returns_counts_per_frame`) | PASS |
| VID-23 | Config constants | `ALLOWED_VIDEO_EXTENSIONS`, `MAX_VIDEO_UPLOAD_BYTES`, etc. present and correct. | Correct. (`test_config_has_video_constants`) | PASS |
| VID-24 | Streamlit Video tab renders without upload | No exception; honesty text in markdown. | Correct. (`test_video_tab_upload_renders_without_exception`) | PASS |

Video cases are in [`tests/test_video_upload.py`](../tests/test_video_upload.py) and
[`tests/test_video_counting.py`](../tests/test_video_counting.py).
