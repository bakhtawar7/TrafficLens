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
