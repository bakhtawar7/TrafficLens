# Evaluation on audited evaluation-image labels

**Status: provisional audited-label subset, not a complete evaluation over all evaluation images.** The evaluation roster contains 65 evaluation images; only 13 have supplied first-pass audited counts, so 52 remain unlabelled. All metrics below use Pass 1 counts as reference and exclude the development subset.

Both arms used `configs/final_settings.json` (SHA-256 `0c73720e5ee97e9a83ef6a654440226f367f3ebcc47e17bcce12ce4a37ea670e`), per-model configured confidence, and 2000 scene-cluster bootstrap resamples (95% percentile intervals). Metrics are covered-count error (car, bus, truck, motorcycle); all inference outputs were validated for settings fingerprint, image hash, confidence threshold, and complete item/model/arm coverage.

Every condition shown has fewer than 15 labelled images and is **exploratory**. Conditions overlap when an image has multiple tags. Signed error is prediction minus Pass 1 covered total; negative means undercount. Intervals for MAE and signed error resample whole scenes.

| Arm | Model | Condition | n | MAE [scene 95% CI] | Mean signed error [scene 95% CI] | Under / over / exact | Mean inference (ms/image) |
|---|---|---:|---:|---:|---:|---:|---:|
| standard | YOLOv8n | Overall | 13 | 4.00 [2.70, 5.31] | -3.69 [-5.31, -1.70] | 10 / 1 / 2 | 167.65 |
| standard | YOLOv8n | dense (exploratory) | 6 | 5.83 [4.60, 6.62] | -5.83 [-6.62, -4.60] | 6 / 0 / 0 | 179.08 |
| standard | YOLOv8n | day (exploratory) | 8 | 3.88 [2.00, 5.20] | -3.38 [-5.20, -0.50] | 5 / 1 / 2 | 182.29 |
| standard | YOLOv8n | distant (exploratory) | 7 | 5.57 [3.55, 10.00] | -5.57 [-10.00, -3.55] | 7 / 0 / 0 | 165.33 |
| standard | YOLOv8n | glare (exploratory) | 5 | 3.40 [2.00, 6.00] | -2.60 [-6.00, 2.00] | 4 / 1 / 0 | 149.54 |
| standard | YOLOv8n | night (exploratory) | 4 | 3.75 [3.00, 6.00] | -3.75 [-6.00, -3.00] | 4 / 0 / 0 | 144.47 |
| standard | YOLOv8n | sparse (exploratory) | 5 | 2.20 [1.00, 6.00] | -1.40 [-6.00, 1.00] | 3 / 1 / 1 | 159.72 |
| standard | YOLOv8s | Overall | 13 | 3.00 [1.40, 4.50] | -2.54 [-4.31, -0.50] | 9 / 2 / 2 | 350.85 |
| standard | YOLOv8s | dense (exploratory) | 6 | 4.17 [1.60, 6.43] | -3.50 [-6.38, 0.00] | 5 / 1 / 0 | 420.27 |
| standard | YOLOv8s | day (exploratory) | 8 | 3.62 [1.57, 5.29] | -3.38 [-5.29, -0.88] | 5 / 1 / 2 | 397.37 |
| standard | YOLOv8s | distant (exploratory) | 7 | 3.57 [1.56, 8.60] | -3.00 [-8.00, -0.22] | 6 / 1 / 0 | 358.01 |
| standard | YOLOv8s | glare (exploratory) | 5 | 1.80 [1.00, 4.00] | -1.40 [-4.00, 1.00] | 4 / 1 / 0 | 263.78 |
| standard | YOLOv8s | night (exploratory) | 4 | 2.00 [1.33, 4.00] | -2.00 [-4.00, -1.33] | 4 / 0 / 0 | 259.84 |
| standard | YOLOv8s | sparse (exploratory) | 5 | 2.00 [0.50, 6.00] | -1.60 [-6.00, 0.50] | 3 / 1 / 1 | 297.84 |
| standard | RT-DETR-L | Overall | 13 | 7.08 [4.64, 10.45] | 5.54 [3.21, 9.90] | 3 / 10 / 0 | 2754.45 |
| standard | RT-DETR-L | dense (exploratory) | 6 | 11.00 [6.33, 15.60] | 8.67 [3.43, 15.60] | 1 / 5 / 0 | 3048.05 |
| standard | RT-DETR-L | day (exploratory) | 8 | 6.50 [2.67, 10.20] | 4.25 [0.83, 9.40] | 2 / 6 / 0 | 2475.51 |
| standard | RT-DETR-L | distant (exploratory) | 7 | 9.00 [5.56, 15.40] | 6.71 [0.60, 15.20] | 2 / 5 / 0 | 2953.89 |
| standard | RT-DETR-L | glare (exploratory) | 5 | 5.60 [4.67, 8.00] | 5.20 [4.00, 8.00] | 1 / 4 / 0 | 3224.43 |
| standard | RT-DETR-L | night (exploratory) | 4 | 5.50 [4.67, 8.00] | 5.00 [4.00, 8.00] | 1 / 3 / 0 | 3338.95 |
| standard | RT-DETR-L | sparse (exploratory) | 5 | 2.80 [1.00, 5.00] | 1.60 [-2.00, 5.00] | 2 / 3 / 0 | 2539.49 |
| sliced | YOLOv8n | Overall | 13 | 2.38 [1.17, 3.67] | -1.46 [-2.58, -0.60] | 8 / 3 / 2 | 2268.50 |
| sliced | YOLOv8n | dense (exploratory) | 6 | 3.33 [1.00, 5.14] | -2.00 [-3.17, -1.00] | 4 / 1 / 1 | 2813.82 |
| sliced | YOLOv8n | day (exploratory) | 8 | 3.25 [1.50, 4.43] | -1.75 [-4.20, -0.40] | 4 / 3 / 1 | 3237.01 |
| sliced | YOLOv8n | distant (exploratory) | 7 | 2.43 [1.00, 5.20] | -2.43 [-5.20, -1.00] | 6 / 0 / 1 | 1650.56 |
| sliced | YOLOv8n | glare (exploratory) | 5 | 1.00 [1.00, 1.00] | -0.60 [-1.00, 1.00] | 3 / 1 / 1 | 650.70 |
| sliced | YOLOv8n | night (exploratory) | 4 | 1.00 [1.00, 1.00] | -1.00 [-1.00, -1.00] | 3 / 0 / 1 | 786.90 |
| sliced | YOLOv8n | sparse (exploratory) | 5 | 1.80 [0.50, 5.00] | -1.40 [-5.00, 0.50] | 3 / 1 / 1 | 686.86 |
| sliced | YOLOv8s | Overall | 13 | 2.77 [1.33, 4.12] | 1.23 [-0.25, 2.69] | 4 / 8 / 1 | 5460.04 |
| sliced | YOLOv8s | dense (exploratory) | 6 | 3.17 [1.60, 4.38] | 1.83 [-0.20, 3.00] | 2 / 4 / 0 | 7223.92 |
| sliced | YOLOv8s | day (exploratory) | 8 | 3.00 [0.86, 4.73] | 1.00 [-2.20, 3.17] | 3 / 4 / 1 | 7945.60 |
| sliced | YOLOv8s | distant (exploratory) | 7 | 2.14 [1.60, 2.80] | 1.00 [-0.40, 2.60] | 2 / 5 / 0 | 3992.01 |
| sliced | YOLOv8s | glare (exploratory) | 5 | 2.00 [1.00, 3.00] | 1.20 [0.67, 3.00] | 1 / 4 / 0 | 1352.78 |
| sliced | YOLOv8s | night (exploratory) | 4 | 2.25 [2.00, 3.00] | 1.25 [0.67, 3.00] | 1 / 3 / 0 | 1630.03 |
| sliced | YOLOv8s | sparse (exploratory) | 5 | 1.60 [0.50, 4.00] | -0.80 [-4.00, 0.50] | 2 / 2 / 1 | 1473.21 |
| sliced | RT-DETR-L | Overall | 13 | 14.38 [5.55, 23.44] | 14.38 [5.55, 23.44] | 0 / 12 / 1 | 28012.85 |
| sliced | RT-DETR-L | dense (exploratory) | 6 | 17.67 [8.40, 26.50] | 17.67 [8.40, 26.50] | 0 / 6 / 0 | 33707.41 |
| sliced | RT-DETR-L | day (exploratory) | 8 | 18.12 [5.43, 28.09] | 18.12 [5.43, 28.09] | 0 / 8 / 0 | 40560.65 |
| sliced | RT-DETR-L | distant (exploratory) | 7 | 11.00 [6.18, 20.20] | 11.00 [6.18, 20.20] | 0 / 6 / 1 | 17649.08 |
| sliced | RT-DETR-L | glare (exploratory) | 5 | 6.00 [4.33, 12.00] | 6.00 [4.33, 12.00] | 0 / 4 / 1 | 7502.64 |
| sliced | RT-DETR-L | night (exploratory) | 4 | 6.25 [4.33, 12.00] | 6.25 [4.33, 12.00] | 0 / 3 / 1 | 8670.85 |
| sliced | RT-DETR-L | sparse (exploratory) | 5 | 8.20 [1.50, 30.00] | 8.20 [1.50, 30.00] | 0 / 4 / 1 | 8918.38 |

## H4: distant-condition paired sliced effect

Effect is sliced MAE minus standard MAE on the same distant-tagged images; negative favors sliced inference. Intervals bootstrap scenes and preserve pairing within each sampled image. A difference is called an improvement only when its interval excludes zero in the negative direction.

- **YOLOv8n** (n=7): paired MAE difference -3.14, 95% scene interval [-5.00, -2.00]. Improvement: interval excludes zero in the negative direction.
- **YOLOv8s** (n=7): paired MAE difference -1.43, 95% scene interval [-6.60, 0.57]. No established difference: interval includes zero.
- **RT-DETR-L** (n=7): paired MAE difference 2.00, 95% scene interval [-2.86, 10.14]. No established difference: interval includes zero.

## Audited labels and pass agreement

Pass 1 totals across 13 labelled evaluation images: **covered 110**; **all seven labelled categories 151**. By category: car 50, bus 24, truck 9, motorcycle 27, rickshaw 27, loader_rickshaw 0, other 14.
Pass 2 totals for the same images: covered 100; all seven categories 144. The evaluator's legacy all-vehicles helper excludes `other`; under that definition Pass 1 totals 137. The MAE table uses covered totals only.
Pass 1 vs Pass 2 agreement (13 paired items): mean absolute difference in covered totals 1.69 (95% scene interval [0.67, 2.64]); identical covered totals on 5/13 images. Across all seven count fields, 71/91 item-category cells match exactly; mean absolute difference in seven-category item totals 1.62 (95% scene interval [0.67, 2.46]).

## Caveats, surprises, and unverifiable items

- This is not a complete 65-image final evaluation: 52 evaluation images have no supplied Pass 1 audited count. Results represent only the 13 labelled images and should not be generalized to the entire selected evaluation set.
- The supplied audited Pass 1 and Pass 2 counts are present in `data/testset/LABELS.csv`; the evaluation pack retains a matching copy for this 13-image audited evaluation subset. The source label file also contains 10 development-subset Pass 1 rows, which are not included in the evaluation metrics.
- **The earlier draft figure of 134 undercounted vehicles came from unverified labels and must not be cited.**
- The evaluator's legacy all-vehicles helper excludes `other`; the seven-category audited total is also shown explicitly.
- The images directory contains 10 image(s) outside the selected audit roster; they were excluded by using the roster.
- Label correctness beyond the supplied human counts, usage rights, and ambiguous/occluded object decisions could not be independently verified.

## Provenance

- Settings: `configs/final_settings.json`; SHA-256 `0c73720e5ee97e9a83ef6a654440226f367f3ebcc47e17bcce12ce4a37ea670e`.
- Image/tag roster: `results/eval_labelling_pack/evaluation_list.csv`; scene metadata: `data/testset/SOURCES.csv`; development exclusion: `configs/testset_plan.json`.
- Bootstrap: scene-level percentile 95% intervals; seed inherited from `core.config.SEED`. Mean inference time excludes model loading and warmup.
