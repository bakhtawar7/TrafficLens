# Week 3 Report: Pilot Evaluation and Remaining Work

## Executive summary

Week 3 aimed to compare pretrained vehicle detectors against audited counts, test whether reliability signals can identify cases for human review, examine local vehicle types, and prepare failure cases. The full evaluation was not run. Instead, a separate pilot was run because only 13 evaluation images had complete audited first-pass counts, below the 30-image minimum for the planned H1/H2 analysis. The pilot also found incomplete frozen-settings cache coverage for RT-DETR-L. It used existing cache entries only: it did not run inference, tune settings, or run the predictor.

The pilot provides preliminary error summaries for the models and descriptive analyses for H3, H4, and a needs-review flag. It does not establish a validated predictor, a complete three-model comparison, or a final evaluation. The persistent `results/summary.json` still records zero evaluated rows and null metrics; it is a stale final-evaluation artifact, not the source of the pilot results below. The pilot results are in `results/week3_summary.md`.

## Purpose and what changed

The planned Week 3 workflow was a frozen-settings evaluation on audited labels, with scene-level uncertainty intervals; reliability-predictor analyses; a local-vehicle analysis; and reviewable failure cases. The repository has an end-to-end runner, but its full analysis requires substantially more audited evaluation labels than were available.

The decision changed to a pilot rather than a full evaluation. The audited sample contained 13 evaluation first passes, while H1 and H2 require at least 30 evaluation images. In addition, the existing frozen-settings cache had complete standard results from RT-DETR-L for only two of those 13 images; YOLOv8n and YOLOv8s each had results for all 13. The pilot runner therefore summarized available cache entries, displayed the per-model sample sizes, and did not pool development images with evaluation images. No predictor was run and no settings were selected or adjusted from the pilot.

The pilot report lists 10 development images separately because settings were tuned on that subset. Its development results are explicitly optimistic and are not combined with the evaluation results. The separate failure-case review selected nine unique evaluation images from the pilot data; it did not use them to tune anything.

## Labelling and reference-count reliability

There are 23 labelled images with first-pass counts: 10 development images and 13 evaluation images. The 13 evaluation images also have second-pass counts. The mean absolute difference between the two passes on the covered total (car, bus, truck, and motorcycle) was **1.6923 vehicles**, with a scene-level bootstrap 95% interval of **0.67 to 2.64** across the 13 double-labelled evaluation images. This is the observed labeller-consistency figure for this pilot. Model errors below this level cannot be distinguished from labelling noise.

These are reference counts, not perfect ground truth. The interval quantifies variation in the observed pass differences under scene resampling; it does not eliminate systematic labelling error.

## Pilot error results

The pre-registered reference for the main pilot results is first-pass counts. Mean absolute error is the average size of the difference between a model's covered total and the reference. Mean signed error is model total minus reference: a negative value indicates undercounting on average and a positive value indicates overcounting. Bracketed ranges are scene-level bootstrap 95% intervals. The small RT-DETR-L sample makes its point estimates especially uncertain and not directly comparable to the other models.

| Model, frozen standard mode | Evaluation images | Scenes | Mean absolute error (95% interval) | Mean signed error (95% interval) |
|---|---:|---:|---:|---:|
| YOLOv8n | 13 | 7 | 4.00 (2.70 to 5.31) | -3.6923 (-5.31 to -1.70) |
| YOLOv8s | 13 | 7 | 3.00 (1.40 to 4.50) | -2.5385 (-4.31 to -0.50) |
| RT-DETR-L | 2 | 1 | 9.50 (9.50 to 9.50) | 2.50 (2.50 to 2.50) |

The RT-DETR-L intervals are based on only two images in one scene. Its cache is absent for the other labelled evaluation images, so the estimate is not a full-sample result. For the other two models, the pilot found more undercounted than overcounted or exact images; this remains a small, pilot-only sample.

Second-pass sensitivity analysis reused the 13 double-labelled evaluation images where the relevant cached model result existed:

| Model | Images | Mean absolute error (95% interval) | Mean signed error (95% interval) |
|---|---:|---:|---:|
| YOLOv8n | 13 | 3.3846 (2.11 to 5.00) | -2.9231 (-4.89 to -1.07) |
| YOLOv8s | 13 | 3.00 (1.67 to 4.67) | -1.7692 (-3.70 to 0.55) |
| RT-DETR-L | 2 | 6.00 (6.00 to 6.00) | 6.00 (6.00 to 6.00) |

Development results are shown separately and are optimistic because settings were tuned on those images:

| Model, development set | Images | Scenes | Mean absolute error (95% interval) | Mean signed error (95% interval) |
|---|---:|---:|---:|---:|
| YOLOv8n | 10 | 8 | 5.20 (1.50 to 9.55) | -5.00 (-9.46 to -1.00) |
| YOLOv8s | 10 | 8 | 3.50 (1.38 to 5.77) | -2.70 (-5.36 to 0.12) |
| RT-DETR-L | 4 | 3 | 4.00 (1.00 to 9.00) | 3.00 (1.00 to 9.00) |

These are pilot summaries, not final rankings. Cache coverage differs by model; the estimates should not be read as if all models were measured on the same set of evaluation images.

## Hypotheses

| Hypothesis | Status | Pilot evidence and reason |
|---|---|---|
| H1: learned error predictor versus mean-confidence baseline | **Not tested** | There are 13 audited evaluation images, below the 30-image minimum. The predictor was not run. |
| H2: cross-family disagreement versus within-family disagreement | **Not tested** | There are 13 audited evaluation images, below the 30-image minimum. The predictor and registered H2 analysis were not run. |
| H3: local vehicle types and counting error | **Descriptive only** | Pilot signed covered error was grouped by the first-pass rickshaw count. These groups do not measure category-level confusion or prove that a rickshaw was missed. |
| H4: sliced inference on distant scenes | **Descriptive only** | Seven labelled evaluation images were tagged distant, and all seven had cached standard and sliced results for the configured sliced model, YOLOv8n. This is a small pilot comparison, not a full confirmatory test. |

No H1 or H2 verdict is available. No hypothesis is described as established by this pilot.

### Local vehicles and H3

For images where the labeller counted three or more rickshaws, the mean signed covered error was -3.25 for YOLOv8n (4 images), -1.00 for YOLOv8s (4 images), and +12.00 for RT-DETR-L (1 image with a matching cache entry). For the zero-rickshaw group, the corresponding means were -5.67 (3 images), -6.00 (3 images), and -7.00 (1 image). For the one-or-two-rickshaw group, they were -3.00 (6 images), -1.83 (6 images), and unavailable for RT-DETR-L (no matching cached images). These are descriptive covered-total errors grouped by human-counted rickshaw count. They are **not a confusion analysis**: the models have no rickshaw class, and the groups cannot show whether a particular rickshaw was missed or assigned another class.

A separate exploratory H3 box analysis exists in `results/h3_confusion.csv` and `results/h3_confusion_metadata.json`. It is annotation-referenced and explicitly marked unverified. Its metadata reports 49 source images, 217 annotated boxes, and 651 model-box outcomes; the audit sheet has 30 sampled boxes and no verified-class entries. Therefore, **there are no verified H3 box results** to report as confirmed evidence. That source-annotation analysis must not be used to select settings or treated as a verified confusion result.

### Distant-image comparison

For YOLOv8n on the seven labelled distant images, standard-mode mean absolute error was 5.57 and sliced-mode mean absolute error was 2.43. The paired standard-minus-sliced error difference was 3.14, with a scene-level interval from 2.00 to 5.00, across seven cached pairs. The interval is above zero, but this remains a descriptive pilot comparison on a small sample; it does not replace the planned full evaluation.

## Needs-review flag

The pilot's simple flag selects the quarter of images with the greatest cross-family disagreement, using only images with cached standard results for all three models. Only two images met that common-cache requirement, so the flag selected one and left one unflagged. Their mean absolute median-model errors were 1.00 for the flagged image and 13.00 for the unflagged image (one image in each group).

This is **unvalidated** and is only a descriptive association on two images; it is not evidence that disagreement predicts error and is not a guarantee of reliability. The learned predictor was not run. Validation requires a sufficiently large audited evaluation sample, complete matching frozen-settings detections, a held-out scene-based evaluation of the pre-registered flag rule, and reported flag precision and recall with intervals. Results from development images must not be used as held-out validation because settings were tuned there.

## Failure cases

The failure-case review selected eight images because the best eligible model's absolute covered error exceeded the labeller-consistency figure, up to a maximum of 12, and five images for the largest available-model disagreement, up to a maximum of five. Four appeared in both groups, leaving nine unique figures. The best eligible model was YOLOv8s, selected as the lowest-MAE model among those with at least 10 image-hash-matched evaluation results.

The most frequent visual judgments were **distant or small** (five selected images), **dense overlap** (three), and **night or glare** (one). These causes are unconfirmed judgments from viewing images, not verified explanations. For example, `tl0038` was described as a wide-road scene with small distant vehicles; `tl0007` as busy traffic with partly occluded foreground vehicles; and `tl0046` as a dim scene where contrast may be reduced. Each remains unconfirmed and should be checked by the labeller.

The report flags every selected case for a possible blind recount; this is a request to review, not a claim that the labeller miscounted. No source dataset boxes could be aligned byte-for-byte to the selected image copies, so no dataset boxes were drawn. Cached RT-DETR-L detections were unavailable for seven of the nine selected images; the figures mark these missing results rather than generating detections. See `docs/failure_cases.md` and `results/failures/`.

## Limitations, unfinished work, and Week 4 plan

The pilot covers 13 audited evaluation images from Dhaka imagery. The labelled condition counts are descriptive and overlap because an image may have multiple tags: day 8, dense 6, distant 7, glare 5, night 4, and sparse 5. Results may not transfer to other places, viewpoints, lighting, or traffic patterns. Human counts are reference labels, not perfect truth. The models are pretrained general-purpose detectors, not models trained and validated for every local vehicle category. Incomplete cache coverage limits direct comparisons, especially for RT-DETR-L.

The persistent `results/summary.json` and `results/evaluation_table.csv` do not contain the pilot results: the JSON currently records zero evaluated items and null error metrics, and the table is header-only. The pilot findings come from `results/week3_summary.md` and are intentionally provisional. A full evaluation, H1/H2 predictor validation, verified source-box H3 analysis, and independent confirmation of failure causes remain unfinished. The persisted test log records an earlier run; the current full-suite result is 368 passed and 5 skipped.

For Week 4:

1. Increase audited evaluation first-pass coverage to at least 30 images and complete the planned second-pass review, while keeping development data separate.
2. Ensure frozen-settings cache coverage is complete for all models and required modes before comparing them on a common sample.
3. Run the full evaluation and registered H1/H2 predictor only after readiness checks pass; report scene-level uncertainty and validation precision/recall intervals for any flag.
4. Verify source annotations through the H3 audit process before interpreting box-level local-vehicle outcomes.
5. Recheck failure cases blind, confirm or revise each cause, and keep the labels unchanged unless the labeller records a correction through the established process.

## Questions for the mentor

1. Is a minimum of 30 audited evaluation images adequate for the planned H1/H2 claims, or should the sample be expanded before those analyses?
2. What independent verification protocol should be required before source-annotation H3 results for rickshaws and loader-rickshaws are interpreted?
3. Should Week 4 prioritize complete common-cache coverage across models, or expand the evaluation sample across scenes and rare conditions first?

## Number inventory and sources

All quantitative values in this report are traceable to the sources below. Item identifiers in the failure-case examples and the per-case cause totals are from `docs/failure_cases.md`. “95% interval” means a scene-level bootstrap interval where the pilot report labels it as such.

| Numbers used | Meaning | Source |
|---|---|---|
| 13 evaluation first passes; 10 development first passes; 23 first-pass-labelled images; 13 second-pass images | Label coverage by split and pass | `results/week3_summary.md`; `data/testset/LABELS.csv`; `configs/testset_plan.json` |
| 30 | Minimum audited evaluation sample required by H1/H2 readiness | `results/week3_summary.md`; `scripts/run_week3.py` |
| 1.6923; 0.67–2.64; n=13 | Mean absolute covered difference between labelling passes, interval, and paired sample size | `results/week3_summary.md` |
| Evaluation table: model n, scenes, MAE, signed error, and intervals; all values in the evaluation results table above | Pilot standard-mode evaluation | `results/week3_summary.md` |
| Second-pass table: model n, MAE, signed error, and intervals; all values in the sensitivity table above | Pilot sensitivity analysis | `results/week3_summary.md` |
| Development table: model n, scenes, MAE, signed error, and intervals; all values in the development table above | Separate, optimistic development results | `results/week3_summary.md` |
| 0 evaluated; 0 labelled outside development; null MAE/bias; zero table data rows | Stale stored final-evaluation artifact, distinguished from the pilot | `results/summary.json`; `results/evaluation_table.csv` |
| Condition counts 8, 6, 7, 5, 4, 5 | Day, dense, distant, glare, night, sparse counts respectively | `results/week3_summary.md` |
| H3 rickshaw-group sample sizes and signed errors | Descriptive pilot grouping by human-counted rickshaw count | `results/week3_summary.md` |
| 7 distant images; 5.57 standard MAE; 2.43 sliced MAE; 3.14 difference; 2.00–5.00 interval; 7 pairs | H4 descriptive paired comparison | `results/week3_summary.md` |
| 2 common-cache images; 1 flagged and 1 unflagged; mean errors 1.00 and 13.00; group n=1 each | Unvalidated descriptive needs-review flag | `results/week3_summary.md` |
| Selection limits 12 and 5; 8 error-selected; 5 disagreement-selected; 4 overlaps; 9 unique; cause frequencies 5, 3, 1; best-model eligibility minimum 10 | Failure-case selection and cause counts | `docs/failure_cases.md` |
| 49 source images; 217 boxes; 651 model-box outcomes; 30 sampled audit boxes; 0 verified classes | Exploratory, unverified H3 box analysis | `results/h3_confusion_metadata.json`; `results/h3_box_audit/verification_sheet.csv` |
| 368 passed; 5 skipped | Current full test-suite result after the failure-case implementation | Full test-suite run recorded in the task execution |
| Screenshot filenames below | Proposed filenames, not measured results | Recommendations in this report |

## Screenshots to capture

No screenshots of completed UI validation or a verified H3 audit were produced for this report. These are capture recommendations:

| Suggested filename | What it should show |
|---|---|
| `week3_pilot_summary.png` | The pilot model table, its per-model sample sizes, and the separate optimistic development table |
| `week3_labeller_consistency.png` | The covered-total pass-difference estimate, its interval, and the labelling-noise caveat |
| `week3_needs_review_pilot.png` | The descriptive flag explanation, its two-image limitation, and the unvalidated caveat |
| `week3_failure_cases.png` | A failure figure with its audited count panel, available model overlays, and explicit missing-data notes |
| `week3_h3_verification.png` | The source-box verification sheet after verified classes have been entered; until then it should show the unverified state |

## Could not verify

- A complete final evaluation from `results/summary.json` or `results/evaluation_table.csv`; both remain empty for evaluation.
- RT-DETR-L error on the full labelled evaluation set, because matching cached results exist for only two images in the pilot.
- Any H1/H2 predictor verdict or validated needs-review precision/recall.
- Verified H3 source-box outcomes; the audit sheet has no verified-class entries.
- Whether any failure-case cause is correct or whether any audited count needs correction; the causes remain unconfirmed pending blind review.
- Transfer of these results beyond the Dhaka imagery and the small pilot sample.
