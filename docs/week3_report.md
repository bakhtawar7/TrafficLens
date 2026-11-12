# Week 3 report: evaluation readiness and provisional findings

## Executive summary

Week 3 set out to evaluate standard and sliced vehicle counting against audited
labels, test whether reliability features predict counting error, examine confusion
on local vehicle types, and prepare failure cases for human review. The repository
contains the evaluation and predictor implementations, a readiness-gated runner,
the benchmark cache, and a provisional source-annotation H3 analysis. It does **not**
contain a completed final-settings evaluation or a stored predictor report.

The current label file has first-pass counts for the development subset only. The
evaluation table has no data rows, and its stored summary reports no scored items.
Consequently, there is no defensible evaluation MAE, signed bias, H1/H2 verdict,
H4 paired comparison, or needs-review performance to report. The source-annotation
H3 analysis is available, but its annotations are explicitly unverified and it is
not confirmatory evidence. No failure-case report or screenshots are present.

This report describes artifacts as they currently exist; it does not rerun
inference, evaluation, or tests.

## What Week 3 set out to do and what was achieved

The plan was to use the frozen final settings to compare standard and sliced
inference, estimate uncertainty by resampling whole scenes, evaluate a scene-grouped
error predictor, inspect local vehicle confusions, and prepare a reviewable failure
gallery. A runner exists at `scripts/run_week3.py`; it checks label coverage and the
test-set checker before running its stages. The completed code also includes
evaluation, prediction, H3 analysis, and synthetic test infrastructure.

The planned end-to-end result was not achieved: the evaluation has no scored
evaluation images, and the predictor and Week 3 summary artifacts do not exist.
The benchmark cache is not complete for all three registered models at the frozen
settings, so it cannot substitute for the requested full evaluation. Implementation
availability is not evidence that the planned analysis ran successfully.

## Evaluation and label consistency

The source list and frozen plan describe an image set split into development and
evaluation subsets. The current label file has first-pass rows for development
images, but none for the evaluation subset. The stored `results/summary.json`
reports zero evaluated rows and null MAE/bias values; `results/evaluation_table.csv`
contains only its header. Thus standard/sliced results by model or condition,
including scene-level intervals and exploratory-condition flags, are unavailable.
No condition can be responsibly labelled exploratory from observed evaluation
counts because there are no scored condition rows.

The current labels contain no second-pass rows. Inter-pass category differences,
covered-total differences, exact-match share, and a scene-bootstrap interval are
therefore not estimable. Labeller consistency cannot be used as an empirical
measurement floor yet; until it is measured, model errors smaller than that
unmeasured disagreement cannot be distinguished from label noise.

## Hypotheses

Verdicts below use only stored evidence. “Inconclusive” means the required
evaluation evidence is absent; it is not a negative result.

| Hypothesis | Pre-registered decision rule | Available evidence | Verdict |
|---|---|---|---|
| H1: learned error ranking | Support only when ridge Spearman exceeds mean-confidence Spearman by at least 0.1, the scene-bootstrap interval for the difference excludes zero, and ridge captures more worst-error cases at the same fixed review budget. | No predictor report and no scored evaluation rows. The comparison and interval are unavailable. | **Inconclusive** |
| H2: cross-family disagreement | Support only when cross-family disagreement has a higher held-out Spearman correlation with error than within-family disagreement and the scene-bootstrap interval for the difference excludes zero. | No predictor report and no scored evaluation rows. Neither correlation nor their difference can be verified. | **Inconclusive** |
| H3: local vehicle confusion | The claim concerns systematic misses or absorption of local vehicle types; the documented rejection rule asks whether recall is comparable with covered classes and the miss-rate gap is small and inconsistent across scenes. | A provisional box-level analysis exists, but it compares detections to unverified source annotations. The available sample and rates are listed below. No human audit-sheet classes were entered, and the documented rejection rule does not set a numerical gap threshold. | **Inconclusive** |
| H4: distant-scene slicing | Support requires lower absolute counting error on distant images for sliced inference at the same threshold; the preregistered description also calls out small-object recall and double counting. | No audited evaluation rows or final evaluation summary. The paired distant-scene error difference and interval are unavailable. | **Inconclusive** |

No difference is described as an improvement: there is no evaluation interval that
excludes zero.

## Local vehicle confusion: provisional, annotation-referenced

The existing H3 output records fresh local inference on the source archive, not the
test-set benchmark cache. Its metadata labels the analysis “annotation referenced,
unverified” and “exploratory.” It records 49 analyzed source images, 217 annotated
boxes, and 651 model-box outcomes. The rickshaw category has 128 boxes across 15
scenes; loader-rickshaw has 6 boxes across 5 scenes and is explicitly exploratory
because its category sample is small. The 30 selected audit crops have a
verification sheet, but the `verified_class` cells are blank. Therefore **none of
these annotations has been verified through that sample**.

| Source category | Model | Boxes / scenes | Misses at default threshold | Miss rate (95% scene bootstrap interval) |
|---|---|---:|---:|---:|
| Rickshaw | YOLOv8n | 128 / 15 | 103 | 80.5% (72.0%–87.3%) |
| Rickshaw | YOLOv8s | 128 / 15 | 93 | 72.7% (63.5%–82.0%) |
| Rickshaw | RT-DETR-L | 128 / 15 | 51 | 39.8% (30.3%–48.9%) |
| Loader-rickshaw | YOLOv8n | 6 / 5 | 3 | 50.0% (16.7%–83.3%), exploratory |
| Loader-rickshaw | YOLOv8s | 6 / 5 | 4 | 66.7% (37.5%–100.0%), exploratory |
| Loader-rickshaw | RT-DETR-L | 6 / 5 | 1 | 16.7% (0.0%–60.0%), exploratory |

These are descriptive comparisons with source boxes, not error rates against
verified ground truth. They must not be used to select a setting or establish
confirmatory H3 support.

## Needs-review flag

The predictor implementation and its pre-registered review rules are documented in
the repository. However, `results/predictor_report.json` is absent, and no stored
H1/H2 verdict, validation precision/recall, or interval is available. The flag has
not been demonstrated to predict error in the current evaluation artifacts.
Accordingly, no flag performance or validated status is claimed here.

## Failure cases

`docs/failure_cases.md` and a failure gallery are absent. There are no verified
ranked cases from which to name main causes or provide examples. In particular,
occlusion, distance, night/glare, local vehicle types, dense overlap, and possible
annotation or labelling errors must not be assigned as causes without reviewing
the corresponding images. No case is claimed to show that the user miscounted.

## Unfinished and provisional work

* Evaluation and predictor reports are absent; final standard/sliced performance
  and hypothesis decisions remain unfinished.
* Evaluation first-pass labels and the requested second-pass labels are not present
  for the evaluation images. The checker and Week 3 runner must pass after labels
  are completed before analysis is run.
* H3 remains provisional until source boxes are checked; its verification sheet
  currently records no verified classes.
* Failure-case ranking, image review, cause assignments, and user confirmation are
  unfinished.
* Video counting is implemented with a provenance gate and synthetic tests, but it
  is not part of the image evaluation. No licensed clip run or video accuracy claim
  is verified here.
* The persisted pytest log is evidence only of the run it records; it does not
  establish the status of every later repository change.

## Limitations

The test set is small and comes from Dhaka road imagery; results cannot be assumed
to transfer to other places, camera positions, weather, or traffic patterns.
Reference counts depend on human labelling and are not perfect truth. The H3 source
annotations are separately unverified. All detectors are pretrained models rather
than models trained or validated for every local vehicle category. Rickshaws and
other local vehicle types may not map to a COCO detector class. Occlusion, overlap,
small or distant vehicles, night conditions, and glare can affect counts. No
accuracy percentage is supported by the current evaluation artifacts.

## Week 4 plan and mentor questions

1. Complete audited first-pass counts for the evaluation images and the planned
   second-pass recounts; preserve the blind review process and run the test-set
   checker.
2. Verify a sample of the source boxes and record true classes in the audit sheet;
   rerun H3 only with those explicit overrides and keep any unverified results
   clearly provisional.
3. Run the final-settings standard and sliced benchmark and evaluation only after
   readiness passes; then run the predictor, inspect the failure gallery blind,
   and confirm or reject each proposed cause before publication.
4. Report scene-level intervals, small-condition exploratory status, labeller
   disagreement, and the measured flag metrics without converting uncertainty into
   an accuracy claim.

**Questions for the mentor**

1. Is the planned evaluation-labeling and second-pass workload sufficient for the
   intended claims, or should the sample design be expanded before drawing
   conclusions?
2. What level of independent verification is required before source-annotation H3
   results for rickshaws and loader-rickshaws can be treated as evidence?
3. Should Week 4 prioritize completing the audited evaluation and H1/H2/H4
   decisions, or first broaden the sampled scenes to improve coverage of rare
   conditions and local vehicle types?

## Number inventory and sources

This inventory identifies the source for every quantitative result stated above.
Percentages and intervals in the H3 table are the rounded values stored in the H3
aggregate CSV. Missing results are described as unavailable rather than replaced
with estimates.

| Number(s) used | Meaning | Source file |
|---|---|---|
| 75; 10; 65 | Total source images; development subset; remaining evaluation images | `data/testset/SOURCES.csv`; `configs/testset_plan.json` |
| 0 evaluation rows; 0 labelled outside development; null MAE/bias | Stored final evaluation summary values | `results/summary.json` |
| 0 evaluation-table data rows | Evaluation table has a header only | `results/evaluation_table.csv` |
| 0 second-pass rows; 10 first-pass label rows | Current label-file pass counts | `data/testset/LABELS.csv` |
| 49; 217; 651 | H3 source images, annotation boxes, and model-box outcomes | `results/h3_confusion_metadata.json` |
| 30 selected audit boxes; 0 verified classes | Audit sample and blank verification sheet | `results/h3_confusion_metadata.json`; `results/h3_box_audit/verification_sheet.csv` |
| H3 category counts, scene counts, miss counts, rates, and intervals in the table above | Provisional per-category/model outcomes | `results/h3_confusion.csv` |
| 161 passed; 4 skipped | Persisted test-log result; not independently rerun for this report | `results/pytest_audit.log` |
| 0 predictor reports; 0 failure-case reports | Those artifacts are not present in the inspected `results/` and `docs/` directories | Repository artifact inventory |
| 15 images | Documented minimum condition sample size below which a condition is exploratory | `core/evaluation.py` (`EXPLORATORY_MIN_IMAGES`); `results/summary.json` |
| 0.1; 20% | H1 minimum Spearman margin and fixed worst-case review budget | `docs/hypotheses.md`; `core/predictor.py` |
| 0.5 IoU; 0.15; 0.25 | H3 box-match threshold; low-confidence sweep floor; default reporting threshold | `core/confusion.py`; `docs/metrics.md`; `results/h3_confusion_metadata.json` |

## Suggested screenshots to capture

No screenshots of completed evaluation or review screens were found. Existing H3
charts and audit crops are analysis artifacts, not screenshots of those screens.
The following are capture recommendations, not screenshots already produced.

| Suggested filename | What it should show |
|---|---|
| `week3_evaluation_tab.png` | Evaluation tab after a readiness-passing final evaluation, including sample size and condition results |
| `week3_predictor_validation.png` | Stored H1/H2 verdicts, scene-grouped validation, and needs-review precision/recall with intervals |
| `week3_h3_verification_sheet.png` | The H3 audit sheet with the verified-class entries visible and the annotation caveat |
| `week3_failure_gallery.png` | Human-reviewed failure gallery with dataset and model boxes clearly distinguished |
| `week3_testset_readiness.png` | Successful test-set checker output after required audited labels are complete |

## Could not verify

* A completed final evaluation on audited evaluation labels, including model-by-
  condition MAE and signed bias with scene-level intervals.
* Any paired-pass consistency estimate or its labelling-noise floor.
* Predictor validation, H1/H2 verdicts, needs-review precision/recall, and their
  intervals.
* A confirmatory H3 decision; all 30 selected box-audit entries remain unverified.
* A paired distant-condition H4 effect and interval.
* Ranked failure cases, image-viewed causes, possible user miscounts, or human
  confirmation of those judgments.
* Screenshots of completed evaluation or review screens.
* Whether the persisted test log covers the current repository revision; the log
  reports a test result but is not a fresh run for this report.
