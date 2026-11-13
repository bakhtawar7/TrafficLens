# TrafficLens: Design, Implementation, and Pilot Evaluation Report

**Status:** documentation draft based on the repository's existing artefacts. The full test
suite, clean-environment run, and usability sessions have not been executed for this report.
Sections that require those activities are marked **pending**.

## Abstract

TrafficLens is a prototype vehicle-counting application for road-scene images. It combines
pretrained object detectors, image annotation, per-class count aggregation, model-to-model
reliability signals, offline evaluation against human reference counts, local run history,
and a blind-first human correction workflow. It also contains a separate optional
fine-tuned one-class rickshaw detector and a provenance-gated video line-crossing prototype.

The available evidence is a pilot on Dhaka imagery. The selected image set contains a
development partition and a larger evaluation partition, but only a small subset of
evaluation images has audited human labels. The reported human-pass consistency is a mean
absolute total-count difference of 1.62 vehicles across 13 paired images in 7 scenes
(95% scene-bootstrap interval 0.67–2.46). The general pretrained evaluation artefacts report
covered-class errors on 13 labelled evaluation images. The fine-tuned detector is evaluated
separately because of training/test overlap; its eligible first-pass audited comparison is
smaller still. These results are descriptive pilot evidence, not a deployment validation.

The project keeps the hypotheses' registered evidence labels: H1 and H2 are **not tested**;
H3 and H4 are **descriptive only**. Reliability agreement is not correctness. The video
prototype has not been evaluated against labelled video ground truth. Automated tests and
usability findings are pending fresh execution.

## 1. Problem and objectives

Manual vehicle counting from road imagery is time-consuming and can be inconsistent,
especially in dense traffic, under occlusion, at night, and where local vehicle types do not
map cleanly to standard object-detection categories. A detector can produce a count quickly,
but a single number without context can hide missed vehicles, duplicate detections, or
systematic class mismatch.

TrafficLens addresses this as a decision-support prototype. Its objectives are:

1. Accept a road-scene image and validate/decode it safely.
2. Run a pretrained detector and report selected vehicle counts with visual annotations.
3. Compare multiple pretrained detectors and expose their count range and agreement.
4. Evaluate cached model outputs against human reference counts without using the development
   subset as evaluation evidence.
5. Keep run metadata locally and enable blind-first human review with append-only corrections.
6. Explore a separate one-class fine-tuned rickshaw detector without conflating it with
   pretrained multi-class results.
7. Explore video vehicle crossing counts while explicitly documenting tracking limitations.

The application does not control traffic signals, enforce laws, identify drivers, or establish
a legally authoritative vehicle count.

## 2. Requirements and scope

### 2.1 Functional scope

The current application includes image upload, validation, counting, annotated output,
JSON/CSV export, three pretrained detector registry entries, sliced inference, model
agreement metrics, an offline evaluation pipeline, SQLite history, and a human correction
loop. The Image tab also includes optional polygon-zone analysis and a separate fine-tuned
rickshaw toggle. The Video tab uses provenance-gated uploads and a virtual counting line.

### 2.2 Scope change

The project began as a pretrained image counter. It was extended with an optional, off-by-
default, one-class fine-tuned rickshaw detector. The frozen pretrained outputs and reliability
comparisons remain separate. Mentor approval for the scope change is recorded as not yet
obtained. The fine-tuned detector does not detect cars, buses, trucks, or motorcycles and
cannot distinguish loader rickshaws.

### 2.3 Non-functional and evidence requirements

The system should preserve configuration provenance, avoid storing uploaded image bytes in
history, keep inference usable when history persistence fails, and disclose limitations.
The project safeguards prohibit modifying frozen labels, settings, class mappings, cut-offs,
and cached detections during documentation work. Evaluation hypotheses must not be upgraded
beyond their registered evidence status.

## 3. System design

### 3.1 Image workflow

The Streamlit entry point builds the sidebar and tab shell. Image bytes pass through
extension/size validation and OpenCV decoding into a BGR image. A detector wrapper produces
class-labelled bounding boxes; counting filters selected classes and aggregates totals.
The UI displays original and annotated images, per-class metrics, inference timing, and
downloadable JSON/CSV output.

The pretrained registry currently contains YOLOv8n, YOLOv8s, and RT-DETR-L. The detector
protocol allows the counting pipeline to use different model implementations without
changing the result contract. Sliced inference runs overlapping image tiles and merges
detections using class-aware non-maximum suppression.

### 3.2 Reliability workflow

The Reliability tab compares pretrained models on the same input. It reports count range,
normalised per-class count disagreement, and class-aware box agreement. These values describe
how much models agree with one another. They do not establish that any model is correct.
The needs-review status is validation-gated; when the required validated configuration is
unavailable, the interface must retain the “not yet validated” state.

### 3.3 Evaluation workflow

The offline evaluation pipeline joins the selected image roster, human labels, and cached
detections. It excludes the development subset and records settings provenance. It reports
mean absolute error (MAE), signed error, under/over/exact tallies, inference timing, and
scene-cluster bootstrap intervals. Conditions with fewer than 15 labelled evaluation images
are marked exploratory.

### 3.4 History and human review

SQLite stores run metadata, model results, reliability metadata, and corrections. Each
correction save inserts a new row rather than editing an earlier correction; however, this
is not an immutable audit log because deleting a run or clearing local history cascades to
its dependent corrections. Foreign keys and indexes support dependent-row cleanup and
lookups. Image bytes are not stored. A run is deduplicated by image hash, settings hash,
selected classes, threshold, rickshaw mode, and inference mode.

The review queue is blind-first. A reviewer must re-upload the exact image with a matching
SHA-256, enter seven category counts, and then reveal model results. Corrections are
user-reported records; they do not update `LABELS.csv`, model weights, or inference settings.

### 3.5 Separate experimental paths

The optional fine-tuned rickshaw model is a one-class detector with its own settings,
evaluation, and contamination exclusions. It is not included in pretrained reliability
comparisons. The video prototype uses a local pretrained detector, ByteTrack, and a directed
virtual line to report inbound/outbound crossings. Track identity switches and duplicate IDs
under occlusion can change crossing totals; video has not been evaluated.

## 4. Data and labelling

### 4.1 Image set

The repository records a 75-image Dhaka road-scene test set with 65 evaluation items and
10 development items. Selection is governed by the frozen seed, tag cut-offs, and plan.
Scene identifiers and condition tags are recorded in the source inventory. The selection
pipeline is designed to avoid using model detections to choose evaluation images.

### 4.2 Human labelling

The labelling protocol defines seven categories: car, bus, truck, motorcycle, rickshaw,
loader rickshaw, and other. It describes what constitutes a vehicle, partial visibility,
occlusion, a minimum visible size rule, parked vehicles, and second-pass adjudication.
The available audited evaluation pack contains 13 evaluation images with two human
passes. The current `data/testset/LABELS.csv` has 23 Pass 1 rows (10 development items
and 13 evaluation items) plus 13 Pass 2 rows for the same 13 evaluation items. The
`benchmark/audited_labels.csv` file mirrors those 26 evaluation-pass records. The broader
65-image evaluation roster still has 52 items without supplied first-pass audited counts.
Consequently, the general evaluation results represent the 13-image labelled evaluation
subset, not the entire roster.

### 4.3 Consistency

Across the 13 paired human-labelled images in 7 scenes, the mean absolute difference in
seven-category total counts between passes is 1.62 vehicles per image, with a 95% scene-
bootstrap interval of 0.67–2.46. For the four covered classes only, the mean absolute
difference is 1.69, with interval 0.67–2.64. The estimate describes pass-to-pass consistency,
not the correctness of the labels.

### 4.4 Licensing and provenance

The source inventory records the dataset licence as unverified. The project notes that the
Dhaka-AI licence is listed inconsistently across mirrors: CC0 1.0 on one page and GNU GPL 2.0
on another. This must be confirmed against the original source before redistribution. The
original source citation and URL are not established in the supplied provenance. The licence
for project-authored annotations remains a placeholder.

The fine-tuned detector's source archive also lacks a verified source URL, authorship, licence,
download date, and paper citation. Redistribution of that source data is not authorized by
the current documentation.

## 5. Evaluation method

The general pretrained evaluation uses the first human labelling pass as reference, frozen
final settings, per-model configured confidence, and 2,000 scene-cluster bootstrap resamples.
The evaluation roster is 65 items, but only 13 have supplied first-pass audited counts. All
condition-level subsets in the current report are small and explicitly exploratory.

The covered-class target is car + bus + truck + motorcycle. Rickshaw and loader-rickshaw are
outside the pretrained COCO classes; the all-seven human target is shown separately where
relevant. The evaluation report also records a legacy all-vehicles helper that excludes the
`other` category; this distinction must be retained when comparing totals.

The fine-tuned evaluation is separate. Its training selection was reproduced as 242 images.
Filename and perceptual-hash matching identified 49 overlaps with the test set (43 evaluation
and 6 development images); known overlaps are excluded from fine-tuned evaluation. The
uncontaminated evaluation pool contains 22 images, but only 7 images across 3 scenes have
eligible first-pass labels for the audited-count comparison. Development results are kept
separate and are explicitly optimistic because settings were tuned there.

## 6. Results

### 6.1 Pretrained detector pilot

The following table reports overall standard-mode covered-class MAE from the provisional
audited-label subset. It does not provide per-image outputs. The sample size is 13 images;
the evaluation roster has 65 images, so the result is incomplete.

| Model | MAE (vehicles/image) | 95% scene-bootstrap interval | Source |
| --- | ---: | ---: | --- |
| YOLOv8n | 4.00 | 2.70–5.31 | `results/eval_labelling_pack/final_evaluation_report.md` |
| YOLOv8s | 3.00 | 1.40–4.50 | `results/eval_labelling_pack/final_evaluation_report.md` |
| RT-DETR-L | 7.08 | 4.64–10.45 | `results/eval_labelling_pack/final_evaluation_report.md` |

These are pilot estimates on a small labelled subset. They should not be generalized to all
65 evaluation images or other cities and camera conditions.

### 6.2 Sliced inference pilot

For the same labelled subset, the report records overall standard-versus-sliced results.
Sliced inference reduced YOLOv8n's observed MAE in this sample, but this descriptive result
does not establish a general improvement across models or scenes.

| Model / mode | MAE (vehicles/image) | 95% scene-bootstrap interval | Source |
| --- | ---: | ---: | --- |
| YOLOv8n standard | 4.00 | 2.70–5.31 | `results/eval_labelling_pack/final_evaluation_report.md` |
| YOLOv8n sliced | 2.38 | 1.17–3.67 | `results/eval_labelling_pack/final_evaluation_report.md` |
| YOLOv8s standard | 3.00 | 1.40–4.50 | `results/eval_labelling_pack/final_evaluation_report.md` |
| YOLOv8s sliced | 2.77 | 1.33–4.12 | `results/eval_labelling_pack/final_evaluation_report.md` |
| RT-DETR-L standard | 7.08 | 4.64–10.45 | `results/eval_labelling_pack/final_evaluation_report.md` |
| RT-DETR-L sliced | 14.38 | 5.55–23.44 | `results/eval_labelling_pack/final_evaluation_report.md` |

The registered H4 paired distant-condition analysis reports a YOLOv8n sliced-minus-standard
MAE difference of -3.14 with a 95% scene interval of -5.00 to -2.00 on 7 distant-tagged
images. YOLOv8s and RT-DETR-L intervals include zero. The evidence is descriptive and
limited to the available Dhaka pilot; it is not a broad validation of sliced inference.

### 6.3 Fine-tuned rickshaw pilot

The fine-tuned detector is a one-class rickshaw model. In its eligible first-pass comparison
(7 images, 3 scenes), its covered-total MAE is 6.00 vehicles per image (95% scene interval
1.50–9.50), compared with 3.29 (1.00–6.00) for YOLOv8n. For the combined rickshaw-any
target, fine-tuned MAE is 1.86 (0.67–3.00), compared with 2.00 (1.00–3.00) for YOLOv8n.
The paired covered-total absolute-error difference is +2.71 (0.50–3.67), which is a
regression signal in this small pilot sample; it does not establish a universal result.
These figures come from `docs/rickshaw_model.md` and its referenced evaluation artefacts.

The model's development results are separate and must not be pooled with evaluation.
Its one-class output is zero for other vehicle categories. It cannot separate loader
rickshaws from passenger rickshaws.

### 6.4 Hypothesis status

| Hypothesis | Status | Interpretation |
| --- | --- | --- |
| H1: learned error predictor out-ranks mean-confidence baseline | Not tested | Current evidence does not meet the registered held-out evidence rule. |
| H2: cross-family disagreement correlates with error more than within-family disagreement | Not tested | Current pilot does not establish the registered comparison. |
| H3: local vehicle types are systematically missed/misclassified | Descriptive only | The pilot shows a class-mapping limitation and exploratory observations, not a sufficient general test. |
| H4: sliced inference reduces distant vehicle error | Descriptive only | A small paired result exists for one model; the broader hypothesis remains descriptive. |

## 7. Failure analysis

The repository's failure-case documentation selects difficult pilot examples using existing
cached detections and the labelled subset. Proposed causes such as occlusion, distant/small
objects, glare, and dense overlap are explicitly unconfirmed judgments pending blind human
review. Missing cached model outputs are marked unavailable rather than recreated.
Annotation boxes are not overlaid when their alignment to the selected image copies cannot
be verified.

A detector disagreement does not establish which count is wrong. Human labels themselves
have pass-to-pass variation. The 52 unlabelled evaluation items and the small number of
scenes limit the conclusions that can be drawn.

## 8. Testing and usability

### 8.1 Automated tests

The repository contains unit and workflow test modules for validation, counting, detectors,
reliability, evaluation, history, corrections, curation, video counting, and other workflows.
A module-by-module inventory and execution instructions are in `docs/test_report.md`.
**Fresh full-suite results are pending:** no test suite was executed during this documentation
work because execution was deferred by the project owner. Exact pass/fail/skip counts and
test-passing dependency versions are therefore pending.

### 8.2 Usability

A 20-minute first-time user protocol with six tasks, observation sheet, SUS questionnaire,
citation, and consent note is in `docs/usability_test.md`. Participant sessions have not
been conducted. Completion rates, times, errors, SUS scores, and usability findings are
**pending**. No participant results are claimed.

## 9. Limitations

- The dataset is a small Dhaka imagery pilot and may not represent other cities, roads,
  cameras, seasons, weather, or traffic densities.
- Only 13 of 65 evaluation items have supplied audited first-pass labels in the general
  evaluation artefact.
- Human reference counts vary between passes.
- The pretrained COCO models do not provide dedicated rickshaw and loader-rickshaw classes.
- The fine-tuned detector has unresolved source-data licence/citation and known overlap
  exclusions; its eligible audited evaluation is very small.
- Model agreement is not correctness and cannot be interpreted as a calibrated probability.
- H1 and H2 are not tested; H3 and H4 are descriptive only.
- Video tracking may switch or duplicate identities under occlusion and is not evaluated.
- Formal usability results and fresh automated test results are pending.
- The source dataset licence and the annotation licence must be confirmed before release.

## 10. Future work

1. Extend the audited evaluation set to at least 100 images, with broader scene coverage and
   independently reviewed labels.
2. Complete the preregistered H1 and H2 evaluation on adequate held-out data.
3. Conduct verified box-level analysis with source/image alignment confirmed.
4. Evaluate video crossing counts against a separately annotated video ground-truth set.
5. Confirm the original dataset and fine-tuning dataset licences and citations.
6. Run the full automated suite and clean-environment submission audit.
7. Conduct usability sessions and address only observed blocking or repeated slowing issues.
8. Reconsider optional heatmap and HTML report features only after the required earlier
   parts and test gates are complete.

## 11. Conclusion

TrafficLens has progressed beyond a single-image counter into a multi-model prototype with
reliability comparisons, an offline human-reference evaluation pipeline, local run history,
blind-first review, an optional fine-tuned rickshaw model, and a video-crossing prototype.
The available evidence remains a small pilot on Dhaka imagery. Human references have
documented pass variation, most evaluation images lack supplied audited labels, the
fine-tuned detector has a separate and contamination-limited evaluation, and video is not
evaluated. The system is useful as a research and demonstration prototype, not as a
validated operational traffic measurement system.

## Appendix A. Commands

```bash
python -m pip install -r requirements.txt
streamlit run app/main.py
python scripts/check_testset.py
python scripts/run_benchmark.py --dev
python scripts/run_benchmark.py --evaluation-only
python scripts/run_predictor.py
python -m pytest
```

These commands are listed for reproducibility; they have not all been run in a clean
environment during this report's preparation.

## Appendix B. Metric definitions

- **Count:** number of retained detections after class and confidence filtering.
- **MAE:** mean absolute difference between predicted and reference count.
- **Signed error:** predicted count minus reference count; negative indicates undercount.
- **Count disagreement:** mean of the normalised per-class count differences between models.
- **Box agreement:** class-aware, one-to-one box matching using intersection over union.
- **Scene-cluster bootstrap interval:** percentile interval from resampling scenes rather than
  treating all images as independent.
- **Count range:** minimum-to-maximum total among the compared model outputs; it is not a
  statistical confidence interval.

Full formulas and edge cases are defined in `docs/metrics.md`.

## Appendix C. Database schema

The local database schema, relationships, migration details, and example queries are in
`docs/database_schema.md`.

## Appendix D. Figures and screenshots required

- System architecture diagram (`docs/architecture.mmd`).
- Dashboard overview and Image tab result.
- Reliability tab showing count range and honest validation status.
- Evaluation tab with pilot-scope warning.
- Fine-tuned rickshaw result with separate-evaluation/contamination note.
- Blind review before reveal and saved append-only correction.
- History tab and limitations panel.
- Video crossing screen labelled as unevaluated.

The screenshot inventory is in `docs/screenshots.md`. No screenshots or demo recording are
claimed to exist.

## Appendix E. Numeric provenance register

Every numeric project result or sample-size statement used in this report is listed here
with its source. Counts in this table are aggregate only; no per-image model outputs are
reproduced.

| Number / quantity | Meaning | Source file |
| --- | --- | --- |
| 1.62 vehicles/image | Mean absolute difference between human pass totals across seven categories | `results/eval_labelling_pack/final_evaluation_report.md`; `results/eval_labelling_pack/final_evaluation_summary.json` |
| 13 images; 7 scenes | Paired human-label sample for pass consistency | `results/eval_labelling_pack/final_evaluation_report.md` |
| 0.67–2.46 vehicles | 95% scene-bootstrap interval for seven-category pass consistency | `results/eval_labelling_pack/final_evaluation_report.md` |
| 1.69 vehicles/image; 0.67–2.64 | Covered-class pass consistency mean and interval | `results/eval_labelling_pack/final_evaluation_report.md` |
| 75 images; 65 evaluation; 10 development | Selected test-set size and partition | `README.md`; `configs/testset_plan.json`; `results/eval_labelling_pack/final_evaluation_summary.json` |
| 52 images | Evaluation roster items without supplied first-pass audited counts | `results/eval_labelling_pack/final_evaluation_report.md`; `results/eval_labelling_pack/final_evaluation_summary.json` |
| 13 labelled evaluation images | General pretrained evaluation sample size | `results/eval_labelling_pack/final_evaluation_report.md` |
| 2,000 resamples; seed 20261028 | Scene-cluster bootstrap configuration | `results/eval_labelling_pack/final_evaluation_summary.json`; `core/config.py` |
| YOLOv8n MAE 4.00; interval 2.70–5.31 | Standard-mode overall covered-class pilot result | `results/eval_labelling_pack/final_evaluation_report.md` |
| YOLOv8s MAE 3.00; interval 1.40–4.50 | Standard-mode overall covered-class pilot result | `results/eval_labelling_pack/final_evaluation_report.md` |
| RT-DETR-L MAE 7.08; interval 4.64–10.45 | Standard-mode overall covered-class pilot result | `results/eval_labelling_pack/final_evaluation_report.md` |
| YOLOv8n sliced MAE 2.38; interval 1.17–3.67 | Sliced-mode overall covered-class pilot result | `results/eval_labelling_pack/final_evaluation_report.md` |
| YOLOv8s sliced MAE 2.77; interval 1.33–4.12 | Sliced-mode overall covered-class pilot result | `results/eval_labelling_pack/final_evaluation_report.md` |
| RT-DETR-L sliced MAE 14.38; interval 5.55–23.44 | Sliced-mode overall covered-class pilot result | `results/eval_labelling_pack/final_evaluation_report.md` |
| 7 distant images; -3.14; -5.00 to -2.00 | YOLOv8n paired sliced-minus-standard distant-condition MAE difference and interval | `results/eval_labelling_pack/final_evaluation_report.md` |
| 242 training images | Reproduced fine-tuning training subset size | `docs/rickshaw_model.md` |
| 49 overlaps; 43 evaluation; 6 development | Fine-tuned model training/test contamination matches and partition | `docs/rickshaw_model.md` |
| 22 uncontaminated evaluation images | Fine-tuned model remaining evaluation pool after known overlap exclusions | `docs/rickshaw_model.md` |
| 7 images; 3 scenes | Eligible fine-tuned first-pass audited comparison | `docs/rickshaw_model.md` |
| Fine-tuned covered-total MAE 6.00; interval 1.50–9.50 | Fine-tuned pilot covered-total error | `docs/rickshaw_model.md` |
| YOLOv8n covered-total MAE 3.29; interval 1.00–6.00 | Pretrained comparator in the fine-tuned eligible subset | `docs/rickshaw_model.md` |
| Fine-tuned rickshaw-any MAE 1.86; interval 0.67–3.00 | Fine-tuned pilot rickshaw-any error | `docs/rickshaw_model.md` |
| YOLOv8n rickshaw-any MAE 2.00; interval 1.00–3.00 | Pretrained comparator for rickshaw-any | `docs/rickshaw_model.md` |
| +2.71; 0.50–3.67 | Fine-tuned minus YOLOv8n paired covered-total absolute-error difference | `docs/rickshaw_model.md` |
| 3 pretrained models | Current pretrained detector registry: YOLOv8n, YOLOv8s, RT-DETR-L | `core/config.py`; `README.md` |
| 7 label categories | Human labelling vocabulary | `docs/labelling_protocol.md` |
| 12 pixels | Minimum visible vehicle dimension in the labelling protocol | `docs/labelling_protocol.md` |
| 100 images | Future audited-set target | Project brief in the original task instructions |
| 20 minutes; 6 tasks; 10 SUS items | Planned usability session structure and questionnaire | `docs/usability_test.md` |
| 3–5 minutes | Planned demo duration | `docs/demo_script.md` |

