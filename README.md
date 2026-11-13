# TrafficLens

## Overview

TrafficLens is a Streamlit prototype for counting vehicles in road-scene images. It runs
pretrained object detectors, filters detections to selected vehicle classes, annotates the
image, and reports per-class and total counts. Additional workflows compare detector
outputs, evaluate cached detections against human labels, keep local run history, and support
a blind-first human correction loop. A separate video prototype counts virtual-line crossings.

**Evidence boundary:** results come from a pilot on Dhaka imagery. Reference counts are
human counts; the documented consistency estimate is a mean absolute pass difference of
1.62 vehicles (95% scene-bootstrap interval 0.67–2.46; 13 paired images across 7 scenes).
The optional fine-tuned rickshaw mode is a separate one-class model with its own evaluation
and contamination exclusions. It does not replace the pretrained multi-class detectors.
H1 and H2 are not tested under the pre-registered evidence rules; H3 and H4 are descriptive
only. Video has not been evaluated against labelled video ground truth.

## Features and status

| Feature | Status |
| --- | --- |
| Image upload, validation, decode, annotated count | Built |
| Per-class counts and JSON/CSV export | Built |
| Three pretrained detectors (YOLOv8n, YOLOv8s, RT-DETR-L) | Built |
| Sliced/tiled inference | Built |
| Reliability tab: model disagreement, box agreement, count range | Built; agreement is not correctness |
| Human-reference evaluation pipeline and dashboard | Built; pilot evidence only |
| Local SQLite run history and CSV export | Built |
| Blind-first review queue and append-only user corrections | Built |
| Optional fine-tuned rickshaw detector | Pilot; separate evaluation and contamination exclusions |
| Learned error predictor and needs-review flag | Experimental workflow; user-facing flag remains not validated unless validated configuration is available |
| Video virtual-line counting with ByteTrack | Prototype; not evaluated |
| Formal usability findings | Pending participant sessions |
| Density heatmap and HTML summary report | Skipped; Part E gate requires completed A–D and passing tests |

## Installation

TrafficLens targets Python 3.10 or newer. Create a virtual environment from the project
root.

**Windows PowerShell**

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Linux / macOS**

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The current `requirements.txt` specifies minimum versions, not an exact reproducible
environment lock. No constraints file of test-passing versions is supplied because the
full suite has not been run in this work session. Runtime compatibility should be confirmed
in a clean environment before submission.

## Run

From the repository root:

```bash
streamlit run app/main.py
```

Open the local Streamlit URL shown in the terminal. Upload a supported image, select vehicle
classes, and adjust the confidence threshold. The Image tab displays the original and
annotated image, count metrics, detections, and downloads.

The optional **Rickshaw aware mode (fine tuned)** toggle runs a separate one-class rickshaw
detector. It does not detect the other vehicle categories and does not alter pretrained
counts, ranges, reliability comparisons, or evaluation. Its source-data licence/citation
metadata remains unresolved; see `docs/rickshaw_model.md`.

The Video tab requires a provenance record in `data/samples/video/PROVENANCE.csv`. It uses
local weights and does not download videos or model weights. Video crossings are not part
of the evaluation and carry no accuracy claim.

## Configuration and data

- `configs/default_settings.json`: standard per-model inference configuration used by the
  benchmark runner unless overridden.
- `configs/final_settings.json`: frozen settings used by the final application paths.
- `configs/finetuned_settings.json`: separate fine-tuned detector inference settings.
- `configs/class_mapping.json`: proposed dataset-to-category mapping for review; it is
  not an approved detector mapping.
- `configs/tag_cutoffs.json` and `configs/testset_plan.json`: condition cut-offs and
  seeded test-set selection metadata.
- `data/testset/SOURCES.csv`: image provenance, split, scene, and condition tags.
- `data/testset/LABELS.csv`: manually recorded human counts (23 Pass 1 rows: 10
  development and 13 evaluation; plus 13 Pass 2 rows for the same evaluation items).
  Do not replace these with annotation-derived draft counts.
- `data/history.db`: local SQLite run history, ignored by Git. Image bytes are not stored.
- Model weights and generated result caches are not intended for version control.

The source inventory records the dataset licence as unverified. The benchmark metadata
package notes the licence inconsistency across mirrors (CC0 1.0 versus GNU GPL 2.0); confirm
the original source terms before redistribution. The licence for project-authored annotations
is still a placeholder.

## Run history and human review

The History tab stores filenames, image hashes, dimensions, settings hashes, counts, and
reliability metadata locally. It provides filters, run details, CSV export, deletion, and
clear-history confirmation. A locked or unwritable history store should not block image
counting; the UI reports a warning.

The Human review queue is blind by default. A reviewer re-uploads the exact image (SHA-256
must match), enters the seven category counts, and then reveals the model comparison.
Corrections are append-only user-reported records. They are never written to
`data/testset/LABELS.csv` and never update model weights or inference settings.

See `docs/database_schema.md` for the schema and `docs/usability_test.md` for the
first-time usability session protocol.

## Evaluation and pilot results

The project test set contains 75 Dhaka images: 65 evaluation and 10 development items,
selected using the frozen seeded plan. Human pass consistency is documented for 13 paired
images across 7 scenes: mean absolute total-count difference 1.62 vehicles, with a 95%
scene-bootstrap interval of 0.67–2.46. This is a human-label consistency statistic, not
model performance.

The optional fine-tuned rickshaw model has a separate pilot evaluation. Its documented
training selection overlaps 49 of 75 test images; known overlaps are excluded from its
evaluation figures. Only 7 uncontaminated evaluation images across 3 scenes have eligible
first-pass labels for its audited-count comparison. The checkpoint is a one-class detector
and cannot distinguish loader rickshaws. Its data licence and citation remain unresolved.
See `docs/rickshaw_model.md` for its separately reported results and limitations.

The hypotheses retain their registered evidence labels: H1 and H2 are **not tested**;
H3 and H4 are **descriptive only**. Model disagreement is not ground truth, and a difference
is not an improvement unless the relevant interval excludes zero.

## Project structure

```text
app/                 Streamlit entry point and tab renderers
core/                validation, detectors, counting, reliability, evaluation, history
configs/             frozen settings, mappings, cut-offs and selection plan
data/testset/        curated test images and source/label metadata
data/samples/video/  video provenance template
benchmark/           image-free pilot annotation metadata package
docs/                problem, scope, metrics, architecture, protocol and reports
scripts/             curation, benchmark, evaluation and analysis workflows
tests/               automated test suite
results/              generated evaluation artefacts (not committed)
requirements.txt      runtime and test dependency floors
```

## Useful commands

```bash
python scripts/check_testset.py
python scripts/run_benchmark.py --dev
python scripts/run_benchmark.py --evaluation-only
python scripts/run_predictor.py
pytest
```

The commands above are documented entry points; they have not been executed as part of the
current documentation-only work. Do not interpret this list as a passing test report.

## Limitations

- The evidence is a small pilot on Dhaka imagery and may not transfer to other cities,
  cameras, weather, or traffic conditions.
- Human reference counts have measurable pass-to-pass variation.
- The selected test set is limited in size and scene coverage.
- Pretrained COCO models do not have dedicated rickshaw or loader-rickshaw classes.
- The fine-tuned detector is a separate one-class pilot with known contamination exclusions
  and unresolved dataset licensing/citation.
- Reliability agreement describes model consistency, not correctness.
- H1 and H2 are not tested; H3 and H4 are descriptive only.
- Video tracking can produce duplicate or switched IDs under occlusion and is not evaluated.
- The first-time usability protocol has not yet been run; findings are pending.

## Roadmap

1. Complete the manual defect review and run the requested regression/full test suites when
   authorized.
2. Conduct the 20-minute first-time usability sessions and report only observed results.
3. Confirm source dataset and annotation licences before any redistribution.
4. Extend the audited human-labelled set to at least 100 images and broaden scene coverage.
5. Test H1 and H2 under their pre-registered rules; conduct verified box-level analysis.
6. Evaluate video against annotated video ground truth.
7. Revisit optional heatmap and HTML report extras only after Parts A–D are complete and
   tests pass.
