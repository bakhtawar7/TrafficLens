# TrafficLens

## Overview

TrafficLens is a vehicle counting system for images and, in later weeks, video. It
runs pretrained object detectors over a supplied input, filters the results down to
the vehicle classes of interest, and reports a per class count together with an
annotated image. The Week 1 scope delivers the counting foundation: a working image
counter, result exports, a clean dashboard shell, documentation, and scaffolding for a
local road scene test set.

The system deliberately exposes its own uncertainty. Counting is only useful when the
operator understands how far that count can be trusted, so every screen carries honest
limitations and later weeks add reliability scoring on top of the raw count.

## Features

Status legend: **Built** means available and tested in the current codebase.
**Built (script/library)** means the capability runs from a command line script or a
`core/` module with its own tests, but is not yet surfaced in a dashboard tab.
**Planned for Week N** means the interface exists but the behaviour does not yet, and the
app shows a clearly labelled placeholder panel rather than fake data.

| Feature | Status |
| --- | --- |
| Image vehicle counting (default YOLOv8n) | Built |
| Per class counts (car, motorcycle, bus, truck) | Built |
| Confidence threshold control (0.05 to 0.95) | Built |
| Annotated image output | Built |
| JSON and CSV export | Built |
| Input validation with clear errors | Built |
| Archive indexing, screening, and review sheets | Built |
| Dataset cleaning with logged drops (unreadable, unannotated, near duplicates) | Built |
| Quartile condition tagging with cut offs pinned before sampling | Built |
| Seeded stratified test set build with audit overlays and sampling plan | Built |
| Dashboard shell with custom theme | Built |
| Model registry, three pretrained detectors (YOLOv8n, YOLOv8s, RT-DETR-L) | Built |
| Optional one-class fine-tuned rickshaw pilot, isolated from pretrained counts | Built (pilot; see `docs/rickshaw_model.md`) |
| Sliced (tiled) inference with class-aware merge | Built |
| H2 agreement measures (count disagreement, class-aware box agreement) | Built |
| Reliability tab: per model counts, count range, both agreement scores, sliced toggle | Built |
| Evaluation tab: scored table, error by condition chart, labelling pass agreement | Built (needs benchmark results) |
| Warmup separated from load and inference timing | Built (script/library) |
| Threshold sweep derived from one low-threshold run | Built (script/library) |
| Detector comparison report (run_detectors) | Built (script/library) |
| Local road scene benchmark runner with detection cache | Built (script/library) |
| Settings provenance: defaults file hashed into the summary | Built (script/library) |
| Development-only read guard for tuning (dev_guard) | Built (script/library) |
| Signed error, bias, and scene-level cluster bootstrap intervals | Built (script/library) |
| Blind manual count audit app (label_app) with atomic saves | Built (script/library) |
| Local run history with SQLite, filters, details, CSV export, and deletion | Built |
| Label-free reliability feature table (build_features) | Built (script/library) |
| Metrics reference with every formula (`docs/metrics.md`) | Built |
| Provenance-gated ByteTrack video line counting | Built; not evaluated |
| Learned error predictor and reliability score | Planned for Week 3 |
| Blind review queue with append-only human corrections | Planned |
| Needs review flag | Built; not yet validated |

## Installation

TrafficLens targets Python 3.10 or newer. Its standard counting and reliability workflows
use pretrained models; an optional, separate one-class rickshaw pilot is available in the
Image tab. That experimental model does not replace pretrained counts or reliability
signals. See `docs/rickshaw_model.md` for its provenance, contamination, and pilot limits.
Create a clean virtual environment and install the pinned requirements.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The pretrained weights download automatically on first use and are cached in the project
directory: `yolov8n.pt` and `yolov8s.pt` for the YOLO detectors and `rtdetr-l.pt` for
RT-DETR-L. They are excluded from version control by `.gitignore`. The dashboard uses
YOLOv8n; the comparison and benchmark scripts exercise all three.

## Running the app

From the project root:

```powershell
streamlit run app/main.py
```

The browser opens the dashboard on `http://localhost:8501`. Upload an image, choose the
classes to count, adjust the confidence slider, and the Image tab shows the annotated
result, the counts, and the export downloads. The off-by-default **Rickshaw aware mode
(fine tuned)** toggle runs the separate one-class pilot; it uses a distinct rickshaw box
colour and exports its model name and weights hash. It cannot detect the other vehicle
classes and does not change pretrained counts, ranges, agreement scores, or flags.
The History tab stores run metadata, filenames, hashes, counts, and comparison signals in
the ignored local `data/history.db`; it never stores image bytes. See
`docs/database_schema.md` for the schema. History is local to this checkout and is not
ground truth.

## Project structure

```text
trafficlens/
  app/            Streamlit entry point and per tab renderers
    main.py       Application entry point
    ui/           Tab renderers (image, video, reliability, evaluation, history)
  core/           Detector registry, counting, annotation, export, validation, dataset,
                  curation, reliability (sweep and agreement), evaluation, settings
                  provenance, development read guard, label free features, config
  configs/        Reviewed class mapping, quartile tag cut offs, sampling plan (seed,
                  subsets), default model settings
  data/           Test set media (the 75 curated images are versioned), inventory and logs
  docs/           Problem statement, hypotheses, protocol, architecture, wireframe
  results/        Generated artefacts (curation log, overlays, audit sheet, raw detection
                  cache, evaluation table, summary, feature table); untracked
  scripts/        Indexing, cleaning, tagging, test set build, expansion and checks, tag
                  proposals, review sheets, blind labelling app (label_app), detector
                  comparison (run_detectors), benchmark runner (run_benchmark), feature
                  table (build_features)
  tests/          pytest suite
  .streamlit/     Theme configuration
  requirements.txt
  README.md
```

## Test set

`data/testset/` holds the local road scene benchmark: 75 Dhaka street images (65
evaluation, 10 development) drawn by a seeded, stratified, byte-for-byte reproducible
selection. The chosen images are committed, so a fresh clone can be checked and scored
without the archive. TrafficLens never fabricates or downloads footage; the media comes
from an annotated archive supplied by the operator, whose usage rights are recorded as
UNVERIFIED until confirmed.
`docs/test_set_guide.md` states the collection rules, `docs/labelling_protocol.md` states
the counting rules, and `docs/testset_workflow.md` states the mechanics.

The build is a three stage pipeline; every stage reads the archive and writes only inside
this project, and running any stage twice with the same inputs gives identical output:

```powershell
python scripts/clean_dataset.py --source "C:/path/to/archive"          # drop unreadable, unannotated, and near duplicate images
python scripts/tag_dataset.py --source "C:/path/to/archive"            # quartile cut offs, written before any sampling
python scripts/build_testset.py --source "C:/path/to/archive" --apply  # seeded draw, copies images, writes artefacts
python scripts/check_testset.py                                        # validate the folder
```

What the pipeline guarantees, and where each result lives:

* Sampling quotas, all stored in `core/config.py` with seed 20261028: the original build
  targeted 45 images, at least 8 per condition tag, at least 15 carrying a rickshaw or CNG
  annotation, and at most 3 per apparent scene. The target of 60 was not reachable under
  that scene cap for this archive; the build reports shortfalls honestly and pads nothing.
* `scripts/expand_testset.py` then grew the set from 45 to 75 items under a pre-registered
  cap rule: take the smallest per-scene cap in {3, 4, 5} that yields at least 100 evaluation
  images with at least 15 per tag; no cap qualified, so cap 5 was used for the maximum
  feasible 65 evaluation images, with the shortfall reported and nothing padded. The
  expansion is append-only: the original 45 items and the same 10 development images are
  preserved verbatim, selection reuses the frozen seed and cut offs, and no model detection
  enters any selection decision. Evaluation per tag after expansion: day 22, night 19,
  glare 19, dense 32, sparse 16, distant 21, with 283 rickshaw or CNG boxes across 54 images.
* `configs/tag_cutoffs.json`: the quartile cut offs proposed from the cleaned pool, fixed
  before any sampling happens.
* `configs/testset_plan.json`: the full draw, the 10 image development subset, and the
  seeded random 20 percent (13 items) marked for a second labelling pass.
* `data/testset/SOURCES.csv`: provenance per item (dataset, split, scene, source, licence,
  url, condition tags), plus `tag_source` and `tag_confirmed` recording how each tag was
  assigned and whether the labeller has confirmed it.
* `scripts/tag_proposal.py` handles items the frozen cut-offs left tagless: it recomputes
  the screening features, re-applies the same cut-offs without changing them, writes only
  the tags the rule supports (a mid-band image may honestly carry none), and records
  `tag_source` as `auto_proposed` with review images under `results/tag_review/`.
  `scripts/check_testset.py` accepts these rows while printing how many tags remain
  unconfirmed.
* `data/testset/annotation_counts.csv`: counts derived from the archive annotations only,
  never from a detector; they are a draft for the labeller, not ground truth.
* `results/audit_overlays/` and `results/audit_sheet.csv`: every chosen image with its
  annotation boxes drawn, and a counting sheet whose `my_` columns are blank for the
  labeller. Both are generated outputs and stay untracked.
* `LABELS.csv` is human ground truth and the build never writes it. The auto derived draft
  that once filled it described a different, earlier selection, so it now sits in the
  untracked `results/draft_auto_labels.csv` and `LABELS.csv` is a header only scaffold
  waiting for the first labelling pass. The evaluator refuses any row whose notes still
  record the counts as auto derived, so no error statistic can be built from unverified
  numbers.
* `scripts/label_app.py` is the blind counting app for the labelling passes: it shows the
  image only, never a model detection or count, saves each row atomically with a
  timestamped backup, requires explicit confirmation before overwriting a saved row, and
  lets the labeller confirm condition tags after counting.

The older manual workflow (`select_testset.py`, `make_review_sheet.py`, `apply_review.py`)
remains available for hand picking a different archive, but rerunning it would overwrite
the current `SOURCES.csv` and `LABELS.csv`.

## Detector comparison and benchmark

Week 2 added a model registry, sliced inference, the reliability agreement measures, and the
ground-truth evaluation; Week 4 added settings provenance and the development read guard on
top. Each piece is a `core/` module with its own tests and runs from a command line script.
The pieces are:

* `core.detector.build_detector(key)` returns any registry model behind one `Detector`
  protocol, with weights load time and warmup reported separately from inference time.
  `SlicedDetector` wraps any detector in overlapping-tile inference merged with
  class-aware non-max suppression.
* `core.reliability` sweeps one low-threshold run to derive higher thresholds by
  confidence filtering, and computes the H2 agreement measures (normalised per class
  count disagreement and class-aware box agreement at IoU 0.5).
* `core.evaluation` joins `LABELS.csv`, `SOURCES.csv`, and the detection cache to score
  covered and all-vehicle ground truth, excluding the development subset. It reports mean
  absolute error, signed error and bias, undercount/overcount/exact tallies, and mean
  relative error; every interval is a scene-level cluster bootstrap, and any condition
  with fewer than 15 evaluation images is marked exploratory.
* `core.settings` loads the settings file a run was produced under (default
  `configs/default_settings.json`, which records the current per-model confidence, input
  size, sliced mode, tile settings, and NMS IoU). The benchmark stamps every cached entry
  with the file's SHA-256 plus the input size and NMS fields, and the evaluator records the
  same provenance in `results/summary.json`, so every statistic stays traceable to its
  configuration. The loader only reads; it selects nothing.
* `core.dev_guard` is the leakage guard for tuning: it returns only the seeded development
  subset ids and raises `DevelopmentOnlyError` before any bytes are read when code asks for
  a non-development image.
* `core.features` computes label-free reliability features per image from the cached
  standard-mode detections and the pixels alone: confidence statistics, threshold
  sensitivity, the H2 agreement pairs and family aggregates, scene statistics, and box
  geometry. `scripts/build_features.py` writes `results/features.csv` stamped with the
  settings hash, reuses the table until that hash changes, and prints only aggregate
  summaries (feature means, ranges, missing counts). It reads no label or annotation file,
  so the table can be built before labelling and cannot leak a label into a feature.
* `docs/metrics.md` defines every reported formula (the three ranges, both agreement scores,
  mean absolute and signed error, bias, relative error, and the cluster bootstrap interval)
  so each number is reproducible.

On the dashboard, the Reliability tab shows per model counts, the count range, and both
agreement values for the uploaded image, with a sidebar toggle to switch the models between
full-frame and sliced inference; every figure is labelled as measured from the models, with no
verdict about trust. The Evaluation tab shows the scored table, error by condition as a chart,
and the labelling pass agreement when `results/evaluation_table.csv` exists, and keeps its
placeholder otherwise.

```powershell
python scripts/run_detectors.py                     # per-image counts and timings, all three detectors
python scripts/run_benchmark.py --dev               # cache raw detections for the development subset
python scripts/run_benchmark.py                     # full test set, standard and sliced modes
python scripts/run_benchmark.py --settings configs/default_settings.json   # explicit settings file
python scripts/tune_on_dev.py                       # guarded coarse grid and frozen final settings
python scripts/run_benchmark.py --evaluation-only --settings configs/final_settings.json --configured-confidence
python -c "from core.config import FINAL_SETTINGS_JSON; from core.evaluation import evaluate; evaluate(settings_json=FINAL_SETTINGS_JSON)"
```

The benchmark writes one line per (image, model, mode) to `results/raw_detections.jsonl`
and skips entries already cached for the selected settings fingerprint, so reruns only do
the missing work. `scripts/tune_on_dev.py` evaluates standard inference only on the guarded
development images, writes its per-setting metrics to `results/tune_dev_table.csv`, and
selects a read-only `configs/final_settings.json`. Pass that file to the evaluation-only
benchmark and evaluator to keep the final cache and `results/summary.json` tied to the
frozen settings hash. The sliced benchmark arm reuses the selected model settings with
the default tile configuration. A missing settings file stops a run rather than letting
it proceed without provenance.
Evaluation over that cache produces `results/evaluation_table.csv` and `results/summary.json`;
run the full benchmark first, because the development subset is excluded from every reported
statistic and only cached detections can be scored.

## Limitations

* Occlusion and overlap between vehicles cause undercounting.
* Night scenes and glare degrade detection quality.
* Small and distant vehicles are frequently missed.
* The pretrained COCO detectors have no rickshaw class and may miss or mislabel local
  vehicle types. The optional one-class fine-tuned rickshaw pilot is separate, cannot
  detect other categories, and is not a validated replacement.
* Video tracking may assign duplicate or switched IDs. Occlusion can interrupt or merge
  tracks, so virtual-line inbound/outbound counts are not guaranteed to match unique vehicles.
  Video results are outside the evaluation and carry no accuracy claim.
* Dashboard counts come from pretrained models and are not yet checked against ground
  truth in the app; `core.evaluation` computes that comparison offline, but its numbers
  are only as good as the human labelling pass they depend on.
* The test set counts derived from archive annotations still require a human labelling
  pass, and the licence of the supplied archive is unverified.

## Roadmap

* **Week 1 (delivered):** image counter, exports, validation, dashboard shell,
  documentation, test set scaffold.
* **Week 2 (delivered):** model registry with three pretrained detectors, warmup and load
  timing, threshold sweep, detector comparison, and sliced inference, all behind one
  `Detector` protocol and covered by the test suite; plus the Reliability tab showing per
  model counts, the count range, and the two H2 agreement measures (count disagreement and
  class-aware box agreement) with a sidebar sliced toggle.
* **Week 3 (partial):** provenance-gated video counting with a virtual line and ByteTrack is
  built. Add a row to `data/samples/video/PROVENANCE.csv` with the video filename, source URL,
  author, licence text, and ISO download date before processing; the app never downloads a
  clip or model weights. Video results are not evaluated and carry no accuracy claim. The
  learned error predictor is not built; the Reliability tab reports agreement only and draws
  no trust verdict.
* **Week 4 (in progress):** stratified test set build with audit artefacts (expanded to 75
  images under the pre-registered cap rule), the local road scene benchmark runner, the
  ground-truth evaluation module, and the Evaluation tab (scored table, error by condition,
  labelling pass agreement) are delivered, together with the blind labelling app, settings
  provenance, and the development read guard that prepare leakage-free tuning. Tuning itself
  waits for the audited first-pass labels; the needs review flag and SQLite history are
  still to come.
