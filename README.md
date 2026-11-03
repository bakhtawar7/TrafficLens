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
| Sliced (tiled) inference with class-aware merge | Built |
| H2 agreement measures (count disagreement, class-aware box agreement) | Built |
| Reliability tab: per model counts, count range, both agreement scores, sliced toggle | Built |
| Evaluation tab: scored table, error by condition chart, labelling pass agreement | Built (needs benchmark results) |
| Warmup separated from load and inference timing | Built (script/library) |
| Threshold sweep derived from one low-threshold run | Built (script/library) |
| Detector comparison report (run_detectors) | Built (script/library) |
| Local road scene benchmark runner with detection cache | Built (script/library) |
| Metrics reference with every formula (`docs/metrics.md`) | Built |
| ByteTrack video counting | Planned for Week 3 |
| Learned error predictor and reliability score | Planned for Week 3 |
| Needs review flag and SQLite history | Planned for Week 4 |

## Installation

TrafficLens targets Python 3.10 or newer and uses pretrained models only. No training
or fine tuning is performed. Create a clean virtual environment and install the pinned
requirements.

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
result, the counts, and the export downloads.

## Project structure

```text
trafficlens/
  app/            Streamlit entry point and per tab renderers
    main.py       Application entry point
    ui/           Tab renderers (image, video, reliability, evaluation, history)
  core/           Detector registry, counting, annotation, export, validation, dataset,
                  curation, reliability (sweep and agreement), evaluation, config
  configs/        Reviewed class mapping, quartile tag cut offs, sampling plan (seed, subsets)
  data/           Test set media (the 45 curated images are versioned), inventory and logs
  docs/           Problem statement, hypotheses, protocol, architecture, wireframe
  results/        Generated artefacts (curation log, overlays, audit sheet, raw detection
                  cache, evaluation table, summary); untracked
  scripts/        Indexing, cleaning, tagging, test set build and checks, review sheets,
                  detector comparison (run_detectors), benchmark runner (run_benchmark)
  tests/          pytest suite
  .streamlit/     Theme configuration
  requirements.txt
  README.md
```

## Test set

`data/testset/` holds the local road scene benchmark: 45 Dhaka street images drawn by a
seeded, stratified, byte-for-byte reproducible selection. The chosen images are committed,
so a fresh clone can be checked and scored without the archive. TrafficLens never
fabricates or downloads footage; the media comes from an annotated archive supplied by the
operator, whose usage rights are recorded as UNVERIFIED until confirmed.
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

* Sampling quotas, all stored in `core/config.py` with seed 20261028: target 45, at least
  8 images per condition tag, at least 15 carrying a rickshaw or CNG annotation, and at
  most 3 per apparent scene. The target of 60 was not reachable under the scene cap for
  this archive; the build reports shortfalls honestly and pads nothing.
* `configs/tag_cutoffs.json`: the quartile cut offs proposed from the cleaned pool, fixed
  before any sampling happens.
* `configs/testset_plan.json`: the full draw, the 10 image development subset, and the
  seeded random 20 percent marked for a second labelling pass.
* `data/testset/SOURCES.csv`: provenance per item (dataset, split, scene, source, licence,
  url, condition tags).
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

The older manual workflow (`select_testset.py`, `make_review_sheet.py`, `apply_review.py`)
remains available for hand picking a different archive, but rerunning it would overwrite
the current `SOURCES.csv` and `LABELS.csv`.

## Detector comparison and benchmark

Week 2 added a model registry, sliced inference, the reliability agreement measures, and the
ground-truth evaluation. Each is a `core/` module with its own tests, each runs from a command
line script, and each now surfaces in the dashboard. The pieces are:

* `core.detector.build_detector(key)` returns any registry model behind one `Detector`
  protocol, with weights load time and warmup reported separately from inference time.
  `SlicedDetector` wraps any detector in overlapping-tile inference merged with
  class-aware non-max suppression.
* `core.reliability` sweeps one low-threshold run to derive higher thresholds by
  confidence filtering, and computes the H2 agreement measures (normalised per class
  count disagreement and class-aware box agreement at IoU 0.5).
* `core.evaluation` joins `LABELS.csv`, `SOURCES.csv`, and the detection cache to score
  covered and all-vehicle ground truth, excluding the development subset.
* `docs/metrics.md` defines every reported formula (the three ranges, both agreement scores,
  mean absolute error, and the bootstrap interval) so each number is reproducible.

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
```

The benchmark writes one line per (image, model, mode) to `results/raw_detections.jsonl`
and skips entries already cached, so reruns only do the missing work. Evaluation over that
cache produces `results/evaluation_table.csv` and `results/summary.json`; run the full
benchmark first, because the development subset is excluded from every reported statistic
and only cached detections can be scored.

## Limitations

* Occlusion and overlap between vehicles cause undercounting.
* Night scenes and glare degrade detection quality.
* Small and distant vehicles are frequently missed.
* Rickshaws and other local vehicle types have no COCO class and may be missed or
  mislabelled. TrafficLens never claims to detect rickshaws.
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
* **Week 3 (planned):** video counting with ByteTrack and the learned error predictor that
  would turn model agreement into a trust score. Neither is built; the Reliability tab
  reports agreement only and draws no trust verdict.
* **Week 4 (in progress):** stratified test set build with audit artefacts, the local road
  scene benchmark runner, the ground-truth evaluation module, and the Evaluation tab (scored
  table, error by condition, labelling pass agreement) are delivered; the needs review flag
  and SQLite history are still to come.
