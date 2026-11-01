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
**Planned for Week N**
means the interface exists but the behaviour does not yet, and the app shows a clearly
labelled placeholder panel rather than fake data.

| Feature | Status |
| --- | --- |
| Image vehicle counting with YOLOv8n | Built |
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
| Second and third detectors (YOLOv8s, RT-DETR-L) | Planned for Week 2 |
| Threshold sweep and detector comparison | Planned for Week 2 |
| Sliced inference for distant vehicles | Planned for Week 2 |
| ByteTrack video counting | Planned for Week 3 |
| Learned error predictor and reliability score | Planned for Week 3 |
| Needs review flag and SQLite history | Planned for Week 4 |
| Local road scene benchmark and evaluation | Planned for Week 4 |

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

The YOLOv8n weights download automatically on first use and are cached in the project
directory. They are excluded from version control by `.gitignore`.

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
  core/           Detector, counting, annotation, export, validation, dataset, curation, config
  configs/        Reviewed class mapping, quartile tag cut offs, sampling plan (seed, subsets)
  data/           Test set media (the 45 curated images are versioned), inventory and logs
  docs/           Problem statement, hypotheses, protocol, architecture, wireframe
  results/        Generated audit artefacts (curation log, overlays, audit sheet); untracked
  scripts/        Indexing, cleaning, tagging, test set build and checks, review sheets
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
* `LABELS.csv` is human ground truth and the build never writes it. It still holds the
  earlier reviewed draft whose ids predate the current selection, so `check_testset.py`
  reports the mismatch until a labeller pass reconciles the two files.

The older manual workflow (`select_testset.py`, `make_review_sheet.py`, `apply_review.py`)
remains available for hand picking a different archive, but rerunning it would overwrite
the current `SOURCES.csv` and `LABELS.csv`.

## Limitations

* Occlusion and overlap between vehicles cause undercounting.
* Night scenes and glare degrade detection quality.
* Small and distant vehicles are frequently missed.
* Rickshaws and other local vehicle types have no COCO class and may be missed or
  mislabelled. TrafficLens never claims to detect rickshaws.
* All results come from pretrained models without verification against ground truth in
  Week 1.
* The test set counts derived from archive annotations still require a human labelling
  pass, and the licence of the supplied archive is unverified.

## Roadmap

* **Week 1 (delivered):** image counter, exports, validation, dashboard shell,
  documentation, test set scaffold.
* **Week 2:** additional detectors, threshold sweep, detector comparison, sliced
  inference.
* **Week 3:** video counting with ByteTrack, learned error predictor, reliability
  scoring.
* **Week 4 (in progress):** stratified test set build with audit artefacts delivered;
  needs review flag, SQLite history, and local benchmark evaluation still to come.
