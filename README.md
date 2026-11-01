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

Status legend: **Built** means available and tested in Week 1. **Planned for Week N**
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
  core/           Detector, counting, annotation, export, validation, dataset, config
  data/           Test set media, inventory and logs, sample media placeholder
  docs/           Problem statement, hypotheses, protocol, architecture, wireframe
  scripts/        Archive indexing, test set selection, review sheets, test set checks
  tests/          pytest suite
  .streamlit/     Theme configuration
  requirements.txt
  README.md
```

## Test set

`data/testset/` holds the local road scene benchmark. TrafficLens never fabricates or
downloads footage; the media comes from an archive you have the right to use.
`docs/test_set_guide.md` states the collection rules, `docs/labelling_protocol.md` states the
counting rules, and `docs/testset_workflow.md` states the mechanics.

When an annotated archive is available, the scripts do the filtering and you do the
verifying:

```powershell
python scripts/index_dataset.py --source "C:/path/to/archive"  # measure and screen every image
python scripts/select_testset.py                               # preview a diverse 55 item plan
python scripts/select_testset.py --apply --clean               # copy picks, draft SOURCES and LABELS
python scripts/make_review_sheet.py                            # contact sheets and REVIEW.csv
python scripts/apply_review.py                                 # record the verified counts as pass 2
python scripts/check_testset.py                                # validate the folder
```

Rare conditions are reserved rather than hoped for: add `--min-tag night=12` to the
selection command to guarantee a floor of that condition in the set.

`SOURCES.csv` records provenance and permission for each item and `LABELS.csv` records the
ground truth counts, draft pass 1 rows first and authoritative pass 2 rows after review. An
archive without annotations can still be used: select the items by hand, then label them
following the protocol.

## Limitations

* Occlusion and overlap between vehicles cause undercounting.
* Night scenes and glare degrade detection quality.
* Small and distant vehicles are frequently missed.
* Rickshaws and other local vehicle types have no COCO class and may be missed or
  mislabelled. TrafficLens never claims to detect rickshaws.
* All results come from pretrained models without verification against ground truth in
  Week 1.

## Roadmap

* **Week 1 (delivered):** image counter, exports, validation, dashboard shell,
  documentation, test set scaffold.
* **Week 2:** additional detectors, threshold sweep, detector comparison, sliced
  inference.
* **Week 3:** video counting with ByteTrack, learned error predictor, reliability
  scoring.
* **Week 4:** needs review flag, SQLite history, local benchmark evaluation.
