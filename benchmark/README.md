# TrafficLens pilot annotation benchmark

This directory contains the metadata for a pilot annotation set of 75 Dhaka street
images drawn from the Dhaka-AI annotated dataset. It is **not** a production benchmark
and must be treated as a pilot annotation set with the limitations described below.

No image files are included. Images are stored in `data/testset/images/` and are
excluded from the benchmark package; use `benchmark/verify.py` to check their
SHA-256 checksums.

---

## Purpose

The benchmark supports offline evaluation of the TrafficLens vehicle counter against
human-labelled counts. It records image identifiers, their original dataset filenames and
splits, scene identifiers, condition tags, labeller-assigned vehicle counts, and the
labelling consistency measured between two independent passes.

The evaluation workflow is in `core/evaluation.py`; the benchmark scripts are in
`scripts/`. The benchmark is a pilot: it uses a small set from one city, one labeller
(with a subset reviewed twice), and one dataset with an unverified licence. See
the Limitations section.

---

## Seeded selection rule

Images were drawn from the Dhaka-AI annotated archive using a three-stage pipeline:

1. **Clean** (`scripts/clean_dataset.py`): drop unreadable, unannotated, and near-duplicate
   images. Near-duplicates are defined as pairs with pHash distance ≤ 10.
2. **Tag** (`scripts/tag_dataset.py`): assign condition tags (day, night, glare, dense,
   sparse, distant) using quartile cut-offs computed from the cleaned pool and frozen
   before any sampling. Cut-offs are stored in `configs/tag_cutoffs.json` and are never
   recomputed after the first run.
3. **Build** (`scripts/build_testset.py`): seeded stratified draw targeting 75 images
   total, at most 3 (later 5) per apparent scene, at least 8 per condition tag, and at
   least 15 carrying a rickshaw or CNG annotation.

**Seed**: 20261028. Stored in `configs/testset_plan.json` and `core/config.py`.

The original target was 60 images under a 3-per-scene cap; that target was not reachable
for this archive. The set was expanded to 75 images under a 5-per-scene cap
(`scripts/expand_testset.py`), giving 65 evaluation and 10 development images. The
shortfall from the 100-image target is documented and not padded.

The build is append-only: the original 10 development images and the first 45 items are
preserved verbatim in the expansion. No model detection enters any selection decision.

To rebuild the test set from a local copy of the archive, run::

    python benchmark/rebuild_testset.py --source /path/to/archive

Then verify SHA-256 checksums::

    python benchmark/verify.py

---

## Labelling procedure

Vehicle counts were assigned by one main labeller following the protocol in
`docs/labelling_protocol.md`. The key rules are:

* Seven categories: car, bus, truck, motorcycle, rickshaw, loader_rickshaw, other.
* Count vehicles as objects, not occupants. Partially visible vehicles are counted when
  their category can be determined with confidence.
* Minimum visible size: 12 pixels in the longest dimension. Vehicles smaller than this
  are treated as background detail.
* Parked and stationary vehicles on the road or roadside are counted.
* `other` category catches ambulances, construction plant, tractors, and other road
  vehicles that fit no primary category.
* The pretrained COCO detectors have no rickshaw class. Rickshaws are labelled as ground
  truth for evaluation purposes.

The blind labelling app (`scripts/label_app.py`) was used for all labelling passes. It
shows the image with no model detections, requires explicit confirmation before
overwriting a saved row, and takes a timestamped backup on each session. This prevents
model outputs from influencing the human count.

---

## Labeller consistency

A seeded random 20% of the evaluation set (13 images in 7 scenes) was relabelled in a
second independent pass. The seed for the second-pass selection is 20261028.

**Consistency figure**: mean absolute total-count difference between pass 1 and pass 2
is **1.62 vehicles per image**. 95% scene-bootstrap interval: **0.67 to 2.46 vehicles**.
Computed from 13 paired images, 7 scenes, 2000 bootstrap samples.

This interval describes within-labeller variability, not model error. The bootstrap
clusters by scene so that the interval is not falsely narrow from repeated images in
the same location.

The figure is sourced from `results/eval_labelling_pack/final_evaluation_report.md` and
is computed by `core/evaluation.py`.

---

## Class mapping

The mapping from Dhaka-AI dataset category names to TrafficLens project categories is
in `benchmark/class_mapping.json` (a copy of `configs/class_mapping.json`). The same
file is used by `core/dataset.py` for all annotation-based counts. Category decisions
are recorded with reasoning; the mapping was not changed after the first labelling pass.

A note on COCO pretrained detectors: the pretrained models map COCO class 2 → car,
class 3 → motorcycle, class 5 → bus, and class 7 → truck. They have no rickshaw or
loader_rickshaw class. The fine-tuned rickshaw detector uses a one-class YOLO model
trained on a user-supplied archive; it is evaluated separately.

---

## Package contents

| File | Description |
|---|---|
| `README.md` | This datasheet |
| `image_ids.csv` | 75 image identifiers with dataset filename, split, scene, condition tags, development/evaluation role, and second-pass status |
| `audited_counts.csv` | Human-labelled counts (pass 1 and pass 2 where available), sourced from `data/testset/LABELS.csv` |
| `class_mapping.json` | Mapping from Dhaka-AI category names to TrafficLens categories |
| `checksums.json` | SHA-256 of every image file in `data/testset/images/` |
| `rebuild_testset.py` | Script to rebuild the test set from the original archive |
| `verify.py` | Checks package integrity: no image files, checksums, coverage, count validity |

---

## Limitations

* **Small size**: 75 images, 65 evaluation images. This is a pilot; it is not large enough
  for reliable condition-stratified statistics. Any per-condition result with fewer than 15
  images is marked exploratory.
* **Single location**: all images are from Dhaka, Bangladesh. Results do not generalise to
  other cities without further evaluation.
* **One main labeller**: all pass-1 labels come from one labeller. The 1.62-vehicle
  consistency figure is within-labeller, not between-labeller.
* **Pilot annotation**: counts for the development subset were used for tuning decisions
  (`configs/final_settings.json`). Development images are excluded from every reported
  evaluation statistic.
* **Unverified licence**: the Dhaka-AI dataset licence is listed inconsistently across
  mirrors. One mirror records CC0 1.0; another records GNU GPL 2.0. The licence must be
  confirmed at the original source before the benchmark is used in any publication or
  redistributed. `data/testset/SOURCES.csv` records the licence field as "UNVERIFIED".
* **Rickshaw and CNG count mix**: the archive labels both auto rickshaws and CNG three-
  wheelers as the same category; both are mapped to `rickshaw` in this project.

---

## Source dataset citation

Dhaka-AI annotated dataset:
> Dhaka AI Lab (2020). *Dhaka-AI: Vehicle detection and classification in Dhaka traffic*.
> Dataset supplied by operator; hosted at Roboflow Universe and other mirrors.
> Licence: **listed inconsistently across mirrors (CC0 1.0 on one page, GNU GPL 2.0 on
> another); must be confirmed at the original source before use**.
> Downloaded from operator-supplied archive.zip; original URL not recorded.

COCO pretrained model weights: Ultralytics YOLOv8 and RT-DETR are trained on
Microsoft COCO 2017 (CC BY 4.0). See `https://cocodataset.org` for the licence.

---

## Licence for these annotations

**[PLACEHOLDER — confirm before publication]**

The labelling work in this project is currently unlicensed pending confirmation of the
source dataset licence. Until the source licence is resolved, the annotations in
`benchmark/audited_counts.csv` and `data/testset/LABELS.csv` may not be redistributed.
