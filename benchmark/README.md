# TrafficLens pilot annotation set

## Purpose and status

This directory is a **pilot annotation set and metadata package**, not a new image
dataset release. It contains no image files. It records the selected dataset filenames,
split and scene metadata, condition tags, the available audited human label passes,
the class definitions, and the labelling protocol. Rebuild the images from the original
dataset folder only when you have the original archive and the project's required local
inputs.

TrafficLens results come from a pilot on Dhaka imagery. Reference counts are human counts.
The documented consistency estimate is a mean absolute total-count difference of 1.62
vehicles across 13 paired images in 7 scenes (95% scene-bootstrap interval 0.67–2.46).
The optional fine-tuned rickshaw detector is a separate one-class model with its own
evaluation and contamination exclusions.

## Contents

- `image_manifest.csv`: selected item IDs, original dataset filenames, split, scene,
  source, licence status, condition tags and notes, copied from the project source inventory.
- `audited_labels.csv`: the audited human pass records supplied in the evaluation labelling
  pack. These are human labels, not model outputs. Both available passes are retained.
- `labelling_protocol.md`: category definitions, inclusion rules, first/second pass procedure,
  and adjudication guidance.
- `class_mapping_proposal.json`: explicitly proposed archive-label mapping for review;
  it is not applied to detector classes and must not be treated as an approved mapping.
- `verify.py`: checks that the package contains no image files and, when generated,
  validates the SHA-256 manifest.
- `generate_checksums.py`: generates checksums for the package files after checkout.

## Selection and split

The selected set contains 75 images recorded in `data/testset/SOURCES.csv` and
`image_manifest.csv`. The manifest's `split` column is the **original archive split**;
all 75 selected rows currently carry `train` in that source-dataset field. It is not the
project's evaluation/development partition. That separate partition is defined by
`configs/testset_plan.json`: 10 item IDs are in `development_subset`, and the remaining
65 selected items form the evaluation roster. Scene identifiers, condition tags, and
source provenance are recorded separately. The plan and project selection log hold the
seeded selection details. Consult the plan and `docs/test_set_guide.md` before rebuilding.

## Human labelling

`audited_labels.csv` contains 26 records: Pass 1 and Pass 2 for each of the 13
evaluation items with audited human counts. These records match the corresponding rows
in `data/testset/LABELS.csv`. That source file also contains 10 additional Pass 1 rows
for the development subset; those development rows are not included in this package's
audited evaluation subset. The labelling protocol defines seven categories: car, bus,
truck, motorcycle, rickshaw, loader rickshaw, and other. It specifies partial visibility,
occlusion, a minimum visible dimension rule, parked vehicles, and second-pass adjudication.

The consistency figure above describes agreement between human passes; it is not model
performance. The sample is small, drawn from Dhaka imagery, and has one main labeller.
Do not treat this as a broadly representative traffic benchmark. H1 and H2 are not tested;
H3 and H4 are descriptive only under the current pilot evidence rules.

## Source dataset licence and citation

The project source inventory currently records the dataset licence as unverified and
requires confirmation before redistribution. The Dhaka-AI licence is listed inconsistently
across mirrors: CC0 1.0 on one page and GNU GPL 2.0 on another. This discrepancy must be
confirmed against the original source before any image or annotation redistribution.
The original dataset citation and URL have not been established in the supplied project
provenance; do not invent them.

## Annotation licence

**Placeholder:** the licence for the project's own human annotations must be selected and
confirmed by the annotation owner before redistribution.

## Rebuild from original data

This package deliberately contains no images. From the repository root, with the original
dataset archive extracted into the location expected by the project's configuration, use
the existing project pipeline:

```bash
python scripts/check_testset.py
python scripts/build_testset.py
```

Check `python scripts/build_testset.py --help` and `docs/testset_workflow.md` for the
actual supported arguments and source-folder requirements before running. The rebuild
depends on the original archive and the frozen project plan; neither is bundled here.

## Checksums

Run `python benchmark/generate_checksums.py` to write `benchmark/SHA256SUMS`, then
run `python benchmark/verify.py`. A checksum file is not claimed to have been generated
in this repository revision. Checksums must be regenerated after any package file changes.
