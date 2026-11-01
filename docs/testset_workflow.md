# Test Set Workflow

This document describes how a raw annotated archive becomes the TrafficLens benchmark in
`data/testset`. It exists because the guide in `test_set_guide.md` explains the rules, while
this file explains the machinery and the division of labour between the scripts and the
person doing the verification.

The archive is never modified. Every step reads it and writes into the project.

## Who does what

| Step | Performed by | Output |
| --- | --- | --- |
| Index and measure every image | Script | `data/testset/inventory.csv` |
| Screen, rank, and select 50 to 60 items | Script | `data/testset/selection_log.csv`, images copied, draft `SOURCES.csv` and `LABELS.csv` |
| Render verification material | Script | `data/testset/review/` sheets and `REVIEW.csv` |
| Confirm tags, counts, and privacy | Person | Completed `REVIEW.csv` |
| Fold the answers back in | Script | Pass 2 authoritative `LABELS.csv`, corrected `SOURCES.csv` |
| Validate consistency | Script | Output of `check_testset.py` |

You are not asked to choose images one by one from thousands. You are asked to confirm
or correct a short list that the scripts have already justified, which is where human
judgement actually adds value.

## Step 1: index the archive

```powershell
python scripts/index_dataset.py --source "C:/path/to/archive"
```

For every image the script finds the PASCAL VOC XML beside it, decodes the picture, and
records dimensions, a sharpness score, mean brightness, the share of blown out pixels, the
ground truth count per TrafficLens category, suggested condition tags, and a perceptual
hash. Each row also carries the reason the item was rejected, or an empty cell when it is
fit for selection.

Rejections are about the file, not the scene: images with no annotation, unreadable files,
frames shorter than 360 pixels on the short edge, visibly blurry frames, annotation sizes
that disagree with the picture, fewer than two countable vehicles, and frames so crowded
that no person could verify them.

## Step 2: select the set

```powershell
python scripts/select_testset.py                    # dry run, prints the plan
python scripts/select_testset.py --apply --clean    # write the chosen items
```

The dry run prints coverage before anything is written, which is the point of the step: if
the plan has no night scenes, adjust the parameters and look again.

Selection is a deterministic greedy pass over the ranked candidates. A candidate is skipped
when its perceptual hash is within eight bits of something already chosen, when its area
already holds the maximum number of items, or when its exact tag combination is already
full. A second pass relaxes the caps if the target cannot otherwise be reached.

Rare conditions lose that ranking to abundant ones, so `--min-tag TAG=COUNT` reserves slots
for them before the general pass runs. The reserved items still obey the duplicate rule, and
the caps are relaxed for them when the quota cannot otherwise be met:

```powershell
python scripts/select_testset.py --target 55 --min-tag night=12 --min-tag sparse=8
```

If the pool cannot satisfy a quota the report says `quota NOT reachable`, which is a fact
about the archive rather than a failure.

Chosen images are copied as `<item_id>_<area>_<conditions>.<ext>`, and draft rows are added
to `SOURCES.csv` and `LABELS.csv` with `labeller_pass` set to `1`. Two fields cannot be
known from an archive and are marked rather than invented:

* `permission` is written as `UNVERIFIED` and must be replaced with the real usage grant.
* `location_area` falls back to the archive batch name when the file name holds no
  recognised place, and the note says so.

## Step 3: build the verification package

```powershell
python scripts/make_review_sheet.py
```

Each item is redrawn with its ground truth boxes and a caption carrying the item id, the
suggested tags, and the counts. Detectable categories use their usual colours while
rickshaws, loaders, and other vehicles are drawn in grey, because a COCO detector can never
report them and the reviewer should see that gap immediately. Items are tiled into contact
sheets, and `data/testset/review/REVIEW.csv` is written with one row per item.

A deterministic twenty percent draw is marked `second_pass_due`, which satisfies the second
pass requirement in `labelling_protocol.md`.

## Step 4: verify

Open the sheets and answer four questions per row in `REVIEW.csv`:

* `keep`: `no` removes the item from the set.
* `tags_are_correct`: `yes` or `no`. When `no`, edit `corrected_condition_tags` using only
  the six permitted tags, separated by semicolons.
* `counts_are_correct`: `yes` or `no`. When `no`, overwrite the number columns with what you
  can actually see, applying the twelve pixel rule and the parked vehicle rule.
* `privacy_ok`: `yes` only when no clear face or readable number plate survives. Anything
  else means blur the frame or mark `keep` as `no`.

Fill `reviewer_initials` on every row you touched. Rows left unanswered make
`apply_review.py` refuse to run, which is intentional.

## Step 5: fold the answers back in

```powershell
python scripts/apply_review.py --dry-run
python scripts/apply_review.py
```

Pass 1 draft rows are kept so labeller agreement can be measured later, and reviewed rows
are appended with `labeller_pass` set to `2` and a note listing the categories that changed.
Corrected tags are written to `SOURCES.csv`, and a file whose tags changed is renamed to
match the convention. Add `--drop-excluded` to delete images marked `keep` as `no`.

## Step 6: validate

```powershell
python scripts/check_testset.py
```

A clean run means every file on disk has a source row, every row has a file, no permission
field is blank, and every tag is one of the six permitted values. That script is unchanged
and remains the single gate on the folder.

## Reproducing a different set

Every step takes parameters, so a different set is a different command rather than a code
change. Common adjustments:

| Want | Change |
| --- | --- |
| More or fewer items | `--target 60` |
| A floor for a rare condition | `--min-tag night=12` |
| Stricter area spread | `--max-per-area 5` |
| More scene variety per tag mix | `--max-per-signature 8` |
| Larger contact sheet images | `--max-edge 1000 --per-sheet 4` |
| A fresh pass over a new archive | re-run step 1, then step 2 with `--clean --force` |

## Limits of the automatic pass

Lighting is decided from three signals: overall luma, the share of the frame that is well
lit, and the brightness of the top fifth of the frame. The middle signal matters because a
dashcam frame of a sunny street is dominated by dark asphalt, so mean brightness alone
called hundreds of ordinary day scenes `night`. Dusk frames that sit near the threshold
still can be, and that is a review correction rather than a script change.
The `glare` tag is a highlight statistic and is frequently a suggestion rather than a fact.
The `distant` tag depends on box sizes in the annotation, which reflect the labeller rather
than the camera. Density is derived from the count, so an undercounted annotation produces a
weaker `dense` tag. All four are marked for confirmation in `notes` and are corrected in
step 4.
