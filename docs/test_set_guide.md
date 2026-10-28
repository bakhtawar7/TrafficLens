# Test Set Guide

TrafficLens is only as trustworthy as the data used to judge it. This guide describes how
to build the local road scene test set that Week 4 uses for evaluation. TrafficLens never
fabricates or downloads footage. Every item is collected by you, and every item must be
recorded honestly.

## Target size

Collect at least 50 distinct items. An item is either a still image placed in
`data/testset/images/` or a short clip placed in `data/testset/clips/`. Aim for a spread
across the condition tags below rather than 50 near identical daytime shots.

## Condition tags

Each item carries one or more condition tags. Tags are drawn from this fixed set and
nothing else:

| Tag | Meaning |
| --- | --- |
| day | Captured in normal daylight. |
| night | Captured after dark or under low ambient light. |
| glare | Strong sun, headlight bloom, or wet surface reflections. |
| dense | Traffic closely packed, heavy mutual occlusion. |
| sparse | Only a few vehicles visible with wide gaps. |
| distant | Most vehicles are small in frame, far from the camera. |

A single item may hold several tags, for example a busy night scene could be
`night;dense;distant`. Separate multiple tags with a semicolon.

## Naming convention

Name every media file with its item id so the CSV rows and the files line up exactly:

```text
<item_id>_<area>_<conditions>.<ext>
```

For example, `tl001_mainst_day.jpg` or `tl014_ringroad_night-dense.mp4`. Keep item ids
unique, zero padded, and stable once assigned. The `filename` column in `SOURCES.csv`
must match the file on disk character for character.

## Recording permission

Before counting an item as part of the set, record its provenance in `SOURCES.csv`:

* `item_id`: the unique id used in the file name.
* `filename`: the exact file name on disk.
* `type`: `image` or `clip`.
* `source`: where it came from (own capture, a named licence, a partner).
* `permission`: how you are allowed to use it, for example `own`, `cc-by`, or a written
  grant reference. This field must not be left blank.
* `date_captured`: ISO date the item was captured, or the acquisition date.
* `location_area`: a coarse area name, not an exact address.
* `condition_tags`: one or more tags from the fixed set.
* `notes`: anything a reviewer should know.

## Privacy rules

* Avoid identifiable people. Do not include clear faces or pedestrians used as subjects.
* Avoid readable number plates. If plates are legible, blur them before adding the item,
  or exclude the item. TrafficLens performs no plate reading and none is ever added.
* Favour wide street views where individual vehicles matter more than individuals.

## Validating the set

After adding files and rows, run the checker from the project root:

```powershell
python scripts/check_testset.py
```

It reports files missing from `SOURCES.csv`, rows with no matching file, blank permission
fields, and invalid condition tags, then prints a count per condition. It exits non zero
when problems are found, so a clean run means the scaffold is consistent.
