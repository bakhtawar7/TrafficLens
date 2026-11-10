# Metrics

Every number TrafficLens reports is defined here so it can be reproduced from the raw
detections without reading the source. Each entry names the exact code that computes it,
the inputs it needs, and the edge cases. Constants come from `core/config.py` and are
listed at the bottom. Where a value is rounded for display, the rounding is stated; the
unrounded value is what downstream formulas use.

Notation. An *item* is one test set image. A *model* is a registry entry. `a_c` and `b_c`
are the counts of class `c` from two models; `n_a` and `n_b` are total detection counts of
two models after class filtering. The selected vehicle classes are `C = {car, motorcycle,
bus, truck}`. Boxes are `(x_min, y_min, x_max, y_max)` in pixels.

## Counting (`core/counting.py`)

* A detection is kept when `confidence >= threshold` (lower bound, inclusive). This is the
  ultralytics semantic and is what both the dashboard and the sweep rely on.
* `counts_by_class[c]` is the number of kept detections whose `class_name == c`, for every
  `c` in the selected classes; a class with no matches is recorded as `0`.
* `total_count` is the number of kept detections after dropping any class not selected.
  Detections of classes outside `C` (for example `person`) never enter a count.

## Threshold sweep (`core/reliability.py`)

The sweep runs a model once at the lowest threshold `t_min = min(thresholds)` and derives
every higher threshold `t` by filtering that single run: keep detections with
`confidence >= t`, then count as above.

* Exactness: `conf` is a `>=` bound and ultralytics greedy NMS emits boxes in descending
  score order, so the survivors above any `t >= t_min` are precisely the `t_min` survivors
  whose confidence passes `t`. A derived count equals a direct run at that threshold.
* Documented deviation: ultralytics returns at most `max_det` boxes (300 default), highest
  scores first. A dense frame whose `t_min` run exceeds the cap loses low-confidence boxes
  before derivation, so a derived higher threshold can undercount versus a direct run. On
  such frames the direct run is the reference.

### Ranges

Three different ranges are reported; they are not interchangeable.

* Model `threshold_range` = `(min, max)` of one model's `total_count` across the sweep
  thresholds. Computed in `sweep_detector`.
* Report `full_range` = `(min, max)` of `total_count` across every model and every threshold
  together. Computed in `sweep_all`.
* Benchmark / Evaluation `Count Range` = `min`-`max` of the three standard model totals at a
  single confidence, and the same spread on the Reliability tab (metrics "Lowest / Highest
  model total" and "Count range"). Computed in `core/evaluation.build_rows` and
  `app/ui/reliability_tab._render_counts`.

## Agreement measures for H2 (`core/reliability.py`)

Both compare two models on one image, after dropping classes outside the selected set.

### Count disagreement

Per class `c` in `C`:

```
d_c = |a_c - b_c| / (a_c + b_c)     when a_c + b_c > 0
d_c = 0                             when a_c = b_c = 0
```

`d_c` lies in `[0, 1]`: `0` means the two models counted class `c` identically, `1` means
every counted vehicle of that class was claimed by only one of them. The headline
`count_disagreement` is the unweighted mean of `d_c` over `C`. `docs/hypotheses.md` H2 names
this "the normalised difference in per class counts" but gives no closed form; the formula
above is the chosen normalisation and is part of the metric's definition.

### Box agreement

Intersection over union of two boxes:

```
intersection = max(0, min(ax1, bx1) - max(ax0, bx0)) * max(0, min(ay1, by1) - max(ay0, by0))
union        = area(a) + area(b) - intersection
IoU(a, b)    = intersection / union            (0 when union <= 0)
```

Matching is greedy, class-aware, and one-to-one at `iou_threshold = config.BOX_IOU_THRESHOLD`
(0.5):

1. Build every same-class pair with `IoU >= iou_threshold`.
2. Sort pairs by `(-IoU, index_a, index_b)` so the result is deterministic under ties.
3. Walk the sorted list, claiming a pair only when neither box is already matched.

```
box_agreement = 2 * matches / (n_a + n_b)
```

This symmetric F1 form lies in `[0, 1]`: identical sets score `1`, disjoint sets of the same
size score `0`. Two empty sets are defined as `1`; one empty and one non-empty set are `0`.
The Reliability tab reports the mean of `box_agreement` over the three model pairs as its
agreement column.

### Family aggregation

Pairs are labelled `within` when the two models share a registry family (`yolo` or `detr`)
and `cross` otherwise.

* `within_family` = mean of each measure over all within-family pairs. `None` when no such
  pair exists.
* `best_yolo` = the YOLO model with the highest capacity suffix rank
  (`n < s < m < l < x`, from the last character of the display name). With the current
  registry this is YOLOv8s.
* `cross_family` = mean of each measure over the cross-family pairs that contain `best_yolo`.
  `None` when there is no YOLO model or no other-family model.

## Ground truth targets (`core/evaluation.py`)

Read from `data/testset/LABELS.csv`, first labelling pass, counts normalised to ints:

* `covered_total` = car + bus + truck + motorcycle, the classes a COCO model can honestly
  claim.
* `all_vehicles_total` = covered_total + rickshaw + loader_rickshaw. No model detects the
  last two, so this target bakes in the rickshaw gap and bounds the honest error floor before
  any model fault. The `other` column is excluded from both.

The development subset in `configs/testset_plan.json` is removed before every statistic and
from the written table, so no reported error is tuned on data used to build the pipeline.

A `LABELS.csv` row whose `notes` still carry the draft wording described in
`labelling_protocol.md` is refused when the labels are loaded, before any target above is
built. Counts the pipeline derived from the archive annotation therefore contribute no
statistic at all. How many rows were refused, which items, and why are recorded in
`summary.json` under `refusals`, and the same statement is printed when the evaluation runs.

## Mean absolute error (`core/evaluation.py`)

Over a set of items `I` (per model, per target, or per condition):

```
MAE = (1 / |I|) * sum over i in I of |count_i - ground_truth_i|
```

`count_i` is a model's total for item `i`; `ground_truth_i` is the covered or all-vehicles
total for that item. Reported to 4 decimals. Per-condition MAE groups items by each tag in
their `condition_tags` (an item with several tags contributes to each tag's bucket).

## Signed error (`core/evaluation.error_stats`)

MAE discards direction, so it cannot say whether a model runs consistently low or high.
The signed error keeps the sign. For one item and one target (covered or all-vehicles),
with `p` the model count and `t` the audited ground-truth total:

```
signed_error = p - t        (negative = undercounting, positive = overcounting)
```

The per-item signed error is written to `results/evaluation_table.csv` as one column per
model for each of the two targets (for example `YOLOv8n Covered Signed Error` and
`YOLOv8n All Vehicles Signed Error`). It is also the raw material for every aggregate below.

### Bias (mean signed error)

```
bias = (1 / n) * sum over i of (p_i - t_i)
```

A negative bias means the model undercounts on average; `+0.4` means it trends high.
`bias` sits next to `mae` in each `model_mae[...]["covered"]`, `model_mae[...]["all_vehicles"]`
and `condition_mae[tag][model]` block in `summary.json`.

### Undercounted, overcounted, exact

```
undercounted = |{ i : p_i - t_i < 0 }|
overcounted  = |{ i : p_i - t_i > 0 }|
exact        = |{ i : p_i - t_i = 0 }|
undercounted + overcounted + exact = n
```

These three counts are recorded per model, per target, overall and per condition.

### Mean relative error (`core/evaluation.error_stats`)

```
relative_error_i = |p_i - t_i| / t_i      when t_i > 0
```

`mean_relative_error` is the mean of `relative_error_i` over the items with `t_i > 0`.

Rule for a zero true count: `t_i = 0` makes the ratio undefined (division by zero), so
zero-truth images are **excluded from the relative mean** and reported separately as
`zero_true_items`. Excluding them does not hide anything, because those same images still
count toward `bias`, `mae`, and the under/over/exact totals - an empty scene a model
hallucinates vehicles into is flagged as overcounted there. `relative_error_items` records
how many images the relative mean actually used; when that is `0` (every true count was
zero), `mean_relative_error` is `None`.

Worked example. For `(p, t)` pairs `(3,5)`, `(7,5)`, `(4,4)`, `(0,0)`, `(2,0)`:
signed errors are `-2, +2, 0, 0, +2`, so `undercounted = 1`, `overcounted = 2`,
`exact = 2`, `bias = 2/5 = 0.4`, and `mae = (2+2+0+0+2)/5 = 1.2`. Relative error uses the
three `t > 0` items - `0.4, 0.4, 0.0` - giving `mean_relative_error = 0.2667`,
`relative_error_items = 3`, and `zero_true_items = 2`.

## Exploratory conditions (`core/evaluation.annotate_exploratory`)

A condition tag carried by fewer than `EXPLORATORY_MIN_IMAGES` (15) evaluation images is
statistically thin, so it is flagged **exploratory** wherever it is reported, and every
condition is labelled with the number of distinct scenes behind it:

* In the table, each row's `Exploratory Conditions` cell lists which of that row's own tags
  are exploratory (blank when none are).
* In `summary.json`, `condition_detail[tag]` records `n_images` (evaluation images carrying
  the tag), `scenes` (distinct scene labels among them), and `exploratory` (true when
  `n_images < 15`).

The scene count matters because 15 images drawn from 2 scenes carry far less information
than 15 drawn from 8, and the cluster bootstrap reflects exactly that.

## Bootstrap confidence interval (`core/evaluation.cluster_bootstrap_ci`)

Every confidence interval the evaluator writes is a **scene-level (cluster) percentile
bootstrap** of the mean of a list of per-item absolute errors. Images copied from one
scene (a burst of near-identical frames, recorded in the `scene` column of `SOURCES.csv`)
are not independent observations, so resampling them as if they were would understate the
uncertainty. Resampling whole scenes keeps that dependence and widens the interval to an
honest width.

1. Group the values by their scene label into clusters `G = {g_1, ..., g_K}`.
2. Seed a `random.Random(config.SEED)` generator.
3. Draw `B = 2000` resamples. Each resample picks `K` clusters with replacement, then takes
   the mean of every value in the drawn clusters (a cluster picked twice contributes its
   values twice).
4. Sort the `B` means ascending. Lower = index `int(0.025 * B)`; upper = index
   `min(int(0.975 * B), B - 1)`. Each is rounded to 4 decimals.

An empty input yields `(None, None)`. The seed and resample count are the same as the old
image-level bootstrap; only the resampling unit changed. Because the seed is fixed and
clusters are grouped in first-appearance order over the item-sorted rows, the interval
reproduces exactly across runs.

`core/evaluation.bootstrap_ci` is kept as the plain image-level routine (resample
individual items). It is no longer used by the pipeline but is retained so the cluster
interval can be compared against the image-level interval it replaced.

Worked example. Ten scenes of three items each, where the three items in a scene share one
value and scene values alternate between 0 and 10. All 30 items average 5.0. The
image-level bootstrap of the 30 items returns about `[3.333, 6.667]` (width 3.33), while the
scene-level cluster bootstrap returns `[2.0, 8.0]` (width 6.0) - the whole-scene draws can
pick mostly-0 or mostly-10 scenes, so the interval is at least as wide, which is the point.

## Labeller pass agreement (`core/evaluation.summarise`)

Over items labelled in both the first and second pass and outside the development subset:

* `items_both_passes` = number of such items.
* `covered_mean_abs_difference` = mean of `|covered_total(pass1) - covered_total(pass2)|`,
  the same MAE formula with the two human passes standing in for model and ground truth.
  Bootstrapped with the routine above for `covered_ci95`.
* `all_vehicles_mean_abs_difference` = same statistic on `all_vehicles_total`.
* `identical_fraction` = fraction of items whose covered difference is exactly `0`.

## Label free reliability features (`core/features.py`)

Features that may predict counting error, computed from inference time information
only: the cached standard-mode detections of all three registry models and the image
pixels. `scripts/build_features.py` never opens `LABELS.csv`, `annotation_counts.csv`,
or any dataset annotation, so the table exists before labelling and cannot leak a
label into a feature. The cache is read at the lowest sweep threshold (0.15);
confidence, agreement, and geometry features then filter at `DEFAULT_CONFIDENCE`
(0.25), while threshold sensitivity uses the full low-threshold list. The table covers
every cached image, development subset included, because no label is involved.

Undefined values are `None` (empty CSV cells), never a fabricated zero. An empty
detection set has no mean confidence but an honestly zero sensitivity and density.

### Confidence (per model)

Over one model's detections at the default confidence:

```
mean_confidence = mean of the detection scores
min_confidence  = lowest detection score
mid_band_share  = fraction with 0.25 <= confidence <= 0.50   (bounds inclusive)
```

Empty set: all three are `None`. Why it may predict error: a low mean or minimum
means the model is unsure about what it saw, and a large mid-band share means many
detections sit near the decision boundary, so the count depends heavily on the
threshold and is fragile.

### Threshold sensitivity (per model)

With `n_t` the model's total vehicle count derived at each sweep threshold
`t` in `(0.15, 0.25, 0.35, 0.50)`:

```
threshold_sensitivity = max(n_t) - min(n_t)
```

Empty set: `0.0` (zero at every threshold). Why it may predict error: a count that
moves widely with the threshold is built from unstable detections, so any single
threshold choice carries a large error bar.

### Agreement (pairs and families)

Exactly the H2 measures defined above, computed at the default confidence:
`box_agreement` and `count_disagreement` for each of the three model pairs, plus
`within_family` (mean over the same-family pair, YOLOv8n vs YOLOv8s) and
`cross_family` (mean over the cross-family pairs containing `best_yolo`, currently
YOLOv8s vs RT-DETR-L). A missing model leaves its pairs and any aggregate without
pairs as `None`. Why it may predict error: independently trained models rarely fail
the same way, so low agreement flags an image where at least one model is likely
wrong (hypothesis H2).

### Scene statistics (pixels only)

```
mean_brightness = mean of the grayscale image
contrast        = standard deviation of the grayscale image
saturated_share = fraction of pixels with any BGR channel clipped at 255
blur            = variance of the Laplacian (CV_32F on grayscale)
```

Brightness and blur use the same definitions as `core.dataset.measure_image`
(`mean_luma` and `sharpness`). Saturation here is true channel clipping, distinct
from the screening metric `highlight_fraction` (grayscale >= 245). Missing or
undecodable image: all four `None`. Why they may predict error: darkness, glare,
low contrast, and blur are the documented degradation conditions; each raises the
miss and false detection rate.

### Geometry (pooled over every present model)

Boxes from all models at the default confidence are pooled; a vehicle seen by three
models contributes three boxes, because the features describe detector output, not
unique vehicles. With `A = width * height`:

```
box_area_fraction     = box area / A
median_box_area_fraction = median of the fractions
small_box_share       = fraction of boxes with box_area_fraction < 0.001   (strict)
detection_density     = number of boxes * 1_000_000 / A    (boxes per megapixel)
```

The small box threshold `SMALL_BOX_AREA_FRACTION = 0.001` (0.1% of the frame, about
a 30x30 box in a 1000x1000 image) is a scale invariant stand-in for the COCO small
object cutoff. No boxes: median and share are `None`, density is `0.0`; no image:
all three `None`. Why they may predict error: small boxes mark distant vehicles,
which detectors miss most often, and high density means overlap and occlusion,
where both missed vehicles and NMS mistakes concentrate.

### Table mechanics

`results/features.csv` carries `item_id`, the `settings_sha256` of the settings file
used (default `configs/default_settings.json`), and the 29 feature columns in the
stable order of `core.features.TABLE_COLUMNS`. The runner reuses an existing table
whose every row carries the current hash and recomputes when the hash changes
(`core.features.needs_recompute`). It prints only aggregate summaries (per feature
mean, range, and missing count); per-image detections and counts never reach the
console. Feature constants live in `core/features.py`: `MID_CONFIDENCE_BAND`
`(0.25, 0.50)` and `SMALL_BOX_AREA_FRACTION` `0.001`.

## Constants (`core/config.py`)

| Name | Value | Role |
| --- | --- | --- |
| `DEFAULT_CONFIDENCE` | 0.25 | dashboard and agreement default threshold |
| `MIN_CONFIDENCE` / `MAX_CONFIDENCE` | 0.05 / 0.95 | allowed threshold range |
| `THRESHOLD_SWEEP` | (0.15, 0.25, 0.35, 0.50) | sweep values; lowest is the run threshold |
| `BOX_IOU_THRESHOLD` | 0.5 | IoU at which same-class boxes count as one vehicle |
| `TARGET_CLASS_MAP` | 2 car, 3 motorcycle, 5 bus, 7 truck | counted classes |
| `SLICE_TILE_SIZE` | 640 | sliced inference tile edge |
| `SLICE_OVERLAP_RATIO` | 0.2 | tile overlap as a fraction of one edge |
| `SLICE_IOU_THRESHOLD` | 0.5 | sliced merge non-max suppression IoU |
| `SEED` | 20261028 | master seed, drives the bootstrap intervals |
| `EXPLORATORY_MIN_IMAGES` | 15 | a condition tag with fewer evaluation images is marked exploratory |
