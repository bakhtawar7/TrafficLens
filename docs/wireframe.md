# Wireframe

This is an annotated layout of the TrafficLens dashboard as built through Week 2. It shows
the sidebar and each tab, what is live now, and what is a labelled placeholder.

## Overall frame

```text
+------------------------------------------------------------------------------------+
| TrafficLens                                                                          |
| Vehicle counting from a single image, with honest limits.                            |
+---------------------+----------------------------------------------------------------+
| SIDEBAR             |  [ Image ] [ Video ] [ Reliability ] [ Evaluation ] [ History ] |
|                     +----------------------------------------------------------------+
| Input               |                                                                 |
|  [ Upload image  ]  |                   SELECTED TAB CONTENT                          |
|                     |                                                                 |
| Classes to count    |                                                                 |
|  [x] car            |                                                                 |
|  [x] motorcycle     |                                                                 |
|  [x] bus            |                                                                 |
|  [x] truck          |                                                                 |
|                     |                                                                 |
| Confidence [0.25]   |                                                                 |
|                     |                                                                 |
| Model: YOLOv8n      |                                                                 |
|                     |                                                                 |
| (o) Sliced inference|  (toggles the Reliability tab between full-frame and tiled)    |
|                     |                                                                 |
| [ Limitations v ]   |                                                                 |
+---------------------+----------------------------------------------------------------+
```

## Sidebar (live)

| Control | Behaviour |
| --- | --- |
| Upload image | Accepts JPG, JPEG, PNG, WEBP up to 15 MB. Invalid inputs raise a clear message in the Image and Reliability tabs. |
| Classes to count | Multiselect of car, motorcycle, bus, truck. All four selected by default. |
| Confidence | Slider from 0.05 to 0.95, default 0.25, step 0.05. Does not reload the model. |
| Model indicator | Static label showing YOLOv8n, the Image tab detector. The Reliability tab compares all registry models. |
| Sliced inference | Toggle, off by default. When on, the Reliability tab runs each model inside the tiled wrapper. No effect on the Image tab. |
| Limitations | Persistent expander listing the honest caveats. |

## Image tab (live)

```text
+------------------------------------------------------------------+
| [info banner: "0 vehicles detected" shown only when total is 0]  |
|                                                                  |
| +---------------------------+  +---------------------------+     |
| | Original image            |  | Annotated image           |     |
| +---------------------------+  +---------------------------+     |
|                                                                  |
| [Total] [Car] [Motorcycle] [Bus] [Truck] [Inference ms]  metrics |
|                                                                  |
| Detections                                                       |
| +--------------------------------------------------------------+ |
| | class | confidence | x_min | y_min | x_max | y_max           | |
| +--------------------------------------------------------------+ |
|                                                                  |
| [ Download JSON ]        [ Download CSV ]                        |
+------------------------------------------------------------------+
```

* Before an upload, the tab shows a prompt to choose an image.
* Boxes use a consistent colour per class, and labels stay readable on dark and bright
  scenes.

## Reliability tab (live)

Compares the registry models against each other on the uploaded image. Every figure is
labelled "measured from the models" and carries no verdict about correctness.

```text
+-----------------------------------------------------------------------------+
| banner: numbers are measured between the models, not against real traffic   |
| caption: confidence, mode (standard or sliced), model count, classes        |
|                                                                             |
| Per model counts (measured from the models)                                 |
| +-------------------------------------------------------------------------+ |
| | Model | Total | car | motorcycle | bus | truck                          | |
| +-------------------------------------------------------------------------+ |
| [Lowest model total] [Highest model total] [Count range]        metrics    |
|                                                                             |
| Model agreement (measured from the models)                                  |
| +-------------------------------------------------------------------------+ |
| | Pair | Relation | Count disagreement | Box agreement                     | |
| +-------------------------------------------------------------------------+ |
| [Within: count disagreement] [Within: box agreement]                        |
| [Cross:  count disagreement] [Cross:  box agreement]            metrics    |
+-----------------------------------------------------------------------------+
```

* Before an upload, the tab prompts for an image and shows no metrics.
* Count disagreement is the mean normalised per-class difference in [0, 1]; box agreement is
  class-aware IoU matching at 0.5. Definitions live in `docs/metrics.md`.
* The sidebar Sliced inference toggle reruns every model in tiled mode; the tables and
  metrics keep the same shape.

## Evaluation tab (live when the benchmark has been run)

If `results/evaluation_table.csv` does not exist, the tab keeps the honest "Planned for
Week 4" placeholder. Once `core.evaluation` has produced it, the tab shows the scored table,
the error by condition as a Plotly chart, and the labelling pass agreement.

```text
+-----------------------------------------------------------------------------+
| banner: measured against human ground truth, development subset excluded     |
| Per item table (Item, Condition, Ground Truth, models, range, agreement...)  |
| [Plotly grouped bars: mean absolute error by condition, one series per model]|
| Labelling pass agreement                                                     |
| [Items labelled twice] [Covered mean abs diff] [Identical fraction] [CI95]   |
+-----------------------------------------------------------------------------+
```

* The warning that fewer than 30 labelled items exist is surfaced when the summary sets it.

## Video tab (placeholder)

A single panel headed "Planned for Week 3". It lists the future ByteTrack capabilities:
clip upload, per frame detection, tracking each vehicle once, unique totals, and playback.
No data is shown.

## History tab (live)

The local history tab discloses that uploaded filenames are stored and that image bytes
are not. It provides filename and model filters, a run table, a model-results detail view,
CSV export, per-run deletion, and clear-history confirmation. SQLite database errors are
shown without disabling image counting. Human correction review remains planned.

```text
+--------------------------------------------------------------------------+
| Pilot, Dhaka imagery; human pass consistency; fine-tuned pilot and overlap|
| Filter by filename [             ]   Filter by model [                 ] |
| Timestamp | Filename | Mode | Rickshaw mode | Flag | Run ID              |
| View run details                                                       |
| Model | Total | Counts by class | Inference | Weights SHA-256          |
| [Export history as CSV]                                                 |
| [Select run] [Confirm deletion] [Delete selected run]                  |
| [Confirm clear history] [Clear history]                                |
+--------------------------------------------------------------------------+
```
