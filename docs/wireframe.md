# Wireframe

This is an annotated layout of the TrafficLens dashboard as built in Week 1. It shows the
sidebar and each tab, what is live now, and what is a labelled placeholder.

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
| [ Limitations v ]   |                                                                 |
+---------------------+----------------------------------------------------------------+
```

## Sidebar (live)

| Control | Behaviour |
| --- | --- |
| Upload image | Accepts JPG, JPEG, PNG, WEBP up to 15 MB. Invalid inputs raise a clear message in the Image tab. |
| Classes to count | Multiselect of car, motorcycle, bus, truck. All four selected by default. |
| Confidence | Slider from 0.05 to 0.95, default 0.25, step 0.05. Does not reload the model. |
| Model indicator | Static label showing YOLOv8n (pretrained). |
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

## Video tab (placeholder)

A single panel headed "Planned for Week 3". It lists the future ByteTrack capabilities:
clip upload, per frame detection, tracking each vehicle once, unique totals, and playback.
No data is shown.

## Reliability tab (placeholder)

A single panel headed "Planned for Week 3". It describes the future learned error predictor,
the baseline comparison, the needs review flag, and the reliability score. No data is shown.

## Evaluation tab (placeholder)

A single panel headed "Planned for Week 4". It describes the future benchmark comparison,
mean absolute error, error by condition tag, and local vehicle type breakdown. No data is
shown.

## History tab (placeholder)

A single panel headed "Planned for Week 4". It describes the future SQLite store, run
browsing, filtering, and range export. No data is shown.
