# Fine-tuned rickshaw model pilot

## Status and checkpoint inspection

This is an experimental, optional one-class detector in the Image tab. The frozen
pretrained detectors remain the default, and this model is excluded from reliability
agreement, flags, and pretrained evaluation columns.

The checkpoint
`runs/rickshaw_finetune/yolov8n_rickshaw_plus10/weights/best.pt` was readable. Its
SHA-256 is
`44a6a3e383f074c79a56d9160acbf3f627b056a94a54f843f9ee7f0cef942c24`.
Ultralytics 8.4.191 loaded it as a `detect` model with `DetectionModel` architecture,
class index `0: rickshaw`, and image size 640. The checkpoint metadata records the
task, data/model references, input size, and class information. The continuation
arguments in `runs/rickshaw_finetune/yolov8n_rickshaw_plus10/args.yaml` include
10 epochs, a 0.1 training fraction, batch size 1, CPU device, image size 640, and NMS
IoU 0.7. Those training arguments describe the experiment, not inference settings.

## Provenance and class mapping

The user-supplied input is `C:\Users\bakht\Downloads\archive.zip`, whose recorded
SHA-256 is
`9de3fc20cca3f6d0d75d0f66123d4dd0d36ec455dca2cfd5f87315e46791bc9f`. The supplied
task input did not include the dataset source URL, authors, licence text, download
date, or paper citation. These fields are unknown and must not be inferred from the
archive filename or annotation labels. Redistribution is not authorized by this
documentation.

The archive identity as Auto-RickshawImageBD could not be verified from the supplied
provenance. If it is that dataset, its requested paper citation still needs to be
provided and recorded before making a provenance-complete claim.

| Model class | Project mapping |
| --- | --- |
| `0: rickshaw` | `rickshaw` |

The detector cannot separate loader rickshaws. For count evaluation its output is
compared with the combined audited reference `rickshaw_any = rickshaw +
loader_rickshaw`. Its predictions for car, bus, truck, and motorcycle are zero; it is
not a multi-class vehicle detector.

## Frozen inference settings

The isolated inference settings are confidence 0.25, image size 640, and NMS IoU 0.7.
They match the frozen base configuration, whose SHA-256 is
`0c73720e5ee97e9a83ef6a654440226f367f3ebcc47e17bcce12ce4a37ea670e`. The fine-tuned
settings-file SHA-256 is
`5fa521eb12fe6f27ee81f4ed37b62b175c4dc583beed094db806c10c8a14f32d`. The settings
file is read-only. The runner records the model and settings hashes in
`results/summary.json`.

## Contamination and evaluation split

The actual Ultralytics training selection was reproduced as the sorted first 0.1 of
the prepared training images: 242 images. Test images were compared by filename and
with the project perceptual hash; a Hamming distance strictly below 8 counts as an
overlap. There were 49 overlapping test images: 43 evaluation-subset images and 6
development-subset images. All known overlaps are excluded from fine-tuned
evaluation figures. The pretrained cache entries were byte-preserved while adding
the separate fine-tuned cache records.

There are 22 uncontaminated evaluation images remaining, but only 7 images across 3
scenes have eligible first-pass labels for the audited-count comparison. The
development results use 4 images across 3 scenes and are separate; they are labelled
“settings were tuned here, results are optimistic”. No development item is pooled
with evaluation. The evaluation pilot is labelled `n=7 images, 3 scenes`; the box
analysis uses a separate annotation-referenced set of 22 images, 36 boxes, and 5
scenes.

## Evaluation results

Errors below are vehicles per image. Signed error is prediction minus reference;
absolute error is the absolute value of that error. Intervals are scene-cluster
bootstrap 95% intervals. The first-pass tables show every registered model; the
separate machine-readable summary also contains second-pass sensitivity results.
All first-pass entries below use 7 images and 3 scenes.

### First-pass signed error (95% interval)

| Target | YOLOv8n | YOLOv8s | RT-DETR-L | Fine-tuned |
| --- | ---: | ---: | ---: | ---: |
| Rickshaw any | -2.0000 [-3.0000, -1.0000] | -2.0000 [-3.0000, -1.0000] | -2.0000 [-3.0000, -1.0000] | -1.2857 [-2.5000, -0.6667] |
| Covered total | -2.7143 [-6.0000, 1.0000] | -1.8571 [-5.0000, 0.5000] | 4.0000 [3.0000, 5.0000] | -6.0000 [-9.5000, -1.5000] |
| All-vehicle total | -4.7143 [-9.0000, -1.5000] | -3.8571 [-8.0000, -2.0000] | 2.0000 [0.0000, 3.0000] | -7.2857 [-10.5000, -4.0000] |
| Car | -1.1429 [-3.5000, 1.5000] | -1.0000 [-2.5000, 0.5000] | 2.4286 [1.5000, 3.5000] | -3.5714 [-7.0000, 0.0000] |
| Bus | 0.0000 [-0.5000, 0.5000] | 0.0000 [-0.5000, 0.3333] | 0.8571 [0.5000, 1.3333] | -0.4286 [-0.6667, 0.0000] |
| Truck | -0.2857 [-1.0000, 0.0000] | 0.1429 [0.0000, 0.3333] | 1.2857 [0.6667, 2.5000] | -0.7143 [-1.5000, 0.0000] |
| Motorcycle | -1.2857 [-2.0000, 0.0000] | -1.0000 [-2.0000, 0.0000] | -0.5714 [-2.0000, 0.5000] | -1.2857 [-2.0000, 0.0000] |

### First-pass absolute error (95% interval)

| Target | YOLOv8n | YOLOv8s | RT-DETR-L | Fine-tuned |
| --- | ---: | ---: | ---: | ---: |
| Rickshaw any | 2.0000 [1.0000, 3.0000] | 2.0000 [1.0000, 3.0000] | 2.0000 [1.0000, 3.0000] | 1.8571 [0.6667, 3.0000] |
| Covered total | 3.2857 [1.0000, 6.0000] | 2.1429 [0.5000, 5.0000] | 4.8571 [4.6667, 5.0000] | 6.0000 [1.5000, 9.5000] |
| All-vehicle total | 4.7143 [1.5000, 9.0000] | 3.8571 [2.0000, 8.0000] | 3.7143 [2.5000, 4.3333] | 7.2857 [4.0000, 10.5000] |
| Car | 2.0000 [1.3333, 3.5000] | 1.2857 [0.5000, 2.5000] | 2.4286 [1.5000, 3.5000] | 3.5714 [0.0000, 7.0000] |
| Bus | 0.2857 [0.0000, 0.5000] | 0.2857 [0.0000, 0.5000] | 0.8571 [0.5000, 1.3333] | 0.4286 [0.0000, 0.6667] |
| Truck | 0.5714 [0.0000, 1.0000] | 0.4286 [0.0000, 1.0000] | 1.5714 [1.0000, 2.5000] | 0.7143 [0.0000, 1.5000] |
| Motorcycle | 1.2857 [0.0000, 2.0000] | 1.0000 [0.0000, 2.0000] | 1.1429 [0.5000, 2.0000] | 1.2857 [0.0000, 2.0000] |

### Paired comparison with YOLOv8n

The difference is fine-tuned absolute error minus YOLOv8n absolute error. Counts are
the number of evaluation images improved / worsened / unchanged. A flag is supported
only when the interval is strictly above zero.

| Target | Mean paired difference (95% interval) | Improved / worsened / unchanged | Regression flag |
| --- | ---: | ---: | --- |
| Covered total | 2.7143 [0.5000, 3.6667] | 0 / 4 / 3 | Yes |
| Car | 1.5714 [-1.5000, 3.5000] | 1 / 3 / 3 | No; point estimate increased |
| Bus | 0.1429 [-0.5000, 0.6667] | 1 / 2 / 4 | No; point estimate increased |
| Truck | 0.1429 [0.0000, 0.5000] | 0 / 1 / 6 | No; interval touches zero |
| Motorcycle | 0.0000 [0.0000, 0.0000] | 0 / 0 / 7 | No difference in this sample |
| Rickshaw any | -0.1429 [-0.3333, 0.0000] | 1 / 0 / 6 | No |

The covered-total interval is strictly positive. Car, bus, and truck have higher
point estimates of absolute error, but their intervals do not exclude zero (the truck
interval touches zero). Motorcycle error is unchanged in this small sample; this does
not make the one-class detector capable of detecting motorcycles.

### Second-pass sensitivity

Second-pass sensitivity also has 7 images and 3 scenes. The following table compares
the fine-tuned model with YOLOv8n. It is an alternate label pass, not an additional
sample and not pooled with the first pass.

| Target | Model | Signed error (95% interval) | Absolute error (95% interval) |
| --- | --- | ---: | ---: |
| Rickshaw any | YOLOv8n | -1.8571 [-2.5000, -1.0000] | 1.8571 [1.0000, 2.5000] |
| Rickshaw any | Fine-tuned | -1.1429 [-2.5000, -0.5000] | 1.7143 [0.6667, 2.5000] |
| Covered total | YOLOv8n | -3.1429 [-7.0000, 1.0000] | 3.7143 [1.0000, 7.0000] |
| Covered total | Fine-tuned | -6.4286 [-10.5000, -1.5000] | 6.4286 [1.5000, 10.5000] |
| All-vehicle total | YOLOv8n | -5.0000 [-9.5000, -1.5000] | 5.0000 [1.5000, 9.5000] |
| All-vehicle total | Fine-tuned | -7.5714 [-11.0000, -4.0000] | 7.5714 [4.0000, 11.0000] |
| Car | YOLOv8n | -1.5714 [-4.5000, 1.5000] | 2.7143 [1.5000, 4.5000] |
| Car | Fine-tuned | -4.0000 [-8.0000, 0.0000] | 4.0000 [0.0000, 8.0000] |
| Bus | YOLOv8n | 0.0000 [-0.5000, 0.5000] | 0.2857 [0.0000, 0.5000] |
| Bus | Fine-tuned | -0.4286 [-0.6667, 0.0000] | 0.4286 [0.0000, 0.6667] |
| Truck | YOLOv8n | -0.2857 [-1.0000, 0.0000] | 0.5714 [0.0000, 1.0000] |
| Truck | Fine-tuned | -0.7143 [-1.5000, 0.0000] | 0.7143 [0.0000, 1.5000] |
| Motorcycle | YOLOv8n | -1.2857 [-2.0000, 0.0000] | 1.2857 [0.0000, 2.0000] |
| Motorcycle | Fine-tuned | -1.2857 [-2.0000, 0.0000] | 1.2857 [0.0000, 2.0000] |

### Development results (kept separate)

These are not combined with evaluation and are labelled “settings were tuned here,
results are optimistic”. They cover 4 images and 3 scenes; there were no eligible
development items with a second pass. These values are included to keep the partition
visible, not to claim independent performance.

| Target | YOLOv8n absolute error (95% interval) | Fine-tuned absolute error (95% interval) |
| --- | ---: | ---: |
| Rickshaw any | 2.7500 [0.0000, 9.0000] | 2.2500 [0.5000, 7.0000] |
| Covered total | 3.5000 [2.0000, 4.5000] | 8.0000 [4.0000, 10.0000] |
| All-vehicle total | 6.2500 [3.0000, 11.0000] | 9.7500 [3.0000, 17.0000] |
| Car | 0.5000 [0.0000, 1.0000] | 4.0000 [1.0000, 7.0000] |
| Bus | 2.0000 [0.0000, 3.0000] | 2.0000 [0.0000, 3.0000] |
| Truck | 0.5000 [0.0000, 1.0000] | 0.5000 [0.0000, 1.0000] |
| Motorcycle | 1.5000 [0.0000, 3.0000] | 1.5000 [0.0000, 3.0000] |

## Annotation-referenced box analysis

This analysis is labelled **annotation referenced, unverified; may be contaminated**.
The evaluation-subset images with known training overlap were excluded. The archive
source image was linked by a unique project perceptual-hash match, and all 22 linked
images had matching image and annotation dimensions. At IoU 0.5, class-agnostic,
one-to-one matching, there were 36 annotated rickshaw boxes in 5 scenes:

| Model | Rickshaw | Car | Motorcycle | Truck | Bus | Other | Missed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n | 0/36 | 12/36 | 3/36 | 1/36 | 2/36 | 0/36 | 18/36 |
| Fine-tuned | 12/36 | 0/36 | 0/36 | 0/36 | 0/36 | 0/36 | 24/36 |

These are annotation-referenced box matches, not a human-verified detector benchmark.
They must not be interpreted as generalization or as a replacement for the audited
count evaluation.

## Limitations and remaining provenance

* The audited evaluation has only 7 images across 3 scenes after contamination
  exclusion; intervals are unstable at this sample size.
* The archive licence, authorship, source URL, download date, and paper citation were
  not supplied. Dataset redistribution or a provenance-complete citation claim is not
  supported.
* The model merges rickshaw types and cannot distinguish loader rickshaws.
* Fine-tuning a one-class detector removes the pretrained model's other class outputs;
  the paired evaluation shows increased covered-total error in this pilot. Transfer
  beyond the training scenes is unknown.
* The test scenes are Dhaka imagery. This result is not a claim about local roads or
  other geographies.

## Reproducible output files

* `configs/finetuned_class_mapping.json`: class map and `rickshaw_any` definition.
* `configs/finetuned_settings.json`: frozen inference settings and base-settings hash.
* `results/finetuned_summary.json`: contamination, first- and second-pass aggregate
  metrics, paired intervals, and partition sizes.
* `results/finetuned_evaluation.csv` and
  `results/finetuned_second_pass_evaluation.csv`: eligible evaluation rows only.
* `results/finetuned_development.csv`: development rows, reported separately.
* `results/finetuned_box_analysis.json`: aggregate annotation-referenced box outcomes.
