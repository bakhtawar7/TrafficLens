# Rickshaw detector fine-tuning

The archive at `C:\Users\bakht\Downloads\archive.zip` is used as a separate,
user-authorized training source. Fine-tuning produces an additional one-class
YOLOv8n detector; it does not change the existing model registry, class mapping,
frozen settings, benchmark labels, or cached detections.

## Training-only class mapping

The single target class, `rickshaw`, combines the existing dataset taxonomy
categories `rickshaw` and `loader_rickshaw`. This follows
`core.dataset.LABEL_TO_CATEGORY`, so the source aliases include `rickshaw`,
`three wheelers (CNG)`, `auto rickshaw`, `human hauler`, and `wheelbarrow`.
This mapping is local to the fine-tuning dataset and is not applied to the
existing detector classes.

## Dataset preparation

Run:

```powershell
python scripts\prepare_rickshaw_dataset.py --archive C:\Users\bakht\Downloads\archive.zip
```

The script writes to the ignored `runs\rickshaw_finetune\dataset` directory and
refuses to overwrite an existing output. It reads annotated images only from the
archive's `train` folder. XML boxes are checked against decoded image dimensions
and converted to normalized YOLO labels. Invalid XML and images with malformed
or out-of-bounds boxes are excluded and recorded in `rejected.csv`; images with
no target-class boxes remain as negative examples.

The archive has no annotations under its `test` folders. The `Pias` source group
is therefore held out as validation, and the repository's existing perceptual
hash duplicate rule keeps duplicate-connected training images in the same split.
Images in a component that matches either archive test folder are excluded from
training. The generated `summary.json` records the archive hash and split
counts. The test folders remain sealed and are not used as labeled evaluation
data.

## Training and pilot status

Results are a pilot on Dhaka imagery. Human reference counts have a mean absolute
total-count difference of 1.62 vehicles between passes across 13 paired images and 7
scenes (95% scene-bootstrap interval 0.67 to 2.46). The fine-tuned mode is a separate
optional model with its own evaluation and contamination exclusion.

Dataset preparation is implemented and tested. The initial 100-epoch CPU run was
stopped early; a quick three-epoch experiment was followed by a ten-epoch
continuation using a 0.1 training fraction, batch size 1, and 640-pixel images on
CPU. The current checkpoint is
`runs\rickshaw_finetune\yolov8n_rickshaw_plus10\weights\best.pt`. It remains an
experimental checkpoint, not a production-ready model.

The checkpoint is integrated only as a separate, off-by-default Image-tab mode.
Its contamination check and small audited pilot are documented in
`rickshaw_model.md`; those results are not independent evidence of generalization.
The evaluation explicitly excludes known overlaps and keeps development results
separate. Training-data licence and citation information were not included in the
supplied task input and remain unresolved.

After preparation, a GPU-enabled training environment can start the standard
Ultralytics training run with:

```powershell
yolo detect train model=yolov8n.pt data=runs\rickshaw_finetune\dataset\dataset.yaml project=runs\rickshaw_finetune name=yolov8n_rickshaw device=0
```

The command uses the trainer's default schedule and the separate dataset. Any
future experiment must preserve the test labels and frozen settings and use a
fresh, uncontaminated, independently labelled holdout for evaluation.
