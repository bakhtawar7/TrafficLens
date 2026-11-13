# TrafficLens demo run sheet (3–5 minutes)

**Status:** script only. No demo recording or screenshots have been created. The current
tracked `data/samples/` folder contains only a placeholder; suitable demo images must be
added with verified source and licence information before this run sheet can be completed.

All results are a pilot on Dhaka imagery. Reference counts are human counts with a
documented mean absolute pass difference of 1.62 vehicles (95% scene-bootstrap interval
0.67–2.46; 13 paired images across 7 scenes). The optional fine-tuned rickshaw mode is a
separate one-class model with its own evaluation and contamination exclusions. Do not make
accuracy claims. The reliability flag must be described as not yet validated when the
validated configuration is unavailable.

## Run sheet

| Time | Demonstration | What to say / show |
| --- | --- | --- |
| 0:00–0:35 | Open app and orient viewer | Explain that this is a Dhaka imagery pilot, not a production traffic authority system. |
| 0:35–1:05 | Normal image count and range | Use an approved easy daytime sample. Show the annotated image, per-class count, and count range. Explain that model outputs can disagree and range is not a confidence interval or guarantee. |
| 1:05–1:35 | Reliability status | Show the reliability tab. If the current state is not validated, say exactly that. Explain that agreement is between models, not ground truth. |
| 1:35–2:05 | High-disagreement case | Use an approved dense or difficult sample with known model disagreement. Do not show evaluation-image per-item results in a public demo. |
| 2:05–2:40 | Rickshaw-aware mode | Use a licensed rickshaw scene. Explain that the fine-tuned detector is one-class and separate; it does not replace pretrained multi-class counts. Mention separate evaluation and contamination exclusions. |
| 2:40–3:25 | Human correction | Select a run in the blind review queue, re-upload the exact image, enter reviewer counts while blind, reveal, and save a correction. Explain that each save appends a record but the run can still be deleted from local history; this does not alter LABELS.csv or model settings. |
| 3:25–3:50 | History | Show stored filename/hash metadata, filters, details and CSV export. Explain that image bytes are not stored. |
| 3:50–4:20 | Limitations | Highlight small Dhaka pilot, human label variation, local vehicle class limitations, separate fine-tuned model, and unevaluated video. |

## Pre-demo checklist

- [ ] Add 8–10 approved samples with source, author, licence, download date and selection reason.
- [ ] Include an easy day scene, dense scene, low-light scene, rickshaw scene and a deliberate
      failure case; keep development examples separate from evaluation examples.
- [ ] Confirm model weights are already available and warm the models before the audience
      arrives. Do not rely on an internet connection during the live demo.
- [ ] Confirm the app starts and each selected sample is readable.
- [ ] Confirm the reliability flag wording is honest and no validation claim is overstated.
- [ ] Confirm the review image hash matches the stored run.
- [ ] Confirm the history database contains only demo-safe runs and no sensitive filenames.
- [ ] Prepare fallback screenshots and cached outputs only after they have actually been
      generated from the approved demo samples.
- [ ] Do not show per-image model counts for evaluation images.
- [ ] Do not claim that the demo establishes model accuracy or that a hypothesis is supported.

## Fallback plan

If model loading or inference fails, explain the failure and switch to screenshots or cached
outputs generated in advance from the same approved samples. Label cached results with the
sample filename, model, settings provenance, and generation date. Do not invent substitute
counts or present stale results as live inference.
