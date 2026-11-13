# Demo script

A 3–5 minute run sheet for demonstrating TrafficLens to a first-time audience.
Timestamps are approximate; adjust to your machine's speed.

No accuracy claims are made at any point. All results are described as pilot
outputs on Dhaka imagery.

---

## Pre-demo checklist

- [ ] Virtual environment activated: `.venv\Scripts\Activate.ps1` (Windows) or
  `source .venv/bin/activate` (Linux/macOS)
- [ ] App running: `streamlit run app/main.py` — browser open at `http://localhost:8501`
- [ ] Sample images ready: copies of `data/samples/` images (or any street images) on
  the desktop or in Downloads
- [ ] A short video clip (≤ 60 s, street-level, ideally with crossing traffic) on the
  desktop — your own footage or a CC0 clip
- [ ] Fallback: screenshots in `docs/screenshots.md` and cached results in
  `data/history.db` if live processing is slow

---

## Segment 1 — Normal image count and range (0:00–1:00)

1. Open the **Image** tab.  Show the empty state: "Upload a JPG, JPEG, PNG, or WEBP
   image in the sidebar."
2. Upload a **day, medium-density** sample image from the desktop.
3. Point out: original image on the left, annotated image on the right, metric tiles
   below.  Read the total vehicle count and per-class breakdown aloud.
4. Move to the **Reliability** tab.  Point out the count range (min–max across
   models) and the two agreement scores.  Say: "These are measured between the
   models, not against ground truth."
5. Show the flag status.  Say: "This run has not been validated; the flag is
   informational only, not a verdict on whether the count is correct."

---

## Segment 2 — Threshold change (1:00–1:30)

1. Return to the sidebar.  Move the confidence threshold from 0.25 to 0.50.
2. Watch the count update.  Explain: "Higher threshold removes lower-confidence
   detections; the count usually falls.  Whether the lower or higher setting is
   more correct depends on the scene and has not been evaluated here."

---

## Segment 3 — High-disagreement case and reliability status (1:30–2:00)

1. Upload a **dense or night** sample image.
2. On the Reliability tab, show a wider count range and lower agreement scores.
3. Say: "The models disagree more on this image.  The flag may be set to 'needs
   review', which puts it at the top of the review queue.  It does not mean the
   count is wrong; it means there is more uncertainty."

---

## Segment 4 — Rickshaw-aware toggle (2:00–2:30)

1. Upload a **rickshaw scene** sample image (or any image).
2. On the Image tab, scroll to the toggle "Rickshaw aware mode (fine tuned)" and
   turn it on.
3. Point out the separate rickshaw detection box and the `Rickshaw count (fine tuned)`
   metric.  Say: "This is a separate one-class pilot model.  It does not change the
   pretrained count or reliability signals above.  It was fine-tuned on a user-supplied
   archive whose licence is unverified; 49 of 75 test images overlap training data and
   are excluded from its evaluation."
4. Turn the toggle off.  Confirm the pretrained counts are unchanged.

---

## Segment 5 — Video upload, line placement, and totals (2:30–3:30)

1. Click the **Video** tab.
2. Read the honesty note aloud: "Video results are a demonstration, not an evaluation,
   and carry no accuracy claim."
3. Click "Upload a video file" and upload your demo clip.
4. The tab shows resolution, frame rate, and duration.
5. Under **Counting line**, select the "Horizontal mid-frame" preset.  The first frame
   appears with the yellow line overlaid.  Optionally rename the directions.
6. Click **Run view check** — the app samples 3 frames and shows how many vehicles it
   found.  If zero: explain that the angle or scene may not suit these models.
7. Click **Process video**.  Show the progress bar with frames/s and estimated time
   remaining.
8. When done: read the inbound and outbound totals, the track count, and the ID
   fragment count.  Say: "Identity fragments mean the tracker lost and re-acquired a
   vehicle, which can cause a double count or a miss."
9. Click **Download annotated video** and open it to show the annotated output.

---

## Segment 6 — Human correction through the review loop (3:30–4:15)

1. Click the **History** tab → **Human review** sub-tab.
2. Show the review queue.  Explain: "Flagged runs appear first.  The order within
   flagged is by model disagreement."
3. Select the first run.  Show that model counts and the count range are hidden until
   reveal.
4. Upload the matching image (must have the same SHA-256).  Enter a count for each
   vehicle category.  Click **Reveal model results for comparison**.
5. Show the signed-error table: how far each model was from the corrected count.  
   Say: "This is not a measurement of model quality; it is one reviewer's count on
   one image.  It is never written to LABELS.csv or used to change any model."
6. Click **Save correction (append-only)**.

---

## Segment 7 — History tab (4:15–4:45)

1. Switch to the **Run history** sub-tab.
2. Show the filterable table of runs.  Open one run's detail view.
3. Click **Export history as CSV**.  Say: "This exports filenames, hashes, counts,
   and reliability metadata.  No image bytes are stored."
4. If time allows, show the clear-history confirmation flow.

---

## Limitations (4:45–5:00)

Point to the **Limitations** expander in the sidebar.  Read one or two items aloud.
Finish with: "These results are a pilot on Dhaka street imagery.  The labelling has
been done by one labeller with a measured consistency of 1.62 vehicles mean absolute
difference per image.  Hypotheses H1 and H2 are not yet tested.  The fine-tuned
model's training-data licence is unverified."

---

## Fallback plan

If the app is slow or the network is unavailable:

1. Use pre-captured screenshots listed in `docs/screenshots.md`.
2. Use runs already in `data/history.db` (History tab → Run history).
3. For the video segment: show the crossing CSV from a previous run as a download.

---

## Screenshots needed

See `docs/screenshots.md` for the full list.  The minimum set for this script:

| Screenshot | Segment |
|---|---|
| Image tab with annotated output and metrics | 1 |
| Reliability tab with count range and agreement | 1 |
| Reliability tab with high-disagreement image | 3 |
| Rickshaw mode — fine-tuned box and metric | 4 |
| Video tab — first frame with line overlay | 5 |
| Video tab — progress bar during processing | 5 |
| Video tab — results with inbound/outbound metrics | 5 |
| History tab — review queue and correction form | 6 |
| History tab — run detail view | 7 |
