# Usability test plan

TrafficLens — pilot usability evaluation for a first-time user.

Scope: image counting workflow, threshold control, reliability signals, rickshaw-aware mode,
and the human review loop. Video counting is out of scope for this session.
Results are for local improvement only; no personal data is collected.

---

## Session overview

| Item | Detail |
|---|---|
| Duration | 20 minutes |
| Participants | Individual sessions; aim for 5 participants to surface recurring patterns |
| Format | Think-aloud with one observer |
| Equipment | Laptop with the app running at `http://localhost:8501`; 3 sample images from `data/samples/` on the desktop |
| Consent | Verbal; see consent note at the end of this document |
| Recording | Paper observation sheet only; no audio or screen recording without explicit written consent |

---

## Task set

Each task is given to the participant as a printed card. The observer reads it aloud
if the participant asks. The observer never answers "how to" questions; they may
re-read the task.

### Task 1 — Upload an image and read the count

> Open the TrafficLens app. Upload the image called `easy_day.jpg` from the desktop.
> Tell me the total number of vehicles the app found.

Success criterion: participant reads the correct total from the metric tile in the Image tab.  
Failure: participant reads from the wrong field or cannot find the total.  
Time limit: 3 minutes.

### Task 2 — Change the threshold and explain the change

> Move the confidence threshold slider to 0.50. What changed in the count?
> In your own words, what does the threshold control do?

Success criterion: participant moves the slider and can describe the threshold as a
filter on detection confidence (more words allowed; the key idea is that higher threshold
removes lower-confidence detections).  
Failure: participant cannot locate the slider, or cannot explain the effect.  
Time limit: 3 minutes.

### Task 3 — Find the count range and explain it

> Click on the Reliability tab. Tell me the count range that is shown. What do you think
> it means?

Success criterion: participant reads the min–max range and can say it represents the spread
across the models or the spread across thresholds.  
Failure: participant cannot find the range, or describes it as an error bar on accuracy.  
Time limit: 3 minutes.

### Task 4 — Find the reliability status and explain it

> Still on the Reliability tab, find the reliability flag or status. What does it say?
> What do you think it means in practice?

Success criterion: participant finds the flag status and can say it describes whether the
run was flagged for review (not a trust verdict).  
Failure: participant cannot find the status, or describes it as a quality score.  
Time limit: 2 minutes.

### Task 5 — Turn on rickshaw-aware mode and describe what changed

> Go back to the Image tab. Find the rickshaw-aware mode toggle and turn it on.
> What changed?

Success criterion: participant locates the toggle and can describe the separate rickshaw
detection output (separate model, separate colour, does not change the main count).  
Failure: participant cannot find the toggle, or believes it improved the main count.  
Time limit: 2 minutes.

### Task 6 — Review a flagged run and enter a corrected count

> Click on the History tab, then open the Human review sub-tab. Find the first run in the
> review queue. Enter your own count for the vehicles in that image, then click Reveal to
> compare it to the model count.

Success criterion: participant navigates to the review queue, enters any non-negative count
for each category, and clicks Reveal without guidance.  
Failure: participant cannot find the review queue, or cannot submit a correction.  
Time limit: 4 minutes.

### Task 7 — Upload a video, place the line, and read the totals

> Click on the Video tab. Upload the video called `demo_clip.mp4` from the desktop.
> Place the counting line horizontally across the middle of the frame using the preset.
> Click "Process video" and read the inbound and outbound totals.

Success criterion: participant uploads the video, selects the horizontal mid-frame preset,
clicks Process video, and reads the inbound and outbound counts from the metric tiles.  
Failure: participant cannot locate the uploader, cannot find the preset, or cannot read the
totals after processing.  
Time limit: 4 minutes.

---

## Observer sheet

Print one copy per participant. Record observations by task.

```
Participant ID (anonymous): ____________  Date: ____________  Observer: ____________

Task 1 — Upload and read count
  Completed without help [ ]   Completed with observer re-read [ ]   Did not complete [ ]
  Time (seconds): ______
  Errors observed: _______________________________________________
  Comments: _____________________________________________________

Task 2 — Threshold change
  Completed without help [ ]   Completed with observer re-read [ ]   Did not complete [ ]
  Time (seconds): ______
  Errors observed: _______________________________________________
  Comments: _____________________________________________________

Task 3 — Count range
  Completed without help [ ]   Completed with observer re-read [ ]   Did not complete [ ]
  Time (seconds): ______
  Errors observed: _______________________________________________
  Comments: _____________________________________________________

Task 4 — Reliability flag
  Completed without help [ ]   Completed with observer re-read [ ]   Did not complete [ ]
  Time (seconds): ______
  Errors observed: _______________________________________________
  Comments: _____________________________________________________

Task 5 — Rickshaw-aware mode
  Completed without help [ ]   Completed with observer re-read [ ]   Did not complete [ ]
  Time (seconds): ______
  Errors observed: _______________________________________________
  Comments: _____________________________________________________

Task 6 — Review and correction
  Completed without help [ ]   Completed with observer re-read [ ]   Did not complete [ ]
  Time (seconds): ______
  Errors observed: _______________________________________________
  Comments: _____________________________________________________

Task 7 — Video upload, line placement, totals
  Completed without help [ ]   Completed with observer re-read [ ]   Did not complete [ ]
  Time (seconds): ______
  Errors observed: _______________________________________________
  Comments: _____________________________________________________

Overall observations and quotes:
________________________________________________________________
________________________________________________________________

Post-session SUS score (see SUS section below): ______
```

---

## System Usability Scale (SUS)

After completing the tasks, ask the participant to complete the standard 10-item SUS
questionnaire. Use a 1 (strongly disagree) to 5 (strongly agree) Likert scale for
each item.

> Source: Brooke, J. (1996). SUS: A "quick and dirty" usability scale. In P. W. Jordan,
> B. Thomas, B. A. Weerdmeester, & A. L. McClelland (Eds.), *Usability evaluation in
> industry* (pp. 189–194). Taylor & Francis.

Items (read as written; do not paraphrase):

1. I think that I would like to use this system frequently.
2. I found the system unnecessarily complex.
3. I thought the system was easy to use.
4. I think that I would need the support of a technical person to be able to use this
   system.
5. I found the various functions in this system were well integrated.
6. I thought there was too much inconsistency in this system.
7. I would imagine that most people would learn to use this system very quickly.
8. I found the system very cumbersome to use.
9. I felt very confident using the system.
10. I needed to learn a lot of things before I could get going with this system.

Scoring: for odd-numbered items subtract 1 from the response; for even-numbered items
subtract the response from 5. Sum all 10 scores and multiply by 2.5 to get a score out
of 100. A score above 68 is considered above average usability.

---

## Consent note

Participation is voluntary, anonymous, and may stop at any time. No audio or screen
recording is made without explicit prior written consent. No personal data (name, contact
details, demographic information) is collected or stored. The session takes about 20 minutes.
Observations recorded on the paper sheet are used only to improve the TrafficLens
interface and are stored locally with no identifying information. You may withdraw at
any point without consequence.

---

## Debrief note for the facilitator

After the session, tell the participant: "Thank you. The app counts vehicles in traffic
images using object detection models. The count range and agreement scores show how much
the models agree with each other, not how close the count is to ground truth. The human
review loop lets you add your own counts for comparison." Answer questions openly.

---

## What to run after sessions are complete

If `results/usability/results_filled.csv` exists (at least one completed session), run:

```text
python scripts/analyse_usability.py results/usability/results_filled.csv
```

That script does not yet exist. Until it does, tabulate the SUS scores and task completion
rates from the filled CSV by hand and record them in `docs/usability_findings.md`.
Report the sample size (n), mean SUS score with its range, task completion rate per task,
and any qualitative theme that appeared in at least two participant comments. Do not
interpolate or generalise from fewer than 3 completed sessions.
