# First-time usability session plan

**Status:** protocol only. No participant sessions or usability findings are claimed here.

## Purpose

Observe whether a first-time user can complete the core TrafficLens tasks and correctly explain what the displayed counts and reliability signals mean. This is a formative 20-minute session, not a benchmark of model quality.

TrafficLens results come from a pilot on Dhaka imagery. Reference counts are human counts; the currently documented consistency is a mean absolute pass difference of 1.62 vehicles (95% scene-bootstrap interval 0.67–2.46; 13 paired images across 7 scenes). The optional fine-tuned rickshaw mode is a separate model with its own evaluation and contamination status. Do not present its results as part of the pretrained model results.

## Session plan (20 minutes)

| Time | Activity | Facilitator guidance |
| --- | --- | --- |
| 0–2 min | Consent and context | Read the consent note. Explain that the interface, not the participant, is being evaluated. |
| 2–4 min | Orientation | Ask the participant to think aloud while looking at the dashboard. Do not teach the interface unless they are blocked. |
| 4–7 min | Task 1 | Upload an image and read the vehicle count. |
| 7–9 min | Task 2 | Change the confidence threshold and explain what changed. |
| 9–11 min | Task 3 | Find the count range and explain what it represents. |
| 11–13 min | Task 4 | Find the reliability status and explain it. |
| 13–16 min | Task 5 | Turn on rickshaw-aware mode and describe what changed. |
| 16–19 min | Task 6 | Review a flagged run and enter a corrected count. |
| 19–20 min | Wrap-up | Collect the ten SUS responses and one open-ended comment. |

## Participant tasks

1. **Upload and count.** Upload the supplied demonstration image. Point to the total vehicle count and explain what it counts.
2. **Threshold.** Change the confidence threshold. Explain what a higher threshold does to detections and what changed on screen.
3. **Count range.** Find the count range. Explain why different models can produce different totals and whether the range is a guarantee.
4. **Reliability status.** Find the reliability status. Explain what “not yet validated” means. The correct interpretation is that the flag is not validated as a predictor of counting error; model agreement is not ground truth.
5. **Rickshaw-aware mode.** Turn on the optional rickshaw-aware mode. Describe which separate result appears and what it does not replace. It is a one-class pilot, not a replacement for the pretrained multi-class results.
6. **Human review.** Open a flagged run, enter corrected counts while the model results remain hidden, reveal the comparison, and describe how the correction is stored. Corrections are user-reported, append-only review data; they do not change labels, model weights, or settings.

Use only approved demonstration images. Do not use evaluation images if doing so would expose per-image evaluation counts or compromise the blind review workflow.

## Observation sheet

Record observations without collecting names, contact details, or other personal data.

| Participant code | Task | Completed? | Time (seconds) | Errors / missteps | Needed help? | Exact point of hesitation or misunderstanding |
| --- | --- | --- | ---: | --- | --- | --- |
| P__ | 1 | | | | | |
| P__ | 2 | | | | | |
| P__ | 3 | | | | | |
| P__ | 4 | | | | | |
| P__ | 5 | | | | | |
| P__ | 6 | | | | | |

**Facilitator notes:** distinguish a participant's interpretation error from a system defect. Record the screen, control, wording, and action involved. Do not infer a cause without evidence.

## System Usability Scale (SUS)

After the tasks, ask the participant to rate each statement from 1 (strongly disagree) to 5 (strongly agree).

1. I think that I would like to use this system frequently.
2. I found the system unnecessarily complex.
3. I thought the system was easy to use.
4. I think that I would need the support of a technical person to be able to use this system.
5. I found the various functions in this system were well integrated.
6. I thought there was too much inconsistency in this system.
7. I would imagine that most people would learn to use this system very quickly.
8. I found the system very cumbersome to use.
9. I felt very confident using the system.
10. I needed to learn a lot of things before I could get going with this system.

Calculate SUS using the standard method: for odd-numbered items subtract 1 from the response; for even-numbered items subtract the response from 5; sum the adjusted values and multiply by 2.5. The score ranges from 0 to 100 and is not a percentage.

**Citation:** Brooke, J. (1996). “SUS: A ‘Quick and Dirty’ Usability Scale.” In P. W. Jordan, B. Thomas, B. A. Weerdmeester, and A. L. McClelland (eds.), *Usability Evaluation in Industry*, pp. 189–194. Taylor & Francis.

## Consent note

Taking part is voluntary. The session is anonymous and does not require personal data. You may skip any task or stop at any time without giving a reason. Notes will record task completion, time, interface errors, and comments only. Do not record audio, video, names, or contact details. The session evaluates the interface, not the participant.

## Reporting rules

- Do not write findings until sessions have actually been conducted and the filled results file has been reviewed.
- Report the number of participants, task completion, time, observed errors, SUS scores, and qualitative comments separately.
- Keep qualitative observations descriptive; do not turn them into unsupported claims.
- Fix usability issues only when they blocked or slowed at least two participants, plus any single blocking issue, as specified in the project brief.
- If no completed file exists, leave docs/usability_findings.md unwritten and report that sessions are still pending.
