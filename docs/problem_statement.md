# Problem Statement

## The problem

Counting vehicles on a road is used for traffic studies, planning, and enforcement
support. Doing it by hand from images or video is slow, inconsistent, and hard to audit.
Off the shelf detectors produce a number quickly, but that number carries no indication of
how reliable it is. In local road scenes, dense traffic, occlusion, night lighting, glare,
small distant vehicles, and vehicle types that no public dataset covers all push the raw
count away from the truth. The real problem is not producing a count. It is producing a
count together with an honest sense of how much it can be trusted.

## Users

* Traffic analysts and planners who need repeatable counts across many scenes.
* Interns and field researchers running a bounded study without a machine learning team.
* Reviewers who must decide, quickly, which counts are safe to use and which need a human
  to look again.

## Why uncertainty matters in counting

A count is only actionable when its error is understood. Two images can both return a count
of 40 cars while one is accurate and the other is badly wrong because of heavy occlusion at
night. Without a reliability signal, users treat every count as equally trustworthy, which is
worse than no number at all because it creates false confidence. Exposing uncertainty lets a
reviewer spend attention where the model is likely wrong, supports fair comparison between
detectors, and turns the system from a black box into a measurable instrument. This project
treats the reliability estimate as a first class output, not an afterthought.

## Scope

Week 1 delivers a working single image vehicle counter with class filtering, a confidence
threshold, annotated output, JSON and CSV export, input validation, a clean multi tab
dashboard shell, documentation, and a scaffold for a local road scene test set. Later weeks
add multiple detectors, threshold sweeps, sliced inference, video counting with tracking, a
learned error predictor, a needs review flag, stored history, and benchmark evaluation. The
target classes are the COCO vehicle categories car, motorcycle, bus, and truck.

## Exclusions

* No training or fine tuning of any detector. Only pretrained models are used.
* No rickshaw detection is claimed. Rickshaws have no COCO class and are handled only as a
  documented gap and as ground truth labels, never as a detector output class.
* No number plate reading and no identification of individuals.
* No live camera features. Inputs are uploaded images and, later, uploaded video clips.
* No cloud service or account. The app runs locally.
