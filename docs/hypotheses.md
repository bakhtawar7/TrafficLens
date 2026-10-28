# Hypotheses

Each hypothesis is written so it can be tested against the local road scene test set once
the relevant week is built. For every hypothesis the metric and the rejection condition are
stated up front. Rejecting a hypothesis is a valid, reportable outcome, not a failure.

## H1. A learned error predictor ranks inputs by true error better than a mean confidence baseline

*Claim.* A supervised model that maps detection features to counting error will order inputs
from most to least accurate better than simply ranking inputs by their mean detection
confidence.

*Metric.* Spearman rank correlation between the predicted error score and the actual absolute
count error over the test set, plus the fraction of the worst 20 percent of inputs captured at
a fixed review budget.

*Reject when.* The learned predictor's Spearman correlation is not greater than the mean
confidence baseline's correlation by a clear margin (an improvement of at least 0.1, confirmed
with a bootstrap confidence interval that excludes zero difference), or when it captures no
more of the worst inputs than the baseline at the same budget.

## H2. Cross architecture disagreement correlates with error more than within family disagreement

*Claim.* Disagreement between different model families, for example YOLOv8n against RT-DETR-L,
is a stronger signal of counting error than disagreement within the same family, for example
YOLOv8n against YOLOv8s.

*Metric.* Pearson or Spearman correlation, on held out items, between each disagreement measure
and the actual absolute count error. Disagreement is expressed as the normalised difference in
per class counts.

*Reject when.* The cross architecture disagreement correlation is not higher than the within
family disagreement correlation, or when the difference is not significant under a bootstrap
interval.

## H3. Local vehicle types are misclassified or missed systematically

*Claim.* Vehicle types common in local scenes but absent from COCO, above all rickshaws and
loader rickshaws, are either missed entirely or absorbed into another class at a rate far above
their share of the scenes.

*Metric.* Per type recall and confusion counts against ground truth. Report the share of ground
truth rickshaws and loader rickshaws that produce no matching detection, and the share absorbed
as car or truck.

*Reject when.* Recall for these types is comparable to recall for covered classes, meaning the
miss rate gap is small and not consistent across the scenes.

## H4. Sliced inference reduces distant vehicle error

*Claim.* Splitting a large image into overlapping tiles and running the detector on each tile
reduces counting error for distant and small vehicles compared with a single full frame pass.

*Metric.* Absolute count error on items tagged `distant`, and small object recall, compared
between tiled inference and a single full frame pass at the same confidence threshold.

*Reject when.* Tiled inference does not lower the mean absolute error on distant items, or when
any gain is offset by a rise in double counting that leaves the overall error flat or worse.
