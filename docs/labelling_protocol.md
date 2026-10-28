# Labelling Protocol

Ground truth labels let Week 4 measure how far a TrafficLens count can be trusted. This
protocol makes every counting decision concrete so two labellers reach the same number.
Labels go in `data/testset/LABELS.csv`. TrafficLens never claims to detect rickshaws
because they have no COCO class, but a labeller still records them as ground truth so the
gap can be measured honestly.

## Vehicle categories

Record a count for each category, using zero when none are present:

* `car`: passenger cars, including hatchbacks, sedans, SUVs, and vans used as cars.
* `bus`: public or private buses and coaches.
* `truck`: goods vehicles, lorries, pickups used for cargo, and tankers.
* `motorcycle`: motorcycles and scooters, with or without a rider.
* `rickshaw`: auto rickshaws and cycle rickshaws.
* `loader_rickshaw`: small motorised load carriers and three wheel cargo vehicles.
* `other`: any vehicle that fits none above, for example tractors, construction plant, or
  emergency vehicles. Note the specifics in the `notes` column.

Vehicles are counted as objects, not as occupants or trips.

## What counts as a vehicle

A vehicle is counted when all of these hold:

* It is a real road vehicle, not a sign, mural, or billboard image of one.
* It is not permanently fixed as scenery, for example a car mounted as a monument.

## Partially visible and occluded vehicles

* Count a vehicle that is partly out of frame as long as enough is visible to identify its
  category with confidence.
* Count a vehicle hidden behind another vehicle, a barrier, or foliage when you can
  reasonably tell that it is a separate vehicle. Use judgement: a wheel and a roof behind a
  truck is one occluded car, but a sliver too ambiguous to classify is not counted.
* Do not guess vehicles you cannot see any evidence of.

## Minimum visible size rule

* Count a vehicle whose longest visible dimension is at least 12 pixels in the image as
  stored. Below 12 pixels, treat it as background detail and exclude it, even if you can
  tell it is a vehicle.
* If the frame is very large and distant traffic is tiny, note this in `notes` and tag the
  item `distant` in `SOURCES.csv`.

## Parked vehicles

* A parked or stationary vehicle on the roadway, shoulder, or roadside is counted like any
  other, because it is physically present in the scene.
* Vehicles inside a fully enclosed showroom, garage, or a parking structure not visible from
  the road are not counted.
* Consistency matters more than philosophy: if one labeller counts roadside parked cars, all
  labellers do.

## Recording a label

For each item, fill one `LABELS.csv` row with `item_id`, the seven category counts, the
`labeller_pass` value, and optional `notes`. Set `labeller_pass` to `1` for the first pass
and `2` for the review pass described below.

## Second pass procedure

1. Finish a full first pass over all items.
2. Draw a random 20 percent of the items for review, rounded up so at least one item is
   always reviewed. Keep the sample spread across condition tags rather than only easy
   scenes.
3. Re label each sampled item from scratch, without looking at the first pass number, and
   record it as a `2` in `labeller_pass`.
4. Compare the two passes. For any category where they differ by more than one vehicle, or
   by more than 10 percent, adjudicate a final agreed number and write it as the authoritative
   row, noting the disagreement in `notes`.
5. Keep both pass rows so Week 4 can report labeller agreement as a quality signal.

The second pass exists to catch drift and ambiguity, not to overwrite quickly. When in
doubt, apply the rules above and document the decision.
