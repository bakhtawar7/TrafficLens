# Pilot failure cases

Selection used the frozen final settings and existing cached detections only; these images were not used to tune anything. The best model was YOLOv8s, the lowest-MAE model among those with at least 10 image-hash-matched cached evaluation results. The labeller-consistency threshold was 1.6923 covered vehicles. The first selection contains 8 of up to 12 images whose best-model absolute covered error strictly exceeds that threshold. The separate disagreement selection contains 5 of up to 5 images, ranked by the range between available models' covered totals. 4 images overlap; 9 unique images are shown.

Causes are judgments from viewing the images and are **unconfirmed** until the labeller confirms them. Possible labeller miscounts are flagged for blind review; a model discrepancy alone does not establish that the label is wrong.

## Important overlay limitations

The available cache has incomplete RT-DETR-L results for selected images; figures mark missing entries rather than generating detections. The source annotation files do not match the selected image copies byte-for-byte, so source boxes are not drawn because their alignment cannot be verified. Each figure states this explicitly. Audited labels and all available model counts are displayed for reference.

## Selected images

| Image id | Condition tags | Selection | Audited pass-1 counts | Model counts (covered classes) | Likely cause | Explanation / review |
|---|---|---|---|---|---|---|
| tl0007 | day, dense, distant | best-model error, model disagreement | car=5, bus=4, truck=0, motorcycle=9, rickshaw=0, loader_rickshaw=0, other=2 | YOLOv8n: car=1, motorcycle=0, bus=4, truck=0<br>YOLOv8s: car=1, motorcycle=0, bus=4, truck=0<br>RT-DETR-L: car=7, motorcycle=0, bus=3, truck=1 | dense overlap — unconfirmed | Several foreground vehicles overlap or are partly occluded in busy traffic. Possible labeller miscount; blind recheck requested. |
| tl0038 | day, sparse | best-model error | car=2, bus=0, truck=0, motorcycle=4, rickshaw=2, loader_rickshaw=0, other=2 | YOLOv8n: car=0, motorcycle=0, bus=0, truck=0<br>YOLOv8s: car=0, motorcycle=0, bus=0, truck=0<br>RT-DETR-L: cache unavailable | distant or small — unconfirmed | The road scene is wide and multiple vehicles appear small in the distance. Possible labeller miscount; blind recheck requested. |
| tl0022 | day, dense | best-model error | car=0, bus=8, truck=1, motorcycle=0, rickshaw=0, loader_rickshaw=0, other=1 | YOLOv8n: car=0, motorcycle=0, bus=3, truck=2<br>YOLOv8s: car=0, motorcycle=0, bus=3, truck=1<br>RT-DETR-L: cache unavailable | distant or small — unconfirmed | Vehicles are spread across a broad roadway and some are small in the frame. Possible labeller miscount; blind recheck requested. |
| tl0070 | night, glare, distant | best-model error | car=12, bus=1, truck=0, motorcycle=0, rickshaw=4, loader_rickshaw=0, other=0 | YOLOv8n: car=7, motorcycle=0, bus=0, truck=0<br>YOLOv8s: car=9, motorcycle=0, bus=0, truck=0<br>RT-DETR-L: cache unavailable | dense overlap — unconfirmed | Multiple nearby road users occupy the same busy area and partly overlap. Possible labeller miscount; blind recheck requested. |
| tl0008 | day, dense | best-model error, model disagreement | car=4, bus=1, truck=2, motorcycle=2, rickshaw=3, loader_rickshaw=0, other=3 | YOLOv8n: car=4, motorcycle=4, bus=0, truck=0<br>YOLOv8s: car=1, motorcycle=4, bus=0, truck=2<br>RT-DETR-L: car=10, motorcycle=4, bus=2, truck=5 | distant or small — unconfirmed | Most vehicles are small or far down the roadway. Possible labeller miscount; blind recheck requested. |
| tl0029 | day, dense, distant | best-model error, model disagreement | car=4, bus=4, truck=1, motorcycle=2, rickshaw=1, loader_rickshaw=0, other=0 | YOLOv8n: car=3, motorcycle=2, bus=0, truck=1<br>YOLOv8s: car=3, motorcycle=3, bus=0, truck=3<br>RT-DETR-L: cache unavailable | distant or small — unconfirmed | The elevated view places many vehicles at a small apparent size. Possible labeller miscount; blind recheck requested. |
| tl0044 | night, glare, sparse, distant | best-model error | car=1, bus=1, truck=0, motorcycle=2, rickshaw=1, loader_rickshaw=0, other=0 | YOLOv8n: car=1, motorcycle=0, bus=1, truck=0<br>YOLOv8s: car=1, motorcycle=0, bus=1, truck=0<br>RT-DETR-L: cache unavailable | distant or small — unconfirmed | Several vehicles are small against the broad road and distant background. Possible labeller miscount; blind recheck requested. |
| tl0046 | dense, distant | best-model error, model disagreement | car=10, bus=1, truck=0, motorcycle=4, rickshaw=9, loader_rickshaw=0, other=5 | YOLOv8n: car=9, motorcycle=0, bus=0, truck=0<br>YOLOv8s: car=13, motorcycle=4, bus=0, truck=0<br>RT-DETR-L: cache unavailable | night or glare — unconfirmed | The dim dusk lighting reduces contrast around some vehicles. Possible labeller miscount; blind recheck requested. |
| tl0043 | night, glare, dense, distant | model disagreement | car=10, bus=1, truck=1, motorcycle=3, rickshaw=1, loader_rickshaw=0, other=0 | YOLOv8n: car=6, motorcycle=0, bus=1, truck=2<br>YOLOv8s: car=7, motorcycle=2, bus=2, truck=3<br>RT-DETR-L: cache unavailable | dense overlap — unconfirmed | Crowded traffic includes vehicles close together and partially occluded. Possible labeller miscount; blind recheck requested. |

## Cause frequency

With this few selected images, this pattern is descriptive only.

| Cause | Images |
|---|---:|
| occlusion | 0 |
| distant or small | 5 |
| night or glare | 1 |
| local vehicle type | 0 |
| annotation or labelling issue | 0 |
| dense overlap | 3 |
| other | 0 |
