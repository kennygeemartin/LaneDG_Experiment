# Benchmark data sources and preprocessing

The local experiment environment contains 420 selected road images and
original-format annotations under `datasets`, with per-file provenance and
SHA-256. Images and annotations are not committed to this repository. The
published manifests retain image IDs, relative paths, annotation sources,
sampling roles, and hashes so selections can be restored and checked. The
distribution mirrors are third parties. Official release checksums were
unavailable, so independent certification of original-release bytes is not
claimed. Do not publish the benchmark images with the code.

## TuSimple

Official format: https://github.com/TuSimple/tusimple-benchmark/blob/master/doc/lane_detection/readme.md

Mirror: https://www.kaggle.com/datasets/manideep1108/tusimple

Recover the required source JSON files with:

```powershell
.venv/Scripts/python.exe -m benchmark.acquire tusimple manideep1108/tusimple TUSimple/test_label.json --local annotations/test_label.json
.venv/Scripts/python.exe -m benchmark.acquire tusimple manideep1108/tusimple TUSimple/train_set/label_data_0313.json --local annotations/label_data_0313.json
.venv/Scripts/python.exe -m benchmark.acquire tusimple manideep1108/tusimple TUSimple/train_set/label_data_0531.json --local annotations/label_data_0531.json
.venv/Scripts/python.exe -m benchmark.acquire tusimple manideep1108/tusimple TUSimple/train_set/label_data_0601.json --local annotations/label_data_0601.json
```

Each manifest identifies its source JSON and line. Negative x samples split
polylines; no gap is bridged. Validation is reserved from labeled training;
labeled test is held out. Clip IDs are disjoint between roles. Adjacent clips
could still originate from the same longer drive; complete recording linkage
was unavailable. A perceptual-hash audit found no cross-role candidates at
Hamming distance at most four.

## CULane

Official source: https://xingangpan.github.io/projects/CULane.html

Mirror: https://www.kaggle.com/datasets/manideep1108/culane

```powershell
.venv/Scripts/python.exe -m benchmark.acquire culane manideep1108/culane CULane/list/list/train_gt.txt --local list/train_gt.txt
.venv/Scripts/python.exe -m benchmark.acquire culane manideep1108/culane CULane/list/list/val_gt.txt --local list/val_gt.txt
.venv/Scripts/python.exe -m benchmark.acquire culane manideep1108/culane CULane/list/list/test.txt --local list/test.txt
```

Original train/val/test membership is retained. MP4 directory IDs are disjoint
across selected roles. Validation groups are allocated first because original
training and validation lists can share videos. Train/val polylines come from
`annotations_new`, the corrected release identified by the official website.
Test polylines come from the corresponding test archives. Annotated occluded
portions remain foreground; empty annotations remain negative images.

Some individual-file endpoints return 404 for entries present in the ZIP.
`benchmark.index_archives` caches offsets for the frozen selections, and the
downloader recovers those exact entries through HTTP Range requests. ZIP CRC,
uncompressed length and subsequent SHA-256 are checked. Image IDs do not change.

## BDD100K

Official format: https://github.com/bdd100k/bdd100k/blob/master/doc/source/format.rst

Official legacy path viewer: https://github.com/ucbdrive/bdd100k/blob/master/bdd100k/show_labels.py

Image mirror: https://www.kaggle.com/datasets/marquis03/bdd100k

Original-format JSON mirror: https://www.kaggle.com/datasets/happytren/bdd100k-labels

The complete image archive index confirms 70,000 train, 10,000 val and 20,000
test IDs. Selection uses the complete inventories, not an API's first page.
A partial alternate mirror was rejected before sampling. The image paths are
`train/images/ID.jpg` and `val/images/ID.jpg`; matching per-image JSON is under
`100k/train/ID.json` or `100k/val/ID.json`. JSON stems must match image IDs.
An initial image-path correction is logged without changing the sample IDs.

Training uses original train. Validation and evaluation use separate labeled
val subsets; no test labels are inferred. Shared filename prefixes are grouped
conservatively across roles. Exact recording relationships remain unavailable.

Positive classes are single/double white, yellow and other longitudinal lane
markings. Curbs, crosswalks, vertical markings and explicitly ignored/occluded
lane supports are excluded from loss and metrics. Their ignored support is
rasterized at the same width as foreground. Other unannotated pixels remain
background. Dashed markings retain the continuous geometry actually annotated.

Legacy cubic command sequences such as `CCCL` are interpreted as in the official
Matplotlib viewer: the first point is a move and a cubic command consumes three
following vertices. Each cubic is sampled at 64 intervals before rasterization.
Simply joining the control vertices with straight lines would be incorrect.

## Shared protocol

Sampling seed 42 selects one random frame per random sequence group, allocating
20 validation groups, 40 evaluation groups and 80 training groups. This is
sequence-balanced sampling, not uniform image sampling. Existing selection
plans and manifests are preserved rather than silently resampled on retries.

OpenCV decodes images, converts BGR to RGB and stretches to 256x256 with bilinear
interpolation. Coordinates transform as x*256/width and y*256/height and are
rounded with numpy.rint. OpenCV LINE_8 polylines have width three output pixels.
No crop, letterbox, extrapolation or filling between lane boundaries occurs.
Per-image transformations, masks, valid-pixel maps and sampled overlays are
generated under `benchmark_artifacts/qa` when data are restored locally. Only
QA metadata and transformations are published; benchmark-derived overlay
images, masks, and valid-pixel maps are excluded. These choices define a
harmonized semantic evaluation, not an official lane-instance leaderboard
protocol.
