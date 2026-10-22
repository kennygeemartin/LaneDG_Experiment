# Exploratory cross-dataset lane segmentation

This repository contains a reproducible real-image rerun on selected TuSimple,
CULane, and BDD100K images. The evaluated model is
**LaneDG-Semantic-Reimplementation-v1**, a documented semantic adaptation
inspired by DGFBlock, C2f FPN-PAN, and EUCB. It has 1,141,793 trainable
parameters and one binary-mask output. It is **not a verified reproduction**
of the cited YOLOv8n-LaneDG instance-segmentation model.

The historical manuscript results are not included as verified benchmark
results. The local recovery audit found no image manifests or per-image
outputs tying those results to benchmark images. This does not establish how
the historical runs were produced. The public repository's prior runner used
synthetic scenes; its old model/results/figures have been removed from the
current branch to avoid presenting them as benchmark measurements. They remain
available in Git history.

## Environment

The recorded rerun used Python 3.14.7 and CPU PyTorch. The exact environment
is listed in `requirements-benchmark.lock.txt`; general dependency ranges are
in `requirements.txt`.

```powershell
uv venv .venv --python 3.14 --cache-dir .runtime/uv-cache
uv pip install --python .venv/Scripts/python.exe --cache-dir .runtime/uv-cache -r requirements-benchmark.lock.txt
.venv/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
```

## Data preparation

Benchmark images and annotations are **not** stored in this repository. Obtain
them from the documented benchmark sources or mirrors. Follow
`benchmark_artifacts/DATA_SOURCES.md`, then run:

```powershell
.venv/Scripts/python.exe scripts/restore_benchmark_data.py
.venv/Scripts/python.exe -m benchmark.prepare
.venv/Scripts/python.exe -m benchmark.qa
```

Frozen manifests under `benchmark_artifacts/manifests/` record selected image
IDs, source partitions, annotation paths, sampling metadata, and hashes. Do not
resample the selected IDs when reproducing the reported run. Images and
generated image overlays, masks, checkpoints, and per-image probability maps
are excluded from version control because they are benchmark-derived or
unnecessarily large. QA audit metadata, coordinate transforms, preprocessing
policy, selection manifests, aggregate results, and figures are included.

Each dataset uses 80 training, 20 validation, and 40 evaluation images,
selected with seed 42, one image per sampled sequence group. TuSimple
validation is held out from training; CULane's original train/validation/test
roles are retained; BDD100K validation and evaluation use disjoint subsets of
the labeled validation partition. Complete longer-drive linkage is
unavailable, and BDD filename prefixes are conservative sequence proxies.
The datasets were recovered through third-party mirrors; official release
checksums were unavailable, so independent certification of mirror bytes is
not claimed.

## Preprocessing

Images are stretched directly to 256x256. Lane coordinates are scaled
separately in x and y; centerlines are rasterized as width-three output
polylines. BDD lane categories, ignored regions, and cubic curves follow the
documented policy; annotated CULane occluded geometry is retained. Exact
transforms and the preprocessing policy are in `benchmark_artifacts/qa/`.

This harmonized binary semantic-mask protocol is not official lane-instance
evaluation for any benchmark. It does not measure instance matching, boxes,
class confidence, or NMS.

## Training and frozen evaluation

```powershell
.venv/Scripts/python.exe -m benchmark.run
.venv/Scripts/python.exe scripts/verify_benchmark.py
.venv/Scripts/python.exe scripts/report_benchmark.py
```

The recorded run used seeds 42/43/44, eight epochs, batch size four, AdamW at
0.001, weight decay 0.0005, three warm-up epochs followed by cosine decay,
ImageNet normalization, and random initialization. The loss is masked BCE +
Dice + 0.5 bidirectional lane continuity. Augmentation is applied only to
training data.

Each checkpoint and threshold is selected on its source validation subset and
frozen for all three target datasets. All nine training runs finish before
target evaluation. The immutable execution protocol and verification record
are published under `benchmark_artifacts/reproducibility/`. The local
`runs/real_v1/` directory contains full checkpoints and per-image outputs but
is intentionally not committed.

## Metrics and results

Pixel precision, recall, and F1 use pooled valid-pixel counts. Mean image IoU
averages images equally. Empty-image behavior and negative-image counts are
reported explicitly. Thresholded mask-IoU rates are called **mask-IoU success
rates**, not AP or mAP.

`benchmark_artifacts/report/RESULTS.md` presents the frozen 3x3 results,
three-seed mean and sample SD, conditional sequence-cluster bootstrap
intervals, diagnostics, and limitations. Bootstrap intervals describe
evaluation-sample uncertainty conditional on each checkpoint; they are not
training-seed intervals. The small subsets, short schedule, semantic
adaptation, and incomplete sequence linkage support exploratory observations
only—not convergence or state-of-the-art claims.

## Provenance

The public code and result artifacts document the real-image rerun only. They
do not establish the data provenance of the historical manuscript results.
The updated manuscript draft and reviewer response are maintained separately
from this code repository and are not published here.
