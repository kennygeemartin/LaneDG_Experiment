# YOLOv8n-LaneDG cross-dataset experiment

Run the scaled reproducible experiment with:

```powershell
python lanedg_experiment/run_experiment.py
```

Checkpoints, raw results, and figures are saved under `lanedg_experiment/`.
Defaults are 640Ã—640, 500 epochs, batch 16, AdamW (initial learning rate
0.01), three warm-up epochs, cosine decay, ImageNet normalization, and a
frozen 3Ã—3 source-to-target evaluation. Full-protocol artifacts are written
under `lanedg_experiment/full_protocol/`.

The publication-ready interpretation and chart index are in
[`lanedg_experiment/results/RESULTS.md`](lanedg_experiment/results/RESULTS.md).
Every chart is exported as PNG and PDF.

