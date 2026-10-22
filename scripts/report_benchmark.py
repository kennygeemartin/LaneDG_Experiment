"""Generate tables, figures, and manuscript-ready wording from verified results."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from benchmark.prepare import DATASETS

METRICS = ("pixel_precision", "pixel_recall", "pixel_f1", "mean_image_iou")
LABELS = {"tusimple": "TuSimple", "culane": "CULane", "bdd100k": "BDD100K"}


def save_figure(figure, path):
    figure.savefig(path.with_suffix(".png"), dpi=250, bbox_inches="tight")
    figure.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=BASE/"runs/real_v1")
    args = parser.parse_args()
    run = args.run_dir
    verification = json.loads((run/"verification.json").read_text())
    if not verification["passed"]:
        raise ValueError("Verify all saved maps and counts before reporting")
    results = json.loads((run/"results.json").read_text())
    protocol = json.loads((run/"protocol.json").read_text())
    seeds = results["seeds"]
    matrices = results["matrix_by_seed"]
    output = BASE/"benchmark_artifacts/report"
    output.mkdir(parents=True, exist_ok=True)
    (output/"figures").mkdir(exist_ok=True)
    means, standard_deviations, rows = {}, {}, []
    for source in DATASETS:
        means[source], standard_deviations[source] = {}, {}
        for target in DATASETS:
            values = {metric: np.asarray([matrices[str(seed)][source][target]["metrics"][metric] for seed in seeds]) for metric in METRICS}
            means[source][target] = {m: float(v.mean()) for m, v in values.items()}
            standard_deviations[source][target] = {m: float(v.std(ddof=1)) if len(v)>1 else 0. for m, v in values.items()}
            for seed in seeds:
                cell = matrices[str(seed)][source][target]
                row = {"training_seed": seed, "source": source, "target": target,
                       "checkpoint_sha256": cell["checkpoint_sha256"], "threshold": cell["threshold"], **cell["metrics"]}
                for metric, interval in cell["bootstrap"]["intervals_95"].items():
                    row[metric+"_ci95_low"], row[metric+"_ci95_high"] = interval
                records = json.loads((run/f"seed_{seed}"/source/"evaluation"/target/"per_image.json").read_text())
                positive = [r for r in records if not r["truth_empty"]]
                row["positive_images"] = len(positive)
                row["negative_images"] = len(records)-len(positive)
                row["positive_only_mean_image_iou"] = float(np.mean([r["iou"] for r in positive])) if positive else None
                row["empty_prediction_images"] = sum(r["prediction_empty"] for r in records)
                row["all_background_mean_image_iou_baseline"] = row["negative_images"]/len(records)
                rows.append(row)
    with (output/"metrics_by_seed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    aggregate_rows = []
    for source in DATASETS:
        for target in DATASETS:
            row = {"source": source, "target": target}
            for metric in METRICS:
                row[metric+"_mean"] = means[source][target][metric]
                row[metric+"_training_seed_sd"] = standard_deviations[source][target][metric]
            aggregate_rows.append(row)
    with (output/"three_seed_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate_rows[0]))
        writer.writeheader()
        writer.writerows(aggregate_rows)

    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.7), constrained_layout=True)
    for ax, metric, title in zip(axes, ("pixel_f1", "mean_image_iou"), ("Pixel F1", "Mean image IoU")):
        array = np.array([[means[s][t][metric] for t in DATASETS] for s in DATASETS])
        view = ax.imshow(array, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(3), [LABELS[d] for d in DATASETS])
        ax.set_yticks(range(3), [LABELS[d] for d in DATASETS])
        ax.set_xlabel("Held-out target")
        ax.set_ylabel("Training source")
        ax.set_title(title)
        for i, source in enumerate(DATASETS):
            for j, target in enumerate(DATASETS):
                ax.text(j, i, f"{array[i,j]:.3f}\n±{standard_deviations[source][target][metric]:.3f}",
                        ha="center", va="center", color="white" if array[i,j]>.55 else "black")
        fig.colorbar(view, ax=ax, shrink=.8)
    fig.suptitle("Real-image semantic rerun · mean ± training-seed SD (n=3)", fontsize=12)
    save_figure(fig, output/"figures/frozen_matrices")

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, constrained_layout=True)
    for i, seed in enumerate(seeds):
        for j, source in enumerate(DATASETS):
            history = json.loads((run/f"seed_{seed}"/source/"history.json").read_text())
            ax = axes[i, j]
            ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], "o-", markersize=3, label="Training")
            ax.plot([r["epoch"] for r in history], [r["validation_loss"] for r in history], "s-", markersize=3, label="Source validation")
            selection = json.loads((run/f"seed_{seed}"/source/"selection.json").read_text())
            ax.axvline(selection["best_epoch"], color="gray", linestyle=":", linewidth=1)
            ax.set_title(f"{LABELS[source]} · seed {seed}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
    axes[0, 0].legend(fontsize=8)
    save_figure(fig, output/"figures/training_histories")

    transfer = {source: [float(np.mean([matrices[str(seed)][source][target]["metrics"]["pixel_f1"]
                                     for target in DATASETS if target!=source])) for seed in seeds] for source in DATASETS}
    fig, ax = plt.subplots(figsize=(6.3, 4.2), constrained_layout=True)
    for i, source in enumerate(DATASETS):
        values = np.asarray(transfer[source])
        ax.bar(i, values.mean(), width=.55, color="#7198bb", alpha=.6)
        ax.errorbar(i, values.mean(), yerr=values.std(ddof=1), color="#163955", capsize=5)
        ax.scatter(i+np.linspace(-.12,.12,len(seeds)), values, color="#163955", s=22, zorder=3)
    ax.set_xticks(range(3), [LABELS[d] for d in DATASETS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Mean off-domain pixel F1")
    ax.set_title("Descriptive source transfer · three training seeds")
    save_figure(fig, output/"figures/source_transfer")

    lines = ["# Real-image cross-dataset semantic segmentation rerun", "",
             "The rerun is complete: nine independently trained source/seed models and 27 frozen source-target evaluations on real benchmark images. All 1,080 evaluation probability maps and count records passed independent reconstruction checks.", "",
             "The model is **LaneDG-Semantic-Reimplementation-v1**, with 1,141,793 trainable parameters. It is a documented semantic adaptation, not a verified reproduction of the cited YOLOv8n-LaneDG instance network. These results replace neither the provenance nor the architectural claims of the original manuscript automatically.", "",
             "## Frozen protocol", "",
             "Each benchmark contributes 80 training, 20 validation and 40 held-out evaluation images, with sampling seed 42 and disjoint declared sequence groups. BDD100K validation and evaluation use separate labeled val subsets. Training uses seeds 42, 43 and 44; eight epochs; 256×256 input; batch four; AdamW learning rate 0.001; weight decay 0.0005; three warm-up epochs and cosine decay; random initialization. All nine trainings finish before any target evaluation.", "",
             "A source checkpoint minimizes source-validation loss, and its mask probability threshold maximizes source-validation pixel F1. Both are frozen across targets. Images are stretched directly to a square; lane coordinates are transformed and rasterized at three output pixels. BDD category exclusions, cubic-curve handling and ignored pixels are recorded in the preprocessing policy. See [data sources](../DATA_SOURCES.md), [QA audit](../qa/audit.json), and [executed protocol](../../runs/real_v1/protocol.json).", "",
             "## Three-seed results", "", "Values are mean ± sample SD across three training seeds. These SDs are not confidence intervals.", "",
             "| Source | Target | Pixel precision | Pixel recall | Pixel F1 | Mean image IoU |",
             "|---|---|---:|---:|---:|---:|"]
    for source in DATASETS:
        for target in DATASETS:
            cells = [f"{means[source][target][m]:.3f} ± {standard_deviations[source][target][m]:.3f}" for m in METRICS]
            lines.append("| "+" | ".join([LABELS[source], LABELS[target], *cells])+" |")
    lines += ["", "![Frozen F1 and IoU matrices](figures/frozen_matrices.png)", "",
              "## Conditional 95 percent intervals", "",
              "Each checkpoint has 2,000 sequence-cluster percentile bootstrap resamples, with pooled pixel counts recomputed in each resample. Because selection takes one image per sequence group, each target has 40 groups. These intervals capture sampled-evaluation uncertainty conditional on the trained checkpoint. They do not include uncertainty from resampling training images, training seeds, unknown longer-drive relationships, or preprocessing choices.", "",
              "Seed 42 is shown below; [the complete 27-cell CSV](metrics_by_seed.csv) contains all seeds, all four primary metrics, success-rate intervals and diagnostics.", "",
              "| Source | Target | Pixel F1 [95% CI] | Mean image IoU [95% CI] |",
              "|---|---|---:|---:|"]
    for row in rows:
        if row["training_seed"] != 42:
            continue
        cells = [f"{row[m]:.3f} [{row[m+'_ci95_low']:.3f}, {row[m+'_ci95_high']:.3f}]" for m in ("pixel_f1", "mean_image_iou")]
        lines.append("| "+" | ".join([LABELS[row["source"]], LABELS[row["target"]], *cells])+" |")
    lines += ["", "## Foreground and negative-image diagnostics", "",
              "Empty ground truth and empty prediction have image IoU 1. Consequently, an all-background model can obtain a nonzero mean image IoU. The table below exposes this baseline; pixel F1 for an all-background model is zero. Positive-only image IoU is averaged over images with annotated foreground, then across training seeds.", "",
              "| Target | Positive evaluation images | Negative evaluation images | All-background mean image IoU |",
              "|---|---:|---:|---:|"]
    for target in DATASETS:
        row = next(r for r in rows if r["target"] == target)
        lines.append(f"| {LABELS[target]} | {row['positive_images']} | {row['negative_images']} | {row['all_background_mean_image_iou_baseline']:.3f} |")
    lines += ["", "| Source | Target | Positive-only mean image IoU (three-seed mean) |", "|---|---|---:|"]
    for source in DATASETS:
        for target in DATASETS:
            value = np.mean([r["positive_only_mean_image_iou"] for r in rows if r["source"]==source and r["target"]==target])
            lines.append(f"| {LABELS[source]} | {LABELS[target]} | {value:.3f} |")
    lines += ["", "## Training and interpretation", "", "![Training histories](figures/training_histories.png)", "",
              "![Descriptive transfer](figures/source_transfer.png)", "",
              "Mean off-domain pixel F1 across seeds:"]
    for source in DATASETS:
        values = np.asarray(transfer[source])
        lines.append(f"- {LABELS[source]}: {values.mean():.3f} ± {values.std(ddof=1):.3f}.")
    lines += ["", "Source ordering is descriptive for these sampled subsets and short runs. The study cannot establish that dataset diversity caused the observed ordering. Eight epochs and 80 training images may undertrain the model; annotation conventions, negative-image prevalence, sampling and the semantic adaptation can all affect transfer. There is no claim of convergence or state-of-the-art benchmark performance.", "",
              "TuSimple clip IDs and CULane MP4 IDs are separated, but complete longer-drive linkage is unavailable. BDD filename prefixes are conservative groups, not independently verified recording identifiers. Perceptual-hash checking found no cross-role candidates at Hamming distance ≤4; it does not prove semantic or temporal independence.", "",
              "The optional mask-IoU success rates are not AP/mAP. No detection boxes, instance matching, confidence-ranked precision-recall curves or NMS are used. Official benchmark lane-instance metrics are not claimed.", "",
              "## Preserved provenance and outputs", "",
              "The original checkpoints and aggregate histories were recovered, including a separate partial 500-epoch-configured TuSimple job saved at epoch nine. Real-image manifests and per-image evaluation records for those historical tables were not recovered. The current public source contains a synthetic generator. These findings do not establish how the original results were produced. See [recovery audit](../recovery/recovery_audit.json).", "",
              "The executed code and manifests are archived in `runs/real_v1/source_snapshot.zip`. Raw checkpoint-specific maps and metrics are under `runs/real_v1/seed_SEED/SOURCE/evaluation/TARGET/`. [Verification](../../runs/real_v1/verification.json), [all seed metrics](metrics_by_seed.csv), and [three-seed summary](three_seed_summary.csv) are machine-readable. PNG and editable SVG figures are provided. No changes have been pushed to GitHub.", ""]
    (output/"RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    summary = {"status": "complete", "model": results["model"], "training_seeds": seeds,
               "mean": means, "training_seed_sample_sd": standard_deviations,
               "off_domain_pixel_f1": {s: {"mean": float(np.mean(v)), "training_seed_sample_sd": float(np.std(v, ddof=1))} for s, v in transfer.items()},
               "protocol_sha256": results["protocol_sha256"]}
    (output/"summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"report": str(output/"RESULTS.md"), "off_domain_pixel_f1": summary["off_domain_pixel_f1"]}, indent=2))


if __name__ == "__main__":
    main()
