"""Pixel micro metrics and sequence bootstrap, never detection AP."""
import numpy as np


def per_image(probability, truth, valid, threshold):
    predicted = probability >= threshold
    truth, valid = truth.astype(bool), valid.astype(bool)
    tp = int((predicted & truth & valid).sum())
    fp = int((predicted & ~truth & valid).sum())
    fn = int((~predicted & truth & valid).sum())
    union = tp+fp+fn
    return {"tp": tp, "fp": fp, "fn": fn, "valid_pixels": int(valid.sum()),
            "iou": tp/union if union else 1.0,
            "truth_empty": not bool((truth & valid).any()), "prediction_empty": not bool((predicted & valid).any())}


def aggregate(rows):
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    ious = np.asarray([r["iou"] for r in rows])
    return {"pixel_precision": tp/(tp+fp) if tp+fp else 0.0,
            "pixel_recall": tp/(tp+fn) if tp+fn else 0.0,
            "pixel_f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
            "mean_image_iou": float(ious.mean()),
            "mask_iou_success_50": float((ious >= .5).mean()),
            "mask_iou_success_50_95": float(np.mean([np.mean(ious >= t) for t in np.linspace(.5, .95, 10)])),
            "tp": tp, "fp": fp, "fn": fn, "images": len(rows)}


def bootstrap(rows, seed=20261022, repetitions=2000):
    groups = {}
    for row in rows:
        groups.setdefault(row["sequence_id"], []).append(row)
    keys = sorted(groups)
    rng = np.random.default_rng(seed)
    names = [k for k in aggregate(rows) if k not in ("tp", "fp", "fn", "images")]
    values = {k: [] for k in names}
    for _ in range(repetitions):
        sampled = [row for i in rng.integers(0, len(keys), len(keys)) for row in groups[keys[i]]]
        result = aggregate(sampled)
        for k in names:
            values[k].append(result[k])
    return {"method": "sequence-cluster percentile bootstrap; pooled pixel counts recomputed in every resample",
            "seed": seed, "repetitions": repetitions, "sequence_groups": len(keys),
            "intervals_95": {k: [float(x) for x in np.percentile(v, [2.5, 97.5])] for k, v in values.items()},
            "scope": "evaluation sample uncertainty conditional on this trained checkpoint; not training-seed uncertainty"}
