"""Independently reconstruct saved counts and summaries from probability maps."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from benchmark.data import read_manifest
from benchmark.prepare import DATASETS


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=BASE/"runs/real_v1")
    args = parser.parse_args()
    run = args.run_dir
    results = json.loads((run/"results.json").read_text())
    protocol = json.loads((run/"protocol.json").read_text())
    assert results["status"] == "complete"
    expected_protocol = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    assert expected_protocol == results["protocol_sha256"]
    for dataset in DATASETS:
        assert digest(BASE/"benchmark_artifacts/manifests"/f"{dataset}.json") == protocol["manifest_sha256"][dataset]
    checked_images = 0
    checked_cells = 0
    for seed in results["seeds"]:
        for source in DATASETS:
            parent = run/f"seed_{seed}"/source
            selection = json.loads((parent/"selection.json").read_text())
            checkpoint_hash = digest(parent/"best.pt")
            assert checkpoint_hash == selection["checkpoint_sha256"]
            checkpoint = torch.load(parent/"best.pt", weights_only=True, map_location="cpu")
            assert checkpoint["protocol_sha256"] == expected_protocol
            assert checkpoint["source"] == source and checkpoint["seed"] == seed
            history = json.loads((parent/"history.json").read_text())
            assert len(history) == protocol["config"]["epochs"]
            assert checkpoint["epoch"] == min(history, key=lambda r: r["validation_loss"])["epoch"]
            curve = selection["threshold_curve"]
            chosen = max(curve, key=lambda r: (r["pixel_f1"], -abs(r["threshold"]-.5), r["threshold"]))
            assert chosen["threshold"] == selection["threshold"]
            assert {r["image_id"] for r in selection["per_image_validation"]} == {
                r["image_id"] for r in read_manifest(source) if r["role"] == "validation"}
            for target in DATASETS:
                folder = parent/"evaluation"/target
                records = json.loads((folder/"per_image.json").read_text())
                expected_ids = {r["image_id"] for r in read_manifest(target) if r["role"] == "evaluation"}
                assert len(records) == len(expected_ids) == 40
                assert {r["image_id"] for r in records} == expected_ids
                sums = np.zeros(3, dtype=np.int64)
                ious = []
                for row in records:
                    assert row["checkpoint_sha256"] == checkpoint_hash
                    assert row["threshold"] == selection["threshold"]
                    path = folder/row["probability_map"]
                    assert digest(path) == row["probability_sha256"]
                    with np.load(path) as saved:
                        p, y, valid = saved["probability"], saved["truth"].astype(bool), saved["valid"].astype(bool)
                        assert p.dtype == np.float32 and p.shape == y.shape == valid.shape == (256, 256)
                        assert np.isfinite(p).all() and p.min() >= 0 and p.max() <= 1
                        prediction = p >= row["threshold"]
                        tp = np.count_nonzero(prediction[y & valid])
                        fp = np.count_nonzero(prediction[~y & valid])
                        fn = np.count_nonzero(~prediction[y & valid])
                        counts = np.array([tp, fp, fn])
                        assert counts.tolist() == [row["tp"], row["fp"], row["fn"]]
                        union = int(counts.sum())
                        iou = float(tp/union) if union else 1.
                        assert abs(iou-row["iou"]) < 1e-12
                        sums += counts
                        ious.append(iou)
                    checked_images += 1
                tp, fp, fn = [int(x) for x in sums]
                independent = {"pixel_precision": tp/(tp+fp) if tp+fp else 0.,
                               "pixel_recall": tp/(tp+fn) if tp+fn else 0.,
                               "pixel_f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                               "mean_image_iou": sum(ious)/len(ious)}
                stored = results["matrix_by_seed"][str(seed)][source][target]["metrics"]
                for name, value in independent.items():
                    assert abs(value-stored[name]) < 1e-12
                checked_cells += 1
    output = {"passed": True, "trained_checkpoints": len(results["seeds"])*3,
              "evaluation_cells": checked_cells, "probability_maps_and_count_records": checked_images,
              "checks": ["checkpoint identity and frozen protocol", "source-validation-only epoch and threshold selection",
                         "held-out target identity", "float32 probability maps and hashes",
                         "independently reconstructed TP/FP/FN/IoU", "independently reconstructed pixel metrics"],
              "protocol_sha256": expected_protocol}
    (run/"verification.json").write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
