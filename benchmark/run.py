"""Train independently per source and seed, then freeze and evaluate all targets.

Run only after benchmark.prepare and visually verified benchmark.qa output.
Artifacts are immutable protocol snapshots; unfinished training resumes only
when the configuration, data manifests, and source code hashes match exactly.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from benchmark.acquire import BASE
from benchmark.data import POLICY, LaneDataset, audit_manifests, read_manifest
from benchmark.metrics import aggregate, bootstrap, per_image
from benchmark.model import ARCHITECTURE, MODEL_NAME, LaneDGSemantic, segmentation_loss
from benchmark.prepare import DATASETS, sha256


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path, payload):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def infer(model, dataset, device, batch):
    model.eval()
    output = []
    for x, y, valid, indices in DataLoader(dataset, batch_size=batch, shuffle=False):
        probability = model(x.to(device)).sigmoid().cpu().numpy()[:, 0]
        for p, truth, v, index in zip(probability, y.numpy()[:, 0], valid.numpy()[:, 0], indices.tolist()):
            output.append((index, p, truth, v))
    return output


def train(source, seed, args, protocol_hash, run, device):
    directory = run/f"seed_{seed}"/source
    directory.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = LaneDGSemantic().to(device)
    rows = read_manifest(source)
    training = LaneDataset([r for r in rows if r["role"] == "train"], args.size, args.line_width, True, seed)
    validation = LaneDataset([r for r in rows if r["role"] == "validation"], args.size, args.line_width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    last, best_path = directory/"last.pt", directory/"best.pt"
    best, history, first = float("inf"), [], 0
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=True)
        if state["protocol_sha256"] != protocol_hash:
            raise ValueError("Cannot resume a checkpoint under a different protocol")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        first, best, history = state["epoch"], state["best_validation_loss"], state["history"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.type == "cuda" and "cuda_rng" in state:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    start = time.perf_counter()
    for epoch in range(first, args.epochs):
        training.epoch = epoch
        factor = ((epoch+1)/args.warmup_epochs if epoch < args.warmup_epochs else
                  .01+.99*.5*(1+math.cos(math.pi*(epoch-args.warmup_epochs)/max(1, args.epochs-args.warmup_epochs))))
        for group in optimizer.param_groups:
            group["lr"] = args.lr*factor
        generator = torch.Generator().manual_seed(seed+epoch*10000)
        loader = DataLoader(training, batch_size=args.batch, shuffle=True, generator=generator)
        model.train()
        total = 0.0
        for x, y, valid, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            value = segmentation_loss(model(x.to(device)), y.to(device), valid.to(device))
            if not torch.isfinite(value):
                raise ValueError("Non-finite training loss")
            value.backward()
            optimizer.step()
            total += value.detach().item()*len(x)
        model.eval()
        val_total = 0.0
        with torch.no_grad():
            for x, y, valid, _ in DataLoader(validation, batch_size=args.batch):
                value = segmentation_loss(model(x.to(device)), y.to(device), valid.to(device))
                val_total += value.item()*len(x)
        row = {"epoch": epoch+1, "train_loss": total/len(training), "validation_loss": val_total/len(validation),
               "learning_rate": args.lr*factor}
        history.append(row)
        print(f"seed={seed} source={source} {json.dumps(row)}", flush=True)
        if row["validation_loss"] < best:
            best = row["validation_loss"]
            save_checkpoint(best_path, {"model": model.state_dict(), "source": source, "seed": seed,
                                       "epoch": epoch+1, "validation_loss": best,
                                       "model_name": MODEL_NAME, "protocol_sha256": protocol_hash})
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch+1,
                   "best_validation_loss": best, "history": history, "torch_rng": torch.get_rng_state(),
                   "source": source, "seed": seed, "protocol_sha256": protocol_hash}
        if device.type == "cuda":
            payload["cuda_rng"] = torch.cuda.get_rng_state_all()
        save_checkpoint(last, payload)
        write_json(directory/"history.json", history)
    state = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(state["model"])
    validation_predictions = infer(model, validation, device, args.batch)
    threshold_curve = []
    for threshold in np.linspace(.1, .9, 17):
        metrics = aggregate([per_image(p, truth, valid, float(threshold)) for _, p, truth, valid in validation_predictions])
        threshold_curve.append({"threshold": float(threshold), **metrics})
    selected = max(threshold_curve, key=lambda r: (r["pixel_f1"], -abs(r["threshold"]-.5), r["threshold"]))
    val_directory = directory/"val_maps"
    val_directory.mkdir(exist_ok=True)
    val_records = []
    for index, probability, truth, valid in validation_predictions:
        np.savez_compressed(val_directory/f"{index:03d}.npz", probability=probability, truth=truth.astype(np.uint8), valid=valid.astype(np.uint8))
        val_records.append({"image_id": validation.rows[index]["image_id"], "index": index,
                            **per_image(probability, truth, valid, selected["threshold"])})
    selection = {"source": source, "seed": seed, "checkpoint_sha256": sha256(best_path),
                 "best_epoch": state["epoch"], "validation_loss": state["validation_loss"],
                 "threshold": selected["threshold"], "threshold_policy": "maximize pooled source-validation pixel F1; ties closest to 0.5, then higher threshold",
                 "threshold_curve": threshold_curve, "per_image_validation": val_records,
                 "training_seconds_this_invocation": time.perf_counter()-start,
                 "parameters": sum(p.numel() for p in model.parameters())}
    write_json(directory/"selection.json", selection)
    return best_path, selection


def evaluate(source, seed, checkpoint, selection, args, protocol_hash, run, device):
    model = LaneDGSemantic().to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if state["protocol_sha256"] != protocol_hash:
        raise ValueError("Checkpoint protocol mismatch")
    model.load_state_dict(state["model"])
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    matrix = {}
    for target in DATASETS:
        directory = run/f"seed_{seed}"/source/"evaluation"/target
        directory.mkdir(parents=True, exist_ok=True)
        rows = [r for r in read_manifest(target) if r["role"] == "evaluation"]
        dataset = LaneDataset(rows, args.size, args.line_width)
        records = []
        for index, probability, truth, valid in infer(model, dataset, device, args.batch):
            name = f"{index:03d}.npz"
            np.savez_compressed(directory/name, probability=probability, truth=truth.astype(np.uint8), valid=valid.astype(np.uint8))
            row = rows[index]
            record = {"source": source, "target": target, "training_seed": seed, "index": index,
                      "image_id": row["image_id"], "sequence_id": row["sequence_id"],
                      "image_sha256": row["image_sha256"], "annotation_sha256": row["annotation_sha256"],
                      "checkpoint_sha256": selection["checkpoint_sha256"], "threshold": selection["threshold"],
                      "probability_map": name, "probability_sha256": sha256(directory/name),
                      **per_image(probability, truth, valid, selection["threshold"])}
            records.append(record)
            if index < 3:
                image = dataset.samples[index][0]
                mask = (probability >= selection["threshold"]) & valid.astype(bool)
                prediction = image.copy()
                prediction[mask] = (prediction[mask]*.4+np.array([20, 230, 30])*.6).astype(np.uint8)
                truth_view = image.copy()
                truth_view[truth.astype(bool)] = (truth_view[truth.astype(bool)]*.4+np.array([240, 30, 30])*.6).astype(np.uint8)
                cv2.imwrite(str(directory/f"{index:03d}_prediction.png"), cv2.cvtColor(np.concatenate([truth_view, prediction], 1), cv2.COLOR_RGB2BGR))
        with (directory/"per_image.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        write_json(directory/"per_image.json", records)
        metrics = aggregate(records)
        uncertainty = bootstrap(records, args.bootstrap_seed, args.bootstrap_repetitions)
        matrix[target] = {"metrics": metrics, "bootstrap": uncertainty,
                          "checkpoint_sha256": selection["checkpoint_sha256"], "threshold": selection["threshold"]}
        write_json(directory/"metrics.json", matrix[target])
        print(f"FROZEN EVALUATION seed={seed} {source}->{target}: {json.dumps(metrics)}", flush=True)
    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=BASE/"runs/real_v1")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=.0005)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--line-width", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--bootstrap-seed", type=int, default=20261022)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    args = parser.parse_args()
    if args.size % 32 or args.epochs < 1 or not 0 <= args.warmup_epochs < args.epochs:
        parser.error("size must be divisible by 32; epochs positive and greater than warmup")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    audit = audit_manifests(DATASETS)
    qa_path = BASE/"benchmark_artifacts/qa/visual_review.json"
    if not qa_path.exists() or not json.loads(qa_path.read_text())["passed"]:
        raise ValueError("Visual overlay review must pass before training")
    visual_review = json.loads(qa_path.read_text())
    for dataset in DATASETS:
        if visual_review["manifest_sha256"][dataset] != sha256(BASE/"benchmark_artifacts/manifests"/f"{dataset}.json"):
            raise ValueError("Manifest changed since visual QA")
        if visual_review["overlay_sha256"][dataset] != sha256(BASE/"benchmark_artifacts/qa"/f"{dataset}_overlays.png"):
            raise ValueError("Overlay changed since visual QA")
    if (args.size, args.line_width) != (256, 3):
        raise ValueError("Current visual QA covers size=256 and width=3 only; review a separate protocol for other settings")
    if any(any(audit[d]["counts"].get(r) != n for r, n in (("train", 80), ("validation", 20), ("evaluation", 40))) for d in DATASETS):
        raise ValueError("This frozen run requires the declared 80/20/40 design")
    run = args.run_dir.resolve()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    code_hashes = {str(p.relative_to(BASE)).replace("\\", "/"): sha256(p) for p in (BASE/"benchmark").glob("*.py")}
    protocol = {"config": config, "architecture": ARCHITECTURE, "preprocessing": POLICY,
                "manifest_sha256": {d: sha256(BASE/"benchmark_artifacts/manifests"/f"{d}.json") for d in DATASETS},
                "source_code_sha256": code_hashes, "visual_review_sha256": sha256(qa_path),
                "selection": "minimum source-validation loss checkpoint; threshold chosen from source-validation data only",
                "evaluation": "all three source models frozen; same three held-out target subsets; all training complete before any target metrics",
                "loss": "masked BCE + per-image Dice + 0.5 bidirectional valid-lane continuity",
                "augmentation": "epoch-varying horizontal flip p=0.5 and brightness +/-0.08, contrast 0.85..1.15; training only",
                "precision_zero_division": 0, "recall_zero_division": 0,
                "uncertainty": "2000 sequence-cluster bootstrap resamples per checkpoint; training seeds reported separately"}
    protocol_hash = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    if (run/"protocol.json").exists():
        saved = json.loads((run/"protocol.json").read_text())
        if saved != protocol:
            raise ValueError("Run directory already belongs to a different frozen protocol")
    write_json(run/"protocol.json", protocol)
    write_json(run/"audit.json", audit)
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                   "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "threads": args.threads,
                   "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
                   "protocol_sha256": protocol_hash}
    write_json(run/"environment.json", environment)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"DEVICE={device} MODEL={MODEL_NAME} PROTOCOL={protocol_hash}", flush=True)
    selections = {}
    for seed in args.seeds:
        selections[seed] = {}
        for source in DATASETS:
            selections[seed][source] = train(source, seed, args, protocol_hash, run, device)
    results = {}
    for seed in args.seeds:
        results[str(seed)] = {}
        for source in DATASETS:
            checkpoint, selection = selections[seed][source]
            results[str(seed)][source] = evaluate(source, seed, checkpoint, selection, args, protocol_hash, run, device)
    write_json(run/"results.json", {"status": "complete", "model": MODEL_NAME, "protocol_sha256": protocol_hash,
                                    "seeds": args.seeds, "matrix_by_seed": results})
    print("COMPLETE: nine trained checkpoints and 27 source-target evaluations" if len(args.seeds) == 3 else "COMPLETE", flush=True)


if __name__ == "__main__":
    main()
