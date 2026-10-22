"""Export preprocessing evidence and check all frozen files before training."""
import json
from pathlib import Path

import cv2
import numpy as np

from benchmark.acquire import BASE
from benchmark.data import POLICY, audit_manifests, load_sample, read_manifest
from benchmark.prepare import DATASETS


def main():
    destination = BASE/"benchmark_artifacts"/"qa"
    destination.mkdir(parents=True, exist_ok=True)
    report = audit_manifests(DATASETS)
    for dataset in DATASETS:
        records = read_manifest(dataset)
        transforms = []
        cards = []
        hashes = []
        (destination/dataset).mkdir(exist_ok=True)
        selected = {role: 0 for role in ("train", "validation", "evaluation")}
        for index, row in enumerate(records):
            image, mask, valid, transform = load_sample(row)
            transform.update(image_id=row["image_id"], role=row["role"], foreground_pixels=int(mask.sum()),
                             ignored_pixels=int((1-valid).sum()))
            transforms.append(transform)
            np.savez_compressed(destination/dataset/f"{index:03d}_mask.npz", mask=mask, valid=valid)
            raw = cv2.imread(str(BASE/row["image_path"]))
            small = cv2.resize(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32)
            coefficients = cv2.dct(small)[:8, :8].flatten()[1:]
            hashes.append(coefficients > np.median(coefficients))
            if selected[row["role"]] < 3:
                selected[row["role"]] += 1
                overlay = image.copy()
                positive = mask.astype(bool)
                overlay[positive] = (0.4*overlay[positive]+0.6*np.array([255, 30, 30])).astype(np.uint8)
                overlay[~valid.astype(bool)] = np.array([255, 200, 0], np.uint8)
                # Image and overlay side by side make coordinate mistakes visible.
                card = np.concatenate([image, overlay], axis=1)
                card = cv2.copyMakeBorder(card, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
                cv2.putText(card, f"{dataset} {row['role']} #{index} lane=red ignore=yellow", (6, 18),
                            cv2.FONT_HERSHEY_SIMPLEX, .42, (20, 20, 20), 1, cv2.LINE_AA)
                cards.append(card)
        candidates = []
        for i in range(len(records)):
            for j in range(i+1, len(records)):
                if records[i]["role"] != records[j]["role"]:
                    distance = int(np.count_nonzero(hashes[i] != hashes[j]))
                    if distance <= 4:
                        candidates.append({"first": records[i]["image_id"], "second": records[j]["image_id"],
                                           "hamming_distance": distance})
        report[dataset]["near_duplicate_candidates_phash_distance_le_4"] = candidates
        report[dataset]["empty_masks"] = sum(t["foreground_pixels"] == 0 for t in transforms)
        report[dataset]["positive_masks"] = sum(t["foreground_pixels"] > 0 for t in transforms)
        if report[dataset]["positive_masks"] == 0:
            raise ValueError(f"No positive lane annotations in {dataset}")
        if candidates:
            raise ValueError(f"Review near-duplicate candidates before training: {candidates}")
        sheet = np.concatenate([np.concatenate(cards[i:i+3], axis=1) for i in range(0, 9, 3)], axis=0)
        cv2.imwrite(str(destination/f"{dataset}_overlays.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        (destination/f"{dataset}_transforms.json").write_text(json.dumps(transforms, indent=2))
    (destination/"preprocessing.json").write_text(json.dumps(POLICY, indent=2))
    (destination/"audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
