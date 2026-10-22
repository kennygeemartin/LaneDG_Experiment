"""Annotation parsing, documented rasterization, and leakage checks."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from benchmark.acquire import BASE

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
POLICY = {
    "task": "binary semantic lane segmentation; not official instance/lane metrics",
    "resize": "direct stretch to square; OpenCV INTER_LINEAR RGB image; no cropping or letterbox",
    "coordinates": "x_out=x_in*size/original_width; y_out=y_in*size/original_height; round with numpy.rint",
    "rasterization": "OpenCV LINE_8 polylines after coordinate transform; width 3 output pixels; no polygon filling",
    "tusimple": "connect consecutive valid x>=0 samples; split at missing samples; do not extrapolate",
    "culane": "all annotated lane polylines; corrected annotations_new for train/val; retain annotated occluded portions",
    "bdd100k_positive": ["lane/single white", "lane/single yellow", "lane/single other",
                         "lane/double white", "lane/double yellow", "lane/double other"],
    "bdd100k_ignore": "crosswalks, curbs, vertical lanes, explicit ignore=true and occluded=true lane objects: rasterized support excluded from loss and metrics",
    "bdd100k_curves": "cubic Bezier segments sampled at 64 intervals; match official Matplotlib Path semantics: first point MOVETO; a C consumes the next three vertices irrespective of their subsequent codes (supports legacy CCCL)",
    "bdd100k_dashed": "annotated continuous geometry retained, without inventing dash gaps",
    "occlusion": "no synthesized occluded geometry; retain provided TuSimple/CULane geometry; explicitly occluded BDD lanes ignored",
    "background": "all pixels not marked positive or ignored; unlabeled lanes are background under this harmonized protocol",
    "empty_masks": "retain negative images; image IoU=1 for empty prediction and empty truth; otherwise union IoU",
    "normalization": {"color": "RGB", "scale": "uint8/255", "mean": MEAN.tolist(), "std": STD.tolist()},
}


def read_manifest(dataset):
    path = BASE / "benchmark_artifacts" / "manifests" / f"{dataset}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def source_annotation(row):
    path = BASE / row["annotation_path"]
    if row["dataset"] == "tusimple":
        with path.open(encoding="utf-8") as handle:
            for n, line in enumerate(handle, 1):
                if n == row["annotation_line"]:
                    item = json.loads(line)
                    if item["raw_file"] != row["image_id"]:
                        raise ValueError("TuSimple annotation/image mismatch")
                    return item
        raise ValueError("Missing annotation line")
    if row["dataset"] == "bdd100k":
        return json.loads(path.read_text(encoding="utf-8"))
    return path.read_text(encoding="utf-8")


def bezier_path(vertices, types):
    if len(vertices) != len(types) or not len(vertices):
        raise ValueError("Invalid poly2d vertices/types")
    points = [np.asarray(vertices[0], dtype=float)]
    index = 1
    while index < len(vertices):
        if types[index] == "L":
            points.append(np.asarray(vertices[index], dtype=float))
            index += 1
        elif types[index] == "C":
            # The official legacy viewer maps C to Matplotlib CURVE4. A CURVE4
            # consumes three vertices, even where a legacy endpoint has code L.
            if index + 2 >= len(vertices):
                raise ValueError("Cubic Bezier segment must have three control/end vertices")
            p0 = points[-1]
            p1, p2, p3 = np.asarray(vertices[index:index+3], dtype=float)
            for t in np.linspace(0, 1, 65)[1:]:
                points.append((1-t)**3*p0 + 3*(1-t)**2*t*p1 + 3*(1-t)*t*t*p2 + t**3*p3)
            index += 3
        else:
            raise ValueError(f"Unsupported poly2d command {types[index]}")
    return np.asarray(points)


def paths(row):
    """Return (points, ignored) paths; missing samples never bridge a gap."""
    annotation = source_annotation(row)
    if row["dataset"] == "tusimple":
        result = []
        for lane in annotation["lanes"]:
            if len(lane) != len(annotation["h_samples"]):
                raise ValueError("TuSimple sample lengths differ")
            segment = []
            for x, y in zip(lane, annotation["h_samples"]):
                if x >= 0:
                    segment.append([x, y])
                else:
                    if len(segment) >= 2:
                        result.append((np.asarray(segment), False))
                    segment = []
            if len(segment) >= 2:
                result.append((np.asarray(segment), False))
        return result
    if row["dataset"] == "culane":
        result = []
        for line in annotation.splitlines():
            values = np.asarray([float(x) for x in line.split()])
            if len(values) % 2:
                raise ValueError("CULane annotation has an odd coordinate count")
            if len(values) >= 4:
                result.append((values.reshape(-1, 2), False))
        return result
    result = []
    for obj in annotation["frames"][0]["objects"]:
        category = obj["category"]
        if not category.startswith("lane/"):
            continue
        attr = obj.get("attributes", {})
        ignored = (category not in POLICY["bdd100k_positive"]
                   or attr.get("direction", attr.get("laneDirection")) == "vertical"
                   or bool(attr.get("ignore", False)) or bool(attr.get("occluded", False)))
        geometry = obj.get("poly2d", [])
        if not geometry:
            raise ValueError("Lane object missing poly2d geometry")
        if isinstance(geometry[0], dict):
            curves = [(p["vertices"], p["types"]) for p in geometry]
        else:
            curves = [([p[:2] for p in geometry], [p[2] for p in geometry])]
        result.extend((bezier_path(v, t), ignored) for v, t in curves)
    return result


def rasterize(row, width, height, size=256, line_width=3):
    mask = np.zeros((size, size), dtype=np.uint8)
    ignore = np.zeros_like(mask)
    for points, ignored in paths(row):
        if not np.isfinite(points).all():
            raise ValueError("Non-finite lane coordinates")
        points = np.rint(points * np.array([size/width, size/height])).astype(np.int32)
        cv2.polylines(ignore if ignored else mask, [points], False, 1, line_width, cv2.LINE_8)
    mask[ignore.astype(bool)] = 0
    return mask, (1-ignore)


def load_sample(row, size=256, line_width=3):
    raw = cv2.imread(str(BASE / row["image_path"]), cv2.IMREAD_COLOR)
    if raw is None:
        raise ValueError(f"Unreadable image {row['image_path']}")
    height, width = raw.shape[:2]
    mask, valid = rasterize(row, width, height, size, line_width)
    image = cv2.cvtColor(cv2.resize(raw, (size, size), interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    return image, mask, valid, {"original_width": width, "original_height": height,
                                "scale_x": size/width, "scale_y": size/height,
                                "output_size": size, "line_width": line_width}


def audit_manifests(datasets):
    """Reject identity, content, decoded-pixel, and sequence leakage."""
    global_content = {}
    report = {}
    for dataset in datasets:
        rows = read_manifest(dataset)
        identity, sequences, content = {}, {}, {}
        for row in rows:
            role = row["role"]
            image_path = BASE / row["image_path"]
            image_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
            annotation_hash = hashlib.sha256((BASE/row["annotation_path"]).read_bytes()).hexdigest()
            if image_hash != row["image_sha256"] or annotation_hash != row["annotation_sha256"]:
                raise ValueError("Manifest/source hash mismatch")
            raw = cv2.imread(str(image_path))
            if raw is None:
                raise ValueError("Unreadable image")
            pixel_hash = hashlib.sha256(raw.tobytes()).hexdigest()
            for mapping, key in ((identity, row["image_id"]), (sequences, row["sequence_id"]),
                                 (content, image_hash), (content, pixel_hash)):
                if key in mapping and mapping[key] != role:
                    raise ValueError(f"{dataset} leakage across {mapping[key]} and {role}: {key}")
                mapping[key] = role
            if pixel_hash in global_content:
                raise ValueError(f"Duplicate image content: {global_content[pixel_hash]} and {row['image_id']}")
            global_content[pixel_hash] = row["image_id"]
        report[dataset] = {"counts": dict(Counter(r["role"] for r in rows)),
                           "unique_images": len(identity), "unique_sequence_groups": len(sequences),
                           "image_hashes_verified": True, "annotation_hashes_verified": True,
                           "role_leakage": False, "duplicate_decoded_images": False,
                           "limitation": "TuSimple clip IDs and CULane video IDs verified; longer-drive linkage beyond supplied IDs unavailable. BDD uses conservative filename-prefix groups."}
    return report


class LaneDataset(Dataset):
    def __init__(self, rows, size=256, line_width=3, augment=False, seed=42):
        self.rows, self.augment, self.seed, self.epoch = rows, augment, seed, 0
        self.samples = [load_sample(row, size, line_width)[:3] for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        image, mask, valid = (x.copy() for x in self.samples[index])
        if self.augment:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            if rng.random() < 0.5:
                image, mask, valid = image[:, ::-1].copy(), mask[:, ::-1].copy(), valid[:, ::-1].copy()
            image = np.clip((image.astype(np.float32)-127.5)*rng.uniform(.85, 1.15)
                            +127.5+rng.uniform(-.08, .08)*255, 0, 255)
        x = ((image.astype(np.float32)/255 - MEAN)/STD).transpose(2, 0, 1)
        return torch.from_numpy(x.copy()), torch.from_numpy(mask[None].astype(np.float32)), torch.from_numpy(valid[None].astype(np.float32)), index
