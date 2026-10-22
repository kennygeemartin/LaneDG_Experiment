"""Freeze disjoint real-image subsets and download their source files."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import urllib.request
from pathlib import Path, PurePosixPath

from benchmark.acquire import BASE, DATA, RemoteZip, fetch_file

MANIFESTS = BASE / "benchmark_artifacts" / "manifests"
DATASETS = ("tusimple", "culane", "bdd100k")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hf_inventory(split):
    """Enumerate the entire original split, not just an accessible first page."""
    target = DATA / "bdd100k" / f"inventory_{split}.json"
    if target.exists():
        return json.loads(target.read_text(encoding="utf-8"))
    url = f"https://huggingface.co/api/datasets/minhletran/BDD100K-lane/tree/main/image/{split}?limit=1000"
    files = []
    while url:
        with urllib.request.urlopen(url, timeout=90) as response:
            items = json.load(response)
            link = response.headers.get("Link", "")
        files.extend(x["path"].rsplit("/", 1)[-1] for x in items
                     if x["type"] == "file" and x["path"].endswith(".jpg"))
        url = next((p.split(">")[0].strip()[1:] for p in link.split(",") if 'rel="next"' in p), None)
        print(f"BDD100K {split} inventory: {len(files)}", flush=True)
    files = sorted(set(files))
    expected = {"train": 70000, "val": 10000}[split]
    if len(files) != expected:
        raise ValueError(f"Incomplete {split} inventory: {len(files)}, expected {expected}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(files, indent=2), encoding="utf-8")
    return files


def bdd_inventory():
    """Read the ZIP central directory only, avoiding a multi-GB download."""
    import zipfile
    target = DATA / "bdd100k" / "complete_image_inventory.json"
    if target.exists():
        inventory = json.loads(target.read_text(encoding="utf-8"))
    else:
        url = "https://www.kaggle.com/api/v1/datasets/download/marquis03/bdd100k"
        with zipfile.ZipFile(RemoteZip(url)) as archive:
            inventory = {split: sorted(PurePosixPath(n).name for n in archive.namelist()
                                       if n.startswith(split + "/") and n.endswith(".jpg"))
                         for split in ("train", "val", "test")}
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(inventory, indent=2), encoding="utf-8")
    for split, expected in (("train", 70000), ("val", 10000), ("test", 20000)):
        if len(set(inventory[split])) != expected:
            raise ValueError(f"Incomplete original {split} image inventory")
    return inventory


def tusimple_candidates():
    pools = {"train": [], "evaluation": []}
    for name in ("label_data_0313.json", "label_data_0531.json", "label_data_0601.json", "test_label.json"):
        annotation = DATA / "tusimple" / "annotations" / name
        role = "evaluation" if name == "test_label.json" else "train"
        original = "test" if role == "evaluation" else "train"
        for line_number, line in enumerate(annotation.read_text(encoding="utf-8").splitlines(), 1):
            item = json.loads(line)
            raw = item["raw_file"]
            sequence = str(PurePosixPath(raw).parent)
            pools[role].append({"dataset": "tusimple", "image_id": raw, "original_partition": original,
                                "sequence_id": sequence, "annotation_path": str(annotation.relative_to(BASE)).replace("\\", "/"),
                                "annotation_line": line_number,
                                "remote_image": f"TUSimple/{original}_set/{raw}",
                                "image_mirror": "manideep1108/tusimple"})
    pools["validation"] = pools["train"]
    return pools


def culane_candidates():
    pools = {}
    for role, original, filename in (("train", "train", "train_gt.txt"),
                                     ("validation", "val", "val_gt.txt"),
                                     ("evaluation", "test", "test.txt")):
        listing = DATA / "culane" / "list" / filename
        records = []
        for line_number, line in enumerate(listing.read_text(encoding="utf-8").splitlines(), 1):
            raw = line.split()[0].lstrip("/")
            driver = raw.split("/")[0]
            records.append({"dataset": "culane", "image_id": raw, "original_partition": original,
                            "sequence_id": str(PurePosixPath(raw).parent),
                            "partition_list": str(listing.relative_to(BASE)).replace("\\", "/"),
                            "partition_line": line_number,
                            "remote_image": f"CULane/{driver}/{raw}", "image_mirror": "manideep1108/culane"})
        pools[role] = records
    return pools


def bdd_candidates():
    pools = {}
    inventory = bdd_inventory()
    for original in ("train", "val"):
        rows = [{"dataset": "bdd100k", "image_id": name, "original_partition": original,
                 # Conservative grouping by shared recording prefix, retained in the manifest.
                 "sequence_id": name.split("-")[0],
                 "sequence_grouping": "conservative shared filename prefix; exact recording linkage unavailable",
                 "remote_image": f"{original}/images/{name}", "image_mirror": "marquis03/bdd100k"}
                for name in inventory[original]]
        if original == "train":
            pools["train"] = rows
        else:
            pools["validation"] = rows
            pools["evaluation"] = rows
    return pools


def select_disjoint(pools, seed, counts):
    """One random frame per random sequence group, with no group shared by roles.

    This is explicitly sequence-balanced sampling, not uniform image sampling.
    Validation is allocated first because CULane's smaller val pool shares videos
    with its original train pool. Official partitions are never reassigned.
    """
    rng = random.Random(seed)
    used = set()
    selected = []
    for role in ("validation", "evaluation", "train"):
        groups = {}
        for row in pools[role]:
            if row["sequence_id"] not in used:
                groups.setdefault(row["sequence_id"], []).append(row)
        keys = sorted(groups)
        rng.shuffle(keys)
        if len(keys) < counts[role]:
            raise ValueError(f"Only {len(keys)} disjoint groups available for {role}")
        for key in keys[:counts[role]]:
            candidates = sorted(groups[key], key=lambda x: x["image_id"])
            row = dict(rng.choice(candidates), role=role, sampling_seed=seed)
            selected.append(row)
            used.add(key)
    return selected


def download_row(row):
    row = dict(row)
    dataset = row["dataset"]
    relative = f"images/{row['original_partition']}/{row['image_id']}"
    image = fetch_file(dataset, row["image_mirror"], row["remote_image"], relative)
    row["image_path"] = str(image.relative_to(BASE)).replace("\\", "/")
    row["image_sha256"] = sha256(image)
    if dataset == "culane":
        raw = row["image_id"]
        label = str(PurePosixPath(raw).with_suffix(".lines.txt"))
        if row["original_partition"] in ("train", "val"):
            remote = f"CULane/annotations_new/{label}"
        else:
            remote = f"CULane/{raw.split('/')[0]}/{label}"
        annotation = fetch_file(dataset, "manideep1108/culane", remote, f"annotations/{label}")
        row["annotation_path"] = str(annotation.relative_to(BASE)).replace("\\", "/")
    elif dataset == "bdd100k":
        name = PurePosixPath(row["image_id"]).with_suffix(".json").name
        annotation = fetch_file(dataset, "happytren/bdd100k-labels",
                                f"100k/{row['original_partition']}/{name}",
                                f"annotations/{row['original_partition']}/{name}")
        item = json.loads(annotation.read_text(encoding="utf-8"))
        if item["name"] != PurePosixPath(row["image_id"]).stem:
            raise ValueError("BDD image/annotation identity mismatch")
        objects = item["frames"][0]["objects"]
        # Missing lanes are legitimate negatives; missing geometry in *all* data
        # is caught later by preprocessing QA.
        row["lane_object_count"] = sum(o["category"].startswith("lane/") for o in objects)
        row["annotation_path"] = str(annotation.relative_to(BASE)).replace("\\", "/")
    row["annotation_sha256"] = sha256(BASE / row["annotation_path"])
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train", type=int, default=80)
    parser.add_argument("--validation", type=int, default=20)
    parser.add_argument("--evaluation", type=int, default=40)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    args = parser.parse_args()
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    counts = {r: getattr(args, r) for r in ("train", "validation", "evaluation")}
    for dataset in args.datasets:
        manifest = MANIFESTS / f"{dataset}.json"
        if manifest.exists():
            print(f"Preserving existing frozen manifest {manifest}", flush=True)
            continue
        pools = {"tusimple": tusimple_candidates, "culane": culane_candidates, "bdd100k": bdd_candidates}[dataset]()
        rows = select_disjoint(pools, args.seed, counts)
        plan = MANIFESTS / f"{dataset}_selection.json"
        if plan.exists():
            saved = json.loads(plan.read_text(encoding="utf-8"))
            if saved != rows:
                raise ValueError("Selection plan changed; use a separate artifact directory")
        else:
            plan.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        completed = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            for row in executor.map(download_row, rows):
                completed.append(row)
                print(f"{dataset}: {len(completed)}/{len(rows)} {row['role']} {row['image_id']}", flush=True)
        manifest.write_text(json.dumps(completed, indent=2), encoding="utf-8")
        print(f"FROZEN {dataset}: {sha256(manifest)}", flush=True)


if __name__ == "__main__":
    main()
