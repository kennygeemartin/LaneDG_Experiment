"""Safe bootstrap controller for the LaneDG cross-dataset experiment."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


DATASETS = {
    "tusimple": ("manideep1108/tusimple", 24_304_438_810),
    "culane": ("manideep1108/culane", 46_058_728_472),
    "bdd100k": ("marquis03/bdd100k", 7_451_243_332),
}
EXTRACT_HEADROOM = 1.35
FULL_EXPERIMENT_MIN_BYTES = 120_000_000_000


def gib(value: int) -> str:
    return f"{value / 1024**3:.1f} GiB"


def torch_status() -> tuple[bool, str]:
    try:
        import torch
    except ImportError:
        return False, "PyTorch is not installed"
    available = torch.cuda.is_available()
    detail = torch.cuda.get_device_name(0) if available else f"torch {torch.__version__} (CPU only)"
    return available, detail


def preflight(data_root: Path) -> int:
    data_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(data_root).free
    cuda, device = torch_status()
    compressed = sum(item[1] for item in DATASETS.values())
    checks = {
        "data_root": str(data_root.resolve()),
        "free_space": gib(free),
        "compressed_downloads": gib(compressed),
        "minimum_working_space": gib(FULL_EXPERIMENT_MIN_BYTES),
        "cuda": cuda,
        "device": device,
        "lanedg_source_present": Path("model/lanedg.py").is_file(),
        "ready": free >= FULL_EXPERIMENT_MIN_BYTES and cuda and Path("model/lanedg.py").is_file(),
    }
    print(json.dumps(checks, indent=2))
    if not checks["ready"]:
        print("\nBLOCKED: satisfy disk, CUDA, and exact LaneDG-source requirements before training.")
        return 2
    return 0


def download(name: str, data_root: Path) -> int:
    handle, compressed_size = DATASETS[name]
    data_root.mkdir(parents=True, exist_ok=True)
    required = int(compressed_size * EXTRACT_HEADROOM)
    free = shutil.disk_usage(data_root).free
    if free < required:
        print(f"Refusing {name}: {gib(free)} free; at least {gib(required)} required.", file=sys.stderr)
        return 2
    try:
        import kagglehub
    except ImportError:
        print("Install kagglehub first: python -m pip install kagglehub", file=sys.stderr)
        return 2
    resolved = kagglehub.dataset_download(handle, output_dir=str(data_root / name))
    locations_file = Path("data/locations.json")
    locations_file.parent.mkdir(parents=True, exist_ok=True)
    locations = json.loads(locations_file.read_text()) if locations_file.exists() else {}
    locations[name] = str(Path(resolved).resolve())
    locations_file.write_text(json.dumps(locations, indent=2) + "\n")
    print(f"Downloaded {name} to {resolved}")
    return 0


def audit(data_root: Path) -> int:
    image_ext = {".jpg", ".jpeg", ".png"}
    report = {}
    for name in DATASETS:
        root = data_root / name
        files = list(root.rglob("*")) if root.exists() else []
        images = sum(p.suffix.lower() in image_ext for p in files if p.is_file())
        annotations = sum(p.suffix.lower() in {".json", ".txt"} for p in files if p.is_file())
        report[name] = {"exists": root.exists(), "images": images, "annotation_files": annotations}
    print(json.dumps(report, indent=2))
    missing = [name for name, item in report.items() if not item["exists"] or not item["images"]]
    if missing:
        print(f"\nIncomplete datasets: {', '.join(missing)}", file=sys.stderr)
        return 2
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("preflight", "audit"):
        p = sub.add_parser(command)
        p.add_argument("--data-root", type=Path, default=Path("datasets"))
    p = sub.add_parser("download")
    p.add_argument("dataset", choices=DATASETS)
    p.add_argument("--data-root", type=Path, default=Path("datasets"))
    args = parser.parse_args()
    if args.command == "preflight":
        return preflight(args.data_root)
    if args.command == "download":
        return download(args.dataset, args.data_root)
    return audit(args.data_root)


if __name__ == "__main__":
    raise SystemExit(main())


