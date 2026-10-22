"""Restore frozen real-image selections, including from a fresh checkout.

Existing selected IDs are never resampled. Every restored source file must
match the frozen manifest hash before it is accepted for a rerun.
"""
import concurrent.futures
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
from benchmark.acquire import fetch_file
from benchmark.prepare import DATASETS, MANIFESTS, download_row


def main():
    source_files = [
        ("tusimple", "manideep1108/tusimple", "TUSimple/test_label.json", "annotations/test_label.json"),
        *[("tusimple", "manideep1108/tusimple", f"TUSimple/train_set/label_data_{date}.json", f"annotations/label_data_{date}.json")
          for date in ("0313", "0531", "0601")],
        *[("culane", "manideep1108/culane", f"CULane/list/list/{name}", f"list/{name}")
          for name in ("train_gt.txt", "val_gt.txt", "test.txt")],
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for path in executor.map(lambda args: fetch_file(*args), source_files):
            print("SOURCE", path.relative_to(BASE), flush=True)
    if not all((MANIFESTS/f"{d}_selection.json").exists() for d in DATASETS):
        from benchmark.prepare import main as prepare
        prepare()
        return
    from benchmark.index_archives import main as index_archives
    index_archives()
    for dataset in DATASETS:
        manifest = MANIFESTS/f"{dataset}.json"
        if not manifest.exists():
            raise ValueError("Frozen manifests required for restoration; run benchmark.prepare to finish initial selection")
        frozen = json.loads(manifest.read_text(encoding="utf-8"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            for original, restored in zip(frozen, executor.map(download_row, frozen)):
                for field in ("image_id", "role", "original_partition", "image_sha256", "annotation_sha256"):
                    if original[field] != restored[field]:
                        raise ValueError(f"Restored data differs from frozen {field}: {original['image_id']}")
        print(f"RESTORED AND HASH-VERIFIED {dataset}: {len(frozen)} images", flush=True)


if __name__ == "__main__":
    main()
