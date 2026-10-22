"""Cache ZIP metadata only for previously selected images and annotations."""
import json
import zipfile
from pathlib import PurePosixPath

from benchmark.acquire import BASE, DATA, RemoteZip


def main():
    for dataset, mirror in (("bdd100k", "marquis03/bdd100k"), ("culane", "manideep1108/culane")):
        destination = DATA/dataset/"selected_archive_index.json"
        if destination.exists():
            continue
        rows = json.loads((BASE/"benchmark_artifacts/manifests"/f"{dataset}_selection.json").read_text())
        wanted = set()
        for row in rows:
            wanted.add(row["remote_image"])
            if dataset == "culane":
                label = str(PurePosixPath(row["image_id"]).with_suffix(".lines.txt"))
                wanted.add(f"CULane/annotations_new/{label}" if row["original_partition"] != "test"
                           else f"CULane/{row['image_id'].split('/')[0]}/{label}")
        with zipfile.ZipFile(RemoteZip(f"https://www.kaggle.com/api/v1/datasets/download/{mirror}")) as archive:
            entries = {info.filename: {"offset": info.header_offset, "compression": info.compress_type,
                                       "compressed_size": info.compress_size, "size": info.file_size, "crc": info.CRC}
                       for info in archive.infolist() if info.filename in wanted}
        if len(entries) != len(wanted):
            raise ValueError(f"Selected paths absent from {dataset} archive: {wanted-entries.keys()}")
        destination.write_text(json.dumps(entries, indent=2))
        print(dataset, len(entries), "selected archive entries indexed", flush=True)


if __name__ == "__main__":
    main()
