"""Download individual public mirror files, retaining hashes and provenance.

This module never generates road images or ground-truth annotations.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
import struct
import zlib
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
DATA = BASE / "datasets"


class RemoteZip(io.RawIOBase):
    """Seekable HTTP Range reader; refuses servers that return the whole archive."""
    def __init__(self, url):
        self.url = url
        self.position = 0
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
        with urllib.request.urlopen(req, timeout=90) as response:
            if response.status != 206:
                raise ValueError("Server does not support range requests")
            self.size = int(response.headers["Content-Range"].split("/")[-1])

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = offset + (0 if whence == 0 else self.position if whence == 1 else self.size)
        return self.position

    def read(self, count=-1):
        count = min(self.size - self.position, count if count >= 0 else self.size)
        if count <= 0:
            return b""
        req = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-{self.position+count-1}"})
        with urllib.request.urlopen(req, timeout=90) as response:
            if response.status != 206:
                raise ValueError("Server ignored range request")
            data = response.read()
        if len(data) != count:
            raise ValueError("Truncated range response")
        self.position += count
        return data


def request(url: str, attempts: int = 4) -> bytes:
    for attempt in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "LaneDG-reproducibility/1.0"})
            with urllib.request.urlopen(req, timeout=90) as response:
                return response.read()
        except Exception as error:
            if isinstance(error, urllib.error.HTTPError) and error.code in (403, 404):
                raise
            if attempt == attempts - 1:
                raise
            time.sleep(1 + attempt)
    raise RuntimeError("Unreachable")


def archive_member(dataset, mirror, remote):
    index_path = DATA / dataset / "selected_archive_index.json"
    if not index_path.exists():
        raise FileNotFoundError("No selected archive index available")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    item = index[remote]
    reader = RemoteZip(f"https://www.kaggle.com/api/v1/datasets/download/{mirror}")
    reader.seek(item["offset"])
    header = reader.read(30)
    if header[:4] != b"PK\x03\x04":
        raise ValueError("Invalid archive member header")
    name_length, extra_length = struct.unpack_from("<HH", header, 26)
    reader.seek(item["offset"]+30+name_length+extra_length)
    compressed = reader.read(item["compressed_size"])
    data = zlib.decompress(compressed, -15) if item["compression"] == 8 else compressed
    if len(data) != item["size"] or zlib.crc32(data) != item["crc"]:
        raise ValueError("Archive member size/CRC mismatch")
    return data


def fetch_file(dataset: str, mirror: str, remote: str, local: str | None = None) -> Path:
    relative = Path(local or remote)
    root = (DATA / dataset).resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Download path escapes dataset directory")
    metadata = target.with_name(target.name + ".provenance.json")
    if target.exists() and metadata.exists():
        saved = json.loads(metadata.read_text())
        if hashlib.sha256(target.read_bytes()).hexdigest() == saved["sha256"]:
            return target
        raise ValueError(f"Cached download hash mismatch: {target}")
    url = f"https://www.kaggle.com/api/v1/datasets/download/{mirror}?" + urllib.parse.urlencode({"fileName": remote})
    try:
        data = request(url)
        retrieval = "individual file"
    except urllib.error.HTTPError as error:
        if error.code != 404:
            raise
        data = archive_member(dataset, mirror, remote)
        retrieval = "HTTP Range extraction from full archive; CRC checked"
    if data[:4] == b"PK\x03\x04":
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            files = [f for f in archive.infolist() if not f.is_dir()]
            if len(files) != 1:
                raise ValueError(f"Expected one file, received {len(files)} for {remote}")
            data = archive.read(files[0])
    if data.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        raise ValueError(f"Received HTML instead of {remote}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".part")
    temporary.write_bytes(data)
    temporary.replace(target)
    metadata.write_text(json.dumps({"mirror": mirror, "remote_path": remote, "url": url, "retrieval": retrieval,
                                    "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}, indent=2))
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset")
    parser.add_argument("mirror")
    parser.add_argument("remote")
    parser.add_argument("--local")
    args = parser.parse_args()
    print(fetch_file(args.dataset, args.mirror, args.remote, args.local), flush=True)


if __name__ == "__main__":
    main()
