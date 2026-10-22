"""Command dispatcher for the traceable real-image benchmark pipeline."""
from __future__ import annotations
import argparse
import importlib
import json
import runpy
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "qa", "train", "audit", "verify", "report", "preflight"])
    args, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0], *rest]
    root = Path(__file__).resolve().parent
    if args.command in ("verify", "report"):
        script = {"verify": "verify_benchmark.py", "report": "report_benchmark.py"}[args.command]
        runpy.run_path(str(root/"scripts"/script), run_name="__main__")
    elif args.command in ("audit", "preflight"):
        import torch
        from benchmark.data import audit_manifests
        from benchmark.prepare import DATASETS
        print(json.dumps({"device": "cuda" if torch.cuda.is_available() else "cpu", "torch": torch.__version__, "datasets": audit_manifests(DATASETS)}, indent=2))
    else:
        module = "run" if args.command == "train" else args.command
        importlib.import_module("benchmark."+module).main()

if __name__ == "__main__":
    main()
