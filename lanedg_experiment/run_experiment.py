"""Compatibility entry point for the real-image semantic benchmark runner."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.run import main
if __name__ == "__main__":
    main()
