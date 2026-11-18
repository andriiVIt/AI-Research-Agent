"""pytest setup. Lets the tests run from a fresh checkout even before `pip install -e .`.

The tests themselves are plain unittest cases, so `python -m unittest` works too
(after installing the package, or with PYTHONPATH=src).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
