"""Test package bootstrap for variopt.

The repository uses a src-layout package. Test modules are imported through the
``tests`` package so the package initializer can add ``src`` to ``sys.path``
once without per-file path hacks or import-order suppressions.
"""

import sys
from pathlib import Path

_SRC_PATH = Path(__file__).resolve().parents[1] / "src"
_SRC_PATH_STR = str(_SRC_PATH)
if _SRC_PATH_STR not in sys.path:
    sys.path.insert(0, _SRC_PATH_STR)
