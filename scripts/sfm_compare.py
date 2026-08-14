#!/usr/bin/env python3
"""Local entry point for collecting/rendering the six-row SFM comparison."""
from __future__ import annotations

import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE = os.path.join(ROOT, "source_snapshot", "overnight_run_07_12_sfm")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

from sfm_six_row_compare import main  # noqa: E402


if __name__ == "__main__":
    main()
