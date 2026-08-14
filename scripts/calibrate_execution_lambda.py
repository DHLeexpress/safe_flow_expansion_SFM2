#!/usr/bin/env python3
"""CLI entry point for the task-agnostic execution-lambda preflight."""

from __future__ import annotations

import sys
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "source_snapshot" / "overnight_run_07_12_sfm"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

from execution_lambda_preflight import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
