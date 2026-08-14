#!/usr/bin/env python3
"""Render one saved comparison-video frame as a vector PDF."""
from __future__ import annotations

import argparse
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SOURCE = os.path.join(ROOT, "source_snapshot", "overnight_run_07_12_sfm")
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

from sfm_six_row_compare import snapshot  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-bundle", required=True)
    parser.add_argument("--render-json", required=True)
    parser.add_argument("--frame-index", required=True, type=int)
    parser.add_argument("--output-pdf", required=True)
    args = parser.parse_args(argv)
    result = snapshot(
        args.trace_bundle, args.render_json, args.frame_index, args.output_pdf,
    )
    print(result["output_pdf"])


if __name__ == "__main__":
    main()
