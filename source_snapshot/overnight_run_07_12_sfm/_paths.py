"""Local import bootstrap for the isolated SFM Hp10/B1 study."""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RUN_02 = os.path.join(ROOT, "overnight_run_2026-07-02")
RUN_01 = os.path.join(ROOT, "overnight_run_2026-07-01")
RUN_TODAY_SRC = os.path.join(ROOT, "overnight_run_today", "src")
RUN_28 = os.path.join(ROOT, "overnight_run_2026-06-28")
IEEE_SRC = os.path.join(ROOT, "ieee_compact_polytope_verifier_package", "src")
BEST_CONFIG = os.path.join(RUN_28, "best_area_mode4.json")

for _path in (HERE, RUN_02, RUN_01, RUN_TODAY_SRC, RUN_28, IEEE_SRC, ROOT):
    if _path not in sys.path:
        sys.path.insert(0, _path)
