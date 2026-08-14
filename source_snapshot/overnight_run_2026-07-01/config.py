"""Central knobs (change ideas here). TWO scene configurations — run the whole pipeline for each.

  gap    (上下): two obstacles stacked vertically → SafeMPPI THREADS the gap (γ: center↔upper/lower).
  slalom (左右): two obstacles sequential (A upper-left, B lower-right) → SafeMPPI GOES AROUND;
                 expansion + the fitted verifier discover the weave / tighter modes.
Both use POINT robot (r_robot=0, matches the sm=0 best_area_mode4 planner; NO safety_margin) and the
ORIGINAL ieee unbounded max-margin verifier (it EXPANDS to certify what the fixed nominal polytope rejects).
"""
from __future__ import annotations

import os

import _paths

# --- scene registry ---
SCENES = {
    "gap":    dict(kind="gap",    gap_offset=0.55, gap_r=0.35, r_robot=0.0, T=80),
    "slalom": dict(kind="slalom", ax=2.4, ay=0.45, bx=3.6, by=-0.45, r=0.55, r_robot=0.0, T=80),
}
SCENE_NAMES = list(SCENES)

# --- horizons (decision variables; future sweep H∈{5,10,15,20}) ---
H_PRED = 10          # MPPI expert window & FM output horizon
H_EXEC = 1           # executed FM controls per replan
VERIFIER_WINDOW = 10

# --- gamma grid ---
GAMMAS = [0.1, 0.5, 1.0]

# --- verifier knobs (ORIGINAL ieee unbounded max-margin; m_max=None) ---
VERIFIER = dict(gamma_max=0.7, m_max=None, K=12, rho_art=0.16, R=2.5, H_win=VERIFIER_WINDOW, stride=2)

# --- paths / W&B ---
ROOT = _paths.ROOT
RESULTS = os.path.join(_paths.HERE, "results")
FIGURES = os.path.join(_paths.HERE, "figures")
WANDB_PROJECT = "cfm-mppi-safeflow"


def make_scene(name, T=None, device="cpu"):
    import scenes
    s = dict(SCENES[name])
    kind = s.pop("kind")
    if T is not None:
        s["T"] = T
    if kind == "gap":
        return scenes.make_narrow_gap(**s, device=device)
    if kind == "slalom":
        return scenes.make_slalom(**s, device=device)
    return scenes.make_single_obstacle(**s, device=device)


def dataset_dir(name):
    return os.path.join(ROOT, "dataset", f"windowed_{name}")


def scene_fig(name, fname):
    d = os.path.join(FIGURES, name)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, fname)


def scene_result(name, fname):
    d = os.path.join(RESULTS, name)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, fname)
