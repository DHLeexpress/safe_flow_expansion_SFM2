"""STAGE 1 — narrow-gap SafeMPPI γ-sweep sanity viz (the approval gate).

Deploys the frozen best_area_mode4 planner on the STATIC narrow-gap scene at γ∈{0.1,0.5,1.0}, renders
di_grid-style (blue nominal polytope + 3-mode accept/reject + executed ✗ + PERSISTENT BLACK-DOT trail),
and reports whether the conservative planner THREADS the gap (expected: NO — it should detour).

    python stage1_narrow_gap_sweep.py
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import _paths
import scenes
import di_grid_viz as V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.1, 0.5, 1.0])
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--gap-offset", type=float, default=0.75)
    ap.add_argument("--gap-r", type=float, default=0.35)
    ap.add_argument("--out", default=os.path.join(_paths.HERE, "figures", "viz_narrow_gap_sweep"))
    ap.add_argument("--no-mp4", action="store_true")
    args = ap.parse_args()

    cfg = V.load_best_config()
    env = scenes.make_narrow_gap(gap_offset=args.gap_offset, gap_r=args.gap_r, T=args.steps)
    gx, hc, hb, outer = scenes.gap_geometry(env)
    print(f"=== STAGE 1: narrow-gap γ-sweep ===", flush=True)
    print(f"scene: obstacles (3, ±{args.gap_offset}) r={args.gap_r} | robot-center corridor |y|≤{hc:.2f} "
          f"(half_body {hb:.2f}, around ≥{outer:.2f}) | start (0,0)→goal (6,0)", flush=True)

    data, verdicts = {}, {}
    for g in args.gammas:
        rec, path = V.mppi_rollout(env, g, cfg, steps=args.steps, seed_base=int(g * 1000))
        data[g] = (rec, path)
        tg = scenes.threads_gap(path, env)
        verdicts[g] = tg
        state = ("THREADED gap" if tg["threaded"] else
                 "went AROUND" if tg["went_around"] else
                 "did not reach gap plane" if not tg["reached_gap_plane"] else "grazed (between)")
        cy = "n/a" if tg["cross_y"] is None else f"{tg['cross_y']:+.2f}"
        final = np.linalg.norm(path[-1] - env.goal.detach().cpu().numpy())
        print(f"  γ={g}: {state}  (cross_y={cy}, final_dist_to_goal={final:.2f})", flush=True)

    V.render_grid(env, data, args.gammas, args.out, polytope_mode="nominal",
                  title="SafeMPPI narrow-gap γ-sweep (blue = nominal polytope level sets)",
                  mp4=not args.no_mp4)

    # ---- verdict summary ----
    any_thread = any(v["threaded"] for v in verdicts.values())
    print("\n=== VERDICT ===", flush=True)
    for g, v in verdicts.items():
        print(f"  γ={g}: threaded={v['threaded']} around={v['went_around']} cross_y="
              f"{'n/a' if v['cross_y'] is None else round(v['cross_y'],2)}", flush=True)
    if any_thread:
        print("⚠️  At least one γ THREADED the narrow gap — NOT the expected conservative behavior. "
              "Flagging for the user (may need a tighter gap or lower γ).", flush=True)
    else:
        print("✅ As expected: the conservative planner does NOT thread the narrow gap at any γ "
              "(it detours / avoids). This leaves the gap-threading behavior for SAFE EXPANSION to discover.", flush=True)
    with open(os.path.join(_paths.HERE, "results", "stage1_verdict.json"), "w") as f:
        json.dump({str(g): {k: (None if v[k] is None else v[k]) for k in
                            ["threaded", "went_around", "reached_gap_plane", "cross_y", "half_center"]}
                   for g, v in verdicts.items()}, f, indent=2)


if __name__ == "__main__":
    main()
