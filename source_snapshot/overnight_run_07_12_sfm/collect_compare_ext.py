"""Collect comparison cases under the re-targeting (video-only) crowd variant.

Same full-bank CRN replay as ``collect_compare.py`` but executed inside
``ped_extend.retargeting_pedestrians``.  Because the crowd now keeps walking,
the rollouts necessarily differ from the stored fixed-bank rows, so the stored
row is *not* used as an oracle: every case is validated against its own
replayed row (``trace_case`` still enforces the internal shape/validity
contract).  Nothing produced here may be reported as a fixed-bank metric.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_hp100_branch_viz as BRANCH
import sfm_hp100_eval as RAW
import sfm_hp100_paper_video_style as STYLE

import ped_extend


def _self_reference(row: dict) -> dict:
    verified = RAW._verify_executed_episode(row)
    return {
        "status": row["status"], "steps": int(row["steps"]),
        "min_clearance": row.get("min_clearance"),
        "successful_clearance": row.get("successful_clearance"),
        "time_to_goal": row.get("time_to_goal"),
        "validity": verified["validity"],
        "valid_windows": verified["valid_windows"],
    }


def _validity(row) -> float:
    traces = row.get("traces") or ()
    if not traces:
        return float("nan")
    positive = sum(t["proposal_label"] == "full_h_positive" for t in traces)
    return positive / len(traces)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--expected-sha", required=True)
    ap.add_argument("--ep0", type=int, default=920000)
    ap.add_argument("--episodes", required=True, help="comma separated")
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--evaluation", required=True,
                    help="this checkpoint's own M50 json; supplies the CRN noise "
                         "so the only change versus the delivered video is the crowd")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    gammas = (0.1, 0.5, 1.0)

    sha = STYLE.sha256_file(args.checkpoint)
    if sha != args.expected_sha:
        raise RuntimeError(f"checkpoint sha mismatch: {sha}")
    policy, _ckpt = GPS.load_sfm_hp100_policy(args.checkpoint, device=args.device)
    policy.eval()

    payload = json.loads(Path(args.evaluation).read_text())
    noise = BRANCH.validate_evaluation(
        payload, scene_profile="double_density_velocity_ood",
        ep0=int(args.ep0), checkpoint_sha256=sha, policy_d=policy.d,
    )

    with ped_extend.retargeting_pedestrians():
        replay = RAW.run_batched_raw(
            policy, scene_profile="double_density_velocity_ood",
            ep0=int(args.ep0), M=50, noise=noise, device=args.device,
            retain_proposals=True,
        )

    by_key = {
        (round(float(r["gamma"]), 8), int(r["episode"])): r for r in replay
    }
    ledger = []
    for ep in range(int(args.ep0), int(args.ep0) + 50):
        rows = [by_key[(round(float(g), 8), ep)] for g in gammas]
        entry = {
            "episode": ep,
            "status": {f"{g:g}": str(r["status"]) for g, r in zip(gammas, rows)},
            "steps": {f"{g:g}": int(r["steps"]) for g, r in zip(gammas, rows)},
            "validity": {f"{g:g}": _validity(r) for g, r in zip(gammas, rows)},
        }
        entry["allsucc"] = all(v == "success" for v in entry["status"].values())
        entry["allvalid"] = all(v >= 0.999 for v in entry["validity"].values())
        entry["fastest_last"] = bool(
            entry["steps"]["1"] <= entry["steps"]["0.5"]
            and entry["steps"]["1"] <= entry["steps"]["0.1"]
        )
        ledger.append(entry)

    cases, motion = {}, {}
    for ep in [int(x) for x in args.episodes.split(",")]:
        for g in gammas:
            row = by_key[(round(float(g), 8), ep)]
            key = f"ep{ep}_g{g:g}".replace(".", "p")
            case = BRANCH.trace_case(
                row, scene_profile="double_density_velocity_ood",
                rollout_index=int(ep - int(args.ep0)),
                expected_row=_self_reference(row),
            )
            case["crowd_variant"] = "retargeting_pedestrians (video only)"
            cases[key] = case
            motion[key] = ped_extend.crowd_motion_profile(
                [np.asarray(t["ped_xy"], float) for t in case["traces"]]
            )

    torch.save({
        "label": args.label, "checkpoint_sha256": sha, "ep0": int(args.ep0),
        "gammas": gammas, "cases": cases,
        "crowd_variant": "retargeting_pedestrians",
        "not_a_fixed_bank_metric": True,
    }, out / f"{args.label}_cases.pt")
    summary = {
        "label": args.label, "episodes": args.episodes,
        "crowd_motion": motion,
        "story_candidates": [
            e["episode"] for e in ledger if e["allsucc"] and e["allvalid"]
        ],
        "allsucc": [e["episode"] for e in ledger if e["allsucc"]],
        "requested": {
            f"ep{e['episode']}": {"status": e["status"], "steps": e["steps"],
                                  "validity": e["validity"]}
            for e in ledger if str(e["episode"]) in args.episodes.split(",")
        },
    }
    (out / f"{args.label}_selection.json").write_text(
        json.dumps({"summary": summary, "ledger": ledger}, indent=2,
                   allow_nan=True)
    )
    print(json.dumps(summary, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
