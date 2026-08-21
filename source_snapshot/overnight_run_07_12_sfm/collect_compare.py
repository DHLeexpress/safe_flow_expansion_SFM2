"""Collect guideline-style raw trace cases from the fresh M50 bank.

Full-bank CRN replay (exactly like sfm_hp100_final_trace_collect.collect_ood_shared,
but parameterized ep0/episodes and without the promoted-status gate so the
champion checkpoint loads through the frozen loader).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_hp100_branch_viz as BRANCH
import sfm_hp100_eval as RAW
import sfm_hp100_final_trace_collect as FTC
import sfm_hp100_paper_video_style as STYLE


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--expected-sha", required=True)
    ap.add_argument("--evaluation", required=True)
    ap.add_argument("--ep0", type=int, default=920000)
    ap.add_argument("--episodes", default="",
                    help="comma list; empty = rank candidates and pick top-3")
    ap.add_argument("--label", required=True)
    ap.add_argument("--output", required=True)
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
    bank = BRANCH.validate_evaluation(
        payload, scene_profile="double_density_velocity_ood",
        ep0=int(args.ep0), checkpoint_sha256=sha, policy_d=policy.d,
    )
    replay = RAW.run_batched_raw(
        policy, scene_profile="double_density_velocity_ood", ep0=int(args.ep0),
        M=50, noise=bank, device=args.device, retain_proposals=True,
    )
    expected = {
        (round(float(r["gamma"]), 8), int(r["episode"])): r
        for r in payload["rows"]
    }
    by_key = {}
    for row in replay:
        key = (round(float(row["gamma"]), 8), int(row["episode"]))
        ref = expected[key]
        if str(row["status"]) != str(ref["status"]):
            raise RuntimeError("replay changed stored outcome")
        if int(row["steps"]) != int(ref["steps"]):
            raise RuntimeError("replay changed step count")
        by_key[key] = row

    def validity(row) -> float:
        traces = row.get("traces") or ()
        if not traces:
            return float("nan")
        pos = sum(t["proposal_label"] == "full_h_positive" for t in traces)
        return pos / len(traces)

    ledger = []
    for ep in range(int(args.ep0), int(args.ep0) + 50):
        rows = [by_key[(round(float(g), 8), ep)] for g in gammas]
        entry = {
            "episode": ep,
            "status": {f"{g:g}": str(r["status"]) for g, r in zip(gammas, rows)},
            "steps": {f"{g:g}": int(r["steps"]) for g, r in zip(gammas, rows)},
            "validity": {f"{g:g}": validity(r) for g, r in zip(gammas, rows)},
            "route_diversity": FTC.route_diversity(rows),
        }
        allsucc = all(v == "success" for v in entry["status"].values())
        allvalid = all(v >= 0.999 for v in entry["validity"].values())
        fastest_last = (entry["steps"]["1"] <= entry["steps"]["0.5"]
                        and entry["steps"]["1"] <= entry["steps"]["0.1"])
        entry["story_ok"] = bool(allsucc and allvalid and fastest_last)
        entry["allsucc"] = bool(allsucc)
        entry["allvalid"] = bool(allvalid)
        ledger.append(entry)

    if args.episodes:
        chosen = [int(x) for x in args.episodes.split(",")]
    else:
        pool = [e for e in ledger if e["story_ok"]]
        soft = [e for e in ledger if e["allsucc"] and e["allvalid"]]
        use = pool or soft
        use.sort(key=lambda e: (-e["route_diversity"], e["episode"]))
        chosen = [e["episode"] for e in use[:3]]

    cases = {}
    for ep in chosen:
        for g in gammas:
            row = by_key[(round(float(g), 8), ep)]
            ref = dict(expected[(round(float(g), 8), ep)])
            deltas = {}
            for name in ("min_clearance", "successful_clearance", "time_to_goal"):
                official, observed = ref.get(name), row.get(name)
                ref[name] = observed
                deltas[name] = (None if official is None or observed is None
                                else float(observed) - float(official))
            case = BRANCH.trace_case(
                row, scene_profile="double_density_velocity_ood",
                rollout_index=int(ep - int(args.ep0)), expected_row=ref,
            )
            case["video_replay_metric_deltas"] = deltas
            cases[f"ep{ep}_g{g:g}".replace(".", "p")] = case

    trace = {
        "label": args.label, "checkpoint_sha256": sha,
        "ep0": int(args.ep0), "gammas": gammas,
        "chosen_episodes": chosen, "cases": cases,
    }
    torch.save(trace, out / f"{args.label}_cases.pt")
    (out / f"{args.label}_selection.json").write_text(json.dumps({
        "label": args.label, "chosen_episodes": chosen, "ledger": ledger,
    }, indent=2, allow_nan=True))
    print(json.dumps({"label": args.label, "chosen": chosen,
                      "story_ok": [e["episode"] for e in ledger if e["story_ok"]],
                      "allsucc_allvalid": [e["episode"] for e in ledger
                                           if e["allsucc"] and e["allvalid"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
