"""Run the locked HP100 Kazuki comparator on chosen episodes/gammas.

Scoped wrappers capture, per step, the selected control window (for the
audit-only exact verification badge/polytope) and the integrated guidance
components (for the goal/safety arrows). No frozen file is edited; the
comparator's execution path is byte-identical.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_hp100_branch_viz as BRANCH
import sfm_hp100_kazuki as KAZ


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--episodes", required=True)
    ap.add_argument("--gammas", default="0.1,0.5,1.0")
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--goal-coef", type=float, default=None,
                    help="override the locked 0.5 (reported in provenance)")
    ap.add_argument("--safe-coef", type=float, default=None,
                    help="override the locked 0.3 (reported in provenance)")
    args = ap.parse_args()

    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    policy, _ = GPS.load_sfm_hp100_policy(args.checkpoint, device=args.device)
    policy.eval()

    episodes = [int(x) for x in args.episodes.split(",")]
    gammas = [float(x) for x in args.gammas.split(",")]

    original_refine = KAZ.BASE.flow_mppi_refine
    original_generate = KAZ.BASE.guided_generate
    original_locked = KAZ.locked_config

    if args.goal_coef is not None or args.safe_coef is not None:
        def patched_locked():
            config = original_locked()
            overrides = {}
            if args.goal_coef is not None:
                overrides["goal_coef"] = float(args.goal_coef)
            if args.safe_coef is not None:
                overrides["safe_coefs"] = (float(args.safe_coef),)
            return KAZ.BASE.replace(config, **overrides).validate()
        KAZ.locked_config = patched_locked

    summary = []
    try:
        for episode in episodes:
            for gamma in gammas:
                windows, components_log = [], []

                def refine_wrapper(*a, **k):
                    selected, diag = original_refine(*a, **k)
                    windows.append(
                        selected.detach().cpu().numpy().astype(np.float32)
                    )
                    return selected, diag

                def generate_wrapper(*a, **k):
                    z, trace, unguided, components = original_generate(*a, **k)
                    components_log.append({
                        "goal": components["goal"].detach().cpu(),
                        "safety": components["safety"].detach().cpu(),
                    })
                    return z, trace, unguided, components

                KAZ.BASE.flow_mppi_refine = refine_wrapper
                KAZ.BASE.guided_generate = generate_wrapper
                try:
                    run = KAZ.kazuki_hp100_deploy(
                        policy, episode, gamma,
                        scene_profile="double_density_velocity_ood",
                        device=args.device, collect_diagnostics=True,
                    )
                finally:
                    KAZ.BASE.flow_mppi_refine = original_refine
                    KAZ.BASE.guided_generate = original_generate

                horizon = int(policy.H_pred)
                u_max = float(policy.u_max)
                audits = []
                for row, window, components in zip(
                    run["trace"], windows, components_log,
                ):
                    result = BRANCH.verify_raw_proposal(
                        row["state"], window, row["pedestrian_xy"],
                        row["pedestrian_velocity"], gamma,
                    )
                    seed = int(row["refinement"]["selected_generated_index"])
                    goal_action = (
                        components["goal"][seed].reshape(horizon, 2)[0].numpy()
                        * u_max
                    )
                    safety_action = (
                        components["safety"][seed].reshape(horizon, 2)[0].numpy()
                        * u_max
                    )
                    audits.append({
                        "step": int(row["step"]),
                        "selected_window": window,
                        "verify": result,
                        "goal_guidance_action": goal_action.astype(np.float32),
                        "safety_guidance_action": safety_action.astype(np.float32),
                    })
                run["audits"] = audits
                run["goal_coef_override"] = args.goal_coef
                key = f"ep{episode}_g{gamma:g}".replace(".", "p")
                torch.save(run, out / f"kazuki_{key}.pt")
                valid_steps = sum(
                    1 for a in audits
                    if a["verify"].get("resolved")
                    and int(a["verify"].get("y", 0)) == 1
                )
                summary.append({
                    "episode": episode, "gamma": gamma,
                    "success": bool(run["success"]),
                    "collision": bool(run["collision"]),
                    "steps": int(run["steps"]),
                    "min_clear": float(run["min_clear"]),
                    "valid_steps": valid_steps,
                    "valid_fraction": (valid_steps / len(audits)
                                       if audits else None),
                })
                print(json.dumps(summary[-1]))
    finally:
        KAZ.locked_config = original_locked

    (out / "KAZUKI_SUMMARY.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
