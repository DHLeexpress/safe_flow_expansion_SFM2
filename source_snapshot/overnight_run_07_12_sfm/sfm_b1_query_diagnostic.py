"""Paired closed-loop query diagnostics for the two B1 execution selectors.

The same checkpoint, recent GP support, scenarios, gammas, and random seeds are
used in both runs.  Only the post-verification execution selector changes.
These traces are diagnostic-only and never enter D, D+, GP state, or training.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import os

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm as GPS
import sfm_b1_expand as BX
import sfm_b1_store as BS
import sfm_protocol as SP
import sfm_scene as SS


DIAGNOSTIC_GAMMAS = (0.1, 0.5, 1.0)


def _write_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2)
    os.replace(temporary, path)


def choose_shared_interaction_steps(traces_by_selector, scenarios, gammas=DIAGNOSTIC_GAMMAS):
    """Choose one shared time per scenario by minimum mean robot--pedestrian distance.

    The rule is declared before rendering and uses both selectors symmetrically.
    A step is eligible whenever every selector/gamma trace exists.  NVP cells
    remain visible rather than being filtered out to manufacture a successful
    picture.  This avoids visual curation and gives paired axes/snapshots.
    """
    indices = {}
    for selector, traces in traces_by_selector.items():
        indices[selector] = {
            (int(row["scenario_id"]), round(float(row["gamma"]), 8), int(row["step"])): row
            for row in traces
        }
    chosen = []
    for scenario in map(int, scenarios):
        common = None
        for selector, index in indices.items():
            for gamma in gammas:
                steps = {step for (sid, value, step) in index
                         if sid == scenario and abs(value - float(gamma)) <= 1e-8}
                common = steps if common is None else common & steps
        if not common:
            raise RuntimeError(f"no shared interaction step for scenario {scenario}")
        scored = []
        for step in sorted(common):
            distances = []
            for index in indices.values():
                for gamma in gammas:
                    row = index[(scenario, round(float(gamma), 8), step)]
                    distances.append(float(np.linalg.norm(
                        np.asarray(row["ped_xy"], float) - np.asarray(row["state"], float)[:2], axis=1
                    ).min()))
            scored.append((float(np.mean(distances)), int(step)))
        chosen.append(dict(
            scenario_id=scenario, step=min(scored)[1],
            rule="minimum mean robot-pedestrian center distance over both selectors and all displayed gammas",
        ))
    return chosen


def _run_selector(checkpoint, recent_dir, round_i, scenarios, *, selector, ell, cap,
                  scene_profile, device, verifier_workers, seed):
    policy, _ = GPS.load_sfm_policy(checkpoint, device=device)
    policy.eval()
    phi_policy = copy.deepcopy(policy).eval()
    for parameter in phi_policy.parameters():
        parameter.requires_grad_(False)
    recent = BS.RecentRounds(recent_dir, SP.W)
    recent.load_through(round_i)
    gp, gp_ids = BX.gp_from_recent(
        phi_policy, recent, ell=ell, cap=cap, lam=1.0e-2, phi_s=.9,
        device=device, seed=seed + 101,
    )
    environment = SS.scene_profile(scene_profile)
    replicas = [
        BX.Replica(
            scenario, gamma, n_ped=environment["n_ped"],
            ped_speed_range=tuple(environment["ped_speed_range"]),
        )
        for scenario in scenarios for gamma in DIAGNOSTIC_GAMMAS
    ]
    cfg = BX.ArmConfig(
        name=("A" if selector == "margin" else "B"), selector=selector, alpha=0.0,
        rounds=1, scene_profile=scene_profile, verifier_workers=verifier_workers, seed=seed,
    ).validate()
    beta, calibrated_ess = BX._initial_beta(phi_policy, gp, replicas, cfg, device, seed + 1009)
    generator = torch.Generator(device=device).manual_seed(seed + 2003)
    shard = BS.RoundShard(round_i + 1)
    with ProcessPoolExecutor(max_workers=verifier_workers) as executor:
        gather = BX.gather_macro_round(
            policy, phi_policy, gp, beta, replicas, cfg, shard, device, executor, generator,
            record_all_traces=True,
        )
    return gather["traces"], dict(
        selector=selector, environment=environment, checkpoint=os.path.abspath(checkpoint),
        checkpoint_sha256=BX._sha256_file(checkpoint), recent_dir=os.path.abspath(recent_dir),
        recent_through_round=int(round_i), gp_buffer_ids=gp_ids, gp=gp.diagnostics(),
        beta=float(beta), calibrated_ess_over_K=float(calibrated_ess),
        gather={key: value for key, value in gather.items() if key != "traces"},
    )


def run(checkpoint, recent_dir, round_i, *, scenarios, ell, cap, scene_profile,
        device, verifier_workers, seed, outdir):
    if scene_profile not in ("requested_ood", "density_ood"):
        raise ValueError("paired query diagnostic requires an explicit supported OOD profile")
    if len(tuple(scenarios)) != 3 or len(set(map(int, scenarios))) != 3:
        raise ValueError("exactly three distinct diagnostic scenarios are required")
    os.makedirs(outdir, exist_ok=True)
    traces_by_selector, reports = {}, {}
    for selector in ("margin", "safemppi_cost"):
        traces, report = _run_selector(
            checkpoint, recent_dir, round_i, tuple(map(int, scenarios)), selector=selector,
            ell=ell, cap=cap, scene_profile=scene_profile, device=device,
            verifier_workers=verifier_workers, seed=seed,
        )
        traces_by_selector[selector] = traces
        reports[selector] = report
        torch.save(traces, os.path.join(outdir, f"{selector}_traces.pt"))
    steps = choose_shared_interaction_steps(traces_by_selector, scenarios)
    # At t=0 both runs must have identical proposals and acquisition choices.
    for scenario in scenarios:
        for gamma in DIAGNOSTIC_GAMMAS:
            first = [row for row in traces_by_selector["margin"]
                     if row["scenario_id"] == scenario and row["gamma"] == gamma and row["step"] == 0]
            second = [row for row in traces_by_selector["safemppi_cost"]
                      if row["scenario_id"] == scenario and row["gamma"] == gamma and row["step"] == 0]
            if len(first) != 1 or len(second) != 1:
                raise RuntimeError("missing paired t=0 trace")
            if first[0]["selected_ids"] != second[0]["selected_ids"]:
                raise RuntimeError("selectors changed acquisition at the shared initial context")
            for left, right in zip(first[0]["all_K"], second[0]["all_K"]):
                if not np.array_equal(left["controls"], right["controls"]):
                    raise RuntimeError("selectors changed the shared initial K proposals")
    payload = dict(
        status="PAIRED_QUERY_DIAGNOSTIC_COMPLETE", diagnostic_only=True,
        enters_training_or_gp=False, scenarios=list(map(int, scenarios)),
        gammas=list(DIAGNOSTIC_GAMMAS), shared_interaction_steps=steps,
        shared_contract="same checkpoint, GP buffer, candidate seed, scenarios and initial K/B; only selector differs",
        reports=reports,
    )
    _write_json(os.path.join(outdir, "diagnostic.json"), payload)
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--recent-dir", required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--scenarios", type=int, nargs=3, required=True)
    parser.add_argument("--ell", type=float, required=True); parser.add_argument("--cap", type=int, required=True)
    parser.add_argument("--scene-profile", required=True, choices=("requested_ood", "density_ood"))
    parser.add_argument("--device", default="cuda"); parser.add_argument("--verifier-workers", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260720); parser.add_argument("--outdir", required=True)
    args = parser.parse_args(argv)
    run(
        args.checkpoint, args.recent_dir, args.round, scenarios=args.scenarios,
        ell=args.ell, cap=args.cap, scene_profile=args.scene_profile, device=args.device,
        verifier_workers=args.verifier_workers, seed=args.seed, outdir=args.outdir,
    )


if __name__ == "__main__":
    main()
