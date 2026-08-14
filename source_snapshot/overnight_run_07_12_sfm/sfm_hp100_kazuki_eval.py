"""Evaluate the locked HP100 Kazuki comparator on an M/gamma CRN bank."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess

import numpy as np

import _paths  # noqa: F401
import sfm_hp100_dynamics as DYN
import sfm_hp100_eval as RAW
import sfm_hp100_features as HPF
import sfm_hp100_kazuki as KZ
import sfm_scene as SS


VERSION = "sfm_hp100_kazuki_fixed_bank_v1"


def _git_provenance() -> dict:
    root = Path(__file__).resolve().parents[1]
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    return dict(root=str(root), head=head, clean=not bool(status), status=status)


def _source_hashes() -> dict:
    modules = {
        "evaluator": __file__,
        "hp100_kazuki": KZ.__file__,
        "legacy_kazuki": KZ.BASE.__file__,
        "raw_evaluator": RAW.__file__,
        "dynamics": DYN.__file__,
        "features": HPF.__file__,
        "scene": SS.__file__,
    }
    return {
        name: dict(path=str(Path(path).resolve()), sha256=RAW.sha256_file(path))
        for name, path in modules.items()
    }


def _preflight(checkpoint, expected_checkpoint_sha256, output, expected_source_commit):
    checkpoint = Path(checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    observed_checkpoint_sha256 = RAW.sha256_file(checkpoint)
    if observed_checkpoint_sha256 != str(expected_checkpoint_sha256):
        raise RuntimeError(
            "checkpoint SHA-256 mismatch: "
            f"{observed_checkpoint_sha256} != {expected_checkpoint_sha256}"
        )
    output = Path(output).resolve()
    temporary = Path(str(output) + ".tmp")
    for candidate in (output, temporary):
        if os.path.lexists(candidate):
            raise FileExistsError(f"refusing to overwrite existing output: {candidate}")
    git = _git_provenance()
    if git["head"] != str(expected_source_commit):
        raise RuntimeError(
            f"source commit {git['head']} != expected {expected_source_commit}"
        )
    if not git["clean"]:
        raise RuntimeError("HP100 Kazuki evaluation requires a clean frozen worktree")
    return dict(
        checkpoint=str(checkpoint), checkpoint_sha256=observed_checkpoint_sha256,
        output=str(output), temporary=str(temporary), git=git,
        source_hashes=_source_hashes(),
    )


def _assert_inputs_unchanged(initial: dict) -> None:
    if _git_provenance() != initial["git"]:
        raise RuntimeError("source Git provenance changed during HP100 Kazuki evaluation")
    if _source_hashes() != initial["source_hashes"]:
        raise RuntimeError("source files changed during HP100 Kazuki evaluation")
    if RAW.sha256_file(initial["checkpoint"]) != initial["checkpoint_sha256"]:
        raise RuntimeError("checkpoint changed during HP100 Kazuki evaluation")


def _rollout_gamma(payload):
    checkpoint, scene_profile, ep0, M, gamma, device = payload
    import grid_policy_sfm_hp100 as GPS

    policy, _ = GPS.load_sfm_hp100_policy(checkpoint, device=device)
    policy.eval()
    rows = []
    for episode in range(int(ep0), int(ep0) + int(M)):
        rollout = KZ.kazuki_hp100_deploy(
            policy, episode, float(gamma), scene_profile=scene_profile,
            T=RAW.T, device=device, sample_seed=KZ.SAMPLE_SEED,
            collect_diagnostics=False,
        )
        success = bool(rollout["success"])
        collision = bool(rollout["collision"])
        steps = int(rollout["steps"])
        rows.append(dict(
            episode=int(episode), gamma=float(gamma),
            status=("success" if success else "collision" if collision else "timeout"),
            success=success, collision=collision,
            timeout=bool(not success and not collision), steps=steps,
            time_to_goal=(steps * DYN.DT if success else None),
            min_clearance=float(rollout["min_clear"]),
            successful_clearance=(float(rollout["min_clear"]) if success else None),
            states=np.asarray(rollout["states"], np.float32),
            controls=np.asarray(rollout["controls"], np.float32),
            ped_xy=np.asarray(rollout["peds"], np.float32),
            ped_vel=np.asarray(rollout["ped_vels"], np.float32),
        ))
    return RAW.attach_validity(rows)


def evaluate(
    checkpoint,
    *,
    scene_profile,
    ep0,
    M,
    device,
    workers=1,
):
    checkpoint = os.path.abspath(checkpoint)
    payloads = [
        (checkpoint, scene_profile, int(ep0), int(M), float(gamma), device)
        for gamma in SS.GAMMAS
    ]
    if int(workers) == 1:
        cells = [_rollout_gamma(payload) for payload in payloads]
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=min(len(payloads), int(workers)), mp_context=context,
        ) as executor:
            cells = list(executor.map(_rollout_gamma, payloads))
    rows = [row for cell in cells for row in cell]
    return rows, RAW.summarize(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument(
        "--scene-profile", required=True,
        choices=SS.SCIENTIFIC_EVAL_PROFILES,
    )
    parser.add_argument("--ep0", required=True, type=int)
    parser.add_argument("--M", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    if args.M <= 0 or args.workers <= 0:
        parser.error("--M and --workers must be positive")

    inputs = _preflight(
        args.checkpoint, args.expected_checkpoint_sha256, args.out,
        args.expected_source_commit,
    )

    rows, summary = evaluate(
        inputs["checkpoint"], scene_profile=args.scene_profile, ep0=args.ep0,
        M=args.M, device=args.device, workers=args.workers,
    )
    _assert_inputs_unchanged(inputs)
    config = KZ.locked_config()
    payload = dict(
        status="SFM_HP100_KAZUKI_EVAL_COMPLETE", version=VERSION,
        method="locked Kazuki generate-guide-refine on HP100 prior",
        checkpoint=inputs["checkpoint"],
        checkpoint_sha256=inputs["checkpoint_sha256"],
        source=dict(git=inputs["git"], files=inputs["source_hashes"]),
        scene=SS.scene_profile(args.scene_profile),
        bank=dict(
            ep0=int(args.ep0), M_per_gamma=int(args.M),
            same_scenario_ids_for_every_gamma=True,
            pedestrian_seeding="SS.make_humans(episode, seed=0, profile)",
            kazuki_sample_seed=int(KZ.SAMPLE_SEED),
        ),
        locked_guidance=dict(
            safe_coef=float(KZ.SAFE_COEF), goal_coef=float(KZ.GOAL_COEF),
        ),
        kazuki_config=config.to_dict(), dynamics=DYN.contract(),
        observation=HPF.contract(), summary=summary, rows=rows,
        metric_semantics=dict(
            Validity=(
                "terminal-truncated executed-window mean; exact GREEN verifier "
                "with clipped HP100 robot rollout"
            ),
            comparator=(
                "same HP100 learned prior plus locked ODE goal/CBF guidance and "
                "MPPI refinement; no shield, templates, privileged lookahead, "
                "fallback, or gamma retuning"
            ),
        ),
    )
    path = inputs["output"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = inputs["temporary"]
    try:
        with open(temporary, "x") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
        # link(2) creates the destination only if it is still absent.  Unlike
        # os.replace, it cannot silently overwrite a result created mid-run.
        os.link(temporary, path)
    finally:
        if os.path.lexists(temporary):
            os.unlink(temporary)
    print(json.dumps({
        "status": payload["status"], "out": path,
        "pooled": summary["pooled"],
    }), flush=True)


if __name__ == "__main__":
    main()
