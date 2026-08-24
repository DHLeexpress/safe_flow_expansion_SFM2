#!/usr/bin/env python
"""Kazuki (CFM-MPPI) gamma-cluster deployment runner (ADDITIVE).

Sixth series of the success-conditioned ``clearance x time-to-goal`` cluster
figure.  This module is a thin driver: the controller itself is the LOCKED
HP100 Kazuki comparator ``sfm_hp100_kazuki.kazuki_hp100_deploy`` imported
read-only and called unchanged (``sfm_hp100_kazuki.py``, ``sfm_kazuki.py``,
``sfm_hp100_eval.py``, ``sfm_scene.py``, ``sfm_hp100_dynamics.py``,
``sfm_hp100_features.py`` and ``grid_policy_sfm_hp100.py`` are never edited or
monkeypatched).  The locked recipe is

    guidance   v + goal_coef * grad_goal + safe_coef * rho_H * grad_CBF
               with the locked defaults ``goal_coef = 0.5``, ``safe_coef = 0.3``
    generate   ``n_sample = 200`` flow samples on the locked ODE time grid
    refine     top ``n_elite = 10`` -> ``n_copy = 200`` perturbations ->
               MPPI (``mppi_lambda = 0.1``, ``mppi_sigma = 0.4``,
               ``beta_mppi = 20``) under the ``b1_safemppi`` refinement cost
    warm start ``warm_s = 0.8`` on the shifted previous window

Every knob is pinned by ``sfm_hp100_kazuki.LOCKED_CONFIG_ITEMS`` and audited by
``sfm_hp100_kazuki.locked_config()``, which this driver calls and records.

gamma
-----
gamma reaches the controller through exactly one path: the low-dimensional
observation ``sfm_hp100_features.low5(state, GOAL, gamma)`` that conditions the
CFM prior's context.  Every guidance coefficient, the refinement cost and the
MPPI hyper-parameters are gamma-independent (all ``*_gamma_span`` entries of
the locked config are 0.0 and every ``*_by_gamma`` table is empty), so gamma
only reshapes the generated sample cloud and never the objective that picks
inside it.  ``--gamma-invariance-check`` measures how much of that survives.

Average clearance
-----------------
``episode_average_clearance`` reproduces the expert-side v2 metric exactly: the
arithmetic mean of the nearest-pedestrian *surface* clearance
(``min_p ||robot_xy - ped_xy|| - R_PED``) over every visited state of the
episode, i.e. one sample per pre-step state plus the terminal state, so
``clearance_count == steps + 1``.  The locked rollout returns the visited robot
states and the pedestrian snapshot of every EXECUTED step but not the terminal
snapshot, so the pedestrian bank is replayed deterministically from
``sfm_scene.make_humans`` / ``advance_humans`` against the returned states.  The
replay is verified snapshot-by-snapshot against the recorded pedestrian
positions (``ped_replay_max_abs_deviation``, required to be <= 1e-4) which
makes the reconstructed terminal snapshot as trustworthy as the recorded ones.

Explicit episode lists (ADDITIVE, optional)
------------------------------------------
``--episode-list-json PATH`` replaces the contiguous ``--ep0/--M`` bank with an
explicit, per-gamma scenario list ``{"<gamma>": [episode_id, ...]}``; the same
contract as ``sfm_hp100_cluster_deploy``.  ``--name-suffix`` lets two shards of
one cell be produced by two processes and merged afterwards.

Layout::

    <out-dir>/<cells-dirname>/<series>_g<gamma><suffix>.json    one per gamma
    <out-dir>/<markers-dirname>/<series><suffix>.done           after all gammas
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

DEFAULT_SNAPSHOT = os.path.expanduser(
    "~/projects/safe_flow_expansion_SFM2-claude-cfc09ad/source_snapshot/"
    "overnight_run_07_12_sfm"
)

VERSION = "sfm_hp100_kazuki_cluster_deploy_v1"
CONTROLLER = "kazuki_cfm_mppi_locked"
STATUS_CELL = "SFM2_CLUSTER_CELL_COMPLETE"
PED_REPLAY_TOLERANCE = 1.0e-4


def _bootstrap(snapshot: str):
    snapshot = os.path.abspath(snapshot)
    if snapshot not in sys.path:
        sys.path.insert(0, snapshot)
    import _paths  # noqa: F401


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gamma_tag(gamma: float) -> str:
    """``0.1 -> 0p1``, ``0.15 -> 0p15``, ``1.0 -> 1p0`` (DCBF cell naming)."""
    text = f"{float(gamma):.4f}".rstrip("0")
    if text.endswith("."):
        text += "0"
    return text.replace(".", "p")


def load_episode_lists(path, gammas):
    """Read ``{"<gamma>": [ids...]}`` -> ``(per_gamma_lists, M)``.

    Identical contract to ``sfm_hp100_cluster_deploy.load_episode_lists``:
    non-empty, duplicate-free, equal-length and pairwise disjoint lists.
    """
    with open(path) as stream:
        raw = json.load(stream)
    if not isinstance(raw, dict):
        raise ValueError("--episode-list-json must contain a gamma -> ids object")
    numeric = {float(key): value for key, value in raw.items()}
    lists = {}
    for gamma in gammas:
        matches = [k for k in numeric if abs(k - float(gamma)) <= 1e-12]
        if len(matches) != 1:
            raise ValueError(f"episode list has no unique entry for gamma={gamma:g}")
        ids = [int(v) for v in numeric[matches[0]]]
        if not ids:
            raise ValueError(f"episode list for gamma={gamma:g} is empty")
        if len(set(ids)) != len(ids):
            raise ValueError(f"episode list for gamma={gamma:g} has duplicates")
        if any(v < 0 for v in ids):
            raise ValueError(f"episode list for gamma={gamma:g} has a negative id")
        lists[float(gamma)] = ids
    sizes = {len(v) for v in lists.values()}
    if len(sizes) != 1:
        raise ValueError("every gamma must list the same number of episodes")
    pooled = [v for ids in lists.values() for v in ids]
    if len(set(pooled)) != len(pooled):
        raise ValueError("episode lists must be disjoint across gammas")
    return lists, sizes.pop()


# --------------------------------------------------------------------------
# expert-v2 average clearance from a locked Kazuki rollout
# --------------------------------------------------------------------------
def clearance_trace(rollout, *, scene_profile, modules):
    """Per-visited-state surface clearance of one locked Kazuki rollout.

    ``kazuki_hp100_deploy`` returns ``states`` (steps + 1 visited states) and
    ``peds`` (the pedestrian snapshot of every EXECUTED step, i.e. steps
    entries).  The pedestrian bank is a pure deterministic function of the
    scenario id and of the robot states that were fed to ``advance_humans``, so
    replaying it against the returned states reproduces every recorded snapshot
    bit-for-bit and additionally yields the missing terminal snapshot.
    """
    SS = modules["SS"]
    KZ = modules["KZ"]
    environment = SS.scene_profile(scene_profile)
    humans = SS.make_humans(
        int(rollout["episode"]), seed=0, n_ped=int(environment["n_ped"]),
        speed_range=tuple(environment["ped_speed_range"]),
    )
    states = np.asarray(rollout["states"], np.float32)
    recorded = np.asarray(rollout["peds"], np.float32)
    steps = int(rollout["steps"])
    if states.shape[0] != steps + 1:
        raise RuntimeError(
            f"expected steps + 1 visited states, got {states.shape[0]} for {steps} steps"
        )
    if recorded.shape[0] != steps:
        raise RuntimeError(
            f"expected one pedestrian snapshot per executed step, got {recorded.shape[0]}"
        )
    clearances = []
    deviation = 0.0
    for index in range(steps + 1):
        pedestrian_xy = np.asarray(SS.collect_humans(humans)[0], np.float32)
        if index < steps:
            deviation = max(
                deviation,
                float(np.max(np.abs(pedestrian_xy - recorded[index]))),
            )
        clearance = KZ._clearance(states[index], pedestrian_xy)
        if not np.isfinite(clearance):
            raise RuntimeError("path-average clearance requires at least one pedestrian")
        clearances.append(float(clearance))
        if index < steps:
            SS.advance_humans(humans, states[index + 1])
    if deviation > PED_REPLAY_TOLERANCE:
        raise RuntimeError(
            f"pedestrian replay diverged by {deviation:g} > {PED_REPLAY_TOLERANCE:g}"
        )
    return clearances, deviation


def row_from_rollout(rollout, *, gamma, scene_profile, modules, seconds):
    DYN = modules["DYN"]
    clearances, deviation = clearance_trace(
        rollout, scene_profile=scene_profile, modules=modules
    )
    steps = int(rollout["steps"])
    if len(clearances) != steps + 1:
        raise RuntimeError("clearance trace must contain each visited state exactly once")
    success = bool(rollout["success"])
    collision = bool(rollout["collision"])
    total = float(np.sum(clearances))
    average = total / len(clearances)
    executed = (total - clearances[-1]) / steps if steps else None
    minimum = float(np.min(clearances))
    if abs(minimum - float(rollout["min_clear"])) > 1e-4:
        raise RuntimeError(
            "replayed minimum clearance disagrees with the locked rollout: "
            f"{minimum:g} != {float(rollout['min_clear']):g}"
        )
    return dict(
        episode=int(rollout["episode"]),
        gamma=float(gamma),
        J=None,
        status=("success" if success else "collision" if collision else "timeout"),
        success=success,
        collision=collision,
        timeout=bool(not success and not collision),
        steps=steps,
        time_to_goal=(steps * float(DYN.DT) if success else None),
        episode_average_clearance=float(average),
        episode_min_clearance=float(minimum),
        episode_average_clearance_executed_steps=(
            None if executed is None else float(executed)
        ),
        successful_average_clearance=(float(average) if success else None),
        clearance_sum=float(total),
        clearance_count=int(len(clearances)),
        min_clearance=float(rollout["min_clear"]),
        ped_replay_max_abs_deviation=float(deviation),
        seconds=round(float(seconds), 3),
    )


# --------------------------------------------------------------------------
# one gamma cell
# --------------------------------------------------------------------------
@torch.no_grad()
def run_cell(policy, *, gamma, scene_profile, episode_ids, device, modules,
             sample_seed, T, progress_every=10):
    KZ = modules["KZ"]
    rows = []
    for position, episode in enumerate(episode_ids):
        started = time.time()
        rollout = KZ.kazuki_hp100_deploy(
            policy, int(episode), float(gamma), scene_profile=scene_profile,
            T=int(T), device=device, sample_seed=int(sample_seed),
            collect_diagnostics=False,
        )
        row = row_from_rollout(
            rollout, gamma=gamma, scene_profile=scene_profile, modules=modules,
            seconds=time.time() - started,
        )
        rows.append(row)
        if progress_every and (position == 0 or (position + 1) % int(progress_every) == 0):
            print(json.dumps(dict(
                event="episode_done", gamma=float(gamma), position=position + 1,
                total=len(episode_ids), episode=int(episode),
                status=row["status"], seconds=row["seconds"],
            )), flush=True)
    return rows


def _mean(values):
    finite = [float(value) for value in values if value is not None]
    return None if not finite else float(np.mean(finite))


def summarize_cell(rows) -> dict:
    n = len(rows)
    successes = sum(bool(row["success"]) for row in rows)
    collisions = sum(bool(row["collision"]) for row in rows)
    timeouts = sum(bool(row["timeout"]) for row in rows)
    if successes + collisions + timeouts != n:
        raise RuntimeError("outcomes do not partition the cell")
    successful = [row for row in rows if row["success"]]
    return dict(
        attempts=int(n),
        successes=int(successes),
        collisions=int(collisions),
        timeouts=int(timeouts),
        SR=(successes / n if n else None),
        CR=(collisions / n if n else None),
        timeout_rate=(timeouts / n if n else None),
        all_attempts=dict(
            mean_average_clearance=_mean(
                [row["episode_average_clearance"] for row in rows]
            ),
            mean_min_clearance=_mean([row["episode_min_clearance"] for row in rows]),
        ),
        success_conditioned=dict(
            n=len(successful),
            mean_time_to_goal=_mean([row["time_to_goal"] for row in successful]),
            mean_average_clearance=_mean(
                [row["episode_average_clearance"] for row in successful]
            ),
            mean_min_clearance=_mean(
                [row["episode_min_clearance"] for row in successful]
            ),
        ),
        max_ped_replay_deviation=max(
            [float(row["ped_replay_max_abs_deviation"]) for row in rows] or [0.0]
        ),
    )


# --------------------------------------------------------------------------
# gamma-invariance probe
# --------------------------------------------------------------------------
@torch.no_grad()
def gamma_invariance_check(policy, *, episodes, gammas, scene_profile, device,
                           modules, sample_seed, T):
    """Same episodes and seeds, two gammas: how far do the states drift apart?"""
    KZ = modules["KZ"]
    lo, hi = float(gammas[0]), float(gammas[1])
    report = []
    for episode in episodes:
        rollouts = {}
        for gamma in (lo, hi):
            rollouts[gamma] = KZ.kazuki_hp100_deploy(
                policy, int(episode), float(gamma), scene_profile=scene_profile,
                T=int(T), device=device, sample_seed=int(sample_seed),
                collect_diagnostics=False,
            )
        a = np.asarray(rollouts[lo]["states"], np.float64)
        b = np.asarray(rollouts[hi]["states"], np.float64)
        common = min(a.shape[0], b.shape[0])
        difference = np.abs(a[:common] - b[:common])
        report.append(dict(
            episode=int(episode),
            steps=dict(lo=int(rollouts[lo]["steps"]), hi=int(rollouts[hi]["steps"])),
            status=dict(
                lo=("success" if rollouts[lo]["success"] else
                    "collision" if rollouts[lo]["collision"] else "timeout"),
                hi=("success" if rollouts[hi]["success"] else
                    "collision" if rollouts[hi]["collision"] else "timeout"),
            ),
            common_states=int(common),
            max_abs_state_deviation=float(difference.max()) if common else None,
            max_abs_position_deviation=(
                float(difference[:, :2].max()) if common else None
            ),
            first_step_deviating=(
                int(np.argmax(difference.max(axis=1) > 1e-6))
                if common and bool((difference.max(axis=1) > 1e-6).any()) else None
            ),
            final_position_lo=a[-1, :2].tolist(),
            final_position_hi=b[-1, :2].tolist(),
        ))
    values = [r["max_abs_state_deviation"] for r in report
              if r["max_abs_state_deviation"] is not None]
    return dict(
        gamma_lo=lo, gamma_hi=hi, episodes=[int(e) for e in episodes],
        per_episode=report,
        max_abs_state_deviation=(max(values) if values else None),
        identical_trajectories=bool(values and max(values) <= 1e-6),
        verdict=("gamma_invariant" if values and max(values) <= 1e-6
                 else "gamma_dependent"),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--series", default="kazuki")
    parser.add_argument("--gammas", default="0.1,0.15,0.2,0.5,1.0")
    parser.add_argument("--ep0", type=int, default=940000)
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--episode-list-json", default=None)
    parser.add_argument("--wave", default=None)
    parser.add_argument("--scene-profile", default="double_density_velocity_ood")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--sample-seed", type=int, default=None,
                        help="defaults to the locked sfm_hp100_kazuki.SAMPLE_SEED")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--cells-dirname", default="cells_kazuki")
    parser.add_argument("--markers-dirname", default="markers")
    parser.add_argument("--name-suffix", default="",
                        help="appended to the cell/marker basename (episode shards)")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--gamma-invariance-check", default=None,
                        help=("run the two-gamma same-seed probe instead of the "
                              "cells and write the report to this path"))
    parser.add_argument("--check-episodes", default="940000,940001,940002")
    parser.add_argument("--check-gammas", default="0.1,1.0")
    args = parser.parse_args(argv)

    _bootstrap(args.snapshot)
    from pathlib import Path

    import grid_policy_sfm_hp100 as GPS
    import sfm_hp100_dynamics as DYN
    import sfm_hp100_eval as EVAL
    import sfm_hp100_features as HPF
    import sfm_hp100_kazuki as KZ
    import sfm_scene as SS

    torch.backends.cudnn.benchmark = False
    modules = dict(KZ=KZ, DYN=DYN, HPF=HPF, SS=SS, EVAL=EVAL)

    source = os.path.abspath(__file__)
    source_sha = sha256_file(source)
    sample_seed = int(KZ.SAMPLE_SEED if args.sample_seed is None else args.sample_seed)

    checkpoint_path = os.path.abspath(args.checkpoint)
    actual = sha256_file(checkpoint_path)
    if actual != args.expected_checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint sha mismatch: {actual} != {args.expected_checkpoint_sha256}"
        )
    policy, checkpoint = GPS.load_sfm_hp100_policy(checkpoint_path, device=args.device)
    policy.eval()

    config = KZ.locked_config()          # fails closed if any knob drifted
    locked = dict(
        version=str(KZ.VERSION),
        safe_coef=float(KZ.SAFE_COEF),
        goal_coef=float(KZ.GOAL_COEF),
        sample_seed=sample_seed,
        sample_seed_default=int(KZ.SAMPLE_SEED),
        sample_seed_formula="torch.manual_seed(sample_seed + episode*1000 + step)",
        config=config.to_dict(),
        evaluator_sources={
            name: dict(path=os.path.abspath(module.__file__),
                       sha256=sha256_file(module.__file__))
            for name, module in (
                ("sfm_hp100_kazuki", KZ), ("sfm_kazuki", KZ.BASE),
                ("sfm_scene", SS), ("sfm_hp100_dynamics", DYN),
                ("sfm_hp100_features", HPF), ("sfm_hp100_eval", EVAL),
            )
        },
        gamma_entry_point=(
            "gamma enters ONLY through sfm_hp100_features.low5(state, GOAL, gamma) "
            "-> policy.ctx_from(...) context; every guidance/refinement coefficient "
            "is gamma-independent (all *_gamma_span == 0.0, all *_by_gamma empty)"
        ),
    )

    gammas = [float(value) for value in str(args.gammas).split(",") if value.strip()]
    if not gammas:
        raise ValueError("no gammas requested")
    if len(set(gammas)) != len(gammas):
        raise ValueError("duplicate gamma requested")

    out_root = Path(os.path.abspath(args.out_dir))

    if args.gamma_invariance_check:
        episodes = [int(v) for v in str(args.check_episodes).split(",") if v.strip()]
        probe_gammas = [float(v) for v in str(args.check_gammas).split(",") if v.strip()]
        if len(probe_gammas) != 2:
            raise ValueError("--check-gammas needs exactly two values")
        started = time.time()
        report = gamma_invariance_check(
            policy, episodes=episodes, gammas=probe_gammas,
            scene_profile=args.scene_profile, device=args.device, modules=modules,
            sample_seed=sample_seed, T=EVAL.T,
        )
        report.update(
            status="SFM2_KAZUKI_GAMMA_INVARIANCE_PROBE",
            version=VERSION, source=source, source_sha256=source_sha,
            checkpoint=checkpoint_path, checkpoint_sha256=actual,
            scene=SS.scene_profile(args.scene_profile), locked=locked,
            seconds=round(time.time() - started, 1),
            seconds_per_episode_rollout=round(
                (time.time() - started) / max(1, 2 * len(episodes)), 2
            ),
        )
        target = Path(os.path.abspath(args.gamma_invariance_check))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(json.dumps(dict(
            status=report["status"], verdict=report["verdict"],
            max_abs_state_deviation=report["max_abs_state_deviation"],
            seconds_per_episode_rollout=report["seconds_per_episode_rollout"],
            out=str(target),
        )), flush=True)
        return 0

    episode_lists = None
    episode_list_sha = None
    M = int(args.M)
    if args.episode_list_json:
        episode_lists, M = load_episode_lists(args.episode_list_json, gammas)
        episode_list_sha = sha256_file(args.episode_list_json)
        print(json.dumps({
            "event": "episode_list_loaded",
            "path": os.path.abspath(args.episode_list_json),
            "sha256": episode_list_sha, "M": int(M),
            "gammas": [float(g) for g in gammas],
        }), flush=True)

    cells_dir = out_root / str(args.cells_dirname)
    markers_dir = out_root / str(args.markers_dirname)
    cells_dir.mkdir(parents=True, exist_ok=True)
    markers_dir.mkdir(parents=True, exist_ok=True)

    common = dict(
        status=STATUS_CELL,
        version=VERSION,
        controller=CONTROLLER,
        result_kind="controller_mode",
        authoritative_raw_eval=False,
        series=str(args.series),
        source=source,
        source_sha256=source_sha,
        checkpoint=checkpoint_path,
        checkpoint_sha256=actual,
        checkpoint_scientific_status=(
            checkpoint.get("scientific_status") if isinstance(checkpoint, dict) else None
        ),
        J=None,
        ep0=(None if episode_lists is not None else int(args.ep0)),
        M=int(M),
        wave=(None if args.wave is None else str(args.wave)),
        episode_source=("declared_episode_list" if episode_lists is not None
                        else "contiguous_bank"),
        episode_list_json=(None if episode_lists is None
                           else os.path.abspath(args.episode_list_json)),
        episode_list_sha256=episode_list_sha,
        gammas_requested=[float(value) for value in gammas],
        scene=SS.scene_profile(args.scene_profile),
        locked_kazuki=locked,
        selection_rule=(
            "locked Kazuki generate-guide-refine: 200 guided CFM samples "
            "(v + 0.5*grad_goal + 0.3*rho_H*grad_CBF) -> top-10 elites -> 200 "
            "perturbations -> MPPI refinement under the b1_safemppi cost; warm "
            "start s=0.8; no shield, template, privileged lookahead or fallback"
        ),
        seeds=dict(kazuki_sample_seed=sample_seed),
        latent_bank=(
            "per-step torch.manual_seed(sample_seed + episode*1000 + step) inside "
            "the locked comparator; NOT the canonical sfm_hp100_eval CRN bank"
        ),
        temperature=None,
        NFE=None,
        T=int(EVAL.T),
        H=int(policy.H_pred),
        clearance_metric=dict(
            episode_average_clearance=(
                "arithmetic mean over every visited state (pre-step states plus "
                "the terminal state; clearance_count == steps + 1) of "
                "min_p ||robot_xy - ped_xy|| - R_PED, identical to the expert v2 "
                "metric in sfm_hp100_extended_clearance_eval / "
                "sfm_hp100_mppi_pair_eval"
            ),
            episode_average_clearance_executed_steps=(
                "same mean restricted to the executed pre-step states "
                "(terminal visit dropped)"
            ),
            episode_min_clearance="minimum over the same visited states",
            reconstruction=(
                "the locked rollout returns the visited states and every executed "
                "step's pedestrian snapshot; the pedestrian bank is replayed "
                "deterministically to recover the terminal snapshot and every "
                "recorded snapshot is re-verified "
                "(ped_replay_max_abs_deviation <= 1e-4)"
            ),
        ),
        dynamics=DYN.contract(),
        observation=HPF.contract(),
        device=str(args.device),
        physical_gpu=int(args.physical_gpu),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        policy_config=policy.config(),
        name_suffix=str(args.name_suffix),
    )

    written = []
    for gamma in gammas:
        name = f"{args.series}_g{gamma_tag(gamma)}{args.name_suffix}.json"
        target = cells_dir / name
        if target.exists() and not args.overwrite:
            print(json.dumps({"event": "cell_exists_skipped", "out": str(target)}),
                  flush=True)
            written.append(str(target))
            continue
        if episode_lists is None:
            identifiers = [int(args.ep0) + offset for offset in range(int(M))]
        else:
            identifiers = [int(v) for v in episode_lists[float(gamma)]]
        started = time.time()
        rows = run_cell(
            policy, gamma=float(gamma), scene_profile=args.scene_profile,
            episode_ids=identifiers, device=args.device, modules=modules,
            sample_seed=sample_seed, T=EVAL.T,
            progress_every=int(args.progress_every),
        )
        summary = summarize_cell(rows)
        payload = dict(common)
        payload.update(
            gamma=float(gamma),
            gamma_tag=gamma_tag(gamma),
            episodes=list(identifiers),
            seconds=round(time.time() - started, 1),
            seconds_per_episode=round((time.time() - started) / max(1, len(rows)), 2),
            created_unix=time.time(),
            summary=summary,
            rows=rows,
        )
        temporary = str(target) + ".tmp"
        with open(temporary, "w") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
        os.replace(temporary, str(target))
        written.append(str(target))
        print(json.dumps({
            "event": "cell_complete", "series": args.series, "gamma": float(gamma),
            "successes": summary["successes"], "collisions": summary["collisions"],
            "timeouts": summary["timeouts"],
            "success_conditioned": summary["success_conditioned"],
            "seconds": payload["seconds"],
            "seconds_per_episode": payload["seconds_per_episode"],
            "out": str(target),
        }), flush=True)

    marker = markers_dir / f"{args.series}{args.name_suffix}.done"
    marker.write_text(json.dumps(dict(
        status="SFM2_CLUSTER_RUN_COMPLETE",
        series=str(args.series),
        gammas=[float(value) for value in gammas],
        wave=(None if args.wave is None else str(args.wave)),
        episode_list_json=(None if episode_lists is None
                           else os.path.abspath(args.episode_list_json)),
        episode_list_sha256=episode_list_sha,
        M=int(M), cells=written, checkpoint_sha256=actual,
        source_sha256=source_sha, name_suffix=str(args.name_suffix),
        completed_unix=time.time(),
    ), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "SFM2_CLUSTER_RUN_COMPLETE",
                      "series": args.series, "cells": len(written),
                      "marker": str(marker)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
