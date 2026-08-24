#!/usr/bin/env python
"""Success-conditioned gamma-cluster deployment runner (ADDITIVE).

Produces one per-(series, gamma) cell JSON of closed-loop best-of-J rollouts
for the success-conditioned ``clearance x time-to-goal`` cluster plot.  Nothing
in the frozen pipeline is edited or monkeypatched: ``sfm_hp100_eval``,
``sfm_scene``, ``sfm_hp100_dynamics``, ``sfm_hp100_features``, ``sfm_b1_eval``
and ``grid_policy_sfm_hp100`` are imported read-only.

Controller (identical mechanics to ``bestofN_calibration.run_bestofN`` and
``sfm_hp100_bon_distill_collect.run_block``)
-------------------------------------------------------------------------
Per replan step of every active episode: draw ``J`` temperature-1 NFE-8 flow
proposals at the current context (one batched ``ctx_from`` + one batched
``integrate_latents`` call), score every proposal with the tuned v2 MPC cost

    cost_j = -H10_goal_progress_j
             + lam * sum_{h=1..H} rho**(H-h) * exp((r_eff - d_hj)/sigma_len)

(``lam=4.0, rho=1.1, r_eff=0.45, sigma_len=0.10``, exponent clamped at 60,
``d_hj`` the constant-velocity-propagated min-over-pedestrians surface
clearance), execute the first clipped action of the argmin proposal with the
deterministic tie-break ``(cost, -progress, index)``, replan.

CONTROLLER-MODE numbers
-----------------------
For ``J > 1`` the controller differs from the frozen raw evaluator, so every
number here is a controller-mode number.  It is additionally *not* a canonical
CRN run at any ``J``: because this study needs the off-grid gamma 0.15, the
whole latent bank is generated from declared ``numpy`` generators rather than
from ``sfm_hp100_eval.noise_bank`` (whose seven slots are tied to
``sfm_scene.GAMMAS``).  Proposal ``j = 0`` is therefore NOT the canonical CRN
latent, and every payload declares ``latent_bank = "declared_rng,
controller_mode"``.

Average clearance
-----------------
``episode_average_clearance`` reproduces the expert-side v2 metric exactly:
the arithmetic mean of the nearest-pedestrian *surface* clearance
(``min_p ||robot_xy - ped_xy|| - R_PED``) over every visited state of the
episode, i.e. one sample per pre-step state plus the terminal state, so
``clearance_count == steps + 1`` -- the same accounting as
``sfm_hp100_extended_clearance_eval`` and ``sfm_hp100_mppi_pair_eval`` (the
MPPI-DCBF anchor cells this plot joins against).  The executed-step-only mean
(terminal visit dropped) is reported alongside as
``episode_average_clearance_executed_steps``.

Layout::

    <out-dir>/cells/<series>_g<gamma>_J<J>.json     one per gamma
    <out-dir>/markers/<series>_J<J>.done            after ALL gammas landed
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


def _bootstrap(snapshot: str):
    snapshot = os.path.abspath(snapshot)
    if snapshot not in sys.path:
        sys.path.insert(0, snapshot)
    import _paths  # noqa: F401


VERSION = "sfm_hp100_cluster_deploy_v1"
CONTROLLER = "bestofJ_mpc_select"
STATUS_CELL = "SFM2_CLUSTER_CELL_COMPLETE"

# Tuned MPC-cost rule (bestofN_calibration / sfm_hp100_predictive_execution_v2).
LAM = 4.0
RHO = 1.1
R_EFF = 0.45
SIGMA_LEN = 0.10
MAX_EXPONENT = 60.0


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


# --------------------------------------------------------------------------
# MPC cost, batched over (active episodes) x (J candidates)
# --------------------------------------------------------------------------
def batched_mpc_cost(states, ped_xy, ped_vel, windows, *, DYN, SS):
    """Vectorised v2 MPC cost for every candidate of every active episode.

    states  : (A, 4)       float32 current robot states
    ped_xy  : (A, P, 2)    float32 current pedestrian positions
    ped_vel : (A, P, 2)    float32 current pedestrian velocities
    windows : (A, J, H, 2) float32 candidate H-step action windows

    Returns (cost, progress, clearances) with shapes (A, J), (A, J), (A, J, H).
    """
    A, J, H, _ = windows.shape
    current = np.repeat(states[:, None, :], J, axis=1).astype(np.float32, copy=True)
    positions = np.empty((A, J, H + 1, 2), np.float32)
    positions[:, :, 0] = current[..., :2]
    for h in range(H):
        current = DYN.step_numpy(current, windows[:, :, h, :]).astype(
            np.float32, copy=False
        )
        positions[:, :, h + 1] = current[..., :2]

    horizon_time = np.arange(H + 1, dtype=np.float32) * float(SS.DT)
    pedestrians = (
        ped_xy[:, None, :, :]
        + horizon_time[None, :, None, None] * ped_vel[:, None, :, :]
    )  # (A, H+1, P, 2)
    if pedestrians.shape[2]:
        difference = positions[:, :, :, None, :] - pedestrians[:, None, :, :, :]
        distance = np.linalg.norm(difference, axis=-1) - float(SS.R_PED)
        clearances = distance[:, :, 1:, :].min(axis=-1)  # (A, J, H)
    else:
        clearances = np.full((A, J, H), np.inf, np.float32)

    goal = np.asarray(SS.GOAL, float)
    initial = np.linalg.norm(states[:, :2].astype(float) - goal[None], axis=1)
    final = np.linalg.norm(
        positions[:, :, H, :].astype(float) - goal[None, None, :], axis=2
    )
    progress = initial[:, None] - final  # (A, J)

    weights = RHO ** (H - np.arange(1, H + 1, dtype=float))  # rho**(H-h)
    exponent = np.minimum(
        (R_EFF - clearances.astype(float)) / SIGMA_LEN, MAX_EXPONENT
    )
    proximity = (weights[None, None, :] * np.exp(exponent)).sum(axis=2)
    cost = -progress + LAM * proximity
    return cost, progress, clearances


def select_argmin(cost_row, progress_row):
    """argmin cost with the v2 deterministic tie-break (cost, -progress, index)."""
    return min(
        range(len(cost_row)),
        key=lambda j: (float(cost_row[j]), -float(progress_row[j]), int(j)),
    )


# --------------------------------------------------------------------------
# declared latent bank (NOT the canonical CRN bank -- see the module docstring)
# --------------------------------------------------------------------------
def declared_latent_banks(*, n_gammas, M, T, d, J, base_seed, extra_seed):
    base = np.random.default_rng(int(base_seed)).standard_normal(
        (int(n_gammas), int(M), int(T), int(d)), dtype=np.float32
    )
    if int(J) <= 1:
        return base, None
    extra = np.random.default_rng(int(extra_seed)).standard_normal(
        (int(n_gammas), int(M), int(T), int(J) - 1, int(d)), dtype=np.float32
    )
    return base, extra


# --------------------------------------------------------------------------
# closed-loop best-of-J rollout for one gamma cell
# --------------------------------------------------------------------------
@torch.no_grad()
def run_cell(
    policy,
    *,
    gamma,
    gamma_index,
    scene_profile,
    ep0,
    M,
    J,
    base,
    extra,
    device,
    modules,
    progress_every=20,
):
    EVAL = modules["EVAL"]
    BASE = modules["BASE"]
    DYN = modules["DYN"]
    HPF = modules["HPF"]
    SS = modules["SS"]
    T, H = EVAL.T, EVAL.H
    J = int(J)

    if J > 1 and (extra is None or extra.shape[3] < J - 1):
        raise ValueError("extra latent bank too small for the requested J")

    environment = SS.scene_profile(scene_profile)
    episodes = [
        EVAL.Episode(
            gamma_index=int(gamma_index),
            rollout_index=rollout_index,
            episode=int(ep0) + rollout_index,
            gamma=float(gamma),
            humans=SS.make_humans(
                int(ep0) + rollout_index,
                seed=0,
                n_ped=int(environment["n_ped"]),
                speed_range=tuple(environment["ped_speed_range"]),
            ),
        )
        for rollout_index in range(int(M))
    ]
    # Per-episode nearest-pedestrian surface-clearance trace accounting.  One
    # sample per visited state (pre-step states + the terminal state), exactly
    # the expert-side v2 convention (clearance_count == steps + 1).
    trace = [
        dict(sum=0.0, count=0, minimum=float("inf"), last=None)
        for _ in episodes
    ]

    def visit(index, episode, pedestrian_xy):
        """Accumulate this visited state's clearance, then frozen terminal check."""
        clearance = EVAL._clearance(episode.state, pedestrian_xy)
        if not np.isfinite(clearance):
            raise RuntimeError(
                "path-average clearance requires at least one pedestrian"
            )
        item = trace[index]
        item["sum"] += float(clearance)
        item["count"] += 1
        item["minimum"] = min(item["minimum"], float(clearance))
        item["last"] = float(clearance)
        return EVAL._terminal_check(episode, pedestrian_xy)

    selection = dict(
        replans=0,
        changed=0,
        chosen_index_histogram=[0] * J,
        cost_chosen=0.0,
        cost_first=0.0,
        minclear_chosen=0.0,
        minclear_first=0.0,
        first_negative_minclear_steps=0,
        chosen_negative_minclear_steps=0,
        rescued_steps=0,
    )

    for step in range(T):
        active = []
        hp_histories, low5, control_histories, latents = [], [], [], []
        for index, episode in enumerate(episodes):
            if episode.status is not None:
                continue
            pedestrian_xy, pedestrian_velocity = SS.collect_humans(episode.humans)
            pedestrian_xy = np.asarray(pedestrian_xy, np.float32)
            pedestrian_velocity = np.asarray(pedestrian_velocity, np.float32)
            if visit(index, episode, pedestrian_xy):
                continue
            frame = HPF.hp100_frame(
                episode.state[:2],
                EVAL._obstacles(pedestrian_xy),
                sensing=SS.R_SENSE,
                n_base=HPF.POLYTOPE_N_BASE,
                obstacle_velocities=pedestrian_velocity,
                robot_velocity=episode.state[2:4],
                predict_gain=HPF.PREDICT_GAIN,
                predict_tau=HPF.PREDICT_TAU,
            )
            active.append((episode, pedestrian_xy.copy(), pedestrian_velocity.copy()))
            hp_histories.append(episode.hp_history.append(frame))
            low5.append(
                torch.as_tensor(HPF.low5(episode.state, SS.GOAL, episode.gamma))
            )
            control_histories.append(
                torch.as_tensor(HPF.hist_pad(episode.controls[-HPF.K_HIST:]))
            )
            head = base[episode.gamma_index, episode.rollout_index, step][None]
            if J > 1:
                tail = extra[
                    episode.gamma_index, episode.rollout_index, step, : J - 1
                ]
                latents.append(np.concatenate((head, tail), axis=0))
            else:
                latents.append(head)
        if not active:
            break

        hp_tensor = torch.stack(hp_histories).to(device)
        low_tensor = torch.stack(low5).to(device)
        history_tensor = torch.stack(control_histories).to(device)
        context = policy.ctx_from(hp_tensor, low_tensor, history_tensor)
        expanded = context.repeat_interleave(J, dim=0)
        latent_array = np.asarray(latents, np.float32).reshape(len(active) * J, -1)
        latent_tensor = torch.as_tensor(latent_array, device=device)
        windows = BASE.integrate_latents(
            policy, EVAL.TEMPERATURE * latent_tensor, expanded, nfe=EVAL.NFE
        ).reshape(len(active) * J, H, 2)
        windows = (
            windows.detach().cpu().numpy().astype(np.float32)
            .reshape(len(active), J, H, 2)
        )

        states = np.stack([item[0].state for item in active]).astype(np.float32)
        peds_xy = np.stack([item[1] for item in active]).astype(np.float32)
        peds_vel = np.stack([item[2] for item in active]).astype(np.float32)
        cost, progress, clearances = batched_mpc_cost(
            states, peds_xy, peds_vel, windows, DYN=DYN, SS=SS
        )

        for index, (episode, pedestrian_xy, pedestrian_velocity) in enumerate(active):
            choice = select_argmin(cost[index], progress[index])
            selection["replans"] += 1
            selection["chosen_index_histogram"][choice] += 1
            selection["changed"] += int(choice != 0)
            selection["cost_chosen"] += float(cost[index, choice])
            selection["cost_first"] += float(cost[index, 0])
            chosen_min = float(clearances[index, choice].min())
            first_min = float(clearances[index, 0].min())
            selection["minclear_chosen"] += chosen_min
            selection["minclear_first"] += first_min
            selection["first_negative_minclear_steps"] += int(first_min < 0.0)
            selection["chosen_negative_minclear_steps"] += int(chosen_min < 0.0)
            selection["rescued_steps"] += int(first_min < 0.0 <= chosen_min)

            window = windows[index, choice]
            action = DYN.clip_action_numpy(window[0]).astype(np.float32, copy=False)
            episode.ped_xy.append(pedestrian_xy)
            episode.ped_vel.append(pedestrian_velocity)
            episode.controls.append(action.copy())
            episode.state = DYN.step_numpy(episode.state, action).astype(
                np.float32, copy=False
            )
            episode.states.append(episode.state.copy())
            SS.advance_humans(episode.humans, episode.state)

        if progress_every and (step == 0 or (step + 1) % int(progress_every) == 0):
            print(json.dumps(dict(
                event="rollout_step", gamma=float(gamma), J=J,
                step=step + 1, active=len(active), total=int(M),
            )), flush=True)

    rows = []
    for index, episode in enumerate(episodes):
        if episode.status is None:
            pedestrian_xy, _ = SS.collect_humans(episode.humans)
            if not visit(index, episode, pedestrian_xy):
                episode.status = "timeout"
        item = trace[index]
        steps = len(episode.controls)
        if item["count"] != steps + 1:
            raise RuntimeError(
                "clearance trace must contain each visited state exactly once "
                f"(count={item['count']}, steps={steps})"
            )
        average = item["sum"] / item["count"]
        executed_average = (
            (item["sum"] - item["last"]) / steps if steps else None
        )
        success = episode.status == "success"
        rows.append(dict(
            episode=int(episode.episode),
            gamma=float(gamma),
            J=J,
            status=str(episode.status),
            success=bool(success),
            collision=episode.status == "collision",
            timeout=episode.status == "timeout",
            steps=int(steps),
            time_to_goal=(steps * float(DYN.DT) if success else None),
            episode_average_clearance=float(average),
            episode_min_clearance=float(item["minimum"]),
            episode_average_clearance_executed_steps=(
                None if executed_average is None else float(executed_average)
            ),
            successful_average_clearance=(float(average) if success else None),
            clearance_sum=float(item["sum"]),
            clearance_count=int(item["count"]),
            min_clearance=float(episode.minimum_clearance),
        ))

    replans = max(1, selection["replans"])
    stats = dict(
        replans=int(selection["replans"]),
        selection_changed_fraction=selection["changed"] / replans,
        chosen_index_histogram=[int(v) for v in selection["chosen_index_histogram"]],
        mean_cost_chosen=selection["cost_chosen"] / replans,
        mean_cost_first_candidate=selection["cost_first"] / replans,
        mean_min_horizon_clearance_chosen=selection["minclear_chosen"] / replans,
        mean_min_horizon_clearance_first_candidate=(
            selection["minclear_first"] / replans
        ),
        cv_predicted_contact_steps_first_candidate=int(
            selection["first_negative_minclear_steps"]
        ),
        cv_predicted_contact_steps_chosen=int(
            selection["chosen_negative_minclear_steps"]
        ),
        cv_predicted_contact_steps_rescued=int(selection["rescued_steps"]),
    )
    return rows, stats


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
            mean_min_clearance=_mean(
                [row["episode_min_clearance"] for row in rows]
            ),
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
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--series", required=True, help="series label, e.g. pre/sfe")
    parser.add_argument("--gammas", default="0.1,0.15,0.2,0.5,1.0")
    parser.add_argument("--ep0", type=int, default=940000)
    parser.add_argument("--M", type=int, default=100)
    parser.add_argument("--J", type=int, required=True)
    parser.add_argument("--base-seed", type=int, required=True)
    parser.add_argument("--extra-seed", type=int, required=True)
    parser.add_argument("--scene-profile", default="double_density_velocity_ood")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--out-dir", required=True,
                        help="cluster_plot root; cells/ and markers/ live under it")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true",
                        help="recompute cells whose JSON already exists")
    args = parser.parse_args(argv)

    _bootstrap(args.snapshot)
    from pathlib import Path

    import grid_policy_sfm_hp100 as GPS
    import sfm_b1_eval as BASE
    import sfm_hp100_dynamics as DYN
    import sfm_hp100_eval as EVAL
    import sfm_hp100_features as HPF
    import sfm_scene as SS

    torch.backends.cudnn.benchmark = False
    modules = dict(EVAL=EVAL, BASE=BASE, DYN=DYN, HPF=HPF, SS=SS)

    source = os.path.abspath(__file__)
    source_sha = sha256_file(source)

    checkpoint_path = os.path.abspath(args.checkpoint)
    actual = sha256_file(checkpoint_path)
    if actual != args.expected_checkpoint_sha256:
        raise RuntimeError(
            f"checkpoint sha mismatch: {actual} != {args.expected_checkpoint_sha256}"
        )
    policy, checkpoint = GPS.load_sfm_hp100_policy(checkpoint_path, device=args.device)

    gammas = [float(value) for value in str(args.gammas).split(",") if value.strip()]
    if not gammas:
        raise ValueError("no gammas requested")
    if len(set(gammas)) != len(gammas):
        raise ValueError("duplicate gamma requested")

    base, extra = declared_latent_banks(
        n_gammas=len(gammas), M=int(args.M), T=EVAL.T, d=int(policy.d),
        J=int(args.J), base_seed=int(args.base_seed),
        extra_seed=int(args.extra_seed),
    )

    out_root = Path(os.path.abspath(args.out_dir))
    cells_dir = out_root / "cells"
    markers_dir = out_root / "markers"
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
            checkpoint.get("scientific_status")
            if isinstance(checkpoint, dict) else None
        ),
        J=int(args.J),
        ep0=int(args.ep0),
        M=int(args.M),
        gammas_requested=[float(value) for value in gammas],
        scene=SS.scene_profile(args.scene_profile),
        mpc_params=dict(lam=LAM, rho=RHO, r_eff=R_EFF, sigma_len=SIGMA_LEN,
                        max_exponent=MAX_EXPONENT),
        selection_rule=(
            "pure argmin of the tuned v2 MPC cost over J raw temperature-1 "
            "NFE-8 samples; tie-break (cost, -progress, index); no exact "
            "verifier, no acquisition/ESS, no retry/fallback"
        ),
        seeds=dict(base_seed=int(args.base_seed), extra_seed=int(args.extra_seed)),
        temperature=float(EVAL.TEMPERATURE),
        NFE=int(EVAL.NFE),
        T=int(EVAL.T),
        H=int(EVAL.H),
        latent_bank="declared_rng, controller_mode",
        latent_bank_manifest=dict(
            convention=(
                "base bank = default_rng(base_seed).standard_normal("
                "(len(gammas), M, T, d)); extra bank = default_rng(extra_seed)"
                ".standard_normal((len(gammas), M, T, J-1, d)); candidate j=0 "
                "takes the base bank, j>=1 the extra bank.  This is NOT the "
                "canonical sfm_hp100_eval.noise_bank CRN: the custom gamma "
                "grid (0.15 is off the canonical seven-gamma grid) forces a "
                "declared bank, so j=0 is NOT the canonical raw proposal."
            ),
            base_seed=int(args.base_seed),
            base_shape=list(base.shape),
            base_sha256=EVAL.array_sha256(base),
            extra_seed=int(args.extra_seed),
            extra_shape=(None if extra is None else list(extra.shape)),
            extra_sha256=(None if extra is None else EVAL.array_sha256(extra)),
        ),
        clearance_metric=dict(
            episode_average_clearance=(
                "arithmetic mean over every visited state (pre-step states plus "
                "the terminal state; clearance_count == steps + 1) of "
                "min_p ||robot_xy - ped_xy|| - R_PED, identical to the expert "
                "v2 metric in sfm_hp100_extended_clearance_eval / "
                "sfm_hp100_mppi_pair_eval"
            ),
            episode_average_clearance_executed_steps=(
                "same mean restricted to the executed pre-step states "
                "(terminal visit dropped)"
            ),
            episode_min_clearance="minimum over the same visited states",
        ),
        dynamics=DYN.contract(),
        device=str(args.device),
        physical_gpu=int(args.physical_gpu),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        policy_config=policy.config(),
    )

    written = []
    for gamma_index, gamma in enumerate(gammas):
        name = f"{args.series}_g{gamma_tag(gamma)}_J{int(args.J)}.json"
        target = cells_dir / name
        if target.exists() and not args.overwrite:
            print(json.dumps({"event": "cell_exists_skipped",
                              "out": str(target)}), flush=True)
            written.append(str(target))
            continue
        started = time.time()
        rows, stats = run_cell(
            policy, gamma=float(gamma), gamma_index=int(gamma_index),
            scene_profile=args.scene_profile, ep0=int(args.ep0), M=int(args.M),
            J=int(args.J), base=base, extra=extra, device=args.device,
            modules=modules, progress_every=int(args.progress_every),
        )
        summary = summarize_cell(rows)
        payload = dict(common)
        payload.update(
            gamma=float(gamma),
            gamma_tag=gamma_tag(gamma),
            gamma_index=int(gamma_index),
            seconds=round(time.time() - started, 1),
            created_unix=time.time(),
            selection=stats,
            summary=summary,
            rows=rows,
        )
        temporary = str(target) + ".tmp"
        with open(temporary, "w") as stream:
            json.dump(payload, stream, indent=2, allow_nan=False)
        os.replace(temporary, str(target))
        written.append(str(target))
        print(json.dumps({
            "event": "cell_complete", "series": args.series,
            "gamma": float(gamma), "J": int(args.J),
            "successes": summary["successes"],
            "collisions": summary["collisions"],
            "timeouts": summary["timeouts"],
            "success_conditioned": summary["success_conditioned"],
            "selection_changed_fraction": round(
                stats["selection_changed_fraction"], 4
            ),
            "seconds": payload["seconds"], "out": str(target),
        }), flush=True)

    marker = markers_dir / f"{args.series}_J{int(args.J)}.done"
    marker.write_text(json.dumps(dict(
        status="SFM2_CLUSTER_RUN_COMPLETE",
        series=str(args.series), J=int(args.J),
        gammas=[float(value) for value in gammas],
        cells=written,
        checkpoint_sha256=actual,
        source_sha256=source_sha,
        completed_unix=time.time(),
    ), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "SFM2_CLUSTER_RUN_COMPLETE",
                      "series": args.series, "J": int(args.J),
                      "cells": len(written), "marker": str(marker)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
