#!/usr/bin/env python
"""N-sample MPC-select inference switch: mechanism + per-model N calibration.

DEFAULT-OFF DEPLOYMENT SWITCH.  This file is ADDITIVE.  It edits nothing in the
frozen pipeline: it imports ``sfm_hp100_eval`` (the canonical raw evaluator),
``sfm_hp100_dynamics``, ``sfm_hp100_features``, ``sfm_scene``, ``sfm_metrics2``
and ``sfm_b1_eval`` read-only and re-implements only the *selection* step.

Mechanism (per replan step of a closed-loop rollout)
----------------------------------------------------
1. Draw ``N`` independent temperature-1, NFE-8 flow proposals from the policy
   at the current context (one batched call; the context is shared).
2. Score every proposal with the tuned MPC cost of
   ``sfm_hp100_predictive_execution_v2``::

       cost_j = -H10_goal_progress_j
                + lam * sum_{h=1..10} rho**(10-h) * exp((r_eff - d_hj)/sigma)

   with ``lam=4.0, rho=1.1, r_eff=0.45, sigma=0.10`` and ``d_hj`` the
   constant-velocity-propagated min-over-pedestrians clearance of proposal
   ``j`` at horizon step ``h`` (identical construction to
   ``sfm_hp100_predictive_execution_v2.horizon_clearances``).
3. Execute the first clipped action of the ARGMIN-cost proposal; replan.

What this is NOT
----------------
There is no exact GREEN verifier in the loop, no RBF acquisition, no ESS/B
sub-selection, and no retry/fallback schedule.  Eligibility is not filtered:
the rule is a pure argmin of the scalar cost over the ``N`` raw samples.  The
CV cost is therefore a *heuristic* safety surrogate, not a certificate.
Because the controller differs from the frozen evaluator for ``N > 1``, every
number produced here is a CONTROLLER-MODE number and is labelled as such.  It
must never be quoted as a raw-eval number.

N = 1 fallback
--------------
The sampling convention is nested: proposal ``j = 0`` at every (gamma,
rollout, step) is exactly the canonical CRN latent of
``sfm_hp100_eval.noise_bank(seed=<noise-seed>)``; proposals ``j >= 1`` come
from a separate declared bank and are a prefix in ``j`` across the whole N
grid.  Selection is ``argmin`` over the N candidates with no special case, so
at N = 1 the executed proposal is the canonical raw proposal.  ``--prove-n1``
runs the frozen ``sfm_hp100_eval.run_batched_raw`` on the same bank and asserts
bitwise array equality of states / controls / pedestrians / outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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
    import _paths  # noqa: F401  (installs the study's sibling package paths)


VERSION = "sfm_hp100_bestofN_mpcselect_v1"
CONTROLLER = "bestofN_mpc_select"
STATUS = "SFM_HP100_BESTOFN_CALIBRATION_COMPLETE"

# Tuned MPC-cost rule (mirrors sfm_hp100_predictive_execution_v2.MPCRuleParams
# with the calibrated lam / r_eff used by the current execution-rule study).
LAM = 4.0
RHO = 1.1
R_EFF = 0.45
SIGMA_LEN = 0.10
MAX_EXPONENT = 60.0  # identical clamp to the v2 module

# Declared latent-bank seeds.  BASE_SEED must match the M20 screening bank so
# that proposal j=0 is the canonical raw proposal.
DEFAULT_BASE_SEED = 20260814
DEFAULT_EXTRA_SEED = 20260819


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------
# MPC cost, batched over (active episodes) x (N candidates)
# --------------------------------------------------------------------------
def batched_mpc_cost(states, ped_xy, ped_vel, windows, *, EVAL, DYN, SS):
    """Vectorised v2 MPC cost for every candidate of every active episode.

    states  : (A, 4)      float32 current robot states
    ped_xy  : (A, P, 2)   float32 current pedestrian positions
    ped_vel : (A, P, 2)   float32 current pedestrian velocities
    windows : (A, N, H, 2) float32 candidate H-step action windows

    Returns (cost, progress, clearances) with shapes (A, N), (A, N), (A, N, H).
    """
    A, N, H, _ = windows.shape
    current = np.repeat(states[:, None, :], N, axis=1).astype(np.float32, copy=True)
    positions = np.empty((A, N, H + 1, 2), np.float32)
    positions[:, :, 0] = current[..., :2]
    for h in range(H):
        current = DYN.step_numpy(current, windows[:, :, h, :]).astype(
            np.float32, copy=False
        )
        positions[:, :, h + 1] = current[..., :2]

    # constant-velocity pedestrian propagation, identical to
    # sfm_metrics2.predict_pedestrians broadcast over the episode axis
    horizon_time = np.arange(H + 1, dtype=np.float32) * float(SS.DT)
    pedestrians = (
        ped_xy[:, None, :, :]
        + horizon_time[None, :, None, None] * ped_vel[:, None, :, :]
    )  # (A, H+1, P, 2)

    if pedestrians.shape[2]:
        difference = positions[:, :, :, None, :] - pedestrians[:, None, :, :, :]
        distance = np.linalg.norm(difference, axis=-1) - float(SS.R_PED)
        clearances = distance[:, :, 1:, :].min(axis=-1)  # (A, N, H)
    else:
        clearances = np.full((A, N, H), np.inf, np.float32)

    goal = np.asarray(SS.GOAL, float)
    initial = np.linalg.norm(states[:, :2].astype(float) - goal[None], axis=1)
    final = np.linalg.norm(
        positions[:, :, H, :].astype(float) - goal[None, None, :], axis=2
    )
    progress = initial[:, None] - final  # (A, N)

    weights = RHO ** (H - np.arange(1, H + 1, dtype=float))  # rho**(H-h)
    exponent = np.minimum(
        (R_EFF - clearances.astype(float)) / SIGMA_LEN, MAX_EXPONENT
    )
    proximity = (weights[None, None, :] * np.exp(exponent)).sum(axis=2)
    cost = -progress + LAM * proximity
    return cost, progress, clearances


def reference_cost(state, ped_xy, ped_vel, window, *, EVAL, VERIFY, SS):
    """Scalar reference built ONLY from frozen helpers, for self-checking.

    Reproduces ``sfm_hp100_predictive_execution_v2.horizon_clearances`` and
    ``.mpc_cost`` literally (``EVAL.clipped_rollout_positions`` is the same
    capped integration as ``PORT.clipped_plan_states``).
    """
    controls = np.asarray(window, np.float32).reshape(-1, 2)
    segment = EVAL.clipped_rollout_positions(state, controls)
    pedestrians = VERIFY.predict_pedestrians(ped_xy, ped_vel, H=len(controls))
    if pedestrians.shape[1]:
        distance = np.linalg.norm(segment[:, None, :] - pedestrians, axis=2) - float(
            SS.R_PED
        )
        clearances = [float(v) for v in distance[1:].min(axis=1)]
    else:
        clearances = [float("inf")] * len(controls)
    goal = np.asarray(SS.GOAL, float)
    initial = float(np.linalg.norm(np.asarray(state[:2], float) - goal))
    progress = float(initial - np.linalg.norm(segment[-1] - goal))
    horizon = len(clearances)
    proximity = 0.0
    for index, clearance in enumerate(clearances):
        h = index + 1
        exponent = (R_EFF - float(clearance)) / SIGMA_LEN
        proximity += RHO ** (horizon - h) * math.exp(min(exponent, MAX_EXPONENT))
    return -progress + LAM * proximity, progress, clearances


def select_argmin(cost_row, progress_row):
    """argmin cost with the v2 deterministic tie-break (cost, -progress, index)."""
    return min(
        range(len(cost_row)),
        key=lambda j: (float(cost_row[j]), -float(progress_row[j]), int(j)),
    )


# --------------------------------------------------------------------------
# latent bank
# --------------------------------------------------------------------------
def latent_banks(*, M, d, n_gammas, T, base_seed, extra_seed, n_extra, EVAL):
    """Nested proposal bank.  index 0 == the canonical raw CRN latent."""
    base = EVAL.noise_bank(M=M, d=d, seed=base_seed)  # (G, M, T, d)
    if base.shape != (n_gammas, M, T, d):
        raise RuntimeError("canonical noise bank shape drifted")
    if n_extra <= 0:
        return base, None
    generator = np.random.default_rng(int(extra_seed))
    extra = generator.standard_normal(
        (n_gammas, M, T, int(n_extra), d), dtype=np.float32
    )
    return base, extra


# --------------------------------------------------------------------------
# the N-sample MPC-select closed-loop controller
# --------------------------------------------------------------------------
@torch.no_grad()
def run_bestofN(
    policy,
    *,
    scene_profile,
    ep0,
    M,
    base,
    extra,
    N,
    device,
    modules,
    self_check_steps=0,
):
    EVAL = modules["EVAL"]
    BASE = modules["BASE"]
    DYN = modules["DYN"]
    HPF = modules["HPF"]
    SS = modules["SS"]
    VERIFY = modules["VERIFY"]
    T, H = EVAL.T, EVAL.H

    if N > 1 and (extra is None or extra.shape[3] < N - 1):
        raise ValueError("extra latent bank too small for the requested N")

    environment = SS.scene_profile(scene_profile)
    episodes = [
        EVAL.Episode(
            gamma_index=gamma_index,
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
        for gamma_index, gamma in enumerate(SS.GAMMAS)
        for rollout_index in range(int(M))
    ]

    selection = dict(
        replans=0,
        changed=0,
        chosen_index_histogram=[0] * int(N),
        cost_chosen=0.0,
        cost_raw=0.0,
        minclear_chosen=0.0,
        minclear_raw=0.0,
        raw_negative_minclear_steps=0,
        chosen_negative_minclear_steps=0,
        rescued_steps=0,
    )
    checks = []

    for step in range(T):
        active = []
        hp_histories, low5, control_histories, latents = [], [], [], []
        for episode in episodes:
            if episode.status is not None:
                continue
            pedestrian_xy, pedestrian_velocity = SS.collect_humans(episode.humans)
            pedestrian_xy = np.asarray(pedestrian_xy, np.float32)
            pedestrian_velocity = np.asarray(pedestrian_velocity, np.float32)
            if EVAL._terminal_check(episode, pedestrian_xy):
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
                torch.as_tensor(HPF.hist_pad(episode.controls[-HPF.K_HIST :]))
            )
            head = base[episode.gamma_index, episode.rollout_index, step][None]
            if N > 1:
                tail = extra[
                    episode.gamma_index, episode.rollout_index, step, : N - 1
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
        expanded = context.repeat_interleave(int(N), dim=0)
        latent_array = np.asarray(latents, np.float32).reshape(len(active) * N, -1)
        latent_tensor = torch.as_tensor(latent_array, device=device)
        windows = BASE.integrate_latents(
            policy, EVAL.TEMPERATURE * latent_tensor, expanded, nfe=EVAL.NFE
        ).reshape(len(active) * N, H, 2)
        windows = (
            windows.detach()
            .cpu()
            .numpy()
            .astype(np.float32)
            .reshape(len(active), N, H, 2)
        )

        states = np.stack([item[0].state for item in active]).astype(np.float32)
        peds_xy = np.stack([item[1] for item in active]).astype(np.float32)
        peds_vel = np.stack([item[2] for item in active]).astype(np.float32)
        cost, progress, clearances = batched_mpc_cost(
            states, peds_xy, peds_vel, windows, EVAL=EVAL, DYN=DYN, SS=SS
        )

        if self_check_steps and step < self_check_steps:
            for a in range(min(len(active), 3)):
                for j in range(N):
                    ref_cost, ref_progress, ref_clear = reference_cost(
                        states[a], peds_xy[a], peds_vel[a], windows[a, j],
                        EVAL=EVAL, VERIFY=VERIFY, SS=SS,
                    )
                    checks.append(
                        dict(
                            step=int(step),
                            cost_abs_error=abs(float(cost[a, j]) - float(ref_cost)),
                            progress_abs_error=abs(
                                float(progress[a, j]) - float(ref_progress)
                            ),
                            clearance_max_abs_error=float(
                                np.max(
                                    np.abs(
                                        np.asarray(clearances[a, j], float)
                                        - np.asarray(ref_clear, float)
                                    )
                                )
                            ),
                        )
                    )

        for index, (episode, pedestrian_xy, pedestrian_velocity) in enumerate(active):
            choice = select_argmin(cost[index], progress[index])
            selection["replans"] += 1
            selection["chosen_index_histogram"][choice] += 1
            selection["changed"] += int(choice != 0)
            selection["cost_chosen"] += float(cost[index, choice])
            selection["cost_raw"] += float(cost[index, 0])
            chosen_min = float(clearances[index, choice].min())
            raw_min = float(clearances[index, 0].min())
            selection["minclear_chosen"] += chosen_min
            selection["minclear_raw"] += raw_min
            selection["raw_negative_minclear_steps"] += int(raw_min < 0.0)
            selection["chosen_negative_minclear_steps"] += int(chosen_min < 0.0)
            selection["rescued_steps"] += int(raw_min < 0.0 <= chosen_min)

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

    rows = []
    for episode in episodes:
        if episode.status is None:
            pedestrian_xy, _ = SS.collect_humans(episode.humans)
            if not EVAL._terminal_check(episode, pedestrian_xy):
                episode.status = "timeout"
        success = episode.status == "success"
        rows.append(
            dict(
                episode=int(episode.episode),
                gamma=float(episode.gamma),
                status=str(episode.status),
                success=bool(success),
                collision=episode.status == "collision",
                timeout=episode.status == "timeout",
                steps=len(episode.controls),
                time_to_goal=(len(episode.controls) * DYN.DT if success else None),
                min_clearance=float(episode.minimum_clearance),
                successful_clearance=(
                    float(episode.minimum_clearance) if success else None
                ),
                states=np.asarray(episode.states, np.float32),
                controls=np.asarray(episode.controls, np.float32).reshape(-1, 2),
                ped_xy=np.asarray(episode.ped_xy, np.float32).reshape(
                    len(episode.controls), int(environment["n_ped"]), 2
                ),
                ped_vel=np.asarray(episode.ped_vel, np.float32).reshape(
                    len(episode.controls), int(environment["n_ped"]), 2
                ),
            )
        )

    replans = max(1, selection["replans"])
    stats = dict(
        replans=int(selection["replans"]),
        selection_changed_fraction=selection["changed"] / replans,
        chosen_index_histogram=[
            int(v) for v in selection["chosen_index_histogram"]
        ],
        mean_cost_chosen=selection["cost_chosen"] / replans,
        mean_cost_raw=selection["cost_raw"] / replans,
        mean_min_horizon_clearance_chosen=selection["minclear_chosen"] / replans,
        mean_min_horizon_clearance_raw=selection["minclear_raw"] / replans,
        cv_predicted_contact_steps_raw=int(selection["raw_negative_minclear_steps"]),
        cv_predicted_contact_steps_chosen=int(
            selection["chosen_negative_minclear_steps"]
        ),
        cv_predicted_contact_steps_rescued=int(selection["rescued_steps"]),
    )
    if checks:
        stats["cost_self_check"] = dict(
            samples=len(checks),
            max_cost_abs_error=max(c["cost_abs_error"] for c in checks),
            max_progress_abs_error=max(c["progress_abs_error"] for c in checks),
            max_clearance_abs_error=max(
                c["clearance_max_abs_error"] for c in checks
            ),
        )
    return rows, stats


# --------------------------------------------------------------------------
# N = 1 fallback proof
# --------------------------------------------------------------------------
def prove_n1(policy, *, scene_profile, ep0, M, base, device, modules):
    EVAL = modules["EVAL"]
    mine, _ = run_bestofN(
        policy, scene_profile=scene_profile, ep0=ep0, M=M, base=base, extra=None,
        N=1, device=device, modules=modules, self_check_steps=0,
    )
    frozen = EVAL.run_batched_raw(
        policy, scene_profile=scene_profile, ep0=ep0, M=M, noise=base, device=device,
    )
    if len(mine) != len(frozen):
        raise RuntimeError("N=1 proof: episode count mismatch")
    mismatches = []
    arrays = ("states", "controls", "ped_xy", "ped_vel")
    scalars = ("episode", "gamma", "status", "success", "collision", "timeout",
               "steps", "min_clearance")
    for a, b in zip(mine, frozen):
        for key in arrays:
            if not np.array_equal(a[key], b[key]):
                mismatches.append(
                    dict(episode=int(a["episode"]), gamma=float(a["gamma"]),
                         field=key,
                         max_abs_diff=float(
                             np.max(np.abs(np.asarray(a[key], float)
                                           - np.asarray(b[key], float)))
                         ) if np.shape(a[key]) == np.shape(b[key]) else None,
                         shape_a=list(np.shape(a[key])),
                         shape_b=list(np.shape(b[key]))))
        for key in scalars:
            if a[key] != b[key]:
                mismatches.append(dict(episode=int(a["episode"]), field=key,
                                       mine=a[key], frozen=b[key]))
    return dict(
        claim=(
            "N=1 MPC-select controller == frozen sfm_hp100_eval.run_batched_raw "
            "on the identical CRN latent bank"
        ),
        episodes_compared=len(mine),
        arrays_compared=list(arrays),
        scalars_compared=list(scalars),
        bitwise_identical=not mismatches,
        mismatches=mismatches[:20],
        n_mismatches=len(mismatches),
        frozen_evaluator_source=os.path.abspath(EVAL.__file__),
        frozen_evaluator_sha256=sha256_file(EVAL.__file__),
    )


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def compact(rows):
    drop = ("states", "controls", "ped_xy", "ped_vel")
    return [{k: v for k, v in row.items() if k not in drop} for row in rows]


def fit_exponential(ns, crs):
    """Least squares of log CR on N.  CR(N) ~ exp(a) * p**N, p = exp(b)."""
    pairs = [(float(n), float(c)) for n, c in zip(ns, crs) if c > 0.0]
    if len(pairs) < 2:
        return None
    x = np.array([p[0] for p in pairs])
    y = np.log(np.array([p[1] for p in pairs]))
    matrix = np.vstack([np.ones_like(x), x]).T
    (a, b), residuals, *_ = np.linalg.lstsq(matrix, y, rcond=None)
    predicted = a + b * x
    ss_res = float(((y - predicted) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return dict(
        model="log CR(N) = a + b*N",
        a=float(a), b=float(b),
        per_sample_risk_factor=float(np.exp(b)),
        r2=(None if ss_tot == 0.0 else 1.0 - ss_res / ss_tot),
        points_used=[[float(p[0]), float(p[1])] for p in pairs],
        zero_cr_points=[int(n) for n, c in zip(ns, crs) if c <= 0.0],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--N-grid", default="1,2,4,8,16")
    parser.add_argument("--scene-profile", default="double_density_velocity_ood")
    parser.add_argument("--ep0", type=int, default=900000)
    parser.add_argument("--M", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    parser.add_argument("--extra-seed", type=int, default=DEFAULT_EXTRA_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prove-n1", action="store_true")
    parser.add_argument("--self-check-steps", type=int, default=2)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    _bootstrap(args.snapshot)
    import grid_policy_sfm_hp100 as GPS
    import sfm_b1_eval as BASE
    import sfm_hp100_dynamics as DYN
    import sfm_hp100_eval as EVAL
    import sfm_hp100_features as HPF
    import sfm_metrics2 as VERIFY
    import sfm_scene as SS

    modules = dict(EVAL=EVAL, BASE=BASE, DYN=DYN, HPF=HPF, SS=SS, VERIFY=VERIFY)
    torch.backends.cudnn.benchmark = False

    grid = [int(v) for v in str(args.N_grid).split(",") if v.strip()]
    if not grid or min(grid) < 1:
        raise ValueError("N grid must be positive integers")

    policy, checkpoint = GPS.load_sfm_hp100_policy(args.checkpoint, device=args.device)
    base, extra = latent_banks(
        M=args.M, d=policy.d, n_gammas=len(SS.GAMMAS), T=EVAL.T,
        base_seed=args.base_seed, extra_seed=args.extra_seed,
        n_extra=max(grid) - 1, EVAL=EVAL,
    )

    provenance = dict(
        status=STATUS,
        version=VERSION,
        controller=CONTROLLER,
        authoritative_raw_eval=False,
        result_kind="controller_mode",
        label=str(args.label),
        source=os.path.abspath(__file__),
        source_sha256=sha256_file(__file__),
        checkpoint=os.path.abspath(args.checkpoint),
        checkpoint_sha256=sha256_file(args.checkpoint),
        checkpoint_scientific_status=checkpoint.get("scientific_status"),
        mpc_params=dict(lam=LAM, rho=RHO, r_eff=R_EFF, sigma_len=SIGMA_LEN,
                        max_exponent=MAX_EXPONENT),
        selection_rule=(
            "pure argmin of the v2 MPC cost over N raw temperature-1 NFE-8 "
            "samples; tie-break (cost, -progress, index); NO exact verifier, "
            "NO acquisition/ESS, NO retry/fallback"
        ),
        scene=SS.scene_profile(args.scene_profile),
        ep0=int(args.ep0), M_per_gamma=int(args.M),
        temperature=EVAL.TEMPERATURE, NFE=EVAL.NFE, T=EVAL.T, H=EVAL.H,
        latent_bank=dict(
            base_seed=int(args.base_seed),
            base_sha256=EVAL.array_sha256(base),
            base_shape=list(base.shape),
            extra_seed=int(args.extra_seed),
            extra_shape=(None if extra is None else list(extra.shape)),
            extra_sha256=(None if extra is None else EVAL.array_sha256(extra)),
            convention=(
                "proposal j=0 == canonical sfm_hp100_eval CRN latent; j>=1 from "
                "the declared extra bank; prefix-nested in j across the N grid"
            ),
        ),
        dynamics=DYN.contract(),
        N_grid=grid,
        device=str(args.device),
        validity_note=(
            "exact GREEN Validity is NOT computed here; the controller uses no "
            "exact verifier and this run reports outcome rates only"
        ),
    )

    results = {}
    if args.prove_n1:
        started = time.time()
        provenance["n1_fallback_proof"] = prove_n1(
            policy, scene_profile=args.scene_profile, ep0=args.ep0, M=args.M,
            base=base, device=args.device, modules=modules,
        )
        provenance["n1_fallback_proof"]["seconds"] = time.time() - started
        print(json.dumps({"n1_proof": {
            k: v for k, v in provenance["n1_fallback_proof"].items()
            if k in ("bitwise_identical", "n_mismatches", "episodes_compared")
        }}), flush=True)

    for N in grid:
        started = time.time()
        rows, stats = run_bestofN(
            policy, scene_profile=args.scene_profile, ep0=args.ep0, M=args.M,
            base=base, extra=extra, N=N, device=args.device, modules=modules,
            self_check_steps=int(args.self_check_steps),
        )
        small = compact(rows)
        summary = EVAL.summarize(small)
        results[str(N)] = dict(
            N=int(N), summary=summary, selection=stats,
            seconds=time.time() - started, rows=small,
        )
        pooled = summary["pooled"]
        print(json.dumps({
            "label": args.label, "N": N, "CR": pooled["CR"], "SR": pooled["SR"],
            "timeout": pooled["timeout"],
            "TtG": pooled["successful_time_to_goal"],
            "selection_changed_fraction": stats["selection_changed_fraction"],
            "seconds": round(results[str(N)]["seconds"], 1),
        }), flush=True)

    crs = [results[str(N)]["summary"]["pooled"]["CR"] for N in grid]
    n_star = next((N for N, cr in zip(grid, crs) if cr < 0.10), None)
    provenance["n_star"] = (None if n_star is None else int(n_star))
    provenance["n_star_criterion"] = "smallest N in the grid with pooled CR < 0.10"
    provenance["cr_decay_fit"] = fit_exponential(grid, crs)
    provenance["results"] = results

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    temporary = out + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(provenance, stream, indent=2, allow_nan=False)
    os.replace(temporary, out)
    print(json.dumps({"status": STATUS, "label": args.label, "out": out,
                      "n_star": provenance["n_star"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
