#!/usr/bin/env python
"""Best-of-N MPC-select self-distillation collector.

ADDITIVE module: closed-loop best-of-N teacher rollouts whose executed steps
become D+ demonstrations in the exact extended-archive + raw-obs shard format
consumed by ``sfm_hp100_raw_train.py``.  Nothing in the frozen pipeline is
edited or monkeypatched.

Controller (identical to ``bestofN_calibration.run_bestofN``)
------------------------------------------------------------
Per replan step: draw ``N`` temperature-1 NFE-8 flow proposals at the current
context (proposal j=0 is the canonical ``sfm_hp100_eval.noise_bank`` CRN
latent for this block's declared base seed), score every proposal with the
tuned v2 MPC cost (lam=4, rho=1.1, r_eff=0.45, sigma=0.10), execute the
argmin's first clipped action, replan.  CONTROLLER-MODE rollouts: no exact
verifier in the control loop, no acquisition/ESS, no retry schedule.

Demonstration harvest
---------------------
For every executed step of every episode that terminates in SUCCESS:

* the exact raw encoder inputs that produced the context token (ten-frame
  Hp100 raster stack, low5, GRU control history) go to a raw-obs shard
  directory in the ``sfm_hp100_raw_obs_capture.ShardWriter`` format, keyed
  ``(gamma, scenario_id, step)``;
* the CHOSEN window is certified post-hoc by the exact GREEN verifier
  (``sfm_metrics2.verify_query`` in worker processes — the same full-H10
  conjunction the acquisition adapter declares as ``valid``), and ONLY
  verifier-valid steps become archive rows (role="positive"); the invalid
  fraction is reported, never trained on;
* rows carry the full extended-archive field set.  Fields whose acquisition
  semantics do not exist here are declared honestly: ``native_cost`` is NaN
  (not computed), ``selected_sigma``/``marginal_sigma``/``conditional_ess``/
  ``marginal_ESS_over_K`` are NaN (no acquisition), ``K_index``/``B_local``
  hold the BoN choice index, ``mode_tags.changed`` means "a non-canonical
  proposal (j>0) was chosen" (BoN analog of the shadow-selector tag), and
  ``mode_tags.interaction`` uses the same CV min-clearance < 0.45 rule as
  ``sfm_hp100_mode_tags``.

Episodes ending in collision or timeout contribute NOTHING.  Negatives are
not fabricated; trainings that want the alpha-hinge term add an existing
negative-row archive alongside these blocks.

Output layout (one directory per scenario block, crash-tolerant)::

    <out-dir>/block_0000/raw_obs/RAW_OBS_MANIFEST.json + shards
    <out-dir>/block_0000/bon_archive.pt      rows: verified positives only
    <out-dir>/block_0000/bon_pairs.pt        all-N windows + costs sidecar
    <out-dir>/block_0000/BLOCK_COMPLETE.json
    <out-dir>/COLLECT_COMPLETE.json          written at the end / deadline
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


VERSION = "sfm_hp100_bon_distill_collect_v1"
STATUS_ARCHIVE = "SFM2_BON_DISTILL_ARCHIVE"
STATUS_BLOCK = "SFM2_BON_DISTILL_BLOCK_COMPLETE"
STATUS_RUN = "SFM2_BON_DISTILL_COLLECT_COMPLETE"

LAM = 4.0
RHO = 1.1
R_EFF = 0.45
SIGMA_LEN = 0.10
MAX_EXPONENT = 60.0
INTERACTION_CLEARANCE = 0.45  # sfm_hp100_mode_tags interaction rule


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batched_mpc_cost(states, ped_xy, ped_vel, windows, *, DYN, SS):
    """Vectorised v2 MPC cost; also returns the integrated plan positions."""
    A, N, H, _ = windows.shape
    current = np.repeat(states[:, None, :], N, axis=1).astype(np.float32, copy=True)
    positions = np.empty((A, N, H + 1, 2), np.float32)
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
    )
    if pedestrians.shape[2]:
        difference = positions[:, :, :, None, :] - pedestrians[:, None, :, :, :]
        distance = np.linalg.norm(difference, axis=-1) - float(SS.R_PED)
        clearances = distance[:, :, 1:, :].min(axis=-1)
    else:
        clearances = np.full((A, N, H), np.inf, np.float32)

    goal = np.asarray(SS.GOAL, float)
    initial = np.linalg.norm(states[:, :2].astype(float) - goal[None], axis=1)
    final = np.linalg.norm(
        positions[:, :, H, :].astype(float) - goal[None, None, :], axis=2
    )
    progress = initial[:, None] - final

    weights = RHO ** (H - np.arange(1, H + 1, dtype=float))
    exponent = np.minimum(
        (R_EFF - clearances.astype(float)) / SIGMA_LEN, MAX_EXPONENT
    )
    proximity = (weights[None, None, :] * np.exp(exponent)).sum(axis=2)
    cost = -progress + LAM * proximity
    return cost, progress, clearances, positions


def select_argmin(cost_row, progress_row):
    return min(
        range(len(cost_row)),
        key=lambda j: (float(cost_row[j]), -float(progress_row[j]), int(j)),
    )


def nominal_step_margin(robot, action, ped_xy, ped_vel, gamma, *, HPF, DYN,
                        EVAL, SS):
    """Adapter's one-step nominal Hp margin (verification.step_margin)."""
    _, geometry = HPF.hp100_frame(
        robot[:2], EVAL._obstacles(ped_xy), sensing=SS.R_SENSE,
        n_base=HPF.POLYTOPE_N_BASE, obstacle_velocities=ped_vel,
        robot_velocity=robot[2:4], predict_gain=HPF.PREDICT_GAIN,
        predict_tau=HPF.PREDICT_TAU, return_geometry=True,
    )
    after = DYN.step_numpy(robot, action)
    A = np.asarray(geometry["A"], np.float64)
    b = np.asarray(geometry["b"], np.float64)
    margins = np.asarray(geometry["margins"], np.float64)
    old = float(np.min((b - A @ robot[:2]) / margins))
    new = float(np.min((b - A @ after[:2]) / margins))
    return float(new - (1.0 - float(gamma)) * old)


def verification_dict(result, *, step_margin, progress):
    """Archive-row verification dict from a ``verify_query`` result.

    ``valid`` is exactly the adapter's declared conjunction (``y`` = taskspace
    AND collision-free AND certificate, at full H=10).  ``native_cost`` is NOT
    computed here (NaN, declared); ``H10_progress`` is the plan progress from
    the same capped integration the MPC cost used.
    """
    diagnostics = result.get("diagnostics") or {}
    return dict(
        valid=bool(result["y"]),
        hp_eligible=bool(step_margin >= -1.0e-9),
        margin=float(diagnostics.get("slack", float("nan"))),
        native_cost=float("nan"),
        H10_progress=float(progress),
        progress_eligible=True,
        error=bool(not result.get("resolved", False)),
        step_margin=float(step_margin),
    )


def run_block(
    policy, *, block, ep0, M, N, base_seed, extra_seed, scene_profile,
    device, modules, verify_pool, teacher_sha, block_dir, hp_flush_every,
    reencode_every, reencode_atol,
):
    import sfm_hp100_raw_obs_capture as CAP

    EVAL = modules["EVAL"]
    BASE = modules["BASE"]
    DYN = modules["DYN"]
    HPF = modules["HPF"]
    SS = modules["SS"]
    VERIFY = modules["VERIFY"]
    T, H = EVAL.T, EVAL.H

    base = EVAL.noise_bank(M=M, d=policy.d, seed=int(base_seed))
    generator = np.random.default_rng(int(extra_seed))
    extra = generator.standard_normal(
        (len(SS.GAMMAS), M, T, int(N) - 1, policy.d), dtype=np.float32
    )

    environment = SS.scene_profile(scene_profile)
    n_ped = int(environment["n_ped"])
    episodes = [
        EVAL.Episode(
            gamma_index=gamma_index,
            rollout_index=rollout_index,
            episode=int(ep0) + rollout_index,
            gamma=float(gamma),
            humans=SS.make_humans(
                int(ep0) + rollout_index,
                seed=0,
                n_ped=n_ped,
                speed_range=tuple(environment["ped_speed_range"]),
            ),
        )
        for gamma_index, gamma in enumerate(SS.GAMMAS)
        for rollout_index in range(int(M))
    ]

    buffers = {id(episode): [] for episode in episodes}
    stats = dict(
        replans=0, changed=0, chosen_valid=0, chosen_invalid=0,
        rescued_steps=0, reencode_checked=0, reencode_max_dev=0.0,
    )

    for step in range(T):
        active, hp_stacks, low5s, histories, latents = [], [], [], [], []
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
            hp_stacks.append(episode.hp_history.append(frame))
            low5s.append(
                torch.as_tensor(HPF.low5(episode.state, SS.GOAL, episode.gamma))
            )
            histories.append(
                torch.as_tensor(HPF.hist_pad(episode.controls[-HPF.K_HIST:]))
            )
            head = base[episode.gamma_index, episode.rollout_index, step][None]
            tail = extra[episode.gamma_index, episode.rollout_index, step, : N - 1]
            latents.append(np.concatenate((head, tail), axis=0))
        if not active:
            break

        hp_tensor = torch.stack(hp_stacks).to(device)
        low_tensor = torch.stack(low5s).to(device)
        history_tensor = torch.stack(histories).to(device)
        with torch.no_grad():
            context = policy.ctx_from(hp_tensor, low_tensor, history_tensor)
            expanded = context.repeat_interleave(int(N), dim=0)
            latent_array = np.asarray(latents, np.float32).reshape(
                len(active) * N, -1
            )
            latent_tensor = torch.as_tensor(latent_array, device=device)
            windows = BASE.integrate_latents(
                policy, EVAL.TEMPERATURE * latent_tensor, expanded, nfe=EVAL.NFE
            ).reshape(len(active) * N, H, 2)
        windows = (
            windows.detach().cpu().numpy().astype(np.float32)
            .reshape(len(active), N, H, 2)
        )
        tokens = context.detach().cpu().to(torch.float32)

        states = np.stack([item[0].state for item in active]).astype(np.float32)
        peds_xy = np.stack([item[1] for item in active]).astype(np.float32)
        peds_vel = np.stack([item[2] for item in active]).astype(np.float32)
        cost, progress, clearances, positions = batched_mpc_cost(
            states, peds_xy, peds_vel, windows, DYN=DYN, SS=SS
        )

        # -- select, then certify every chosen window in the worker pool
        choices = [
            select_argmin(cost[index], progress[index])
            for index in range(len(active))
        ]
        payloads = [
            (index, choices[index], states[index],
             windows[index, choices[index]], peds_xy[index], peds_vel[index],
             float(active[index][0].gamma))
            for index in range(len(active))
        ]
        verify_results = {
            context_id: result
            for context_id, _, result in verify_pool.map(
                VERIFY.verify_in_worker, payloads, chunksize=8
            )
        }

        for index, (episode, pedestrian_xy, pedestrian_velocity) in enumerate(active):
            choice = choices[index]
            window = windows[index, choice]
            stats["replans"] += 1
            stats["changed"] += int(choice != 0)
            chosen_min = float(clearances[index, choice].min())
            raw_min = float(clearances[index, 0].min())
            stats["rescued_steps"] += int(raw_min < 0.0 <= chosen_min)

            grid_cpu = hp_stacks[index].detach().cpu().to(torch.float32).clone()
            low5_cpu = low5s[index].detach().cpu().to(torch.float32).clone()
            history_cpu = histories[index].detach().cpu().to(torch.float32).clone()
            token_cpu = tokens[index].clone()
            if stats["replans"] % int(reencode_every) == 0:
                redone = CAP.reencode(policy, grid_cpu, low5_cpu, history_cpu)
                deviation = float((redone - token_cpu).abs().max())
                stats["reencode_checked"] += 1
                stats["reencode_max_dev"] = max(
                    stats["reencode_max_dev"], deviation
                )
                if deviation > float(reencode_atol):
                    raise RuntimeError(
                        f"raw capture re-encode deviation {deviation} > "
                        f"{reencode_atol} (block {block}, step {step})"
                    )

            packed = torch.cat([
                token_cpu,
                torch.as_tensor(states[index], dtype=torch.float32),
                torch.as_tensor(pedestrian_xy.reshape(-1), dtype=torch.float32),
                torch.as_tensor(pedestrian_velocity.reshape(-1), dtype=torch.float32),
            ])

            result = verify_results[index]
            step_margin = nominal_step_margin(
                states[index], window[0], pedestrian_xy, pedestrian_velocity,
                episode.gamma, HPF=HPF, DYN=DYN, EVAL=EVAL, SS=SS,
            )
            verification = verification_dict(
                result, step_margin=step_margin,
                progress=progress[index, choice],
            ) if result.get("resolved") else dict(
                valid=False, hp_eligible=False, margin=float("nan"),
                native_cost=float("nan"), H10_progress=0.0,
                progress_eligible=True, error=True, step_margin=step_margin,
            )
            stats["chosen_valid" if verification["valid"] else "chosen_invalid"] += 1

            plan_pos = positions[index, choice]
            goal = np.asarray(SS.GOAL, float)
            one_step = float(
                np.linalg.norm(plan_pos[0].astype(float) - goal)
                - np.linalg.norm(plan_pos[1].astype(float) - goal)
            )
            center_dist = np.linalg.norm(
                pedestrian_xy - states[index][:2][None], axis=1
            )
            horizon_time = (
                np.arange(H + 1, dtype=np.float32)[:, None, None] * float(SS.DT)
            )
            ped_traj = pedestrian_xy[None] + horizon_time * pedestrian_velocity[None]
            entering = np.linalg.norm(
                plan_pos[:, None, :] - ped_traj, axis=2
            ).min(axis=0)
            prediction_audit = dict(
                predicted_min_clearance=chosen_min,
                predicted_collision_free=bool(chosen_min > 0.0),
                sensed_pedestrians=int((center_dist <= float(SS.R_SENSE)).sum()),
                sensed_or_predicted_entering_pedestrians=int(
                    (entering <= float(SS.R_SENSE)).sum()
                ),
                H10_goal_progress=float(progress[index, choice]),
                one_step_goal_progress=one_step,
                mpc_horizon_clearances=[
                    float(v) for v in clearances[index, choice]
                ],
                mpc_cost=float(cost[index, choice]),
            )

            row = dict(
                role="positive",
                gamma=float(episode.gamma),
                replica=int(episode.rollout_index),
                lineage=f"bon16:g{episode.gamma:g}:ep{episode.episode}",
                scenario_id=int(episode.episode),
                step=int(step),
                attempt=0,
                executed=True,
                negative_reason=None,
                context=packed,
                candidate=torch.as_tensor(window.copy(), dtype=torch.float32),
                flow_base=torch.as_tensor(
                    latents[index][choice].reshape(H, 2).copy(),
                    dtype=torch.float32,
                ),
                verification=verification,
                prediction_audit=prediction_audit,
                round=1,
                block=int(block),
                block_seed=int(base_seed),
                K_index=int(choice),
                B_local=int(choice),
                base_std=1.0,
                beta=float("nan"),
                selected_sigma=float("nan"),
                marginal_sigma=float("nan"),
                conditional_ess=float("nan"),
                marginal_ESS_over_K=float("nan"),
                positive_first16=-1,
                positive_B32=-1,
                attempts_used=1,
                scene_profile=str(scene_profile),
                scenario_start=int(ep0),
                sampling_seed=int(extra_seed),
                checkpoint_sha256=str(teacher_sha),
                reference_checkpoint_sha256=str(teacher_sha),
                source_sha256=SOURCE_SHA,
                mode_tags=dict(
                    interaction=bool(chosen_min < INTERACTION_CLEARANCE),
                    changed=bool(choice != 0),
                ),
                bon_controller=dict(
                    N=int(N), choice=int(choice),
                    raw_min_clearance=raw_min,
                    cost_chosen=float(cost[index, choice]),
                    cost_raw=float(cost[index, 0]),
                ),
            )

            buffers[id(episode)].append(dict(
                raw=dict(
                    grid=grid_cpu, low5=low5_cpu, history=history_cpu,
                    token=token_cpu,
                    meta=dict(
                        core_episode=int(episode.episode),
                        robot=[float(v) for v in states[index]],
                        controller="bon16_mpc_select",
                        choice=int(choice),
                    ),
                ),
                row=(row if verification["valid"] else None),
                pair=dict(
                    windows=torch.as_tensor(
                        windows[index].copy(), dtype=torch.float16
                    ),
                    costs=torch.as_tensor(
                        cost[index].copy(), dtype=torch.float32
                    ),
                    choice=int(choice),
                    chosen_valid=bool(verification["valid"]),
                ),
                key=(f"{episode.gamma:g}", int(episode.episode), int(step)),
            ))

            action = DYN.clip_action_numpy(window[0]).astype(np.float32, copy=False)
            episode.ped_xy.append(pedestrian_xy)
            episode.ped_vel.append(pedestrian_velocity)
            episode.controls.append(action.copy())
            episode.state = DYN.step_numpy(episode.state, action).astype(
                np.float32, copy=False
            )
            episode.states.append(episode.state.copy())
            SS.advance_humans(episode.humans, episode.state)

    for episode in episodes:
        if episode.status is None:
            pedestrian_xy, _ = SS.collect_humans(episode.humans)
            if not EVAL._terminal_check(episode, pedestrian_xy):
                episode.status = "timeout"

    # -- flush success episodes only
    writer = CAP.ShardWriter(
        block_dir / "raw_obs", flush_every=int(hp_flush_every),
    )
    rows, pairs, outcome = [], [], dict(success=0, collision=0, timeout=0)
    per_gamma_rows = {}
    for episode in episodes:
        outcome[str(episode.status)] = outcome.get(str(episode.status), 0) + 1
        if episode.status != "success":
            continue
        for record in buffers[id(episode)]:
            writer.add(
                record["key"],
                grid=record["raw"]["grid"], low5=record["raw"]["low5"],
                history=record["raw"]["history"], token=record["raw"]["token"],
                meta=record["raw"]["meta"],
            )
            pairs.append(dict(key=record["key"], **record["pair"]))
            if record["row"] is not None:
                rows.append(record["row"])
                label = f"{episode.gamma:g}"
                per_gamma_rows[label] = per_gamma_rows.get(label, 0) + 1
    manifest = writer.close(extra=dict(
        controller="bon16_mpc_select", teacher_sha256=str(teacher_sha),
        block=int(block), ep0=int(ep0),
        reencode_checked=stats["reencode_checked"],
        reencode_max_abs_dev=stats["reencode_max_dev"],
        reencode_note=(
            "single-row re-encode vs batched forward; float-level agreement "
            "expected (not bitwise across batch shapes)"
        ),
    ))

    header = dict(
        status=STATUS_ARCHIVE,
        version=VERSION,
        semantics=dict(
            controller="bon16_mpc_select",
            teacher_checkpoint_sha256=str(teacher_sha),
            selection=(
                "argmin tuned v2 MPC cost over N raw temp-1 NFE-8 proposals; "
                "tie-break (cost, -progress, index); j=0 = canonical CRN latent"
            ),
            mpc_params=dict(lam=LAM, rho=RHO, r_eff=R_EFF, sigma_len=SIGMA_LEN),
            positives=(
                "executed steps of SUCCESS episodes whose chosen window passes "
                "the exact full-H10 GREEN verifier (verify_query.y) post-hoc"
            ),
            declared_gaps=(
                "native_cost/selected_sigma/marginal_sigma/conditional_ess/"
                "marginal_ESS_over_K are NaN (no acquisition here); "
                "positive_first16/positive_B32 are -1; mode_tags.changed means "
                "a non-canonical proposal was chosen"
            ),
            negatives="none in this archive (add an external negative archive)",
        ),
        config=dict(
            scene_profile=str(scene_profile), ep0=int(ep0), M=int(M), N=int(N),
            base_seed=int(base_seed), extra_seed=int(extra_seed),
            temperature=float(EVAL.TEMPERATURE), NFE=int(EVAL.NFE),
            T=int(T), H=int(H), n_ped=n_ped,
        ),
        provenance=dict(
            source=os.path.abspath(__file__), source_sha256=SOURCE_SHA,
            created_unix=time.time(),
        ),
        rows=rows,
    )
    torch.save(header, block_dir / "bon_archive.pt")
    torch.save(
        dict(status="SFM2_BON_DISTILL_PAIRS", version=VERSION,
             block=int(block), pairs=pairs),
        block_dir / "bon_pairs.pt",
    )

    summary = dict(
        status=STATUS_BLOCK, block=int(block), ep0=int(ep0),
        episodes=len(episodes), outcome=outcome,
        replans=stats["replans"],
        selection_changed_fraction=(
            stats["changed"] / max(1, stats["replans"])
        ),
        rescued_steps=stats["rescued_steps"],
        chosen_valid=stats["chosen_valid"],
        chosen_invalid=stats["chosen_invalid"],
        chosen_valid_fraction=(
            stats["chosen_valid"]
            / max(1, stats["chosen_valid"] + stats["chosen_invalid"])
        ),
        rows=len(rows), per_gamma_rows=per_gamma_rows,
        raw_records=manifest["rows"],
        reencode_checked=stats["reencode_checked"],
        reencode_max_abs_dev=stats["reencode_max_dev"],
    )
    (block_dir / "BLOCK_COMPLETE.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


SOURCE_SHA = None  # filled in main() after the file exists on disk


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--scene-profile", default="double_density_velocity_ood")
    parser.add_argument("--ep0-start", type=int, required=True)
    parser.add_argument("--block-stride", type=int, default=1000)
    parser.add_argument("--blocks", type=int, default=40)
    parser.add_argument("--M", type=int, default=20)
    parser.add_argument("--N", type=int, default=16)
    parser.add_argument("--base-seed-start", type=int, required=True)
    parser.add_argument("--extra-seed-start", type=int, required=True)
    parser.add_argument("--deadline-epoch", type=int, default=0,
                        help="unix time; no new block starts after this")
    parser.add_argument("--verify-workers", type=int, default=8)
    parser.add_argument("--hp-flush-every", type=int, default=2000)
    # Batched forward vs single-row re-encode differs at kernel-accumulation
    # order (measured up to ~7e-4 at production batch shapes); 5e-3 matches
    # the campaign's raw-audit atol.  The max deviation is recorded per block.
    parser.add_argument("--reencode-every", type=int, default=500)
    parser.add_argument("--reencode-atol", type=float, default=5.0e-3)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args(argv)

    _bootstrap(args.snapshot)
    global SOURCE_SHA
    SOURCE_SHA = sha256_file(os.path.abspath(__file__))

    import multiprocessing as mp
    from pathlib import Path

    import grid_policy_sfm_hp100 as GPS
    import sfm_b1_eval as BASE
    import sfm_hp100_dynamics as DYN
    import sfm_hp100_eval as EVAL
    import sfm_hp100_features as HPF
    import sfm_metrics2 as VERIFY
    import sfm_scene as SS

    torch.backends.cudnn.benchmark = False
    modules = dict(EVAL=EVAL, BASE=BASE, DYN=DYN, HPF=HPF, SS=SS, VERIFY=VERIFY)

    actual = sha256_file(args.checkpoint)
    if actual != args.expected_checkpoint_sha256:
        raise RuntimeError(
            f"teacher checkpoint sha mismatch: {actual} != "
            f"{args.expected_checkpoint_sha256}"
        )
    policy, _ = GPS.load_sfm_hp100_policy(args.checkpoint, device=args.device)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "RUN_ARGS.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()},
                   indent=2, sort_keys=True) + "\n"
    )

    context = mp.get_context("spawn")
    verify_pool = context.Pool(int(args.verify_workers))
    summaries, stopped_early = [], False
    try:
        for block in range(int(args.blocks)):
            if args.deadline_epoch and time.time() > float(args.deadline_epoch):
                stopped_early = True
                break
            block_dir = out_dir / f"block_{block:04d}"
            done = block_dir / "BLOCK_COMPLETE.json"
            if done.exists():
                summaries.append(json.loads(done.read_text()))
                continue
            if block_dir.exists():
                import shutil
                shutil.rmtree(block_dir)
            block_dir.mkdir(parents=True)
            started = time.time()
            summary = run_block(
                policy,
                block=block,
                ep0=int(args.ep0_start) + block * int(args.block_stride),
                M=int(args.M), N=int(args.N),
                base_seed=int(args.base_seed_start) + block,
                extra_seed=int(args.extra_seed_start) + block,
                scene_profile=args.scene_profile,
                device=args.device, modules=modules,
                verify_pool=verify_pool,
                teacher_sha=args.expected_checkpoint_sha256,
                block_dir=block_dir,
                hp_flush_every=int(args.hp_flush_every),
                reencode_every=int(args.reencode_every),
                reencode_atol=float(args.reencode_atol),
            )
            summary["seconds"] = round(time.time() - started, 1)
            summaries.append(summary)
            print(json.dumps({
                "label": args.label, "block": block,
                "rows": summary["rows"],
                "chosen_valid_fraction": round(
                    summary["chosen_valid_fraction"], 4
                ),
                "outcome": summary["outcome"],
                "seconds": summary["seconds"],
            }), flush=True)
    finally:
        verify_pool.close()
        verify_pool.join()

    total_rows = sum(s["rows"] for s in summaries)
    run_summary = dict(
        status=STATUS_RUN, label=str(args.label), version=VERSION,
        blocks_completed=len(summaries), stopped_early=stopped_early,
        total_rows=total_rows,
        teacher_checkpoint=os.path.abspath(args.checkpoint),
        teacher_checkpoint_sha256=args.expected_checkpoint_sha256,
        summaries=summaries,
    )
    (out_dir / "COLLECT_COMPLETE.json").write_text(
        json.dumps(run_summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"status": STATUS_RUN, "label": args.label,
                      "blocks": len(summaries), "rows": total_rows}),
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
