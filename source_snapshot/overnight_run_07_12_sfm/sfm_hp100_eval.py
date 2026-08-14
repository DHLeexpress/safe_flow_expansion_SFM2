"""Canonical raw evaluation for the clipped-dynamics SFM HP100 policy.

The learned policy is evaluated without acquisition, verifier selection,
guidance, or fallback: one temperature-one flow window is generated at each
context, its first action is executed, and the system replans.  The same
current-position tangent nominal ``H_P`` observation and componentwise action/velocity
caps used by HP100 demonstration collection are used here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import math
import multiprocessing as mp
import os

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_b1_eval as BASE
import sfm_hp100_dynamics as DYN
import sfm_hp100_features as HPF
import sfm_hp100_history as HPH
import sfm_metrics2 as VERIFY
import sfm_scene as SS


VERSION = "sfm_hp100_raw_clipped_v1"
T = 180
H = 10
NFE = 8
TEMPERATURE = 1.0
DEFAULT_NOISE_SEED = 2_026_080_2
DEFAULT_VERIFIER_WORKERS = 32


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Content hash used to authenticate a reconstructed CRN noise bank."""
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def noise_bank(*, M: int, d: int, seed: int = DEFAULT_NOISE_SEED) -> np.ndarray:
    generator = np.random.default_rng(int(seed))
    return generator.standard_normal(
        (len(SS.GAMMAS), int(M), T, int(d)), dtype=np.float32,
    )


@dataclass
class Episode:
    gamma_index: int
    rollout_index: int
    episode: int
    gamma: float
    humans: list
    state: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float32))
    hp_history: HPH.Hp100History = field(default_factory=HPH.Hp100History)
    controls: list[np.ndarray] = field(default_factory=list)
    proposals: list[np.ndarray] = field(default_factory=list)
    states: list[np.ndarray] = field(
        default_factory=lambda: [np.zeros(4, np.float32)]
    )
    ped_xy: list[np.ndarray] = field(default_factory=list)
    ped_vel: list[np.ndarray] = field(default_factory=list)
    status: str | None = None
    minimum_clearance: float = float("inf")


def _clearance(state, pedestrian_xy) -> float:
    pedestrian_xy = np.asarray(pedestrian_xy, np.float32).reshape(-1, 2)
    if not len(pedestrian_xy):
        return float("inf")
    return float(
        np.linalg.norm(pedestrian_xy - state[:2][None], axis=1).min()
        - SS.R_PED
    )


def _terminal_check(episode: Episode, pedestrian_xy) -> bool:
    clearance = _clearance(episode.state, pedestrian_xy)
    episode.minimum_clearance = min(episode.minimum_clearance, clearance)
    if clearance < 0.0:
        episode.status = "collision"
    elif float(np.linalg.norm(episode.state[:2] - SS.GOAL)) < 0.5:
        episode.status = "success"
    return episode.status is not None


def _obstacles(pedestrian_xy) -> np.ndarray:
    pedestrian_xy = np.asarray(pedestrian_xy, np.float32).reshape(-1, 2)
    return np.concatenate(
        (
            pedestrian_xy,
            np.full((len(pedestrian_xy), 1), SS.R_PED, np.float32),
        ),
        axis=1,
    )


@torch.no_grad()
def run_batched_raw(
    policy,
    *,
    scene_profile: str,
    ep0: int,
    M: int,
    noise: np.ndarray,
    device: str,
    retain_proposals: bool = False,
) -> list[dict]:
    """Run the fixed CRN bank using the canonical unguided HP100 policy."""
    if tuple(noise.shape) != (len(SS.GAMMAS), int(M), T, int(policy.d)):
        raise ValueError("HP100 raw noise bank shape mismatch")
    if noise.dtype != np.float32:
        raise ValueError("HP100 raw noise bank must be float32")
    environment = SS.scene_profile(scene_profile)
    episodes = [
        Episode(
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

    for step in range(T):
        active = []
        hp_histories, low5, control_histories, latents = [], [], [], []
        for episode in episodes:
            if episode.status is not None:
                continue
            pedestrian_xy, pedestrian_velocity = SS.collect_humans(episode.humans)
            pedestrian_xy = np.asarray(pedestrian_xy, np.float32)
            pedestrian_velocity = np.asarray(pedestrian_velocity, np.float32)
            if _terminal_check(episode, pedestrian_xy):
                continue
            frame = HPF.hp100_frame(
                episode.state[:2],
                _obstacles(pedestrian_xy),
                sensing=SS.R_SENSE,
                n_base=HPF.POLYTOPE_N_BASE,
                obstacle_velocities=pedestrian_velocity,
                robot_velocity=episode.state[2:4],
                predict_gain=HPF.PREDICT_GAIN,
                predict_tau=HPF.PREDICT_TAU,
            )
            active.append((episode, pedestrian_xy.copy(), pedestrian_velocity.copy()))
            hp_histories.append(episode.hp_history.append(frame))
            low5.append(torch.as_tensor(HPF.low5(
                episode.state, SS.GOAL, episode.gamma,
            )))
            control_histories.append(torch.as_tensor(HPF.hist_pad(
                episode.controls[-HPF.K_HIST:],
            )))
            latents.append(noise[
                episode.gamma_index, episode.rollout_index, step,
            ])
        if not active:
            break

        hp_tensor = torch.stack(hp_histories).to(device)
        low_tensor = torch.stack(low5).to(device)
        history_tensor = torch.stack(control_histories).to(device)
        context = policy.ctx_from(hp_tensor, low_tensor, history_tensor)
        latent_tensor = torch.as_tensor(np.asarray(latents), device=device)
        windows = BASE.integrate_latents(
            policy, TEMPERATURE * latent_tensor, context, nfe=NFE,
        ).reshape(len(active), H, 2)
        windows = windows.detach().cpu().numpy().astype(np.float32)

        for (episode, pedestrian_xy, pedestrian_velocity), window in zip(
            active, windows
        ):
            action = DYN.clip_action_numpy(window[0]).astype(np.float32, copy=False)
            if retain_proposals:
                episode.proposals.append(window.copy())
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
            if not _terminal_check(episode, pedestrian_xy):
                episode.status = "timeout"
        success = episode.status == "success"
        row = dict(
            episode=int(episode.episode), gamma=float(episode.gamma),
            status=str(episode.status), success=bool(success),
            collision=episode.status == "collision",
            timeout=episode.status == "timeout", steps=len(episode.controls),
            time_to_goal=(len(episode.controls) * DYN.DT if success else None),
            min_clearance=float(episode.minimum_clearance),
            successful_clearance=(
                float(episode.minimum_clearance) if success else None
            ),
            states=np.asarray(episode.states, np.float32),
            controls=np.asarray(episode.controls, np.float32).reshape(-1, 2),
            ped_xy=np.asarray(episode.ped_xy, np.float32).reshape(
                len(episode.controls), int(environment["n_ped"]), 2,
            ),
            ped_vel=np.asarray(episode.ped_vel, np.float32).reshape(
                len(episode.controls), int(environment["n_ped"]), 2,
            ),
        )
        if retain_proposals:
            row["proposals"] = np.asarray(
                episode.proposals, np.float32,
            ).reshape(-1, H, 2)
            if row["proposals"].shape != (len(episode.controls), H, 2):
                raise RuntimeError("retained raw proposals do not align with executed contexts")
        rows.append(row)
    return rows


def clipped_rollout_positions(state, controls) -> np.ndarray:
    current = np.asarray(state, np.float32).reshape(4).copy()
    positions = [current[:2].copy()]
    for action in np.asarray(controls, np.float32).reshape(-1, 2):
        current = DYN.step_numpy(current, action).astype(np.float32, copy=False)
        positions.append(current[:2].copy())
    return np.asarray(positions, np.float32)


def verify_executed_window(state, controls, ped_xy, ped_vel, gamma) -> dict:
    """Exact GREEN verification using the same capped physical rollout."""
    try:
        controls = np.asarray(controls, np.float32).reshape(-1, 2)
        if not 1 <= len(controls) <= H:
            raise ValueError("executed HP100 window needs 1 <= H_t <= 10")
        robot = clipped_rollout_positions(state, controls)
        pedestrian = VERIFY.predict_pedestrians(ped_xy, ped_vel, H=len(controls))
        task = VERIFY.taskspace_ok(robot)
        collision = VERIFY.collision_free_time_indexed(robot, pedestrian)
        certificate, _, diagnostics = VERIFY.certify_moving_window(
            robot, pedestrian, gamma,
        )
        return dict(
            resolved=True, y=int(task and collision and certificate),
            taskspace=bool(task), collision_free=bool(collision),
            certificate=bool(certificate), diagnostics=diagnostics,
            window_horizon=len(controls),
        )
    except Exception as error:
        return dict(resolved=False, error=f"{type(error).__name__}: {error}")


def _verify_executed_episode(row: dict) -> dict:
    """Verify one complete episode; top-level for spawn-process pickling."""
    valid = 0
    controls = row["controls"]
    for start in range(int(row["steps"])):
        stop = min(start + H, int(row["steps"]))
        result = verify_executed_window(
            row["states"][start], controls[start:stop], row["ped_xy"][start],
            row["ped_vel"][start], row["gamma"],
        )
        if not result.get("resolved", False):
            raise RuntimeError(result.get("error", "HP100 verifier failed"))
        valid += int(result["y"])
    count = int(row["steps"])
    value = {
        key: item for key, item in row.items()
        if key not in ("states", "controls", "ped_xy", "ped_vel")
    }
    value.update(
        validity=(valid / count if count else 0.0),
        valid_windows=int(valid), evaluated_windows=count, verifier_errors=0,
    )
    return value


def attach_validity(rows: list[dict], *, executor=None) -> list[dict]:
    """Attach exact window validity, optionally ordered by episode workers."""
    if executor is None:
        return [_verify_executed_episode(row) for row in rows]
    return list(executor.map(_verify_executed_episode, rows, chunksize=1))


def _wilson(successes: int, total: int, z=1.959963984540054):
    if not total:
        return [None, None]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(
        proportion * (1 - proportion) / total + z * z / (4 * total * total)
    ) / denominator
    return [center - radius, center + radius]


def _mean(values):
    finite = [float(value) for value in values if value is not None]
    return None if not finite else float(np.mean(finite))


def summarize(rows: list[dict]) -> dict:
    def one(cell):
        n = len(cell)
        success = sum(bool(row["success"]) for row in cell)
        collision = sum(bool(row["collision"]) for row in cell)
        timeout = sum(bool(row["timeout"]) for row in cell)
        if success + collision + timeout != n:
            raise RuntimeError("HP100 outcomes do not partition the cell")
        return dict(
            n=n, SR=success / n, SR_wilson95=_wilson(success, n),
            CR=collision / n, CR_wilson95=_wilson(collision, n),
            timeout=timeout / n, timeout_wilson95=_wilson(timeout, n),
            Validity=_mean([row.get("validity") for row in cell]),
            successful_clearance=_mean([
                row["successful_clearance"] for row in cell
            ]),
            successful_time_to_goal=_mean([
                row["time_to_goal"] for row in cell
            ]),
        )
    return dict(
        pooled=one(rows),
        per_gamma={
            str(gamma): one([
                row for row in rows if float(row["gamma"]) == float(gamma)
            ])
            for gamma in SS.GAMMAS
        },
    )


def evaluate(
    policy,
    *,
    scene_profile: str,
    ep0: int,
    M: int,
    device: str,
    seed: int = DEFAULT_NOISE_SEED,
    with_validity: bool = True,
    validity_executor=None,
) -> tuple[list[dict], dict]:
    noise = noise_bank(M=M, d=policy.d, seed=seed)
    rows = run_batched_raw(
        policy, scene_profile=scene_profile, ep0=ep0, M=M,
        noise=noise, device=device,
    )
    compact = attach_validity(rows, executor=validity_executor) if with_validity else [
        {
            key: value for key, value in row.items()
            if key not in ("states", "controls", "ped_xy", "ped_vel")
        }
        for row in rows
    ]
    return compact, summarize(compact)


def id_raw_gate(
    policy, *, M: int, ep0: int, device: str,
    seed: int = DEFAULT_NOISE_SEED,
    validity_executor=None,
) -> dict:
    rows, summary = evaluate(
        policy, scene_profile="matched_id", ep0=ep0, M=M, device=device,
        seed=int(seed), with_validity=True,
        validity_executor=validity_executor,
    )
    del rows
    return dict(
        bank="fixed matched-ID scenario bank, raw temp=1",
        ep0=int(ep0), M_per_gamma=int(M), NFE=NFE,
        noise_seed=int(seed), temperature=TEMPERATURE,
        pooled=summary["pooled"],
        per_gamma={
            gamma: {
                key: cell[key]
                for key in (
                    "SR", "CR", "timeout", "Validity",
                    "successful_clearance", "successful_time_to_goal",
                )
            }
            for gamma, cell in summary["per_gamma"].items()
        },
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--scene-profile", required=True, choices=SS.SCIENTIFIC_EVAL_PROFILES)
    parser.add_argument("--ep0", required=True, type=int)
    parser.add_argument("--M", required=True, type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--noise-seed", type=int, default=DEFAULT_NOISE_SEED)
    parser.add_argument(
        "--verifier-workers", type=int, default=DEFAULT_VERIFIER_WORKERS,
    )
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    policy, checkpoint = GPS.load_sfm_hp100_policy(args.checkpoint, device=args.device)
    if int(args.verifier_workers) <= 0:
        raise ValueError("verifier-workers must be positive")
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=int(args.verifier_workers), mp_context=context,
    ) as validity_executor:
        rows, summary = evaluate(
            policy, scene_profile=args.scene_profile, ep0=args.ep0, M=args.M,
            device=args.device, seed=args.noise_seed, with_validity=True,
            validity_executor=validity_executor,
        )
    declared_noise = noise_bank(M=args.M, d=policy.d, seed=args.noise_seed)
    payload = dict(
        status="SFM_HP100_RAW_EVAL_COMPLETE", version=VERSION,
        evaluator_source=os.path.abspath(__file__),
        evaluator_sha256=sha256_file(__file__),
        checkpoint=os.path.abspath(args.checkpoint),
        checkpoint_sha256=sha256_file(args.checkpoint),
        architecture=checkpoint["config"], dynamics=DYN.contract(),
        verifier=dict(
            contract=VERIFY.verifier_manifest(),
            evaluator_source=os.path.abspath(inspect.getsourcefile(VERIFY)),
            evaluator_sha256=sha256_file(inspect.getsourcefile(VERIFY)),
            polytope_source=os.path.abspath(
                inspect.getsourcefile(VERIFY.VP)
            ),
            polytope_sha256=sha256_file(inspect.getsourcefile(VERIFY.VP)),
        ),
        observation=HPF.contract() if hasattr(HPF, "contract") else dict(
            shape=[10, 32, 100], polytope_n_base=16,
        ),
        scene=SS.scene_profile(args.scene_profile), ep0=args.ep0,
        M_per_gamma=args.M, temperature=TEMPERATURE, NFE=NFE,
        noise_seed=int(args.noise_seed),
        noise_bank=dict(
            seed=int(args.noise_seed), shape=list(declared_noise.shape),
            dtype=str(declared_noise.dtype), sha256=array_sha256(declared_noise),
        ),
        validity_parallelism=dict(
            workers=int(args.verifier_workers), start_method="spawn",
            task="one complete episode; ordered executor.map",
        ),
        summary=summary, rows=rows,
        semantics=(
            "unguided raw flow; temp=1; NFE=8; one H10 window/context; "
            "execute first clipped action; componentwise clipped physical velocity"
        ),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    temporary = os.path.abspath(args.out) + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    os.replace(temporary, os.path.abspath(args.out))
    print(json.dumps({
        "status": payload["status"], "out": os.path.abspath(args.out),
        "pooled": summary["pooled"],
    }), flush=True)


if __name__ == "__main__":
    main()
