"""Batched SFM checkpoint curves with leak-free temperature selection.

This evaluator never filters or changes an action.  The full-curve command:

1. jointly selects one latent-noise temperature per gamma on a fixed
   M=10/gamma validation bank;
2. freezes that seven-temperature vector; and
3. evaluates both canonical temperature one and the selected vector on
   a disjoint M=50/gamma screening bank.

The candidate command applies the same M=10 temperature-vector selection to
one checkpoint and evaluates only that locked vector on the disjoint M=50
bank.  The single-vector confirmation option similarly evaluates exactly one
predeclared vector on the final M=100 bank, so canonical and tuned policies can
be launched independently without duplicate rollouts.

Only compact metrics are written.  ``V_safe`` is an episode predicate: the
executed trajectory must be physically collision/task-space safe and the full
generated H=10 plan at every executed planning context must pass the same exact
moving-pedestrian certificate used by B1 acquisition.  Reaching the goal ends
execution, but never shortens the last plan submitted to the verifier.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import _paths  # noqa: F401
import grid_feats as GF
import grid_policy_sfm as GPS
import sfm_b1_eval as BE
import sfm_hp_history as HH
import sfm_metrics2 as SM
import sfm_protocol as SP
import sfm_scene as SS


VERSION = "sfm_b1_curve_eval_v2"
TEMPERATURES = (0.90, 0.95, 1.00, 1.05, 1.10)
TUNE_EP0 = 300_000
TUNE_M = 10
SCREEN_EP0 = 320_000
SCREEN_M = 50
NFE = 8
T = SP.T
H = SP.H


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _write_json(path: str | os.PathLike[str], payload: Any) -> None:
    path = os.fspath(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    os.replace(temporary, path)


def parse_rounds(value: str) -> list[int]:
    if ":" in value:
        start, stop = map(int, value.split(":"))
        rounds = list(range(start, stop + 1))
    else:
        rounds = [int(item) for item in value.split(",")]
    if not rounds or min(rounds) < 0 or len(rounds) != len(set(rounds)):
        raise ValueError("rounds must be unique non-negative integers")
    return rounds


def assert_disjoint_banks(tune_ep0: int, tune_m: int, screen_ep0: int, screen_m: int) -> None:
    tune = set(range(int(tune_ep0), int(tune_ep0) + int(tune_m)))
    screen = set(range(int(screen_ep0), int(screen_ep0) + int(screen_m)))
    if int(tune_m) < 1 or int(screen_m) < 1 or tune & screen:
        raise ValueError("temperature-selection and screening scenario banks must be nonempty and disjoint")


def assert_final_confirmation_bank(ep0: int, M: int) -> None:
    declared = {int(SP.FINAL_CONFIRM_EP0), int(SP.ADAPTIVE_CONFIRM_EP0)}
    if int(ep0) not in declared or int(M) != 100:
        raise ValueError(
            "scientific confirmation is locked to the declared disjoint M100 bank"
        )
    assert_disjoint_banks(TUNE_EP0, TUNE_M, int(ep0), int(M))
    assert_disjoint_banks(SCREEN_EP0, SCREEN_M, int(ep0), int(M))


def noise_bank(*, scene_profile: str, ep0: int, M: int, d: int, T_steps: int = T):
    contract = dict(
        version=VERSION, scene_profile=str(scene_profile), ep0=int(ep0), M=int(M),
        gammas=list(map(float, SP.GAMMAS)), T=int(T_steps), d=int(d),
    )
    seed = int(_sha256_json(contract)[:16], 16) % (2**63 - 1)
    generator = np.random.default_rng(seed)
    values = generator.standard_normal(
        (len(SP.GAMMAS), int(M), int(T_steps), int(d)), dtype=np.float32,
    )
    return values, dict(
        **contract, seed=seed, dtype="float32", shape=list(values.shape),
        sha256=hashlib.sha256(values.tobytes(order="C")).hexdigest(),
        CRN=("identical per (gamma,episode,step) across checkpoints and temperatures; "
             "independent latent slice across gamma while scenario IDs remain paired"),
    )


def temperature_vector(value: float | dict[str, float]) -> dict[str, float]:
    """Return a complete, JSON-stable gamma -> temperature mapping."""
    if isinstance(value, dict):
        expected = {str(gamma) for gamma in SP.GAMMAS}
        if set(value) != expected:
            raise ValueError("temperature vector must contain every declared gamma exactly once")
        result = {str(gamma): float(value[str(gamma)]) for gamma in SP.GAMMAS}
    else:
        result = {str(gamma): float(value) for gamma in SP.GAMMAS}
    if any(not math.isfinite(item) or item <= 0.0 for item in result.values()):
        raise ValueError("temperatures must be finite and positive")
    return result


def _is_canonical_temperature(value: float | dict[str, float]) -> bool:
    return all(item == 1.0 for item in temperature_vector(value).values())


def cell_contract_key(*, checkpoint_sha256: str, round_i: int, scene_profile: str,
                      bank: dict, temperature: float | dict[str, float], role: str) -> str:
    temperatures = temperature_vector(temperature)
    return _sha256_json(dict(
        version=VERSION, evaluator_sha256=_sha256_file(__file__),
        checkpoint_sha256=str(checkpoint_sha256), round=int(round_i),
        scene_profile=str(scene_profile), bank=bank, temperature_by_gamma=temperatures,
        role=str(role), NFE=NFE, T=T, verifier=SM.verifier_manifest(),
        v_safe=("whole executed trajectory is collision/task-space safe and every generated "
                "full-H=10 planned window passes the exact moving-pedestrian certificate at "
                "the contemporaneous context and episode gamma"),
    ))


@dataclass
class _Episode:
    gamma_index: int
    rollout_index: int
    episode: int
    gamma: float
    humans: list
    state: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float32))
    history: HH.HpHistory = field(default_factory=HH.HpHistory)
    controls: list[np.ndarray] = field(default_factory=list)
    states: list[np.ndarray] = field(default_factory=lambda: [np.zeros(4, np.float32)])
    ped_xy: list[np.ndarray] = field(default_factory=list)
    ped_vel: list[np.ndarray] = field(default_factory=list)
    planned_segments: list[np.ndarray] = field(default_factory=list)
    status: str | None = None
    min_clearance: float = float("inf")


def _clearance(state: np.ndarray, ped_xy: np.ndarray) -> float:
    if not len(ped_xy):
        return float("inf")
    return float(np.linalg.norm(ped_xy - state[:2][None], axis=1).min() - SS.R_PED)


def _terminal_check(episode: _Episode, ped_xy: np.ndarray) -> bool:
    clearance = _clearance(episode.state, ped_xy)
    episode.min_clearance = min(episode.min_clearance, clearance)
    if clearance < 0.0:
        episode.status = "collision"
    elif float(np.linalg.norm(episode.state[:2] - SS.GOAL)) < 0.5:
        episode.status = "success"
    return episode.status is not None


@torch.no_grad()
def run_batched_raw(policy, *, scene_profile: str, ep0: int, M: int,
                    base_noise: np.ndarray, temperature: float | dict[str, float], device: str,
                    T_steps: int = T) -> list[dict]:
    """Run all gamma/episode cells with one policy batch per control tick."""
    temperatures = temperature_vector(temperature)
    environment = SS.scene_profile(scene_profile)
    expected = (len(SP.GAMMAS), int(M), int(T_steps), int(policy.d))
    if tuple(base_noise.shape) != expected or base_noise.dtype != np.float32:
        raise ValueError(f"noise bank {base_noise.shape}/{base_noise.dtype} != {expected}/float32")
    episodes = [
        _Episode(
            gamma_index=gamma_index, rollout_index=rollout_index,
            episode=int(ep0) + rollout_index, gamma=float(gamma),
            humans=SS.make_humans(
                int(ep0) + rollout_index, 0, environment["n_ped"],
                tuple(environment["ped_speed_range"]),
            ),
        )
        for gamma_index, gamma in enumerate(SP.GAMMAS)
        for rollout_index in range(int(M))
    ]
    for step in range(int(T_steps)):
        active, hp10, lows, histories, latents = [], [], [], [], []
        for episode in episodes:
            if episode.status is not None:
                continue
            ped_xy, ped_vel = SS.collect_humans(episode.humans)
            if _terminal_check(episode, ped_xy):
                continue
            obstacles = np.concatenate([
                ped_xy,
                np.full((len(ped_xy), 1), SS.R_PED, np.float32),
            ], axis=1)
            raw_grid = torch.as_tensor(
                GF.axis_grid(episode.state[:2], obstacles, 0.0, R=SS.R_SENSE, sensing=SS.R_SENSE)
            )
            active.append((episode, ped_xy.copy(), ped_vel.copy()))
            hp10.append(episode.history.append(raw_grid))
            lows.append(torch.as_tensor(GF.low5(episode.state, SS.GOAL, episode.gamma)))
            histories.append(torch.as_tensor(GF.hist_pad(
                np.asarray(episode.controls[-16:]) if episode.controls else np.zeros((0, 2)), 16
            )))
            latents.append(base_noise[
                episode.gamma_index, episode.rollout_index, step,
            ] * temperatures[str(episode.gamma)])
        if not active:
            break
        hp10_tensor = torch.stack(hp10).to(device)
        low_tensor = torch.stack(lows).to(device)
        history_tensor = torch.stack(histories).to(device)
        context = policy.ctx_from(hp10_tensor, low_tensor, history_tensor)
        windows = BE.integrate_latents(
            policy, torch.as_tensor(np.asarray(latents), device=device), context, nfe=NFE,
        ).detach().cpu().numpy()
        for (episode, ped_xy, ped_vel), window in zip(active, windows):
            if tuple(window.shape) != (H, 2):
                raise RuntimeError(f"generated plan {tuple(window.shape)} != {(H, 2)}")
            action = np.asarray(window[0], np.float32)
            episode.ped_xy.append(ped_xy)
            episode.ped_vel.append(ped_vel)
            episode.planned_segments.append(
                SM.rollout_positions(episode.state, np.asarray(window, np.float32))
            )
            episode.controls.append(action)
            episode.state = BE._step(episode.state, action)
            episode.states.append(episode.state.copy())
            SS.advance_humans(episode.humans, episode.state)
    rows = []
    for episode in episodes:
        if episode.status is None:
            ped_xy, _ = SS.collect_humans(episode.humans)
            if not _terminal_check(episode, ped_xy):
                episode.status = "timeout"
        success = episode.status == "success"
        rows.append(dict(
            episode=episode.episode, gamma=episode.gamma, status=episode.status,
            success=success, collision=episode.status == "collision",
            timeout=episode.status == "timeout", steps=len(episode.controls),
            time_to_goal=(len(episode.controls) * SS.DT if success else None),
            min_clearance=float(episode.min_clearance),
            successful_clearance=(float(episode.min_clearance) if success else None),
            states=np.asarray(episode.states, np.float32),
            planned_segments=np.asarray(episode.planned_segments, np.float32),
            ped_xy=np.asarray(episode.ped_xy, np.float32),
            ped_vel=np.asarray(episode.ped_vel, np.float32),
        ))
    return rows


def _v_safe_worker(row: dict) -> dict:
    states = np.asarray(row["states"], np.float32)
    planned_segments = np.asarray(row["planned_segments"], np.float32)
    ped_xy = np.asarray(row["ped_xy"], np.float32)
    ped_vel = np.asarray(row["ped_vel"], np.float32)
    n_steps = int(row["steps"])
    if (len(states) != n_steps + 1 or len(planned_segments) != n_steps
            or len(ped_xy) != n_steps or len(ped_vel) != n_steps):
        return dict(v_safe=False, verifier_errors=1, windows=0)
    if n_steps and tuple(planned_segments.shape[1:]) != (H + 1, 2):
        return dict(v_safe=False, verifier_errors=1, windows=0)
    if row["collision"] or not SM.taskspace_ok(states[:, :2]) or n_steps < 1:
        return dict(v_safe=False, verifier_errors=0, windows=0)
    windows = 0
    for segment, current_xy, current_vel in zip(planned_segments, ped_xy, ped_vel):
        try:
            pedestrians = SM.predict_pedestrians(current_xy, current_vel, H=H)
            taskspace = SM.taskspace_ok(segment)
            collision_free = SM.collision_free_time_indexed(segment, pedestrians)
            certified, _, _ = SM.certify_moving_window(segment, pedestrians, row["gamma"])
        except Exception:
            return dict(v_safe=False, verifier_errors=1, windows=windows)
        windows += 1
        if not (taskspace and collision_free and certified):
            return dict(v_safe=False, verifier_errors=0, windows=windows)
    return dict(v_safe=True, verifier_errors=0, windows=windows)


def attach_v_safe(rows: list[dict], executor) -> list[dict]:
    validity = [future.result() for future in _submit_v_safe(rows, executor)]
    return _attach_validity(rows, validity)


def _submit_v_safe(rows: list[dict], executor):
    return [executor.submit(_v_safe_worker, row) for row in rows]


def _attach_validity(rows: list[dict], validity: list[dict]) -> list[dict]:
    output = []
    for row, value in zip(rows, validity):
        compact = {key: item for key, item in row.items()
                   if key not in ("states", "planned_segments", "ped_xy", "ped_vel")}
        compact.update(value)
        output.append(compact)
    return output


def _summarize_one(rows: list[dict], seed: int) -> dict:
    n = len(rows)
    if not n:
        raise ValueError("cannot summarize an empty evaluation cell")
    successes = sum(bool(row["success"]) for row in rows)
    collisions = sum(bool(row["collision"]) for row in rows)
    timeouts = sum(bool(row["timeout"]) for row in rows)
    valid = sum(bool(row["v_safe"]) for row in rows)
    if successes + collisions + timeouts != n:
        raise RuntimeError("SR/CR/timeout do not partition the cell")
    return dict(
        n=n,
        SR=successes / n, SR_wilson95=BE.wilson(successes, n),
        CR=collisions / n, CR_wilson95=BE.wilson(collisions, n),
        timeout=timeouts / n, timeout_wilson95=BE.wilson(timeouts, n),
        V_safe=valid / n, V_safe_wilson95=BE.wilson(valid, n),
        successful_clearance=BE.bootstrap_mean(
            [row["successful_clearance"] for row in rows], seed=seed,
        ),
        successful_time_to_goal=BE.bootstrap_mean(
            [row["time_to_goal"] for row in rows], seed=seed + 1,
        ),
        verifier_errors=sum(int(row["verifier_errors"]) for row in rows),
        certified_windows=sum(int(row["windows"]) for row in rows),
    )


def _cluster_bootstrap_interval(rows: list[dict], key: str, *, seed: int,
                                draws: int = 2_000) -> list[float | None]:
    """Bootstrap paired-gamma rows by scenario/episode, not by individual row."""
    episode_ids = sorted({int(row["episode"]) for row in rows})
    sums, counts = [], []
    for episode in episode_ids:
        values = [row.get(key) for row in rows if int(row["episode"]) == episode]
        finite = [float(value) for value in values
                  if value is not None and math.isfinite(float(value))]
        sums.append(sum(finite)); counts.append(len(finite))
    if not episode_ids or not sum(counts):
        return [None, None]
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, len(episode_ids), size=(int(draws), len(episode_ids)))
    numerator = np.asarray(sums, float)[indices].sum(axis=1)
    denominator = np.asarray(counts, float)[indices].sum(axis=1)
    samples = numerator[denominator > 0] / denominator[denominator > 0]
    if not len(samples):
        return [None, None]
    return list(map(float, np.quantile(samples, [.025, .975])))


def summarize(rows: list[dict], seed: int = 0) -> dict:
    per_gamma = {
        str(gamma): _summarize_one(
            [row for row in rows if float(row["gamma"]) == float(gamma)], seed + index * 10,
        )
        for index, gamma in enumerate(SP.GAMMAS)
    }
    pooled = _summarize_one(rows, seed + 100)
    for metric, key in (("SR", "success"), ("CR", "collision"),
                        ("timeout", "timeout"), ("V_safe", "v_safe")):
        pooled.pop(f"{metric}_wilson95")
        pooled[f"{metric}_cluster_bootstrap95"] = _cluster_bootstrap_interval(
            rows, key, seed=seed + 200 + len(metric),
        )
    pooled["successful_clearance"]["interval95"] = _cluster_bootstrap_interval(
        rows, "successful_clearance", seed=seed + 300,
    )
    pooled["successful_time_to_goal"]["interval95"] = _cluster_bootstrap_interval(
        rows, "time_to_goal", seed=seed + 301,
    )
    pooled["ci_method"] = "scenario-cluster bootstrap across the seven paired gamma rows"
    return dict(pooled=pooled, per_gamma=per_gamma)


def assert_zero_verifier_errors(summary: dict, *, role: str) -> None:
    """Fail closed before any verifier-tainted metric can affect selection."""
    cells = [("pooled", summary["pooled"])] + [
        (str(gamma), summary["per_gamma"][str(gamma)]) for gamma in SP.GAMMAS
    ]
    contaminated = {
        name: int(cell.get("verifier_errors", -1))
        for name, cell in cells if int(cell.get("verifier_errors", -1)) != 0
    }
    if contaminated:
        raise RuntimeError(f"{role} contains verifier errors: {contaminated}")


def _ordering_score(values: list[float | None]) -> tuple[int, float]:
    missing = sum(value is None or not math.isfinite(float(value)) for value in values)
    finite = [None if value is None or not math.isfinite(float(value)) else float(value)
              for value in values]
    # Gamma increases left-to-right. Desired safety adaptation is non-increasing:
    # low gamma has higher clearance and longer successful time-to-goal.
    violation = sum(
        max(0.0, right - left) for left, right in zip(finite, finite[1:])
        if left is not None and right is not None
    )
    return int(missing), float(violation)


def _metric_selection_key(cells: list[dict], pooled: dict,
                          temperatures: dict[str, float]) -> tuple:
    clearance = [cell["successful_clearance"]["mean"] for cell in cells]
    times = [cell["successful_time_to_goal"]["mean"] for cell in cells]
    clearance_missing, clearance_violation = _ordering_score(clearance)
    time_missing, time_violation = _ordering_score(times)
    return (
        max(cell["CR"] for cell in cells), pooled["CR"],
        -min(cell["V_safe"] for cell in cells), -pooled["V_safe"],
        clearance_missing, clearance_violation, time_missing, time_violation,
        max(cell["timeout"] for cell in cells), pooled["timeout"],
        -min(cell["SR"] for cell in cells), -pooled["SR"],
        sum(abs(math.log(temperatures[str(gamma)])) for gamma in SP.GAMMAS),
        *(temperatures[str(gamma)] for gamma in SP.GAMMAS),
    )


def temperature_selection_key(summary: dict,
                              temperature: float | dict[str, float]) -> tuple:
    temperatures = temperature_vector(temperature)
    cells = [summary["per_gamma"][str(gamma)] for gamma in SP.GAMMAS]
    return _metric_selection_key(cells, summary["pooled"], temperatures)


def _validation_vector_key(candidates: dict[float, dict], values: tuple[float, ...]) -> tuple:
    temperatures = {str(gamma): float(value) for gamma, value in zip(SP.GAMMAS, values)}
    cells = [
        candidates[temperatures[str(gamma)]]["per_gamma"][str(gamma)]
        for gamma in SP.GAMMAS
    ]
    pooled = {
        metric: sum(float(cell[metric]) for cell in cells) / len(cells)
        for metric in ("CR", "V_safe", "timeout", "SR")
    }
    return _metric_selection_key(cells, pooled, temperatures)


def select_temperature_vector(candidates: dict[float, dict]) -> tuple[dict[str, float], list]:
    if set(map(float, candidates)) != set(TEMPERATURES):
        raise ValueError("temperature selection requires the complete predeclared grid")
    values = min(
        itertools.product(TEMPERATURES, repeat=len(SP.GAMMAS)),
        key=lambda item: _validation_vector_key(candidates, item),
    )
    chosen = {str(gamma): float(value) for gamma, value in zip(SP.GAMMAS, values)}
    return chosen, list(_validation_vector_key(candidates, values))


@dataclass
class _PendingCell:
    path: str
    payload: dict
    rows: list[dict]
    futures: list


def _begin_cell(policy, *, checkpoint_sha: str, round_i: int, scene_profile: str,
                ep0: int, M: int, noise: np.ndarray, noise_meta: dict,
                temperature: float | dict[str, float], role: str, device: str, executor,
                path: str | os.PathLike[str]) -> dict | _PendingCell:
    temperatures = temperature_vector(temperature)
    key = cell_contract_key(
        checkpoint_sha256=checkpoint_sha, round_i=round_i, scene_profile=scene_profile,
        bank=noise_meta, temperature=temperature, role=role,
    )
    path = os.fspath(path)
    if os.path.exists(path):
        with open(path) as stream:
            payload = json.load(stream)
        if (payload.get("status") != "SFM_B1_CURVE_CELL_COMPLETE"
                or payload.get("cell_key") != key):
            raise RuntimeError(f"stale evaluation cell: {path}")
        return payload
    rows = run_batched_raw(
        policy, scene_profile=scene_profile, ep0=ep0, M=M, base_noise=noise,
        temperature=temperature, device=device,
    )
    payload = dict(
        status="SFM_B1_CURVE_CELL_COMPLETE", cell_key=key, role=role,
        round=int(round_i), checkpoint_sha256=checkpoint_sha,
        scene_profile=scene_profile, ep0=int(ep0), M_per_gamma=int(M),
        temperature_by_gamma=temperatures, noise_bank=noise_meta,
        metric_semantics=dict(
            policy="unguided flow; NFE=8; one generated window/context; execute first action",
            V_safe=("episode conjunction over physical task/collision safety and the exact "
                    "certificate for every generated full-H=10 plan, including the final "
                    "pre-goal planning context"),
            clearance="mean of per-trajectory minimum clearance over successful episodes only",
            time="successful episodes only",
        ),
    )
    return _PendingCell(
        path=path, payload=payload, rows=rows,
        futures=_submit_v_safe(rows, executor),
    )


def _finish_cell(cell: dict | _PendingCell) -> dict:
    if isinstance(cell, dict):
        assert_zero_verifier_errors(cell["summary"], role=cell.get("role", "cached cell"))
        return cell
    validity = [future.result() for future in cell.futures]
    compact = _attach_validity(cell.rows, validity)
    cell.payload["summary"] = summarize(
        compact, seed=int(cell.payload["round"]) * 1000 + int(cell.payload["ep0"]),
    )
    assert_zero_verifier_errors(cell.payload["summary"], role=cell.payload["role"])
    _write_json(cell.path, cell.payload)
    return cell.payload


def _run_cell(policy, **kwargs) -> dict:
    """Synchronous compatibility wrapper; the main evaluator uses begin/finish pipelining."""
    return _finish_cell(_begin_cell(policy, **kwargs))


def _plot(records: list[dict], output_stem: str, *, best_round: int) -> None:
    selected = {row["round"]: row for row in records if row["mode"] == "validation_selected_temperature"}
    canonical = {row["round"]: row for row in records if row["mode"] == "canonical_temp1"}
    rounds = sorted(selected)
    colors = plt.cm.plasma(np.linspace(.05, .95, len(SP.GAMMAS)))
    specs = (
        ("CR", "Collision rate"), ("V_safe", r"$V_{\mathrm{safe}}$"),
        ("clearance", "Successful min. clearance [m]"),
        ("time", "Successful time-to-goal [s]"),
    )

    def value(cell, metric):
        if metric in ("CR", "V_safe"):
            return cell[metric]
        key = "successful_clearance" if metric == "clearance" else "successful_time_to_goal"
        return np.nan if cell[key]["mean"] is None else cell[key]["mean"]

    plt.rcParams.update({
        "font.family": "serif", "mathtext.fontset": "cm",
        "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
        "axes.unicode_minus": False, "axes.formatter.use_mathtext": True,
    })
    figure, axes = plt.subplots(2, 2, figsize=(14.5, 9), constrained_layout=True)
    for axis, (metric, title) in zip(axes.flat, specs):
        for gamma, color in zip(SP.GAMMAS, colors):
            key = str(gamma)
            axis.plot(rounds, [value(canonical[r]["summary"]["per_gamma"][key], metric) for r in rounds],
                      color=color, lw=1.0, ls=":", alpha=.45)
            axis.plot(rounds, [value(selected[r]["summary"]["per_gamma"][key], metric) for r in rounds],
                      color=color, lw=1.5, label=rf"$\gamma={gamma:g}$")
        axis.plot(rounds, [value(canonical[r]["summary"]["pooled"], metric) for r in rounds],
                  color="black", lw=2.0, ls=":", alpha=.55)
        axis.plot(rounds, [value(selected[r]["summary"]["pooled"], metric) for r in rounds],
                  color="black", lw=3.0, label="pooled")
        pooled_cells = [selected[r]["summary"]["pooled"] for r in rounds]
        if metric in ("CR", "V_safe"):
            interval_key = f"{metric}_cluster_bootstrap95"
        else:
            interval_key = "successful_clearance" if metric == "clearance" else "successful_time_to_goal"
        if metric in ("CR", "V_safe"):
            lower = [cell[interval_key][0] for cell in pooled_cells]
            upper = [cell[interval_key][1] for cell in pooled_cells]
        else:
            lower = [cell[interval_key]["interval95"][0] for cell in pooled_cells]
            upper = [cell[interval_key]["interval95"][1] for cell in pooled_cells]
        lower = [np.nan if item is None else float(item) for item in lower]
        upper = [np.nan if item is None else float(item) for item in upper]
        axis.fill_between(rounds, lower, upper, color="black", alpha=.10, lw=0)
        axis.axvline(int(best_round), color="#0072B2", ls="--", lw=1.5)
        axis.set(title=title, xlabel="expansion round")
        axis.grid(alpha=.25)
        if metric in ("CR", "V_safe"):
            axis.set_ylim(-.03, 1.03)
    axes[0, 0].legend(ncol=2, fontsize=8)
    figure.suptitle("Solid: validation-selected per-gamma temperatures; dotted: canonical temperature 1")
    for suffix in ("png", "pdf"):
        figure.savefig(f"{output_stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(figure)


def run(args) -> dict:
    rounds = parse_rounds(args.rounds)
    assert_disjoint_banks(args.tune_ep0, args.tune_M, args.screen_ep0, args.screen_M)
    checkpoint_paths = {
        round_i: os.path.join(args.checkpoint_dir, f"round_{round_i:02d}.pt")
        for round_i in rounds
    }
    missing = [path for path in checkpoint_paths.values() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            f"post-expansion evaluation requires every declared checkpoint; missing {missing[0]}"
        )
    environment = SS.scene_profile(args.scene_profile)
    os.makedirs(args.outdir, exist_ok=True)
    validation_dir = os.path.join(args.outdir, "validation")
    screening_dir = os.path.join(args.outdir, "screening")
    os.makedirs(validation_dir, exist_ok=True)
    os.makedirs(screening_dir, exist_ok=True)

    first_checkpoint = checkpoint_paths[rounds[0]]
    probe, _ = GPS.load_sfm_policy(first_checkpoint, device="cpu")
    tune_noise, tune_meta = noise_bank(
        scene_profile=args.scene_profile, ep0=args.tune_ep0, M=args.tune_M, d=probe.d,
    )
    screen_noise, screen_meta = noise_bank(
        scene_profile=args.scene_profile, ep0=args.screen_ep0, M=args.screen_M, d=probe.d,
    )
    del probe
    schedule, records = [], []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(args.workers), mp_context=context) as executor:
        for round_i in rounds:
            checkpoint = checkpoint_paths[round_i]
            checkpoint_sha = _sha256_file(checkpoint)
            policy, _ = GPS.load_sfm_policy(checkpoint, device=args.device)
            policy.eval()
            validation = {}
            validation_cells = {}
            pending_validation = {}
            for temperature in TEMPERATURES:
                tag = str(temperature).replace(".", "p")
                pending_validation[temperature] = _begin_cell(
                    policy, checkpoint_sha=checkpoint_sha, round_i=round_i,
                    scene_profile=args.scene_profile, ep0=args.tune_ep0, M=args.tune_M,
                    noise=tune_noise, noise_meta=tune_meta, temperature=temperature,
                    role="temperature_validation", device=args.device, executor=executor,
                    path=os.path.join(validation_dir, f"r{round_i:02d}_t{tag}.json"),
                )
            for temperature in TEMPERATURES:
                cell = _finish_cell(pending_validation[temperature])
                validation[float(temperature)] = cell["summary"]
                validation_cells[str(temperature)] = cell["cell_key"]
            chosen, selection_key = select_temperature_vector(validation)
            schedule.append(dict(
                round=round_i, temperature_by_gamma=chosen, selection_key=selection_key,
                validation_cells=validation_cells,
            ))
            pending_canonical = _begin_cell(
                policy, checkpoint_sha=checkpoint_sha, round_i=round_i,
                scene_profile=args.scene_profile, ep0=args.screen_ep0, M=args.screen_M,
                noise=screen_noise, noise_meta=screen_meta, temperature=1.0,
                role="canonical_temp1_screening", device=args.device, executor=executor,
                path=os.path.join(screening_dir, f"r{round_i:02d}_temp1.json"),
            )
            if _is_canonical_temperature(chosen):
                canonical = _finish_cell(pending_canonical)
                selected_cell = canonical
                _write_json(os.path.join(screening_dir, f"r{round_i:02d}_selected.json"), dict(
                    status="SFM_B1_CURVE_SELECTED_ALIAS", round=round_i,
                    selected_temperature_by_gamma=chosen,
                    alias_of=f"r{round_i:02d}_temp1.json",
                    cell_key=canonical["cell_key"],
                ))
            else:
                pending_selected = _begin_cell(
                    policy, checkpoint_sha=checkpoint_sha, round_i=round_i,
                    scene_profile=args.scene_profile, ep0=args.screen_ep0, M=args.screen_M,
                    noise=screen_noise, noise_meta=screen_meta, temperature=chosen,
                    role="validation_selected_temperature_screening", device=args.device,
                    executor=executor,
                    path=os.path.join(screening_dir, f"r{round_i:02d}_selected.json"),
                )
                canonical = _finish_cell(pending_canonical)
                selected_cell = _finish_cell(pending_selected)
            records.extend((
                dict(round=round_i, mode="canonical_temp1",
                     temperature_by_gamma=temperature_vector(1.0),
                     cell_key=canonical["cell_key"], summary=canonical["summary"]),
                dict(round=round_i, mode="validation_selected_temperature",
                     temperature_by_gamma=chosen,
                     cell_key=selected_cell["cell_key"], summary=selected_cell["summary"]),
            ))
            del policy

    schedule_payload = dict(
        status="TEMPERATURE_SCHEDULE_COMPLETE", version=VERSION,
        selection_bank=tune_meta, screening_bank=screen_meta,
        grid=list(TEMPERATURES), shared_across_gammas=False,
        selection_key=("worst CR, pooled CR, -worst V_safe, -pooled V_safe, "
                       "undefined/clearance-order violation, undefined/time-order violation, "
                       "worst/pooled timeout, -worst/pooled SR, distance from temperature 1"),
        entries=schedule,
    )
    schedule_path = os.path.join(args.outdir, "temperature_schedule.json")
    _write_json(schedule_path, schedule_payload)
    metrics_path = os.path.join(args.outdir, "metrics.jsonl")
    with open(metrics_path + ".tmp", "w") as stream:
        for record in records:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    os.replace(metrics_path + ".tmp", metrics_path)
    selected_records = [row for row in records if row["mode"] == "validation_selected_temperature"]
    best = min(selected_records, key=lambda row: temperature_selection_key(
        row["summary"], row["temperature_by_gamma"],
    ))
    _plot(records, os.path.join(args.outdir, "raw_checkpoint_curves"),
          best_round=int(best["round"]))
    artifacts = {}
    for path in sorted(Path(args.outdir).rglob("*")):
        if path.is_file() and path.name != "COMPLETE.json":
            artifacts[str(path.relative_to(args.outdir))] = _sha256_file(path)
    complete = dict(
        status="SFM_B1_CURVE_EVAL_COMPLETE", version=VERSION,
        evaluator_sha256=_sha256_file(__file__),
        checkpoint_dir=os.path.abspath(args.checkpoint_dir), rounds=rounds,
        environment=environment, validation_bank=tune_meta, screening_bank=screen_meta,
        temperature_grid=list(TEMPERATURES), shared_across_gammas=False,
        canonical_control=True, post_expansion_only=True,
        best_screening=dict(
            round=best["round"], temperature_by_gamma=best["temperature_by_gamma"],
            key=list(temperature_selection_key(
                best["summary"], best["temperature_by_gamma"],
            )),
        ),
        no_test_leakage=("the per-gamma temperature vector is selected only on M10 validation; "
                         "M50 screening is evaluated after lock and never changes that vector"),
        interpretation=("solid curves are a validation-tuned deployment policy, not intrinsic "
                        "temperature-1 generator performance; dotted curves are canonical temp=1"),
        no_trajectory_artifacts=True, verifier=SM.verifier_manifest(), artifact_sha256=artifacts,
    )
    _write_json(os.path.join(args.outdir, "COMPLETE.json"), complete)
    return complete


def candidate(args) -> dict:
    """Tune one checkpoint on M10, then screen only the locked vector on M50."""
    assert_disjoint_banks(args.tune_ep0, args.tune_M, args.screen_ep0, args.screen_M)
    os.makedirs(args.outdir, exist_ok=False)
    validation_dir = os.path.join(args.outdir, "validation")
    screening_dir = os.path.join(args.outdir, "screening")
    os.makedirs(validation_dir)
    os.makedirs(screening_dir)

    checkpoint_sha = _sha256_file(args.checkpoint)
    policy, _ = GPS.load_sfm_policy(args.checkpoint, device=args.device)
    policy.eval()
    tune_noise, tune_meta = noise_bank(
        scene_profile=args.scene_profile, ep0=args.tune_ep0, M=args.tune_M, d=policy.d,
    )
    screen_noise, screen_meta = noise_bank(
        scene_profile=args.scene_profile, ep0=args.screen_ep0, M=args.screen_M, d=policy.d,
    )
    validation_summaries = {}
    validation_cells = {}
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(args.workers), mp_context=context) as executor:
        pending_validation = {}
        for temperature in TEMPERATURES:
            tag = str(temperature).replace(".", "p")
            pending_validation[temperature] = _begin_cell(
                policy, checkpoint_sha=checkpoint_sha, round_i=args.round,
                scene_profile=args.scene_profile, ep0=args.tune_ep0, M=args.tune_M,
                noise=tune_noise, noise_meta=tune_meta, temperature=temperature,
                role="candidate_temperature_validation", device=args.device, executor=executor,
                path=os.path.join(validation_dir, f"t{tag}.json"),
            )
        selection_input = {}
        for temperature in TEMPERATURES:
            cell = _finish_cell(pending_validation[temperature])
            selection_input[float(temperature)] = cell["summary"]
            validation_summaries[str(temperature)] = cell["summary"]
            validation_cells[str(temperature)] = cell["cell_key"]
        chosen, selection_key = select_temperature_vector(selection_input)
        screening = _finish_cell(_begin_cell(
            policy, checkpoint_sha=checkpoint_sha, round_i=args.round,
            scene_profile=args.scene_profile, ep0=args.screen_ep0, M=args.screen_M,
            noise=screen_noise, noise_meta=screen_meta, temperature=chosen,
            role="candidate_selected_temperature_screening", device=args.device,
            executor=executor, path=os.path.join(screening_dir, "selected_temperature.json"),
        ))

    record = dict(
        mode="candidate_selected_temperature", round=int(args.round),
        temperature_by_gamma=chosen, summary=screening["summary"],
    )
    metrics_path = os.path.join(args.outdir, "metrics.jsonl")
    with open(metrics_path + ".tmp", "w") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
    os.replace(metrics_path + ".tmp", metrics_path)
    artifacts = {}
    for path in sorted(Path(args.outdir).rglob("*")):
        if path.is_file() and path.name != "COMPLETE.json":
            artifacts[str(path.relative_to(args.outdir))] = _sha256_file(path)
    complete = dict(
        status="SFM_B1_CANDIDATE_SCREEN_COMPLETE", version=VERSION,
        evaluator_sha256=_sha256_file(__file__), round=int(args.round),
        checkpoint=os.path.abspath(args.checkpoint), checkpoint_sha256=checkpoint_sha,
        environment=SS.scene_profile(args.scene_profile),
        temperature_by_gamma=chosen, selection_key=selection_key,
        validation=dict(
            bank=tune_meta, grid=list(TEMPERATURES), summaries=validation_summaries,
            cells=validation_cells,
        ),
        screening=dict(
            mode="candidate_selected_temperature", bank=screen_meta,
            cell_key=screening["cell_key"], summary=screening["summary"],
        ),
        no_test_leakage=("temperature vector selected only on disjoint M10 validation; "
                         "the locked vector alone was evaluated on M50 screening"),
        canonical_screening_run=False, no_trajectory_artifacts=True,
        verifier=SM.verifier_manifest(), artifact_sha256=artifacts,
    )
    _write_json(os.path.join(args.outdir, "COMPLETE.json"), complete)
    return complete


def confirm_single(args) -> dict:
    """Evaluate exactly one supplied temperature vector on the final M100 bank."""
    assert_final_confirmation_bank(args.ep0, args.M)
    selected_temperatures = temperature_vector(
        json.loads(args.temperature_by_gamma)
        if args.temperature_by_gamma is not None else args.temperature
    )
    os.makedirs(args.outdir, exist_ok=False)
    checkpoint_sha = _sha256_file(args.checkpoint)
    policy, _ = GPS.load_sfm_policy(args.checkpoint, device=args.device)
    policy.eval()
    noise, noise_meta = noise_bank(
        scene_profile=args.scene_profile, ep0=args.ep0, M=args.M, d=policy.d,
    )
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(args.workers), mp_context=context) as executor:
        cell = _finish_cell(_begin_cell(
            policy, checkpoint_sha=checkpoint_sha, round_i=args.round,
            scene_profile=args.scene_profile, ep0=args.ep0, M=args.M,
            noise=noise, noise_meta=noise_meta, temperature=selected_temperatures,
            role="final_confirmation_single_temperature", device=args.device,
            executor=executor, path=os.path.join(args.outdir, "single_temperature.json"),
        ))
    record = dict(
        mode="single_temperature_confirmation", round=int(args.round),
        temperature_by_gamma=selected_temperatures, summary=cell["summary"],
    )
    metrics_path = os.path.join(args.outdir, "metrics.jsonl")
    with open(metrics_path + ".tmp", "w") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")
    os.replace(metrics_path + ".tmp", metrics_path)
    artifacts = {
        path.name: _sha256_file(path)
        for path in sorted(Path(args.outdir).iterdir())
        if path.is_file() and path.name != "COMPLETE.json"
    }
    complete = dict(
        status="SFM_B1_SINGLE_CONFIRMATION_COMPLETE", version=VERSION,
        evaluator_sha256=_sha256_file(__file__), round=int(args.round),
        checkpoint=os.path.abspath(args.checkpoint), checkpoint_sha256=checkpoint_sha,
        temperature_by_gamma=selected_temperatures, summary=cell["summary"],
        environment=SS.scene_profile(args.scene_profile), bank=noise_meta,
        selection_frozen_before_confirmation=True, single_vector_only=True,
        verifier=SM.verifier_manifest(), no_trajectory_artifacts=True,
        artifact_sha256=artifacts,
    )
    _write_json(os.path.join(args.outdir, "COMPLETE.json"), complete)
    return complete


def confirm(args) -> dict:
    """One disjoint M100 confirmation after arm/round/temperature-vector selection."""
    if getattr(args, "single_vector_only", False):
        return confirm_single(args)
    assert_final_confirmation_bank(args.ep0, args.M)
    selected_temperatures = temperature_vector(
        json.loads(args.temperature_by_gamma)
        if args.temperature_by_gamma is not None else args.temperature
    )
    os.makedirs(args.outdir, exist_ok=False)
    checkpoint_sha = _sha256_file(args.checkpoint)
    policy, _ = GPS.load_sfm_policy(args.checkpoint, device=args.device)
    policy.eval()
    noise, noise_meta = noise_bank(
        scene_profile=args.scene_profile, ep0=args.ep0, M=args.M, d=policy.d,
    )
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(args.workers), mp_context=context) as executor:
        pending_canonical = _begin_cell(
            policy, checkpoint_sha=checkpoint_sha, round_i=args.round,
            scene_profile=args.scene_profile, ep0=args.ep0, M=args.M,
            noise=noise, noise_meta=noise_meta, temperature=1.0,
            role="final_confirmation_canonical_temp1", device=args.device, executor=executor,
            path=os.path.join(args.outdir, "canonical_temp1.json"),
        )
        if _is_canonical_temperature(selected_temperatures):
            canonical = _finish_cell(pending_canonical)
            selected = canonical
        else:
            pending_selected = _begin_cell(
                policy, checkpoint_sha=checkpoint_sha, round_i=args.round,
                scene_profile=args.scene_profile, ep0=args.ep0, M=args.M,
                noise=noise, noise_meta=noise_meta, temperature=selected_temperatures,
                role="final_confirmation_validation_selected_temperature",
                device=args.device, executor=executor,
                path=os.path.join(args.outdir, "selected_temperature.json"),
            )
            canonical = _finish_cell(pending_canonical)
            selected = _finish_cell(pending_selected)
    result = dict(
        status="SFM_B1_FINAL_CONFIRMATION_COMPLETE", version=VERSION,
        checkpoint=os.path.abspath(args.checkpoint), checkpoint_sha256=checkpoint_sha,
        selected_round=int(args.round), selected_temperature_by_gamma=selected_temperatures,
        environment=SS.scene_profile(args.scene_profile), bank=noise_meta,
        selection_frozen_before_confirmation=True,
        canonical_temp1=canonical["summary"], selected_temperature_result=selected["summary"],
        verifier=SM.verifier_manifest(), no_trajectory_artifacts=True,
    )
    _write_json(os.path.join(args.outdir, "final_confirmation.json"), result)
    artifacts = {
        path.name: _sha256_file(path) for path in sorted(Path(args.outdir).iterdir()) if path.is_file()
    }
    complete = dict(**result, artifact_sha256=artifacts)
    _write_json(os.path.join(args.outdir, "COMPLETE.json"), complete)
    return complete


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("run")
    command.add_argument("--checkpoint-dir", required=True)
    command.add_argument("--scene-profile", required=True, choices=SS.SCIENTIFIC_EVAL_PROFILES)
    command.add_argument("--outdir", required=True)
    command.add_argument("--rounds", default="0:20")
    command.add_argument("--device", default="cuda:0")
    command.add_argument("--workers", type=int, default=32)
    command.add_argument("--tune-ep0", type=int, default=TUNE_EP0)
    command.add_argument("--tune-M", type=int, default=TUNE_M)
    command.add_argument("--screen-ep0", type=int, default=SCREEN_EP0)
    command.add_argument("--screen-M", type=int, default=SCREEN_M)
    candidate_command = sub.add_parser("candidate")
    candidate_command.add_argument("--checkpoint", required=True)
    candidate_command.add_argument("--round", type=int, required=True)
    candidate_command.add_argument(
        "--scene-profile", required=True, choices=SS.SCIENTIFIC_EVAL_PROFILES,
    )
    candidate_command.add_argument("--outdir", required=True)
    candidate_command.add_argument("--device", default="cuda:0")
    candidate_command.add_argument("--workers", type=int, default=32)
    candidate_command.add_argument("--tune-ep0", type=int, default=TUNE_EP0)
    candidate_command.add_argument("--tune-M", type=int, default=TUNE_M)
    candidate_command.add_argument("--screen-ep0", type=int, default=SCREEN_EP0)
    candidate_command.add_argument("--screen-M", type=int, default=SCREEN_M)
    confirmation = sub.add_parser("confirm")
    confirmation.add_argument("--checkpoint", required=True)
    confirmation.add_argument("--round", type=int, required=True)
    temperatures = confirmation.add_mutually_exclusive_group(required=True)
    temperatures.add_argument("--temperature", type=float)
    temperatures.add_argument(
        "--temperature-by-gamma",
        help='JSON object keyed by every declared gamma, e.g. {"0.1":0.95,...}',
    )
    confirmation.add_argument("--scene-profile", required=True, choices=SS.SCIENTIFIC_EVAL_PROFILES)
    confirmation.add_argument("--outdir", required=True)
    confirmation.add_argument("--device", default="cuda:0")
    confirmation.add_argument("--workers", type=int, default=32)
    confirmation.add_argument("--ep0", type=int, default=SP.FINAL_CONFIRM_EP0)
    confirmation.add_argument("--M", type=int, default=100)
    confirmation.add_argument(
        "--single-vector-only", action="store_true",
        help="evaluate only the supplied vector, without a canonical-temperature companion",
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.command == "run":
        run(args)
    elif args.command == "candidate":
        candidate(args)
    elif args.command == "confirm":
        confirm(args)


if __name__ == "__main__":
    main()
