"""Synchronous 56-episode SFM B1 macro-round expansion."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import random
import time

import numpy as np
import torch

import _paths  # noqa: F401
import grid_feats as GF
import grid_policy_sfm as GPS
import sfm_b1_cost as BC
import sfm_b1_eval as BE
import sfm_b1_rbf as BR
import sfm_b1_store as BS
import sfm_hp_history as HH
import sfm_metrics2 as SM
import sfm_protocol as SP
import sfm_scene as SS


@dataclass(frozen=True)
class ArmConfig:
    name: str
    selector: str
    alpha: float
    rounds: int = SP.ROUNDS
    K: int = SP.K
    B: int = SP.B
    T: int = SP.T
    H: int = SP.H
    W: int = SP.W
    batch: int = SP.BATCH
    lr: float = SP.LR
    ess_target: float = SP.ESS_TARGET
    nfe: int = 8
    temp: float = 1.0
    phi_s: float = 0.9
    gp_lam: float = 1.0e-2
    optimizer_steps: int = 1
    inner_epochs: int = 1
    sanity_M: int = 0
    verifier_workers: int = 32
    smoke: bool = False
    seed: int = 20260720
    scene_profile: str = "legacy_velocity_ood"
    acquisition_mode: str = "fixed_b4"
    max_queries: int = SP.B

    def validate(self):
        if (self.T, self.H, self.W, self.batch, self.ess_target) != (
                180, 10, 2, 128, 0.5):
            raise ValueError("scientific B1 knobs differ from the frozen protocol")
        if self.acquisition_mode == "fixed_b4":
            if (self.K, self.B, self.max_queries) != (16, 4, 4):
                raise ValueError("fixed-B control requires K=16 and exactly four queries")
        elif self.acquisition_mode == "adaptive_k64":
            if (self.K, self.B, self.max_queries) != (
                    SP.ADAPTIVE_PROPOSAL_K, SP.ADAPTIVE_QUERY_BATCH,
                    SP.ADAPTIVE_MAX_QUERIES):
                raise ValueError("adaptive acquisition requires K=64 and B=4 up to 64 queries")
        else:
            raise ValueError("unknown acquisition mode")
        if self.selector not in ("margin", "safemppi_cost"):
            raise ValueError("invalid arm selector")
        if not math.isfinite(float(self.alpha)) or float(self.alpha) < 0.0:
            raise ValueError("alpha must be finite and nonnegative")
        if self.scene_profile not in (
                "legacy_velocity_ood", "requested_ood", "density_ood",
                "double_density_velocity_ood"):
            raise ValueError("expansion requires an explicit OOD scene profile")
        if self.name == "A" and (self.selector != "margin" or self.alpha != 0.0
                                 or self.optimizer_steps != 1 or self.inner_epochs != 1):
            raise ValueError("arm A must be margin/alpha=0")
        if self.name in ("B", "C", "D") and self.selector != "safemppi_cost":
            raise ValueError("arms B-D require SafeMPPI cost selection")
        if self.name in ARMS and (
                self.optimizer_steps != 1 or self.inner_epochs != 1
                or self.lr != SP.LR or self.sanity_M != 0
                or self.acquisition_mode != "fixed_b4"):
            raise ValueError("legacy A-D controls use the frozen one-step replay")
        if self.name not in ARMS and self.selector != "margin":
            raise ValueError("alpha/replay sweep arms keep max-step-margin execution fixed")
        if self.name not in ARMS and (
                self.optimizer_steps != SP.OPTIMIZER_CHUNKS
                or self.inner_epochs not in SP.INNER_EPOCHS
                or self.lr != SP.SWEEP_LR or self.sanity_M != SP.SANITY_M):
            raise ValueError(
                "custom sweep requires 16 chunks, inner epochs {1,4,16}, lr=1e-4, and M10 sanity"
            )
        return self


ARMS = {
    "A": dict(selector="margin", alpha=0.0),
    "B": dict(selector="safemppi_cost", alpha=0.0),
    "C": dict(selector="safemppi_cost", alpha=0.001),
    "D": dict(selector="safemppi_cost", alpha=0.01),
}


class Replica:
    def __init__(self, scenario_id, gamma, *, n_ped, ped_speed_range=SS.OOD_PED_SPEED_RANGE):
        self.scenario_id = int(scenario_id)
        self.gamma = float(gamma)
        self.humans = SS.make_humans(scenario_id, 0, n_ped, ped_speed_range)
        self.state = np.zeros(4, np.float32)
        self.states = [self.state.copy()]
        self.controls = []
        self.peds = []
        self.history = HH.HpHistory()
        self.alive = True
        self.status = None
        self.minimum_clearance = float("inf")
        self.prepared = None


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def policy_sha256(policy):
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _prepare(replica, device):
    if replica.prepared is not None:
        return replica.prepared
    ped_xy, ped_vel = SS.collect_humans(replica.humans)
    clearance = float(np.linalg.norm(ped_xy - replica.state[:2][None], axis=1).min() - SS.R_PED)
    replica.minimum_clearance = min(replica.minimum_clearance, clearance)
    if clearance < 0.0:
        replica.alive = False
        replica.status = "collision"
        return None
    if float(np.linalg.norm(replica.state[:2] - SS.GOAL)) < 0.5:
        replica.alive = False
        replica.status = "success"
        return None
    obstacles = np.concatenate([ped_xy, np.full((len(ped_xy), 1), SS.R_PED, np.float32)], axis=1)
    raw_grid = torch.as_tensor(
        GF.axis_grid(replica.state[:2], obstacles, 0.0, R=SS.R_SENSE, sensing=SS.R_SENSE)
    )
    hp10 = replica.history.append(raw_grid)
    low = torch.as_tensor(GF.low5(replica.state, SS.GOAL, replica.gamma))
    hist = torch.as_tensor(GF.hist_pad(
        np.asarray(replica.controls[-16:]) if replica.controls else np.zeros((0, 2)), 16
    ))
    replica.prepared = dict(
        hp10=hp10, low=low, hist=hist, ped_xy=ped_xy.copy(), ped_vel=ped_vel.copy(),
        state=replica.state.copy(),
    )
    return replica.prepared


def _stack_prepared(replicas, device):
    prepared = [_prepare(replica, device) for replica in replicas]
    valid = [(replica, value) for replica, value in zip(replicas, prepared) if value is not None]
    if not valid:
        return [], None
    return [item[0] for item in valid], dict(
        hp10=torch.stack([item[1]["hp10"] for item in valid]).to(device),
        low=torch.stack([item[1]["low"] for item in valid]).to(device),
        hist=torch.stack([item[1]["hist"] for item in valid]).to(device),
    )


@torch.no_grad()
def _features(phi_policy, windows, batch_context, s):
    contexts = phi_policy.ctx_from(batch_context["hp10"], batch_context["low"], batch_context["hist"])
    K = windows.shape[1]
    controls = windows.reshape(-1, windows.shape[-2], 2)
    context = contexts.repeat_interleave(K, dim=0)
    return BR.l2_normalize(phi_policy.phi_s(controls, context, s=float(s))).reshape(len(contexts), K, -1)


def _record_batch(records, device):
    contexts = [shard.contexts[int(query["context_id"])] for shard, query in records]
    hp10 = torch.as_tensor(np.stack([row["hp10"] for row in contexts]), device=device).float()
    low = torch.as_tensor(np.stack([row["low5"] for row in contexts]), device=device).float()
    hist = torch.as_tensor(np.stack([row["hist"] for row in contexts]), device=device).float()
    controls = torch.as_tensor(np.stack([query["controls"] for _, query in records]), device=device).float()
    return hp10, low, hist, controls


@torch.no_grad()
def gp_from_recent(policy, recent, *, ell, cap, lam, phi_s, device, seed):
    del seed  # GP retention is fixed when each round shard is completed.
    if int(cap) != SP.GP_CAP:
        raise ValueError(f"latest B1 GP memory requires cap={SP.GP_CAP}")
    selected = recent.gp_records()
    if len(selected) > int(cap):
        raise AssertionError("retained GP memory exceeds the declared cap")
    gp = BR.RBFGP(ell, lam)
    if selected:
        feature_parts = []
        for start in range(0, len(selected), 256):
            hp10, low, hist, controls = _record_batch(selected[start:start + 256], device)
            feature_parts.append(policy.phi_s(controls, policy.ctx_from(hp10, low, hist), s=phi_s))
        gp.set_buffer(torch.cat(feature_parts))
    return gp, [(shard.round_i, int(query["query_id"])) for shard, query in selected]


def gp_memory_diagnostics(recent):
    records = recent.gp_records()
    per_round, per_gamma = Counter(), Counter()
    identities = []
    for shard, query in records:
        context = shard.contexts[int(query["context_id"])]
        per_round[int(shard.round_i)] += 1
        per_gamma[str(float(context["gamma"]))] += 1
        identities.append((int(shard.round_i), int(query["query_id"])))
    fingerprint = hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()
    gamma_counts = {str(float(gamma)): int(per_gamma[str(float(gamma))]) for gamma in SP.GAMMAS}
    if gamma_counts and max(gamma_counts.values()) - min(gamma_counts.values()) > 1:
        raise AssertionError("two-round GP memory is not gamma-balanced")
    return dict(
        count=len(records), per_round=dict(sorted(per_round.items())),
        per_gamma=gamma_counts, identity_sha256=fingerprint,
    )


def _initial_beta(phi_policy, gp, replicas, cfg, device, seed):
    live, batch = _stack_prepared(replicas, device)
    generator = torch.Generator(device=device).manual_seed(int(seed))
    with torch.no_grad():
        windows = BE.generate_windows(
            phi_policy, batch["hp10"], batch["low"], batch["hist"],
            K=cfg.K, nfe=cfg.nfe, temp=cfg.temp, generator=generator,
        )
        features = _features(phi_policy, windows, batch, cfg.phi_s)
    beta, ess = BR.calibrate_beta(
        gp, [features[index] for index in range(len(live))], B=cfg.B,
        target=cfg.ess_target, seed=seed + 17,
    )
    if abs(float(ess) - float(cfg.ess_target)) > 1.0e-4:
        raise RuntimeError(
            f"adaptive beta missed normalized ESS target: {ess} != {cfg.ess_target}"
        )
    return beta, ess


def _advance(replica, action):
    replica.state = SM.rollout_positions(replica.state, np.asarray(action)[None])[-1]
    # rollout_positions returns positions only; update velocity explicitly.
    previous = replica.states[-1]
    replica.state = np.concatenate([
        replica.state[:2], previous[2:4] + SS.DT * np.asarray(action, np.float32)
    ]).astype(np.float32)
    replica.controls.append(np.asarray(action, np.float32).copy())
    replica.states.append(replica.state.copy())
    replica.peds.append(replica.prepared["ped_xy"].copy())
    replica.prepared = None
    SS.advance_humans(replica.humans, replica.state)


def nvp_fail_closed(replica):
    """Terminate exactly one replica; the macro-round remains live."""
    replica.alive = False
    replica.status = "nvp"


def gather_macro_round(policy, phi_policy, gp, beta, replicas, cfg, shard, device, executor, generator,
                       *, record_all_traces=False):
    """Freeze all acquisition state and gather one complete 8x7 macro-round."""
    timers = Counter()
    sigma_all, sigma_selected, ess_values = [], [], []
    modes = {key: Counter() for key in ("all_K", "selected_B", "Dplus", "executed")}
    traces = []
    frozen_hash = policy_sha256(policy)
    for step in range(cfg.T):
        start = time.perf_counter()
        live = [replica for replica in replicas if replica.alive]
        live, batch = _stack_prepared(live, device)
        timers["sfm_stepping"] += time.perf_counter() - start
        if not live:
            break
        start = time.perf_counter()
        with torch.no_grad():
            windows = BE.generate_windows(
                policy, batch["hp10"], batch["low"], batch["hist"], K=cfg.K,
                nfe=cfg.nfe, temp=cfg.temp, generator=generator,
            )
        timers["flow_proposal"] += time.perf_counter() - start
        start = time.perf_counter()
        with torch.no_grad():
            features = _features(phi_policy, windows, batch, cfg.phi_s)
        base_sigma = gp.acquisition_sigma_batched(features)
        selected_by_context, acquisition_traces = gp.sequential_acquire_batched(
            features, cfg.B, beta, generator=generator
        )
        sigma_all.extend(map(float, base_sigma.detach().cpu().reshape(-1)))
        for acquisition in acquisition_traces:
            sigma_selected.extend(float(row["chosen_sigma"]) for row in acquisition)
            ess_values.extend(row["ess_norm"] for row in acquisition)
        timers["phi_rbf"] += time.perf_counter() - start
        tasks = []
        for context_index, replica in enumerate(live):
            prepared = replica.prepared
            for candidate_id in selected_by_context[context_index]:
                tasks.append((
                    context_index, candidate_id, prepared["state"],
                    windows[context_index, candidate_id].detach().cpu().numpy(),
                    prepared["ped_xy"], prepared["ped_vel"], replica.gamma,
                ))
        start = time.perf_counter()
        results = list(executor.map(SM.verify_in_worker, tasks))
        timers["verifier"] += time.perf_counter() - start
        by_context = defaultdict(list)
        for context_index, candidate_id, result in results:
            by_context[context_index].append((candidate_id, result))
        start = time.perf_counter()
        for context_index, replica in enumerate(live):
            prepared = replica.prepared
            context_id = shard.add_context(
                scenario_id=replica.scenario_id, gamma=replica.gamma, step=step,
                state=prepared["state"], hp10=prepared["hp10"].numpy(),
                low5=prepared["low"].numpy(), hist=prepared["hist"].numpy(),
                ped_xy=prepared["ped_xy"], ped_vel=prepared["ped_vel"],
            )
            ped_prediction = SM.predict_pedestrians(prepared["ped_xy"], prepared["ped_vel"], cfg.H)
            all_rows = []
            for candidate_id in range(cfg.K):
                controls = windows[context_index, candidate_id].detach().cpu().numpy()
                segment = SM.rollout_positions(prepared["state"], controls)
                mode = BE.classify_candidate(segment, ped_prediction)
                modes["all_K"][mode] += 1
                all_rows.append(dict(candidate_id=candidate_id, controls=controls, segment=segment, mode=mode))
            query_rows = []
            lookup = {candidate: result for candidate, result in by_context[context_index]}
            for acquisition_step, candidate_id in enumerate(selected_by_context[context_index]):
                controls = windows[context_index, candidate_id].detach().cpu().numpy()
                result = lookup[candidate_id]
                mode = all_rows[candidate_id]["mode"]
                modes["selected_B"][mode] += 1
                if not result.get("resolved"):
                    shard.add_error(
                        context_key=(replica.scenario_id, replica.gamma, step),
                        candidate_id=candidate_id, error=result.get("error"),
                    )
                    continue
                pending_sigma = float(acquisition_traces[context_index][acquisition_step]["chosen_sigma"])
                query_id = shard.add_resolved_query(
                    context_id, candidate_id, controls,
                    sigma=pending_sigma,
                    result=result, acquisition_step=acquisition_step, mode=mode,
                    gp_base_sigma=float(base_sigma[context_index, candidate_id]),
                )
                row = dict(candidate_id=candidate_id, query_id=query_id, controls=controls,
                           result=result, mode=mode)
                query_rows.append(row)
                if result["y"] == 1 and result["full_h"]:
                    modes["Dplus"][mode] += 1
            chosen = BC.select_admissible(
                query_rows, selector=cfg.selector, state=prepared["state"],
                ped_xy=prepared["ped_xy"], ped_vel=prepared["ped_vel"], gamma=replica.gamma,
            )
            for row in query_rows:
                stored = shard.queries[row["query_id"]]
                if "hp_margin" in row:
                    stored["hp_margin"] = float(row["hp_margin"])
                if "step_progress" in row:
                    stored["step_progress"] = float(row["step_progress"])
                if "expert_cost" in row:
                    stored["expert_cost"] = float(row["expert_cost"])
            trace = dict(
                round=shard.round_i, step=step, scenario_id=replica.scenario_id,
                gamma=replica.gamma, state=prepared["state"], ped_xy=prepared["ped_xy"],
                ped_vel=prepared["ped_vel"], all_K=all_rows,
                selected_ids=list(selected_by_context[context_index]), query_rows=query_rows,
                acquisition=acquisition_traces[context_index], executed_id=None,
            )
            if chosen is None:
                nvp_fail_closed(replica)
            else:
                shard.mark_executed(
                    chosen["query_id"], hp_margin=chosen["hp_margin"],
                    expert_cost=chosen.get("expert_cost"),
                )
                modes["executed"][chosen["mode"]] += 1
                trace["executed_id"] = int(chosen["candidate_id"])
                _advance(replica, chosen["controls"][0])
            if record_all_traces or len(traces) < 64 or chosen is None:
                traces.append(trace)
        timers["sfm_stepping"] += time.perf_counter() - start
    for replica in replicas:
        if replica.alive:
            terminal_xy, _ = SS.collect_humans(replica.humans)
            terminal_clearance = float(
                np.linalg.norm(terminal_xy - replica.state[:2][None], axis=1).min() - SS.R_PED
            )
            replica.minimum_clearance = min(replica.minimum_clearance, terminal_clearance)
            if terminal_clearance < 0.0:
                replica.status = "collision"
            elif float(np.linalg.norm(replica.state[:2] - SS.GOAL)) < 0.5:
                replica.status = "success"
            else:
                replica.status = "timeout"
            replica.alive = False
    if policy_sha256(policy) != frozen_hash:
        raise RuntimeError("policy changed during frozen macro-round")
    return dict(
        timers=dict(timers), sigma=BR.acquisition_diagnostics(sigma_all, sigma_selected),
        beta=float(beta), realized_ess_over_K=float(np.mean(ess_values)),
        modes={key: dict(value) for key, value in modes.items()}, traces=traces,
        outcomes=[dict(
            scenario_id=replica.scenario_id, gamma=replica.gamma, status=replica.status,
            success=replica.status == "success", collision=replica.status == "collision",
            nvp=replica.status == "nvp", steps=len(replica.controls),
            min_clearance=replica.minimum_clearance,
        ) for replica in replicas],
    )


def adaptive_query_slices(order, *, batch, max_queries):
    """Return the declared without-replacement query batches for one context."""
    values = list(map(int, order))
    if len(values) != len(set(values)):
        raise ValueError("adaptive acquisition order contains duplicates")
    if int(batch) < 1 or int(max_queries) < int(batch):
        raise ValueError("invalid adaptive query budget")
    if int(max_queries) > len(values) or int(max_queries) % int(batch):
        raise ValueError("adaptive query budget must be a batched prefix of the proposal order")
    return [
        values[start:start + int(batch)]
        for start in range(0, int(max_queries), int(batch))
    ]


def _adaptive_rescue_bucket(query_count):
    if int(query_count) <= 4:
        return "initial_1_4"
    if int(query_count) <= 16:
        return "rescued_5_16"
    if int(query_count) <= 32:
        return "rescued_17_32"
    return "rescued_33_64"


def gather_macro_round_adaptive(policy, phi_policy, gp, beta, replicas, cfg, shard, device,
                                executor, generator, *, record_all_traces=False):
    """Gather with K=64 and verifier batches of four until one is admissible.

    This path is intentionally separate from :func:`gather_macro_round` so the
    frozen K=16/B=4 control is not refactored or numerically perturbed.
    """
    if cfg.acquisition_mode != "adaptive_k64":
        raise ValueError("adaptive gather requires acquisition_mode=adaptive_k64")
    timers = Counter()
    sigma_all, sigma_selected, base_sigma_selected, ess_over_remaining = [], [], [], []
    modes = {key: Counter() for key in ("all_K", "queried", "Dplus", "executed")}
    rescue = Counter()
    realized_queries = []
    traces = []
    frozen_hash = policy_sha256(policy)
    for step in range(cfg.T):
        start = time.perf_counter()
        live = [replica for replica in replicas if replica.alive]
        live, batch = _stack_prepared(live, device)
        timers["sfm_stepping"] += time.perf_counter() - start
        if not live:
            break
        start = time.perf_counter()
        with torch.no_grad():
            windows = BE.generate_windows(
                policy, batch["hp10"], batch["low"], batch["hist"], K=cfg.K,
                nfe=cfg.nfe, temp=cfg.temp, generator=generator,
            )
        timers["flow_proposal"] += time.perf_counter() - start
        start = time.perf_counter()
        with torch.no_grad():
            features = _features(phi_policy, windows, batch, cfg.phi_s)
        base_sigma = gp.acquisition_sigma_batched(features)
        acquisition_orders, acquisition_traces = gp.sequential_acquire_batched(
            features, cfg.max_queries, beta, generator=generator,
        )
        sigma_all.extend(map(float, base_sigma.detach().cpu().reshape(-1)))
        timers["phi_rbf"] += time.perf_counter() - start

        context_ids, all_rows_by_context = [], []
        query_rows_by_context = [[] for _ in live]
        queried_ids = [[] for _ in live]
        chosen_by_context = [None for _ in live]
        verifier_failed_by_context = [False for _ in live]
        for context_index, replica in enumerate(live):
            prepared = replica.prepared
            context_ids.append(shard.add_context(
                scenario_id=replica.scenario_id, gamma=replica.gamma, step=step,
                state=prepared["state"], hp10=prepared["hp10"].numpy(),
                low5=prepared["low"].numpy(), hist=prepared["hist"].numpy(),
                ped_xy=prepared["ped_xy"], ped_vel=prepared["ped_vel"],
            ))
            ped_prediction = SM.predict_pedestrians(
                prepared["ped_xy"], prepared["ped_vel"], cfg.H,
            )
            all_rows = []
            for candidate_id in range(cfg.K):
                controls = windows[context_index, candidate_id].detach().cpu().numpy()
                segment = SM.rollout_positions(prepared["state"], controls)
                mode = BE.classify_candidate(segment, ped_prediction)
                modes["all_K"][mode] += 1
                all_rows.append(dict(
                    candidate_id=candidate_id, controls=controls, segment=segment, mode=mode,
                ))
            all_rows_by_context.append(all_rows)

        batch_slices = [
            adaptive_query_slices(order, batch=cfg.B, max_queries=cfg.max_queries)
            for order in acquisition_orders
        ]
        for batch_index in range(cfg.max_queries // cfg.B):
            unresolved = [
                index for index, chosen in enumerate(chosen_by_context)
                if chosen is None and not verifier_failed_by_context[index]
            ]
            if not unresolved:
                break
            tasks = []
            for context_index in unresolved:
                replica = live[context_index]
                prepared = replica.prepared
                for candidate_id in batch_slices[context_index][batch_index]:
                    tasks.append((
                        context_index, candidate_id, prepared["state"],
                        windows[context_index, candidate_id].detach().cpu().numpy(),
                        prepared["ped_xy"], prepared["ped_vel"], replica.gamma,
                    ))
                    queried_ids[context_index].append(int(candidate_id))
                    acquisition_step = len(queried_ids[context_index]) - 1
                    trace_row = acquisition_traces[context_index][acquisition_step]
                    sigma_selected.append(float(trace_row["chosen_sigma"]))
                    base_sigma_selected.append(
                        float(base_sigma[context_index, candidate_id])
                    )
                    ess_over_remaining.append(dict(
                        rank=acquisition_step + 1,
                        value=float(trace_row["ess_norm"]),
                    ))
            start = time.perf_counter()
            results = list(executor.map(SM.verify_in_worker, tasks))
            timers["verifier"] += time.perf_counter() - start
            by_context = defaultdict(list)
            for context_index, candidate_id, result in results:
                by_context[context_index].append((candidate_id, result))
            start = time.perf_counter()
            for context_index in unresolved:
                replica = live[context_index]
                prepared = replica.prepared
                lookup = {candidate: result for candidate, result in by_context[context_index]}
                first_acquisition_step = batch_index * cfg.B
                for local_offset, candidate_id in enumerate(batch_slices[context_index][batch_index]):
                    controls = windows[context_index, candidate_id].detach().cpu().numpy()
                    result = lookup[candidate_id]
                    mode = all_rows_by_context[context_index][candidate_id]["mode"]
                    modes["queried"][mode] += 1
                    if not result.get("resolved"):
                        verifier_failed_by_context[context_index] = True
                        shard.add_error(
                            context_key=(replica.scenario_id, replica.gamma, step),
                            candidate_id=candidate_id, error=result.get("error"),
                        )
                        continue
                    acquisition_step = first_acquisition_step + local_offset
                    pending_sigma = float(
                        acquisition_traces[context_index][acquisition_step]["chosen_sigma"]
                    )
                    query_id = shard.add_resolved_query(
                        context_ids[context_index], candidate_id, controls,
                        sigma=pending_sigma, result=result, acquisition_step=acquisition_step,
                        mode=mode, gp_base_sigma=float(base_sigma[context_index, candidate_id]),
                    )
                    row = dict(
                        candidate_id=candidate_id, query_id=query_id, controls=controls,
                        result=result, mode=mode,
                    )
                    query_rows_by_context[context_index].append(row)
                    if result["y"] == 1 and result["full_h"]:
                        modes["Dplus"][mode] += 1
                chosen_by_context[context_index] = (
                    None if verifier_failed_by_context[context_index]
                    else BC.select_admissible(
                        query_rows_by_context[context_index], selector=cfg.selector,
                        state=prepared["state"], ped_xy=prepared["ped_xy"],
                        ped_vel=prepared["ped_vel"], gamma=replica.gamma,
                    )
                )
                for row in query_rows_by_context[context_index]:
                    stored = shard.queries[row["query_id"]]
                    if "hp_margin" in row:
                        stored["hp_margin"] = float(row["hp_margin"])
                    if "step_progress" in row:
                        stored["step_progress"] = float(row["step_progress"])
                    if "expert_cost" in row:
                        stored["expert_cost"] = float(row["expert_cost"])
            timers["sfm_stepping"] += time.perf_counter() - start

        for context_index, replica in enumerate(live):
            prepared = replica.prepared
            chosen = chosen_by_context[context_index]
            query_count = len(queried_ids[context_index])
            realized_queries.append(query_count)
            trace = dict(
                round=shard.round_i, step=step, scenario_id=replica.scenario_id,
                gamma=replica.gamma, state=prepared["state"], ped_xy=prepared["ped_xy"],
                ped_vel=prepared["ped_vel"], all_K=all_rows_by_context[context_index],
                selected_ids=list(queried_ids[context_index]),
                query_rows=query_rows_by_context[context_index],
                acquisition=acquisition_traces[context_index][:query_count], executed_id=None,
                query_budget=dict(K=cfg.K, batch=cfg.B, max_queries=cfg.max_queries,
                                  realized=query_count),
            )
            if chosen is None:
                resolved = query_rows_by_context[context_index]
                verifier_errors = query_count - len(resolved)
                if verifier_errors:
                    rescue["verifier_error_fail_closed"] += 1
                    replica.alive = False
                    replica.status = "verifier_error"
                elif any(int(row["result"]["y"]) == 1 for row in resolved):
                    rescue["hp_gate_failure"] += 1
                else:
                    rescue["finite_K_no_y"] += 1
                if not verifier_errors:
                    if query_count != cfg.max_queries:
                        raise RuntimeError(
                            "adaptive NVP occurred before exhausting the query budget"
                        )
                    nvp_fail_closed(replica)
            else:
                rescue[_adaptive_rescue_bucket(query_count)] += 1
                shard.mark_executed(
                    chosen["query_id"], hp_margin=chosen["hp_margin"],
                    expert_cost=chosen.get("expert_cost"),
                )
                modes["executed"][chosen["mode"]] += 1
                trace["executed_id"] = int(chosen["candidate_id"])
                _advance(replica, chosen["controls"][0])
            if record_all_traces or len(traces) < 64 or chosen is None:
                traces.append(trace)

    for replica in replicas:
        if replica.alive:
            terminal_xy, _ = SS.collect_humans(replica.humans)
            terminal_clearance = float(
                np.linalg.norm(terminal_xy - replica.state[:2][None], axis=1).min() - SS.R_PED
            )
            replica.minimum_clearance = min(replica.minimum_clearance, terminal_clearance)
            if terminal_clearance < 0.0:
                replica.status = "collision"
            elif float(np.linalg.norm(replica.state[:2] - SS.GOAL)) < 0.5:
                replica.status = "success"
            else:
                replica.status = "timeout"
            replica.alive = False
    if policy_sha256(policy) != frozen_hash:
        raise RuntimeError("policy changed during frozen macro-round")
    query_array = np.asarray(realized_queries, dtype=float)
    query_summary = ({"count": 0, "mean": None, "q50": None, "q90": None, "max": None}
                     if not len(query_array) else dict(
                         count=len(query_array), mean=float(query_array.mean()),
                         q50=float(np.quantile(query_array, .5)),
                         q90=float(np.quantile(query_array, .9)),
                         max=int(query_array.max()),
                     ))
    pending_diagnostic = BR.acquisition_diagnostics(sigma_all, sigma_selected)
    base_diagnostic = BR.acquisition_diagnostics(sigma_all, base_sigma_selected)
    sigma_diagnostic = base_diagnostic
    sigma_diagnostic.update(
        queried_base_sigma=base_diagnostic["selected_B_sigma"],
        base_uplift=base_diagnostic["uplift"],
        pending_conditioned_selected_sigma=pending_diagnostic["selected_B_sigma"],
        pending_conditioned_uplift=pending_diagnostic["uplift"],
        comparison_note=(
            "base_uplift compares selected and pool candidates under the same "
            "pre-acquisition posterior; pending_conditioned_uplift is retained "
            "only for continuity with the completed fixed-B diagnostics"
        ),
    )
    ess_values = np.asarray(
        [row["value"] for row in ess_over_remaining], dtype=float,
    )
    ess_by_rank = {}
    for label, lower, upper in (
            ("1_4", 1, 4), ("5_16", 5, 16),
            ("17_32", 17, 32), ("33_64", 33, 64)):
        values = [
            row["value"] for row in ess_over_remaining
            if lower <= row["rank"] <= upper
        ]
        ess_by_rank[label] = None if not values else float(np.mean(values))
    return dict(
        timers=dict(timers), sigma=sigma_diagnostic,
        beta=float(beta), realized_ess_over_remaining=(
            None if not len(ess_values) else float(np.mean(ess_values))
        ),
        modes={key: dict(value) for key, value in modes.items()}, traces=traces,
        adaptive_acquisition=dict(
            mode=cfg.acquisition_mode, K=cfg.K, query_batch=cfg.B,
            max_queries=cfg.max_queries, outcomes=dict(rescue),
            realized_queries=query_summary,
            ess_normalization="remaining candidate pool at each sequential rank",
            ess_by_rank=ess_by_rank,
            beta_calibration_scope="initial sequential ranks 1-4",
        ),
        outcomes=[dict(
            scenario_id=replica.scenario_id, gamma=replica.gamma, status=replica.status,
            success=replica.status == "success", collision=replica.status == "collision",
            nvp=replica.status == "nvp", steps=len(replica.controls),
            verifier_error=replica.status == "verifier_error",
            min_clearance=replica.minimum_clearance,
        ) for replica in replicas],
    )


def _save_checkpoint(policy, path, extra):
    temporary = path + ".tmp"
    GPS.save_sfm_policy(policy, temporary, extra=extra)
    os.replace(temporary, path)
    complete = path + ".COMPLETE.json"
    with open(complete + ".tmp", "w") as stream:
        json.dump(dict(status="COMPLETE", path=os.path.abspath(path), sha256=_sha256_file(path)), stream, indent=2)
    os.replace(complete + ".tmp", complete)


def _run_sanity(policy, cfg, *, round_i, device, executor, noise, noise_meta, outdir):
    """Canonical temperature-one M10 monitor; never used as training data."""
    import sfm_b1_curve_eval as CE

    policy.eval()
    rollout_start = time.perf_counter()
    rows = CE.run_batched_raw(
        policy, scene_profile=cfg.scene_profile, ep0=SP.SMOKE_EVAL_EP0,
        M=cfg.sanity_M, base_noise=noise, temperature=1.0, device=device,
    )
    rollout_seconds = time.perf_counter() - rollout_start
    certificate_start = time.perf_counter()
    rows = CE._attach_validity(rows, [
        future.result() for future in CE._submit_v_safe(rows, executor)
    ])
    certificate_seconds = time.perf_counter() - certificate_start
    summary = CE.summarize(rows, seed=cfg.seed + int(round_i) * 31)
    CE.assert_zero_verifier_errors(summary, role=f"round-{int(round_i)} sanity")
    payload = dict(
        status="SFM_B1_ROUND_SANITY_COMPLETE", round=int(round_i),
        role="development_monitor_only", temperature_by_gamma={
            str(gamma): 1.0 for gamma in SP.GAMMAS
        },
        bank=noise_meta, summary=summary,
        timers=dict(
            gpu_policy_rollout=rollout_seconds,
            full_H10_certificate_wait=certificate_seconds,
            total=rollout_seconds + certificate_seconds,
        ),
        no_training_feedback=True, no_trajectory_artifacts=True,
    )
    path = os.path.join(outdir, f"sanity_r{int(round_i):02d}.json")
    temporary = path + ".tmp"
    with open(temporary, "w") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
    os.replace(temporary, path)
    return payload


def run_arm(checkpoint, outdir, cfg, *, ell, cap, device):
    cfg.validate()
    if cfg.name not in ARMS and int(cap) != SP.GP_CAP:
        raise ValueError(f"custom B1 sweep requires GP cap={SP.GP_CAP}")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    environment = SS.scene_profile(cfg.scene_profile)
    os.makedirs(outdir, exist_ok=True)
    policy, source_checkpoint = GPS.load_sfm_policy(checkpoint, device=device)
    frozen_parameters = BS.configure_expansion_trainability(policy)
    encoder_before = BS.module_sha256(policy.enc_grid)
    source_sha = _sha256_file(checkpoint)
    optimizer = torch.optim.Adam(
        [parameter for parameter in policy.parameters() if parameter.requires_grad], lr=cfg.lr
    )
    sanity_noise = sanity_meta = None
    if cfg.sanity_M:
        import sfm_b1_curve_eval as CE
        sanity_noise, sanity_meta = CE.noise_bank(
            scene_profile=cfg.scene_profile, ep0=SP.SMOKE_EVAL_EP0,
            M=cfg.sanity_M, d=policy.d,
        )
    recent = BS.RecentRounds(os.path.join(outdir, "round_shards"), cfg.W)
    history = []
    torch_generator = torch.Generator(device=device).manual_seed(cfg.seed)
    _save_checkpoint(policy, os.path.join(outdir, "round_00.pt"), dict(
        round=0, arm=cfg.name, source_checkpoint=os.path.abspath(checkpoint),
        source_sha256=source_sha, encoder_sha256=encoder_before,
        training_randomness="per-query SHA256-keyed CFM x0/tau/residual-dropout design",
    ))
    baseline_sanity = None
    with ProcessPoolExecutor(max_workers=cfg.verifier_workers) as executor:
        if cfg.sanity_M:
            baseline_sanity = _run_sanity(
                policy, cfg, round_i=0, device=device, executor=executor,
                noise=sanity_noise, noise_meta=sanity_meta, outdir=outdir,
            )
        for round_i in range(1, cfg.rounds + 1):
            round_start = time.perf_counter()
            scenarios = SP.expansion_scenarios(round_i, smoke=cfg.smoke)
            if len(set(scenarios)) != 8:
                raise RuntimeError("macro-round scenarios are not distinct")
            replicas = [
                Replica(
                    scenario, gamma, n_ped=environment["n_ped"],
                    ped_speed_range=tuple(environment["ped_speed_range"]),
                )
                for scenario in scenarios for gamma in SP.GAMMAS
            ]
            if len(replicas) != 56:
                raise RuntimeError("B1 macro-round must contain exactly 56 independent episodes")
            policy.eval()
            phi_policy = copy.deepcopy(policy).eval()
            if getattr(phi_policy, "gru", None) is not None:
                # deepcopy breaks cuDNN's packed GRU storage; repack without
                # changing any parameter or numerical policy output.
                phi_policy.gru.flatten_parameters()
            for parameter in phi_policy.parameters():
                parameter.requires_grad_(False)
            gp, gp_ids = gp_from_recent(
                phi_policy, recent, ell=ell, cap=cap, lam=cfg.gp_lam,
                phi_s=cfg.phi_s, device=device, seed=cfg.seed + round_i * 101,
            )
            gp_memory = gp_memory_diagnostics(recent)
            beta, calibrated_ess = _initial_beta(
                phi_policy, gp, replicas, cfg, device, cfg.seed + round_i * 1009
            )
            shard = BS.RoundShard(round_i)
            gather_fn = (gather_macro_round_adaptive
                         if cfg.acquisition_mode == "adaptive_k64"
                         else gather_macro_round)
            gather = gather_fn(
                policy, phi_policy, gp, beta, replicas, cfg, shard, device, executor,
                torch_generator,
            )
            retention = BS.retain_gp_upper_quartile(
                shard, SP.GAMMAS, seed=cfg.seed + round_i * 7919,
                total=SP.GP_RETAIN_PER_ROUND,
            )
            shard_manifest = recent.append_and_save(shard)
            replay_start = time.perf_counter()
            replay_kwargs = dict(
                alpha=cfg.alpha, batch=cfg.batch, device=device, seed=cfg.seed + round_i,
            )
            if cfg.name in ARMS:
                replay_kwargs["optimizer_steps"] = cfg.optimizer_steps
            else:
                replay_kwargs.update(
                    optimizer_chunks=cfg.optimizer_steps, inner_epochs=cfg.inner_epochs,
                )
            replay = BS.signed_update(policy, optimizer, recent, **replay_kwargs)
            gather["timers"]["replay"] = time.perf_counter() - replay_start
            encoder_after_round = BS.module_sha256(policy.enc_grid)
            if encoder_after_round != encoder_before:
                raise RuntimeError("visual encoder SHA changed during expansion")
            checkpoint_path = os.path.join(outdir, f"round_{round_i:02d}.pt")
            _save_checkpoint(policy, checkpoint_path, dict(
                round=round_i, arm=cfg.name, source_checkpoint=os.path.abspath(checkpoint),
                source_sha256=source_sha, encoder_sha256=encoder_after_round,
                recipe=asdict(cfg), ell=float(ell), cap=int(cap), beta=float(beta),
                training_randomness="per-query SHA256-keyed CFM x0/tau/residual-dropout design",
            ))
            sanity = None
            if cfg.sanity_M:
                sanity = _run_sanity(
                    policy, cfg, round_i=round_i, device=device, executor=executor,
                    noise=sanity_noise, noise_meta=sanity_meta, outdir=outdir,
                )
                gather["timers"]["sanity"] = sanity["timers"]["total"]
            record = dict(
                round=round_i, scenarios=list(scenarios), beta=float(beta),
                verifier=SM.verifier_manifest(),
                calibrated_ess_over_K=float(calibrated_ess), gp_buffer_ids=gp_ids,
                gp=gp.diagnostics(), gp_memory=gp_memory, gp_retention=retention,
                gather={key: value for key, value in gather.items() if key != "traces"},
                replay=replay, sanity=sanity, shard=shard_manifest, encoder_sha256=encoder_after_round,
                checkpoint=os.path.abspath(checkpoint_path), checkpoint_sha256=_sha256_file(checkpoint_path),
                wall_seconds=time.perf_counter() - round_start,
            )
            history.append(record)
            with open(os.path.join(outdir, "metrics.jsonl"), "a") as stream:
                stream.write(json.dumps(record) + "\n")
            torch.save(gather["traces"], os.path.join(outdir, f"query_trace_r{round_i:02d}.pt"))
            print(json.dumps({key: record[key] for key in ("round", "beta", "wall_seconds")}), flush=True)
            if os.path.exists(os.path.join(outdir, "STOP_AFTER_CURRENT_ROUND")):
                raise RuntimeError(
                    f"development monitor requested stop after completed round {round_i}"
                )
    encoder_after = BS.module_sha256(policy.enc_grid)
    if encoder_after != encoder_before:
        raise RuntimeError("arm-level visual encoder SHA mismatch")
    manifest = dict(
        status="ARM_COMPLETE", arm=cfg.name, source_checkpoint=os.path.abspath(checkpoint),
        source_sha256=source_sha, recipe=asdict(cfg), ell=float(ell), cap=int(cap),
        verifier=SM.verifier_manifest(),
        environment=environment, frozen_parameters=frozen_parameters, encoder_sha_before=encoder_before,
        encoder_sha_after=encoder_after, rounds=len(history), history=history,
        baseline_sanity=baseline_sanity,
        training_randomness=dict(
            global_seed=int(cfg.seed),
            design="SHA256(seed,sign,round,query_id) -> CFM x0/tau/residual-dropout mask",
            invariant_to_batching_and_optimizer_step_partition=True,
            deterministic_algorithms=True,
        ),
    )
    with open(os.path.join(outdir, "method_manifest.json"), "w") as stream:
        json.dump(manifest, stream, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", required=True)
    arm_group = parser.add_mutually_exclusive_group(required=True)
    arm_group.add_argument("--arm", choices=tuple(ARMS))
    arm_group.add_argument("--custom-name")
    parser.add_argument("--selector", choices=("margin", "safemppi_cost"))
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--optimizer-steps", type=int)
    parser.add_argument("--inner-epochs", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--sanity-M", type=int)
    parser.add_argument(
        "--adaptive-k64", action="store_true",
        help=("Generate 64 learned proposals and query the exact verifier four at a time "
              "until one batch is admissible or all 64 are exhausted."),
    )
    parser.add_argument("--ell", type=float, required=True)
    parser.add_argument("--cap", type=int, choices=(256, 512), required=True)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--verifier-workers", type=int, default=32)
    parser.add_argument(
        "--scene-profile", required=True,
        choices=("legacy_velocity_ood", "requested_ood", "density_ood",
                 "double_density_velocity_ood"),
        help=("Explicit expansion environment: legacy reproduces 103476d, "
              "requested_ood shifts density and speed, density_ood uses n_ped=50 at training speeds, "
              "and double_density_velocity_ood uses n_ped=40 at 1.0--2.0 m/s."),
    )
    args = parser.parse_args()
    if args.arm is not None:
        if any(value is not None for value in (
                args.selector, args.alpha, args.optimizer_steps, args.inner_epochs,
                args.lr, args.sanity_M)) or args.adaptive_k64:
            parser.error("legacy --arm A-D cannot be combined with custom sweep knobs")
        arm = ARMS[args.arm]
        name = args.arm
        optimizer_steps, inner_epochs, lr, sanity_M = 1, 1, SP.LR, 0
    else:
        if args.selector is None or args.alpha is None:
            parser.error("--custom-name requires --selector and --alpha")
        arm = dict(selector=args.selector, alpha=args.alpha)
        name = args.custom_name
        optimizer_steps = SP.OPTIMIZER_CHUNKS if args.optimizer_steps is None else args.optimizer_steps
        inner_epochs = 1 if args.inner_epochs is None else args.inner_epochs
        lr = SP.SWEEP_LR if args.lr is None else args.lr
        sanity_M = SP.SANITY_M if args.sanity_M is None else args.sanity_M
    cfg = ArmConfig(
        name=name, selector=arm["selector"], alpha=arm["alpha"],
        rounds=args.rounds, smoke=args.smoke, seed=args.seed,
        verifier_workers=args.verifier_workers, scene_profile=args.scene_profile,
        optimizer_steps=optimizer_steps, inner_epochs=inner_epochs,
        lr=lr, sanity_M=sanity_M,
        K=(SP.ADAPTIVE_PROPOSAL_K if args.adaptive_k64 else SP.K),
        B=SP.B,
        acquisition_mode=("adaptive_k64" if args.adaptive_k64 else "fixed_b4"),
        max_queries=(SP.ADAPTIVE_MAX_QUERIES if args.adaptive_k64 else SP.B),
    )
    run_arm(args.checkpoint, args.outdir, cfg, ell=args.ell, cap=args.cap, device=args.device)


if __name__ == "__main__":
    main()
