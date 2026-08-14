"""Task-agnostic, single-arm Safe Flow Expansion reference.

The task adapter owns dynamics, context construction, nominal ``H_P``, the full
verifier, and the execution cost.  This file owns only the B1 expansion loop:
RBF uncertainty, budgeted acquisition, fail-closed execution, recent positive
replay, and optional signed negative gradients.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
import pickle
import time
from typing import Any, Callable, Protocol, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class ExpansionConfig:
    rounds: int = 10
    gammas: tuple[float, ...] = (0.1, 0.3, 0.5, 1.0)
    parallel_episodes: int = 2
    verifier_workers: int = 1
    max_retry_batches: int = 4
    max_steps: int = 100
    max_steps_status: str = "TIMEOUT"
    K: int = 16
    B: int = 4
    batch_size: int = 32
    inner_steps: int | None = None  # None = one exact pass over every eligible D+ row.
    microbatch_repeats: int = 1
    learning_rate: float = 3.0e-5
    first_layer_lr_scale: float = 1.0
    freeze_visual_encoder: bool = False
    head_only_update: bool = False
    optimizer_scope: str = "configured_trainable"
    replay_rounds: int = 2
    gp_buffer_cap: int = 256
    gp_noise: float = 1.0e-2
    rbf_lengthscale: float | None = None
    beta: float = 0.05
    adaptive_beta: bool = False
    ess_target: float = 0.5
    negative_alpha: float = 0.0
    replay_top_fraction: float = 1.0
    replay_selector: str = "sigma_top"
    replay_context_quota: int = 4
    replay_action_weight: float = 0.5
    replay_cluster_count: int = 32
    flow_base_std: float = 1.0
    candidate_perturb_std: float = 0.0
    candidate_perturb_scope: str = "coherent_horizon"
    execution_rule: str = "min_cost"
    execution_step_margin_weight: float = 0.0
    execution_ess_target: float = 0.25
    execution_uncertainty_weight: float = 1.0
    archive_rule: str = "all_queries"
    successful_trajectory_selector: str = "lowest_episode_id"
    successful_trajectories_per_gamma: int = 1
    replay_acceptance: str = "execution_eligible"
    paired_noised_representation: bool = False
    seed: int = 0
    acquisition_feature: str = "learned_phi"
    coverage_replay: str = "none"
    replay_augmentation: str = "none"
    target_gate_start_round: int | None = None
    gp_reference_mode: str = "recent_current_phi"
    gp_reference_rounds: int | None = None
    gp_sliding_row_selector: str = "trajectory_uniform"
    gp_exact_max_rows_per_gamma: int = 1024

    def validate(self) -> None:
        if (self.rounds < 1 or self.parallel_episodes < 1
                or self.verifier_workers < 1
                or self.max_retry_batches < 1 or self.max_steps < 1):
            raise ValueError(
                "rounds, parallel_episodes, verifier_workers, "
                "max_retry_batches, and max_steps must be positive"
            )
        if self.max_steps_status not in {"TIMEOUT", "EARLY_CUTOFF"}:
            raise ValueError(
                "max_steps_status must be TIMEOUT or EARLY_CUTOFF"
            )
        if not (1 <= self.B <= self.K):
            raise ValueError("require 1 <= B <= K")
        if (self.batch_size < 1 or self.microbatch_repeats < 1
                or (self.inner_steps is not None and self.inner_steps < 1)):
            raise ValueError("batch_size and an explicit inner_steps must be positive")
        if self.replay_rounds < 1 or self.gp_buffer_cap < 2:
            raise ValueError("replay_rounds must be positive and gp_buffer_cap at least two")
        if self.gp_noise <= 0.0 or self.beta <= 0.0 or self.learning_rate <= 0.0:
            raise ValueError("gp_noise, beta, and learning_rate must be positive")
        if not 0.0 < self.first_layer_lr_scale <= 1.0:
            raise ValueError("first_layer_lr_scale must lie in (0,1]")
        if self.optimizer_scope not in {
            "configured_trainable", "head_only", "last_block_and_head",
        }:
            raise ValueError(
                "optimizer_scope must be configured_trainable, head_only, or "
                "last_block_and_head"
            )
        if self.head_only_update and self.optimizer_scope != "configured_trainable":
            raise ValueError(
                "legacy head_only_update cannot be combined with optimizer_scope"
            )
        if (
            (self.head_only_update or self.optimizer_scope != "configured_trainable")
            and self.first_layer_lr_scale != 1.0
        ):
            raise ValueError(
                "first_layer_lr_scale applies only to configured_trainable scope"
            )
        if not 0.0 < self.ess_target <= 1.0 or self.negative_alpha < 0.0:
            raise ValueError("ess_target must lie in (0,1] and negative_alpha be nonnegative")
        if self.adaptive_beta and self.ess_target < 1.0 / self.K:
            raise ValueError(
                "adaptive ESS target is infeasible: normalized ESS is at "
                f"least 1/K={1.0 / self.K:g} for K={self.K}"
            )
        if not 0.0 < self.replay_top_fraction <= 1.0:
            raise ValueError("replay_top_fraction must lie in (0,1]")
        if self.replay_selector not in {
                "sigma_top", "uniform", "context_kcenter", "cluster_balanced"}:
            raise ValueError(
                "replay_selector must be sigma_top, uniform, context_kcenter, "
                "or cluster_balanced"
            )
        if (self.replay_context_quota < 1 or self.replay_cluster_count < 1
                or self.replay_action_weight < 0.0):
            raise ValueError(
                "replay quotas/counts must be positive and replay_action_weight nonnegative"
            )
        if not math.isfinite(self.flow_base_std) or self.flow_base_std < 0.0:
            raise ValueError("flow_base_std must be finite and nonnegative")
        if self.candidate_perturb_std < 0.0:
            raise ValueError("candidate_perturb_std must be nonnegative")
        if self.candidate_perturb_scope not in {"first_action", "coherent_horizon"}:
            raise ValueError("candidate_perturb_scope must be first_action or coherent_horizon")
        if self.execution_rule not in {
                "min_cost", "max_margin", "uniform_positive", "softmin_cost",
                "max_uncertainty", "uncertainty_cost", "max_step_margin"}:
            raise ValueError(
                "execution_rule must be min_cost, max_margin, uniform_positive, "
                "softmin_cost, max_uncertainty, uncertainty_cost, or "
                "max_step_margin"
            )
        if (
            not math.isfinite(self.execution_step_margin_weight)
            or self.execution_step_margin_weight < 0.0
        ):
            raise ValueError(
                "execution_step_margin_weight must be finite and nonnegative"
            )
        if (
            self.execution_step_margin_weight > 0.0
            and self.execution_rule != "min_cost"
        ):
            raise ValueError(
                "execution_step_margin_weight requires execution_rule=min_cost"
            )
        if not 0.0 < self.execution_ess_target <= 1.0:
            raise ValueError("execution_ess_target must lie in (0,1]")
        if self.execution_uncertainty_weight < 0.0:
            raise ValueError("execution_uncertainty_weight must be nonnegative")
        if self.archive_rule not in {
                "all_queries", "executed_only", "executed_plus_nvp_negative",
                "successful_executed_windows", "preterminal_resolved_queries"}:
            raise ValueError("archive_rule must be all_queries, executed_only, or "
                             "executed_plus_nvp_negative, or "
                             "successful_executed_windows, or "
                             "preterminal_resolved_queries")
        if self.archive_rule == "successful_executed_windows":
            if self.negative_alpha != 0.0:
                raise ValueError(
                    "successful_executed_windows requires negative_alpha=0"
                )
            if self.target_gate_start_round is not None:
                raise ValueError(
                    "successful_executed_windows requires an inactive target gate"
                )
        if self.successful_trajectory_selector not in {
                "lowest_episode_id", "random_success",
                "max_above_fraction", "max_mean_z"}:
            raise ValueError(
                "successful_trajectory_selector must be lowest_episode_id, "
                "random_success, max_above_fraction, or max_mean_z"
            )
        if self.archive_rule == "successful_executed_windows":
            if self.successful_trajectories_per_gamma < 1:
                raise ValueError(
                    "successful_trajectories_per_gamma must be positive for "
                    "successful_executed_windows"
                )
            if (
                self.successful_trajectories_per_gamma
                > self.parallel_episodes * self.max_retry_batches
            ):
                raise ValueError(
                    "successful_trajectories_per_gamma cannot exceed "
                    "parallel_episodes * max_retry_batches"
                )
        elif self.successful_trajectories_per_gamma not in {0, 1}:
            raise ValueError(
                "successful_trajectories_per_gamma is inactive outside "
                "successful_executed_windows and must be 0 (explicitly "
                "disabled) or the legacy default 1"
            )
        if self.replay_acceptance not in {"execution_eligible", "safety_valid"}:
            raise ValueError(
                "replay_acceptance must be execution_eligible or safety_valid"
            )
        if (self.replay_acceptance == "safety_valid"
                and self.archive_rule not in {
                    "all_queries", "preterminal_resolved_queries",
                }):
            raise ValueError(
                "safety_valid replay_acceptance requires an archive containing "
                "every successfully evaluated selected-B query"
            )
        if (
            self.archive_rule == "preterminal_resolved_queries"
            and self.replay_acceptance != "safety_valid"
        ):
            raise ValueError(
                "preterminal_resolved_queries requires replay_acceptance="
                "safety_valid so every exact-positive selected-B query enters D+"
            )
        if self.acquisition_feature not in {"learned_phi", "task_angular"}:
            raise ValueError("acquisition_feature must be learned_phi or task_angular")
        if self.coverage_replay not in {"none", "circular_voronoi"}:
            raise ValueError("coverage_replay must be none or circular_voronoi")
        if self.replay_augmentation not in {"none", "task_d4"}:
            raise ValueError("replay_augmentation must be none or task_d4")
        if (self.target_gate_start_round is not None
                and self.target_gate_start_round < 1):
            raise ValueError("target_gate_start_round must be positive")
        if self.gp_reference_mode not in {
                "recent_current_phi", "round1_fixed_frozen_phi",
                "cumulative_accepted_frozen_phi",
                "cumulative_success_per_gamma_frozen_phi_exact",
                "sliding_success_per_gamma_frozen_phi",
                "sliding_success_per_gamma_current_phi",
                "sliding_positive_per_gamma_frozen_phi",
                "sliding_positive_per_gamma_current_phi",
                "sliding_executed_positive_per_gamma_frozen_phi"}:
            raise ValueError(
                "gp_reference_mode must be recent_current_phi or "
                "round1_fixed_frozen_phi or cumulative_accepted_frozen_phi or "
                "cumulative_success_per_gamma_frozen_phi_exact or "
                "sliding_success_per_gamma_frozen_phi or "
                "sliding_success_per_gamma_current_phi or "
                "sliding_positive_per_gamma_frozen_phi or "
                "sliding_positive_per_gamma_current_phi or "
                "sliding_executed_positive_per_gamma_frozen_phi"
            )
        if self.gp_reference_rounds is not None:
            if self.gp_reference_rounds < 1:
                raise ValueError("gp_reference_rounds must be positive")
            if not self.gp_reference_mode.startswith("sliding_"):
                raise ValueError(
                    "gp_reference_rounds applies only to sliding GP modes"
                )
        if self.gp_reference_mode == "cumulative_accepted_frozen_phi":
            gamma_count = len(self.gammas)
            if (self.acquisition_feature != "learned_phi"
                    or self.gp_buffer_cap % (2 * gamma_count) != 0
                    or self.gp_buffer_cap < 4 * gamma_count):
                raise ValueError(
                    "cumulative_accepted_frozen_phi requires learned_phi and a "
                    "gp_buffer_cap divisible by twice the gamma count, with at "
                    "least four rows per gamma"
                )
        if self.gp_reference_mode in {
            "cumulative_success_per_gamma_frozen_phi_exact",
            "sliding_success_per_gamma_frozen_phi",
            "sliding_success_per_gamma_current_phi",
        }:
            if (
                self.archive_rule != "successful_executed_windows"
                or self.acquisition_feature != "learned_phi"
            ):
                raise ValueError(
                    "successful per-gamma learned-phi GP modes require "
                    "successful_executed_windows and learned_phi"
                )
        if self.gp_reference_mode in {
            "sliding_positive_per_gamma_frozen_phi",
            "sliding_positive_per_gamma_current_phi",
        }:
            if (
                self.archive_rule != "preterminal_resolved_queries"
                or self.acquisition_feature != "learned_phi"
            ):
                raise ValueError(
                    "preterminal positive per-gamma learned-phi GP modes require "
                    "preterminal_resolved_queries and learned_phi"
                )
        if self.gp_reference_mode == (
            "sliding_executed_positive_per_gamma_frozen_phi"
        ):
            if (
                self.archive_rule != "executed_plus_nvp_negative"
                or self.acquisition_feature != "learned_phi"
            ):
                raise ValueError(
                    "sliding executed-positive learned-phi GP mode requires "
                    "executed_plus_nvp_negative and learned_phi"
                )
        if self.gp_reference_mode in {
            "sliding_success_per_gamma_frozen_phi",
            "sliding_success_per_gamma_current_phi",
            "sliding_positive_per_gamma_frozen_phi",
            "sliding_positive_per_gamma_current_phi",
            "sliding_executed_positive_per_gamma_frozen_phi",
        }:
            gamma_count = len(self.gammas)
            if (
                self.gp_buffer_cap % gamma_count != 0
                or self.gp_buffer_cap // gamma_count < 2
            ):
                raise ValueError(
                    "sliding successful-window GP modes require a total "
                    "gp_buffer_cap divisible by the gamma count and at least "
                    "two rows per gamma"
                )
        if self.gp_sliding_row_selector not in {
            "trajectory_uniform", "fifo_tail",
        }:
            raise ValueError(
                "gp_sliding_row_selector must be trajectory_uniform or fifo_tail"
            )
        if self.gp_exact_max_rows_per_gamma < 1:
            raise ValueError("gp_exact_max_rows_per_gamma must be positive")


@dataclass(frozen=True)
class Verification:
    """One successfully evaluated full-verifier result.

    ``valid`` is the full-H label stored in D+. ``hp_eligible`` is retained as
    a nominal one-step diagnostic but is not an execution gate.
    ``progress_eligible`` is a separate performance gate. In the ball task it
    requires positive x progress at every plan knot, without using the
    unexecuted tail's distance to the goal.
    ``execution_cost`` ranks candidates satisfying ``valid`` and
    ``progress_eligible``.
    """

    valid: bool
    hp_eligible: bool
    margin: float
    execution_cost: float
    progress: float = 0.0
    progress_eligible: bool = True
    error: bool = False
    target_eligible: bool = True
    step_margin: float | None = None


@dataclass
class QueryRecord:
    round: int
    gamma: float
    episode: int
    context_id: int
    context: torch.Tensor
    candidate: torch.Tensor
    verification: Verification
    acquisition_sigma: float = 0.0
    flow_base: torch.Tensor | None = None
    nvp_context: bool = False
    acquisition_feature: torch.Tensor | None = None
    replay_eligible: bool = True
    retry_batch: int | None = None
    replica: int | None = None
    trajectory_id: str | None = None
    window_id: str | None = None
    window_start: int | None = None
    valid_horizon: int | None = None
    loss_mask: torch.Tensor | None = None
    executed: bool = False


def _successful_trajectory_id(round_i: int, gamma: float, episode: int) -> str:
    return f"r{round_i:04d}:g{gamma:.9g}:e{episode:06d}"


def _successful_window_id(trajectory_id: str, start: int) -> str:
    return f"{trajectory_id}:w{start:06d}"


def _preterminal_query_id(
    trajectory_id: str,
    start: int,
    local_query: int,
) -> str:
    return f"{trajectory_id}:w{start:06d}:q{local_query:02d}"


def _counter_seed(master_seed: int, *coordinates: int | str) -> int:
    """Stable order-independent seed for success-commit gathering CRNs."""
    digest = hashlib.sha256()
    digest.update(str(int(master_seed)).encode())
    for coordinate in coordinates:
        digest.update(b":")
        digest.update(str(coordinate).encode())
    return int.from_bytes(digest.digest()[:8], "big") % (2**63 - 1)


class ExpansionPolicy(Protocol):
    def parameters(self): ...
    def sample(self, context: torch.Tensor, count: int,
               generator: torch.Generator,
               base_std: float = 1.0) -> torch.Tensor: ...
    def sample_with_base(self, context: torch.Tensor, count: int,
                         generator: torch.Generator,
                         base_std: float = 1.0,
                         ) -> tuple[torch.Tensor, torch.Tensor]: ...
    def embed(self, context: torch.Tensor, candidates: torch.Tensor,
              flow_time: float = 0.9,
              base: torch.Tensor | None = None) -> torch.Tensor: ...
    def cfm_loss(self, contexts: torch.Tensor, candidates: torch.Tensor,
                 reduction: str = "none",
                 loss_mask: torch.Tensor | None = None) -> torch.Tensor: ...
    def state_dict(self): ...


class ExpansionTask(Protocol):
    def reset(self, gamma: float, episode: int, seed: int) -> Any: ...
    def context(self, state: Any, gamma: float) -> torch.Tensor: ...
    def verify(self, context: torch.Tensor, candidates: torch.Tensor,
               gamma: float) -> Sequence[Verification]: ...
    def advance(self, state: Any, candidate: torch.Tensor) -> Any: ...
    def terminal(self, state: Any) -> str | None: ...
    def successful_trajectory_above_fraction(
        self, executed_states: Sequence[Any]
    ) -> float: ...
    def successful_trajectory_mean_z(
        self, executed_states: Sequence[Any]
    ) -> float: ...


_VERIFY_WORKER_TASK: ExpansionTask | None = None


def _initialize_verify_worker(task: ExpansionTask) -> None:
    """Install one immutable task per persistent CPU verifier process."""
    global _VERIFY_WORKER_TASK
    _VERIFY_WORKER_TASK = task
    torch.set_num_threads(1)


def _verify_worker(
    payload: tuple[np.ndarray, np.ndarray, float],
) -> tuple[Verification, ...]:
    """Evaluate one active context in a worker without policy/GP state."""
    if _VERIFY_WORKER_TASK is None:
        raise RuntimeError("verifier worker task was not initialized")
    context, candidates, gamma = payload
    return tuple(_VERIFY_WORKER_TASK.verify(
        torch.from_numpy(context),
        torch.from_numpy(candidates),
        float(gamma),
    ))


class _OrderedVerifier:
    """Serial or spawn-process verifier with deterministic input ordering."""

    def __init__(self, task: ExpansionTask, workers: int):
        self.task = task
        self.workers = int(workers)
        self.pool: ProcessPoolExecutor | None = None
        if self.workers > 1:
            try:
                pickle.dumps(task)
            except Exception as error:
                raise TypeError(
                    "verifier_workers>1 requires a pickleable, CPU-safe task"
                ) from error

    def __enter__(self) -> "_OrderedVerifier":
        return self

    def __exit__(self, *_exc_info) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None

    def verify_many(
        self,
        blocks: Sequence[tuple[torch.Tensor, torch.Tensor, float]],
    ) -> list[tuple[Verification, ...]]:
        if self.workers == 1:
            return [
                tuple(self.task.verify(context, candidates, gamma))
                for context, candidates, gamma in blocks
            ]
        if self.pool is None:
            self.pool = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=mp.get_context("spawn"),
                initializer=_initialize_verify_worker,
                initargs=(self.task,),
            )
        payloads = [
            (
                np.ascontiguousarray(context.detach().cpu().numpy()),
                np.ascontiguousarray(candidates.detach().cpu().numpy()),
                float(gamma),
            )
            for context, candidates, gamma in blocks
        ]
        return list(self.pool.map(_verify_worker, payloads, chunksize=1))


def _normalize(values: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    return values / values.norm(dim=-1, keepdim=True).clamp_min(eps)


def _stable_accumulation_dtype(values: torch.Tensor) -> torch.dtype:
    """Use float64 where supported and float32 on Apple MPS."""
    return torch.float32 if values.device.type == "mps" else torch.float64


def mean_pairwise_lengthscale(features: torch.Tensor) -> float:
    # Calibration uses only 50 rows; keeping this reduction on CPU preserves
    # float64 numerics and avoids the unsupported MPS pdist operator.
    values = _normalize(features.detach().cpu().to(torch.float64))
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("RBF calibration needs at least two embedding rows")
    lengthscale = float(torch.pdist(values).mean())
    if not math.isfinite(lengthscale) or lengthscale <= 0.0:
        raise ValueError("pretrained embeddings do not define a positive length scale")
    return lengthscale


class RBFPosterior:
    """Exact RBF posterior variance on a finite previous-round support set."""

    def __init__(self, lengthscale: float, noise: float):
        if lengthscale <= 0.0 or noise <= 0.0:
            raise ValueError("RBF lengthscale and observation noise must be positive")
        self.lengthscale, self.noise = float(lengthscale), float(noise)
        self.X: torch.Tensor | None = None
        self.L: torch.Tensor | None = None

    @staticmethod
    def _sqdist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return ((a * a).sum(1, keepdim=True) + (b * b).sum(1)[None]
                - 2.0 * a @ b.T).clamp_min(0.0)

    def kernel(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.exp(-self._sqdist(a, b) / (2.0 * self.lengthscale ** 2))

    @torch.no_grad()
    def set_buffer(self, features: torch.Tensor | None) -> None:
        if features is None or len(features) == 0:
            self.X = self.L = None
            return
        stored = (
            features.detach().cpu()
            if features.device.type == "mps" else features.detach()
        )
        self.X = _normalize(stored)
        factor_dtype = torch.float64
        kernel = self.kernel(self.X, self.X).to(factor_dtype)
        eye = torch.eye(
            len(kernel), dtype=factor_dtype, device=kernel.device,
        )
        jitter = self.noise
        for _ in range(6):
            factor, info = torch.linalg.cholesky_ex(kernel + jitter * eye)
            if int(info.max()) == 0:
                self.L = factor
                return
            jitter *= 10.0
        raise RuntimeError("RBF posterior Cholesky failed")

    @torch.no_grad()
    def covariance(self, features: torch.Tensor) -> torch.Tensor:
        query = _normalize(
            features.detach().cpu()
            if features.device.type == "mps" else features.detach()
        )
        covariance = self.kernel(query, query)
        if self.X is not None:
            cross = self.kernel(query, self.X)
            solved = torch.cholesky_solve(cross.T.to(self.L.dtype), self.L)
            covariance -= cross @ solved.to(cross.dtype)
        covariance = 0.5 * (covariance + covariance.T)
        covariance += self.noise * torch.eye(
            len(query), dtype=query.dtype, device=query.device)
        return covariance

    @torch.no_grad()
    def sigma(self, features: torch.Tensor) -> torch.Tensor:
        diagonal = torch.diagonal(self.covariance(features)) - self.noise
        return diagonal.clamp_min(0.0).sqrt()

    @torch.no_grad()
    def acquire_from_covariance(
        self,
        covariance: torch.Tensor,
        B: int,
        beta: float,
        generator: torch.Generator,
        *,
        sampling_device: torch.device,
    ) -> tuple[list[int], list[float], list[float]]:
        """Sequentially acquire from a precomputed posterior covariance."""
        covariance = covariance.clone()
        remaining = torch.arange(len(covariance), device=covariance.device)
        selected, selected_sigma, ess = [], [], []
        for _ in range(B):
            # The Gibbs tilt is defined on posterior standard deviation, not variance.
            # Subtract the observation-noise diagonal added by ``covariance`` so the
            # first draw agrees with ``sigma(features)`` and beta calibration.
            scores = (
                torch.diagonal(covariance) - self.noise
            ).clamp_min(0.0).sqrt()
            weights = torch.exp(((scores - scores.max()) / beta).clamp(-30.0, 30.0))
            probability = weights / weights.sum()
            if sampling_device.type == "mps":
                uniform = float(torch.rand(
                    (), device=sampling_device, generator=generator,
                ).cpu())
                local = min(
                    int(torch.searchsorted(
                        probability.cumsum(0),
                        torch.tensor(uniform, dtype=probability.dtype),
                    )),
                    len(probability) - 1,
                )
            else:
                local = int(torch.multinomial(
                    probability, 1, generator=generator,
                ))
            selected.append(int(remaining[local]))
            selected_sigma.append(float(scores[local]))
            ess.append(float(1.0 / (probability.to(
                _stable_accumulation_dtype(probability)
            ).square().sum()
                                    * probability.numel())))
            keep = torch.ones(len(remaining), dtype=torch.bool, device=remaining.device)
            keep[local] = False
            if not bool(keep.any()):
                break
            cross = covariance[keep, local]
            denominator = covariance[local, local].clamp_min(1.0e-12)
            covariance = covariance[keep][:, keep] - torch.outer(cross, cross) / denominator
            covariance = 0.5 * (covariance + covariance.T)
            remaining = remaining[keep]
        return selected, selected_sigma, ess

    @torch.no_grad()
    def acquire(self, features: torch.Tensor, B: int, beta: float,
        generator: torch.Generator) -> tuple[list[int], list[float], list[float]]:
        return self.acquire_from_covariance(
            self.covariance(features), B, beta, generator,
            sampling_device=features.device,
        )


def normalized_ess(scores: torch.Tensor, beta: float) -> float:
    weights = torch.exp(((scores - scores.max()) / beta).clamp(-30.0, 30.0))
    probability = weights / weights.sum()
    return float(1.0 / (probability.to(
        _stable_accumulation_dtype(probability)
    ).square().sum()
                        * probability.numel()))


def calibrate_fixed_beta(score_pools: Sequence[torch.Tensor], target: float,
                         lower: float = 1.0e-5, upper: float = 10.0) -> float:
    """Choose beta once from representative pools; expansion keeps it fixed."""
    if not score_pools or not 0.0 < target <= 1.0:
        raise ValueError("provide score pools and an ESS target in (0,1]")
    for _ in range(60):
        middle = math.sqrt(lower * upper)
        value = float(np.median([normalized_ess(pool, middle) for pool in score_pools]))
        if value < target:
            lower = middle
        else:
            upper = middle
    return math.sqrt(lower * upper)


def _recent(records: Sequence[QueryRecord], round_i: int, width: int,
            *, positive: bool | None = None) -> list[QueryRecord]:
    first = max(0, round_i - width + 1)
    output = [row for row in records if first <= row.round <= round_i]
    if positive is not None:
        output = [row for row in output if row.verification.valid is positive]
    return output


def _top_uncertainty_by_round(
    records: Sequence[QueryRecord],
    fraction: float,
    *,
    group_by_gamma: bool = False,
) -> list[QueryRecord]:
    """Keep causal sigma tails per round, optionally preserving every gamma."""
    if fraction >= 1.0:
        return list(records)
    groups: dict[tuple[int, float | None], list[QueryRecord]] = {}
    for row in records:
        key = (row.round, row.gamma if group_by_gamma else None)
        groups.setdefault(key, []).append(row)
    kept = []
    for key in sorted(groups):
        rows = sorted(
            groups[key], key=lambda row: row.acquisition_sigma, reverse=True
        )
        count = max(1, int(math.ceil(fraction * len(rows))))
        threshold = rows[count - 1].acquisition_sigma
        # A top-fraction ranking is undefined when uncertainties tie.  Include the
        # complete boundary tie rather than turning stable archive order into a
        # hidden gamma/context/acquisition-slot filter.
        kept.extend(row for row in rows if row.acquisition_sigma >= threshold)
    return kept


def _balanced_cap(records: Sequence[QueryRecord], cap: int,
                  rng: np.random.Generator) -> list[QueryRecord]:
    """Round/gamma-balanced sampling without replacement for the GP only."""
    cells: dict[tuple[int, float], list[QueryRecord]] = {}
    for row in records:
        cells.setdefault((row.round, row.gamma), []).append(row)
    chosen: list[QueryRecord] = []
    while len(chosen) < cap and any(cells.values()):
        for key in sorted(cells):
            if len(chosen) == cap:
                break
            cell = cells[key]
            if cell:
                chosen.append(cell.pop(int(rng.integers(len(cell)))))
    return chosen


def _equal_mass_weights(records: Sequence[QueryRecord]) -> torch.Tensor:
    """Equal mass over gamma -> (round, episode) -> context -> positive query."""
    gammas = sorted({row.gamma for row in records})
    lineages: dict[float, set[tuple[int, int]]] = {}
    contexts: dict[tuple[float, int, int], set[int]] = {}
    queries: dict[tuple[float, int, int, int], int] = {}
    for row in records:
        lineage = (row.round, row.episode)
        lineages.setdefault(row.gamma, set()).add(lineage)
        contexts.setdefault((row.gamma, *lineage), set()).add(row.context_id)
        query = (row.gamma, *lineage, row.context_id)
        queries[query] = queries.get(query, 0) + 1
    weights = []
    for row in records:
        lineage = (row.gamma, row.round, row.episode)
        query = (*lineage, row.context_id)
        weights.append(1.0 / (
            len(gammas) * len(lineages[row.gamma]) * len(contexts[lineage]) * queries[query]
        ))
    values = torch.tensor(weights, dtype=torch.float32)
    return values / values.mean()


def _circular_voronoi_weights(
    records: Sequence[QueryRecord],
    lineage_weights: torch.Tensor,
) -> torch.Tensor:
    """Multiply lineage mass by conditional periodic angular Voronoi mass.

    The task supplies a two-dimensional direction descriptor.  Exact duplicate
    directions share one Voronoi cell, so archive multiplicity cannot create
    extra angular mass.  Density correction is performed independently for each
    conditional query context; pooling contexts would let a common direction at
    one state erase uncertainty at another state.  Per-context rescaling keeps
    the original gamma/lineage/context mass unchanged.
    """
    if len(records) != len(lineage_weights):
        raise ValueError("records and lineage_weights must have the same length")
    multipliers = torch.zeros(len(records), dtype=lineage_weights.dtype)
    context_groups: dict[tuple[float, int, int, int], list[int]] = {}
    for index, row in enumerate(records):
        key = (row.gamma, row.round, row.episode, row.context_id)
        context_groups.setdefault(key, []).append(index)
    for key in sorted(context_groups):
        indices = context_groups[key]
        descriptors = []
        directed_indices = []
        for index in indices:
            value = getattr(records[index], "acquisition_feature", None)
            if value is None:
                raise ValueError(
                    "circular_voronoi replay requires a stored 2-D task "
                    "acquisition feature for every replay row"
                )
            descriptor = torch.as_tensor(value, dtype=torch.float64).reshape(-1)
            if descriptor.numel() != 2 or not bool(torch.isfinite(descriptor).all()):
                raise ValueError(
                    "circular_voronoi replay requires finite 2-D descriptors"
                )
            if float(descriptor.norm()) > 1.0e-12:
                directed_indices.append(index)
                descriptors.append(descriptor / descriptor.norm())
            else:
                # An exactly axial plan has no defined angle. Keep it at neutral
                # lineage mass instead of assigning an arbitrary route direction.
                multipliers[index] = 1.0
        if not descriptors:
            continue
        directions = torch.stack(descriptors)
        angles = torch.remainder(
            torch.atan2(directions[:, 1], directions[:, 0]), 2.0 * math.pi
        ).cpu().numpy()
        unique_angles, inverse, counts = np.unique(
            angles, return_inverse=True, return_counts=True
        )
        if len(unique_angles) == 1:
            cell_mass = np.asarray([2.0 * math.pi], dtype=np.float64)
        else:
            right_gaps = np.remainder(
                np.roll(unique_angles, -1) - unique_angles, 2.0 * math.pi
            )
            cell_mass = 0.5 * (np.roll(right_gaps, 1) + right_gaps)
        per_row_mass = cell_mass[inverse] / counts[inverse]
        # Unit mean within gamma makes this a density correction rather than a
        # hidden learning-rate change.
        per_row_mass *= len(directed_indices) / per_row_mass.sum()
        multipliers[directed_indices] = torch.as_tensor(
            per_row_mass, dtype=multipliers.dtype
        )

    combined = lineage_weights * multipliers
    for key in sorted(context_groups):
        indices = context_groups[key]
        original_mass = lineage_weights[indices].sum()
        combined_mass = combined[indices].sum()
        if float(combined_mass) <= 0.0:
            raise ValueError("circular Voronoi replay produced zero context mass")
        combined[indices] *= original_mass / combined_mass
    return combined


def _stack(records: Sequence[QueryRecord]) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.stack([row.context for row in records]),
            torch.stack([row.candidate for row in records]))


def _stack_loss_masks(records: Sequence[QueryRecord]) -> torch.Tensor | None:
    masks = [getattr(row, "loss_mask", None) for row in records]
    if all(mask is None for mask in masks):
        return None
    values = []
    for row, mask in zip(records, masks):
        if mask is None:
            values.append(torch.ones_like(row.candidate))
        else:
            if tuple(mask.shape) != tuple(row.candidate.shape):
                raise ValueError("replay loss mask must match the stored candidate shape")
            values.append(mask)
    return torch.stack(values)


@torch.no_grad()
def _embed_records(
    policy: ExpansionPolicy,
    records: Sequence[QueryRecord],
    device: torch.device,
) -> torch.Tensor:
    """Re-embed archive rows with their originally paired CFM base noise.

    ``getattr`` keeps archives serialized before ``flow_base`` was introduced
    readable.  Mixed legacy/paired rows are embedded in two batches and then
    restored to archive order.
    """
    if not records:
        raise ValueError("cannot embed an empty record sequence")
    contexts, candidates = (value.to(device) for value in _stack(records))
    bases = [getattr(row, "flow_base", None) for row in records]
    if all(base is None for base in bases):
        return policy.embed(contexts, candidates)
    if all(base is not None for base in bases):
        return policy.embed(
            contexts, candidates,
            base=torch.stack([base for base in bases if base is not None]).to(device),
        )
    output: list[torch.Tensor | None] = [None] * len(records)
    for has_base in (False, True):
        indices = [
            index for index, base in enumerate(bases)
            if (base is not None) is has_base
        ]
        if not indices:
            continue
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=device)
        block_contexts = contexts.index_select(0, index_tensor)
        block_candidates = candidates.index_select(0, index_tensor)
        if has_base:
            block = policy.embed(
                block_contexts,
                block_candidates,
                base=torch.stack([
                    bases[index] for index in indices
                    if bases[index] is not None
                ]).to(device),
            )
        else:
            block = policy.embed(block_contexts, block_candidates)
        for local, index in enumerate(indices):
            output[index] = block[local]
    return torch.stack([value for value in output if value is not None])


@torch.no_grad()
def _task_angular_features(
    task: ExpansionTask,
    context: torch.Tensor,
    candidates: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Validate and normalize a task-supplied two-dimensional direction."""
    feature_fn = getattr(task, "acquisition_features", None)
    if callable(feature_fn):
        raw_features = feature_fn(context, candidates, gamma)
    else:
        feature_fn = getattr(task, "angular_descriptors", None)
        if callable(feature_fn):
            raw_features = feature_fn(context, candidates)
        else:
            raw_features = None
    if raw_features is None:
        raise TypeError(
            "task_angular acquisition or circular_voronoi replay requires "
            "task.acquisition_features(context, candidates, gamma) or "
            "task.angular_descriptors(context, candidates)"
        )
    features = torch.as_tensor(
        raw_features,
        dtype=candidates.dtype,
        device=candidates.device,
    ).detach()
    if features.shape != (len(candidates), 2):
        raise ValueError(
            "task.acquisition_features must return one 2-D row per candidate"
        )
    if not bool(torch.isfinite(features).all()):
        raise ValueError("task acquisition features must be finite")
    return _normalize(features)


def _stack_acquisition_features(
    records: Sequence[QueryRecord],
    device: torch.device,
) -> torch.Tensor:
    values = []
    for row in records:
        value = getattr(row, "acquisition_feature", None)
        if value is None:
            raise ValueError(
                "task_angular GP refit requires a stored 2-D acquisition "
                "feature for every positive buffer row"
            )
        values.append(torch.as_tensor(value))
    features = torch.stack(values).to(device)
    if features.shape != (len(records), 2):
        raise ValueError("stored task acquisition features must be two-dimensional")
    return features


def _record_identity_hash(records: Sequence[QueryRecord]) -> str:
    """Stable provenance hash for a frozen GP reference row set."""
    digest = hashlib.sha256()
    for row in records:
        digest.update(
            f"{row.round}:{row.gamma:.9g}:{row.episode}:{row.context_id}\n"
            .encode("utf-8")
        )
        candidate = row.candidate.detach().cpu().contiguous().numpy()
        digest.update(candidate.tobytes(order="C"))
        flow_base = getattr(row, "flow_base", None)
        if flow_base is not None:
            digest.update(
                flow_base.detach().cpu().contiguous().numpy().tobytes(order="C")
            )
        loss_mask = getattr(row, "loss_mask", None)
        if loss_mask is not None:
            digest.update(loss_mask.detach().cpu().contiguous().numpy().tobytes(order="C"))
        digest.update(f":valid_horizon={getattr(row, 'valid_horizon', None)}\n".encode())
    return digest.hexdigest()


def _tensor_identity_hash(values: torch.Tensor) -> str:
    array = values.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(tuple(array.shape)).encode())
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _policy_state_hash(policy: ExpansionPolicy) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(policy.state_dict().items()):
        digest.update(name.encode())
        digest.update(_tensor_identity_hash(value).encode())
    return digest.hexdigest()


def _gp_window_start_diagnostics(
    records: Sequence[QueryRecord],
    gammas: Sequence[float],
) -> dict[str, dict[str, int | float | None]]:
    output: dict[str, dict[str, int | float | None]] = {}
    for gamma in map(float, gammas):
        rows = [row for row in records if row.gamma == gamma]
        starts = np.asarray([
            int(row.window_start)
            for row in rows if row.window_start is not None
        ], dtype=int)
        trajectories = {
            str(row.trajectory_id)
            for row in rows if row.trajectory_id is not None
        }
        if len(starts):
            quantiles = np.quantile(starts, [0.0, 0.25, 0.5, 0.75, 1.0])
            stats = {
                "rows": int(len(rows)),
                "trajectories": int(len(trajectories)),
                "start_min": int(quantiles[0]),
                "start_q25": float(quantiles[1]),
                "start_median": float(quantiles[2]),
                "start_q75": float(quantiles[3]),
                "start_max": int(quantiles[4]),
                "start_le_4": int(np.sum(starts <= 4)),
                "start_le_9": int(np.sum(starts <= 9)),
            }
        else:
            stats = {
                "rows": int(len(rows)),
                "trajectories": int(len(trajectories)),
                "start_min": None,
                "start_q25": None,
                "start_median": None,
                "start_q75": None,
                "start_max": None,
                "start_le_4": 0,
                "start_le_9": 0,
            }
        output[f"{gamma:.9g}"] = stats
    return output


def _record_identity_key(row: QueryRecord) -> tuple[int, float, int, int, str]:
    """Order-independent deterministic key for frozen-feature coreset ties."""
    candidate = row.candidate.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256(candidate.tobytes(order="C"))
    flow_base = getattr(row, "flow_base", None)
    if flow_base is not None:
        digest.update(
            flow_base.detach().cpu().contiguous().numpy().tobytes(order="C")
        )
    loss_mask = getattr(row, "loss_mask", None)
    if loss_mask is not None:
        digest.update(loss_mask.detach().cpu().contiguous().numpy().tobytes(order="C"))
    digest.update(f":valid_horizon={getattr(row, 'valid_horizon', None)}".encode())
    candidate_hash = digest.hexdigest()
    return (row.round, row.gamma, row.episode, row.context_id, candidate_hash)


def _sliding_success_gp_rows(
    records: Sequence[QueryRecord],
    gammas: Sequence[float],
    total_cap: int,
    *,
    through_round: int,
    selector: str = "trajectory_uniform",
    reference_rounds: int | None = None,
) -> list[QueryRecord]:
    """Bounded deterministic support from exact-positive lineages, per gamma.

    ``records`` remains the immutable cumulative audit log.  Only the active
    posterior support changes. Negatives and current/future-round rows are
    rejected defensively even when callers pass an unfiltered archive.
    ``trajectory_uniform`` gives each retained
    trajectory nearly equal capacity and samples its window starts uniformly
    from departure through tail. ``fifo_tail`` exactly preserves the legacy
    high-window-start behavior for rollback. Both selectors are independent of
    archive/list order and preserve strict previous-round causality.
    """
    gamma_values = tuple(map(float, gammas))
    if total_cap % len(gamma_values) != 0:
        raise ValueError("sliding GP total cap must be divisible by gamma count")
    if selector not in {"trajectory_uniform", "fifo_tail"}:
        raise ValueError(
            "sliding GP selector must be trajectory_uniform or fifo_tail"
        )
    if reference_rounds is not None and reference_rounds < 1:
        raise ValueError("reference_rounds must be positive")
    first_round = (
        None
        if reference_rounds is None
        else int(through_round) - int(reference_rounds) + 1
    )
    per_gamma_cap = total_cap // len(gamma_values)

    def fifo_key(row: QueryRecord) -> tuple[int, int, int, int, str, str]:
        identity = _record_identity_key(row)[-1]
        return (
            int(row.round),
            int(row.window_start if row.window_start is not None else -1),
            int(row.episode),
            int(row.context_id),
            str(row.window_id or ""),
            identity,
        )

    active: list[QueryRecord] = []
    for gamma in sorted(gamma_values):
        rows = sorted(
            (
                row for row in records
                if (
                    row.gamma == gamma
                    and row.round <= through_round
                    and (first_round is None or row.round >= first_round)
                    and row.verification.valid
                    and not row.verification.error
                )
            ),
            key=fifo_key,
        )
        if selector == "fifo_tail":
            active.extend(rows[-per_gamma_cap:])
            continue

        trajectories: dict[str, list[QueryRecord]] = {}
        for row in rows:
            if row.trajectory_id is None or row.window_start is None:
                raise ValueError(
                    "trajectory_uniform GP support requires trajectory_id and "
                    "window_start on every successful-window row"
                )
            trajectories.setdefault(str(row.trajectory_id), []).append(row)
        trajectory_rows = [
            (
                max(int(row.round) for row in block),
                trajectory_id,
                sorted(block, key=fifo_key),
            )
            for trajectory_id, block in trajectories.items()
        ]
        trajectory_rows.sort(key=lambda item: (item[0], item[1]))
        if len(trajectory_rows) > per_gamma_cap:
            trajectory_rows = trajectory_rows[-per_gamma_cap:]

        quotas = [0] * len(trajectory_rows)
        remaining = min(
            per_gamma_cap,
            sum(len(block) for _, _, block in trajectory_rows),
        )
        while remaining:
            progressed = False
            # Newer trajectories receive any deterministic remainder.
            for index in range(len(trajectory_rows) - 1, -1, -1):
                block = trajectory_rows[index][2]
                if quotas[index] >= len(block):
                    continue
                quotas[index] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
            if not progressed:
                break

        trajectory_count = len(trajectory_rows)
        for trajectory_index, ((_, _, block), quota) in enumerate(
            zip(trajectory_rows, quotas)
        ):
            if quota == 0:
                continue
            by_start: dict[int, list[QueryRecord]] = {}
            for row in block:
                by_start.setdefault(int(row.window_start), []).append(row)
            stages = [
                sorted(by_start[start], key=fifo_key)
                for start in sorted(by_start)
            ]
            stage_quotas = [0] * len(stages)
            if quota == 1:
                # When the number of retained trajectories reaches the
                # per-gamma cap, selecting block[0] for every trajectory
                # creates the mirror-image failure of the legacy tail bias:
                # an all-departure GP. Stratify the single retained row from
                # departure through tail across the retained trajectories.
                denominator = max(trajectory_count - 1, 1)
                stage_quotas[
                    (trajectory_index * (len(stages) - 1)) // denominator
                ] = 1
            elif quota <= len(stages):
                for position in range(quota):
                    stage_quotas[
                        (position * (len(stages) - 1)) // (quota - 1)
                    ] = 1
            else:
                remaining_stage_rows = quota
                # Equalize time-stage mass before retaining a second query from
                # any stage. Rotating the remainder avoids a systematic early-
                # trajectory or departure preference.
                stage_order = list(range(len(stages)))
                rotation = trajectory_index % len(stage_order)
                stage_order = stage_order[rotation:] + stage_order[:rotation]
                while remaining_stage_rows:
                    progressed = False
                    for stage_index in stage_order:
                        if stage_quotas[stage_index] >= len(stages[stage_index]):
                            continue
                        stage_quotas[stage_index] += 1
                        remaining_stage_rows -= 1
                        progressed = True
                        if remaining_stage_rows == 0:
                            break
                    if not progressed:
                        break
            for stage_index, stage_quota in enumerate(stage_quotas):
                if stage_quota == 0:
                    continue
                stage = stages[stage_index]
                if stage_quota == 1:
                    selected_indices = [trajectory_index % len(stage)]
                else:
                    selected_indices = [
                        (position * (len(stage) - 1)) // (stage_quota - 1)
                        for position in range(stage_quota)
                    ]
                active.extend(stage[index] for index in selected_indices)
    return active


@torch.no_grad()
def _frozen_phi_farthest_first(
    policy: ExpansionPolicy,
    records: Sequence[QueryRecord],
    count: int,
    device: torch.device,
    *,
    reference: Sequence[QueryRecord] = (),
) -> list[QueryRecord]:
    """Deterministic route-label-free coreset in a frozen normalized ``phi``.

    With a reference set, candidates farthest from the reference and from
    already selected candidates enter first. Without one, the deterministic
    first center is farthest from the candidate centroid. Stable identity
    sorting resolves exact geometric ties without depending on archive order.
    """
    if count < 1 or not records:
        return []
    rows = sorted(records, key=_record_identity_key)
    if len(rows) <= count:
        return rows
    features = _normalize(_embed_records(policy, rows, device))
    if reference:
        reference_rows = sorted(reference, key=_record_identity_key)
        reference_features = _normalize(
            _embed_records(policy, reference_rows, device)
        )
        squared = (
            (features * features).sum(1, keepdim=True)
            + (reference_features * reference_features).sum(1)[None]
            - 2.0 * features @ reference_features.T
        ).clamp_min(0.0)
        minimum_distance = squared.min(dim=1).values
    else:
        centroid = features.mean(dim=0, keepdim=True)
        first_distance = (features - centroid).square().sum(dim=1)
        first = int(torch.argmax(first_distance))
        minimum_distance = (features - features[first]).square().sum(dim=1)
        selected = [first]
        minimum_distance[first] = -1.0

    if reference:
        selected = []
    while len(selected) < min(count, len(rows)):
        local = int(torch.argmax(minimum_distance))
        selected.append(local)
        distance = (features - features[local]).square().sum(dim=1)
        minimum_distance = torch.minimum(minimum_distance, distance)
        minimum_distance[selected] = -1.0
    return [rows[index] for index in selected]


def _cumulative_gp_quotas(config: ExpansionConfig) -> tuple[int, int, int]:
    """Return per-gamma total, immutable-anchor, and ingress quotas."""
    per_gamma = config.gp_buffer_cap // len(config.gammas)
    anchor = per_gamma // 2
    adaptive = per_gamma - anchor
    ingress = max(1, adaptive // 2)
    return per_gamma, anchor, ingress


@torch.no_grad()
def _context_kcenter_replay(
    records: Sequence[QueryRecord],
    policy: ExpansionPolicy,
    quota: int,
    action_weight: float,
    *,
    embedding_batch_size: int = 4096,
) -> list[QueryRecord]:
    """Select a route-label-free geometric coreset independently per query context.

    The current model supplies normalized ``phi`` and the normalized raw plan supplies
    complementary geometry.  Farthest-first selection acts only on verified positives;
    it does not alter GP fitting, acquisition, labels, or execution.
    """
    if not records:
        return []
    device = next(policy.parameters()).device
    joint_blocks = []
    for start in range(0, len(records), embedding_batch_size):
        block = records[start:start + embedding_batch_size]
        _, candidates = (value.to(device) for value in _stack(block))
        phi = _normalize(_embed_records(policy, block, device))
        action = _normalize(candidates.reshape(len(candidates), -1))
        joint_blocks.append(
            torch.cat([phi, action_weight * action], dim=1)
            .div(math.sqrt(1.0 + action_weight ** 2))
            .cpu()
        )
    joint = torch.cat(joint_blocks)
    groups: dict[tuple[int, float, int, int], list[int]] = {}
    for index, row in enumerate(records):
        groups.setdefault(
            (row.round, row.gamma, row.episode, row.context_id), []
        ).append(index)

    selected_global: list[int] = []
    for key in sorted(groups):
        indices = groups[key]
        if len(indices) <= quota:
            selected_global.extend(indices)
            continue
        features = joint[indices]
        distances = (
            (features * features).sum(1, keepdim=True)
            + (features * features).sum(1)[None]
            - 2.0 * features @ features.T
        ).clamp_min(0.0)
        flat = int(torch.argmax(distances))
        first, second = divmod(flat, len(indices))
        chosen = [first]
        if second != first:
            chosen.append(second)
        min_distance = distances[:, chosen].min(dim=1).values
        while len(chosen) < quota:
            min_distance[chosen] = -1.0
            next_index = int(torch.argmax(min_distance))
            if next_index in chosen:
                # All features are identical; stable archive order is the only
                # label-free deterministic tie break.
                next_index = next(i for i in range(len(indices)) if i not in chosen)
            chosen.append(next_index)
            min_distance = torch.minimum(min_distance, distances[:, next_index])
        selected_global.extend(indices[index] for index in chosen)
    return [records[index] for index in sorted(selected_global)]


@torch.no_grad()
def _cluster_balanced_replay(
    records: Sequence[QueryRecord],
    policy: ExpansionPolicy,
    cluster_count: int,
    action_weight: float,
    budget: int,
    generator: torch.Generator,
    *,
    iterations: int = 8,
    embedding_batch_size: int = 4096,
) -> list[QueryRecord]:
    """Flatten current-feature density without using route labels.

    Mini-batch-sized replay mass is divided equally across gamma and then across
    k-means cells in normalized ``[phi, U]``.  This is a training selector only;
    it does not alter the GP posterior or verifier.
    """
    if not records or budget >= len(records):
        return list(records)
    device = next(policy.parameters()).device
    blocks = []
    for start in range(0, len(records), embedding_batch_size):
        block = records[start:start + embedding_batch_size]
        _, candidates = (value.to(device) for value in _stack(block))
        phi = _normalize(_embed_records(policy, block, device))
        action = _normalize(candidates.reshape(len(candidates), -1))
        blocks.append(
            torch.cat([phi, action_weight * action], dim=1)
            .div(math.sqrt(1.0 + action_weight ** 2))
            .cpu()
        )
    features = torch.cat(blocks)
    gamma_groups: dict[float, list[int]] = {}
    for index, row in enumerate(records):
        gamma_groups.setdefault(row.gamma, []).append(index)
    gammas = sorted(gamma_groups)
    base, remainder = divmod(budget, len(gammas))
    selected_global: list[int] = []
    for gamma_index, gamma in enumerate(gammas):
        indices = gamma_groups[gamma]
        gamma_budget = min(len(indices), base + int(gamma_index < remainder))
        if gamma_budget >= len(indices):
            selected_global.extend(indices)
            continue
        x = features[indices]
        k = min(cluster_count, len(indices), gamma_budget)
        first = int(torch.randint(
            len(indices), (1,), generator=generator, device=x.device
        ))
        seeds = [first]
        min_seed_distance = (
            (x - x[first]).square().sum(dim=1)
        )
        while len(seeds) < k:
            min_seed_distance[seeds] = -1.0
            next_seed = int(torch.argmax(min_seed_distance))
            seeds.append(next_seed)
            min_seed_distance = torch.minimum(
                min_seed_distance,
                (x - x[next_seed]).square().sum(dim=1),
            )
        centers = x[seeds].clone()
        assignments = torch.zeros(len(x), dtype=torch.long)
        distances = torch.empty(len(x))
        for _ in range(iterations):
            squared = (
                (x * x).sum(1, keepdim=True)
                + (centers * centers).sum(1)[None]
                - 2.0 * x @ centers.T
            ).clamp_min(0.0)
            distances, assignments = squared.min(dim=1)
            sums = torch.zeros_like(centers)
            counts = torch.zeros(k, dtype=x.dtype)
            sums.index_add_(0, assignments, x)
            counts.index_add_(0, assignments, torch.ones(len(x), dtype=x.dtype))
            nonempty = counts > 0
            centers[nonempty] = sums[nonempty] / counts[nonempty, None]
            centers = _normalize(centers)

        cells = [
            torch.nonzero(assignments == cluster, as_tuple=False).flatten().tolist()
            for cluster in range(k)
        ]
        for cluster, cell in enumerate(cells):
            cell.sort(key=lambda local: (float(distances[local]), local))
        chosen_local: list[int] = []
        cursor = 0
        while len(chosen_local) < gamma_budget:
            progressed = False
            for cell in cells:
                if cursor < len(cell):
                    chosen_local.append(cell[cursor])
                    progressed = True
                    if len(chosen_local) == gamma_budget:
                        break
            if not progressed:
                break
            cursor += 1
        selected_global.extend(indices[local] for local in chosen_local)
    return [records[index] for index in sorted(selected_global)]


def _softmin_choice(costs: Sequence[float], target: float,
                    generator: torch.Generator, device: torch.device) -> tuple[int, float]:
    """Sample one cost with a per-context normalized-ESS target."""
    values = torch.as_tensor(costs, dtype=torch.float32, device=device)
    if len(values) == 1 or float(values.max() - values.min()) <= 1.0e-12:
        probability = torch.full_like(values, 1.0 / len(values))
    else:
        scores = -values
        temperature = calibrate_fixed_beta(
            [scores.detach().cpu()], target, lower=1.0e-8, upper=1.0e8
        )
        probability = torch.softmax((scores - scores.max()) / temperature, dim=0)
    choice = int(torch.multinomial(probability, 1, generator=generator))
    ess = float(
        1.0 / (
            probability.to(
                _stable_accumulation_dtype(probability)
            ).square().sum()
            * len(probability)
        )
    )
    return choice, ess


def _min_cost_score(
    result: Verification,
    step_margin_weight: float,
) -> float:
    if step_margin_weight > 0.0 and result.step_margin is None:
        raise ValueError(
            "execution_step_margin_weight requires verifier step_margin values"
        )
    return float(
        result.execution_cost
        - step_margin_weight * float(result.step_margin or 0.0)
    )


def perturb_plan_candidates(policy: ExpansionPolicy, candidates: torch.Tensor, std: float,
                            generator: torch.Generator,
                            scope: str = "coherent_horizon") -> torch.Tensor:
    """Add one action-space Gaussian vector per candidate plan.

    The canonical ``coherent_horizon`` law broadcasts the same 3-D vector over H.
    ``first_action`` is retained only as an explicit ablation.  The returned plans are
    embedded, verified, stored, executed, and used as CFM targets.
    """
    if std == 0.0:
        return candidates
    if candidates.ndim < 2:
        raise ValueError("candidate plans must include batch and action dimensions")
    if scope not in {"first_action", "coherent_horizon"}:
        raise ValueError("scope must be first_action or coherent_horizon")
    noise_shape = (len(candidates), candidates.shape[-1])
    noise = torch.randn(noise_shape, dtype=candidates.dtype, device=candidates.device,
                        generator=generator) * std
    perturbed = candidates.clone()
    if candidates.ndim == 2:
        perturbed += noise
    elif scope == "first_action":
        perturbed[:, 0] += noise
    else:
        broadcast = noise.reshape(len(candidates), *([1] * (candidates.ndim - 2)),
                                  candidates.shape[-1])
        perturbed += broadcast
    limit = getattr(policy, "control_limit", None)
    return perturbed.clamp(-float(limit), float(limit)) if limit is not None else perturbed


def _gradient_norm(gradients: Sequence[torch.Tensor | None], device) -> torch.Tensor:
    return torch.sqrt(sum((gradient.square().sum() for gradient in gradients
                           if gradient is not None),
                          torch.zeros((), device=device)).clamp_min(0.0))


def _update(policy: ExpansionPolicy, task: ExpansionTask,
            optimizer: torch.optim.Optimizer,
            positives: Sequence[QueryRecord], negatives: Sequence[QueryRecord],
            cfg: ExpansionConfig, generator: torch.Generator) -> dict[str, float | int | None]:
    if not positives:
        return {"steps": 0, "positive_loss": None, "negative_loss": None}
    parameters = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    device = parameters[0].device
    order = torch.randperm(
        len(positives), generator=generator, device=device
    ).cpu().tolist()
    if cfg.inner_steps is not None:
        order = order[:cfg.inner_steps * cfg.batch_size]
    weights = _equal_mass_weights(positives)
    if cfg.coverage_replay == "circular_voronoi":
        weights = _circular_voronoi_weights(positives, weights)
    positive_values, negative_values = [], []
    for start in range(0, len(order), cfg.batch_size):
        ids = order[start:start + cfg.batch_size]
        batch = [positives[index] for index in ids]
        contexts, candidates = (value.to(device) for value in _stack(batch))
        loss_mask = _stack_loss_masks(batch)
        if loss_mask is not None:
            loss_mask = loss_mask.to(device)
        batch_weights = weights[ids].to(device)
        if cfg.replay_augmentation == "task_d4":
            augment = getattr(task, "d4_replay_batch", None)
            if not callable(augment):
                raise TypeError(
                    "task_d4 replay requires task.d4_replay_batch(contexts, candidates)"
                )
            contexts, candidates = augment(contexts, candidates)
            if len(contexts) != 4 * len(batch) or len(candidates) != len(contexts):
                raise ValueError("task_d4 replay must return four transforms per row")
            batch_weights = batch_weights.repeat(4)
            if loss_mask is not None:
                loss_mask = loss_mask.repeat(4, *([1] * (loss_mask.ndim - 1)))
        for _ in range(cfg.microbatch_repeats):
            if loss_mask is None:
                per_sample = policy.cfm_loss(
                    contexts, candidates, reduction="none",
                )
            else:
                per_sample = policy.cfm_loss(
                    contexts, candidates, reduction="none",
                    loss_mask=loss_mask,
                )
            positive_loss = (per_sample * batch_weights).mean()
            optimizer.zero_grad()
            if cfg.negative_alpha and negatives:
                negative_ids = torch.randint(
                    len(negatives), (len(batch),), generator=generator, device=device
                ).cpu().tolist()
                neg_context, neg_candidate = _stack([negatives[index] for index in negative_ids])
                negative_loss = policy.cfm_loss(
                    neg_context.to(device), neg_candidate.to(device), reduction="mean"
                )
                positive_grad = torch.autograd.grad(
                    positive_loss, parameters, allow_unused=True)
                negative_grad = torch.autograd.grad(
                    negative_loss, parameters, allow_unused=True)
                pos_norm = _gradient_norm(positive_grad, device)
                neg_norm = _gradient_norm(negative_grad, device)
                rho = cfg.negative_alpha * pos_norm / (neg_norm + 1.0e-12)
                for parameter, pos, neg in zip(parameters, positive_grad, negative_grad):
                    if pos is None and neg is None:
                        parameter.grad = None
                    elif pos is None:
                        parameter.grad = -rho * neg.detach()
                    elif neg is None:
                        parameter.grad = pos.detach()
                    else:
                        parameter.grad = pos.detach() - rho * neg.detach()
                negative_values.append(float(negative_loss.detach()))
            else:
                positive_loss.backward()
            optimizer.step()
            positive_values.append(float(positive_loss.detach()))
    return {
        "steps": len(positive_values),
        "positive_loss": float(np.mean(positive_values)),
        "negative_loss": (float(np.mean(negative_values)) if negative_values else None),
    }


def _head_only_parameters(policy: ExpansionPolicy) -> list[torch.nn.Parameter]:
    """Freeze the policy backbone and return only its final-head parameters."""
    head = getattr(policy, "head", None)
    if not isinstance(head, torch.nn.Module):
        raise TypeError(
            "head_only_update requires a policy with a torch Module head"
        )
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    parameters = list(head.parameters())
    if not parameters:
        raise ValueError("policy head has no parameters")
    for parameter in parameters:
        parameter.requires_grad_(True)
    return parameters


def _run_safe_expansion_impl(
    policy: ExpansionPolicy,
    task: ExpansionTask,
    output_dir: str | Path,
    *,
    config: ExpansionConfig = ExpansionConfig(),
    calibration_features: torch.Tensor | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    round_callback: Callable[[dict[str, Any]], None] | None = None,
    verifier: _OrderedVerifier,
) -> dict[str, Any]:
    """Run expert-free B1 expansion; task callbacks supply every task-specific fact."""
    config.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError("expansion output_dir must be empty")
    if config.rbf_lengthscale is None:
        if calibration_features is None:
            raise ValueError("provide 50 pretrained embeddings or set rbf_lengthscale explicitly")
        if (config.acquisition_feature == "task_angular"
                and (calibration_features.ndim != 2
                     or calibration_features.shape[1] != 2)):
            raise ValueError(
                "task_angular RBF calibration requires two-dimensional "
                "task acquisition features"
            )
        lengthscale = mean_pairwise_lengthscale(calibration_features)
    else:
        lengthscale = float(config.rbf_lengthscale)
    archive: list[QueryRecord] = []
    if config.freeze_visual_encoder:
        freeze_visual_encoder = getattr(
            policy, "freeze_visual_encoder_for_expansion", None,
        )
        if not callable(freeze_visual_encoder):
            raise TypeError(
                "freeze_visual_encoder requires a policy with "
                "freeze_visual_encoder_for_expansion()"
            )
        freeze_visual_encoder()
    effective_optimizer_scope = (
        "head_only" if config.head_only_update else config.optimizer_scope
    )
    if effective_optimizer_scope == "head_only":
        optimizer_parameters = _head_only_parameters(policy)
    elif effective_optimizer_scope == "last_block_and_head":
        scoped_parameters = getattr(
            policy, "expansion_optimizer_parameters", None,
        )
        if not callable(scoped_parameters):
            raise TypeError(
                "last_block_and_head requires a policy with "
                "expansion_optimizer_parameters(scope)"
            )
        optimizer_parameters = list(scoped_parameters(effective_optimizer_scope))
        if not optimizer_parameters:
            raise ValueError("last_block_and_head returned no parameters")
    elif config.first_layer_lr_scale < 1.0:
        parameter_groups = getattr(policy, "expansion_parameter_groups", None)
        if not callable(parameter_groups):
            raise TypeError(
                "first_layer_lr_scale<1 requires "
                "policy.expansion_parameter_groups(base_lr, scale)"
            )
        optimizer_parameters = parameter_groups(
            config.learning_rate, config.first_layer_lr_scale,
        )
    else:
        optimizer_parameters = [
            parameter for parameter in policy.parameters() if parameter.requires_grad
        ]
    trainable_parameter_names = [
        name for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    ]
    trainable_parameter_count = sum(
        parameter.numel() for parameter in policy.parameters()
        if parameter.requires_grad
    )
    optimizer = torch.optim.Adam(optimizer_parameters, lr=config.learning_rate)
    rng = np.random.default_rng(config.seed)
    device = next(policy.parameters()).device
    acquisition_policy = policy
    if config.gp_reference_mode in {
            "round1_fixed_frozen_phi", "cumulative_accepted_frozen_phi",
            "cumulative_success_per_gamma_frozen_phi_exact",
            "sliding_success_per_gamma_frozen_phi",
            "sliding_positive_per_gamma_frozen_phi",
            "sliding_executed_positive_per_gamma_frozen_phi"}:
        if config.acquisition_feature != "learned_phi":
            raise ValueError(
                f"{config.gp_reference_mode} requires learned_phi acquisition"
            )
        acquisition_policy = copy.deepcopy(policy)
        for parameter in acquisition_policy.parameters():
            parameter.requires_grad_(False)
        eval_fn = getattr(acquisition_policy, "eval", None)
        if callable(eval_fn):
            eval_fn()
    torch_rng = torch.Generator(device=device)
    torch_rng.manual_seed(config.seed)
    torch.manual_seed(config.seed)
    beta = float(config.beta)
    round_rows = []
    frozen_gp_rows: list[QueryRecord] | None = None
    frozen_gp_hash: str | None = None
    round1_gp_candidates: list[QueryRecord] = []
    gp_evidence: list[QueryRecord] = []
    cumulative_anchors = {float(gamma): [] for gamma in config.gammas}
    cumulative_adaptive = {float(gamma): [] for gamma in config.gammas}
    torch.save({"round": 0, "model": policy.state_dict(),
                "config": asdict(config), "pretrained": True},
               output_dir / "checkpoint_000.pt")
    for round_i in range(1, config.rounds + 1):
        round_started = time.perf_counter()
        beta_used = beta
        gp_anchor_count = 0
        gp_adaptive_count = 0
        acquisition_representation_hash = _policy_state_hash(
            acquisition_policy
        )
        if config.gp_reference_mode == "round1_fixed_frozen_phi":
            if round_i == 1:
                gp_rows = []
            else:
                if frozen_gp_rows is None:
                    round1_positive = list(round1_gp_candidates)
                    if not round1_positive:
                        raise RuntimeError(
                            "round1_fixed_frozen_phi found no eligible round-1 "
                            "positive reference rows"
                        )
                    frozen_rng = np.random.default_rng(config.seed + 7919)
                    frozen_gp_rows = _balanced_cap(
                        round1_positive, config.gp_buffer_cap, frozen_rng
                    )
                    frozen_gp_hash = _record_identity_hash(frozen_gp_rows)
                gp_rows = frozen_gp_rows
        elif config.gp_reference_mode == "cumulative_accepted_frozen_phi":
            if round_i == 1:
                gp_rows = []
            else:
                per_gamma_cap, anchor_cap, ingress_cap = (
                    _cumulative_gp_quotas(config)
                )
                adaptive_cap = per_gamma_cap - anchor_cap
                if round_i == 2:
                    for gamma in sorted(cumulative_anchors):
                        candidates = [
                            row for row in gp_evidence
                            if row.round == 1 and row.gamma == gamma
                        ]
                        cumulative_anchors[gamma] = _frozen_phi_farthest_first(
                            acquisition_policy, candidates, anchor_cap, device
                        )
                else:
                    for gamma in sorted(cumulative_adaptive):
                        incoming = [
                            row for row in gp_evidence
                            if row.round == round_i - 1 and row.gamma == gamma
                        ]
                        reference = (
                            cumulative_anchors[gamma]
                            + cumulative_adaptive[gamma]
                        )
                        admitted = _frozen_phi_farthest_first(
                            acquisition_policy, incoming, ingress_cap, device,
                            reference=reference,
                        )
                        cumulative_adaptive[gamma] = (
                            _frozen_phi_farthest_first(
                                acquisition_policy,
                                cumulative_adaptive[gamma] + admitted,
                                adaptive_cap,
                                device,
                                reference=cumulative_anchors[gamma],
                            )
                        )
                gp_rows = [
                    row
                    for gamma in sorted(cumulative_anchors)
                    for row in (
                        cumulative_anchors[gamma]
                        + cumulative_adaptive[gamma]
                    )
                ]
                frozen_gp_hash = (
                    _record_identity_hash(gp_rows) if gp_rows else None
                )
                gp_anchor_count = sum(map(len, cumulative_anchors.values()))
                gp_adaptive_count = sum(map(len, cumulative_adaptive.values()))
        elif (
            config.gp_reference_mode
            == "cumulative_success_per_gamma_frozen_phi_exact"
        ):
            # Every row was reconstructed from a committed terminal-SUCCESS
            # trajectory in an earlier round. Exact means no cap-based thinning:
            # the explicit support limit is checked atomically before commit.
            gp_rows = list(gp_evidence)
            frozen_gp_hash = (
                _record_identity_hash(gp_rows) if gp_rows else None
            )
        elif config.gp_reference_mode in {
            "sliding_success_per_gamma_frozen_phi",
            "sliding_success_per_gamma_current_phi",
            "sliding_positive_per_gamma_frozen_phi",
            "sliding_positive_per_gamma_current_phi",
            "sliding_executed_positive_per_gamma_frozen_phi",
        }:
            gp_rows = _sliding_success_gp_rows(
                gp_evidence,
                config.gammas,
                config.gp_buffer_cap,
                through_round=round_i - 1,
                selector=config.gp_sliding_row_selector,
                reference_rounds=config.gp_reference_rounds,
            )
            if any(row.round >= round_i for row in gp_rows):
                raise RuntimeError(
                    "sliding GP support must contain only previous-round evidence"
                )
            frozen_gp_hash = (
                _record_identity_hash(gp_rows) if gp_rows else None
            )
        else:
            previous_positive = _recent(
                archive, round_i - 1, config.replay_rounds, positive=True
            )
            previous_positive = _top_uncertainty_by_round(
                previous_positive, config.replay_top_fraction,
                group_by_gamma=(
                    config.archive_rule == "successful_executed_windows"
                ),
            )
            gp_rows = _balanced_cap(previous_positive, config.gp_buffer_cap, rng)
        shared_gp: RBFPosterior | None = None
        gp_by_gamma: dict[float, RBFPosterior] = {}
        gp_feature_hash_by_gamma: dict[str, str | None] = {
            f"{float(gamma):.9g}": None for gamma in config.gammas
        }
        if (
            config.acquisition_feature == "task_angular"
            or config.gp_reference_mode in {
                "cumulative_success_per_gamma_frozen_phi_exact",
                "sliding_success_per_gamma_frozen_phi",
                "sliding_success_per_gamma_current_phi",
                "sliding_positive_per_gamma_frozen_phi",
                "sliding_positive_per_gamma_current_phi",
                "sliding_executed_positive_per_gamma_frozen_phi",
            }
        ):
            gp_by_gamma = {
                float(gamma): RBFPosterior(lengthscale, config.gp_noise)
                for gamma in config.gammas
            }
            for gamma, gp in gp_by_gamma.items():
                gamma_rows = [row for row in gp_rows if row.gamma == gamma]
                if gamma_rows:
                    if config.acquisition_feature == "task_angular":
                        features = _stack_acquisition_features(
                            gamma_rows, device
                        )
                    else:
                        features = _embed_records(
                            acquisition_policy, gamma_rows, device
                        )
                    gp_feature_hash_by_gamma[f"{gamma:.9g}"] = (
                        _tensor_identity_hash(_normalize(features))
                    )
                    gp.set_buffer(features)
        else:
            shared_gp = RBFPosterior(lengthscale, config.gp_noise)
            if gp_rows:
                shared_gp.set_buffer(
                    _embed_records(acquisition_policy, gp_rows, device)
                )
        gp_refit_s = time.perf_counter() - round_started
        gather_started = time.perf_counter()
        episodes: list[dict[str, Any]] = []
        retry_capability_reverify_queries = 0
        retry_batches_by_gamma = {
            float(gamma): 0 for gamma in config.gammas
        }
        completed_retry_gammas: set[float] = set()
        exhausted_retry_gammas: set[float] = set()

        def launch_episode_batch(gamma_value: float) -> None:
            retry_batch = retry_batches_by_gamma[gamma_value]
            for replica in range(config.parallel_episodes):
                episode = len(episodes)
                if config.archive_rule in {
                    "successful_executed_windows",
                    "preterminal_resolved_queries",
                    "executed_plus_nvp_negative",
                }:
                    reset_seed = _counter_seed(
                        config.seed, "reset", round_i, retry_batch, replica
                    )
                else:
                    reset_seed = config.seed + 10000 * round_i + episode
                episodes.append({
                    "gamma": gamma_value, "episode": episode,
                    "replica": replica, "retry_batch": retry_batch,
                    "step": 0, "status": None,
                    "state": task.reset(
                        gamma_value, episode, reset_seed
                    ),
                    "executed_steps": [],
                })
            retry_batches_by_gamma[gamma_value] += 1

        def reverify_executed_suffix(
            executed_steps: Sequence[dict[str, Any]],
            start: int,
            horizon: int,
            gamma_value: float,
        ) -> tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Verification
        ]:
            """Verify the available executed suffix and pad only for fixed-shape CFM."""
            valid_horizon = min(horizon, len(executed_steps) - start)
            if valid_horizon < 1:
                raise ValueError("executed suffix must contain at least one action")
            context_window = executed_steps[start]["context"].to(device)
            actual_window = torch.stack([
                executed_steps[offset]["action"]
                for offset in range(start, start + valid_horizon)
            ]).to(device)
            padded_window = torch.zeros(
                (horizon, *actual_window.shape[1:]),
                dtype=actual_window.dtype,
                device=device,
            )
            padded_window[:valid_horizon] = actual_window
            loss_mask = torch.zeros_like(padded_window)
            loss_mask[:valid_horizon] = 1.0
            verification = list(task.verify(
                context_window, actual_window.unsqueeze(0), gamma_value
            ))
            if len(verification) != 1:
                raise RuntimeError(
                    "reconstructed executed-window verification must return "
                    "exactly one result"
                )
            return (
                context_window, actual_window, padded_window, loss_mask,
                verification[0],
            )

        def paired_executed_suffix_base(
            executed_steps: Sequence[dict[str, Any]],
            start: int,
            horizon: int,
        ) -> torch.Tensor | None:
            """Compose a successful window from its actual action/base pairs."""
            if not config.paired_noised_representation:
                return None
            valid_horizon = min(horizon, len(executed_steps) - start)
            paired_actions = []
            for step_record in executed_steps[start:start + valid_horizon]:
                flow_base_action = step_record.get("flow_base_action")
                if flow_base_action is None:
                    raise RuntimeError(
                        "paired successful-window replay is missing an executed "
                        "first-action flow base"
                    )
                paired_actions.append(flow_base_action.to(device))
            actual_base = torch.stack(paired_actions)
            if valid_horizon == horizon:
                return actual_base
            padding = torch.zeros(
                (horizon - valid_horizon, *actual_base.shape[1:]),
                dtype=actual_base.dtype,
                device=device,
            )
            return torch.cat([actual_base, padding], dim=0)

        def has_commit_capable_window(
            episode_record: dict[str, Any],
            gamma_value: float,
        ) -> bool:
            """Reverify executed sliding windows before accepting a retry batch."""
            nonlocal retry_capability_reverify_queries
            executed_steps = episode_record["executed_steps"]
            if not executed_steps:
                return False
            horizons = {
                int(step_record["plan_horizon"])
                for step_record in executed_steps
            }
            if len(horizons) != 1:
                raise RuntimeError(
                    "executed candidates changed horizon within one "
                    "successful trajectory"
                )
            horizon = horizons.pop()
            action_shapes = {
                tuple(step_record["action"].shape)
                for step_record in executed_steps
            }
            if len(action_shapes) != 1:
                raise RuntimeError(
                    "executed action shape changed within one successful trajectory"
                )
            for start in range(len(executed_steps)):
                *_, result = reverify_executed_suffix(
                    executed_steps, start, horizon, gamma_value,
                )
                retry_capability_reverify_queries += 1
                if (
                    not result.error
                    and result.valid
                    and result.progress_eligible
                ):
                    return True
            return False

        for gamma in config.gammas:
            launch_episode_batch(float(gamma))
        ess_values, marginal_ess_values, score_pools, nvp = [], [], [], 0
        sigma_pool_means: list[float] = []
        sigma_selected_means: list[float] = []
        execution_ess_values = []
        verifier_queries = verifier_positives = 0
        verifier_dispatch_s = 0.0
        verifier_jobs = 0
        verifier_candidates = 0
        context_id = 0

        def consume_verified(
            prepared: dict[str, Any],
            result_block: Sequence[Verification],
        ) -> None:
            """Apply ordered verifier results without changing rollout semantics."""
            nonlocal nvp, verifier_queries, verifier_positives
            episode = prepared["episode"]
            step = prepared["step"]
            gamma = prepared["gamma"]
            gather_rng = prepared["gather_rng"]
            state_before = prepared["state_before"]
            context = prepared["context"]
            flow_bases = prepared["flow_bases"]
            base_candidates = prepared["base_candidates"]
            candidates = prepared["candidates"]
            task_features = prepared["task_features"]
            sigma_k = prepared["sigma_k"]
            selected = prepared["selected"]
            selected_sigma = prepared["selected_sigma"]
            queried = prepared["queried"]
            current_context_id = prepared["context_id"]
            results = list(result_block)
            if len(results) != len(selected):
                raise RuntimeError(
                    "task.verify must return exactly one result per selected query"
                )
            gate_active = (
                config.target_gate_start_round is not None
                and round_i >= config.target_gate_start_round
            )
            current_records = {}
            lineage_trajectory_id = (
                _successful_trajectory_id(
                    round_i, gamma, int(episode["episode"])
                )
                if config.archive_rule in {
                    "preterminal_resolved_queries",
                    "executed_plus_nvp_negative",
                }
                else None
            )
            for local, (candidate, result, acquisition_sigma) in enumerate(zip(
                    queried, results, selected_sigma)):
                if result.error:
                    continue
                verifier_queries += 1
                verifier_positives += int(result.valid)
                if config.replay_acceptance == "safety_valid":
                    replay_eligible = bool(result.valid)
                else:
                    replay_eligible = bool(
                        result.valid
                        and (
                            config.target_gate_start_round is None
                            or (
                                result.progress_eligible
                                and (not gate_active or result.target_eligible)
                            )
                        )
                    )
                record = QueryRecord(
                    round_i, gamma, int(episode["episode"]),
                    current_context_id,
                    context.detach().cpu(), candidate.detach().cpu(), result,
                    acquisition_sigma=float(sigma_k[selected[local]]),
                    flow_base=(
                        flow_bases[selected[local]].detach().cpu()
                        if flow_bases is not None else None
                    ),
                    acquisition_feature=(
                        task_features[selected[local]].detach().cpu()
                        if task_features is not None else None
                    ),
                    replay_eligible=replay_eligible,
                    retry_batch=int(episode["retry_batch"]),
                    replica=int(episode["replica"]),
                    trajectory_id=lineage_trajectory_id,
                    window_id=(
                        _preterminal_query_id(
                            lineage_trajectory_id, step, local,
                        )
                        if lineage_trajectory_id is not None else None
                    ),
                    window_start=(
                        step if lineage_trajectory_id is not None else None
                    ),
                    valid_horizon=(
                        int(len(candidate))
                        if lineage_trajectory_id is not None else None
                    ),
                )
                current_records[local] = record
                if (
                    config.archive_rule != "successful_executed_windows"
                    and config.gp_reference_mode == "round1_fixed_frozen_phi"
                    and round_i == 1
                    and replay_eligible
                ):
                    round1_gp_candidates.append(record)
                if (
                    config.archive_rule != "successful_executed_windows"
                    and config.gp_reference_mode
                    == "cumulative_accepted_frozen_phi"
                    and replay_eligible
                ):
                    gp_evidence.append(record)
            if config.archive_rule == "preterminal_resolved_queries":
                # This is an append-only observation archive, not a terminal-
                # success selector. Preserve every resolved B query immediately;
                # only exact positives may support the next round's GP.
                archive.extend(current_records.values())
                gp_evidence.extend(
                    record for record in current_records.values()
                    if record.verification.valid
                )
            eligible = [
                index for index, result in enumerate(results)
                if (
                    not result.error
                    and result.valid
                    and result.progress_eligible
                    and (not gate_active or result.target_eligible)
                )
            ]
            nvp_reason = None
            if not eligible:
                episode["status"] = "NVP"
                nvp += 1
                safe = [
                    result for result in results
                    if not result.error and result.valid
                ]
                forward = [
                    result for result in safe if result.progress_eligible
                ]
                if gate_active and forward and not any(
                        result.target_eligible for result in forward):
                    nvp_reason = "TARGET"
                elif safe and not forward:
                    nvp_reason = "PROGRESS"
                else:
                    nvp_reason = "VERIFIER"
                archived_local = None
                if config.archive_rule == "all_queries":
                    for record in current_records.values():
                        record.nvp_context = True
                        archive.append(record)
                elif config.archive_rule == "preterminal_resolved_queries":
                    for record in current_records.values():
                        record.nvp_context = True
                elif config.archive_rule == "executed_plus_nvp_negative":
                    rejected = [
                        local for local, record in current_records.items()
                        if not record.verification.valid
                    ]
                    if rejected:
                        archived_local = min(
                            rejected,
                            key=lambda local: _min_cost_score(
                                results[local],
                                config.execution_step_margin_weight,
                            ),
                        )
                        current_records[archived_local].nvp_context = True
                        current_records[archived_local].executed = False
                        archive.append(current_records[archived_local])
                chosen = None
            else:
                archived_local = None
                if config.execution_rule == "max_margin":
                    chosen = max(
                        eligible, key=lambda index: results[index].margin
                    )
                elif config.execution_rule == "max_step_margin":
                    if any(
                        results[index].step_margin is None
                        for index in eligible
                    ):
                        raise ValueError(
                            "max_step_margin requires verifier step_margin values"
                        )
                    chosen = max(
                        eligible,
                        key=lambda index: float(results[index].step_margin),
                    )
                elif config.execution_rule == "max_uncertainty":
                    chosen = max(
                        eligible,
                        key=lambda index: float(sigma_k[selected[index]]),
                    )
                elif config.execution_rule == "uncertainty_cost":
                    costs = torch.as_tensor(
                        [results[index].execution_cost for index in eligible],
                        dtype=torch.float64,
                    )
                    uncertainties = torch.as_tensor(
                        [
                            float(sigma_k[selected[index]])
                            for index in eligible
                        ],
                        dtype=torch.float64,
                    )
                    cost_span = (
                        costs.max() - costs.min()
                    ).clamp_min(1.0e-12)
                    sigma_span = (
                        uncertainties.max() - uncertainties.min()
                    ).clamp_min(1.0e-12)
                    score = (
                        config.execution_uncertainty_weight
                        * (uncertainties - uncertainties.min()) / sigma_span
                        - (costs - costs.min()) / cost_span
                    )
                    chosen = eligible[int(torch.argmax(score))]
                elif config.execution_rule == "uniform_positive":
                    draw = int(torch.randint(
                        len(eligible), (1,), generator=gather_rng,
                        device=device,
                    ))
                    chosen = eligible[draw]
                elif config.execution_rule == "softmin_cost":
                    draw, execution_ess = _softmin_choice(
                        [
                            results[index].execution_cost
                            for index in eligible
                        ],
                        config.execution_ess_target, gather_rng, device,
                    )
                    execution_ess_values.append(execution_ess)
                    chosen = eligible[draw]
                else:
                    chosen = min(
                        eligible,
                        key=lambda index: _min_cost_score(
                            results[index],
                            config.execution_step_margin_weight,
                        ),
                    )
                if config.archive_rule == "all_queries":
                    archive.extend(current_records.values())
                elif config.archive_rule not in {
                    "successful_executed_windows",
                    "preterminal_resolved_queries",
                }:
                    archive.append(current_records[chosen])
                if chosen in current_records:
                    chosen_record = current_records[chosen]
                    chosen_record.executed = True
                    if chosen_record.trajectory_id is not None:
                        chosen_record.window_id = _successful_window_id(
                            chosen_record.trajectory_id, step,
                        )
                    if config.gp_reference_mode == (
                        "sliding_executed_positive_per_gamma_frozen_phi"
                    ):
                        if not chosen_record.verification.valid:
                            raise RuntimeError(
                                "executed-positive GP evidence must be exact-positive"
                            )
                        gp_evidence.append(chosen_record)
                staged_executed_step = None
                if config.archive_rule == "successful_executed_windows":
                    executed_candidate = queried[chosen].detach()
                    if (
                        executed_candidate.ndim < 2
                        or len(executed_candidate) < 1
                    ):
                        raise ValueError(
                            "successful_executed_windows requires candidates "
                            "with an explicit leading horizon dimension"
                        )
                    staged_executed_step = {
                        "context_id": current_context_id,
                        "context": context.detach().cpu(),
                        "action": executed_candidate[0].detach().cpu(),
                        "plan_horizon": int(len(executed_candidate)),
                    }
                    if config.paired_noised_representation:
                        if flow_bases is None:
                            raise RuntimeError(
                                "paired successful-window replay requires sampled "
                                "flow bases"
                            )
                        staged_executed_step["flow_base_action"] = (
                            flow_bases[selected[chosen]][0].detach().cpu()
                        )
                episode["state"] = task.advance(
                    episode["state"], queried[chosen]
                )
                if staged_executed_step is not None:
                    staged_executed_step["state_after"] = copy.deepcopy(
                        episode["state"]
                    )
                    episode["executed_steps"].append(staged_executed_step)
                episode["status"] = task.terminal(episode["state"])
            if event_callback is not None:
                event = {
                    "round": round_i, "step": step, "gamma": gamma,
                    "episode": int(episode["episode"]),
                    "context_id": current_context_id,
                    "retry_batch": int(episode["retry_batch"]),
                    "replica": int(episode["replica"]),
                    "state_before": state_before,
                    "state_after": episode["state"],
                    "context": context.detach().cpu(),
                    "base_candidates": base_candidates.detach().cpu(),
                    "flow_bases": (
                        flow_bases.detach().cpu()
                        if flow_bases is not None else None
                    ),
                    "candidates": candidates.detach().cpu(),
                    "sigma_K": sigma_k,
                    "selected": selected,
                    "selected_sigma": selected_sigma,
                    "verification": [asdict(result) for result in results],
                    "chosen_local": chosen,
                    "archived_negative_local": archived_local,
                    "status": episode["status"],
                    "nvp_reason": nvp_reason,
                    "target_gate_active": gate_active,
                }
                if task_features is not None:
                    event["task_acquisition_features_K"] = (
                        task_features.detach().cpu()
                    )
                event_callback(event)
            episode["step"] = step + 1
            if (
                episode["status"] is None
                and int(episode["step"]) >= config.max_steps
            ):
                episode["status"] = config.max_steps_status

        while True:
            active = [episode for episode in episodes if episode["status"] is None]
            if not active:
                if config.archive_rule != "successful_executed_windows":
                    break
                launched = False
                for gamma in map(float, config.gammas):
                    if gamma in completed_retry_gammas:
                        continue
                    gamma_episodes = [
                        episode for episode in episodes
                        if float(episode["gamma"]) == gamma
                    ]
                    for episode in gamma_episodes:
                        if (
                            episode["status"] == "SUCCESS"
                            and "commit_capable" not in episode
                        ):
                            episode["commit_capable"] = has_commit_capable_window(
                                episode, gamma
                            )
                    commit_capable_successes = sum(
                        episode["status"] == "SUCCESS"
                        and bool(episode.get("commit_capable", False))
                        for episode in gamma_episodes
                    )
                    if (
                        commit_capable_successes
                        >= config.successful_trajectories_per_gamma
                    ):
                        completed_retry_gammas.add(gamma)
                    elif retry_batches_by_gamma[gamma] < config.max_retry_batches:
                        launch_episode_batch(gamma)
                        launched = True
                    else:
                        completed_retry_gammas.add(gamma)
                        exhausted_retry_gammas.add(gamma)
                if not launched:
                    break
                active = [
                    episode for episode in episodes
                    if episode["status"] is None
                ]
            prepared_blocks: list[dict[str, Any]] = []
            for episode in active:
                step = int(episode["step"])
                gamma = float(episode["gamma"])
                if config.archive_rule in {
                    "successful_executed_windows",
                    "preterminal_resolved_queries",
                    "executed_plus_nvp_negative",
                }:
                    gather_rng = torch.Generator(device=device).manual_seed(
                        _counter_seed(
                            config.seed, "gather", round_i,
                            int(episode["retry_batch"]),
                            int(episode["replica"]), step,
                        )
                    )
                else:
                    gather_rng = torch_rng
                state_before = episode["state"]
                context = task.context(episode["state"], gamma).detach().to(device)
                flow_bases = None
                if config.paired_noised_representation:
                    sampler = getattr(policy, "sample_with_base", None)
                    if not callable(sampler):
                        raise TypeError(
                            "paired_noised_representation requires "
                            "policy.sample_with_base(context, count, generator) "
                            "returning (plans, initial_bases)"
                        )
                    if config.flow_base_std == 1.0:
                        base_candidates, flow_bases = sampler(
                            context, config.K, gather_rng
                        )
                    else:
                        base_candidates, flow_bases = sampler(
                            context, config.K, gather_rng,
                            base_std=config.flow_base_std,
                        )
                    base_candidates = base_candidates.detach()
                    flow_bases = flow_bases.detach()
                    if (len(flow_bases) != config.K
                            or flow_bases.reshape(config.K, -1).shape[1]
                            != base_candidates.reshape(config.K, -1).shape[1]):
                        raise ValueError(
                            "sample_with_base must return one plan-shaped initial "
                            "base per sampled candidate"
                        )
                else:
                    if config.flow_base_std == 1.0:
                        base_candidates = policy.sample(
                            context, config.K, gather_rng
                        ).detach()
                    else:
                        base_candidates = policy.sample(
                            context, config.K, gather_rng,
                            base_std=config.flow_base_std,
                        ).detach()
                candidates = perturb_plan_candidates(
                    policy, base_candidates, config.candidate_perturb_std, gather_rng,
                    config.candidate_perturb_scope,
                ).detach()
                task_features = None
                if (config.acquisition_feature == "task_angular"
                        or config.coverage_replay == "circular_voronoi"):
                    task_features = _task_angular_features(
                        task, context, candidates, gamma
                    )
                if config.acquisition_feature == "task_angular":
                    features = task_features
                    acquisition_gp = gp_by_gamma[gamma]
                else:
                    features = acquisition_policy.embed(
                        context, candidates, base=flow_bases
                    ) if flow_bases is not None else acquisition_policy.embed(
                        context, candidates
                    )
                    if (
                        config.gp_reference_mode in {
                            "cumulative_success_per_gamma_frozen_phi_exact",
                            "sliding_success_per_gamma_frozen_phi",
                            "sliding_success_per_gamma_current_phi",
                            "sliding_positive_per_gamma_frozen_phi",
                            "sliding_positive_per_gamma_current_phi",
                            "sliding_executed_positive_per_gamma_frozen_phi",
                        }
                    ):
                        acquisition_gp = gp_by_gamma[gamma]
                    else:
                        if shared_gp is None:
                            raise RuntimeError(
                                "learned_phi acquisition GP was not initialized"
                            )
                        acquisition_gp = shared_gp
                acquisition_covariance = acquisition_gp.covariance(features)
                sigma_k = (
                    torch.diagonal(acquisition_covariance) - acquisition_gp.noise
                ).clamp_min(0.0).sqrt().detach().cpu()
                score_pools.append(sigma_k)
                marginal_ess_values.append(normalized_ess(sigma_k, beta_used))
                selected, selected_sigma, ess = (
                    acquisition_gp.acquire_from_covariance(
                        acquisition_covariance,
                        config.B,
                        beta_used,
                        gather_rng,
                        sampling_device=features.device,
                    )
                )
                ess_values.extend(ess)
                sigma_pool_means.append(float(sigma_k.mean()))
                sigma_selected_means.append(
                    float(sigma_k[torch.as_tensor(selected, dtype=torch.long)].mean())
                )
                queried = candidates[selected]
                prepared_blocks.append({
                    "episode": episode,
                    "step": step,
                    "gamma": gamma,
                    "gather_rng": gather_rng,
                    "state_before": state_before,
                    "context": context,
                    "flow_bases": flow_bases,
                    "base_candidates": base_candidates,
                    "candidates": candidates,
                    "task_features": task_features,
                    "sigma_k": sigma_k,
                    "selected": selected,
                    "selected_sigma": selected_sigma,
                    "queried": queried,
                    "context_id": context_id,
                })
                context_id += 1
            verify_started = time.perf_counter()
            verified_blocks = verifier.verify_many([
                (prepared["context"], prepared["queried"], prepared["gamma"])
                for prepared in prepared_blocks
            ])
            verifier_dispatch_s += time.perf_counter() - verify_started
            verifier_jobs += len(prepared_blocks)
            verifier_candidates += sum(
                len(prepared["queried"]) for prepared in prepared_blocks
            )
            for prepared, results in zip(prepared_blocks, verified_blocks):
                consume_verified(prepared, results)
        gather_s = time.perf_counter() - gather_started
        statuses = [
            episode["status"] or config.max_steps_status
            for episode in episodes
        ]
        if (
            config.archive_rule == "successful_executed_windows"
            and exhausted_retry_gammas
        ):
            missing = ", ".join(
                f"{gamma:g}" for gamma in sorted(exhausted_retry_gammas)
            )
            raise RuntimeError(
                f"round {round_i} exhausted max_retry_batches="
                f"{config.max_retry_batches} before obtaining "
                f"{config.successful_trajectories_per_gamma} commit-capable "
                "terminal SUCCESS trajectories for "
                f"gamma(s): {missing}; round not committed"
            )
        successful_executed_commit_by_gamma: dict[str, dict[str, Any]] = {}
        committed_trajectory_ids: list[str] = []
        committed_window_ids: list[str] = []
        commit_reverify_queries = 0
        commit_reverify_valid = 0
        commit_reverify_progress = 0
        if config.archive_rule == "successful_executed_windows":
            committed_records: list[QueryRecord] = []
            for gamma in config.gammas:
                gamma = float(gamma)
                successful = sorted(
                    (
                        episode for episode in episodes
                        if float(episode["gamma"]) == gamma
                        and episode["status"] == "SUCCESS"
                        and bool(episode.get("commit_capable", False))
                    ),
                    key=lambda episode: int(episode["episode"]),
                )
                above_fraction_fn = getattr(
                    task, "successful_trajectory_above_fraction", None
                )
                mean_z_fn = getattr(
                    task, "successful_trajectory_mean_z", None
                )
                above_fractions: dict[int, float | None] = {}
                mean_z_values: dict[int, float | None] = {}
                for successful_episode in successful:
                    episode_id = int(successful_episode["episode"])
                    if callable(above_fraction_fn):
                        above_fraction = float(above_fraction_fn([
                            step_record["state_after"]
                            for step_record in successful_episode["executed_steps"]
                        ]))
                        if not math.isfinite(above_fraction):
                            raise ValueError(
                                "successful trajectory above fraction must be finite"
                            )
                        above_fractions[episode_id] = above_fraction
                    else:
                        above_fractions[episode_id] = None
                    if callable(mean_z_fn):
                        mean_z = float(mean_z_fn([
                            step_record["state_after"]
                            for step_record in successful_episode["executed_steps"]
                        ]))
                        if not math.isfinite(mean_z):
                            raise ValueError(
                                "successful trajectory mean z must be finite"
                            )
                        mean_z_values[episode_id] = mean_z
                    else:
                        mean_z_values[episode_id] = None
                if (
                    successful
                    and config.successful_trajectory_selector
                    == "max_above_fraction"
                ):
                    if not callable(above_fraction_fn):
                        raise TypeError(
                            "max_above_fraction requires task."
                            "successful_trajectory_above_fraction"
                        )
                    successful.sort(key=lambda episode: (
                        -float(above_fractions[int(episode["episode"])]),
                        int(episode["episode"]),
                    ))
                elif (
                    successful
                    and config.successful_trajectory_selector == "max_mean_z"
                ):
                    if not callable(mean_z_fn):
                        raise TypeError(
                            "max_mean_z requires task."
                            "successful_trajectory_mean_z"
                        )
                    successful.sort(key=lambda episode: (
                        -float(mean_z_values[int(episode["episode"])]),
                        int(episode["episode"]),
                    ))
                elif (
                    successful
                    and config.successful_trajectory_selector == "random_success"
                ):
                    successful.sort(key=lambda episode: (
                        _counter_seed(
                            config.seed, "successful_trajectory_selector",
                            round_i, f"{gamma:.9g}",
                            int(episode["episode"]),
                        ),
                        int(episode["episode"]),
                    ))
                detail: dict[str, Any] = {
                    "selector": config.successful_trajectory_selector,
                    "requested_trajectory_count": (
                        config.successful_trajectories_per_gamma
                    ),
                    "retry_batches_used": retry_batches_by_gamma[gamma],
                    "attempted_episode_count": (
                        retry_batches_by_gamma[gamma]
                        * config.parallel_episodes
                    ),
                    "success_episode_count": len(successful),
                    "success_episode_above_fractions": {
                        str(episode_id): above_fractions[episode_id]
                        for episode_id in sorted(above_fractions)
                    },
                    "success_episode_mean_z": {
                        str(episode_id): mean_z_values[episode_id]
                        for episode_id in sorted(mean_z_values)
                    },
                    "committed_trajectory_id": None,
                    "committed_episode_id": None,
                    "successful_retry_batch": None,
                    "above_fraction": None,
                    "mean_z": None,
                    "executed_steps": 0,
                    "candidate_window_count": 0,
                    "full_h_valid_window_count": 0,
                    "committed_window_count": 0,
                    "committed_window_ids": [],
                    "committed_trajectory_count": 0,
                    "committed_trajectory_ids": [],
                    "committed_episode_ids": [],
                    "committed_trajectories": [],
                    "all_committed_window_count": 0,
                    "all_committed_window_ids": [],
                }
                if len(successful) < config.successful_trajectories_per_gamma:
                    raise RuntimeError(
                        f"round {round_i} has only {len(successful)} ranked "
                        f"commit-capable terminal SUCCESS trajectories for "
                        f"gamma={gamma:g}, fewer than requested "
                        f"{config.successful_trajectories_per_gamma}; "
                        "round not committed"
                    )
                for rank, committed_episode in enumerate(
                    successful[:config.successful_trajectories_per_gamma]
                ):
                    episode_id = int(committed_episode["episode"])
                    trajectory_id = _successful_trajectory_id(
                        round_i, gamma, episode_id
                    )
                    committed_trajectory_ids.append(trajectory_id)
                    detail["committed_trajectory_ids"].append(trajectory_id)
                    detail["committed_episode_ids"].append(episode_id)
                    trajectory_detail: dict[str, Any] = {
                        "rank": rank,
                        "trajectory_id": trajectory_id,
                        "episode_id": episode_id,
                        "successful_retry_batch": int(
                            committed_episode["retry_batch"]
                        ),
                        "above_fraction": above_fractions[episode_id],
                        "mean_z": mean_z_values[episode_id],
                        "executed_steps": 0,
                        "candidate_window_count": 0,
                        "full_h_valid_window_count": 0,
                        "committed_window_count": 0,
                        "committed_window_ids": [],
                    }
                    executed_steps = committed_episode["executed_steps"]
                    trajectory_detail["executed_steps"] = len(executed_steps)
                    if executed_steps:
                        horizons = {
                            int(step_record["plan_horizon"])
                            for step_record in executed_steps
                        }
                        if len(horizons) != 1:
                            raise RuntimeError(
                                "executed candidates changed horizon within one "
                                "successful trajectory"
                            )
                        horizon = horizons.pop()
                        action_shapes = {
                            tuple(step_record["action"].shape)
                            for step_record in executed_steps
                        }
                        if len(action_shapes) != 1:
                            raise RuntimeError(
                                "executed action shape changed within one "
                                "successful trajectory"
                            )
                        window_count = len(executed_steps)
                        trajectory_detail["candidate_window_count"] = (
                            window_count
                        )
                        for start in range(window_count):
                            context_record = executed_steps[start]
                            (
                                context_window,
                                actual_window,
                                candidate_window,
                                loss_mask,
                                result,
                            ) = reverify_executed_suffix(
                                executed_steps, start, horizon, gamma,
                            )
                            valid_horizon = len(actual_window)
                            paired_window_base = paired_executed_suffix_base(
                                executed_steps, start, horizon,
                            )
                            commit_reverify_queries += 1
                            if result.error:
                                continue
                            if result.valid:
                                trajectory_detail[
                                    "full_h_valid_window_count"
                                ] += 1
                                commit_reverify_valid += 1
                            if not (result.valid and result.progress_eligible):
                                continue
                            commit_reverify_progress += 1
                            with torch.no_grad():
                                task_feature = None
                                if (
                                    config.acquisition_feature == "task_angular"
                                    or config.coverage_replay == "circular_voronoi"
                                ):
                                    task_feature = _task_angular_features(
                                        task, context_window,
                                        actual_window.unsqueeze(0), gamma,
                                    )
                                if config.acquisition_feature == "task_angular":
                                    reconstructed_feature = task_feature
                                    reconstructed_gp = gp_by_gamma[gamma]
                                else:
                                    reconstructed_feature = acquisition_policy.embed(
                                        context_window.unsqueeze(0),
                                        candidate_window.unsqueeze(0),
                                        base=(
                                            paired_window_base.unsqueeze(0)
                                            if paired_window_base is not None
                                            else None
                                        ),
                                    )
                                    if (
                                        config.gp_reference_mode in {
                                            "cumulative_success_per_gamma_frozen_phi_exact",
                                            "sliding_success_per_gamma_frozen_phi",
                                            "sliding_success_per_gamma_current_phi",
                                        }
                                    ):
                                        reconstructed_gp = gp_by_gamma[gamma]
                                    else:
                                        if shared_gp is None:
                                            raise RuntimeError(
                                                "learned_phi acquisition GP was "
                                                "not initialized"
                                            )
                                        reconstructed_gp = shared_gp
                                acquisition_sigma = float(
                                    reconstructed_gp.sigma(
                                        reconstructed_feature
                                    )[0].detach().cpu()
                                )
                            window_id = _successful_window_id(
                                trajectory_id, start
                            )
                            committed_window_ids.append(window_id)
                            trajectory_detail["committed_window_ids"].append(
                                window_id
                            )
                            detail["all_committed_window_ids"].append(window_id)
                            committed_records.append(QueryRecord(
                                round_i, gamma, episode_id,
                                int(context_record["context_id"]),
                                context_window.detach().cpu(),
                                candidate_window.detach().cpu(),
                                result,
                                acquisition_sigma=acquisition_sigma,
                                flow_base=(
                                    paired_window_base.detach().cpu()
                                    if paired_window_base is not None else None
                                ),
                                acquisition_feature=(
                                    task_feature[0].detach().cpu()
                                    if task_feature is not None else None
                                ),
                                replay_eligible=True,
                                retry_batch=int(
                                    committed_episode["retry_batch"]
                                ),
                                replica=int(committed_episode["replica"]),
                                trajectory_id=trajectory_id,
                                window_id=window_id,
                                window_start=start,
                                valid_horizon=valid_horizon,
                                loss_mask=loss_mask.detach().cpu(),
                            ))
                    trajectory_detail["committed_window_count"] = len(
                        trajectory_detail["committed_window_ids"]
                    )
                    detail["committed_trajectories"].append(
                        trajectory_detail
                    )
                    if rank == 0:
                        detail["committed_trajectory_id"] = trajectory_id
                        detail["committed_episode_id"] = episode_id
                        detail["successful_retry_batch"] = trajectory_detail[
                            "successful_retry_batch"
                        ]
                        detail["above_fraction"] = trajectory_detail[
                            "above_fraction"
                        ]
                        detail["mean_z"] = trajectory_detail["mean_z"]
                        detail["executed_steps"] = trajectory_detail[
                            "executed_steps"
                        ]
                        detail["candidate_window_count"] = trajectory_detail[
                            "candidate_window_count"
                        ]
                        detail["full_h_valid_window_count"] = trajectory_detail[
                            "full_h_valid_window_count"
                        ]
                        detail["committed_window_count"] = trajectory_detail[
                            "committed_window_count"
                        ]
                        detail["committed_window_ids"] = list(
                            trajectory_detail["committed_window_ids"]
                        )
                detail["committed_trajectory_count"] = len(
                    detail["committed_trajectory_ids"]
                )
                detail["all_committed_window_count"] = len(
                    detail["all_committed_window_ids"]
                )
                successful_executed_commit_by_gamma[f"{gamma:.9g}"] = detail
            if (
                config.gp_reference_mode in {
                    "cumulative_success_per_gamma_frozen_phi_exact",
                    "sliding_success_per_gamma_frozen_phi",
                    "sliding_success_per_gamma_current_phi",
                }
            ):
                for gamma in map(float, config.gammas):
                    new_count = sum(
                        row.gamma == gamma for row in committed_records
                    )
                    if new_count < 1:
                        raise RuntimeError(
                            f"round {round_i} produced no commit-capable "
                            f"SUCCESS window for gamma={gamma:g}; round not committed"
                        )
                    if (
                        config.gp_reference_mode
                        == "cumulative_success_per_gamma_frozen_phi_exact"
                    ):
                        total_count = sum(
                            row.gamma == gamma for row in gp_evidence
                        ) + new_count
                        if total_count > config.gp_exact_max_rows_per_gamma:
                            raise RuntimeError(
                                f"round {round_i} exact GP support for gamma={gamma:g} "
                                f"would exceed gp_exact_max_rows_per_gamma="
                                f"{config.gp_exact_max_rows_per_gamma}; round not committed"
                            )
            archive.extend(committed_records)
            if (
                config.gp_reference_mode == "round1_fixed_frozen_phi"
                and round_i == 1
            ):
                round1_gp_candidates.extend(committed_records)
            if config.gp_reference_mode in {
                "cumulative_accepted_frozen_phi",
                "cumulative_success_per_gamma_frozen_phi_exact",
                "sliding_success_per_gamma_frozen_phi",
                "sliding_success_per_gamma_current_phi",
            }:
                gp_evidence.extend(committed_records)
        eligible_positive = [
            row for row in _recent(
                archive, round_i, config.replay_rounds, positive=True
            )
            if row.replay_eligible
        ]
        eligible_positive_total = len(eligible_positive)
        if config.replay_selector == "sigma_top":
            eligible_positive = _top_uncertainty_by_round(
                eligible_positive, config.replay_top_fraction,
                group_by_gamma=(
                    config.archive_rule == "successful_executed_windows"
                ),
            )
        elif config.replay_selector == "context_kcenter":
            eligible_positive = _context_kcenter_replay(
                eligible_positive, policy, config.replay_context_quota,
                config.replay_action_weight,
            )
        elif config.replay_selector == "cluster_balanced":
            replay_budget = (
                config.inner_steps * config.batch_size
                if config.inner_steps is not None else len(eligible_positive)
            )
            eligible_positive = _cluster_balanced_replay(
                eligible_positive, policy, config.replay_cluster_count,
                config.replay_action_weight, replay_budget, torch_rng,
            )
        negative = _recent(archive, round_i, config.replay_rounds, positive=False)
        update_started = time.perf_counter()
        update = _update(policy, task, optimizer, eligible_positive, negative,
                         config, torch_rng)
        update_s = time.perf_counter() - update_started
        if (config.adaptive_beta and score_pools
                and any(float(pool.max() - pool.min()) > 1.0e-4
                        for pool in score_pools)):
            beta = calibrate_fixed_beta(score_pools, config.ess_target)
        row = {
            "round": round_i,
            "queries": sum(record.round == round_i for record in archive),
            "positives": sum(record.round == round_i and record.verification.valid
                             for record in archive),
            "negatives": sum(
                record.round == round_i and not record.verification.valid
                for record in archive
            ),
            "replay_accepted_positives": sum(
                record.round == round_i and record.replay_eligible
                for record in archive
            ),
            "verifier_queries": verifier_queries,
            "verifier_positives": verifier_positives,
            "gp_buffer": len(gp_rows),
            "gp_buffer_by_gamma": {
                f"{float(gamma):.9g}": sum(
                    row.gamma == float(gamma) for row in gp_rows
                )
                for gamma in config.gammas
            },
            "gp_sliding_row_selector": (
                config.gp_sliding_row_selector
                if config.gp_reference_mode in {
                    "sliding_success_per_gamma_frozen_phi",
                    "sliding_success_per_gamma_current_phi",
                    "sliding_positive_per_gamma_frozen_phi",
                    "sliding_positive_per_gamma_current_phi",
                    "sliding_executed_positive_per_gamma_frozen_phi",
                }
                else None
            ),
            "gp_window_start_by_gamma": _gp_window_start_diagnostics(
                gp_rows, config.gammas,
            ),
            "gp_reference_hash_by_gamma": {
                f"{float(gamma):.9g}": (
                    _record_identity_hash([
                        record for record in gp_rows
                        if record.gamma == float(gamma)
                    ])
                    if any(
                        record.gamma == float(gamma) for record in gp_rows
                    )
                    else None
                )
                for gamma in config.gammas
            },
            "gp_feature_hash_by_gamma": gp_feature_hash_by_gamma,
            "acquisition_representation_hash": (
                acquisition_representation_hash
            ),
            "gp_reference_hash": frozen_gp_hash,
            "gp_anchor_count": gp_anchor_count,
            "gp_adaptive_count": gp_adaptive_count,
            "gp_evidence_count": len(gp_evidence),
            "successful_executed_commit_by_gamma": (
                successful_executed_commit_by_gamma
            ),
            "committed_trajectory_ids": committed_trajectory_ids,
            "committed_window_ids": committed_window_ids,
            "commit_reverify_queries": commit_reverify_queries,
            "retry_capability_reverify_queries": (
                retry_capability_reverify_queries
            ),
            "commit_reverify_valid": commit_reverify_valid,
            "commit_reverify_progress": commit_reverify_progress,
            "replay_positive_total": eligible_positive_total,
            "replay_positives": len(eligible_positive),
            "replay_selector": config.replay_selector,
            "beta": beta_used,
            "beta_next": beta,
            "ESS_over_K": float(np.mean(ess_values)) if ess_values else 1.0,
            "marginal_ESS_over_K": (
                float(np.mean(marginal_ess_values))
                if marginal_ess_values else 1.0
            ),
            "sigma_pool_mean": (
                float(np.mean(sigma_pool_means)) if sigma_pool_means else None
            ),
            "sigma_selected_mean": (
                float(np.mean(sigma_selected_means))
                if sigma_selected_means else None
            ),
            "uncertainty_uplift": (
                float(np.mean(sigma_selected_means) - np.mean(sigma_pool_means))
                if sigma_pool_means else None
            ),
            "execution_ESS": (
                float(np.mean(execution_ess_values))
                if execution_ess_values else None
            ),
            "NVP": nvp,
            "success": statuses.count("SUCCESS"),
            "collision": statuses.count("COLLISION"),
            "oob": statuses.count("OOB"),
            "timeout": statuses.count("TIMEOUT"),
            "early_cutoff": statuses.count("EARLY_CUTOFF"),
            "retry_batches_by_gamma": {
                f"{gamma:.9g}": count
                for gamma, count in retry_batches_by_gamma.items()
            },
            "attempted_episode_count": len(episodes),
            "round_success_commit_complete": (
                config.archive_rule == "successful_executed_windows"
            ),
            "gp_refit_s": gp_refit_s,
            "gather_s": gather_s,
            "verifier_dispatch_s": verifier_dispatch_s,
            "verifier_jobs": verifier_jobs,
            "verifier_candidates": verifier_candidates,
            "update_s": update_s,
            "round_total_s": time.perf_counter() - round_started,
            **update,
        }
        round_rows.append(row)
        if round_callback is not None:
            round_callback(row)
        torch.save({"round": round_i, "model": policy.state_dict(),
                    "config": asdict(config), "pretrained": False},
                   output_dir / f"checkpoint_{round_i:03d}.pt")
        with (output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    manifest = {
        "status": "SAFE_FLOW_EXPANSION_COMPLETE",
        "config": asdict(config),
        "rbf_lengthscale": lengthscale,
        "final_beta": beta,
        "D": len(archive),
        "D_plus": sum(row.verification.valid for row in archive),
        "D_minus": sum(not row.verification.valid for row in archive),
        "D_replay_accepted": sum(row.replay_eligible for row in archive),
        "rounds": round_rows,
        "optimizer_scope": {
            "mode": effective_optimizer_scope,
            "trainable_parameter_names": trainable_parameter_names,
            "trainable_parameter_count": trainable_parameter_count,
        },
        "semantics": {
            "GP": (
                "separate per-gamma RBF posteriors over recent full-H verifier-"
                "positive task angular descriptors; no cross-gamma uncertainty"
                if config.acquisition_feature == "task_angular"
                else {
                    "recent_current_phi": (
                        "recent full-H verifier positives embedded by current phi"
                    ),
                    "round1_fixed_frozen_phi": (
                        "fixed round-1 accepted selected-B evidence embedded by "
                        "frozen pretrained phi"
                    ),
                    "cumulative_accepted_frozen_phi": (
                        "bounded cumulative accepted selected-B evidence coreset "
                        "embedded by frozen pretrained phi"
                    ),
                    "cumulative_success_per_gamma_frozen_phi_exact": (
                        "separate exact per-gamma posteriors over every previously "
                        "committed successful executed window, embedded by frozen "
                        "pretrained endpoint phi; no thinning"
                    ),
                    "sliding_success_per_gamma_frozen_phi": (
                        "separate per-gamma bounded posteriors over committed "
                        "successful executed windows, embedded by frozen "
                        "pretrained endpoint phi"
                    ),
                    "sliding_success_per_gamma_current_phi": (
                        "separate per-gamma bounded posteriors over committed "
                        "successful executed windows, re-embedded at each round "
                        "by the current model endpoint phi"
                    ),
                    "sliding_positive_per_gamma_frozen_phi": (
                        "separate per-gamma bounded posteriors over resolved "
                        "exact-positive preterminal selected-B queries, embedded "
                        "by frozen pretrained endpoint phi"
                    ),
                    "sliding_positive_per_gamma_current_phi": (
                        "separate per-gamma bounded posteriors over resolved "
                        "exact-positive preterminal selected-B queries, re-embedded "
                        "at each round by current endpoint phi"
                    ),
                    "sliding_executed_positive_per_gamma_frozen_phi": (
                        "separate per-gamma bounded posteriors over exact-positive "
                        "plans actually selected for execution, embedded by frozen "
                        "pretrained endpoint phi"
                    ),
                }[config.gp_reference_mode]
            ),
            "acquisition_feature": config.acquisition_feature,
            "replay_augmentation": config.replay_augmentation,
            "D": ({
                "all_queries": "all successful selected-B verifier queries",
                "executed_only": "only the verifier-positive candidate selected for execution",
                "executed_plus_nvp_negative": (
                    "each executed positive plus at most one lowest min-cost-score "
                    "full-verifier reject at terminal NVP; the score includes the "
                    "declared execution step-margin weight and the reject is not "
                    "marked executed"
                ),
                "successful_executed_windows": (
                    "all executed suffix starts reconstructed from "
                    f"{config.successful_trajectories_per_gamma} distinct "
                    "commit-capable successful trajectories per gamma and round, "
                    "selected "
                    f"by {config.successful_trajectory_selector}; each available "
                    "suffix of length <=H is reverified and must be valid with "
                    "positive progress, then zero-padded only for fixed-shape "
                    "embedding with CFM loss masked to its valid horizon"
                ),
                "preterminal_resolved_queries": (
                    "every resolved selected-B query observed along each lineage "
                    "before NVP, success, collision, or timeout; original sampled "
                    "candidate and authoritative flow base are retained"
                ),
            }[config.archive_rule]),
            "D_plus": (
                "reverified valid positive-progress executed <=H suffixes from "
                "commit-capable terminal successes"
                if config.archive_rule == "successful_executed_windows"
                else "full-H verifier positives"
            ),
            "replay_acceptance": (
                "successful-window archive admission overrides per-query replay "
                "acceptance: only reverified valid positive-progress executed "
                "<=H suffixes from committed terminal successes are replayed"
                if config.archive_rule == "successful_executed_windows"
                else (
                    "every full-H verifier-positive selected-B query; independent "
                    "of progress and task target gates"
                    if config.replay_acceptance == "safety_valid"
                    else (
                        "legacy controller-coupled acceptance: validity alone when "
                        "no task target gate is configured; otherwise validity, "
                        "progress, and the active target gate"
                    )
                )
            ),
            "execution": ("positive-per-knot-x verifier positives ranked by "
                          f"{config.execution_rule}; nominal one-step H_P is diagnostic only"),
            "failure": "NVP terminates that episode; no expert fallback",
            "replay": (
                "recent D+ selected by "
                f"{config.replay_selector}"
                + (
                    f" (top fraction={config.replay_top_fraction})"
                    if config.replay_selector == "sigma_top" else ""
                )
                + (
                    " (per-context quota="
                    f"{config.replay_context_quota}, action weight="
                    f"{config.replay_action_weight})"
                    if config.replay_selector == "context_kcenter" else ""
                )
                + (
                    " (clusters/gamma="
                    f"{config.replay_cluster_count}, action weight="
                    f"{config.replay_action_weight})"
                    if config.replay_selector == "cluster_balanced" else ""
                )
                + "; then gamma-lineage-context-query equal mass"
                + (
                    " multiplied by within-gamma periodic circular Voronoi "
                    "arc mass without removing replay rows"
                    if config.coverage_replay == "circular_voronoi" else ""
                )
            ),
            "negative_replay": ({
                "all_queries": "recent selected-B full-verifier rejects",
                "executed_only": "none: temporary non-executed queries never enter D",
                "executed_plus_nvp_negative": (
                    "at most one lowest min-cost-score full-verifier reject per NVP "
                    "rollout; it is a counterfactual D- row, never an executed or GP row"
                ),
                "successful_executed_windows": (
                    "none: NVP, collision, OOB, and timeout trajectories are "
                    "discarded in full"
                ),
                "preterminal_resolved_queries": (
                    "all resolved exact-negative selected-B queries remain truthful "
                    "D- rows for optional signed-alpha replay; none enter the GP"
                ),
            }[config.archive_rule]),
            "candidate_perturbation": ("one Gaussian action vector per candidate applied "
                                       f"with scope={config.candidate_perturb_scope} before "
                                       "embedding and verification"),
            "flow_base_sampling": (
                "expansion-only initial flow latent "
                rf"x0~N(0,{config.flow_base_std ** 2:g}I), "
                "applied before flow integration; raw evaluation remains unit normal"
            ),
            "representation": (
                "paired noised phi_s using each sampled plan's stored initial CFM base"
                if config.paired_noised_representation
                else "endpoint-only phi_s without a paired CFM base"
            ),
            "target_gate": (
                "inactive"
                if config.target_gate_start_round is None
                else (
                    (
                        "task target_eligible gates execution from round "
                        f"{config.target_gate_start_round}; safety_valid replay "
                        "remains independent; safety label unchanged"
                    )
                    if config.replay_acceptance == "safety_valid"
                    else (
                        "task target_eligible gates execution and replay acceptance "
                        f"from round {config.target_gate_start_round}; "
                        "safety label unchanged"
                    )
                )
            ),
            "gp_reference": (
                "rolling recent positives embedded by current phi"
                if config.gp_reference_mode == "recent_current_phi"
                else (
                    "one deterministic gamma-balanced cap of round-1 accepted "
                    "positives, reused with frozen pretrained phi"
                    if config.gp_reference_mode == "round1_fixed_frozen_phi"
                    else (
                        (
                            "exact per-gamma cumulative successful-window support "
                            "through the previous round, with frozen pretrained "
                            "endpoint phi and no thinning"
                        )
                        if config.gp_reference_mode
                        == "cumulative_success_per_gamma_frozen_phi_exact"
                        else (
                            (
                                "per-gamma bounded preterminal exact-positive "
                                "selected-B evidence through the previous round, "
                                "re-embedded by current endpoint phi"
                                if config.gp_reference_mode
                                == "sliding_positive_per_gamma_current_phi"
                                else (
                                    "per-gamma bounded preterminal exact-positive "
                                    "selected-B evidence through the previous round, "
                                    "with frozen pretrained endpoint phi"
                                    if config.gp_reference_mode
                                    == "sliding_positive_per_gamma_frozen_phi"
                                    else (
                                "per-gamma bounded cumulative successful-window "
                                "evidence through the previous round, re-embedded "
                                "by current endpoint phi"
                                if config.gp_reference_mode
                                == "sliding_success_per_gamma_current_phi"
                                else (
                                    "per-gamma bounded cumulative successful-window "
                                    "evidence through the previous round, with "
                                    "frozen pretrained endpoint phi"
                                    if config.gp_reference_mode
                                    == "sliding_success_per_gamma_frozen_phi"
                                    else (
                                        (
                                            "per-gamma bounded exact-positive "
                                            "actually executed plans through the "
                                            "previous round, with frozen pretrained "
                                            "endpoint phi"
                                        )
                                        if config.gp_reference_mode == (
                                            "sliding_executed_positive_per_gamma_"
                                            "frozen_phi"
                                        )
                                        else (
                                            "bounded cumulative frozen-phi coreset "
                                            "with equal gamma capacity, immutable "
                                            "round-1 anchors, and deterministic "
                                            "farthest-first adaptive discoveries"
                                        )
                                    )
                                )
                                    )
                                )
                            )
                        )
                    )
                )
            ),
            "acquisition_diagnostics": (
                "ESS_over_K is the mean sequential conditional normalized ESS "
                "over B without-replacement draws; marginal_ESS_over_K is the "
                "first-draw Gibbs ESS/K computed from marginal sigma_K; uplift "
                "compares selected marginal sigma with the all-K marginal mean"
            ),
            "successful_executed_windows": (
                "inactive"
                if config.archive_rule != "successful_executed_windows"
                else (
                    "after gathering, choose "
                    f"{config.successful_trajectories_per_gamma} distinct terminal "
                    "SUCCESS trajectories per gamma, ranked by "
                    "successful_trajectory_selector; reconstruct all L suffix starts "
                    "from consecutive actually executed first actions; reverify "
                    "only the available <=H states and commit valid "
                    "positive-progress suffixes; zero-pad to H for frozen-phi "
                    "embedding and mask CFM loss to the available actions; "
                    "recompute their sigma "
                    "against the round-start GP; selection is admission ranking "
                    "only and never changes validity or NVP; retry whole batches "
                    "until every gamma has the requested number of commit-capable "
                    "terminal SUCCESS trajectories, with "
                    "top-fraction replay grouped by (round,gamma) and complete "
                    "sigma-boundary ties retained"
                )
            ),
            "episode_retry": (
                "inactive"
                if config.archive_rule != "successful_executed_windows"
                else (
                    f"up to {config.max_retry_batches} whole-episode batches per "
                    f"gamma, {config.parallel_episodes} episodes per batch; stop "
                    "a gamma after accumulating "
                    f"{config.successful_trajectories_per_gamma} commit-capable "
                    "terminal SUCCESS trajectories; no step retry; exhaustion "
                    "aborts the round "
                    "before archive, GP, optimizer, or checkpoint mutation; "
                    "counter-keyed gather RNG coordinates are "
                    "(seed,round,retry_batch,replica,step) and deliberately "
                    "exclude gamma"
                )
            ),
        },
    }
    if config.gp_reference_mode == "cumulative_accepted_frozen_phi":
        per_gamma_cap, anchor_cap, ingress_cap = _cumulative_gp_quotas(config)
        final_active = [
            row
            for gamma in sorted(cumulative_anchors)
            for row in cumulative_anchors[gamma] + cumulative_adaptive[gamma]
        ]
        manifest["gp_reference"] = {
            "mode": config.gp_reference_mode,
            "round": None,
            "through_round": max(0, config.rounds - 1),
            "count": len(final_active),
            "anchor_count": sum(map(len, cumulative_anchors.values())),
            "adaptive_count": sum(map(len, cumulative_adaptive.values())),
            "evidence_count": len(gp_evidence),
            "identity_sha256": frozen_gp_hash,
            "frozen_phi": True,
            "active_cap": config.gp_buffer_cap,
            "per_gamma_total_cap": per_gamma_cap,
            "per_gamma_anchor_cap": anchor_cap,
            "per_gamma_adaptive_cap": per_gamma_cap - anchor_cap,
            "per_gamma_ingress_cap": ingress_cap,
        }
    elif (
        config.gp_reference_mode
        == "cumulative_success_per_gamma_frozen_phi_exact"
    ):
        active_rows = list(gp_rows)
        manifest["gp_reference"] = {
            "mode": config.gp_reference_mode,
            "posterior_through_round": max(0, config.rounds - 1),
            "evidence_through_round": config.rounds,
            "count": len(active_rows),
            "count_by_gamma": {
                f"{float(gamma):.9g}": sum(
                    row.gamma == float(gamma) for row in active_rows
                )
                for gamma in config.gammas
            },
            "evidence_count": len(gp_evidence),
            "evidence_count_by_gamma": {
                f"{float(gamma):.9g}": sum(
                    row.gamma == float(gamma) for row in gp_evidence
                )
                for gamma in config.gammas
            },
            "identity_sha256": frozen_gp_hash,
            "identity_sha256_by_gamma": {
                f"{float(gamma):.9g}": (
                    _record_identity_hash([
                        row for row in active_rows
                        if row.gamma == float(gamma)
                    ])
                    if any(
                        row.gamma == float(gamma) for row in active_rows
                    )
                    else None
                )
                for gamma in config.gammas
            },
            "frozen_phi": True,
            "paired_noised_representation": (
                config.paired_noised_representation
            ),
            "exact": True,
            "thinning": "none",
            "hard_max_rows_per_gamma": config.gp_exact_max_rows_per_gamma,
            "legacy_gp_buffer_cap_used": False,
            "legacy_gp_buffer_cap_ignored_value": config.gp_buffer_cap,
            "window_source": (
                "every reverified available <=H suffix of actually executed "
                "controls from "
                f"{config.successful_trajectories_per_gamma} distinct "
                "commit-capable terminal SUCCESS trajectories per gamma/round; "
                "zero-padded for fixed-shape phi with valid-horizon CFM masks"
            ),
        }
    elif config.gp_reference_mode in {
        "sliding_success_per_gamma_frozen_phi",
        "sliding_success_per_gamma_current_phi",
        "sliding_positive_per_gamma_frozen_phi",
        "sliding_positive_per_gamma_current_phi",
        "sliding_executed_positive_per_gamma_frozen_phi",
    }:
        active_rows = list(gp_rows)
        per_gamma_cap = config.gp_buffer_cap // len(config.gammas)
        manifest["gp_reference"] = {
            "mode": config.gp_reference_mode,
            "posterior_through_round": max(0, config.rounds - 1),
            "evidence_through_round": config.rounds,
            "count": len(active_rows),
            "count_by_gamma": {
                f"{float(gamma):.9g}": sum(
                    row.gamma == float(gamma) for row in active_rows
                )
                for gamma in config.gammas
            },
            "evidence_count": len(gp_evidence),
            "evidence_count_by_gamma": {
                f"{float(gamma):.9g}": sum(
                    row.gamma == float(gamma) for row in gp_evidence
                )
                for gamma in config.gammas
            },
            "identity_sha256": frozen_gp_hash,
            "identity_sha256_by_gamma": {
                f"{float(gamma):.9g}": (
                    _record_identity_hash([
                        row for row in active_rows
                        if row.gamma == float(gamma)
                    ])
                    if any(
                        row.gamma == float(gamma) for row in active_rows
                    )
                    else None
                )
                for gamma in config.gammas
            },
            "frozen_phi": (
                config.gp_reference_mode in {
                    "sliding_success_per_gamma_frozen_phi",
                    "sliding_positive_per_gamma_frozen_phi",
                    "sliding_executed_positive_per_gamma_frozen_phi",
                }
            ),
            "representation": (
                "pretrained_phi0"
                if config.gp_reference_mode in {
                    "sliding_success_per_gamma_frozen_phi",
                    "sliding_positive_per_gamma_frozen_phi",
                    "sliding_executed_positive_per_gamma_frozen_phi",
                }
                else "current_round_phi_reembedded_before_gather"
            ),
            "paired_noised_representation": (
                config.paired_noised_representation
            ),
            "paired_success_window_base": (
                (
                    "original per-query sampled flow base paired with the exact "
                    "selected-B candidate"
                    if config.archive_rule in {
                        "preterminal_resolved_queries",
                        "executed_plus_nvp_negative",
                    }
                    else (
                        "concatenated actual first-action generator latents from "
                        "successive closed-loop decisions; terminal padding is zero"
                    )
                )
                if config.paired_noised_representation else None
            ),
            "exact": False,
            "active_cap": config.gp_buffer_cap,
            "cap_scope": "equal_per_gamma",
            "per_gamma_cap": per_gamma_cap,
            "reference_rounds": config.gp_reference_rounds,
            "row_selector": config.gp_sliding_row_selector,
            "eviction": (
                (
                    "legacy highest-window-start FIFO tail"
                    if config.gp_sliding_row_selector == "fifo_tail"
                    else (
                        "equal trajectory/lineage quotas with uniformly "
                        "spaced departure-to-tail window starts"
                    )
                )
                + "; cumulative evidence log retained"
            ),
            "active_window_start_by_gamma": _gp_window_start_diagnostics(
                active_rows, config.gammas,
            ),
            "window_source": (
                (
                    "every resolved exact-positive selected-B query observed before "
                    "each lineage terminal event, retaining its original H-plan "
                    "and paired flow base"
                    if config.archive_rule == "preterminal_resolved_queries"
                    else (
                        (
                            "every exact-positive H-plan actually selected for "
                            "closed-loop execution before each lineage terminal "
                            "event; terminal NVP negatives never enter the GP"
                        )
                        if config.archive_rule == "executed_plus_nvp_negative"
                        else (
                            "every reverified available <=H suffix of actually "
                            "executed controls from "
                            f"{config.successful_trajectories_per_gamma} distinct "
                            "commit-capable terminal SUCCESS trajectories per "
                            "gamma/round; zero-padded for fixed-shape phi with "
                            "valid-horizon CFM masks"
                        )
                    )
                )
            ),
        }
    else:
        manifest["gp_reference"] = {
            "mode": config.gp_reference_mode,
            "round": (
                1 if config.gp_reference_mode == "round1_fixed_frozen_phi" else None
            ),
            "count": len(frozen_gp_rows or []),
            (
                "source_window_count"
                if config.archive_rule == "successful_executed_windows"
                else "source_query_count"
            ): len(round1_gp_candidates),
            "identity_sha256": frozen_gp_hash,
            "frozen_phi": config.gp_reference_mode == "round1_fixed_frozen_phi",
        }
    torch.save(archive, output_dir / "query_archive.pt")
    if config.gp_reference_mode in {
        "cumulative_accepted_frozen_phi",
        "cumulative_success_per_gamma_frozen_phi_exact",
        "sliding_success_per_gamma_frozen_phi",
        "sliding_success_per_gamma_current_phi",
        "sliding_positive_per_gamma_frozen_phi",
        "sliding_positive_per_gamma_current_phi",
        "sliding_executed_positive_per_gamma_frozen_phi",
    }:
        torch.save(gp_evidence, output_dir / "gp_evidence.pt")
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2,
                                                          allow_nan=False) + "\n")
    return manifest


def run_safe_expansion(
    policy: ExpansionPolicy,
    task: ExpansionTask,
    output_dir: str | Path,
    *,
    config: ExpansionConfig = ExpansionConfig(),
    calibration_features: torch.Tensor | None = None,
    event_callback: Callable[[dict[str, Any]], None] | None = None,
    round_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run expansion with ordered CPU-parallel verification when requested."""
    config.validate()
    with _OrderedVerifier(task, config.verifier_workers) as verifier:
        return _run_safe_expansion_impl(
            policy,
            task,
            output_dir,
            config=config,
            calibration_features=calibration_features,
            event_callback=event_callback,
            round_callback=round_callback,
            verifier=verifier,
        )
