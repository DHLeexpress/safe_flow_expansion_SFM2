"""From-scratch HP100 SFM pretraining with ID-only checkpoint promotion.

The loader keeps each freshly collected ``[N,32,100]`` Hp file memory-mapped.
It stores only ten integer source indices per row and gathers the corresponding
newest-to-oldest history in ``Dataset.__getitem__``.  It therefore never
materializes a second ``[N,10,32,100]`` dataset.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
import hashlib
import importlib
import inspect
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_hp100_features as HPF
import sfm_hp100_dynamics as DYN
import sfm_hp100_history as HPH
import sfm_protocol as SP
import sfm_scene as SS
import stage2_hp100_data as DATA


SUCCESSFUL_LINEAGES_PER_GAMMA = 500
SCHEMA_VERSION = DATA.SCHEMA_VERSION
HISTORY_LENGTH = 10
DEFAULT_EPOCHS = 120
DEFAULT_BATCH = 256
DEFAULT_LR = 3.0e-4
DEFAULT_WEIGHT_DECAY = 1.0e-4
DEFAULT_WARMUP = 5
DEFAULT_CHECKPOINT_EVERY = 1
DEFAULT_VAL_BATCH = 512
DEFAULT_SEED = 20260720
DEFAULT_NUM_WORKERS = 2
DEFAULT_SCREEN_M = 10
DEFAULT_SCREEN_TOP = 3
DEFAULT_SCREEN_FINALISTS = 2
DEFAULT_CONFIRM_M = 50
DEFAULT_SCREEN_EP0 = SP.PRETRAIN_GATE_EP0
DEFAULT_CONFIRM_EP0 = SP.PRETRAIN_CONFIRM_EP0
DEFAULT_VALIDATION_NOISE_SEED = 41017
DEFAULT_SCREEN_NOISE_SEED = 2_026_080_2
DEFAULT_CONFIRM_NOISE_SEED = 2_026_080_3
DEFAULT_VERIFIER_WORKERS = 32


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gamma_path(dataset, gamma) -> Path:
    return Path(dataset).resolve() / f"sfm_hp100_windows_g{float(gamma)}.pt"


def _manifest(dataset) -> tuple[dict, Path, str]:
    path = Path(dataset).resolve() / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing authenticated HP100 manifest: {path}")
    payload = json.loads(path.read_text())
    if payload.get("status") != "HP100_ID_DATASET_COMPLETE":
        raise ValueError("HP100 dataset manifest is not complete")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"HP100 schema changed: {payload.get('schema_version')} != {SCHEMA_VERSION}"
        )
    return payload, path, sha256_file(path)


def _manifest_files(manifest: dict) -> dict[float, dict]:
    rows = {}
    for row in manifest.get("files", []):
        gamma = float(row["gamma"])
        if gamma in rows:
            raise ValueError(f"duplicate manifest file row for gamma {gamma}")
        rows[gamma] = row
    return rows


def _require_equal(label: str, actual, expected) -> None:
    expected = json.loads(json.dumps(expected))
    if actual != expected:
        raise ValueError(f"canonical HP100 manifest mismatch at {label}: {actual!r} != {expected!r}")


def _validate_canonical_manifest(manifest: dict, dataset_dir: Path) -> None:
    """Fail closed unless this is the one declared seven-gamma dataset."""
    _require_equal("canonical_full_run", manifest.get("canonical_full_run"), True)
    _require_equal(
        "role", manifest.get("role"),
        "successful SafeMPPI ID demonstrations for fresh Hp100 pretraining",
    )
    _require_equal(
        "total_successful_lineages",
        manifest.get("total_successful_lineages"),
        DATA.SUCCESSES_PER_GAMMA * len(SP.GAMMAS),
    )
    allocation = manifest.get("episode_allocation", {})
    _require_equal("episode_allocation.start", allocation.get("start"), DATA.EPISODE_START)
    _require_equal(
        "episode_allocation.successful_trajectories_per_gamma",
        allocation.get("successful_trajectories_per_gamma"),
        DATA.SUCCESSES_PER_GAMMA,
    )
    _require_equal(
        "episode_allocation.max_attempts_per_gamma",
        allocation.get("max_attempts_per_gamma"),
        DATA.MAX_ATTEMPTS_PER_GAMMA,
    )
    environment = manifest.get("environment", {})
    expected_environment = dict(
        n_ped=DATA.N_PED,
        ped_speed_range=list(DATA.PED_SPEED_RANGE),
        gammas=list(map(float, SP.GAMMAS)),
        horizon=DATA.HORIZON,
        T=DATA.T,
        goal=np.asarray(SS.GOAL, float).tolist(),
        task_bounds=[float(SS.TASK_LO), float(SS.TASK_HI)],
        pedestrian_radius=float(SS.R_PED),
        sensing_radius=float(SS.R_SENSE),
    )
    _require_equal("environment", environment, expected_environment)
    expert = manifest.get("expert", {})
    _require_equal("expert.name", expert.get("name"), DATA.HP100_EXPERT_NAME)
    _require_equal("expert.config", expert.get("config"), DATA.locked_expert_config())
    _require_equal(
        "expert.execution", expert.get("execution"),
        "CappedSafeMPPIAdapter using sfm_hp100_dynamics for internal and real "
        "steps; current-position tangent nominal geometry (predict_gain=0)",
    )
    _require_equal(
        "expert.supervised_target", expert.get("supervised_target"),
        DATA.SUPERVISED_TARGET,
    )
    _require_equal("target_contract", manifest.get("target_contract"), DATA.target_contract())
    feature = manifest.get("feature", {})
    _require_equal("feature.shape", feature.get("shape"), [32, 100])
    _require_equal("feature.dtype", feature.get("dtype"), "float32")
    _require_equal("feature.radial_bins", feature.get("radial_bins"), 100)
    _require_equal("feature.angular_bins", feature.get("angular_bins"), 32)
    _require_equal("feature.radial_pooling", feature.get("radial_pooling"), "none")
    _require_equal("feature.contract", feature.get("contract"), HPF.contract())
    _require_equal("feature.nominal_polytope_n_base", feature.get("nominal_polytope_n_base"), 16)
    _require_equal("feature.predict_gain", feature.get("predict_gain"), HPF.PREDICT_GAIN)
    _require_equal("feature.predict_tau", feature.get("predict_tau"), HPF.PREDICT_TAU)
    _require_equal("feature.velocity_aware", feature.get("velocity_aware"), False)
    _require_equal(
        "feature.current_position_tangent",
        feature.get("current_position_tangent"), True,
    )
    _require_equal("dynamics", manifest.get("dynamics"), DYN.contract())

    source_git = manifest.get("source_git", {})
    completion = manifest.get("source_completion_audit", {})
    _require_equal("source_git.clean", source_git.get("clean"), True)
    _require_equal("source_completion_audit.git", completion.get("git"), source_git)
    _require_equal("source_completion_audit.source_hashes_equal", completion.get("source_hashes_equal"), True)
    recorded_hashes = manifest.get("source_hashes")
    if not recorded_hashes or not manifest.get("runtime"):
        raise ValueError("canonical HP100 manifest lacks source/runtime provenance")
    for name, current in DATA._source_hashes().items():
        recorded = recorded_hashes.get(name)
        if not isinstance(recorded, dict) or recorded.get("sha256") != current["sha256"]:
            raise RuntimeError(
                f"canonical HP100 data-generation source differs at {name}"
            )

    file_rows = _manifest_files(manifest)
    gamma_device_map = manifest.get("parallelism", {}).get("gamma_device_map", {})
    _require_equal("files.gammas", sorted(file_rows), sorted(map(float, SP.GAMMAS)))
    _require_equal(
        "episode_allocation.terminal_ranges",
        allocation.get("terminal_ranges"),
        {str(gamma): file_rows[gamma].get("episode_range") for gamma in map(float, SP.GAMMAS)},
    )
    _require_equal(
        "total_context_rows", manifest.get("total_context_rows"),
        sum(int(row.get("windows", -1)) for row in file_rows.values()),
    )
    _require_equal(
        "total_eligible_windows", manifest.get("total_eligible_windows"),
        sum(int(row.get("eligible_windows", -1)) for row in file_rows.values()),
    )
    for gamma in map(float, SP.GAMMAS):
        row = file_rows[gamma]
        expected_device = gamma_device_map.get(str(gamma))
        if not expected_device or row.get("runtime", {}).get("requested_device") != expected_device:
            raise ValueError(f"gamma {gamma} worker runtime/device provenance is missing")
        _require_equal(f"files[{gamma}].n_traj", row.get("n_traj"), DATA.SUCCESSES_PER_GAMMA)
        successful = list(map(int, row.get("successful_episodes", [])))
        if (
            successful != sorted(successful)
            or len(successful) != DATA.SUCCESSES_PER_GAMMA
            or len(set(successful)) != len(successful)
        ):
            raise ValueError(f"gamma {gamma} does not declare 500 unique successful lineages")
        attempts = int(row.get("attempted_episodes", -1))
        if not DATA.SUCCESSES_PER_GAMMA <= attempts <= DATA.MAX_ATTEMPTS_PER_GAMMA:
            raise ValueError(f"gamma {gamma} attempt count is outside the canonical bounds")
        episode_range = row.get("episode_range")
        if episode_range != [DATA.EPISODE_START, attempts]:
            raise ValueError(f"gamma {gamma} episode range disagrees with its attempt count")
        rejected = list(map(int, row.get("rejected_episodes", [])))
        expected_ids = set(range(*episode_range))
        if (
            set(successful) & set(rejected)
            or set(successful) | set(rejected) != expected_ids
            or len(rejected) != attempts - DATA.SUCCESSES_PER_GAMMA
        ):
            raise ValueError(f"gamma {gamma} success/rejection ledger is not exhaustive")
        progress = dataset_dir / str(row.get("progress_file", ""))
        if not progress.is_file() or sha256_file(progress) != row.get("progress_sha256"):
            raise RuntimeError(f"gamma {gamma} progress artifact digest mismatch")
        progress_payload = json.loads(progress.read_text())
        _require_equal(
            f"progress[{gamma}].status",
            progress_payload.get("status"),
            "HP100_GAMMA_COLLECTION_COMPLETE",
        )
        _require_equal(
            f"progress[{gamma}].accepted_successes",
            progress_payload.get("accepted_successes"),
            DATA.SUCCESSES_PER_GAMMA,
        )
        _require_equal(
            f"progress[{gamma}].target_successes",
            progress_payload.get("target_successes"),
            DATA.SUCCESSES_PER_GAMMA,
        )
        _require_equal(
            f"progress[{gamma}].attempted_episodes",
            progress_payload.get("attempted_episodes"), attempts,
        )
        _require_equal(
            f"progress[{gamma}].episode_range",
            progress_payload.get("episode_range"), episode_range,
        )
        _require_equal(
            f"progress[{gamma}].data_file",
            progress_payload.get("data_file"), row.get("file"),
        )
        _require_equal(
            f"progress[{gamma}].data_sha256",
            progress_payload.get("data_sha256"), row.get("sha256"),
        )
        for key in (
            "windows", "eligible_windows", "excluded_all_rejected_windows",
            "excluded_weighted_h10_failed_windows",
        ):
            _require_equal(
                f"progress[{gamma}].{key}", progress_payload.get(key), row.get(key),
            )
        if int(row.get("eligible_windows", 0)) <= 0:
            raise ValueError(f"gamma {gamma} contains no eligible weighted-plan targets")


def _history_indices(episodes: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    """Return local newest-to-oldest row indices without copying Hp frames."""
    episodes = torch.as_tensor(episodes, dtype=torch.int64).reshape(-1)
    steps = torch.as_tensor(steps, dtype=torch.int64).reshape(-1)
    if len(episodes) != len(steps):
        raise ValueError("episode/step lengths differ")
    result = torch.empty((len(episodes), HISTORY_LENGTH), dtype=torch.int64)
    for episode in torch.unique(episodes, sorted=True).tolist():
        rows = torch.nonzero(episodes == int(episode), as_tuple=False).flatten()
        order = torch.argsort(steps[rows], stable=True)
        rows = rows[order]
        episode_steps = steps[rows]
        expected = torch.arange(
            int(episode_steps[0]), int(episode_steps[0]) + len(rows), dtype=torch.int64
        )
        if not torch.equal(episode_steps, expected):
            raise ValueError(f"episode {episode} is duplicated, unordered, or non-contiguous")
        for position, row in enumerate(rows.tolist()):
            history_positions = [max(0, position - lag) for lag in range(HISTORY_LENGTH)]
            result[row] = rows[history_positions]
    return result


def _validate_source(payload: dict, path: Path, gamma: float) -> None:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"wrong HP100 schema in {path}")
    if not payload.get("success_only", False):
        raise ValueError(f"pretraining source is not successful-only: {path}")
    if float(payload.get("gamma")) != float(gamma):
        raise ValueError(f"gamma mismatch in {path}")
    required = {
        "hp": (32, 100), "low5": (5,), "hist": (16, 2), "U": (10, 2),
        "plan_weighted_h": (11,),
    }
    size = None
    for key, trailing in required.items():
        if key not in payload:
            raise ValueError(f"missing {key} in {path}")
        tensor = payload[key]
        if tuple(tensor.shape[1:]) != trailing:
            raise ValueError(f"{key} has shape {tuple(tensor.shape)} in {path}")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{key} must be float32 in {path}, got {tensor.dtype}")
        size = len(tensor) if size is None else size
        if len(tensor) != size:
            raise ValueError(f"window tensor lengths differ in {path}")
    for key in ("episode", "step"):
        if key not in payload or len(payload[key]) != size:
            raise ValueError(f"missing or mis-sized {key} in {path}")
    typed = {
        "target_eligible": torch.bool,
        "target_reason_code": torch.int8,
        "plan_candidate_count": torch.int32,
        "plan_accepted_count": torch.int32,
        "plan_rejected_count": torch.int32,
        "plan_first_violation": torch.int16,
        "action_mean_max_abs_error": torch.float32,
    }
    for key, dtype in typed.items():
        if key not in payload or len(payload[key]) != size:
            raise ValueError(f"missing or mis-sized {key} in {path}")
        if payload[key].dtype != dtype:
            raise ValueError(f"{key} must be {dtype} in {path}, got {payload[key].dtype}")
    eligible = payload["target_eligible"]
    logical = (
        (payload["plan_accepted_count"] > 0)
        & (payload["plan_first_violation"] == -1)
    )
    if not torch.equal(eligible, logical):
        raise ValueError(f"target eligibility identity fails in {path}")
    if not torch.equal(
        eligible, payload["target_reason_code"] == DATA.TARGET_ELIGIBLE
    ):
        raise ValueError(f"target reason codes disagree with eligibility in {path}")
    reason = payload["target_reason_code"]
    if torch.any(
        ~torch.isin(reason, torch.tensor(
            [DATA.TARGET_ELIGIBLE, DATA.TARGET_ALL_REJECTED,
             DATA.TARGET_WEIGHTED_H10_FAILED], dtype=reason.dtype
        ))
    ):
        raise ValueError(f"unknown target reason code in {path}")
    if torch.any((reason == DATA.TARGET_ALL_REJECTED) & (payload["plan_accepted_count"] != 0)):
        raise ValueError(f"all-rejected target has accepted candidates in {path}")
    if torch.any(
        (reason == DATA.TARGET_WEIGHTED_H10_FAILED)
        & ((payload["plan_accepted_count"] <= 0) | (payload["plan_first_violation"] < 1))
    ):
        raise ValueError(f"weighted-plan failure reason is inconsistent in {path}")
    if torch.any(
        payload["plan_accepted_count"] + payload["plan_rejected_count"]
        != payload["plan_candidate_count"]
    ):
        raise ValueError(f"planner candidate accounting fails in {path}")
    if payload.get("target_contract") != DATA.target_contract():
        raise ValueError(f"dataset target contract differs in {path}")
    counts = {
        "eligible_windows": int(eligible.sum().item()),
        "excluded_all_rejected_windows": int(
            (reason == DATA.TARGET_ALL_REJECTED).sum().item()
        ),
        "excluded_weighted_h10_failed_windows": int(
            (reason == DATA.TARGET_WEIGHTED_H10_FAILED).sum().item()
        ),
    }
    for key, expected in counts.items():
        if int(payload.get(key, -1)) != expected:
            raise ValueError(f"dataset {key} count differs in {path}")
    unique = torch.unique(payload["episode"].to(torch.int64), sorted=True)
    declared = int(payload.get("n_traj", -1))
    if declared != SUCCESSFUL_LINEAGES_PER_GAMMA or len(unique) != declared:
        raise ValueError(
            f"gamma {gamma} requires exactly {SUCCESSFUL_LINEAGES_PER_GAMMA} "
            f"successful lineages, found declared={declared}, unique={len(unique)}"
        )
    for episode in unique.tolist():
        mask = payload["episode"].to(torch.int64) == int(episode)
        if not bool(eligible[mask].any()):
            raise ValueError(f"successful lineage {episode} has no eligible target in {path}")
    if payload.get("dynamics") != DYN.contract():
        raise ValueError(f"dataset dynamics differ from the HP100 training contract: {path}")


def _validate_source_manifest_binding(
    payload: dict, row: dict, path: Path, gamma: float
) -> None:
    """Bind the authenticated manifest ledger to the tensor payload itself."""
    payload_episodes = sorted(map(
        int, torch.unique(payload["episode"].to(torch.int64), sorted=True).tolist()
    ))
    if payload_episodes != list(map(int, row.get("successful_episodes", []))):
        raise RuntimeError(
            f"gamma {gamma} tensor lineage IDs disagree with the manifest ledger"
        )
    if int(payload.get("n_seeds", -1)) != int(row.get("attempted_episodes", -2)):
        raise RuntimeError(f"gamma {gamma} tensor attempt count disagrees with manifest")
    if [
        int(payload.get("episode_start", -1)),
        int(payload.get("episode_stop_exclusive", -1)),
    ] != list(map(int, row.get("episode_range", []))):
        raise RuntimeError(f"gamma {gamma} tensor episode range disagrees with manifest")
    if len(payload["episode"]) != int(row.get("windows", -1)):
        raise RuntimeError(f"gamma {gamma} tensor window count disagrees with manifest")
    if int(payload["target_eligible"].sum().item()) != int(
        row.get("eligible_windows", -1)
    ):
        raise RuntimeError(f"gamma {gamma} eligible window count disagrees with manifest")
    reason = payload["target_reason_code"]
    excluded_counts = {
        "excluded_all_rejected_windows": int(
            (reason == DATA.TARGET_ALL_REJECTED).sum().item()
        ),
        "excluded_weighted_h10_failed_windows": int(
            (reason == DATA.TARGET_WEIGHTED_H10_FAILED).sum().item()
        ),
    }
    for key, actual in excluded_counts.items():
        if actual != int(row.get(key, -1)):
            raise RuntimeError(f"gamma {gamma} {key} count disagrees with manifest")
    if path.stat().st_size != int(row.get("bytes", -1)):
        raise RuntimeError(f"gamma {gamma} tensor byte count disagrees with manifest")


class HP100WindowDataset(Dataset):
    """A split view over memory-mapped current Hp frames."""

    def __init__(self, sources: list[dict], gamma_rows, source_rows):
        self.sources = sources
        self.gamma_rows = torch.as_tensor(gamma_rows, dtype=torch.int64).contiguous()
        self.source_rows = torch.as_tensor(source_rows, dtype=torch.int64).contiguous()
        if len(self.gamma_rows) != len(self.source_rows):
            raise ValueError("gamma/source row maps differ in length")
        self.episodes = torch.empty(len(self.source_rows), dtype=torch.int64)
        for gamma_index in torch.unique(self.gamma_rows, sorted=True).tolist():
            mask = self.gamma_rows == int(gamma_index)
            rows = self.source_rows[mask]
            self.episodes[mask] = sources[int(gamma_index)]["episode"][rows].to(torch.int64)

    def __len__(self):
        return len(self.source_rows)

    def __getitem__(self, index):
        gamma_index = int(self.gamma_rows[index])
        row = int(self.source_rows[index])
        source = self.sources[gamma_index]
        history_rows = source["_history_indices"][row]
        hp100 = source["hp"].index_select(0, history_rows)
        return (
            hp100,
            source["low5"][row],
            source["hist"][row],
            source["U"][row],
            source["episode"][row].to(torch.int64),
            torch.tensor(gamma_index, dtype=torch.int64),
        )


def load_split(
    dataset,
    gammas=SP.GAMMAS,
    val_frac=0.1,
    seed=DEFAULT_SEED,
    *,
    expected_manifest_sha256: str,
    require_canonical: bool = True,
):
    """Load ID files with one scenario-disjoint validation bank across gammas."""
    if not math.isclose(float(val_frac), 0.1, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("canonical HP100 pretraining requires a 90/10 trajectory split")
    manifest, manifest_path, manifest_sha = _manifest(dataset)
    if manifest_sha != str(expected_manifest_sha256):
        raise RuntimeError(
            f"dataset manifest {manifest_sha} != expected {expected_manifest_sha256}"
        )
    if require_canonical:
        _validate_canonical_manifest(manifest, Path(dataset).resolve())
    file_rows = _manifest_files(manifest)
    sources, source_rows = [], []
    train_gamma, train_rows, val_gamma, val_rows = [], [], [], []
    split_meta = {}
    for gamma_index, gamma in enumerate(map(float, gammas)):
        path = _gamma_path(dataset, gamma)
        row = file_rows.get(gamma)
        if row is None:
            raise ValueError(f"manifest has no file for gamma {gamma}")
        if path.name != row.get("file"):
            raise ValueError(f"manifest filename mismatch for gamma {gamma}")
        actual_hash = sha256_file(path)
        if actual_hash != row.get("sha256"):
            raise RuntimeError(f"dataset digest mismatch: {path}")
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        _validate_source(payload, path, gamma)
        if require_canonical:
            _validate_source_manifest_binding(payload, row, path, gamma)
        payload["_history_indices"] = _history_indices(payload["episode"], payload["step"])
        sources.append(payload)
        source_rows.append(dict(
            gamma_index=gamma_index, gamma=gamma, path=path,
            sha256=actual_hash, payload=payload,
            unique=torch.unique(payload["episode"].to(torch.int64), sorted=True),
        ))

    common = set(source_rows[0]["unique"].tolist())
    for source_row in source_rows[1:]:
        common &= set(source_row["unique"].tolist())
    n_val = int(round(SUCCESSFUL_LINEAGES_PER_GAMMA * float(val_frac)))
    if len(common) < n_val:
        raise RuntimeError(
            f"only {len(common)} successful scenarios are shared across all gammas; "
            f"need {n_val} for the globally disjoint validation bank"
        )
    common = torch.as_tensor(sorted(common), dtype=torch.int64)
    generator = torch.Generator().manual_seed(int(seed))
    val_episodes = common[torch.randperm(len(common), generator=generator)[:n_val]]
    val_episodes = torch.sort(val_episodes).values

    for source_row in source_rows:
        gamma_index = source_row["gamma_index"]
        gamma = source_row["gamma"]
        path = source_row["path"]
        actual_hash = source_row["sha256"]
        payload = source_row["payload"]
        unique = source_row["unique"]
        train_episodes = unique[~torch.isin(unique, val_episodes)]
        episodes = payload["episode"].to(torch.int64)
        is_val = torch.isin(episodes, val_episodes)
        eligible = payload["target_eligible"].to(torch.bool)
        local_val = torch.nonzero(is_val & eligible, as_tuple=False).flatten()
        local_train = torch.nonzero((~is_val) & eligible, as_tuple=False).flatten()
        train_gamma.append(torch.full_like(local_train, gamma_index))
        train_rows.append(local_train)
        val_gamma.append(torch.full_like(local_val, gamma_index))
        val_rows.append(local_val)
        split_meta[str(gamma)] = {
            "file": str(path), "sha256": actual_hash,
            "train_episodes": sorted(map(int, train_episodes.tolist())),
            "val_episodes": sorted(map(int, val_episodes.tolist())),
            "train_lineages": len(train_episodes), "val_lineages": len(val_episodes),
            "train_windows": len(local_train), "val_windows": len(local_val),
            "context_rows": len(payload["episode"]),
            "eligible_windows": int(eligible.sum().item()),
        }
    train = HP100WindowDataset(sources, torch.cat(train_gamma), torch.cat(train_rows))
    val = HP100WindowDataset(sources, torch.cat(val_gamma), torch.cat(val_rows))
    metadata = {
        "manifest": str(manifest_path), "manifest_sha256": manifest_sha,
        "files": split_meta, "split_seed": int(seed), "val_fraction": float(val_frac),
        "shared_validation_episodes": list(map(int, val_episodes.tolist())),
        "required_successful_lineages_per_gamma": SUCCESSFUL_LINEAGES_PER_GAMMA,
        "split_semantics": (
            "one shared set of 50 successful scenario IDs is validation-only "
            "for every gamma; no validation scenario appears in any training gamma"
        ),
    }
    return train, val, metadata


def hierarchical_sampler_weights(episodes, gamma_indices):
    """Uniform objective mass gamma -> successful trajectory -> window."""
    episodes = torch.as_tensor(episodes, dtype=torch.int64)
    gamma_indices = torch.as_tensor(gamma_indices, dtype=torch.int64)
    if len(episodes) != len(gamma_indices):
        raise ValueError("episode/gamma lengths differ")
    weights = torch.zeros(len(episodes), dtype=torch.float64)
    gammas = torch.unique(gamma_indices, sorted=True)
    for gamma in gammas.tolist():
        gamma_mask = gamma_indices == int(gamma)
        trajectories = torch.unique(episodes[gamma_mask], sorted=True)
        for trajectory in trajectories.tolist():
            mask = gamma_mask & (episodes == int(trajectory))
            weights[mask] = 1.0 / (
                len(gammas) * len(trajectories) * int(mask.sum())
            )
    if not torch.isclose(
        weights.sum(), torch.tensor(1.0, dtype=weights.dtype), atol=1.0e-10
    ):
        raise RuntimeError("pretraining hierarchical mass does not sum to one")
    return weights


@contextmanager
def preserve_rng():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@torch.no_grad()
def deterministic_validation(
    policy, dataset, gammas, device,
    batch=DEFAULT_VAL_BATCH, seed=DEFAULT_VALIDATION_NOISE_SEED,
):
    """Fixed CFM draws with gamma -> lineage -> window macro averaging."""
    policy.eval()
    lineage_totals: dict[tuple[int, int], float] = {}
    lineage_counts: dict[tuple[int, int], int] = {}
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    loader = DataLoader(dataset, batch_size=int(batch), shuffle=False, num_workers=0)
    with preserve_rng():
        for hp100, low, hist, controls, episode, gamma_index in loader:
            batch_size = len(controls)
            x1_cpu = (controls.float() / float(policy.u_max)).reshape(batch_size, policy.d)
            x0_cpu = torch.randn(x1_cpu.shape, generator=generator, dtype=torch.float32)
            tau_cpu = torch.rand((batch_size,), generator=generator).clamp(1.0e-4, 1.0)
            hp100 = hp100.to(device, non_blocking=True)
            low = low.to(device, non_blocking=True)
            hist = hist.to(device, non_blocking=True)
            x1 = x1_cpu.to(device, non_blocking=True)
            x0 = x0_cpu.to(device, non_blocking=True)
            tau = tau_cpu.to(device, non_blocking=True)
            x_tau = (1.0 - tau)[:, None] * x0 + tau[:, None] * x1
            target = x1 - x0
            prediction = policy(x_tau, tau, policy.ctx_from(hp100, low, hist))
            losses = ((prediction - target) ** 2).mean(dim=1).cpu()
            for loss, gamma_value, episode_value in zip(
                losses.tolist(), gamma_index.tolist(), episode.tolist(),
            ):
                key = (int(gamma_value), int(episode_value))
                lineage_totals[key] = lineage_totals.get(key, 0.0) + float(loss)
                lineage_counts[key] = lineage_counts.get(key, 0) + 1
    per_gamma = {}
    for index, gamma in enumerate(gammas):
        lineage_means = [
            lineage_totals[key] / lineage_counts[key]
            for key in sorted(lineage_totals)
            if key[0] == index
        ]
        if not lineage_means:
            raise RuntimeError(f"validation split has no lineage support for gamma {gamma}")
        per_gamma[str(float(gamma))] = float(np.mean(lineage_means))
    return float(np.mean(list(per_gamma.values()))), per_gamma


def atomic_save(payload, path) -> None:
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _source_hashes() -> dict:
    modules = {
        "trainer": inspect.getmodule(_source_hashes),
        "architecture": GPS,
        "dynamics": DYN,
        "features": HPF,
        "history": HPH,
        "protocol": SP,
        "scene": SS,
        "dataset_collector": DATA,
        "flow_policy": inspect.getmodule(GPS.FlowPolicy),
        "id_evaluator": importlib.import_module("sfm_hp100_eval"),
        "raw_integrator": importlib.import_module("sfm_b1_eval"),
        "validity_evaluator": importlib.import_module("sfm_metrics2"),
        "exact_verifier_polytope": importlib.import_module("verifier_polytope"),
    }
    result = {}
    for name, module in modules.items():
        path = Path(inspect.getsourcefile(module)).resolve()
        result[name] = {"path": str(path), "sha256": sha256_file(path)}
    for name, value in {
        "human_agent": SS.HumanAgent,
        "human_advance": SS._advance_humans,
    }.items():
        path = Path(inspect.getsourcefile(value)).resolve()
        result[name] = {"path": str(path), "sha256": sha256_file(path)}
    return result


def _runtime_provenance(device: str) -> dict:
    payload = dict(
        requested_device=str(device), python=sys.version,
        numpy=np.__version__, torch=torch.__version__,
        torch_cuda=torch.version.cuda,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_device_order=os.environ.get("CUDA_DEVICE_ORDER"),
    )
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA pretraining requested but torch.cuda is unavailable")
        index = torch.device(device).index
        index = torch.cuda.current_device() if index is None else int(index)
        properties = torch.cuda.get_device_properties(index)
        inventory = subprocess.run(
            [
                "nvidia-smi", "--query-gpu=index,uuid,name,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True, capture_output=True, text=True,
        ).stdout.strip().splitlines()
        payload["cuda"] = dict(
            logical_index=int(index), name=str(properties.name),
            total_memory_bytes=int(properties.total_memory),
            capability=list(torch.cuda.get_device_capability(index)),
            nvidia_smi_inventory=inventory,
        )
    return payload


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
    return {"root": str(root), "head": head, "clean": not bool(status), "status": status}


def _dataset_artifact_hashes(dataset_meta: dict) -> dict[str, str]:
    manifest_path = Path(dataset_meta["manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text())
    result = {str(manifest_path): sha256_file(manifest_path)}
    for row in manifest.get("files", []):
        for key in ("file", "progress_file"):
            path = manifest_path.parent / str(row[key])
            result[str(path.resolve())] = sha256_file(path)
    return result


def _assert_inputs_unchanged(initial: dict, dataset_meta: dict, device: str) -> dict:
    final = dict(
        git=_git_provenance(), source_hashes=_source_hashes(),
        runtime=_runtime_provenance(device),
        dataset_artifact_hashes=_dataset_artifact_hashes(dataset_meta),
    )
    if final != initial:
        differing = sorted(key for key in initial if initial.get(key) != final.get(key))
        raise RuntimeError(
            f"HP100 pretraining inputs changed during the run: {differing}; "
            "refusing to promote a checkpoint"
        )
    return final


def _prepare_restart_only_outputs(outdir, policy_out) -> tuple[Path, Path]:
    output = Path(outdir).resolve()
    promoted = Path(policy_out).resolve()
    if output.exists():
        raise FileExistsError(f"restart-only run refuses existing outdir: {output}")
    if promoted.exists():
        raise FileExistsError(f"refusing to overwrite promoted policy: {promoted}")
    output.mkdir(parents=True, exist_ok=False)
    promoted.parent.mkdir(parents=True, exist_ok=True)
    return output, promoted


def atomic_policy_save(policy, path, *, extra) -> None:
    path = Path(path).resolve()
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing stale promoted-policy temporary: {temporary}")
    GPS.save_sfm_hp100_policy(policy, temporary, extra=extra)
    try:
        # Hard-link publication is atomic and fails if another process created
        # the promoted path after the restart-only preflight.  os.replace would
        # silently overwrite that file.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_authenticated_checkpoint(path, *, device, expected_sha256=None):
    """Load only a checkpoint whose bytes remain unchanged across deserialization."""
    path = Path(path).resolve()
    before = sha256_file(path)
    if expected_sha256 is not None and before != str(expected_sha256):
        raise RuntimeError(
            f"checkpoint digest changed before load: {before} != {expected_sha256}"
        )
    policy, payload = GPS.load_sfm_hp100_policy(path, device=device)
    after = sha256_file(path)
    if after != before:
        raise RuntimeError("checkpoint bytes changed while they were being loaded")
    return policy, payload, before


def checkpoint_payload(policy, epoch, history, args, dataset_meta, provenance):
    return {
        "state_dict": {
            name: value.detach().cpu() for name, value in policy.state_dict().items()
        },
        "config": policy.config(),
        "epoch": int(epoch),
        "history": history,
        "training_args": vars(args),
        "dataset": dataset_meta,
        "provenance": provenance,
        "initialization": "from_scratch",
        "partial_transplant": False,
    }


def _resolve_id_raw_gate():
    """Load the ID-only evaluator without importing any OOD evaluation code."""
    try:
        module = importlib.import_module("sfm_hp100_eval")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "HP100 promotion requires sfm_hp100_eval.id_raw_gate; "
            "refusing to train or promote without the fixed ID temp=1 gate"
        ) from error
    gate = getattr(module, "id_raw_gate", None)
    if not callable(gate):
        raise RuntimeError("sfm_hp100_eval.id_raw_gate is missing or not callable")
    return gate


def _validate_gate(result: dict, gammas, *, M: int, ep0: int, noise_seed: int) -> None:
    if not isinstance(result, dict):
        raise TypeError("sfm_hp100_eval.id_raw_gate must return a dictionary")
    if float(result.get("temperature", float("nan"))) != 1.0:
        raise RuntimeError("HP100 promotion gate must use raw sampling temperature 1")
    if result.get("distribution") != "ID":
        raise RuntimeError("HP100 checkpoint promotion may inspect only ID scenarios")
    if int(result.get("M_per_gamma", -1)) != int(M):
        raise RuntimeError("HP100 ID gate changed its declared M per gamma")
    if int(result.get("ep0", -1)) != int(ep0):
        raise RuntimeError("HP100 ID gate changed its fixed scenario bank")
    if int(result.get("noise_seed", -1)) != int(noise_seed):
        raise RuntimeError("HP100 ID gate changed its fixed noise bank")
    required = {str(float(gamma)) for gamma in gammas}
    if set(result.get("per_gamma", {})) != required:
        raise RuntimeError("HP100 ID gate did not return every declared gamma")
    for gamma, row in result["per_gamma"].items():
        required_metrics = {
            "SR", "CR", "timeout", "Validity",
            "successful_clearance", "successful_time_to_goal",
        }
        if not required_metrics.issubset(row):
            raise RuntimeError(f"HP100 ID gate lacks full raw metrics for gamma {gamma}")


def _gate_score(result: dict, validation_cfm: float, gammas) -> tuple:
    rows = [result["per_gamma"][str(float(gamma))] for gamma in gammas]
    return (
        max(1.0 - float(row["SR"]) for row in rows),
        float(np.mean([1.0 - float(row["SR"]) for row in rows])),
        max(float(row["CR"]) for row in rows),
        float(np.mean([row["CR"] for row in rows])),
        -min(float(row["Validity"]) for row in rows),
        -float(np.mean([row["Validity"] for row in rows])),
        -min(float(row["SR"]) for row in rows),
        -float(np.mean([row["SR"] for row in rows])),
        float(validation_cfm),
    )


def _lr_multiplier(epoch_index: int, epochs: int, warmup: int) -> float:
    if epoch_index < warmup:
        return float(epoch_index + 1) / max(1, int(warmup))
    # LambdaLR evaluates index zero before the first optimizer step.  Reaching
    # one only at index ``epochs`` avoids training epoch 120 at exactly zero LR.
    progress = min(1.0, (epoch_index - warmup) / max(1, epochs - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--policy-out", required=True)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--val-batch", type=int, default=DEFAULT_VAL_BATCH)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--ckpt-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--gate-m", type=int, default=DEFAULT_SCREEN_M)
    parser.add_argument("--gate-top", type=int, default=DEFAULT_SCREEN_TOP)
    parser.add_argument("--gate-finalists", type=int, default=DEFAULT_SCREEN_FINALISTS)
    parser.add_argument("--confirm-m", type=int, default=DEFAULT_CONFIRM_M)
    parser.add_argument("--gate-episode-start", type=int, default=DEFAULT_SCREEN_EP0)
    parser.add_argument("--confirm-episode-start", type=int, default=DEFAULT_CONFIRM_EP0)
    parser.add_argument("--validation-noise-seed", type=int, default=DEFAULT_VALIDATION_NOISE_SEED)
    parser.add_argument("--gate-noise-seed", type=int, default=DEFAULT_SCREEN_NOISE_SEED)
    parser.add_argument("--confirm-noise-seed", type=int, default=DEFAULT_CONFIRM_NOISE_SEED)
    parser.add_argument(
        "--verifier-workers", type=int, default=DEFAULT_VERIFIER_WORKERS,
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument(
        "--smoke", action="store_true",
        help="one-epoch frozen integration smoke; never a promotable scientific checkpoint",
    )
    args = parser.parse_args()
    if args.smoke:
        args.epochs = 1
        args.samples_per_epoch = args.samples_per_epoch or 512
        args.ckpt_every = 1
        args.gate_m = 1
        args.gate_top = 1
        args.gate_finalists = 1
        args.confirm_m = 1
    canonical_knobs = dict(
        epochs=(args.epochs, DEFAULT_EPOCHS), batch=(args.batch, DEFAULT_BATCH),
        val_batch=(args.val_batch, DEFAULT_VAL_BATCH), samples_per_epoch=(args.samples_per_epoch, 0),
        lr=(args.lr, DEFAULT_LR), weight_decay=(args.weight_decay, DEFAULT_WEIGHT_DECAY),
        warmup=(args.warmup, DEFAULT_WARMUP), seed=(args.seed, DEFAULT_SEED),
        ckpt_every=(args.ckpt_every, DEFAULT_CHECKPOINT_EVERY),
        gate_m=(args.gate_m, DEFAULT_SCREEN_M), gate_top=(args.gate_top, DEFAULT_SCREEN_TOP),
        gate_finalists=(args.gate_finalists, DEFAULT_SCREEN_FINALISTS),
        confirm_m=(args.confirm_m, DEFAULT_CONFIRM_M),
        gate_episode_start=(args.gate_episode_start, DEFAULT_SCREEN_EP0),
        confirm_episode_start=(args.confirm_episode_start, DEFAULT_CONFIRM_EP0),
        validation_noise_seed=(args.validation_noise_seed, DEFAULT_VALIDATION_NOISE_SEED),
        gate_noise_seed=(args.gate_noise_seed, DEFAULT_SCREEN_NOISE_SEED),
        confirm_noise_seed=(args.confirm_noise_seed, DEFAULT_CONFIRM_NOISE_SEED),
    )
    if not args.smoke:
        changed = sorted(name for name, (actual, expected) in canonical_knobs.items() if actual != expected)
        if changed:
            raise ValueError(f"canonical HP100 pretraining knobs are frozen; changed={changed}")
    if int(args.num_workers) < 0:
        raise ValueError("num-workers must be nonnegative")
    if int(args.verifier_workers) <= 0:
        raise ValueError("verifier-workers must be positive")
    if int(args.gate_finalists) > int(args.gate_top):
        raise ValueError("gate-finalists cannot exceed gate-top")
    screen_range = set(range(int(args.gate_episode_start), int(args.gate_episode_start) + int(args.gate_m)))
    confirm_range = set(range(int(args.confirm_episode_start), int(args.confirm_episode_start) + int(args.confirm_m)))
    if screen_range & confirm_range:
        raise ValueError("M10 screen and M50 confirmation scenario banks must be disjoint")

    # Fail before a long GPU run if the fixed ID promotion evaluator is absent.
    id_raw_gate = _resolve_id_raw_gate()
    git = _git_provenance()
    if git["head"] != args.expected_source_commit:
        raise RuntimeError(
            f"source commit {git['head']} != expected {args.expected_source_commit}"
        )
    if not git["clean"]:
        raise RuntimeError("HP100 pretraining requires a clean frozen worktree")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train, val, dataset_meta = load_split(
        args.dataset, SP.GAMMAS, val_frac=0.1, seed=args.seed,
        expected_manifest_sha256=args.expected_manifest_sha256,
        require_canonical=True,
    )
    initial_inputs = dict(
        git=git, source_hashes=_source_hashes(),
        runtime=_runtime_provenance(args.device),
        dataset_artifact_hashes=_dataset_artifact_hashes(dataset_meta),
    )
    outdir, policy_out = _prepare_restart_only_outputs(args.outdir, args.policy_out)
    weights = hierarchical_sampler_weights(train.episodes, train.gamma_rows)
    samples_per_epoch = int(args.samples_per_epoch) or len(train)
    policy = GPS.build_sfm_hp100_policy(device=args.device)
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: _lr_multiplier(epoch, args.epochs, args.warmup),
    )
    provenance = {
        "dataset_manifest": dataset_meta["manifest"],
        "dataset_manifest_sha256": dataset_meta["manifest_sha256"],
        "source_hashes": initial_inputs["source_hashes"],
        "source_git": git,
        "runtime": initial_inputs["runtime"],
        "dataset_artifact_hashes": initial_inputs["dataset_artifact_hashes"],
        "dynamics": DYN.contract(),
        "architecture": policy.config(),
        "optimizer": {
            "name": "AdamW", "lr": args.lr, "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup, "schedule": "warmup_then_cosine",
        },
        "sampler_mass": "gamma -> successful demonstration lineage -> window",
        "validation_macro": "gamma -> successful demonstration lineage -> window",
        "validation_noise_seed": int(args.validation_noise_seed),
        "promotion": {
            "validity_parallelism": dict(
                workers=int(args.verifier_workers), start_method="spawn",
                task="one complete episode; ordered executor.map",
            ),
            "validation_candidates": int(args.gate_top),
            "screen": dict(
                M_per_gamma=int(args.gate_m), ep0=int(args.gate_episode_start),
                noise_seed=int(args.gate_noise_seed), temperature=1.0,
            ),
            "screen_finalists": int(args.gate_finalists),
            "m50_finalist_selection": dict(
                M_per_gamma=int(args.confirm_m), ep0=int(args.confirm_episode_start),
                noise_seed=int(args.confirm_noise_seed), temperature=1.0,
            ),
            "semantics": (
                "top globally scenario-disjoint ID-validation checkpoints -> "
                "matched-ID M10 screen -> disjoint matched-ID M50 second-stage selection "
                "(not an untouched confirmation); OOD forbidden"
            ),
        },
        "restart": (
            "restart-only: no optimizer/scheduler resume state; existing outdir or "
            "promoted policy is refused"
        ),
    }

    history, candidates = [], []
    for epoch in range(args.epochs):
        generator = torch.Generator().manual_seed(args.seed * 100003 + epoch)
        sampler = WeightedRandomSampler(
            weights, samples_per_epoch, replacement=True, generator=generator
        )
        loader = DataLoader(
            train, batch_size=args.batch, sampler=sampler, drop_last=True,
            num_workers=args.num_workers, pin_memory=str(args.device).startswith("cuda"),
        )
        policy.train()
        lr_used = float(optimizer.param_groups[0]["lr"])
        losses = []
        for hp100, low, hist, controls, _episode, _gamma in loader:
            hp100 = hp100.to(args.device, non_blocking=True)
            low = low.to(args.device, non_blocking=True)
            hist = hist.to(args.device, non_blocking=True)
            controls = controls.to(args.device, non_blocking=True)
            loss = policy.cfm_loss(controls, policy.ctx_from(hp100, low, hist))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        macro, per_gamma = deterministic_validation(
            policy, val, SP.GAMMAS, args.device, args.val_batch,
            seed=args.validation_noise_seed,
        )
        row = {
            "epoch": epoch + 1,
            "train_cfm": float(np.mean(losses)),
            "val_macro_cfm": macro,
            "val_per_gamma": per_gamma,
            "lr": lr_used,
        }
        history.append(row)
        if (epoch + 1) % args.ckpt_every == 0:
            path = outdir / f"ckpt_{epoch + 1}.pt"
            atomic_save(
                checkpoint_payload(policy, epoch + 1, history, args, dataset_meta, provenance),
                path,
            )
            candidates.append((macro, str(path)))
        print(json.dumps(row, sort_keys=True), flush=True)

    _assert_inputs_unchanged(initial_inputs, dataset_meta, args.device)
    validity_executor = ProcessPoolExecutor(
        max_workers=int(args.verifier_workers), mp_context=mp.get_context("spawn"),
    )
    screen_gates = []
    for validation_cfm, path in sorted(candidates)[:max(1, int(args.gate_top))]:
        candidate, _, checkpoint_sha = load_authenticated_checkpoint(
            path, device=args.device,
        )
        result = id_raw_gate(
            candidate,
            M=int(args.gate_m),
            ep0=int(args.gate_episode_start),
            device=args.device,
            seed=int(args.gate_noise_seed),
            validity_executor=validity_executor,
        )
        # The imported hook is, by construction, the matched-ID-only entry
        # point.  Record that constraint explicitly in the promoted payload.
        result = {"distribution": "ID", **result}
        _validate_gate(
            result, SP.GAMMAS, M=args.gate_m,
            ep0=args.gate_episode_start, noise_seed=args.gate_noise_seed,
        )
        screen_gates.append({
            **result,
            "stage": "M10_screen",
            "checkpoint": str(Path(path).resolve()),
            "checkpoint_sha256": checkpoint_sha,
            "val_macro_cfm": float(validation_cfm),
        })
    finalists = sorted(
        screen_gates,
        key=lambda row: _gate_score(row, row["val_macro_cfm"], SP.GAMMAS),
    )[:int(args.gate_finalists)]
    confirmation_gates = []
    for screen_row in finalists:
        candidate, _, _ = load_authenticated_checkpoint(
            screen_row["checkpoint"], device=args.device,
            expected_sha256=screen_row["checkpoint_sha256"],
        )
        result = id_raw_gate(
            candidate, M=int(args.confirm_m),
            ep0=int(args.confirm_episode_start), device=args.device,
            seed=int(args.confirm_noise_seed),
            validity_executor=validity_executor,
        )
        result = {"distribution": "ID", **result}
        _validate_gate(
            result, SP.GAMMAS, M=args.confirm_m,
            ep0=args.confirm_episode_start, noise_seed=args.confirm_noise_seed,
        )
        confirmation_gates.append({
            **result,
            "stage": "disjoint_M50_finalist_selection",
            "checkpoint": screen_row["checkpoint"],
            "checkpoint_sha256": screen_row["checkpoint_sha256"],
            "val_macro_cfm": screen_row["val_macro_cfm"],
            "screen_score": _gate_score(
                screen_row, screen_row["val_macro_cfm"], SP.GAMMAS,
            ),
        })
    validity_executor.shutdown(wait=True, cancel_futures=True)
    selected = min(
        confirmation_gates,
        key=lambda row: _gate_score(row, row["val_macro_cfm"], SP.GAMMAS),
    )
    final_inputs = _assert_inputs_unchanged(initial_inputs, dataset_meta, args.device)
    promoted, _, _ = load_authenticated_checkpoint(
        selected["checkpoint"], device="cpu",
        expected_sha256=selected["checkpoint_sha256"],
    )
    atomic_policy_save(
        promoted,
        policy_out,
        extra={
            "selected_by": (
                "globally scenario-disjoint ID validation + "
                "fixed matched-ID raw-temp-1 M10 screen + disjoint M50 finalist selection"
            ),
            "selected_gate": selected,
            "screen_gates": screen_gates,
            "confirmation_gates": confirmation_gates,
            "dataset": dataset_meta,
            "provenance": provenance,
            "completion_audit": {"inputs_unchanged": True, "final": final_inputs},
            "pretrained_from_scratch": True,
            "partial_transplant": False,
            "scientific_status": (
                "nonpromotable_smoke" if args.smoke else "canonical_ID_promoted"
            ),
        },
    )
    report = {
        "status": (
            "HP100_PRETRAIN_SMOKE_COMPLETE" if args.smoke
            else "HP100_PRETRAIN_COMPLETE"
        ),
        "policy": str(policy_out),
        "policy_sha256": sha256_file(policy_out),
        "selected": selected,
        "screen_gates": screen_gates,
        "confirmation_gates": confirmation_gates,
        "dataset": dataset_meta,
        "provenance": provenance,
        "completion_audit": {"inputs_unchanged": True, "final": final_inputs},
    }
    report_path = outdir / "pretraining_report.json"
    temporary = report_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True))
    os.replace(temporary, report_path)
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
