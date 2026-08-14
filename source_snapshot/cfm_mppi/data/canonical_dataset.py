from __future__ import annotations

import csv
import json
import math
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


CANONICAL_REQUIRED = (
    "states",
    "controls_dyn",
    "controls_si",
    "start",
    "goal",
    "ego_history",
    "action_history",
    "nearest_obstacle_history",
    "gamma",
    "dynamics_type",
    "safety_margin",
    "source",
    "metadata",
)


def _as_float_tensor(value: Any, *, name: str, ndim: Optional[int] = None) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if ndim is not None and tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dims, got shape {tuple(tensor.shape)}")
    return tensor


def _pad_last_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    if x.shape[-1] == dim:
        return x
    if x.shape[-1] > dim:
        return x[..., :dim]
    pad = torch.zeros(*x.shape[:-1], dim - x.shape[-1], dtype=x.dtype)
    return torch.cat([x, pad], dim=-1)


def _safe_velocity_from_positions(pos: torch.Tensor, dt: float) -> torch.Tensor:
    vel = torch.zeros_like(pos)
    if pos.shape[1] > 1:
        vel[:, 1:] = (pos[:, 1:] - pos[:, :-1]) / max(dt, 1e-6)
        vel[:, 0] = vel[:, 1]
    return vel


def _make_histories(
    states: torch.Tensor,
    controls: torch.Tensor,
    obstacle_rel: torch.Tensor,
    history_len: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n, t_plus, state_dim = states.shape
    horizon = controls.shape[1]
    ego_hist = torch.zeros(n, history_len, state_dim, dtype=torch.float32)
    act_hist = torch.zeros(n, history_len, controls.shape[-1], dtype=torch.float32)
    obs_hist = torch.zeros(n, history_len, obstacle_rel.shape[-1], dtype=torch.float32)
    take_s = min(history_len, t_plus)
    take_u = min(history_len, horizon)
    ego_hist[:, -take_s:] = states[:, :take_s]
    act_hist[:, -take_u:] = controls[:, :take_u]
    obs_hist[:, -take_u:] = obstacle_rel[:, :take_u]
    return ego_hist, act_hist, obs_hist


def _empty_obstacles(n: int, max_obs: int = 1) -> torch.Tensor:
    return torch.full((n, max_obs, 3), float("nan"), dtype=torch.float32)


def validate_canonical(data: Mapping[str, Any]) -> None:
    missing = [k for k in CANONICAL_REQUIRED if k not in data]
    if missing:
        raise ValueError(f"Canonical dataset missing keys: {missing}")
    states = data["states"]
    controls_dyn = data["controls_dyn"]
    controls_si = data["controls_si"]
    if states.ndim != 3:
        raise ValueError(f"states must be [N,T+1,state_dim], got {tuple(states.shape)}")
    if controls_dyn.ndim != 3 or controls_si.ndim != 3:
        raise ValueError("controls_dyn and controls_si must be rank-3 tensors")
    if states.shape[0] != controls_dyn.shape[0] or controls_dyn.shape[:2] != controls_si.shape[:2]:
        raise ValueError("states/controls batch or horizon dimensions do not match")


def build_canonical_from_mizuta(
    data_path: str | Path,
    *,
    history_len: int = 10,
    dt: float = 0.1,
    dynamics_type: str = "singleintegrator",
    safety_margin: float = 0.5,
    max_items: Optional[int] = None,
) -> Dict[str, Any]:
    src = Path(data_path)
    raw = torch.load(src, map_location="cpu")
    if not torch.is_tensor(raw) or raw.ndim != 3 or raw.shape[1] < 4:
        raise ValueError(
            f"Mizuta tensor must be [N,C,T] with C>=4. Got {type(raw)} shape {getattr(raw, 'shape', None)}"
        )
    if max_items is not None:
        raw = raw[: int(max_items)]
    raw = raw.float()
    n, channels, horizon = raw.shape
    pos = raw[:, 0:2, :].transpose(1, 2).contiguous()
    controls = raw[:, 2:4, :].transpose(1, 2).contiguous()
    states = torch.zeros(n, horizon + 1, 4, dtype=torch.float32)
    states[:, :horizon, 0:2] = pos
    states[:, horizon, 0:2] = pos[:, -1] + controls[:, -1] * dt
    states[:, :, 2:4] = _safe_velocity_from_positions(states[:, :, 0:2], dt)
    start = pos[:, 0].contiguous()
    goal = pos[:, -1].contiguous()
    obs_rel = torch.zeros(n, horizon, 4, dtype=torch.float32)
    ego_hist, act_hist, obs_hist = _make_histories(states, controls, obs_rel, history_len)
    metadata = {
        "schema_version": 1,
        "source_format": "mizuta_train80_ego",
        "source_path": str(src.resolve()),
        "raw_shape": list(raw.shape),
        "raw_dtype": str(raw.dtype),
        "raw_channel_note": "Original training consumes channels 0:2 as positions and 2:4 as controls; all channels are preserved in metadata only.",
        "dt": dt,
        "history_len": history_len,
    }
    return {
        "states": states,
        "controls_dyn": controls.clone(),
        "controls_si": controls.clone(),
        "start": start,
        "goal": goal,
        "ego_history": ego_hist,
        "action_history": act_hist,
        "nearest_obstacle_history": obs_hist,
        "obstacles": _empty_obstacles(n),
        "gamma": torch.full((n,), float("nan"), dtype=torch.float32),
        "dynamics_type": [dynamics_type] * n,
        "safety_margin": torch.full((n,), float(safety_margin), dtype=torch.float32),
        "source": ["mizuta"] * n,
        "metadata": metadata,
    }


def _load_pickle_or_torch(path: Path) -> Any:
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu", weights_only=False)
    if path.suffix in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            return pickle.load(f)
    if path.suffix == ".npz":
        return dict(np.load(path, allow_pickle=True))
    if path.suffix == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records
    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    raise ValueError(f"Unsupported safeGPC data file type: {path.suffix}")


def _records_from_loaded(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return [dict(x) for x in obj]
    if isinstance(obj, dict):
        if {"obs", "u", "gamma"}.issubset(obj.keys()):
            obs = np.asarray(obj["obs"])
            u = np.asarray(obj["u"])
            gamma = np.asarray(obj["gamma"])
            n = obs.shape[0]
            return [{"obs": obs[i], "u": u[i], "gamma": gamma[i]} for i in range(n)]
        for key in ("samples", "records", "data"):
            if key in obj:
                return _records_from_loaded(obj[key])
        keys = sorted(obj.keys())
        raise ValueError(f"Dictionary safeGPC artifact does not expose records. Keys: {keys}")
    raise ValueError(f"Cannot interpret safeGPC artifact of type {type(obj)}")


def _coerce_record_value(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"nan", ""}:
            return float("nan")
        try:
            return float(text)
        except ValueError:
            return text
    return value


def _array_from_record(prefix: str, record: Mapping[str, Any], fallback_key: str) -> np.ndarray:
    if fallback_key in record:
        value = record[fallback_key]
        if isinstance(value, str):
            try:
                return np.asarray(json.loads(value), dtype=np.float32)
            except Exception:
                pass
        return np.asarray(value, dtype=np.float32)
    cols = sorted(
        [k for k in record if k.startswith(prefix + "_")],
        key=lambda k: int(k.split("_")[-1]),
    )
    if not cols:
        raise KeyError(f"Record lacks {fallback_key!r} and {prefix}_* columns. Keys: {sorted(record)[:20]}")
    return np.asarray([_coerce_record_value(record[c]) for c in cols], dtype=np.float32)


def _split_safegpc_episodes(records: Sequence[Mapping[str, Any]]) -> List[List[Dict[str, Any]]]:
    episodes: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    for rec_in in records:
        rec = {k: _coerce_record_value(v) for k, v in dict(rec_in).items()}
        gamma = rec.get("gamma", float("nan"))
        try:
            gamma_float = float(gamma)
        except Exception:
            gamma_float = float("nan")
        if math.isnan(gamma_float):
            if cur:
                episodes.append(cur)
                cur = []
            continue
        rec["gamma"] = gamma_float
        cur.append(rec)
    if cur:
        episodes.append(cur)
    return episodes


def build_canonical_from_safegpc(
    data_path: str | Path,
    *,
    history_len: int = 11,
    dt: float = 0.1,
    safety_margin: float = 0.0,
    max_episodes: Optional[int] = None,
) -> Dict[str, Any]:
    path = Path(data_path)
    if path.is_dir():
        records: List[Dict[str, Any]] = []
        for child in sorted(path.iterdir()):
            if child.suffix in {".pt", ".pkl", ".pickle", ".npz", ".jsonl", ".csv"}:
                records.extend(_records_from_loaded(_load_pickle_or_torch(child)))
    else:
        records = _records_from_loaded(_load_pickle_or_torch(path))
    episodes = _split_safegpc_episodes(records)
    if max_episodes is not None:
        episodes = episodes[: int(max_episodes)]
    if not episodes:
        raise ValueError(
            "No safeGPC episodes found. Required per-step fields are obs/obs_* [px,py,vx,vy,nearest_dx,nearest_dy], u/u_* [ax,ay], and gamma with NaN EOS markers."
        )
    max_horizon = max(len(ep) for ep in episodes)
    n = len(episodes)
    states = torch.zeros(n, max_horizon + 1, 4, dtype=torch.float32)
    controls = torch.zeros(n, max_horizon, 2, dtype=torch.float32)
    obs_hist_full = torch.zeros(n, max_horizon, 4, dtype=torch.float32)
    gammas = torch.full((n,), float("nan"), dtype=torch.float32)
    source = []
    lengths = []
    for i, ep in enumerate(episodes):
        obs = np.stack([_array_from_record("obs", r, "obs") for r in ep], axis=0).astype(np.float32)
        u = np.stack([_array_from_record("u", r, "u") for r in ep], axis=0).astype(np.float32)
        t = obs.shape[0]
        lengths.append(t)
        states[i, :t, : min(4, obs.shape[1])] = torch.from_numpy(obs[:, :4])
        controls[i, :t] = torch.from_numpy(u[:, :2])
        if t > 0:
            next_state = states[i, t - 1].clone()
            next_state[0:2] = next_state[0:2] + next_state[2:4] * dt + 0.5 * controls[i, t - 1] * dt * dt
            next_state[2:4] = next_state[2:4] + controls[i, t - 1] * dt
            states[i, t] = next_state
        if obs.shape[1] >= 6:
            obs_hist_full[i, :t, 0:2] = torch.from_numpy(obs[:, 4:6])
        gammas[i] = float(ep[-1].get("gamma", float("nan")))
        source.append("safeGPC")
    start = states[:, 0, 0:2].contiguous()
    goal = torch.zeros(n, 2, dtype=torch.float32)
    goal[:] = torch.tensor([0.0, 0.0])
    ego_hist, act_hist, near_hist = _make_histories(states, controls, obs_hist_full, history_len)
    metadata = {
        "schema_version": 1,
        "source_format": "safeGPC_step_records",
        "source_path": str(path.resolve()),
        "dt": dt,
        "history_len": history_len,
        "episode_lengths": lengths,
        "obs_layout": "[px, py, vx, vy, nearest_surface_dx, nearest_surface_dy]",
        "control_layout": "[ax, ay]",
        "gamma_eos_rule": "Rows with gamma=NaN delimit episodes and are not training samples.",
    }
    return {
        "states": states,
        "controls_dyn": controls.clone(),
        "controls_si": controls.clone(),
        "start": start,
        "goal": goal,
        "ego_history": ego_hist,
        "action_history": act_hist,
        "nearest_obstacle_history": near_hist,
        "obstacles": _empty_obstacles(n),
        "gamma": gammas,
        "dynamics_type": ["doubleintegrator"] * n,
        "safety_margin": torch.full((n,), float(safety_margin), dtype=torch.float32),
        "source": source,
        "metadata": metadata,
    }


def subset_canonical(data: Mapping[str, Any], indices: torch.Tensor) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    index_list = indices.tolist()
    for k, v in data.items():
        if torch.is_tensor(v) and v.shape[:1] == (len(data["states"]),):
            out[k] = v[indices].clone()
        elif isinstance(v, list) and len(v) == len(data["states"]):
            out[k] = [v[i] for i in index_list]
        else:
            out[k] = v
    out["metadata"] = dict(data.get("metadata", {}))
    out["metadata"]["subset_indices"] = index_list[:1000]
    out["metadata"]["subset_size"] = len(index_list)
    return out


def save_canonical_splits(
    data: Mapping[str, Any],
    output_dir: str | Path,
    *,
    seed: int = 0,
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> Dict[str, Path]:
    validate_canonical(data)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = data["states"].shape[0]
    generator = torch.Generator().manual_seed(int(seed))
    perm = torch.randperm(n, generator=generator)
    n_train = int(round(n * splits[0]))
    n_val = int(round(n * splits[1]))
    n_train = min(n_train, n)
    n_val = min(n_val, n - n_train)
    split_indices = {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }
    paths = {}
    for name, idx in split_indices.items():
        split = subset_canonical(data, idx)
        split["metadata"] = dict(split["metadata"], split=name, split_seed=seed)
        path = out_dir / f"{name}.pt"
        torch.save(split, path)
        paths[name] = path
    return paths


def load_canonical_dataset(path: str | Path) -> Dict[str, Any]:
    data = torch.load(path, map_location="cpu", weights_only=False)
    validate_canonical(data)
    return data


class CanonicalDataset(Dataset):
    def __init__(self, path_or_data: str | Path | Mapping[str, Any], *, horizon: Optional[int] = None):
        if isinstance(path_or_data, (str, Path)):
            self.data = load_canonical_dataset(path_or_data)
        else:
            self.data = dict(path_or_data)
            validate_canonical(self.data)
        self.horizon = horizon

    def __len__(self) -> int:
        return int(self.data["controls_si"].shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item: Dict[str, Any] = {}
        for key, value in self.data.items():
            if torch.is_tensor(value) and value.shape[:1] == (len(self),):
                item[key] = value[idx]
            elif isinstance(value, list) and len(value) == len(self):
                item[key] = value[idx]
        if self.horizon is not None:
            h = int(self.horizon)
            item["states"] = item["states"][: h + 1]
            item["controls_dyn"] = item["controls_dyn"][:h]
            item["controls_si"] = item["controls_si"][:h]
        return item


def canonical_collate(batch: Sequence[Mapping[str, Any]], *, random_truncate: bool = False, min_horizon: int = 10) -> Dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    horizon = min(int(b["controls_si"].shape[0]) for b in batch)
    if random_truncate and horizon > min_horizon:
        horizon = int(torch.randint(min_horizon, horizon + 1, (1,)).item())
    out: Dict[str, Any] = {}
    tensor_keys = [
        "states",
        "controls_dyn",
        "controls_si",
        "start",
        "goal",
        "ego_history",
        "action_history",
        "nearest_obstacle_history",
        "gamma",
        "safety_margin",
    ]
    for key in tensor_keys:
        vals = []
        for b in batch:
            v = b[key]
            if key == "states":
                v = v[: horizon + 1]
            elif key in {"controls_dyn", "controls_si"}:
                v = v[:horizon]
            vals.append(v)
        out[key] = torch.stack(vals, dim=0)
    out["dynamics_type"] = [b.get("dynamics_type", "unknown") for b in batch]
    out["source"] = [b.get("source", "unknown") for b in batch]
    return out


def describe_canonical(data: Mapping[str, Any]) -> str:
    validate_canonical(data)
    lines = []
    for key in CANONICAL_REQUIRED:
        value = data[key]
        if torch.is_tensor(value):
            lines.append(f"{key}: shape={tuple(value.shape)} dtype={value.dtype}")
        elif isinstance(value, list):
            lines.append(f"{key}: list len={len(value)} first={value[0] if value else None}")
        else:
            lines.append(f"{key}: {type(value).__name__}")
    if "obstacles" in data and torch.is_tensor(data["obstacles"]):
        lines.append(f"obstacles: shape={tuple(data['obstacles'].shape)} dtype={data['obstacles'].dtype}")
    return "\n".join(lines)
