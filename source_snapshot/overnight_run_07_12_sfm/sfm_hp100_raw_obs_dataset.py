"""Join expansion-archive rows to raw-observation shards for raw training.

The extended collection (``sfm_hp100_raw_obs_capture``) stores, per replan,
the raw encoder inputs that produced the archived 176-token context: the
``[10,32,100]`` Hp raster stack (fp16 on disk by default), the ``low5``
vector, and the GRU control history, indexed by ``(gamma, scenario_id,
step)``.  This module joins archive D+/D- rows to those shards fail-closed,
loads shards lazily behind a small LRU (never the whole run in RAM), and
provides the raw-forward context path: recompute the token through the
frozen encoders with gradient flowing only through ``grid_projection`` —
``grid_conv``/``angular_pool`` run under ``no_grad`` and their features are
detached into the projection, and the low/GRU side stays fully frozen — so
``expansion_update``'s ``context_provider`` hook can train the widened
``trunk_head_and_projection`` scope.  When the projection is excluded from
the optimizer the recomputed token carries no graph at all and the provider
path is loss-equivalent to the stored-token path (bitwise for fp32 shards,
within the recorded fp16 round-trip tolerance otherwise).
"""
from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path

import torch

import _paths  # noqa: F401
import sfm_hp100_ball_adapter as PORT

VERSION = "sfm_hp100_raw_obs_dataset_v1"
DEFAULT_LRU_SHARDS = 4


def row_key(row: dict) -> tuple:
    """The capture key of one archive row: ``(gamma, scenario_id, step)``."""
    return (
        f"{float(row['gamma']):g}", int(row["scenario_id"]), int(row["step"]),
    )


class RawObsDataset:
    """Lazy, LRU-cached view over one or more raw-observation shard dirs."""

    def __init__(self, manifest_dirs, *, lru_shards: int = DEFAULT_LRU_SHARDS):
        if isinstance(manifest_dirs, (str, Path)):
            manifest_dirs = [manifest_dirs]
        self.directories = [Path(directory) for directory in manifest_dirs]
        if not self.directories:
            raise ValueError("raw-obs dataset requires at least one manifest dir")
        self.lru_shards = int(lru_shards)
        if self.lru_shards < 1:
            raise ValueError("lru_shards must be positive")
        self.manifests = []
        self._index: dict[tuple, tuple[int, int, int]] = {}
        self._cache: OrderedDict[tuple[int, int], dict] = OrderedDict()
        for source, directory in enumerate(self.directories):
            manifest_path = directory / "RAW_OBS_MANIFEST.json"
            manifest = json.loads(manifest_path.read_text())
            self.manifests.append(manifest)
            for flat, (shard, offset) in manifest["index"].items():
                gamma, scenario_id, step = flat.split("|")
                key = (gamma, int(scenario_id), int(step))
                if key in self._index:
                    raise RuntimeError(
                        f"duplicate raw-obs key across manifests: {key}"
                    )
                self._index[key] = (source, int(shard), int(offset))

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, key: tuple) -> bool:
        return tuple(key) in self._index

    @property
    def hp_roundtrip_max_abs(self) -> float | None:
        """Worst recorded storage-cast raster deviation across all sources."""
        bounds = [
            manifest.get("hp_roundtrip_max_abs")
            for manifest in self.manifests
            if manifest.get("hp_roundtrip_max_abs") is not None
        ]
        return max(bounds) if bounds else None

    def _shard(self, source: int, shard: int) -> dict:
        cache_key = (source, shard)
        if cache_key in self._cache:
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]
        manifest = self.manifests[source]
        payload = torch.load(
            self.directories[source] / manifest["shards"][shard]["path"],
            map_location="cpu", weights_only=False,
        )
        self._cache[cache_key] = payload
        while len(self._cache) > self.lru_shards:
            self._cache.popitem(last=False)
        return payload

    def fetch(self, key: tuple) -> dict:
        key = tuple(key)
        if key not in self._index:
            raise KeyError(f"raw-obs key not found: {key}")
        source, shard, offset = self._index[key]
        payload = self._shard(source, shard)
        return {
            "key": key,
            "grid": payload["grids"][offset].to(torch.float32),
            "low5": payload["low5"][offset].to(torch.float32),
            "history": payload["history"][offset].to(torch.float32),
            "token": payload["tokens"][offset].to(torch.float32),
            "meta": payload["meta"][offset],
        }

    def assert_joined(self, rows) -> None:
        """Fail closed unless every archive row has a raw record."""
        missing = [row_key(row) for row in rows if row_key(row) not in self._index]
        if missing:
            raise RuntimeError(
                f"{len(missing)}/{len(rows)} archive rows have no raw-obs "
                f"record; first misses: {missing[:5]}"
            )

    def batch(self, rows) -> dict:
        """Stacked raw tensors for a batch of archive rows (join fail-closed)."""
        records = [self.fetch(row_key(row)) for row in rows]
        return {
            "grid": torch.stack([record["grid"] for record in records]),
            "low5": torch.stack([record["low5"] for record in records]),
            "history": torch.stack([record["history"] for record in records]),
            "token": torch.stack([record["token"] for record in records]),
        }


def token_from_raw(
    policy, grid: torch.Tensor, low5: torch.Tensor, history: torch.Tensor,
) -> torch.Tensor:
    """Recompute the 176-token with gradient only through grid_projection.

    Mirrors the frozen ``ctx_from`` exactly: GRU + low encoder and the conv/
    pool visual features run under ``no_grad`` (they are frozen in every
    declared scope), the pooled features are detached, and only the
    ``grid_projection`` call participates in autograd — its output requires
    grad exactly when the projection parameters do, so the same code path
    serves every scope.
    """
    grid, low5, history = policy._batched_inputs(
        grid.to(torch.float32), low5.to(torch.float32),
        history.to(torch.float32),
    )
    with torch.no_grad():
        _, hidden = policy.gru(history)
        raw_low = torch.cat([low5[:, :4], hidden[-1], low5[:, 4:5]], dim=1)
        low_token = policy.enc_low(raw_low)
        features = policy.angular_pool(policy.grid_conv(grid))
    visual_token = policy.grid_projection(features.detach())
    return torch.cat([low_token, visual_token], dim=1)


def make_context_provider(dataset: RawObsDataset, adapter):
    """``expansion_update`` context provider over this dataset."""
    if not isinstance(adapter, PORT.HP100ExpansionPolicy):
        raise TypeError("raw context provider requires the HP100 adapter")

    def provider(rows, device):
        raw = dataset.batch(rows)
        return token_from_raw(
            adapter.policy,
            raw["grid"].to(device), raw["low5"].to(device),
            raw["history"].to(device),
        )

    return provider


def audit_tokens(
    dataset: RawObsDataset, adapter, rows, *, atol: float,
) -> dict:
    """Recompute tokens from raw (no grad) and compare to the stored tokens."""
    raw = dataset.batch(rows)
    device = next(adapter.policy.parameters()).device
    with torch.no_grad():
        redone = token_from_raw(
            adapter.policy,
            raw["grid"].to(device), raw["low5"].to(device),
            raw["history"].to(device),
        )
    deviation = float((redone - raw["token"].to(device)).abs().max())
    return {
        "rows": len(rows),
        "max_abs_deviation": deviation,
        "atol": float(atol),
        "ok": bool(deviation <= float(atol)),
    }
