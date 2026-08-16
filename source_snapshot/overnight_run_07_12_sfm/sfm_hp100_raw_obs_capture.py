"""Scoped raw-observation capture for extended D+ collection.

The expansion archive stores only the frozen 176-token context, so gradients
can never reach the condition encoders from archived rows.  This module
captures, per replan, the exact raw encoder inputs that produced the token —
the ten-frame Hp100 raster stack, the low5 vector, and the GRU control
history — so a later declared phase can retrain deeper surfaces (e.g.
``grid_projection``) from raw observations.

Seam: ``sfm_hp100_expansion_archive.run`` constructs its task through the
module attribute ``PORT.SFMHP100ExpansionTask`` at call time, and
``gather_predictive`` calls ``task.context(state, gamma)`` exactly once per
``(lineage, step)`` before the retry loop.  ``install_raw_obs_capture``
therefore rebinds that one attribute to a subclass for the duration of the
run — the same scoped runtime-monkeypatch discipline as
``install_v2_selector``; no frozen file is edited.  The subclass delegates to
the frozen ``context`` implementation and reads the raster stack back with
``Hp100History.tensor()`` (a pure view of what ``ctx_from`` just consumed);
``low5``/``hist_pad`` are frozen pure functions of the unchanged state, so
recomputing them reproduces the encoder inputs exactly.

Storage: shards of ``flush_every`` records; the raster stack is cast to
``hp_dtype`` (default fp16, ~64 KB/row -> ~6.5 GB per 100k rows) while low5 /
history / token stay fp32.  Every ``verify_every``-th record is re-encoded
inline through the task's own frozen encoder BEFORE any cast and must match
the stored token bitwise (fail closed).  The fp16 round-trip deviation is
measured per shard on its first record and recorded in the manifest; the
capture itself never loses precision at verification time.

Verifier worker processes unpickle the task under spawn; the subclass is
importable at module scope, the recorder lives on the class (never in the
pickled instance ``__dict__``), and a freshly imported worker sees
``_recorder is None`` so capture is a no-op off the main process.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path

import torch

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_features as HPF
import sfm_scene as SS

VERSION = "sfm_hp100_raw_obs_capture_v1"
MANIFEST_STATUS = "SFM2_RAW_OBS_MANIFEST"
TOKEN_DIM = int(GPS.LOW_TOKEN + GPS.VISUAL_TOKEN)
DEFAULT_FLUSH_EVERY = 2000
DEFAULT_VERIFY_EVERY = 500
DEFAULT_HP_DTYPE = torch.float16


def reencode(encoder, grid, low5, history):
    """Frozen-encoder token for one raw record (fp32 inputs, no grad)."""
    device = next(encoder.parameters()).device
    with torch.no_grad():
        token = encoder.ctx_from(
            grid.to(torch.float32).unsqueeze(0).to(device),
            low5.to(torch.float32).unsqueeze(0).to(device),
            history.to(torch.float32).unsqueeze(0).to(device),
        )[0].detach().cpu().to(torch.float32)
    return token


def verify_raw_row(encoder, grid, low5, history, token, *, atol=0.0):
    """Re-encode a raw row and compare against the stored token."""
    redone = reencode(encoder, grid, low5, history)
    deviation = float((redone - token.to(torch.float32)).abs().max())
    ok = deviation <= float(atol)
    return {"ok": bool(ok), "max_abs_deviation": deviation}


class ShardWriter:
    """Streamed shard storage: never accumulates the full run in memory."""

    def __init__(self, directory, *, flush_every=DEFAULT_FLUSH_EVERY,
                 hp_dtype=DEFAULT_HP_DTYPE):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.flush_every = int(flush_every)
        if self.flush_every <= 0:
            raise ValueError("flush_every must be positive")
        self.hp_dtype = hp_dtype
        self._buffer = []
        self._shards = []
        self._index = {}
        self._fp16_roundtrip = []
        self._closed = False

    def __len__(self):
        return len(self._index) + len(self._buffer)

    def add(self, key, *, grid, low5, history, token, meta):
        if self._closed:
            raise RuntimeError("shard writer is closed")
        if key in self._index or any(row["key"] == key for row in self._buffer):
            raise RuntimeError(f"duplicate raw-obs capture key: {key}")
        if tuple(grid.shape) != (10, 32, 100):
            raise ValueError(f"expected [10,32,100] raster stack, got {tuple(grid.shape)}")
        self._buffer.append({
            "key": key,
            "grid": grid.detach().cpu().to(torch.float32).clone(),
            "low5": low5.detach().cpu().to(torch.float32).clone(),
            "history": history.detach().cpu().to(torch.float32).clone(),
            "token": token.detach().cpu().to(torch.float32).clone(),
            "meta": dict(meta),
        })
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self._buffer:
            return
        shard_id = len(self._shards)
        name = f"raw_obs_shard_{shard_id:04d}.pt"
        grids = torch.stack([row["grid"] for row in self._buffer])
        stored = grids.to(self.hp_dtype)
        # Measured storage-cast information loss for this shard (raster only;
        # low5/history/token stay fp32).
        roundtrip = float((stored.to(torch.float32) - grids).abs().max())
        self._fp16_roundtrip.append(roundtrip)
        payload = {
            "status": "SFM2_RAW_OBS_SHARD",
            "version": VERSION,
            "shard": shard_id,
            "hp_dtype": str(self.hp_dtype),
            "hp_roundtrip_max_abs": roundtrip,
            "keys": [row["key"] for row in self._buffer],
            "grids": stored,
            "low5": torch.stack([row["low5"] for row in self._buffer]),
            "history": torch.stack([row["history"] for row in self._buffer]),
            "tokens": torch.stack([row["token"] for row in self._buffer]),
            "meta": [row["meta"] for row in self._buffer],
        }
        torch.save(payload, self.directory / name)
        for offset, row in enumerate(self._buffer):
            self._index[row["key"]] = (shard_id, offset)
        self._shards.append({
            "shard": shard_id, "path": name, "rows": len(self._buffer),
            "hp_roundtrip_max_abs": roundtrip,
        })
        self._buffer = []

    def close(self, *, extra=None):
        if self._closed:
            raise RuntimeError("shard writer already closed")
        self.flush()
        self._closed = True
        manifest = {
            "status": MANIFEST_STATUS,
            "version": VERSION,
            "rows": len(self._index),
            "hp_dtype": str(self.hp_dtype),
            "hp_roundtrip_max_abs": (
                max(self._fp16_roundtrip) if self._fp16_roundtrip else None
            ),
            "shards": self._shards,
            "index": {
                "|".join(str(part) for part in key): [shard, offset]
                for key, (shard, offset) in sorted(self._index.items())
            },
            **(extra or {}),
        }
        path = self.directory / "RAW_OBS_MANIFEST.json"
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        temporary.replace(path)
        return manifest


def load_raw_row(directory, key):
    """Round-trip helper: fetch one record by ``(gamma, scenario_id, step)``."""
    directory = Path(directory)
    manifest = json.loads((directory / "RAW_OBS_MANIFEST.json").read_text())
    flat = "|".join(str(part) for part in key)
    if flat not in manifest["index"]:
        raise KeyError(f"raw-obs key not found: {key}")
    shard_id, offset = manifest["index"][flat]
    payload = torch.load(
        directory / manifest["shards"][shard_id]["path"],
        map_location="cpu", weights_only=False,
    )
    return {
        "key": payload["keys"][offset],
        "grid": payload["grids"][offset].to(torch.float32),
        "low5": payload["low5"][offset],
        "history": payload["history"][offset],
        "token": payload["tokens"][offset],
        "meta": payload["meta"][offset],
        "hp_roundtrip_max_abs": payload["hp_roundtrip_max_abs"],
    }


class Recorder:
    """Bridges the capture subclass to a shard writer with inline audits."""

    def __init__(self, writer, *, verify_every=DEFAULT_VERIFY_EVERY):
        self.writer = writer
        self.verify_every = int(verify_every)
        if self.verify_every <= 0:
            raise ValueError("verify_every must be positive")
        self.records = 0
        self.verified = 0

    def record(self, *, encoder, gamma, scenario_id, step, grid, low5,
               history, token, meta):
        key = (f"{float(gamma):g}", int(scenario_id), int(step))
        if self.records % self.verify_every == 0:
            audit = verify_raw_row(encoder, grid, low5, history, token, atol=0.0)
            if not audit["ok"]:
                raise RuntimeError(
                    "raw-obs capture failed bitwise re-encode verification: "
                    f"{key} deviation={audit['max_abs_deviation']}"
                )
            self.verified += 1
        self.writer.add(
            key, grid=grid, low5=low5, history=history, token=token, meta=meta,
        )
        self.records += 1


class RawObsCaptureTask(PORT.SFMHP100ExpansionTask):
    """Task subclass that mirrors every context build into the recorder.

    The recorder is a class attribute (never instance state), so pickled
    copies sent to spawn verifier workers reconstruct with the fresh import's
    ``None`` and capture stays main-process only.
    """

    _recorder: Recorder | None = None

    def context(self, state, gamma):
        packed = super().context(state, gamma)
        recorder = type(self)._recorder
        if recorder is not None:
            grid = state.hp_history.tensor()
            low5 = torch.from_numpy(HPF.low5(state.robot, SS.GOAL, gamma))
            history = torch.from_numpy(HPF.hist_pad(state.controls))
            recorder.record(
                encoder=self._context_encoder,
                gamma=float(gamma), scenario_id=int(state.scenario_id),
                step=int(state.steps),
                grid=grid, low5=low5, history=history,
                token=packed[:TOKEN_DIM],
                meta={
                    "core_episode": int(state.core_episode),
                    "robot": [float(v) for v in state.robot],
                },
            )
        return packed


@contextmanager
def install_raw_obs_capture(directory, *, flush_every=DEFAULT_FLUSH_EVERY,
                            verify_every=DEFAULT_VERIFY_EVERY,
                            hp_dtype=DEFAULT_HP_DTYPE, extra_manifest=None):
    """Scoped install: PORT.SFMHP100ExpansionTask -> capture subclass."""
    if RawObsCaptureTask._recorder is not None:
        raise RuntimeError("raw-obs capture is already installed")
    writer = ShardWriter(directory, flush_every=flush_every, hp_dtype=hp_dtype)
    recorder = Recorder(writer, verify_every=verify_every)
    original = PORT.SFMHP100ExpansionTask
    RawObsCaptureTask._recorder = recorder
    PORT.SFMHP100ExpansionTask = RawObsCaptureTask
    try:
        yield recorder
    finally:
        PORT.SFMHP100ExpansionTask = original
        RawObsCaptureTask._recorder = None
        writer.close(extra={
            "records": recorder.records,
            "inline_bitwise_verifications": recorder.verified,
            "verify_every": recorder.verify_every,
        })
