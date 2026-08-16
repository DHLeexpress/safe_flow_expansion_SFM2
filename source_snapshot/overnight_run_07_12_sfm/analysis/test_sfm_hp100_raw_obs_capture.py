"""Raw-observation capture: scoping, shard round-trip, re-encode equality."""
import importlib.util
import json
from pathlib import Path

import pytest
import torch

import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_raw_obs_capture as RAW


def _encoder(seed=0):
    torch.manual_seed(seed)
    return GPS.build_sfm_hp100_policy().eval()


def _run_capture(tmp_path, *, steps=3, verify_every=1, flush_every=2,
                 hp_dtype=torch.float16):
    encoder = _encoder()
    with RAW.install_raw_obs_capture(
        tmp_path / "raw_obs", flush_every=flush_every,
        verify_every=verify_every, hp_dtype=hp_dtype,
    ) as recorder:
        assert PORT.SFMHP100ExpansionTask is RAW.RawObsCaptureTask
        task = PORT.SFMHP100ExpansionTask(
            scene_profile="double_density_velocity_ood",
            scenario_start=860_000,
        ).attach_context_encoder(encoder)
        state = task.reset(0.3, episode=0, seed=11)
        contexts = []
        for _ in range(steps):
            contexts.append(task.context(state, 0.3))
            state = task.advance(state, torch.zeros(10, 2))
    return encoder, task, recorder, contexts


def test_scoped_install_restores_the_original_task_class(tmp_path):
    original = PORT.SFMHP100ExpansionTask
    _run_capture(tmp_path, steps=1)
    assert PORT.SFMHP100ExpansionTask is original
    assert RAW.RawObsCaptureTask._recorder is None
    with pytest.raises(RuntimeError):
        # Nested installs are refused while one is active.
        with RAW.install_raw_obs_capture(tmp_path / "again"):
            with RAW.install_raw_obs_capture(tmp_path / "nested"):
                pass
    assert PORT.SFMHP100ExpansionTask is original


def test_shard_round_trip_index_and_bitwise_reencode(tmp_path):
    encoder, task, recorder, contexts = _run_capture(
        tmp_path, steps=3, verify_every=1, flush_every=2,
    )
    manifest = json.loads(
        (tmp_path / "raw_obs" / "RAW_OBS_MANIFEST.json").read_text()
    )
    assert manifest["status"] == RAW.MANIFEST_STATUS
    assert manifest["rows"] == 3 == manifest["records"]
    assert manifest["inline_bitwise_verifications"] == 3
    assert [shard["rows"] for shard in manifest["shards"]] == [2, 1]
    scenario_id = task.scene_ledger[0]["scenario_id"]
    for step, packed in enumerate(contexts):
        row = RAW.load_raw_row(tmp_path / "raw_obs", ("0.3", scenario_id, step))
        # fp32 capture reproduces the stored token bitwise through the
        # frozen encoders; the stored token equals the packed prefix.
        assert torch.equal(row["token"], packed[:RAW.TOKEN_DIM])
        redone = RAW.reencode(encoder, row["grid"], row["low5"], row["history"])
        deviation = float((redone - row["token"]).abs().max())
        # The shard stores the raster in fp16, so the reload carries the
        # measured round-trip loss; the manifest records the exact bound.
        assert deviation <= 5.0e-3
        assert row["hp_roundtrip_max_abs"] <= 2.0e-3


def test_fp32_storage_keeps_reencode_bitwise(tmp_path):
    encoder, task, recorder, contexts = _run_capture(
        tmp_path, steps=2, hp_dtype=torch.float32,
    )
    scenario_id = task.scene_ledger[0]["scenario_id"]
    for step, packed in enumerate(contexts):
        row = RAW.load_raw_row(tmp_path / "raw_obs", ("0.3", scenario_id, step))
        redone = RAW.reencode(encoder, row["grid"], row["low5"], row["history"])
        assert torch.equal(redone, packed[:RAW.TOKEN_DIM])


def test_duplicate_key_fails_closed(tmp_path):
    writer = RAW.ShardWriter(tmp_path / "raw_obs", flush_every=10)
    record = dict(
        grid=torch.zeros(10, 32, 100), low5=torch.zeros(5),
        history=torch.zeros(16, 2), token=torch.zeros(RAW.TOKEN_DIM),
        meta={},
    )
    writer.add(("0.3", 1, 0), **record)
    with pytest.raises(RuntimeError):
        writer.add(("0.3", 1, 0), **record)


def test_verification_failure_fails_closed(tmp_path):
    encoder = _encoder()
    writer = RAW.ShardWriter(tmp_path / "raw_obs")
    recorder = RAW.Recorder(writer, verify_every=1)
    with pytest.raises(RuntimeError):
        recorder.record(
            encoder=encoder, gamma=0.3, scenario_id=1, step=0,
            grid=torch.zeros(10, 32, 100), low5=torch.zeros(5),
            history=torch.zeros(16, 2),
            token=torch.full((RAW.TOKEN_DIM,), 123.0), meta={},
        )


def test_pickled_task_carries_no_recorder_state(tmp_path):
    import pickle

    with RAW.install_raw_obs_capture(tmp_path / "raw_obs"):
        task = PORT.SFMHP100ExpansionTask(
            scene_profile="double_density_velocity_ood",
            scenario_start=860_000,
        )
        payload = pickle.dumps(task)
    state = pickle.loads(payload)
    assert "_recorder" not in state.__dict__
    assert state._context_encoder is None


def test_extended_driver_is_import_safe():
    root = Path(__file__).resolve().parents[3]
    path = root / "scripts" / "mpc_expansion" / "mpc_collect_extended.py"
    spec = importlib.util.spec_from_file_location("mpc_collect_extended", path)
    module = importlib.util.module_from_spec(spec)
    # Importing with __name__ != "__main__" must execute nothing effectful —
    # this is the spawn-guard contract for verifier workers.
    spec.loader.exec_module(module)
    assert callable(module._run)
    assert module.PARAMS.lam == pytest.approx(4.0)
