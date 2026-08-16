"""Raw-observation training path: widened scope, provider, dataset, audits."""
import copy

import pytest
import torch

import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_expansion_update as UPD
import sfm_hp100_raw_obs_capture as RAW
import sfm_hp100_raw_obs_dataset as RDS


def _adapter(seed=1):
    torch.manual_seed(seed)
    return PORT.HP100ExpansionPolicy(GPS.build_sfm_hp100_policy()).eval()


def _raw_inputs(seed):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.rand((10, 32, 100), generator=generator),
        torch.randn(5, generator=generator),
        torch.randn((16, 2), generator=generator) * 0.5,
    )


def _capture(adapter, directory, count, *, hp_dtype=torch.float16, gamma=0.3):
    """Write ``count`` raw records whose tokens come from this adapter."""
    writer = RAW.ShardWriter(directory, flush_every=3, hp_dtype=hp_dtype)
    rows = []
    for index in range(count):
        grid, low5, history = _raw_inputs(100 + index)
        token = adapter.policy.ctx_from(grid, low5, history)[0].detach()
        writer.add(
            (f"{gamma:g}", 900 + index, index),
            grid=grid, low5=low5, history=history, token=token,
            meta={"index": index},
        )
        generator = torch.Generator().manual_seed(500 + index)
        rows.append({
            "role": "positive",
            "gamma": gamma, "replica": 0, "lineage": "g0.3:rep00",
            "scenario_id": 900 + index, "step": index, "attempt": 0,
            "executed": True, "negative_reason": None,
            "context": torch.cat([token, torch.randn(24, generator=generator)]),
            "candidate": torch.randn((10, 2), generator=generator),
            "flow_base": torch.randn((10, 2), generator=generator),
            "verification": {"valid": True},
            "prediction_audit": {},
        })
    writer.close()
    return rows


def test_projection_scope_surface_and_frozen_digest():
    adapter = _adapter()
    parameters, names = UPD.configure_trainable(
        adapter, UPD.PROJECTION_OPTIMIZER_SCOPE,
    )
    assert tuple(names) == UPD.PROJECTION_TRAINABLE_NAMES
    assert sum(parameter.numel() for parameter in parameters) == 1_966_484
    assert "policy.grid_projection.1.weight" in names
    digest = UPD.frozen_surface_sha256(adapter, UPD.PROJECTION_OPTIMIZER_SCOPE)
    assert "grid_projection" not in digest
    for module in ("grid_conv", "enc_low", "gru"):
        assert module in digest
    # The trainable projection must not perturb the trunk-scope digests.
    assert UPD.frozen_surface_sha256(adapter, UPD.OPTIMIZER_SCOPE)[
        "grid_projection"
    ]


def test_raw_forward_token_matches_stored(tmp_path):
    adapter = _adapter()
    rows = _capture(adapter, tmp_path / "fp16", 4, hp_dtype=torch.float16)
    dataset = RDS.RawObsDataset(tmp_path / "fp16")
    audit = RDS.audit_tokens(dataset, adapter, rows, atol=1.0e-2)
    assert audit["ok"] and audit["rows"] == 4
    # fp32 storage reproduces the stored token bitwise at the capture batch
    # size (1); batched recompute may differ at kernel-accumulation order
    # (~1e-7), the same caveat as the archive replay audit.
    rows32 = _capture(adapter, tmp_path / "fp32", 4, hp_dtype=torch.float32)
    dataset32 = RDS.RawObsDataset(tmp_path / "fp32")
    for row in rows32:
        record = dataset32.fetch(RDS.row_key(row))
        with torch.no_grad():
            redone = RDS.token_from_raw(
                adapter.policy, record["grid"], record["low5"],
                record["history"],
            )[0]
        assert torch.equal(redone, record["token"])
    raw = dataset32.batch(rows32)
    with torch.no_grad():
        batched = RDS.token_from_raw(
            adapter.policy, raw["grid"], raw["low5"], raw["history"],
        )
    assert torch.allclose(batched, raw["token"], atol=1.0e-6)


def test_update_moves_projection_and_freezes_deep_encoders(tmp_path):
    adapter = _adapter()
    rows = _capture(adapter, tmp_path / "raw", 6)
    dataset = RDS.RawObsDataset(tmp_path / "raw")
    provider = RDS.make_context_provider(dataset, adapter)
    projection_before = copy.deepcopy(
        adapter.policy.grid_projection.state_dict()
    )
    head_before = adapter.policy.head.weight.detach().clone()
    gru_before = adapter.policy.gru.weight_ih_l0.detach().clone()
    conv_before = adapter.policy.grid_conv[0].weight.detach().clone()
    metrics = UPD.expansion_update(
        adapter, copy.deepcopy(rows), [],
        UPD.UpdateConfig(
            learning_rate=1.0e-3, batch_size=4,
            optimizer_scope=UPD.PROJECTION_OPTIMIZER_SCOPE,
        ),
        round_index=1, context_provider=provider,
    )
    assert metrics["accepted"] and metrics["context_path"] == "raw_forward"
    after = adapter.policy.grid_projection.state_dict()
    assert not torch.equal(after["1.weight"], projection_before["1.weight"])
    assert not torch.equal(adapter.policy.head.weight, head_before)
    assert torch.equal(adapter.policy.gru.weight_ih_l0, gru_before)
    assert torch.equal(adapter.policy.grid_conv[0].weight, conv_before)


def test_projection_scope_refuses_stored_tokens():
    adapter = _adapter()
    with pytest.raises(ValueError, match="context provider"):
        UPD.expansion_update(
            adapter, [], [],
            UPD.UpdateConfig(optimizer_scope=UPD.PROJECTION_OPTIMIZER_SCOPE),
            round_index=1,
        )


def test_token_and_raw_paths_agree_when_projection_frozen(tmp_path):
    stored = _adapter(seed=3)
    rows = _capture(stored, tmp_path / "raw", 6, hp_dtype=torch.float32)
    config = UPD.UpdateConfig(learning_rate=1.0e-5, batch_size=4)
    token_metrics = UPD.expansion_update(
        stored, copy.deepcopy(rows), [], config, round_index=1,
    )
    fresh = _adapter(seed=3)
    dataset = RDS.RawObsDataset(tmp_path / "raw")
    provider = RDS.make_context_provider(dataset, fresh)
    raw_metrics = UPD.expansion_update(
        fresh, copy.deepcopy(rows), [], config, round_index=1,
        context_provider=provider,
    )
    assert raw_metrics["context_path"] == "raw_forward"
    # Tokens were captured at batch size 1; the provider recomputes them in
    # training batches, so kernel-accumulation order admits ~1e-7 deviation
    # (the projection stays frozen here, so that is the only source).
    assert token_metrics["positive_loss_mean"] == pytest.approx(
        raw_metrics["positive_loss_mean"], abs=1.0e-5,
    )
    for (name_a, a), (name_b, b) in zip(
        stored.named_parameters(), fresh.named_parameters(),
    ):
        assert name_a == name_b
        assert torch.allclose(a, b, atol=1.0e-6)


def test_dataset_join_fails_closed(tmp_path):
    adapter = _adapter()
    rows = _capture(adapter, tmp_path / "raw", 2)
    dataset = RDS.RawObsDataset(tmp_path / "raw")
    orphan = dict(rows[0])
    orphan["scenario_id"] = 12345
    with pytest.raises(RuntimeError, match="1/3 archive rows"):
        dataset.assert_joined(rows + [orphan])
    with pytest.raises(KeyError):
        dataset.fetch(("0.3", 12345, 0))


def test_all_open_scope_surface_and_frozen_digest():
    adapter = _adapter()
    parameters, names = UPD.configure_trainable(
        adapter, UPD.ALL_OPEN_OPTIMIZER_SCOPE,
    )
    assert tuple(names) == UPD.ALL_OPEN_TRAINABLE_NAMES
    assert len(names) == 30
    assert sum(parameter.numel() for parameter in parameters) == 1_978_068
    # Every named parameter is trainable; noise_templates is a buffer and
    # never enters the surface.
    assert set(names) == set(dict(adapter.named_parameters()))
    assert "noise_templates" in dict(adapter.policy.named_buffers())
    digest = UPD.frozen_surface_sha256(adapter, UPD.ALL_OPEN_OPTIMIZER_SCOPE)
    assert set(digest) == {"noise_templates", "combined"}


def test_all_open_update_moves_everything_but_noise_templates(tmp_path):
    adapter = _adapter()
    rows = _capture(adapter, tmp_path / "raw", 6)
    dataset = RDS.RawObsDataset(tmp_path / "raw")
    provider = RDS.make_context_provider(dataset, adapter, open_encoders=True)
    assert provider.open_encoders is True
    before = {
        "conv": adapter.policy.grid_conv[0].weight.detach().clone(),
        "enc_low": adapter.policy.enc_low[0].weight.detach().clone(),
        "gru": adapter.policy.gru.weight_ih_l0.detach().clone(),
        "projection": adapter.policy.grid_projection[1].weight.detach().clone(),
        "trunk": adapter.policy.trunk.inp[0].weight.detach().clone(),
        "head": adapter.policy.head.weight.detach().clone(),
    }
    templates_before = adapter.policy.noise_templates.detach().clone()
    metrics = UPD.expansion_update(
        adapter, copy.deepcopy(rows), [],
        UPD.UpdateConfig(
            learning_rate=1.0e-3, batch_size=4,
            optimizer_scope=UPD.ALL_OPEN_OPTIMIZER_SCOPE,
        ),
        round_index=1, context_provider=provider,
    )
    assert metrics["accepted"] and metrics["context_path"] == "raw_forward"
    after = {
        "conv": adapter.policy.grid_conv[0].weight,
        "enc_low": adapter.policy.enc_low[0].weight,
        "gru": adapter.policy.gru.weight_ih_l0,
        "projection": adapter.policy.grid_projection[1].weight,
        "trunk": adapter.policy.trunk.inp[0].weight,
        "head": adapter.policy.head.weight,
    }
    for key, tensor in after.items():
        assert not torch.equal(tensor, before[key]), key
    assert torch.equal(adapter.policy.noise_templates, templates_before)


def test_open_encoders_false_is_value_identical_and_gates_gradients(tmp_path):
    adapter = _adapter()
    grid, low5, history = _raw_inputs(7)
    closed = RDS.token_from_raw(adapter.policy, grid, low5, history)
    opened = RDS.token_from_raw(
        adapter.policy, grid, low5, history, open_encoders=True,
    )
    # Same ops in the same order: forward values are identical (CPU path).
    assert torch.equal(closed.detach(), opened.detach())
    # Gradient reaches grid_conv only through the open path.
    UPD.configure_trainable(adapter, UPD.ALL_OPEN_OPTIMIZER_SCOPE)
    adapter.policy.zero_grad(set_to_none=True)
    closed.sum().backward()
    assert adapter.policy.grid_conv[0].weight.grad is None
    adapter.policy.zero_grad(set_to_none=True)
    opened2 = RDS.token_from_raw(
        adapter.policy, grid, low5, history, open_encoders=True,
    )
    opened2.sum().backward()
    assert adapter.policy.grid_conv[0].weight.grad is not None
    assert adapter.policy.gru.weight_ih_l0.grad is not None


def test_runner_refuses_all_open_with_stored_tokens():
    import types

    import sfm_hp100_raw_train as RT

    with pytest.raises(ValueError, match="all_open requires --context-path raw"):
        RT.run(types.SimpleNamespace(
            optimizer_scope=UPD.ALL_OPEN_OPTIMIZER_SCOPE,
            context_path="stored",
        ))


def test_all_open_refuses_closed_provider(tmp_path):
    adapter = _adapter()
    rows = _capture(adapter, tmp_path / "raw", 2)
    dataset = RDS.RawObsDataset(tmp_path / "raw")
    closed_provider = RDS.make_context_provider(dataset, adapter)
    assert closed_provider.open_encoders is False
    with pytest.raises(ValueError, match="open_encoders=True"):
        UPD.expansion_update(
            adapter, copy.deepcopy(rows), [],
            UPD.UpdateConfig(optimizer_scope=UPD.ALL_OPEN_OPTIMIZER_SCOPE),
            round_index=1, context_provider=closed_provider,
        )


def test_lru_shard_cache(tmp_path):
    adapter = _adapter()
    rows = _capture(adapter, tmp_path / "raw", 7)  # flush_every=3 -> 3 shards
    dataset = RDS.RawObsDataset(tmp_path / "raw", lru_shards=1)
    fetched = [dataset.fetch(RDS.row_key(row)) for row in rows]
    assert len(dataset._cache) == 1
    for row, record in zip(rows, fetched):
        assert record["key"] == RDS.row_key(row)
        assert torch.equal(
            record["token"], row["context"][:RAW.TOKEN_DIM],
        )
