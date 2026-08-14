"""Gates 10 and 11: loss signs/mass and the exact trainable/frozen surface."""
import copy

import pytest
import torch

import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_expansion_update as UPD


CTX = GPS.LOW_TOKEN + GPS.VISUAL_TOKEN


def _adapter(seed=0):
    torch.manual_seed(seed)
    return PORT.HP100ExpansionPolicy(GPS.build_sfm_hp100_policy()).eval()


def _row(seed, *, role="positive", negative_reason=None, valid=None):
    generator = torch.Generator().manual_seed(seed)
    if valid is None:
        valid = role == "positive" or negative_reason != "all_negative_nvp"
    return {
        "role": role,
        "gamma": 0.3, "replica": 0, "lineage": "g0.3:rep00",
        "scenario_id": 860_000 + seed, "step": seed, "attempt": 0,
        "executed": role == "positive" or negative_reason != "all_negative_nvp",
        "negative_reason": negative_reason,
        "context": torch.randn(CTX + 4, generator=generator),
        "candidate": torch.randn(10, 2, generator=generator),
        "flow_base": torch.randn(10, 2, generator=generator),
        "verification": {"valid": valid},
        "prediction_audit": {},
    }


def _positives(count=6):
    return [_row(seed) for seed in range(count)]


def _negatives():
    return [
        _row(100, role="negative", negative_reason="all_negative_nvp"),
        _row(101, role="negative", negative_reason="realized_collision"),
    ]


class DeterministicAdapter(PORT.HP100ExpansionPolicy):
    """CFM loss with frozen x0=0, tau=0.5 so objectives are draw-free."""

    def cfm_loss(self, contexts, candidates, reduction="none", loss_mask=None):
        assert loss_mask is None and reduction == "none"
        count = len(candidates)
        token = self._policy_context(contexts).to(candidates.device)
        x1 = (candidates / float(self.policy.u_max)).reshape(count, self.policy.d)
        tau = torch.full((count,), 0.5, device=x1.device)
        predicted = self.policy(0.5 * x1, tau, token)
        return (predicted - x1).square().reshape(count, 10, 2).mean(dim=(1, 2))


def _deterministic_adapter(seed=0):
    torch.manual_seed(seed)
    return DeterministicAdapter(GPS.build_sfm_hp100_policy()).eval()


def test_trainable_surface_is_exactly_trunk_and_head():
    adapter = _adapter()
    parameters, names = UPD.configure_trainable(adapter)
    assert tuple(names) == UPD.FROZEN_TRAINABLE_NAMES
    assert sum(parameter.numel() for parameter in parameters) == 327_956
    frozen = [
        name for name, parameter in adapter.named_parameters()
        if not parameter.requires_grad
    ]
    assert all(
        name.startswith(("policy.gru", "policy.enc_low", "policy.grid_conv",
                         "policy.grid_projection"))
        for name in frozen
    )


def test_alpha_zero_objective_is_the_positive_mean_and_negatives_are_inert():
    positives = _positives()
    negatives = _negatives()
    config = UPD.UpdateConfig(alpha=0.0, learning_rate=1.0e-4, seed=2)

    with_negatives = _adapter(seed=1)
    metrics = UPD.expansion_update(
        with_negatives, copy.deepcopy(positives), copy.deepcopy(negatives),
        config, round_index=1,
    )
    without_negatives = _adapter(seed=1)
    control = UPD.expansion_update(
        without_negatives, copy.deepcopy(positives), [], config, round_index=1,
    )
    assert metrics["objective_mean"] == pytest.approx(
        metrics["positive_loss_mean"], abs=1.0e-12,
    )
    assert metrics["negative_loss_mean"] is not None
    # Zero negative gradient: both runs end with bitwise-identical parameters.
    for (name_a, a), (name_b, b) in zip(
        with_negatives.named_parameters(), without_negatives.named_parameters(),
    ):
        assert name_a == name_b
        assert torch.equal(a, b)
    assert control["negative_loss_mean"] is None


def test_objective_obeys_the_positive_minus_alpha_negative_identity():
    positives = _positives()
    negatives = _negatives()
    results = {}
    for alpha in (0.0, 0.5):
        adapter = _deterministic_adapter(seed=1)
        results[alpha] = UPD.expansion_update(
            adapter, copy.deepcopy(positives), copy.deepcopy(negatives),
            UPD.UpdateConfig(alpha=alpha, learning_rate=1.0e-5,
                             batch_size=64, seed=2),
            round_index=1,
        )
    for alpha, metrics in results.items():
        assert metrics["objective_mean"] == pytest.approx(
            metrics["positive_loss_mean"] - alpha * metrics["negative_loss_mean"],
            abs=1.0e-9,
        )
    # Losses are recorded before the single step, so d(objective)/d(alpha)
    # is exactly minus the shared negative mean.
    assert results[0.0]["positive_loss_mean"] == pytest.approx(
        results[0.5]["positive_loss_mean"], abs=1.0e-9,
    )
    assert results[0.5]["objective_mean"] - results[0.0]["objective_mean"] == (
        pytest.approx(-0.5 * results[0.5]["negative_loss_mean"], abs=1.0e-9)
    )
    assert results[0.5]["grad_cosine"] is not None
    assert results[0.5]["positive_grad_norm"] > 0.0
    assert results[0.5]["negative_grad_norm"] > 0.0


def test_full_set_mean_gives_duplicated_negatives_no_extra_mass():
    positives = _positives()
    negatives = _negatives()
    single = UPD.expansion_update(
        _deterministic_adapter(seed=1), copy.deepcopy(positives),
        copy.deepcopy(negatives),
        UPD.UpdateConfig(alpha=0.5, learning_rate=1.0e-5, seed=2),
        round_index=1,
    )
    doubled = UPD.expansion_update(
        _deterministic_adapter(seed=1), copy.deepcopy(positives),
        copy.deepcopy(negatives) + copy.deepcopy(negatives),
        UPD.UpdateConfig(alpha=0.5, learning_rate=1.0e-5, seed=2),
        round_index=1,
    )
    # Duplicating the D- set is mass-invariant analytically; float32 reduction
    # order across torch builds shifts the mean by ~1e-7, so bound only that.
    assert doubled["negative_loss_mean"] == pytest.approx(
        single["negative_loss_mean"], abs=1.0e-6,
    )
    assert doubled["objective_mean"] == pytest.approx(
        single["objective_mean"], abs=1.0e-6,
    )
    assert doubled["in_archive_duplicate_rows"] == 2
    assert doubled["unique_negative_samples"] == 2
    assert doubled["negative_count"] == 4


def test_exposure_passes_are_audited_truthfully():
    positives = _positives(6)
    metrics = UPD.expansion_update(
        _deterministic_adapter(seed=1), copy.deepcopy(positives), _negatives(),
        UPD.UpdateConfig(alpha=0.5, learning_rate=1.0e-6, batch_size=4,
                         exposure_passes=4, seed=2),
        round_index=1,
    )
    assert metrics["exposure_passes_declared"] == 4
    assert metrics["exposure_passes_completed"] == 4
    assert metrics["unique_positive_samples"] == 6
    assert metrics["positive_exposures"] == 24
    assert metrics["duplicate_exposures"] == 24 - 6
    # batch_size 4 over 6 rows -> 2 steps per pass, 4 passes.
    assert metrics["steps"] == metrics["adam_steps"] == 8
    assert metrics["negative_exposures"] == 8 * 2
    assert len(metrics["per_pass_positive_loss"]) == 4
    assert len(metrics["per_pass_grad_norm"]) == 4
    # Passes reshuffle deterministically but distinctly.
    single_pass = UPD.expansion_update(
        _deterministic_adapter(seed=1), copy.deepcopy(positives), _negatives(),
        UPD.UpdateConfig(alpha=0.5, learning_rate=1.0e-6, batch_size=4,
                         exposure_passes=1, seed=2),
        round_index=1,
    )
    assert single_pass["positive_exposures"] == 6
    assert single_pass["duplicate_exposures"] == 0


def test_persistent_optimizer_must_hold_the_declared_surface():
    adapter = _adapter(seed=5)
    foreign = torch.optim.Adam(
        [torch.nn.Parameter(torch.zeros(3))], lr=1.0e-5,
    )
    with pytest.raises(RuntimeError, match="persistent optimizer"):
        UPD.expansion_update(
            adapter, _positives(), [],
            UPD.UpdateConfig(seed=2), round_index=1, optimizer=foreign,
        )
    parameters, _ = UPD.configure_trainable(adapter)
    optimizer = torch.optim.Adam(parameters, lr=1.0e-5)
    first = UPD.expansion_update(
        adapter, _positives(), [], UPD.UpdateConfig(seed=2),
        round_index=1, optimizer=optimizer,
    )
    second = UPD.expansion_update(
        adapter, _positives(), [], UPD.UpdateConfig(seed=2),
        round_index=2, optimizer=optimizer,
    )
    assert first["accepted"] and second["accepted"]
    # Momentum persisted: Adam state step count accumulated across rounds.
    steps = {
        int(state["step"])
        for state in optimizer.state.values() if "step" in state
    }
    assert steps == {first["steps"] + second["steps"]}


def test_update_moves_every_trainable_parameter_and_freezes_encoders():
    adapter = _adapter(seed=3)
    before_trainable = {
        name: parameter.detach().clone()
        for name, parameter in adapter.named_parameters()
    }
    encoder_before = UPD.encoder_state_sha256(adapter)
    metrics = UPD.expansion_update(
        adapter, _positives(), _negatives(),
        UPD.UpdateConfig(alpha=0.1, learning_rate=1.0e-3, seed=2),
        round_index=1,
    )
    assert metrics["accepted"] and metrics["finite"]
    assert metrics["steps"] >= 1
    encoder_after = UPD.encoder_state_sha256(adapter)
    assert encoder_after == encoder_before
    assert metrics["encoder_state_sha256_before"] == encoder_before
    for name, parameter in adapter.named_parameters():
        if name in UPD.FROZEN_TRAINABLE_NAMES:
            assert not torch.equal(parameter, before_trainable[name]), name
        else:
            assert torch.equal(parameter, before_trainable[name]), name


def test_role_validation_fails_closed():
    config = UPD.UpdateConfig()
    with pytest.raises(ValueError, match="at least one D"):
        UPD.expansion_update(_adapter(), [], _negatives(), config, round_index=1)
    bad_positive = _row(0, role="positive", valid=False)
    with pytest.raises(ValueError, match="exact positives"):
        UPD._validate_roles([bad_positive], [])
    bad_negative = _row(1, role="negative", negative_reason="all_negative_nvp",
                        valid=True)
    with pytest.raises(ValueError, match="disagrees"):
        UPD._validate_roles([], [bad_negative])
    undeclared = _row(2, role="negative", negative_reason="oops")
    with pytest.raises(ValueError, match="declared negative reason"):
        UPD._validate_roles([], [undeclared])


def test_rejected_updates_restore_parameters():
    adapter = _adapter(seed=4)
    before = {
        name: parameter.detach().clone()
        for name, parameter in adapter.named_parameters()
    }
    metrics = UPD.expansion_update(
        adapter, _positives(), [],
        UPD.UpdateConfig(alpha=0.0, learning_rate=50.0,
                         max_relative_parameter_drift=1.0e-8, seed=2),
        round_index=1,
    )
    assert not metrics["accepted"]
    for name, parameter in adapter.named_parameters():
        assert torch.equal(parameter, before[name]), name


def test_checkpoint_payload_keeps_the_strict_raw_evaluable_schema(tmp_path):
    adapter = _adapter()
    payload = UPD.checkpoint_payload(
        adapter, parent_checkpoint_sha256="a" * 64,
        pretrained_checkpoint_sha256="b" * 64, round_index=1, alpha=0.05,
    )
    assert payload["scientific_status"] == "SFM2_PREDICTIVE_EXPANSION_ROUND"
    assert payload["promotable"] is False
    path = tmp_path / "round.pt"
    torch.save(payload, path)
    policy, loaded = GPS.load_sfm_hp100_policy(str(path), device="cpu")
    assert loaded["alpha"] == 0.05
    assert policy.config() == GPS._canonical_config(loaded["config"])


def test_reduced_scope_declares_blocks_and_head_only():
    adapter = _adapter()
    parameters, names = UPD.configure_trainable(
        adapter, UPD.REDUCED_OPTIMIZER_SCOPE,
    )
    assert tuple(names) == UPD.REDUCED_TRAINABLE_NAMES
    assert not any(name.startswith("policy.trunk.inp") for name in names)
    assert sum(parameter.numel() for parameter in parameters) == 269_332
    # The full surface is unchanged by the reduced declaration.
    parameters, names = UPD.configure_trainable(adapter, UPD.OPTIMIZER_SCOPE)
    assert tuple(names) == UPD.FROZEN_TRAINABLE_NAMES
    assert sum(parameter.numel() for parameter in parameters) == 327_956
    with pytest.raises(ValueError, match="undeclared optimizer scope"):
        UPD.configure_trainable(adapter, "head_only")
    with pytest.raises(ValueError, match="optimizer_scope"):
        UPD.UpdateConfig(optimizer_scope="head_only").validate()


def test_reduced_scope_freezes_trunk_inp_bitwise_while_blocks_and_head_move():
    adapter = _adapter(seed=6)
    before = {
        name: parameter.detach().clone()
        for name, parameter in adapter.named_parameters()
    }
    frozen_before = UPD.frozen_surface_sha256(
        adapter, UPD.REDUCED_OPTIMIZER_SCOPE,
    )
    assert "trunk_inp" in frozen_before
    assert "trunk_inp" not in UPD.frozen_surface_sha256(
        adapter, UPD.OPTIMIZER_SCOPE,
    )
    metrics = UPD.expansion_update(
        adapter, _positives(), _negatives(),
        UPD.UpdateConfig(alpha=0.1, learning_rate=1.0e-3,
                         optimizer_scope=UPD.REDUCED_OPTIMIZER_SCOPE, seed=2),
        round_index=1,
    )
    assert metrics["accepted"] and metrics["finite"]
    assert metrics["optimizer_scope"] == UPD.REDUCED_OPTIMIZER_SCOPE
    assert metrics["trainable_parameters"] == 269_332
    assert metrics["encoder_state_sha256_after"] == frozen_before
    for name, parameter in adapter.named_parameters():
        if name in UPD.REDUCED_TRAINABLE_NAMES:
            assert not torch.equal(parameter, before[name]), name
        else:
            # trunk.inp and every condition encoder stay bitwise frozen.
            assert torch.equal(parameter, before[name]), name


def test_minimal_scope_declares_last_block_and_head_only():
    adapter = _adapter()
    parameters, names = UPD.configure_trainable(
        adapter, UPD.MINIMAL_OPTIMIZER_SCOPE,
    )
    assert tuple(names) == UPD.MINIMAL_TRAINABLE_NAMES
    assert len(names) == 8
    assert not any(
        name.startswith(("policy.trunk.inp", "policy.trunk.blocks.0"))
        for name in names
    )
    assert sum(parameter.numel() for parameter in parameters) == 137_236
    # The wider surfaces are unchanged by the minimal declaration.
    parameters, names = UPD.configure_trainable(adapter, UPD.OPTIMIZER_SCOPE)
    assert sum(parameter.numel() for parameter in parameters) == 327_956


def test_minimal_scope_freezes_inp_and_block0_bitwise_while_block1_head_move():
    adapter = _adapter(seed=7)
    before = {
        name: parameter.detach().clone()
        for name, parameter in adapter.named_parameters()
    }
    frozen_before = UPD.frozen_surface_sha256(
        adapter, UPD.MINIMAL_OPTIMIZER_SCOPE,
    )
    assert "trunk_inp" in frozen_before
    assert "trunk_block_0" in frozen_before
    assert "trunk_block_0" not in UPD.frozen_surface_sha256(
        adapter, UPD.REDUCED_OPTIMIZER_SCOPE,
    )
    metrics = UPD.expansion_update(
        adapter, _positives(), _negatives(),
        UPD.UpdateConfig(alpha=0.1, learning_rate=1.0e-3,
                         optimizer_scope=UPD.MINIMAL_OPTIMIZER_SCOPE, seed=2),
        round_index=1,
    )
    assert metrics["accepted"] and metrics["finite"]
    assert metrics["optimizer_scope"] == UPD.MINIMAL_OPTIMIZER_SCOPE
    assert metrics["trainable_parameters"] == 137_236
    assert metrics["encoder_state_sha256_after"] == frozen_before
    moved = 0
    for name, parameter in adapter.named_parameters():
        if name in UPD.MINIMAL_TRAINABLE_NAMES:
            assert not torch.equal(parameter, before[name]), name
            moved += 1
        else:
            # trunk.inp, blocks[0], and every condition encoder stay
            # bitwise frozen.
            assert torch.equal(parameter, before[name]), name
    assert moved == 8
