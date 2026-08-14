from dataclasses import replace

import pytest
import torch
from torch import nn

import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_exhaustive_hybrid as HYBRID
from sfm_hp100_ball_core.expansion import Verification


def test_default_config_is_the_declared_four_by_four_hybrid_round():
    config = HYBRID.HybridConfig()
    config.validate()

    assert config.gammas == (0.1, 0.3, 0.5, 1.0)
    assert config.lineages_per_gamma == 4
    assert len(HYBRID._lineage_keys(config)) == 16
    assert (config.max_steps, config.max_repair_batches) == (180, 32)
    assert (config.K, config.B) == (64, 16)
    assert (
        config.repair_base_std_start,
        config.repair_base_std_step,
    ) == (1.0, 0.1)
    assert (config.p1_learning_rate, config.p2_learning_rate) == (1.0e-5, 1.0e-3)
    assert config.negative_alpha_base == 1.0e-3
    assert config.ess_target == 0.1
    assert config.gp_buffer_cap == 2_688


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"K": 32}, "fixes K=64 and B=16"),
        ({"B": 4}, "fixes K=64 and B=16"),
        ({"repair_base_std_step": 0.2}, "repair schedule"),
        ({"ess_target": 0.0}, "ESS target"),
        ({"p2_learning_rate": 0.0}, "learning rates"),
        ({"max_relative_parameter_drift": 1.0}, "drift gate"),
    ],
)
def test_config_rejects_protocol_drift(changes, message):
    with pytest.raises(ValueError, match=message):
        replace(HYBRID.HybridConfig(), **changes).validate()


def test_scene_and_sampling_seeds_are_deterministic_and_coordinate_separated():
    low = HYBRID.LineageKey(0.1, 2)
    high = HYBRID.LineageKey(1.0, 2)

    # Matching replicas share the pedestrian scene across gamma.
    assert HYBRID._scene_seed(7, low.replica) == HYBRID._scene_seed(7, high.replica)
    assert HYBRID._scene_seed(7, 2) != HYBRID._scene_seed(7, 3)

    coordinates = dict(microcycle=3, attempt=4)
    seed = HYBRID._sampling_seed(7, "repair", low, 5, **coordinates)
    assert seed == HYBRID._sampling_seed(7, "repair", low, 5, **coordinates)
    assert seed != HYBRID._sampling_seed(7, "repair", high, 5, **coordinates)
    assert seed != HYBRID._sampling_seed(7, "raw_gather", low, 5, **coordinates)
    assert seed != HYBRID._sampling_seed(
        7, "repair", low, 5, microcycle=3, attempt=5,
    )


def test_max_margin_uses_only_exact_rows_and_native_cost_tie_break():
    results = [
        Verification(False, True, 5.0, -100.0, step_margin=5.0),
        Verification(True, True, 1.0, 20.0, step_margin=0.3),
        Verification(True, True, 1.0, -2.0, step_margin=0.3),
        Verification(
            True, True, 1.0, -100.0, progress_eligible=False,
            step_margin=0.9,
        ),
    ]
    # A progress flag is not an execution gate in exhaustive-hybrid repair.
    assert HYBRID._chosen_max_margin(results) == 3
    assert HYBRID._chosen_max_margin([
        Verification(False, True, 1.0, 0.0, step_margin=1.0),
    ]) is None


def test_calibration_support_is_gamma_matched_and_moved_to_requested_device():
    features = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    calibration = {
        "selection": [
            {"gamma": 0.1}, {"gamma": 0.3}, {"gamma": 0.1},
            {"gamma": 1.0}, {"gamma": 0.5},
        ],
    }
    support = HYBRID._calibration_support_by_gamma(
        features, calibration, (0.1, 0.3, 0.5, 1.0),
        device=torch.device("cpu"),
    )
    assert torch.equal(support[0.1], features[[0, 2]])
    assert torch.equal(support[0.3], features[[1]])
    assert torch.equal(support[0.5], features[[4]])
    assert torch.equal(support[1.0], features[[3]])


def test_calibration_support_rejects_missing_declared_gamma():
    with pytest.raises(ValueError, match="has no calibrated reference rows"):
        HYBRID._calibration_support_by_gamma(
            torch.ones(1, 4), {"selection": [{"gamma": 0.1}]},
            (0.1, 0.3), device=torch.device("cpu"),
        )


def test_checkpoint_payload_roundtrips_through_strict_hp100_loader(tmp_path):
    policy = GPS.GridSFMHP100FlowPolicy()
    adapter = PORT.HP100ExpansionPolicy(policy)
    payload = HYBRID._checkpoint_payload(
        adapter, parent_checkpoint_sha="b" * 64,
        root_pretrained_sha="a" * 64,
        optimizer_scope="head_only", microcycle=1, promotable=True,
    )
    path = tmp_path / "hybrid.pt"
    torch.save(payload, path)
    restored, restored_payload = GPS.load_sfm_hp100_policy(path)
    assert restored_payload["scientific_status"] == (
        "HP100_EXHAUSTIVE_HYBRID_ROUND1_COMMITTED"
    )
    assert restored_payload["promotable"] is True
    assert restored_payload["parent_checkpoint_sha256"] == "b" * 64
    assert restored_payload["pretrained_checkpoint_sha256"] == "a" * 64
    assert restored.config() == policy.config()
    for key, value in policy.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value)


def _record(group, result):
    return HYBRID._record(
        group=group,
        key=HYBRID.LineageKey(0.3, 1),
        scenario_id=123,
        microcycle=2,
        step=4,
        context=torch.tensor([0.3, 4.0]),
        candidate=torch.full((10, 2), 0.25),
        flow_base=torch.full((10, 2), -0.5),
        result=result,
        base_std=1.2,
        repair_attempt=2 if group == "P2" else None,
    )


def test_record_groups_preserve_paired_sample_and_exact_label_invariants():
    positive = Verification(
        True, True, 0.4, 3.0, progress=0.8, step_margin=0.2,
    )
    negative = Verification(
        False, False, -0.1, 5.0, progress=0.1, step_margin=-0.2,
    )

    p1 = _record("P1", positive)
    p2 = _record("P2", positive)
    dminus = _record("Dminus", negative)
    assert (p1["group"], p2["group"], dminus["group"]) == (
        "P1", "P2", "Dminus",
    )
    assert p2["lineage"] == "g0.3:rep01"
    assert p2["verification"]["valid"] is True
    assert dminus["verification"]["valid"] is False
    assert torch.equal(p2["candidate"], torch.full((10, 2), 0.25))
    assert torch.equal(p2["flow_base"], torch.full((10, 2), -0.5))

    with pytest.raises(ValueError, match="must be exact-positive"):
        _record("P1", negative)
    with pytest.raises(ValueError, match="must be a resolved exact-negative"):
        _record("Dminus", positive)
    with pytest.raises(ValueError, match="must be a resolved exact-negative"):
        _record("Dminus", replace(negative, error=True))
    with pytest.raises(ValueError, match="unknown exhaustive-hybrid group"):
        _record("teacher", positive)


def _gp_row(group, gamma, replica, microcycle, step):
    return {
        "group": group,
        "gamma": float(gamma),
        "replica": int(replica),
        "microcycle": int(microcycle),
        "step": int(step),
    }


def test_gp_selection_round_robins_gamma_and_lineage_and_excludes_negative_rows():
    rows = [
        _gp_row("Dminus", 0.1, 0, 0, 0),  # index 0 is never eligible.
        _gp_row("P2", 0.1, 0, 1, 1),      # index 1, second for this lineage.
        _gp_row("P1", 0.1, 0, 0, 0),      # index 2, first for this lineage.
        _gp_row("P1", 0.1, 1, 0, 0),      # index 3.
        _gp_row("P1", 0.3, 0, 0, 0),      # index 4.
        _gp_row("P1", 0.3, 1, 0, 0),      # index 5.
        _gp_row("P2", 0.3, 1, 1, 1),      # index 6, second for this lineage.
    ]

    selected = HYBRID._balanced_gp_indices(rows, cap=6)
    assert selected[:4] == [2, 3, 4, 5]
    assert selected[4:] == [1, 6]
    assert 0 not in selected
    assert len(selected) == len(set(selected))


def test_gp_selection_is_temporally_uniform_inside_each_lineage():
    rows = [
        _gp_row("P1", 0.1, 0, microcycle=1, step=step)
        for step in range(10)
    ]
    assert HYBRID._balanced_gp_indices(rows, cap=4) == [0, 3, 6, 9]


class _TinyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Parameter(torch.tensor(3.0), requires_grad=False)
        self.head = nn.Parameter(torch.tensor(1.0))

    def cfm_loss(self, contexts, candidates, reduction="mean"):
        target = candidates.reshape(len(candidates), -1).mean(dim=1)
        prediction = self.head + (contexts.reshape(len(contexts), -1)[:, 0] * 0.0)
        values = (prediction - target).square()
        return values if reduction == "none" else values.mean()


def _update_rows():
    def row(group, target):
        return {
            "group": group,
            "gamma": 0.1,
            "replica": 0,
            "lineage": "g0.1:rep00",
            "microcycle": 1,
            "step": 0,
            "context": torch.tensor([0.1]),
            "candidate": torch.full((10, 2), float(target)),
        }

    return [row("P2", 2.0), row("Dminus", -1.0), row("P1", 0.5)]


def test_phased_update_moves_only_trainable_scope_and_reports_lineage_scaling():
    adapter = _TinyAdapter()
    head_before = adapter.head.detach().clone()
    encoder_before = adapter.encoder.detach().clone()
    config = replace(
        HYBRID.HybridConfig(),
        p2_learning_rate=1.0e-2,
        p1_learning_rate=1.0e-3,
        negative_alpha_base=1.0e-2,
        max_relative_parameter_drift=0.5,
    )

    report = HYBRID.phased_update(adapter, _update_rows(), config, microcycle=1)

    assert report["accepted"] is True
    assert report["phase_order"] == [
        "P2_plus_signed_Dminus", "P1_retention_anchor",
    ]
    assert (report["P2_Dminus_steps"], report["P1_steps"]) == (1, 1)
    assert report["lineage_negative_scaling"] == [{
        "lineage": "g0.1:rep00",
        "P1": 1,
        "P2": 1,
        "Dminus": 1,
        "positive_to_negative_ratio": 2.0,
        "negative_alpha_effective": 0.02,
    }]
    assert not torch.equal(adapter.head, head_before)
    assert torch.equal(adapter.encoder, encoder_before)


def test_phased_update_rolls_back_an_excessive_trainable_drift():
    adapter = _TinyAdapter()
    head_before = adapter.head.detach().clone()
    encoder_before = adapter.encoder.detach().clone()
    config = replace(
        HYBRID.HybridConfig(),
        p2_learning_rate=0.1,
        p1_learning_rate=0.01,
        negative_alpha_base=0.0,
        max_relative_parameter_drift=1.0e-8,
    )

    report = HYBRID.phased_update(adapter, _update_rows(), config, microcycle=1)

    assert report["finite"] is True
    assert report["accepted"] is False
    assert report["relative_parameter_drift"] > report["drift_gate"]
    assert torch.equal(adapter.head, head_before)
    assert torch.equal(adapter.encoder, encoder_before)
