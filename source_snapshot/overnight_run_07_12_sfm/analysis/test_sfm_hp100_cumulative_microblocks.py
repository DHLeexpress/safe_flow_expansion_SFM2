import pytest
import torch
from torch import nn

import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_cumulative_microblocks as CUM
import sfm_hp100_cumulative_microblock_sweep as DRIVER


def _row(group, index, *, gamma=0.1, replica=0, micro_round=1):
    return {
        "group": group,
        "gamma": float(gamma),
        "replica": int(replica),
        "lineage": f"g{gamma:g}:rep{replica:02d}",
        "microcycle": int(micro_round),
        "step": int(index),
        "context": torch.tensor([float(index), float(gamma)]),
        "candidate": torch.full((10, 2), float(index + 1)),
        "flow_base": torch.full((10, 2), -float(index + 1)),
    }


def test_optimizer_scopes_freeze_every_conditioner_and_have_exact_counts():
    for scope, expected_blocks, expected_count in (
        ("last_block_and_head", {1}, 137_236),
        ("last_two_blocks_and_head", {0, 1}, 269_332),
    ):
        adapter = PORT.HP100ExpansionPolicy(GPS.GridSFMHP100FlowPolicy())
        parameters, names = CUM._configure_trainable(adapter, scope)
        assert sum(parameter.numel() for parameter in parameters) == expected_count
        assert set(names) == {
            name for name, parameter in adapter.policy.named_parameters()
            if parameter.requires_grad
        }
        assert all(
            name.startswith("head.")
            or any(name.startswith(f"trunk.blocks.{index}.") for index in expected_blocks)
            for name in names
        )
        assert not any(
            parameter.requires_grad for name, parameter in adapter.policy.named_parameters()
            if name.startswith(("gru.", "enc_low.", "grid_", "trunk.inp."))
        )


class _TinyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))

    def cfm_loss(self, contexts, candidates, reduction="none"):
        del contexts
        target = candidates.reshape(len(candidates), -1).mean(dim=1)
        loss = (self.weight - target).square()
        return loss if reduction == "none" else loss.mean()


def test_stratified_replay_has_exact_dynamic_steps_and_persistent_adam_state():
    groups = {
        "P1": [_row("P1", index) for index in range(33)],
        "P2": [_row("P2", index) for index in range(3)],
        "Ncausal": [_row("Ncausal", index) for index in range(2)],
        "Dminus": [_row("Dminus", index) for index in range(1)],
    }
    adapter = _TinyAdapter()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=3.0e-5)

    first = CUM.stratified_signed_update(
        adapter, optimizer, groups, p2_weight=2.0, negative_alpha=0.01,
        passes=10, seed=17, micro_round=1,
    )
    assert first["N_batches"] == 2
    assert first["N_Adam_steps"] == 20
    assert first["exposure"]["P1"]["total_draws"] == 2 * 32 * 10
    assert first["exposure"]["P2"]["total_draws"] == 2 * 16 * 10
    assert first["exposure"]["Ncausal"]["total_draws"] == 2 * 12 * 10
    assert first["exposure"]["Dminus"]["total_draws"] == 2 * 4 * 10
    assert all(
        report["all_unique_rows_covered_each_pass"]
        for report in first["exposure"].values()
    )
    step_after_first = int(next(iter(optimizer.state.values()))["step"])
    assert step_after_first == 20

    second = CUM.stratified_signed_update(
        adapter, optimizer, groups, p2_weight=2.0, negative_alpha=0.01,
        passes=10, seed=17, micro_round=2,
    )
    assert second["N_Adam_steps"] == 20
    assert int(next(iter(optimizer.state.values()))["step"]) == 40


def test_cyclic_epoch_never_drops_a_unique_row_and_oversamples_sparse_rows():
    dense = CUM._cyclic_epoch_indices(33, 64, seed=2)
    sparse = CUM._cyclic_epoch_indices(2, 24, seed=2)
    assert set(dense.tolist()) == set(range(33))
    assert set(sparse.tolist()) == {0, 1}
    assert len(sparse) == 24
    with pytest.raises(ValueError, match="cannot omit"):
        CUM._cyclic_epoch_indices(5, 4, seed=2)


class _Reference(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def embed(self, contexts, candidates, base=None):
        del contexts
        return torch.stack((
            candidates.reshape(len(candidates), -1).mean(dim=1),
            base.reshape(len(base), -1).mean(dim=1),
        ), dim=1)


def test_gp_uses_recent_two_rounds_and_falls_back_only_for_missing_gamma():
    calibration = {
        gamma: torch.full((2, 2), float(gamma)) for gamma in CUM.GAMMAS
    }
    previous = {
        gamma: torch.full((3, 2), float(gamma) + 10) for gamma in CUM.GAMMAS
    }
    round1 = [_row("P1", 0, gamma=0.1, micro_round=1)]
    round2 = [_row("P2", 1, gamma=0.3, micro_round=2)]
    round3 = [_row("P1", 2, gamma=0.5, micro_round=3)]

    support, report = CUM.gp_support_for_round(
        reference=_Reference(), calibration_support=calibration,
        previous_support=previous, history=[round1, round2, round3],
        cap=2_688, replay_rounds=2,
    )
    assert report["candidate_recent_positive_rows"] == 2
    assert report["rows_by_gamma"]["0.1"] == 0
    assert report["rows_by_gamma"]["0.3"] == 1
    assert report["rows_by_gamma"]["0.5"] == 1
    assert torch.equal(support[0.1], previous[0.1])
    assert torch.equal(support[1.0], previous[1.0])
    assert report["fallback_by_gamma"]["0.3"] is None
    assert report["fallback_by_gamma"]["0.5"] is None
    assert report["nominal_cap_per_active_gamma"] == 672


def test_driver_declares_exact_requested_eight_cells():
    cells = DRIVER.declared_cells()
    assert len(cells) == 8
    assert {cell.optimizer_scope for cell in cells} == {
        "last_block_and_head", "last_two_blocks_and_head",
    }
    assert {
        (cell.config.learning_rate, cell.config.p2_weight, cell.config.negative_alpha)
        for cell in cells
    } == {
        (3.0e-5, 2.0, 0.01), (3.0e-5, 2.0, 0.05),
        (1.0e-4, 2.0, 0.01), (1.0e-4, 2.0, 0.05),
    }
