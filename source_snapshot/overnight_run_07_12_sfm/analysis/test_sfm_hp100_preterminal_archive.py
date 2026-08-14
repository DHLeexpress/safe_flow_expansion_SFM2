from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
from torch import nn

from sfm_hp100_ball_core.expansion import (
    ExpansionConfig,
    QueryRecord,
    RBFPosterior,
    Verification,
    _sliding_success_gp_rows,
    run_safe_expansion,
)


class _Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.0))

    def sample_with_base(self, context, count, generator, base_std=1.0):
        del generator, base_std
        index = torch.arange(count, dtype=context.dtype, device=context.device)
        plans = index[:, None, None].expand(count, 3, 1).clone()
        bases = (100.0 + index)[:, None, None].expand_as(plans).clone()
        return plans, bases

    def sample(self, context, count, generator, base_std=1.0):
        return self.sample_with_base(context, count, generator, base_std)[0]

    def embed(self, context, candidates, flow_time=0.9, base=None):
        del flow_time
        if context.ndim == 1:
            context = context.unsqueeze(0).expand(len(candidates), -1)
        candidate_mean = candidates.reshape(len(candidates), -1).mean(1)
        base_mean = (
            torch.zeros_like(candidate_mean)
            if base is None
            else base.reshape(len(base), -1).mean(1)
        )
        return torch.stack((context[:, 0], candidate_mean, base_mean), dim=1)

    def cfm_loss(
        self, contexts, candidates, reduction="none", loss_mask=None,
    ):
        del contexts, loss_mask
        values = (
            candidates.reshape(len(candidates), -1).mean(1) - self.weight
        ).square()
        if reduction == "none":
            return values
        if reduction == "mean":
            return values.mean()
        raise ValueError(reduction)


def test_precomputed_covariance_acquisition_is_fixed_seed_equivalent():
    features = torch.tensor([
        [1.0, 0.0], [0.7, 0.3], [0.0, 1.0], [-0.6, 0.8],
    ])
    gp = RBFPosterior(lengthscale=0.5, noise=0.01)
    gp.set_buffer(torch.tensor([[1.0, 1.0], [-1.0, 0.0]]))
    covariance = gp.covariance(features)
    untouched = covariance.clone()
    direct_rng = torch.Generator().manual_seed(123)
    cached_rng = torch.Generator().manual_seed(123)

    direct = gp.acquire(features, 3, 0.2, direct_rng)
    cached = gp.acquire_from_covariance(
        covariance, 3, 0.2, cached_rng, sampling_device=features.device,
    )
    assert cached == direct
    assert torch.equal(covariance, untouched)
    assert torch.allclose(
        gp.sigma(features),
        (torch.diagonal(covariance) - gp.noise).clamp_min(0).sqrt(),
    )


@dataclass
class _State:
    step: int = 0


class _PreterminalTask:
    def reset(self, gamma, episode, seed):
        del gamma, episode, seed
        return _State()

    def context(self, state, gamma):
        return torch.tensor([float(state.step), float(gamma)])

    def verify(self, context, candidates, gamma):
        del gamma
        valid = int(context[0]) == 0
        return [
            Verification(
                valid=valid,
                hp_eligible=valid,
                margin=float(local),
                execution_cost=float(local),
                progress_eligible=True,
                error=False,
                step_margin=float(local),
            )
            for local in range(len(candidates))
        ]

    def advance(self, state, candidate):
        del candidate
        state.step += 1
        return state

    def terminal(self, state):
        del state
        return None


def _config(**changes):
    values = dict(
        rounds=2,
        gammas=(0.5,),
        parallel_episodes=1,
        verifier_workers=1,
        max_retry_batches=1,
        max_steps=3,
        K=2,
        B=2,
        batch_size=2,
        inner_steps=1,
        microbatch_repeats=1,
        learning_rate=1.0e-4,
        replay_rounds=1,
        gp_buffer_cap=2,
        gp_noise=1.0e-2,
        rbf_lengthscale=1.0,
        beta=0.1,
        negative_alpha=0.1,
        replay_selector="uniform",
        execution_rule="min_cost",
        archive_rule="preterminal_resolved_queries",
        replay_acceptance="safety_valid",
        paired_noised_representation=True,
        gp_reference_mode="sliding_positive_per_gamma_current_phi",
    )
    values.update(changes)
    return ExpansionConfig(**values)


def test_preterminal_archive_retains_positive_before_nvp_and_truthful_negative(
    tmp_path,
):
    output = tmp_path / "run"
    manifest = run_safe_expansion(
        _Policy(), _PreterminalTask(), output, config=_config(),
    )
    archive = torch.load(output / "query_archive.pt", weights_only=False)
    evidence = torch.load(output / "gp_evidence.pt", weights_only=False)

    assert manifest["D"] == len(archive) == 8
    assert manifest["D_plus"] == 4
    assert manifest["D_minus"] == 4
    assert sum(not row.verification.valid for row in archive) == 4
    assert {row.round for row in evidence} == {1, 2}
    assert len(evidence) == 4
    assert all(row.verification.valid for row in evidence)

    round1 = [row for row in archive if row.round == 1]
    assert sum(row.verification.valid for row in round1) == 2
    assert sum(row.executed for row in round1) == 1
    assert {row.window_start for row in round1 if row.verification.valid} == {0}
    assert all(
        row.nvp_context
        for row in round1 if not row.verification.valid
    )
    assert all(row.flow_base is not None for row in archive)
    assert all(row.trajectory_id is not None for row in archive)
    assert len({row.window_id for row in archive}) == len(archive)

    assert manifest["rounds"][0]["gp_buffer"] == 0
    assert manifest["rounds"][1]["gp_buffer"] == 2
    assert manifest["rounds"][1]["negative_loss"] is not None


def _record(*, gamma, trajectory, start, round_i=1, valid=True):
    return QueryRecord(
        round=round_i,
        gamma=float(gamma),
        episode=int(trajectory),
        context_id=100 * int(trajectory) + int(start),
        context=torch.tensor([float(start)]),
        candidate=torch.tensor([[float(start)]]),
        verification=Verification(
            valid=bool(valid), hp_eligible=True, margin=0.0,
            execution_cost=0.0, error=False,
        ),
        trajectory_id=f"g{gamma}:trajectory{trajectory}",
        window_id=f"g{gamma}:trajectory{trajectory}:w{start}",
        window_start=int(start),
    )


def test_sliding_gp_is_gamma_lineage_time_balanced_and_previous_round_only():
    rows = []
    for gamma in (0.1, 0.5):
        for trajectory in range(3):
            rows.extend(
                _record(gamma=gamma, trajectory=trajectory, start=start)
                for start in range(9)
            )
        rows.append(
            _record(gamma=gamma, trajectory=99, start=0, valid=False)
        )
        rows.append(
            _record(
                gamma=gamma, trajectory=100, start=0, round_i=2,
            )
        )

    selected = _sliding_success_gp_rows(
        rows, (0.1, 0.5), 12, through_round=1,
        selector="trajectory_uniform",
    )
    assert len(selected) == 12
    assert {gamma: sum(row.gamma == gamma for row in selected)
            for gamma in (0.1, 0.5)} == {0.1: 6, 0.5: 6}
    assert all(row.verification.valid and row.round == 1 for row in selected)
    for gamma in (0.1, 0.5):
        gamma_rows = [row for row in selected if row.gamma == gamma]
        assert {
            trajectory: sum(row.episode == trajectory for row in gamma_rows)
            for trajectory in range(3)
        } == {0: 2, 1: 2, 2: 2}
        assert min(row.window_start for row in gamma_rows) == 0
        assert max(row.window_start for row in gamma_rows) == 8


def test_sliding_gp_does_not_turn_many_positive_queries_into_one_stage_bias():
    rows = []
    for local in range(20):
        row = _record(gamma=0.5, trajectory=0, start=0)
        row.candidate = torch.tensor([[float(local)]])
        row.window_id = f"g0.5:trajectory0:w0:q{local}"
        rows.append(row)
    rows.extend(
        _record(gamma=0.5, trajectory=0, start=start)
        for start in range(1, 5)
    )
    selected = _sliding_success_gp_rows(
        rows, (0.5,), 4, through_round=1,
        selector="trajectory_uniform",
    )
    assert [row.window_start for row in selected] == [0, 1, 2, 4]


def test_sliding_gp_retains_older_rows_but_never_current_round():
    rows = [
        _record(gamma=0.5, trajectory=0, start=0, round_i=0),
        _record(gamma=0.5, trajectory=1, start=1, round_i=0),
        _record(gamma=0.5, trajectory=2, start=2, round_i=1),
        _record(gamma=0.5, trajectory=3, start=3, round_i=2),
    ]
    selected = _sliding_success_gp_rows(
        rows, (0.5,), 4, through_round=1,
        selector="trajectory_uniform",
    )
    assert {row.round for row in selected} == {0, 1}
    assert all(row.round < 2 for row in selected)


def test_sliding_gp_can_limit_reference_to_recent_rounds():
    rows = [
        _record(gamma=0.5, trajectory=round_i, start=round_i, round_i=round_i)
        for round_i in range(1, 5)
    ]
    selected = _sliding_success_gp_rows(
        rows, (0.5,), 4, through_round=4,
        selector="trajectory_uniform", reference_rounds=2,
    )
    assert {row.round for row in selected} == {3, 4}


def test_executed_positive_gp_uses_only_chosen_plans_and_truthful_nvp_negative(
    tmp_path,
):
    config = _config(
        archive_rule="executed_plus_nvp_negative",
        replay_acceptance="execution_eligible",
        gp_reference_mode="sliding_executed_positive_per_gamma_frozen_phi",
        gp_reference_rounds=1,
        execution_step_margin_weight=2.0,
    )
    output = tmp_path / "executed"
    manifest = run_safe_expansion(
        _Policy(), _PreterminalTask(), output, config=config,
    )
    archive = torch.load(output / "query_archive.pt", weights_only=False)
    evidence = torch.load(output / "gp_evidence.pt", weights_only=False)

    assert len(archive) == 4
    assert len(evidence) == 2
    assert all(row.verification.valid and row.executed for row in evidence)
    assert {row.window_id for row in evidence}.issubset(
        {row.window_id for row in archive}
    )
    assert all(row.trajectory_id is not None for row in evidence)
    assert all(row.window_start == 0 for row in evidence)
    assert all(row.window_id.endswith(":w000000") for row in evidence)

    negatives = [row for row in archive if not row.verification.valid]
    assert len(negatives) == 2
    assert all(row.nvp_context and not row.executed for row in negatives)
    # Native costs prefer local query 0, but cost - 2*step_margin prefers 1.
    assert all(row.verification.execution_cost == 1.0 for row in negatives)
    assert all(row.verification.step_margin == 1.0 for row in negatives)
    assert not {row.window_id for row in evidence}.intersection(
        {row.window_id for row in negatives}
    )
    assert manifest["rounds"][0]["gp_buffer"] == 0
    assert manifest["rounds"][1]["gp_buffer"] == 1
    assert manifest["gp_reference"]["reference_rounds"] == 1


def test_deliberate_gather_horizon_is_reported_as_early_cutoff(tmp_path):
    class AlwaysValid(_PreterminalTask):
        def verify(self, context, candidates, gamma):
            del context, gamma
            return [
                Verification(
                    valid=True, hp_eligible=True, margin=float(local),
                    execution_cost=float(local), progress_eligible=True,
                    error=False, step_margin=float(local),
                )
                for local in range(len(candidates))
            ]

    manifest = run_safe_expansion(
        _Policy(), AlwaysValid(), tmp_path / "cutoff",
        config=_config(
            rounds=1, max_steps=1, max_steps_status="EARLY_CUTOFF",
            archive_rule="executed_plus_nvp_negative",
            replay_acceptance="execution_eligible",
            gp_reference_mode=(
                "sliding_executed_positive_per_gamma_frozen_phi"
            ),
            negative_alpha=0.0,
        ),
    )
    assert manifest["rounds"][0]["early_cutoff"] == 1
    assert manifest["rounds"][0]["timeout"] == 0


def test_core_delegates_last_block_and_head_scope_without_refreezing(tmp_path):
    class ScopedPolicy(_Policy):
        def __init__(self):
            super().__init__()
            self.other = nn.Parameter(torch.tensor(1.0))
            self.requested_scope = None

        def expansion_optimizer_parameters(self, scope):
            self.requested_scope = scope
            self.weight.requires_grad_(True)
            self.other.requires_grad_(False)
            return [self.weight]

    policy = ScopedPolicy()
    manifest = run_safe_expansion(
        policy, _PreterminalTask(), tmp_path / "scope",
        config=_config(
            rounds=1, optimizer_scope="last_block_and_head",
        ),
    )
    assert policy.requested_scope == "last_block_and_head"
    assert policy.weight.requires_grad and not policy.other.requires_grad
    assert manifest["optimizer_scope"]["mode"] == "last_block_and_head"
    assert manifest["optimizer_scope"]["trainable_parameter_names"] == [
        "weight"
    ]


def test_preterminal_mode_cannot_silently_restore_success_only_filtering():
    _config().validate()
    with pytest.raises(ValueError, match="requires replay_acceptance=safety_valid"):
        _config(replay_acceptance="execution_eligible").validate()
    with pytest.raises(ValueError, match="preterminal_resolved_queries"):
        _config(
            archive_rule="successful_executed_windows",
            negative_alpha=0.0,
            replay_acceptance="execution_eligible",
        ).validate()
    with pytest.raises(ValueError, match="executed_plus_nvp_negative"):
        _config(
            gp_reference_mode=(
                "sliding_executed_positive_per_gamma_frozen_phi"
            ),
        ).validate()
    with pytest.raises(ValueError, match="gp_reference_rounds"):
        _config(gp_reference_rounds=0).validate()


def test_preterminal_reset_scenarios_are_paired_across_gamma(tmp_path):
    class RecordingTask(_PreterminalTask):
        def __init__(self):
            self.resets = []

        def reset(self, gamma, episode, seed):
            self.resets.append((float(gamma), int(episode), int(seed)))
            return _State()

    task = RecordingTask()
    run_safe_expansion(
        _Policy(), task, tmp_path / "paired",
        config=_config(
            rounds=1, gammas=(0.1, 0.5), parallel_episodes=2,
            max_steps=1, gp_buffer_cap=4,
        ),
    )
    by_gamma = {
        gamma: [seed for value, _, seed in task.resets if value == gamma]
        for gamma in (0.1, 0.5)
    }
    assert by_gamma[0.1] == by_gamma[0.5]
    assert len(set(by_gamma[0.1])) == 2


def test_executed_archive_pairs_scenarios_and_flow_bases_across_gamma(tmp_path):
    class RecordingTask(_PreterminalTask):
        def __init__(self):
            self.resets = []

        def reset(self, gamma, episode, seed):
            self.resets.append((float(gamma), int(episode), int(seed)))
            return _State()

    class RandomPolicy(_Policy):
        def sample_with_base(self, context, count, generator, base_std=1.0):
            base = torch.randn(
                count, 3, 1, generator=generator, dtype=context.dtype,
            ) * float(base_std)
            return base.clone(), base

    task = RecordingTask()
    events = []
    run_safe_expansion(
        RandomPolicy(), task, tmp_path / "paired_executed",
        config=_config(
            rounds=1, gammas=(0.1, 0.5), parallel_episodes=2,
            max_steps=1, gp_buffer_cap=4, negative_alpha=0.0,
            archive_rule="executed_plus_nvp_negative",
            replay_acceptance="execution_eligible",
            gp_reference_mode=(
                "sliding_executed_positive_per_gamma_frozen_phi"
            ),
        ),
        event_callback=events.append,
    )
    by_gamma = {
        gamma: [seed for value, _, seed in task.resets if value == gamma]
        for gamma in (0.1, 0.5)
    }
    assert by_gamma[0.1] == by_gamma[0.5]
    by_cell = {
        (event["gamma"], event["replica"]): event["flow_bases"]
        for event in events
    }
    for replica in (0, 1):
        assert torch.equal(
            by_cell[(0.1, replica)], by_cell[(0.5, replica)]
        )
