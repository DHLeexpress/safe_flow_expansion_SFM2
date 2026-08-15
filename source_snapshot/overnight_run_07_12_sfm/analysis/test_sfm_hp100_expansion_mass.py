"""Declared D+ mass modes: balanced/progress weighting, defaults, resume."""
import contextlib

import numpy as np
import pytest
import torch

import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_expansion_round as ROUND
import sfm_hp100_expansion_update as UPD


CTX = GPS.LOW_TOKEN + GPS.VISUAL_TOKEN


def _adapter(seed=0):
    torch.manual_seed(seed)
    return PORT.HP100ExpansionPolicy(GPS.build_sfm_hp100_policy()).eval()


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


def _row(seed, *, gamma=0.1, progress=1.0, role="positive",
         negative_reason=None):
    generator = torch.Generator().manual_seed(seed)
    return {
        "role": role,
        "gamma": gamma, "replica": 0, "lineage": f"g{gamma:g}:rep00",
        "scenario_id": 860_000 + seed, "step": seed, "attempt": 0,
        "executed": role == "positive",
        "negative_reason": negative_reason,
        "context": torch.randn(CTX + 4, generator=generator),
        "candidate": torch.randn(10, 2, generator=generator),
        "flow_base": torch.randn(10, 2, generator=generator),
        "verification": {"valid": role == "positive"},
        "prediction_audit": {"H10_goal_progress": float(progress)},
    }


def _unbalanced_positives():
    # 6 rows at gamma 0.1 versus 2 rows at gamma 1.0.
    return (
        [_row(seed, gamma=0.1, progress=1.0 + seed) for seed in range(6)]
        + [_row(seed, gamma=1.0, progress=1.0 + seed) for seed in (10, 11)]
    )


def test_per_gamma_balanced_gives_each_gamma_equal_total_mass():
    positives = _unbalanced_positives()
    weights = UPD.positive_mass_weights(positives, "per_gamma_balanced")
    assert weights.sum() == pytest.approx(1.0, abs=1.0e-12)
    gammas = np.asarray([row["gamma"] for row in positives])
    assert weights[gammas == 0.1].sum() == pytest.approx(0.5, abs=1.0e-12)
    assert weights[gammas == 1.0].sum() == pytest.approx(0.5, abs=1.0e-12)
    # Uniform inside each gamma.
    assert np.allclose(weights[gammas == 0.1], 0.5 / 6)
    assert np.allclose(weights[gammas == 1.0], 0.25)
    # Pooled reference stays uniform.
    uniform = UPD.positive_mass_weights(positives, "pooled_mean")
    assert np.allclose(uniform, 1.0 / len(positives))


def test_per_gamma_balanced_changes_the_objective_and_audits_mass():
    positives = _unbalanced_positives()
    results = {}
    for mode in ("pooled_mean", "per_gamma_balanced"):
        results[mode] = UPD.expansion_update(
            _deterministic_adapter(seed=1), list(positives), [],
            UPD.UpdateConfig(positive_mass=mode, learning_rate=1.0e-5, seed=2),
            round_index=1,
        )
    assert results["pooled_mean"]["objective_mean"] != pytest.approx(
        results["per_gamma_balanced"]["objective_mean"], abs=1.0e-9,
    )
    audit = results["per_gamma_balanced"]
    assert audit["positive_mass"] == "per_gamma_balanced"
    assert audit["positive_mass_per_gamma"] == {
        "0.1": pytest.approx(0.5), "1": pytest.approx(0.5),
    }
    stats = audit["positive_weight_stats"]
    assert stats["min"] == pytest.approx(0.5 / 6)
    assert stats["max"] == pytest.approx(0.25)
    assert results["pooled_mean"]["positive_mass"] == "pooled_mean"
    assert results["pooled_mean"]["positive_weight_stats"]["min"] == (
        pytest.approx(1.0 / len(positives))
    )


def test_progress_weighted_orders_within_gamma_and_respects_the_floor():
    positives = [
        _row(0, gamma=0.1, progress=2.0),
        _row(1, gamma=0.1, progress=1.0),
        _row(2, gamma=0.1, progress=0.0),
        _row(3, gamma=0.1, progress=-1.0),
    ]
    weights = UPD.positive_mass_weights(positives, "progress_weighted")
    assert weights.sum() == pytest.approx(1.0, abs=1.0e-12)
    assert weights[0] > weights[1] > weights[2]
    # Nonpositive progress rows sit exactly at the (renormalized) floor.
    floor = UPD.PROGRESS_WEIGHT_FLOOR_FRACTION / len(positives)
    assert weights[2] == pytest.approx(weights[3])
    assert weights[2] >= floor / (1.0 + 2 * floor)
    assert weights.min() > 0.0
    # Gamma mass shares stay pooled: 6-vs-2 rows keep 0.75/0.25 shares up to
    # the floor renormalization.
    mixed = _unbalanced_positives()
    mixed_weights = UPD.positive_mass_weights(mixed, "progress_weighted")
    gammas = np.asarray([row["gamma"] for row in mixed])
    assert mixed_weights[gammas == 0.1].sum() == pytest.approx(0.75, abs=0.05)
    assert mixed_weights[gammas == 1.0].sum() == pytest.approx(0.25, abs=0.05)
    # The mode fails closed when the audited progress field is missing.
    broken = [_row(0, gamma=0.1)]
    del broken[0]["prediction_audit"]["H10_goal_progress"]
    with pytest.raises(KeyError):
        UPD.positive_mass_weights(broken, "progress_weighted")
    with pytest.raises(ValueError, match="undeclared positive mass"):
        UPD.positive_mass_weights(positives, "bogus")


def test_pooled_mean_default_and_explicit_are_bitwise_identical():
    positives = _unbalanced_positives()
    default = _deterministic_adapter(seed=1)
    UPD.expansion_update(
        default, list(positives), [],
        UPD.UpdateConfig(learning_rate=1.0e-5, seed=2), round_index=1,
    )
    explicit = _deterministic_adapter(seed=1)
    UPD.expansion_update(
        explicit, list(positives), [],
        UPD.UpdateConfig(positive_mass="pooled_mean", learning_rate=1.0e-5,
                         seed=2),
        round_index=1,
    )
    for (name_a, a), (name_b, b) in zip(
        default.named_parameters(), explicit.named_parameters(),
    ):
        assert name_a == name_b
        assert torch.equal(a, b)


def test_update_config_rejects_an_undeclared_mass_mode():
    with pytest.raises(ValueError, match="positive_mass"):
        UPD.UpdateConfig(positive_mass="per_batch_softmax").validate()


# ---- resume identity ------------------------------------------------------
# The end-to-end harness mirrors test_sfm_hp100_expansion_replay.py.

def _patched_healthy_environment(tmp_path, monkeypatch):
    def _fake_load(path, device="cpu"):
        torch.manual_seed(11)
        return GPS.build_sfm_hp100_policy(), {"scientific_status": "test"}

    checkpoint = tmp_path / "r0.pt"
    torch.save({"state_dict": _adapter(11).policy.state_dict()}, checkpoint)
    import sfm_hp100_predictive_execution as PRED
    sha = PRED.sha256_file(checkpoint)

    monkeypatch.setattr(ROUND.GPS, "load_sfm_hp100_policy", _fake_load)
    monkeypatch.setattr(
        ROUND.BASE, "_gpu_contract", lambda device, gpu: {"device": device},
    )
    monkeypatch.setattr(
        ROUND.BASE, "calibration_features",
        lambda *a, **k: (torch.randn(50, 8), {"count": 50}),
    )
    monkeypatch.setattr(ROUND, "mean_pairwise_lengthscale", lambda f: 1.0)
    monkeypatch.setattr(
        ROUND.HYBRID, "_calibration_support_by_gamma", lambda *a, **k: {},
    )

    class _FakeTask:
        def __init__(self, **kwargs):
            pass

        def attach_context_encoder(self, policy):
            return self

    monkeypatch.setattr(ROUND.PORT, "SFMHP100ExpansionTask", _FakeTask)

    @contextlib.contextmanager
    def _fake_verifier(task, workers):
        yield object()

    monkeypatch.setattr(
        ROUND.HYBRID, "_OrderedSidecarVerifier", _fake_verifier,
    )
    summaries = [{
        "summary": {
            "contexts": 40,
            "retried_context_fraction": 0.1,
            "outcome_counts": {"success": 3, "nvp": 1},
            "pooled": {"attempts": 60},
            "per_gamma": {},
        },
    }]

    def _gather(*args, **kwargs):
        rows = [
            _row(seed, gamma=gamma, progress=1.0 + seed)
            for gamma in (0.1, 1.0) for seed in range(10)
        ]
        rows.append(_row(
            99, gamma=0.1, role="negative",
            negative_reason="all_negative_nvp",
        ))
        return {
            "rows": rows,
            "sample_counts": ROUND.ARCH._sample_counts(rows),
            "block_summaries": summaries,
            "trace_shards": [],
        }

    monkeypatch.setattr(ROUND.ARCH, "gather_round", _gather)
    monkeypatch.setattr(ROUND.ARCH, "assert_bank_disjoint", lambda rows: None)
    return checkpoint, sha


def _run_args(tmp_path, checkpoint, sha, output, *extra):
    return ROUND.parser().parse_args([
        "--checkpoint", str(checkpoint),
        "--expected-checkpoint-sha256", sha,
        "--pretrain-dataset-root", str(tmp_path / "unused"),
        "--expected-pretrain-dataset-manifest-sha256", "0" * 64,
        "--output", str(tmp_path / output),
        "--recipe-id", "E1",
        "--device", "cpu", "--gammas", "0.1,1.0",
        "--lineages-per-gamma", "1", "--seed", "3",
        *extra,
    ])


def test_resume_refuses_a_positive_mass_mismatch(tmp_path, monkeypatch):
    checkpoint, sha = _patched_healthy_environment(tmp_path, monkeypatch)
    final = ROUND.run_recipe(_run_args(
        tmp_path, checkpoint, sha, "arm", "--rounds", "1",
    ))
    assert final["accepted"]
    resume = tmp_path / "arm" / "resume_r1.pt"
    with pytest.raises(RuntimeError, match="identical declared update recipe"):
        ROUND.run_recipe(_run_args(
            tmp_path, checkpoint, sha, "arm",
            "--rounds", "2", "--resume-from", str(resume),
            "--positive-mass", "per_gamma_balanced",
        ))
    # A resume state written before the mass modes existed implies
    # pooled_mean and resumes under the default.
    payload = torch.load(resume, map_location="cpu", weights_only=False)
    payload["recipe_provenance"]["update_config"].pop("positive_mass")
    torch.save(payload, resume)
    resumed = ROUND.run_recipe(_run_args(
        tmp_path, checkpoint, sha, "arm",
        "--rounds", "2", "--resume-from", str(resume),
    ))
    assert resumed["rounds_completed"] == 2
