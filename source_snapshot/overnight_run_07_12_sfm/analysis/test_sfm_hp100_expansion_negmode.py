"""Declared negative-term modes: bounded hinge vs the literal objective."""
import contextlib

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


def _row(seed, *, gamma=0.1, role="positive", negative_reason=None):
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
        "prediction_audit": {"H10_goal_progress": 1.0},
    }


def _positives(count=6):
    return [_row(seed) for seed in range(count)]


def _negatives():
    return [
        _row(100, role="negative", negative_reason="all_negative_nvp"),
        _row(101, role="negative", negative_reason="all_negative_nvp"),
    ]


def _dminus_loss(adapter, negatives) -> float:
    import sfm_hp100_exhaustive_hybrid as HYBRID
    with torch.no_grad():
        return float(adapter.cfm_loss(
            *HYBRID._stack_rows(negatives, torch.device("cpu")),
            reduction="none",
        ).mean())


def test_negative_mode_config_validation():
    with pytest.raises(ValueError, match="negative_mode"):
        UPD.UpdateConfig(negative_mode="soft").validate()
    with pytest.raises(ValueError, match="negative_margin"):
        UPD.UpdateConfig(negative_mode="hinge", negative_margin=0.0).validate()
    UPD.UpdateConfig(negative_mode="hinge", negative_margin=2.0).validate()


def test_hinge_with_satisfied_margin_matches_alpha_zero_bitwise():
    # Margin far below every D- CFM loss: relu(margin - L) == 0 everywhere,
    # so the hinge contributes exactly zero gradient and the update must be
    # bitwise identical to the alpha=0 control.
    hinged = _deterministic_adapter(seed=1)
    metrics = UPD.expansion_update(
        hinged, _positives(), _negatives(),
        UPD.UpdateConfig(alpha=0.5, negative_mode="hinge",
                         negative_margin=1.0e-6, learning_rate=1.0e-4, seed=2),
        round_index=1,
    )
    control = _deterministic_adapter(seed=1)
    UPD.expansion_update(
        control, _positives(), _negatives(),
        UPD.UpdateConfig(alpha=0.0, negative_mode="hinge",
                         negative_margin=1.0e-6, learning_rate=1.0e-4, seed=2),
        round_index=1,
    )
    for (name_a, a), (name_b, b) in zip(
        hinged.named_parameters(), control.named_parameters(),
    ):
        assert name_a == name_b
        assert torch.equal(a, b)
    assert metrics["negative_mode"] == "hinge"
    assert metrics["negative_hinge_mean"] == pytest.approx(0.0, abs=1.0e-12)
    assert metrics["fraction_of_dminus_beyond_margin"] == [1.0]
    assert metrics["objective_mean"] == pytest.approx(
        metrics["positive_loss_mean"], abs=1.0e-12,
    )
    # The raw D- audit statistic is still reported.
    assert metrics["negative_loss_mean"] > 0.0


def test_hinge_below_margin_pushes_dminus_loss_upward():
    adapter = _deterministic_adapter(seed=1)
    negatives = _negatives()
    margin = 10.0
    before = _dminus_loss(adapter, negatives)
    assert before < margin
    metrics = UPD.expansion_update(
        adapter, _positives(), list(negatives),
        UPD.UpdateConfig(alpha=0.5, negative_mode="hinge",
                         negative_margin=margin, learning_rate=1.0e-4, seed=2),
        round_index=1,
    )
    after = _dminus_loss(adapter, negatives)
    assert metrics["accepted"]
    assert after > before
    # With every D- row below the margin, relu(margin - L) == margin - L, so
    # the hinge mean is exactly margin minus the raw D- mean, and the realized
    # objective is positive + alpha * hinge.
    assert metrics["fraction_of_dminus_beyond_margin"] == [0.0]
    # relu(margin - L) == margin - L for every row here; the fp32 graph
    # reduction differs from the float64 identity at ~1e-6 scale.
    assert metrics["negative_hinge_mean"] == pytest.approx(
        margin - metrics["negative_loss_mean"], abs=1.0e-5,
    )
    assert metrics["objective_mean"] == pytest.approx(
        metrics["positive_loss_mean"]
        + 0.5 * metrics["negative_hinge_mean"], rel=1.0e-9,
    )
    assert metrics["negative_grad_norm"] > 0.0
    assert metrics["grad_cosine"] is not None
    assert metrics["negative_margin"] == pytest.approx(margin)


def test_literal_ignores_margin_and_stays_the_default():
    default_run = _deterministic_adapter(seed=1)
    metrics = UPD.expansion_update(
        default_run, _positives(), _negatives(),
        UPD.UpdateConfig(alpha=0.5, learning_rate=1.0e-4, seed=2),
        round_index=1,
    )
    explicit = _deterministic_adapter(seed=1)
    UPD.expansion_update(
        explicit, _positives(), _negatives(),
        UPD.UpdateConfig(alpha=0.5, negative_mode="literal",
                         negative_margin=77.0, learning_rate=1.0e-4, seed=2),
        round_index=1,
    )
    # The margin is inert under literal and the default mode IS literal.
    for (name_a, a), (name_b, b) in zip(
        default_run.named_parameters(), explicit.named_parameters(),
    ):
        assert name_a == name_b
        assert torch.equal(a, b)
    assert metrics["negative_mode"] == "literal"
    assert metrics["negative_hinge_mean"] is None
    assert metrics["fraction_of_dminus_beyond_margin"] == [None]
    assert metrics["objective_mean"] == pytest.approx(
        metrics["positive_loss_mean"]
        - 0.5 * metrics["negative_loss_mean"], rel=1.0e-9,
    )


# ---- resume identity ------------------------------------------------------
# The end-to-end harness mirrors test_sfm_hp100_expansion_mass.py.

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
            _row(seed, gamma=gamma)
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


def test_resume_refuses_a_negative_mode_mismatch(tmp_path, monkeypatch):
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
            "--negative-mode", "hinge",
        ))
    with pytest.raises(RuntimeError, match="identical declared update recipe"):
        ROUND.run_recipe(_run_args(
            tmp_path, checkpoint, sha, "arm",
            "--rounds", "2", "--resume-from", str(resume),
            "--negative-margin", "3.0",
        ))
    # A resume state written before the negative modes existed implies the
    # literal objective and resumes under the defaults.
    payload = torch.load(resume, map_location="cpu", weights_only=False)
    payload["recipe_provenance"]["update_config"].pop("negative_mode")
    payload["recipe_provenance"]["update_config"].pop("negative_margin")
    torch.save(payload, resume)
    resumed = ROUND.run_recipe(_run_args(
        tmp_path, checkpoint, sha, "arm",
        "--rounds", "2", "--resume-from", str(resume),
    ))
    assert resumed["rounds_completed"] == 2
