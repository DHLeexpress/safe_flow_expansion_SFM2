"""Mode tagging and the declared mode_gamma_tree D+ mass mode."""
import contextlib

import numpy as np
import pytest
import torch

import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_expansion_round as ROUND
import sfm_hp100_expansion_update as UPD
import sfm_hp100_mode_tags as TAGS


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


def _row(seed, *, gamma=0.1, clear_min=1.0, step=None, attempt=0,
         lineage=None, role="positive"):
    generator = torch.Generator().manual_seed(seed)
    return {
        "role": role,
        "gamma": gamma, "replica": 0,
        "lineage": lineage or f"g{gamma:g}:rep00",
        "scenario_id": 1_000_000 + seed,
        "step": seed if step is None else step, "attempt": attempt,
        "executed": role == "positive",
        "negative_reason": None,
        "context": torch.randn(CTX + 4, generator=generator),
        "candidate": torch.randn(10, 2, generator=generator),
        "flow_base": torch.randn(10, 2, generator=generator),
        "verification": {"valid": role == "positive"},
        "prediction_audit": {
            "H10_goal_progress": 1.0,
            "mpc_horizon_clearances": [float(clear_min) + 0.05 * h
                                       for h in range(10)],
        },
    }


def _tag(row, *, interaction, changed):
    row["mode_tags"] = {"interaction": interaction, "changed": changed}
    return row


# ---- tagging ---------------------------------------------------------------

def test_interaction_and_changed_tagging(tmp_path):
    rows = [
        _row(0, gamma=0.1, clear_min=0.10, step=5, lineage="g0.1:rep00"),
        _row(1, gamma=0.1, clear_min=0.10, step=6, lineage="g0.1:rep00"),
        _row(2, gamma=0.1, clear_min=0.90, step=7, lineage="g0.1:rep00"),
        _row(3, gamma=1.0, clear_min=0.10, step=2, lineage="g1:rep00"),
    ]
    trace = tmp_path / "trace.pt"
    torch.save({"events": [
        {"lineage": "g0.1:rep00", "step": 5, "attempts": [
            {"attempt": 0, "prediction_audits": [
                {"v2_shadow": {"old_rule_local": 1, "new_rule_local": 2,
                               "changed": True, "eligible": 30}},
            ]},
        ]},
        {"lineage": "g0.1:rep00", "step": 6, "attempts": [
            {"attempt": 0, "prediction_audits": [
                {"v2_shadow": {"old_rule_local": 4, "new_rule_local": 4,
                               "changed": False, "eligible": 30}},
            ]},
        ]},
        # step 7 has no shadow record; g1 lineage never appears.
        {"lineage": "g0.1:rep00", "step": 7, "attempts": [
            {"attempt": 0, "prediction_audits": [{}]},
        ]},
    ]}, trace)
    changed = TAGS.changed_map_from_trace(trace)
    assert changed == {("g0.1:rep00", 5, 0): True, ("g0.1:rep00", 6, 0): False}
    stats = TAGS.tag_rows(rows, changed)
    assert rows[0]["mode_tags"] == {"interaction": True, "changed": True}
    assert rows[1]["mode_tags"] == {"interaction": True, "changed": False}
    assert rows[2]["mode_tags"] == {"interaction": False, "changed": None}
    assert rows[3]["mode_tags"] == {"interaction": True, "changed": None}
    assert stats["0.1"] == {"goal_seeking": 1, "safe_pass": 1,
                            "hard_avoid": 1, "unknown_changed": 0}
    assert stats["1"] == {"goal_seeking": 0, "safe_pass": 0,
                          "hard_avoid": 0, "unknown_changed": 1}
    # A list of maps merges.
    stats_again = TAGS.tag_rows(rows, [changed, {}])
    assert stats_again == stats


def test_interaction_flag_fails_closed_without_clearances():
    row = _row(0)
    del row["prediction_audit"]["mpc_horizon_clearances"]
    with pytest.raises(KeyError):
        TAGS.interaction_flag(row)
    with pytest.raises(KeyError):
        TAGS.tag_rows([row], {})


# ---- tree masses -----------------------------------------------------------

def test_tree_masses_branch_sums_and_leaf_gamma_equality():
    positives = (
        # goal seeking: 2 gammas, unbalanced row counts.
        [_tag(_row(seed, gamma=0.1), interaction=False, changed=None)
         for seed in range(3)]
        + [_tag(_row(10, gamma=1.0), interaction=False, changed=None)]
        # hard avoid: single gamma.
        + [_tag(_row(seed, gamma=0.1), interaction=True, changed=True)
           for seed in (20, 21)]
        # safe pass + unknown share the soft leaf.
        + [_tag(_row(30, gamma=0.1), interaction=True, changed=False),
           _tag(_row(31, gamma=1.0), interaction=True, changed=None)]
    )
    weights = UPD.positive_mass_weights(
        positives, "mode_gamma_tree", avoid_mass=0.6,
    )
    assert weights.sum() == pytest.approx(1.0, abs=1.0e-12)
    goal = weights[:4]
    hard = weights[4:6]
    soft = weights[6:]
    assert goal.sum() == pytest.approx(0.4, abs=1.0e-9)
    assert hard.sum() == pytest.approx(0.6 * UPD.HARD_AVOID_SHARE, abs=1.0e-9)
    assert soft.sum() == pytest.approx(0.6 / 3.0, abs=1.0e-9)
    # Per-gamma equality inside the goal leaf: gamma 0.1 (3 rows) and gamma
    # 1.0 (1 row) each carry 0.2.
    assert goal[:3].sum() == pytest.approx(0.2, abs=1.0e-9)
    assert goal[3] == pytest.approx(0.2, abs=1.0e-9)
    assert np.allclose(goal[:3], 0.2 / 3)
    # Soft leaf: two gammas, one row each -> equal weights.
    assert soft[0] == pytest.approx(soft[1], abs=1.0e-12)


def test_tree_empty_branch_reassignment():
    # No goal-seeking rows: avoidance carries all mass.
    avoidance_only = [
        _tag(_row(0, gamma=0.1), interaction=True, changed=True),
        _tag(_row(1, gamma=0.1), interaction=True, changed=False),
    ]
    weights = UPD.positive_mass_weights(
        avoidance_only, "mode_gamma_tree", avoid_mass=0.65,
    )
    assert weights.sum() == pytest.approx(1.0, abs=1.0e-12)
    assert weights[0] == pytest.approx(UPD.HARD_AVOID_SHARE, abs=1.0e-9)
    # No hard-avoid rows: the soft leaf carries the whole avoidance branch.
    no_hard = [
        _tag(_row(0, gamma=0.1), interaction=False, changed=None),
        _tag(_row(1, gamma=0.1), interaction=True, changed=False),
    ]
    weights = UPD.positive_mass_weights(
        no_hard, "mode_gamma_tree", avoid_mass=0.65,
    )
    assert weights[1] == pytest.approx(0.65, abs=1.0e-9)
    assert weights[0] == pytest.approx(0.35, abs=1.0e-9)
    # No avoidance rows at all: goal seeking anchors everything.
    goal_only = [
        _tag(_row(seed, gamma=0.1), interaction=False, changed=None)
        for seed in range(2)
    ]
    weights = UPD.positive_mass_weights(
        goal_only, "mode_gamma_tree", avoid_mass=0.65,
    )
    assert np.allclose(weights, 0.5)


def test_tree_floor_and_single_renormalization():
    # 95 goal rows against 5 avoidance rows at avoid_mass 0.9 pushes the raw
    # goal weight (0.1/95) below the floor (0.25/100); flooring then one
    # global renormalization must keep the vector a probability vector.
    positives = (
        [_tag(_row(seed, gamma=0.1), interaction=False, changed=None)
         for seed in range(95)]
        + [_tag(_row(200 + seed, gamma=0.1), interaction=True, changed=True)
           for seed in range(5)]
    )
    weights = UPD.positive_mass_weights(
        positives, "mode_gamma_tree", avoid_mass=0.9,
    )
    assert weights.sum() == pytest.approx(1.0, abs=1.0e-12)
    floor = UPD.PROGRESS_WEIGHT_FLOOR_FRACTION / len(positives)
    raw_goal = 0.1 / 95
    assert raw_goal < floor
    # Goal rows sit at the renormalized floor, uniformly.
    assert np.allclose(weights[:95], weights[0])
    assert weights[0] == pytest.approx(floor / weights_sum_before(floor),
                                       rel=1.0e-6)
    assert weights[95] > weights[0]


def weights_sum_before(floor):
    # 95 floored rows + the avoidance branch mass 0.9.
    return 95 * floor + 0.9


def test_tree_fails_closed_on_untagged_rows_and_bad_avoid_mass():
    untagged = [_row(0, gamma=0.1)]
    with pytest.raises(KeyError):
        UPD.positive_mass_weights(untagged, "mode_gamma_tree")
    tagged = [_tag(_row(0, gamma=0.1), interaction=True, changed=True)]
    with pytest.raises(ValueError, match="avoid_mass"):
        UPD.positive_mass_weights(tagged, "mode_gamma_tree", avoid_mass=1.0)
    with pytest.raises(ValueError, match="avoid_mass"):
        UPD.UpdateConfig(positive_mass="mode_gamma_tree",
                         avoid_mass=0.0).validate()


def test_update_audits_branch_masses():
    positives = (
        [_tag(_row(seed, gamma=0.1), interaction=False, changed=None)
         for seed in range(4)]
        + [_tag(_row(20 + seed, gamma=1.0), interaction=True, changed=True)
           for seed in range(3)]
        + [_tag(_row(40, gamma=1.0), interaction=True, changed=False)]
    )
    tree = UPD.expansion_update(
        _deterministic_adapter(seed=1), list(positives), [],
        UPD.UpdateConfig(positive_mass="mode_gamma_tree", avoid_mass=0.65,
                         learning_rate=1.0e-5, seed=2),
        round_index=1,
    )
    pooled = UPD.expansion_update(
        _deterministic_adapter(seed=1), list(positives), [],
        UPD.UpdateConfig(learning_rate=1.0e-5, seed=2), round_index=1,
    )
    assert tree["positive_mass"] == "mode_gamma_tree"
    assert tree["avoid_mass"] == pytest.approx(0.65)
    branch = tree["positive_mass_per_branch"]
    assert branch["rows"] == {"goal_seeking": 4, "hard_avoid": 3,
                              "safe_pass_or_unknown": 1}
    assert branch["mass"]["goal_seeking"] == pytest.approx(0.35, abs=1.0e-9)
    assert branch["mass"]["hard_avoid"] == pytest.approx(
        0.65 * UPD.HARD_AVOID_SHARE, abs=1.0e-9,
    )
    assert sum(branch["mass"].values()) == pytest.approx(1.0, abs=1.0e-9)
    assert tree["objective_mean"] != pytest.approx(
        pooled["objective_mean"], abs=1.0e-9,
    )
    assert pooled["positive_mass_per_branch"] is None
    assert pooled["avoid_mass"] is None


# ---- resume identity --------------------------------------------------------
# Harness mirrors test_sfm_hp100_expansion_mass.py.

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
        rows.append(_row(99, gamma=0.1, role="negative"))
        rows[-1]["negative_reason"] = "all_negative_nvp"
        rows[-1]["executed"] = False
        rows[-1]["verification"] = {"valid": False}
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


def test_resume_refuses_an_avoid_mass_mismatch(tmp_path, monkeypatch):
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
            "--avoid-mass", "0.5",
        ))
    # A resume state written before the mode tree existed carries the default
    # split implicitly and resumes under it.
    payload = torch.load(resume, map_location="cpu", weights_only=False)
    payload["recipe_provenance"]["update_config"].pop("avoid_mass")
    torch.save(payload, resume)
    resumed = ROUND.run_recipe(_run_args(
        tmp_path, checkpoint, sha, "arm",
        "--rounds", "2", "--resume-from", str(resume),
    ))
    assert resumed["rounds_completed"] == 2
