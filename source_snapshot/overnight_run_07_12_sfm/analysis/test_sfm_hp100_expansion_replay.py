"""Declared replay window: union semantics, exact accounting, fresh-only gate."""
import contextlib
import json

import pytest
import torch

import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_expansion_archive as ARCH
import sfm_hp100_expansion_round as ROUND
import sfm_hp100_expansion_update as UPD


CTX = GPS.LOW_TOKEN + GPS.VISUAL_TOKEN
GAMMAS = (0.1, 1.0)


def _adapter(seed=0):
    torch.manual_seed(seed)
    return PORT.HP100ExpansionPolicy(GPS.build_sfm_hp100_policy()).eval()


def _row(seed, *, gamma=0.1, role="positive", negative_reason=None,
         round_index=None):
    generator = torch.Generator().manual_seed(seed)
    value = {
        "role": role,
        "gamma": gamma, "replica": 0, "lineage": f"g{gamma:g}:rep00",
        "scenario_id": 860_000 + seed, "step": seed, "attempt": 0,
        "executed": role == "positive",
        "negative_reason": negative_reason,
        "context": torch.randn(CTX + 4, generator=generator),
        "candidate": torch.randn(10, 2, generator=generator),
        "flow_base": torch.randn(10, 2, generator=generator),
        "verification": {"valid": role == "positive"},
        "prediction_audit": {},
        "positive_first16": 8, "positive_B32": 20, "base_std": 1.0,
        "attempts_used": 1, "beta": 2.0, "marginal_ESS_over_K": 0.1,
        "selected_sigma": 0.5,
    }
    if round_index is not None:
        value["round"] = int(round_index)
    return value


def _healthy_rows(round_index=None):
    rows = [
        _row(seed, gamma=gamma, round_index=round_index)
        for gamma in GAMMAS for seed in range(10)
    ]
    rows.append(_row(
        99, gamma=0.1, role="negative", negative_reason="all_negative_nvp",
        round_index=round_index,
    ))
    return rows


def _archive_file(tmp_path, name, rows):
    path = tmp_path / name
    torch.save({"status": ARCH.ARCHIVE_STATUS, "rows": rows}, path)
    return path


def test_replay_union_window_one_is_exactly_the_fresh_archive():
    fresh = _healthy_rows(round_index=2)
    # The bogus ledger path proves window=1 never touches prior archives.
    ledger = [{"round": 1, "path": "/nonexistent/archive.pt"}]
    union, marker = ROUND.replay_union(
        ledger, fresh, round_index=2, replay_window=1,
    )
    assert union == fresh
    assert [row is fresh_row for row, fresh_row in zip(union, fresh)]
    assert marker["window"] == 1
    assert marker["first_round_in_union"] == 2
    assert len(marker["sources"]) == 1
    assert marker["sources"][0]["fresh"] and marker["sources"][0]["round"] == 2
    assert marker["union_sample_counts"] == ARCH._sample_counts(fresh)


def test_replay_union_window_two_unions_prior_rounds_with_exact_accounting(
    tmp_path,
):
    round1 = _healthy_rows(round_index=1)
    round2 = _healthy_rows(round_index=2)
    ledger = [
        {"round": 1, "path": str(_archive_file(tmp_path, "r1.pt", round1))},
        {"round": 2, "path": "/never/loaded/for/the/fresh/round.pt"},
    ]
    union, marker = ROUND.replay_union(
        ledger, round2, round_index=2, replay_window=2,
    )
    assert len(union) == len(round1) + len(round2)
    assert marker["window"] == 2
    assert marker["first_round_in_union"] == 1
    assert [source["round"] for source in marker["sources"]] == [1, 2]
    assert [source["fresh"] for source in marker["sources"]] == [False, True]
    assert marker["sources"][0]["sample_counts"]["positive"] == 20
    assert marker["union_sample_counts"]["positive"] == 40
    assert marker["union_sample_counts"]["all_negative_nvp"] == 2
    # Replayed rows keep their own round provenance, so the union has no
    # false duplicates across rounds...
    positives = [row for row in union if row["role"] == "positive"]
    assert UPD._in_archive_duplicate_rows(positives) == 0
    assert len(set(UPD._row_keys(positives))) == 40
    # ...while genuinely identical rows (same key including round) are
    # counted truthfully.
    duplicated = positives + [positives[0]]
    assert UPD._in_archive_duplicate_rows(duplicated) == 1
    # A window larger than the history clips at round 1.
    union_wide, marker_wide = ROUND.replay_union(
        ledger, round2, round_index=2, replay_window=10,
    )
    assert len(union_wide) == len(union)
    assert marker_wide["first_round_in_union"] == 1
    with pytest.raises(ValueError, match="positive round count"):
        ROUND.replay_union(ledger, round2, round_index=2, replay_window=0)


def test_update_over_a_window_union_audits_exposure_exactly(tmp_path):
    round1 = _healthy_rows(round_index=1)
    round2 = _healthy_rows(round_index=2)
    ledger = [
        {"round": 1, "path": str(_archive_file(tmp_path, "r1.pt", round1))},
    ]
    union, _ = ROUND.replay_union(
        ledger, round2, round_index=2, replay_window=2,
    )
    positives = [row for row in union if row["role"] == "positive"]
    negatives = [row for row in union if row["role"] == "negative"]
    metrics = UPD.expansion_update(
        _adapter(seed=1), positives, negatives,
        UPD.UpdateConfig(replay_window=2, exposure_passes=2, seed=2),
        round_index=2,
    )
    assert metrics["config"]["replay_window"] == 2
    assert metrics["positive_count"] == 40
    assert metrics["negative_count"] == 2
    assert metrics["unique_positive_samples"] == 40
    assert metrics["positive_exposures"] == 80
    assert metrics["duplicate_exposures"] == 40
    assert metrics["in_archive_duplicate_rows"] == 0


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
        rows = _healthy_rows()
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


def test_window_two_trains_on_the_union_and_gates_on_fresh_only(
    tmp_path, monkeypatch,
):
    checkpoint, sha = _patched_healthy_environment(tmp_path, monkeypatch)
    final = ROUND.run_recipe(_run_args(
        tmp_path, checkpoint, sha, "arm",
        "--rounds", "2", "--replay-window", "2",
    ))
    assert final["accepted"] and final["rounds_completed"] == 2
    output = tmp_path / "arm"
    round1 = json.loads((output / "ROUND_1_COMPLETE.json").read_text())
    assert round1["replay"]["window"] == 2
    assert [s["fresh"] for s in round1["replay"]["sources"]] == [True]
    assert round1["update"]["positive_count"] == 20
    round2 = json.loads((output / "ROUND_2_COMPLETE.json").read_text())
    assert [s["round"] for s in round2["replay"]["sources"]] == [1, 2]
    assert round2["replay"]["union_sample_counts"]["positive"] == 40
    assert round2["update"]["positive_count"] == 40
    assert round2["update"]["negative_count"] == 2
    # The acquisition health gate saw only the fresh round-2 archive.
    health2 = json.loads((output / "ROUND_2_HEALTH.json").read_text())
    assert health2["passed"]
    assert health2["stats"]["dplus_total"] == 20
    audit = json.loads((output / "QUALIFICATION_AUDIT.json").read_text())
    assert audit["rounds"][1]["replay"]["window"] == 2


def test_window_one_cli_matches_the_default_training_path(
    tmp_path, monkeypatch,
):
    checkpoint, sha = _patched_healthy_environment(tmp_path, monkeypatch)
    ROUND.run_recipe(_run_args(
        tmp_path, checkpoint, sha, "arm_default", "--rounds", "1",
    ))
    ROUND.run_recipe(_run_args(
        tmp_path, checkpoint, sha, "arm_window1",
        "--rounds", "1", "--replay-window", "1",
    ))
    markers = [
        json.loads(
            (tmp_path / name / "ROUND_1_COMPLETE.json").read_text()
        )
        for name in ("arm_default", "arm_window1")
    ]
    assert (
        markers[0]["checkpoint"]["policy_state_sha256"]
        == markers[1]["checkpoint"]["policy_state_sha256"]
    )
    assert (
        markers[0]["update"]["positive_loss_mean"]
        == markers[1]["update"]["positive_loss_mean"]
    )
    assert markers[1]["replay"]["window"] == 1


def test_resume_refuses_a_replay_window_mismatch(tmp_path, monkeypatch):
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
            "--replay-window", "2",
        ))
    # A resume state written before the window existed implies window=1 and
    # resumes under the default.
    payload = torch.load(resume, map_location="cpu", weights_only=False)
    payload["recipe_provenance"]["update_config"].pop("replay_window")
    torch.save(payload, resume)
    resumed = ROUND.run_recipe(_run_args(
        tmp_path, checkpoint, sha, "arm",
        "--rounds", "2", "--resume-from", str(resume),
    ))
    assert resumed["rounds_completed"] == 2
