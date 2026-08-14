import hashlib
import json
from pathlib import Path
import subprocess

import torch


ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_promoted_checkpoints_are_exact():
    assert sha256(ROOT / "checkpoints/hp100_pretrained_r0_258999ae.pt") == (
        "258999ae8ccee8aec5aab92a6f751221d3c15583ac26e0a7ec8311f13316ec44"
    )
    assert sha256(ROOT / "checkpoints/hp10_pretrained_r0.pt") == (
        "1b5179c935d3eeff8824967d707d64cc9bab273949ee1f0e4f190172bab1b215"
    )
    assert sha256(ROOT / "checkpoints/b1_legacy_selected_A_r10.pt") == (
        "bf6f521dd2dd6de4cffcce672a8ce4adbf00bb14e71dd9fd27704d205f65744c"
    )
    assert sha256(ROOT / "checkpoints/b1_alpha001_inner16_selected_r3.pt") == (
        "0a152a2926eaf94bf141d37a6748d0b6a83309f8b9a3a25134ba393f72241938"
    )


def test_hp100_checkpoint_contract_is_canonical():
    checkpoint = torch.load(
        ROOT / "checkpoints/hp100_pretrained_r0_258999ae.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = checkpoint["config"]
    assert checkpoint["scientific_status"] == "canonical_ID_promoted"
    assert checkpoint["pretrained_from_scratch"] is True
    assert checkpoint["partial_transplant"] is False
    assert config["arch"] == "v3-sfm-hp100-residual"
    assert tuple(config["grid_shape"]) == (10, 32, 100)
    assert config["polytope_n_base"] == 16
    assert config["predict_gain"] == 0.0
    assert config["radial_pool"] is None
    assert config["u_max"] == config["v_max"] == 2.0


def test_hp100_dataset_and_public_raw_baselines():
    manifest_path = ROOT / "provenance/hp100_pretrain_20260802/dataset_manifest.json"
    assert sha256(manifest_path) == (
        "44f2bfa8afbb2318376ae9e188b1b622f102253a4a91f5c8ca0f9634d5041c94"
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["status"] == "HP100_ID_DATASET_COMPLETE"
    assert manifest["episode_allocation"]["successful_trajectories_per_gamma"] == 500
    assert manifest["feature"]["contract"]["tensor_shape"] == [10, 32, 100]
    assert manifest["feature"]["nominal_polytope_n_base"] == 16
    assert manifest["feature"]["predict_gain"] == 0.0

    expected = {
        "id_m50.json": ("matched_id", 0.9571428571428572, 0.04285714285714286),
        "ood_m50.json": (
            "double_density_velocity_ood",
            0.56,
            0.43714285714285717,
        ),
    }
    for name, (profile, sr, cr) in expected.items():
        result = json.loads((
            ROOT / "provenance/hp100_pretrain_20260802" / name
        ).read_text())
        assert result["scene"]["scene_profile"] == profile
        assert result["M_per_gamma"] == 50
        assert result["temperature"] == 1.0
        assert result["NFE"] == 8
        assert result["summary"]["pooled"]["SR"] == sr
        assert result["summary"]["pooled"]["CR"] == cr


def test_hp100_paper_assets_are_exact():
    assert sha256(
        ROOT / "assets/hp100_20260802/branch_viz/hp100_id_ood_branch.mp4"
    ) == "dd1269a392b3dd14161d2392ad4ee69d278c3bd76253639dac892206370c8547"
    assert sha256(
        ROOT
        / "assets/hp100_20260802/expert_mechanism/"
        "hp100_safemppi_mechanism_g0p2_ep109.mp4"
    ) == "416b77aeb60b36a06e365dcbb06a343b038bea3a4e671d4c4df00b0f74f69876"


def test_completed_banks_are_matched_and_severe_ood():
    matched = json.loads((ROOT / "provenance/pre_expansion/matched_id_metrics.json").read_text())
    shifted = json.loads((ROOT / "provenance/pre_expansion/double_shift_ood_metrics.json").read_text())
    assert matched["environment"]["scene_profile"] == "matched_id"
    assert matched["environment"]["n_ped"] == 20
    assert shifted["environment"]["scene_profile"] == "double_density_velocity_ood"
    assert shifted["environment"]["n_ped"] == 40
    assert shifted["environment"]["ped_speed_range"] == [1.0, 2.0]
    assert matched["M_per_gamma"] == shifted["M_per_gamma"] == 100


def test_alpha_epoch_recipe_and_completion_are_bound():
    recipe = json.loads((ROOT / "configs/e5ab47b_alpha_epoch_sweep_recipe.json").read_text())
    complete = json.loads((
        ROOT / "provenance/e5ab47b_alpha_epoch_sweep/SWEEP_COMPLETE.json"
    ).read_text())
    assert recipe["status"] == "DECLARED_RECIPE"
    assert complete["status"] == "SFM_B1_ALPHA_INNER_EPOCH_SWEEP_COMPLETE"
    assert complete["source"]["commit"] == recipe["source_commit"]
    assert complete["scene_profile"] == recipe["scene_profile"]
    assert complete["checkpoint_sha256"] == sha256(ROOT / "checkpoints/hp10_pretrained_r0.pt")
    assert complete["winner"]["arm"] == "margin_alpha0p001_inner016"
    assert complete["winner"]["round"] == 3
    assert complete["winner"]["checkpoint_sha256"] == sha256(
        ROOT / "checkpoints/b1_alpha001_inner16_selected_r3.pt"
    )
    gate = json.loads((ROOT / "provenance/runtime_gate/RUNTIME_FORECAST.json").read_text())
    assert gate["status"] == "RUNTIME_GATE_PASS"
    assert gate["source_commit"] == recipe["source_commit"]


def test_final_confirmation_is_m100_and_separates_sampling_temperatures():
    base = ROOT / "provenance/e5ab47b_alpha_epoch_sweep/confirmation"
    canonical = json.loads((base / "canonical_temp1/COMPLETE.json").read_text())
    locked = json.loads((base / "locked_selected/COMPLETE.json").read_text())
    assert canonical["bank"] == locked["bank"]
    assert canonical["bank"]["M"] == 100
    assert canonical["bank"]["scene_profile"] == "double_density_velocity_ood"
    assert set(canonical["temperature_by_gamma"].values()) == {1.0}
    assert set(locked["temperature_by_gamma"].values()) != {1.0}
    assert canonical["checkpoint_sha256"] == locked["checkpoint_sha256"]
    assert canonical["summary"]["pooled"]["CR"] == 0.32571428571428573
    assert canonical["summary"]["pooled"]["V_safe"] == 0.0


def test_compact_round_records_reproduce_completed_lineage_totals():
    root = ROOT / "provenance/e5ab47b_alpha_epoch_sweep/round_records"
    manifest = json.loads((root / "COMPACT_RECORDS_MANIFEST.json").read_text())
    assert manifest["status"] == "COMPACT_ROUND_RECORDS_COMPLETE"
    assert len(manifest["files"]) == 9
    nvp = success = positives = 0
    for entry in manifest["files"]:
        path = root / entry["compact_file"]
        assert sha256(path) == entry["compact_sha256"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert [row["round"] for row in rows] == list(range(1, 21))
        assert {row["source_metrics_sha256"] for row in rows} == {
            entry["source_metrics_sha256"]
        }
        for row in rows:
            counts = row["gather"]["outcome_summary"]["status_counts"]
            nvp += int(counts.get("nvp", 0))
            success += int(counts.get("success", 0))
            positives += int(row["shard"]["Dplus"])
            assert row["sanity"]["temperature_by_gamma"] == {
                str(gamma): 1.0 for gamma in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
            }
    assert (nvp, success) == (10_078, 2)
    assert 9 * 39_000 < positives < 9 * 43_000


def test_manifest_matches_tracked_package_surface_when_git_is_available():
    if not (ROOT / ".git").exists():
        return
    output = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, stdout=subprocess.PIPE
    ).stdout
    tracked = {raw.decode() for raw in output.split(b"\0") if raw}
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    declared = {row["path"] for row in manifest["files"]}
    assert declared == tracked - {"SOURCE_MANIFEST.json"}


def test_manifest_covers_documentation_and_source():
    manifest = json.loads((ROOT / "SOURCE_MANIFEST.json").read_text())
    paths = {row["path"] for row in manifest["files"]}
    assert "README.md" in paths
    assert "CLAUDE_HP100_EXPANSION_HANDOFF.md" in paths
    assert "checkpoints/hp100_pretrained_r0_258999ae.pt" in paths
    assert "source_snapshot/overnight_run_07_12_sfm/grid_policy_sfm_hp100.py" in paths
    assert "source_snapshot/overnight_run_07_12_sfm/stage2_hp100_data.py" in paths
    assert "source_snapshot/overnight_run_07_12_sfm/stage3_hp100_pretrain.py" in paths
    assert "assets/hp100_20260802/branch_viz/hp100_id_ood_branch.mp4" in paths
    assert "source_snapshot/overnight_run_07_12_sfm/sfm_b1_expand.py" in paths
    assert "assets/results/pre_expansion/double_shift_ood_metrics.png" in paths
    assert "assets/results/e5ab47b_alpha_epoch_sweep/arm_comparison.png" in paths
