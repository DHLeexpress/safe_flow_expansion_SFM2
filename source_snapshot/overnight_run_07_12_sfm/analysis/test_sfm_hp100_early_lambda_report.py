import json

import pytest

import sfm_hp100_early_lambda_report as R
import sfm_scene as SS


def _summary(s30, net, h10, step, percentile, *, rmst30=25.0, rmst40=32.0):
    return {
        "S30": float(s30),
        "early_gather_RMST_at_30": float(rmst30),
        "early_gather_RMST_to_max_steps": float(rmst40),
        "lineage_macro_net_goal_progress_at_30": float(net),
        "goal_progress": {
            "lineage_macro_chosen_H10_mean_at_30": float(h10),
            "lineage_macro_chosen_one_step_mean_at_30": float(step),
            "chosen_H10_progress_percentile_mean_at_30": float(percentile),
            "chosen_H10_progress_percentile_ge_0p75_fraction_at_30": .8,
        },
    }


def _payload(weight, summary, *, seed_s30=None, seed_net=None):
    lineages = []
    for seed in R.EXPECTED_SEEDS:
        for gamma in (.1, .2, .3, .4, .5, .7, 1.0):
            for replica in range(16):
                lineages.append({
                    "seed": seed, "gamma": gamma, "replica": replica,
                    "scenario_id": seed * 10_000 + replica,
                })
    per_seed = {}
    for seed in R.EXPECTED_SEEDS:
        value = dict(summary)
        value["goal_progress"] = dict(summary["goal_progress"])
        value["S30"] = float(summary["S30"] if seed_s30 is None else seed_s30)
        value["lineage_macro_net_goal_progress_at_30"] = float(
            summary["lineage_macro_net_goal_progress_at_30"]
            if seed_net is None else seed_net
        )
        per_seed[str(seed)] = value
    scene_ledger = [
        {
            "core_episode": row["replica"], "scenario_id": row["scenario_id"],
            "gamma": row["gamma"], "reset_seed": row["scenario_id"],
            "fixed_scenario_audit": False,
        }
        for row in lineages
    ]
    return {
        "status": R.ACQ.STATUS, "version": R.ACQ.VERSION,
        "config": {
            "round": 1, "max_steps": 40, "K": 64, "B": 16, "H": 10,
            "scene_profile": R.EXPECTED_SCENE_PROFILE,
            "scene_profile_contract": SS.scene_profile(R.EXPECTED_SCENE_PROFILE),
            "scenario_start": R.EXPECTED_SCENARIO_START,
            "flow_base_std": 1.4, "beta": R.EXPECTED_BETA,
            "rbf": {"buffer_rows": 0, "lengthscale": .42,
                    "noise": R.EXPECTED_RBF_NOISE},
            "execution_rule": "weighted_cost",
            "execution_step_margin_weight": float(weight),
            "parallel_episodes_per_gamma": 16,
            "seeds": list(R.EXPECTED_SEEDS),
            "gammas": list(map(float, SS.GAMMAS)),
            "audit_unselected_KminusB_at_NVP": False,
        },
        "policy_unchanged": True,
        "policy_state_sha256_before": "a" * 64,
        "policy_state_sha256_after": "a" * 64,
        "checkpoint": {
            "sha256": R.EXPECTED_CHECKPOINT_SHA256,
            "pretrain_dataset_manifest_sha256":
                R.EXPECTED_DATASET_MANIFEST_SHA256,
        },
        "lineages": lineages,
        "scene_ledger": scene_ledger,
        "summary": {"pooled": summary, "per_seed": per_seed},
    }


def _write(path, value):
    path.write_text(json.dumps(value))
    return path


def test_selects_progress_preserving_survival_gain(tmp_path):
    control = _summary(.60, 1.0, 1.0, .10, .92, rmst30=24, rmst40=31)
    candidate = _summary(.75, .90, .90, .08, .80, rmst30=27, rmst40=35)
    report, rows = R.build_report([
        _write(tmp_path / "control.json", _payload(0, control)),
        _write(tmp_path / "candidate.json", _payload(25_000, candidate)),
    ])
    assert report["status"] == R.STATUS
    assert report["selected"]["lambda"] == 25_000
    assert report["selected"]["passes"] is True
    assert len(rows) == 2


def test_knee_gate_can_select_below_aspirational_absolute_S30(tmp_path):
    control = _summary(.13, 2.0, 1.4, .12, .99, rmst30=16, rmst40=17)
    knee = _summary(.30, 1.65, 1.32, .09, .79, rmst30=19, rmst40=21)
    overtilted = _summary(.34, 1.35, 1.24, .07, .59, rmst30=20, rmst40=23)
    report, _ = R.build_report([
        _write(tmp_path / "control.json", _payload(0, control)),
        _write(tmp_path / "knee.json", _payload(15_000, knee)),
        _write(tmp_path / "overtilted.json", _payload(60_000, overtilted)),
    ])
    assert report["version"] == R.VERSION
    assert report["selected"]["lambda"] == 15_000
    assert report["selected"]["passes_aspirational_S30_0p70"] is False


def test_fail_closed_when_survival_discards_too_much_progress(tmp_path):
    control = _summary(.60, 1.0, 1.0, .10, .92, rmst30=24, rmst40=31)
    collapsed = _summary(.90, .50, .60, .05, .60, rmst30=29, rmst40=39)
    report, _ = R.build_report([
        _write(tmp_path / "control.json", _payload(0, control)),
        _write(tmp_path / "collapsed.json", _payload(75_000, collapsed)),
    ])
    assert report["status"] == R.NO_GO
    assert report["selected"] is None


@pytest.mark.parametrize(
    "path,value,match",
    [
        (("version",), "old", "incomplete"),
        (("config", "scenario_start"), 350_001, "scenario_start"),
        (("config", "beta"), .01, "beta"),
        (("config", "rbf", "noise"), .1, "RBF"),
        (("config", "audit_unselected_KminusB_at_NVP"), True, "audit"),
        (("checkpoint", "sha256"), "0" * 64, "checkpoint"),
        (("checkpoint", "pretrain_dataset_manifest_sha256"), "0" * 64,
         "dataset"),
    ],
)
def test_material_cell_contracts_fail_closed(tmp_path, path, value, match):
    payload = _payload(0, _summary(.6, 1., 1., .1, .9))
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        R.build_report([_write(tmp_path / "cell.json", payload)])


def test_zero_control_progress_fails_closed_without_division(tmp_path):
    control = _summary(.60, 0.0, 1.0, .10, .92)
    candidate = _summary(.75, .90, .90, .08, .80)
    with pytest.raises(ValueError, match="baseline net_progress30"):
        R.build_report([
            _write(tmp_path / "control.json", _payload(0, control)),
            _write(tmp_path / "candidate.json", _payload(25_000, candidate)),
        ])
