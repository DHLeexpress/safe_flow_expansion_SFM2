from pathlib import Path

import sfm_b1_expand as X
import sfm_protocol as P


def test_frozen_arm_matrix_and_macro_round_ids():
    expected = {
        "A": ("margin", 0.0), "B": ("safemppi_cost", 0.0),
        "C": ("safemppi_cost", .001), "D": ("safemppi_cost", .01),
    }
    for name, (selector, alpha) in expected.items():
        assert X.ARMS[name] == {"selector": selector, "alpha": alpha}
        X.ArmConfig(name=name, selector=selector, alpha=alpha).validate()
    first = P.expansion_scenarios(1)
    second = P.expansion_scenarios(2)
    assert len(first) == len(set(first)) == 8
    assert set(first).isdisjoint(second)
    assert 8 * len(P.GAMMAS) == 56


def test_expansion_has_no_forbidden_legacy_or_expert_path():
    source = Path(X.__file__).read_text()
    forbidden = (
        "grid_expand_sfm", "grid_metrics_sfm", "sfm_safe_expand_eval",
        "sfm_verified_controller_eval", "sfm_kazuki", "SafeMPPIAdapter",
        "stage_cost_batch",
    )
    assert not any(name in source for name in forbidden)


def test_frozen_knobs_and_recent_window():
    config = X.ArmConfig(name="B", selector="safemppi_cost", alpha=0.0)
    assert (config.K, config.B, config.T, config.H, config.W) == (16, 4, 180, 10, 2)
    assert (config.batch, config.lr, config.ess_target) == (128, 1e-5, .5)
