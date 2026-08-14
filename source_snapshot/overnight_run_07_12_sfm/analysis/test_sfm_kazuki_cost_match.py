import numpy as np
import torch

import sfm_b1_cost as BC
import sfm_kazuki as K
import sfm_scene as SS


def test_default_kazuki_refinement_cost_is_exact_b1_scorer():
    config = K.KazukiConfig().validate()
    assert config.refinement_cost == "b1_safemppi"
    manifest = BC.scorer_manifest()
    assert manifest["semantics"] == "frozen SafeMPPI raw proposal cost terms"

    generator = torch.Generator().manual_seed(7)
    state = torch.tensor([0.2, -0.1, 0.3, -0.2])
    controls = torch.randn(5, 10, 2, generator=generator).clamp(-2, 2)
    goal = torch.tensor(SS.GOAL)
    ped_xy = np.array([[1.0, 0.2], [2.0, 1.5]], np.float32)
    ped_vel = np.array([[0.1, -0.2], [-0.3, 0.1]], np.float32)
    ped_pred = K.predict_pedestrians_t(ped_xy, ped_vel, 10, SS.DT, "cpu", controls.dtype)
    observed = K.refinement_cost_batch(
        state, controls, goal, ped_xy, ped_vel, ped_pred, SS.R_PED + .05, config
    )
    expected = BC.safemppi_proposal_cost(state, controls, goal, ped_xy, ped_vel)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)


def test_flow_refinement_uses_declared_cost_at_all_three_stages(monkeypatch):
    calls = []

    def fake_cost(state, controls, goal, ped_xy, ped_vel, ped_pred, r_col, cfg, prev_U=None):
        calls.append(tuple(controls.shape))
        return controls.square().sum(dim=(1, 2))

    monkeypatch.setattr(K, "refinement_cost_batch", fake_cost)

    class Policy:
        u_max = SS.U_MAX

    config = K.KazukiConfig(n_sample=4, n_elite=2, n_copy=3).validate()
    state = torch.zeros(4)
    controls = torch.linspace(-.4, .4, 4 * 10 * 2).reshape(4, 10, 2)
    ped_xy = np.array([[2.0, 2.0]], np.float32)
    ped_vel = np.zeros((1, 2), np.float32)
    ped_pred = K.predict_pedestrians_t(ped_xy, ped_vel, 10, SS.DT, "cpu", controls.dtype)
    selected, diagnostics = K.flow_mppi_refine(
        Policy(), state, torch.tensor(SS.GOAL), ped_xy, ped_vel, ped_pred,
        SS.R_PED + .05, controls, None, config, collect_diagnostics=False,
    )
    assert selected.shape == (10, 2) and diagnostics is None
    assert calls == [(4, 10, 2), (6, 10, 2), (2, 10, 2)]


def test_legacy_cost_is_explicit_not_silent():
    assert K.KazukiConfig(refinement_cost="legacy_kazuki").validate().refinement_cost == "legacy_kazuki"
