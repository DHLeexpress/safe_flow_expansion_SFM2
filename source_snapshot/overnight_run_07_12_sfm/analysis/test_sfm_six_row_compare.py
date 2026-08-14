import numpy as np
import pytest
import torch

import grid_policy_sfm as GPS
import sfm_six_row_compare as C


def _expert_run():
    traces = []
    for step in range(3):
        traces.append(dict(
            step=step,
            state=np.zeros(4, np.float32),
            controls=np.zeros((10, 2), np.float32),
            ped_xy=np.array([[4.0, 4.0]], np.float32),
            ped_vel=np.zeros((1, 2), np.float32),
        ))
    return dict(
        trace=traces,
        states=np.zeros((4, 4), np.float32),
        controls=np.zeros((3, 2), np.float32),
        peds=np.zeros((3, 1, 2), np.float32),
        ped_vels=np.zeros((3, 1, 2), np.float32),
        success=True, reached=True, collision=False,
    )


def test_declared_episode_banks_keep_expansion_and_visual_eval_disjoint():
    assert C.SP.EXPANSION_EP0 == 20_000
    assert C.SP.DEPLOY_ID_EP0 == 150_000
    assert C.SP.DEPLOY_DOUBLE_SHIFT_EP0 == 250_000
    assert C.SP.DEPLOY_DOUBLE_SHIFT_EP0 > (
        C.SP.EXPANSION_EP0 + C.SP.ROUNDS * C.SP.SCENARIOS_PER_ROUND
    )


def test_gamma_columns_are_a_declared_distinct_subset():
    assert C.validate_gammas((.1, .5, 1.0)) == (.1, .5, 1.0)
    with pytest.raises(ValueError):
        C.validate_gammas((.1, .1))
    with pytest.raises(ValueError):
        C.validate_gammas((.15,))


def test_expert_gate_displays_but_does_not_execute_first_negative(monkeypatch):
    labels = iter((1, 1, 0))

    def verify(*_args, **_kwargs):
        value = next(labels)
        return dict(resolved=True, y=value, full_h=True)

    monkeypatch.setattr(C.SM, "verify_query", verify)
    gated = C._gate_expert_run(_expert_run(), .5)
    assert gated["nvp"] is True
    assert gated["nvp_step"] == 2
    assert len(gated["trace"]) == 3
    assert gated["trace"][-1]["full_h_positive"] is False
    assert gated["steps"] == 2
    assert len(gated["controls"]) == 2
    assert len(gated["states"]) == 3


def test_hp10_policy_exposes_proposal_specific_noised_representation():
    policy = GPS.build_sfm_policy(device="cpu").eval()
    controls = torch.zeros(2, 10, 2)
    context = torch.zeros(2, policy.ctx_dim)
    x0 = torch.stack((torch.zeros(policy.d), torch.ones(policy.d)))
    actual = policy.phi_s_from_x0(controls, context, x0, s=.9)
    expected = policy.features(
        .1 * x0,
        torch.full((2,), .9),
        context,
    )
    torch.testing.assert_close(actual, expected)


def test_render_frame_index_is_not_a_simulator_step():
    report = dict(frames=[0, 2, 4, 5])
    assert report["frames"][2] == 4
    with pytest.raises(IndexError):
        C.draw_frame({"status": C.STATUS}, 4, report["frames"])
