from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np

import sfm_hp100_branch_viz as V
import sfm_hp100_eval as RAW
import sfm_scene as SS


def _row(gamma, episode, status="success"):
    return dict(
        gamma=float(gamma), episode=int(episode), status=str(status),
        success=status == "success", collision=status == "collision",
        timeout=status == "timeout", steps=12, validity=.625,
        min_clearance=.2, successful_clearance=(.2 if status == "success" else None),
        time_to_goal=(1.2 if status == "success" else None),
    )


def _payload(*, ep0=150_000, profile="matched_id", d=4, seed=19):
    rows = [
        _row(gamma, ep0 + rollout)
        for gamma in SS.GAMMAS for rollout in range(V.M_PER_GAMMA)
    ]
    bank = RAW.noise_bank(M=V.M_PER_GAMMA, d=d, seed=seed)
    return dict(
        status="SFM_HP100_RAW_EVAL_COMPLETE",
        version=RAW.VERSION,
        evaluator_sha256=V.sha256_file(V.inspect.getsourcefile(RAW)),
        checkpoint_sha256="a" * 64,
        dynamics=V.DYN.contract(), observation=V.HPF.contract(),
        verifier=dict(
            contract=V.VERIFY.verifier_manifest(),
            evaluator_sha256=V.sha256_file(V.inspect.getsourcefile(V.VERIFY)),
            polytope_sha256=V.sha256_file(
                V.inspect.getsourcefile(V.VERIFY.VP)
            ),
        ),
        scene=SS.scene_profile(profile), ep0=ep0,
        M_per_gamma=V.M_PER_GAMMA, temperature=1.0, NFE=RAW.NFE,
        noise_seed=seed,
        noise_bank=dict(
            seed=seed, shape=list(bank.shape), dtype=str(bank.dtype),
            sha256=V.array_sha256(bank),
        ),
        rows=rows,
    )


def test_raw_m50_contract_reconstructs_authenticated_noise_bank():
    payload = _payload()
    bank = V.validate_evaluation(
        payload, scene_profile="matched_id", ep0=150_000,
        checkpoint_sha256="a" * 64, policy_d=4,
    )
    assert bank.shape == (len(SS.GAMMAS), 50, RAW.T, 4)
    assert V.array_sha256(bank) == payload["noise_bank"]["sha256"]


def test_raw_m50_contract_rejects_missing_or_changed_noise_provenance():
    payload = _payload()
    payload.pop("noise_seed")
    payload.pop("noise_bank")
    try:
        V.validate_evaluation(
            payload, scene_profile="matched_id", ep0=150_000,
            checkpoint_sha256="a" * 64, policy_d=4,
        )
    except ValueError as error:
        assert "noise seed" in str(error)
    else:
        raise AssertionError("missing CRN provenance must fail closed")

    payload = _payload()
    payload["noise_bank"]["sha256"] = "b" * 64
    try:
        V.validate_evaluation(
            payload, scene_profile="matched_id", ep0=150_000,
            checkpoint_sha256="a" * 64, policy_d=4,
        )
    except ValueError as error:
        assert "SHA-256" in str(error)
    else:
        raise AssertionError("changed CRN bank must fail closed")


def test_case_selection_is_first_success_and_collision_with_timeout_only_fallback():
    payload = _payload()
    cell = [row for row in payload["rows"] if row["gamma"] == .5]
    for row in cell:
        row.update(status="success", success=True, collision=False, timeout=False)
    cell[1].update(status="collision", success=False, collision=True)
    cell[3].update(status="collision", success=False, collision=True)
    selected = V.select_cases(payload)
    assert selected["success"]["episode"] == 150_000
    assert selected["failure"]["episode"] == 150_001
    assert selected["failure_selection"] == "first_collision"

    for row in cell:
        row.update(status="success", success=True, collision=False, timeout=False)
    cell[4].update(status="timeout", success=False, timeout=True)
    selected = V.select_cases(payload)
    assert selected["failure"]["episode"] == 150_004
    assert selected["failure_selection"] == "timeout_fallback_no_collision_in_fixed_bank"


def test_proposal_label_comes_from_raw_evaluator_and_retains_exact_geometry(monkeypatch):
    faces = [SimpleNamespace(feasible=True)]
    monkeypatch.setattr(
        V.RAW, "verify_executed_window",
        lambda *args: dict(
            resolved=True, y=1, taskspace=True, collision_free=True,
            certificate=True, diagnostics={"canonical": True},
            window_horizon=10,
        ),
    )
    monkeypatch.setattr(
        V.RAW, "clipped_rollout_positions",
        lambda state, controls: np.zeros((11, 2), np.float32),
    )
    monkeypatch.setattr(
        V.VERIFY, "predict_pedestrians",
        lambda xy, vel, H: np.zeros((11, 0, 2), np.float32),
    )
    monkeypatch.setattr(V.VERIFY, "taskspace_ok", lambda segment: True)
    monkeypatch.setattr(
        V.VERIFY, "collision_free_time_indexed", lambda segment, peds: True,
    )
    monkeypatch.setattr(
        V.VERIFY, "certify_moving_window",
        lambda segment, peds, gamma: (
            True, faces,
            dict(
                solver="paper_static_exact_2d_angular_interval_socp",
                K_artificial=16,
            ),
        ),
    )
    result = V.verify_raw_proposal(
        np.zeros(4), np.zeros((10, 2)), np.zeros((0, 2)),
        np.zeros((0, 2)), .5,
    )
    assert result["y"] == 1 and result["full_h"] is True
    assert result["faces"] is faces
    assert result["diagnostics"]["K_artificial"] == 16


def _face(a, margin, kind, label):
    return SimpleNamespace(
        a=np.asarray(a, float), m=float(margin), kind=kind,
        label=label, feasible=True,
    )


def test_green_geometry_covers_current_sensed_disks_and_exactly_16_outer_faces():
    artificial = [
        _face(
            [np.cos(2 * np.pi * index / 16), np.sin(2 * np.pi * index / 16)],
            2.0, "artificial", f"art{index}",
        )
        for index in range(16)
    ]
    result = dict(
        resolved=True, y=1, segment=np.zeros((11, 2)),
        pedestrian_prediction=np.repeat(
            np.array([[[1.0, 0.0]]]), 11, axis=0,
        ),
        faces=[_face([1., 0.], .5, "real", "ped0"), *artificial],
        diagnostics=dict(
            solver="paper_static_exact_2d_angular_interval_socp", K_artificial=16,
            R_eff=2.0,
        ),
    )
    geometry = V.verifier_geometry(dict(
        gamma=.5, proposal_result=result,
    ))
    assert len(geometry["levels"]) == 10
    assert geometry["artificial_faces"] == 16
    assert geometry["real_face_labels"] == ["ped0"]

    result["pedestrian_prediction"] = np.repeat(
        np.array([[[1.0, 0.0]]]), 11, axis=0,
    )
    result["faces"] = artificial
    try:
        V.verifier_geometry(dict(gamma=.5, proposal_result=result))
    except RuntimeError as error:
        assert "current-sensed" in str(error)
    else:
        raise AssertionError("a sensed pedestrian omitted by GREEN must fail")

    result["pedestrian_prediction"] = np.linspace(
        [2.3, 0.0], [1.0, 0.0], 11,
    )[:, None, :]
    geometry = V.verifier_geometry(dict(gamma=.5, proposal_result=result))
    assert geometry["real_face_labels"] == []


def test_zero_step_collision_is_rendered_without_inventing_a_branch():
    figure, axis = plt.subplots()
    result = V.draw_case(axis, dict(
        traces=[], terminal_snapshot=dict(
            state=np.zeros(4), ped_xy=np.array([[0.1, 0.0]]),
            ped_vel=np.zeros((1, 2)), gamma=.5,
        ),
    ), 0)
    plt.close(figure)
    assert result is None


def test_negative_branch_marks_its_endpoint_not_an_invented_worst_step():
    segment = np.stack((np.arange(11, dtype=float), np.zeros(11)), axis=1)
    trace = dict(
        state=np.zeros(4), next_state=np.zeros(4), gamma=.5,
        ped_xy=np.zeros((0, 2)), ped_vel=np.zeros((0, 2)),
        proposal_label="full_h_negative",
        proposal_result=dict(segment=segment, diagnostics=dict(worst_t=-1)),
    )
    figure, axis = plt.subplots()
    V.draw_case(axis, dict(traces=[trace]), 0)
    markers = [line for line in axis.lines if line.get_marker() == "x"]
    assert len(markers) == 1
    np.testing.assert_array_equal(markers[0].get_xdata(), [segment[-1, 0]])
    np.testing.assert_array_equal(markers[0].get_ydata(), [segment[-1, 1]])
    plt.close(figure)
