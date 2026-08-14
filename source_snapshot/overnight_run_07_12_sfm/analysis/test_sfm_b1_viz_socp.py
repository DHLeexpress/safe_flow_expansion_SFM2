import numpy as np
from scipy.optimize import minimize

import sfm_b1_viz_socp as V
import sfm_metrics2 as M


def test_visualization_and_runtime_import_the_same_verifier_functions():
    assert V.verify_query is M.verify_query
    assert V.verify_in_worker is M.verify_in_worker
    assert V.certify_moving_window is M.certify_moving_window


def test_free_space_exact_socp_uses_sixteen_artificial_faces_without_grid():
    segment = np.zeros((11, 2), np.float64)
    pedestrians = np.zeros((11, 0, 2), np.float64)
    ok, faces, diagnostics = V.certify_moving_window(segment, pedestrians, .5)
    artificial = [face for face in faces if face.kind == "artificial"]
    assert ok
    assert len(artificial) == 16
    assert all(face.feasible for face in artificial)
    assert diagnostics["K_artificial"] == 16
    assert diagnostics["angular_grid"] is False


def test_static_face_matches_independent_conic_reference_and_is_tangent():
    robot = np.stack([np.linspace(0.0, .55, 11),
                      .04 * np.sin(np.linspace(0.0, np.pi, 11))], axis=1)
    obstacle = np.array([1.2, .6])
    beta = 1.0 - .5 ** np.arange(11)
    radius, m_min = .2, 1.0e-4
    face = V.solve_static_face(robot, obstacle, radius, beta, "ped")
    assert face.feasible
    assert np.isclose(np.linalg.norm(face.a), 1.0, atol=1.0e-9)
    assert np.all(robot[1:] @ face.a <= beta[1:] * face.m + 2.0e-8)
    assert np.isclose(
        obstacle @ face.a - face.m,
        radius * np.linalg.norm(face.a), atol=2.0e-9,
    )

    # Independent numerical solve of the original three-variable SOCP block.
    # This does not call the angular interval implementation.
    def constraints(value):
        normal, margin = value[:2], float(value[2])
        norm = float(np.linalg.norm(normal))
        return np.concatenate((
            beta[1:] * margin - robot[1:] @ normal,
            [obstacle @ normal - margin - radius * norm,
             1.0 - norm, margin - m_min],
        ))

    reference = minimize(
        lambda value: -float(value[2]),
        np.r_[face.a, face.m], method="SLSQP",
        constraints={"type": "ineq", "fun": constraints},
        options={"ftol": 1.0e-12, "maxiter": 2000},
    )
    assert reference.success, reference.message
    assert constraints(reference.x).min() >= -2.0e-7
    assert np.isclose(face.m, reference.x[2], atol=2.0e-7)


def test_nominal_radial_face_is_a_feasible_socp_point_when_nominal_gate_holds():
    obstacle = np.array([1.6, .5])
    radius = .2
    normal = obstacle / np.linalg.norm(obstacle)
    nominal_margin = float(np.linalg.norm(obstacle) - radius)
    beta = 1.0 - .5 ** np.arange(11)
    robot = beta[:, None] * (.45 * nominal_margin) * normal[None]
    assert np.all(robot[1:] @ normal <= beta[1:] * nominal_margin + 1.0e-12)
    assert np.isclose(
        obstacle @ normal - nominal_margin,
        radius * np.linalg.norm(normal), atol=1.0e-12,
    )
    face = V.solve_static_face(robot, obstacle, radius, beta, "ped")
    assert face.feasible
    assert face.m >= nominal_margin - 1.0e-9


def test_green_geometry_is_fixed_radius_and_uses_current_sensed_obstacles_only():
    segment = np.zeros((11, 2), np.float64)
    pedestrians = np.zeros((11, 2, 2), np.float64)
    pedestrians[:, 0] = np.stack((
        np.linspace(2.3, 1.0, 11), np.zeros(11),
    ), axis=1)  # predicted to enter, but outside at the current time
    pedestrians[:, 1] = np.stack((
        np.linspace(1.5, 3.0, 11), np.full(11, .4),
    ), axis=1)  # sensed now, even though it later leaves
    ok, faces, diagnostics = V.certify_moving_window(segment, pedestrians, .5)
    assert ok
    real_labels = [face.label for face in faces if face.kind == "real"]
    assert real_labels == ["ped1"]
    assert diagnostics["R_eff"] == M.SS.R_SENSE
    assert diagnostics["sensing_radius"] == M.SS.R_SENSE
    assert diagnostics["sensed_obstacles"] == "current-time centers only"

    long_segment = np.stack((np.linspace(0.0, 3.0, 11), np.zeros(11)), axis=1)
    _, _, long_diagnostics = V.certify_moving_window(
        long_segment, np.zeros((11, 0, 2)), .5,
    )
    assert long_diagnostics["R_eff"] == M.SS.R_SENSE


def test_exact_query_rejects_a_direct_collision():
    state = np.zeros(4, np.float32)
    controls = np.zeros((10, 2), np.float32)
    result = V.verify_query(
        state, controls, np.array([[0.1, 0.0]], np.float32),
        np.zeros((1, 2), np.float32), .5,
    )
    assert result["resolved"]
    assert result["y"] == 0
    assert not result["collision_free"]


def test_faithful_visualization_rejects_non_sixteen_anchor_contract():
    segment = np.zeros((11, 2), np.float64)
    pedestrians = np.zeros((11, 0, 2), np.float64)
    try:
        V.certify_moving_window(segment, pedestrians, .5, K=12)
    except ValueError as error:
        assert "K=16" in str(error)
    else:
        raise AssertionError("K=12 must not be accepted by the faithful renderer")
