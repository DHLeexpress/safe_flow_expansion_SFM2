import numpy as np

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


def test_moving_face_satisfies_every_socp_constraint():
    robot = np.stack([np.linspace(0.0, .8, 11), np.zeros(11)], axis=1)
    pedestrian = np.stack([np.full(11, 1.2), np.linspace(.7, .5, 11)], axis=1)
    beta = 1.0 - .5 ** np.arange(11)
    face = V.solve_moving_face(robot, pedestrian, .2, beta, "ped")
    assert face.feasible
    assert np.isclose(np.linalg.norm(face.a), 1.0, atol=1.0e-9)
    assert np.all(robot[1:] @ face.a <= beta[1:] * face.m + 2.0e-8)
    assert np.all(.2 * np.linalg.norm(face.a) <= pedestrian @ face.a - face.m + 2.0e-8)


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
