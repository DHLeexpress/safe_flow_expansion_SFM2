from types import SimpleNamespace

import numpy as np
import pytest

import sfm_b1_viz as V


SQUARE_A = np.array([
    [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0],
])


def test_halfspace_polygon_recovers_bounded_square():
    polygon = V.halfspace_polygon(SQUARE_A, np.ones(4))
    assert polygon.shape == (4, 2)
    np.testing.assert_allclose(polygon.min(axis=0), [-1.0, -1.0])
    np.testing.assert_allclose(polygon.max(axis=0), [1.0, 1.0])


def test_level_polygons_use_beta_h_and_return_exactly_ten_sets():
    center = np.array([1.0, -2.0])
    polygons = V._level_polygons(SQUARE_A, np.ones(4), center, gamma=.5, H=10)
    assert [horizon for horizon, _ in polygons] == list(range(1, 11))
    first = polygons[0][1]
    tenth = polygons[-1][1]
    np.testing.assert_allclose(first.min(axis=0), center - .5)
    np.testing.assert_allclose(first.max(axis=0), center + .5)
    radius_ten = 1.0 - .5 ** 10
    np.testing.assert_allclose(tenth.min(axis=0), center - radius_ten, atol=1.0e-8)
    np.testing.assert_allclose(tenth.max(axis=0), center + radius_ten, atol=1.0e-8)


def test_verifier_levels_include_feasible_artificial_faces_for_bounded_sets():
    faces = [
        SimpleNamespace(a=normal, m=1.0, feasible=True,
                        kind="real-moving" if index == 0 else "artificial")
        for index, normal in enumerate(SQUARE_A)
    ]
    result = dict(
        resolved=True, y=1, full_h=True, terminal_step=10,
        segment=np.vstack([np.array([2.0, 3.0]), np.zeros((10, 2))]),
        faces=faces,
    )
    trace = dict(gamma=.5)
    polygons = V.verifier_level_polygons(trace, {"result": result})
    assert len(polygons) == 10
    np.testing.assert_allclose(polygons[0][1].min(axis=0), [1.5, 2.5])
    np.testing.assert_allclose(polygons[0][1].max(axis=0), [2.5, 3.5])


def test_candidate_status_rejects_legacy_terminal_prefix():
    trace = dict(query_rows=[dict(
        candidate_id=2,
        result=dict(resolved=True, y=1, full_h=False, terminal_step=3),
    )])
    with pytest.raises(ValueError, match="legacy partial B1 query"):
        V._candidate_status(trace, 2)
