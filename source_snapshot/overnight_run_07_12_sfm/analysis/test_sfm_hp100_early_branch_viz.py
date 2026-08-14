import matplotlib.pyplot as plt
import numpy as np
import pytest

import sfm_hp100_early_branch_viz as V


def _sidecar(segment):
    faces = []
    for index in range(16):
        angle = 2 * np.pi * index / 16
        faces.append(dict(
            a=[np.cos(angle), np.sin(angle)], m=2.0,
            kind="artificial", label=f"art{index}", coefficient=1.0,
            feasible=True, interval=None,
        ))
    return dict(
        resolved=True, y=1, full_h=True,
        segment=np.asarray(segment, np.float32), faces=faces,
        pedestrian_prediction=np.repeat(
            np.array([[[2.0, 2.0]]], np.float32), 11, axis=0,
        ),
        diagnostics=dict(
            solver="paper_static_exact_2d_angular_interval_socp",
            K_artificial=16, R_eff=2.0,
        ),
    )


def _event(gamma, step, *, chosen=3, negative=None, replica=0):
    segments = np.zeros((16, 11, 2), np.float32)
    for index in range(16):
        segments[index, :, 0] = np.linspace(0, 1 + index / 20, 11)
        segments[index, :, 1] = index / 20
    verification = [
        dict(valid=index == chosen, error=False, execution_cost=float(index),
             progress=.1, step_margin=.2)
        for index in range(16)
    ]
    if chosen is None:
        for row in verification:
            row["valid"] = False
    state_before = np.array([step / 10, 0, 1, 0], np.float32)
    state_after = state_before.copy()
    if chosen is not None:
        state_after[0] += .1
    event = dict(
        seed=2, gamma=float(gamma), replica=int(replica), scenario_id=300123,
        step=int(step), K=64, B=16, state_before=state_before,
        state_after=state_after, ped_xy=np.array([[2., 2.]], np.float32),
        ped_vel=np.zeros((1, 2), np.float32),
        queried_candidate_ids=list(range(16)),
        queried_controls=np.zeros((16, 10, 2), np.float32),
        queried_segments=segments, verification=verification,
        chosen_H10_progress_rank=(None if chosen is None else 2),
        eligible_progress_candidates=(0 if chosen is None else 4),
        chosen_local=chosen, archived_negative_local=negative,
        status=("nvp" if chosen is None else None), nvp_cause=None,
    )
    event["chosen_verifier_sidecar"] = (
        None if chosen is None else _sidecar(segments[int(chosen)])
    )
    return event


def _bundle():
    events = []
    for gamma in V.GAMMAS:
        events.append(_event(gamma, 0, chosen=3))
        events.append(_event(gamma, 1, chosen=None, negative=2))
    return dict(
        status=V.TRACE_STATUS, version=V.TRACE_VERSION,
        checkpoint_sha256="a" * 64, policy_state_sha256="b" * 64,
        config=dict(
            max_steps=30, K=64, B=16, flow_base_std=3.0,
            execution_rule="weighted_cost", execution_step_margin_weight=50_000.,
            parallel_episodes_per_gamma=16,
        ),
        lineage_filter=dict(seed=2, replica=0, gammas=list(V.GAMMAS)),
        semantics=dict(green="queries", blue="executed", red="not executed", black="path"),
        events=events,
    )


def test_trace_contract_and_red_endpoint_marker_are_truthful():
    bundle = _bundle()
    V.validate_trace(bundle)
    rows = V.grouped_events(bundle)[.1]
    figure, axis = plt.subplots()
    V.draw_lineage(axis, rows, 1)
    colors = [line.get_color() for line in axis.lines]
    assert colors.count(V.GREEN) == 16
    assert colors.count(V.RED) == 2  # branch plus endpoint-only x
    assert V.BLUE not in colors
    marker = next(line for line in axis.lines if line.get_marker() == "x")
    segment = rows[-1]["queried_segments"][2]
    np.testing.assert_array_equal(marker.get_xdata(), [segment[-1, 0]])
    np.testing.assert_array_equal(marker.get_ydata(), [segment[-1, 1]])
    plt.close(figure)


def test_blue_branch_retains_exact_outer_set_and_ten_level_sets():
    bundle = _bundle()
    event = V.grouped_events(bundle)[.5][0]
    geometry = V.chosen_green_geometry(event)
    assert geometry["artificial_faces"] == 16
    assert len(geometry["levels"]) == 10
    assert geometry["outer"].shape[1] == 2


def test_trace_rejects_noncanonical_candidate_budget():
    bundle = _bundle()
    bundle["config"]["B"] = 4
    with pytest.raises(ValueError, match="canonical 16"):
        V.validate_trace(bundle)


def test_trace_accepts_an_authenticated_nonzero_paired_replica():
    bundle = _bundle()
    bundle["lineage_filter"]["replica"] = 2
    for event in bundle["events"]:
        event["replica"] = 2
    V.validate_trace(bundle)


def test_lambda_zero_reference_is_drawn_without_becoming_execution():
    bundle = _bundle()
    event = bundle["events"][0]
    event["performance_reference_local"] = 0
    event["verification"][0]["valid"] = True
    V.validate_trace(bundle)
    figure, axis = plt.subplots()
    V.draw_lineage(axis, [event], 0)
    colors = [line.get_color() for line in axis.lines]
    assert V.REFERENCE in colors
    assert V.BLUE in colors
    assert any("rank=2/4" in text.get_text() for text in axis.texts)
    plt.close(figure)
