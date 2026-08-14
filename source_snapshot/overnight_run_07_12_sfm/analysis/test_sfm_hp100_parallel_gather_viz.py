import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import sfm_hp100_parallel_gather_viz as V


def _faces():
    radius = 2.0
    outer_margin = radius * np.cos(np.pi / 16)
    faces = [
        dict(
            a=[float(np.cos(2 * np.pi * index / 16)),
               float(np.sin(2 * np.pi * index / 16))],
            m=float(outer_margin), feasible=True,
            kind="artificial", label=f"art{index}",
        )
        for index in range(16)
    ]
    obstacle = np.array([1.0, 1.0])
    normal = obstacle / np.linalg.norm(obstacle)
    faces.append(dict(
        a=normal.tolist(), m=float(np.linalg.norm(obstacle) - .2),
        feasible=True, kind="real", label="ped0",
    ))
    return faces


def _candidate(candidate_id):
    segment = np.stack((
        np.linspace(0, .2 + .01 * candidate_id, 11),
        np.full(11, .005 * candidate_id),
    ), axis=1)
    return dict(candidate_id=candidate_id, segment=segment.tolist())


def _query(candidate, *, positive):
    return dict(
        candidate_id=int(candidate["candidate_id"]),
        execution_eligible=bool(positive),
        result=dict(
            resolved=True, y=int(positive), full_h=True, terminal_step=10,
            segment=candidate["segment"], faces=(_faces() if positive else []),
            diagnostics=dict(
                solver="stored-test-fixture", sensing_radius=2.0,
                R_eff=2.0, rho_art=.16,
            ),
        ),
    )


def _event(replica, *, positive=True, committed=False, scenario_id=None):
    candidates = [_candidate(index) for index in range(16)]
    selected = [0, 1, 2, 3]
    queries = [
        _query(candidates[index], positive=(positive and index < 2))
        for index in selected
    ]
    return dict(
        round=1, gamma=.5, retry_batch=0, replica=replica,
        lineage_id=f"lineage_{replica:02d}", step=0,
        scenario_id=(f"scene_{replica:02d}" if scenario_id is None else scenario_id),
        state=[0., 0., 0., 0.], ped_xy=[[1., 1.]],
        all_K=candidates, selected_ids=selected, query_rows=queries,
        executed_id=(0 if positive else None),
        episode_status=("success" if positive else "nvp"),
        committed_success=bool(committed),
    )


def _metadata():
    return dict(
        schema_version=V.SCHEMA_VERSION, K=16, B=4, H=10,
        parallel_episodes=16, expected_artificial_faces=16,
        verifier_contract=dict(name="exact stored test verifier"),
        plot_bounds=[-1., 3., -1., 3.], pedestrian_radius=.2,
    )


def _events(*, fixed_scene=False):
    return [
        _event(
            replica, positive=(replica % 3 != 0), committed=(replica == 1),
            scenario_id=("fixed_17" if fixed_scene else None),
        )
        for replica in range(16)
    ]


def test_json_and_jsonl_load_the_same_trace(tmp_path):
    metadata, events = _metadata(), _events()
    json_path = tmp_path / "trace.json"
    json_path.write_text(json.dumps(dict(metadata=metadata, events=events)))
    got_metadata, got_events = V.load_trace(json_path)
    assert got_metadata == metadata and got_events == events

    jsonl_path = tmp_path / "trace.jsonl"
    rows = [dict(type="metadata", metadata=metadata)]
    rows.extend(dict(type="event", event=event) for event in events)
    jsonl_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    jsonl_metadata, jsonl_events = V.load_trace(jsonl_path)
    assert jsonl_metadata == metadata and jsonl_events == events


def test_validation_enforces_sixteen_lineages_and_nvp_execution_invariant():
    drifted = _events()
    for event in drifted:
        event["gamma"] = .50000001
    events = V.validate_trace(_metadata(), drifted)
    assert len(events) == 16
    assert len({(event["replica"], event["lineage_id"]) for event in events}) == 16
    assert len(V.select_batch(
        events, round_index=1, gamma=.5, retry_batch=0,
    )) == 16

    broken = _events()
    broken[0]["executed_id"] = 0
    try:
        V.validate_trace(_metadata(), broken)
    except ValueError as error:
        assert "eligible exact-positive" in str(error)
    else:
        raise AssertionError("executing an exact-negative NVP candidate must fail")

    try:
        V.validate_trace(_metadata(), _events()[:-1])
    except ValueError as error:
        assert "expected 16" in str(error)
    else:
        raise AssertionError("an incomplete parallel batch must fail")


def test_positive_candidate_has_ten_polygons_from_stored_faces_only():
    event = _event(0, positive=True)
    query = event["query_rows"][0]
    levels = V.stored_level_polygons(event, query)
    assert [horizon for horizon, _ in levels] == list(range(1, 11))
    assert all(polygon.shape[1] == 2 and len(polygon) >= 3 for _, polygon in levels)


def test_summary_distinguishes_fixed_diagnostic_from_committed_replay():
    summary = V.committed_summary(_events(fixed_scene=True))
    assert summary["lineages"] == 16
    assert summary["outcomes"]["nvp"] == 6
    assert summary["committed_success_lineages"] == 1
    assert summary["panel_scenario_mode"] == "fixed_scenario_diagnostic"
    assert summary["scenario_ids"] == ["fixed_17"]


def test_candidate_panels_and_complete_marker_are_hash_authenticated(tmp_path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(json.dumps(dict(metadata=_metadata(), events=_events())))
    event = _event(1, positive=True, committed=True)
    png = tmp_path / "candidate.png"
    pdf = tmp_path / "candidate.pdf"
    V.render_candidate_specific(_metadata(), event, png, pdf)
    assert png.stat().st_size > 0 and pdf.stat().st_size > 0

    summary = V.committed_summary(_events())
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    marker = V.write_trace_complete(
        tmp_path, source_trace=trace_path,
        selection=dict(round=1, gamma=.5, retry_batch=0), summary=summary,
        artifacts=(png, pdf, summary_path),
    )
    payload = json.loads(marker.read_text())
    assert payload["status"] == V.TRACE_COMPLETE_STATUS
    assert payload["source_trace"]["sha256"] == V.sha256_file(trace_path)
    assert payload["artifacts"][png.name]["sha256"] == V.sha256_file(png)


def test_candidate_face_audit_draws_all_anchors_contacts_and_envelope():
    event = _event(1, positive=True, committed=True)
    query = event["query_rows"][0]
    figure, axis = plt.subplots()
    V._draw_stored_face_audit(
        axis, event, query, _metadata(), (-2.5, 2.5, -2.5, 2.5),
    )
    gids = [artist.get_gid() for artist in axis.get_children()]
    face_gids = [gid for gid in gids if isinstance(gid, str)
                 and gid.startswith("artificial-face-")]
    assert len(face_gids) == 16
    assert "artificial-face-active" in face_gids
    assert "artificial-face-redundant" in face_gids
    assert gids.count("artificial-anchor") == 16
    assert gids.count("tangent-contact") == 17
    assert gids.count("min-affine-envelope") == 1
    plt.close(figure)


def test_candidate_audit_bounds_show_the_complete_fixed_sensing_ring():
    event = _event(1, positive=True)
    bounds = V._candidate_audit_bounds(_metadata(), event)
    center = np.asarray(event["state"][:2])
    assert bounds[0] < center[0] - 2.16 and bounds[1] > center[0] + 2.16
    assert bounds[2] < center[1] - 2.16 and bounds[3] > center[1] + 2.16


def test_draw_nvp_panel_does_not_invent_an_executed_branch():
    event = _event(0, positive=False)
    figure, axis = plt.subplots()
    V.draw_lineage_event(
        axis, event, np.asarray([[0., 0.]]), _metadata(), (-1., 3., -1., 3.),
    )
    blue = [line for line in axis.lines if line.get_color() == V.BLUE]
    red_x = [line for line in axis.lines if line.get_marker() == "x"]
    plt.close(figure)
    assert not blue
    assert len(red_x) == 4


def test_cutoff_requires_explicit_nonreplay_diagnostic_contract():
    events = [_event(replica, positive=True) for replica in range(16)]
    for event in events:
        event["episode_status"] = "cutoff"
        event["committed_success"] = False
    metadata = _metadata()
    try:
        V.validate_trace(metadata, events)
    except ValueError as error:
        assert "diagnostic_only=true" in str(error)
    else:
        raise AssertionError("cutoff without a non-replay diagnostic contract must fail")

    metadata.update(diagnostic_only=True, enters_replay=False, goal=[2., 2.])
    assert len(V.validate_trace(metadata, events)) == 16


def test_declared_goal_is_rendered_as_a_star():
    metadata = _metadata()
    metadata["goal"] = [2., 2.]
    event = _event(0, positive=True)
    figure, axis = plt.subplots()
    V.draw_lineage_event(
        axis, event, np.asarray([[0., 0.]]), metadata, (-1., 3., -1., 3.),
    )
    stars = [line for line in axis.lines if line.get_marker() == "*"]
    plt.close(figure)
    assert len(stars) == 1
    assert np.allclose([stars[0].get_xdata()[0], stars[0].get_ydata()[0]], [2., 2.])
