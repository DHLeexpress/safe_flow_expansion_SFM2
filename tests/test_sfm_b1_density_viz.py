from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

import sfm_b1_density_viz as V
import sfm_b1_expert as X
import sfm_b1_method_viz as MV


SQUARE = tuple(
    SimpleNamespace(
        a=np.array([np.cos(2.0 * np.pi * index / 16), np.sin(2.0 * np.pi * index / 16)]),
        m=1.0, feasible=True, kind="artificial",
    )
    for index in range(16)
)


def _segment(scale=1.0):
    x = np.linspace(0.0, float(scale), 11)
    return np.stack([x, 0.2 * x], axis=1)


def _result(positive, scale=1.0):
    return dict(
        resolved=True, y=int(positive), full_h=True, terminal_step=10,
        segment=_segment(scale), faces=list(SQUARE),
        diagnostics=dict(
            worst_t=4, solver="exact_2d_angular_interval_socp", K_artificial=16,
        ),
    )


def _partial_result(scale=1.0):
    value = _result(True, scale)
    value.update(full_h=False, terminal_step=4)
    return value


def _query_trace(*, scenario=7, gamma=.5, step=3, spread=1.0, negatives=1):
    segments = [_segment(scale) for scale in (1.0, 1.0 + spread, .7, .9)]
    all_k = [
        dict(candidate_id=index, controls=np.full((10, 2), index * spread, np.float32),
             segment=segment, mode="left")
        for index, segment in enumerate(segments)
    ]
    query_rows = []
    for index in range(4):
        positive = index < 4 - int(negatives)
        query_rows.append(dict(
            candidate_id=index, controls=all_k[index]["controls"], result=_result(positive, segments[index][-1, 0]),
        ))
    return dict(
        round=1, scenario_id=scenario, gamma=gamma, step=step,
        state=np.zeros(4), ped_xy=np.array([[4.0, 4.0]]), ped_vel=np.zeros((1, 2)),
        all_K=all_k, selected_ids=[0, 1, 2, 3], query_rows=query_rows, executed_id=0,
    )


def test_margin_frame_has_no_nominal_and_only_executed_verifier_levels():
    figure, axis = plt.subplots()
    report = V.draw_margin_query_frame(axis, _query_trace())
    plt.close(figure)
    assert report["nominal_drawn"] is False
    assert report["executed_verifier"]["rendered_levels"] == 10
    assert report["rejected_ids"] == [3]


def test_legacy_partial_query_is_rejected_instead_of_rendered():
    trace = _query_trace()
    trace["query_rows"][0]["result"] = _partial_result()
    trace["executed_id"] = 0
    figure, axis = plt.subplots()
    try:
        with pytest.raises(ValueError, match="legacy partial B1 query"):
            V.draw_margin_query_frame(axis, trace)
    finally:
        plt.close(figure)


def test_query_snapshot_ranking_prefers_few_rejections_then_control_spread():
    traces = [
        _query_trace(scenario=2, spread=4.0, negatives=2),
        _query_trace(scenario=3, spread=.5, negatives=1),
        _query_trace(scenario=4, spread=2.0, negatives=1),
    ]
    ranked = V.rank_query_snapshots(traces)
    assert ranked["chosen"]["scenario_id"] == 4
    assert ranked["chosen"]["rejected"] == 1


def test_control_spread_uses_driver_normalization_and_is_bounded():
    trace = _query_trace()
    trace["query_rows"][0]["controls"][:] = -2.0
    trace["query_rows"][1]["controls"][:] = 2.0
    trace["query_rows"][2]["controls"][:] = 0.0
    spread, _ = V._positive_control_spread(trace)
    assert spread == 1.0
    row = V._snapshot_row(trace)
    assert "2*u_max*sqrt(2H)" in row["control_spread_normalization"]


def test_candidate_snapshot_never_draws_a_rejected_query_polytope(tmp_path):
    output = tmp_path / "queries.png"
    report = V.render_candidate_query_snapshot([_query_trace()], str(output))
    assert output.exists()
    rejected = [row for row in report["candidates"] if row["status"] == "negative"]
    assert len(rejected) == 1
    assert rejected[0]["verifier"] is None
    assert rejected[0]["rejected_x"] is not None
    positive = [row for row in report["candidates"] if row["status"] == "positive_full_h"]
    assert len(positive) == 3
    assert all(row["verifier"]["rendered_levels"] == 10 for row in positive)


def _run(success, episode=11, clearance=.2):
    return dict(episode=episode, success=success, collision=not success,
                min_clearance=clearance, trace=[dict(step=0)], states=np.zeros((1, 4)))


def _method_runs(episode, *, selected=True, expert=False, kazuki=False, clearance=.2):
    return {
        "expert": {gamma: _run(expert, episode, clearance) for gamma in V.DISPLAY_GAMMAS},
        "selected": {gamma: _run(selected, episode, clearance) for gamma in V.DISPLAY_GAMMAS},
        "kazuki": {gamma: _run(kazuki, episode, clearance) for gamma in V.DISPLAY_GAMMAS},
    }


def test_exhaustive_episode_selection_is_declared_and_deterministic():
    bank = {
        5: _method_runs(5, clearance=.1),
        4: _method_runs(4, clearance=.3),
    }
    selected = V.select_comparison_episode(bank)
    assert selected["chosen"]["episode"] == 4
    assert "exhaustive fixed bank" in selected["rule"]


def test_clean_axis_has_no_subplot_title_or_labels():
    figure, axis = plt.subplots()
    V._set_clean_axis(axis)
    assert axis.get_title() == ""
    assert axis.get_xlabel() == ""
    assert axis.get_ylabel() == ""
    plt.close(figure)


def test_expert_config_and_stored_nominal_geometry_are_faithful(monkeypatch):
    config = X.demonstration_config()
    assert config.horizon == 10 and config.num_samples == 2048
    assert config.temperature == .1 and config.polytope_nbase == 16
    assert config.centroid_gain == .2 and config.centroid_smooth == .25
    assert config.centroid_eps == .15 and config.smooth_weight == .12
    assert config.predict_gain == .25 and config.warm_start
    trace = dict(
        state=np.zeros(4), gamma=.5, ped_xy=np.zeros((0, 2)),
        nominal_polytope=dict(
            A=np.asarray([row.a for row in SQUARE]),
            b=np.ones(16), margins=np.ones(16),
            n_base=16, velocity_used=True,
        ),
    )
    monkeypatch.setattr(V, "polytope_HP", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("stored expert geometry must not be recomputed")))
    nominal = V.nominal_safemppi_levels(trace)
    assert nominal["base_faces"] == 16 and nominal["velocity_used"] is True


def test_expert_trace_accepts_one_float32_ulp_between_action_and_mean(monkeypatch):
    controls = np.zeros((10, 2), np.float32)
    controls[0, 0] = 1.0
    action = controls[0].copy()
    action[0] = np.nextafter(action[0], np.float32(2.0))
    trace = dict(
        step=0, state=np.zeros(4), action=action, controls=controls,
        planned_states=np.zeros((11, 4)), ped_xy=np.zeros((0, 2)),
        ped_vel=np.zeros((0, 2)), sequence_kind="reward_weighted_mean",
    )
    run = dict(states=np.zeros((2, 4)), trace=[trace])
    monkeypatch.setattr(V, "nominal_safemppi_levels", lambda *args, **kwargs: dict(
        polygons=[], outer_polygon=np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]),
        contains_robot=True,
        base_faces=16, detected_faces=0, velocity_used=True,
    ))
    figure, axis = plt.subplots()
    report = V.draw_method_panel(axis, "expert", run, .5, 0)
    plt.close(figure)
    assert report["expert_sequence_kind"] == "reward_weighted_mean"


def test_selected_full_h_positive_draws_all_verifier_levels_without_orange_prefix():
    planned = np.zeros((11, 4), np.float32)
    planned[:, :2] = _segment()
    trace = dict(
        step=0, state=np.zeros(4), controls=np.zeros((10, 2), np.float32),
        planned_states=planned, ped_xy=np.zeros((0, 2)), ped_vel=np.zeros((0, 2)),
    )
    run = dict(states=np.zeros((2, 4)), trace=[trace])
    figure, axis = plt.subplots()
    report = V.draw_method_panel(
        axis, "selected", run, .5, 0, verifier_result=_result(True),
    )
    orange_prefixes = [
        line for line in axis.lines
        if line.get_color() == V.BV.ORANGE and line.get_linestyle() == "--"
    ]
    red_rejections = [
        line for line in axis.lines
        if line.get_color() == V.BV.RED and line.get_marker() == "x"
    ]
    plt.close(figure)
    assert report["rejected"] is False
    assert report["terminal_step"] == 10
    assert report["verifier_full_h_positive"] is True
    assert report["verifier_levels"] == 10
    assert orange_prefixes == []
    assert red_rejections == []


def test_method_bundle_accepts_density_diagnostic_aliases():
    bundle = dict(scenario_id=11, shared_snapshot=dict(step=0), runs={
        "safemppi_expert": {gamma: _run(False) for gamma in V.DISPLAY_GAMMAS},
        "arm_a_r10_raw": {str(gamma): _run(True) for gamma in V.DISPLAY_GAMMAS},
        "default_kazuki": {f"{gamma:g}": _run(False) for gamma in V.DISPLAY_GAMMAS},
    })
    normalized = V.normalize_method_bundle(bundle, scenario_id=11)
    assert tuple(normalized) == V.METHOD_KEYS
    assert all(normalized["selected"][gamma]["success"] for gamma in V.DISPLAY_GAMMAS)


def test_method_only_manifest_authenticates_trace_bundle(tmp_path, monkeypatch):
    method_path = tmp_path / "methods.pt"
    torch.save(dict(shared_snapshot=dict(step=0), runs={}), method_path)
    monkeypatch.setattr(MV.V, "normalize_method_bundle", lambda *args, **kwargs: {})

    def fake_render(_runs, png, *, snapshot_step, output_mp4, report_path):
        for path in (png, output_mp4, report_path):
            __import__("pathlib").Path(path).write_bytes(b"render")
        return dict(explicit_snapshot_step=snapshot_step)

    monkeypatch.setattr(MV.V, "render_method_gamma_comparison", fake_render)
    manifest = MV.render(method_path, tmp_path / "method-render", scenario=7)
    assert manifest["source"] == {
        "path": str(method_path.resolve()),
        "sha256": MV._sha256(method_path),
    }


def test_render_bundle_is_render_only_and_writes_manifest(tmp_path, monkeypatch):
    method_path = tmp_path / "methods.pt"
    trace_path = tmp_path / "traces.pt"
    snapshot_path = tmp_path / "snapshot.pt"
    bundle = dict(scenario_id=7, shared_snapshot=dict(step=6, rule="test shared step"), runs={
        "expert": {gamma: _run(False, 7) for gamma in V.DISPLAY_GAMMAS},
        "selected": {gamma: _run(True, 7) for gamma in V.DISPLAY_GAMMAS},
        "kazuki": {gamma: _run(False, 7) for gamma in V.DISPLAY_GAMMAS},
    })
    trace = _query_trace(scenario=7)
    torch.save(bundle, method_path); torch.save([trace], trace_path); torch.save(trace, snapshot_path)

    seen = {}
    def fake_comparison(_runs, png, *, snapshot_step, output_mp4):
        seen["method_step"] = snapshot_step
        for path in (png, output_mp4):
            __import__("pathlib").Path(path).write_bytes(b"comparison")
        return dict(explicit_snapshot_step=snapshot_step)

    def fake_gathering(_traces, scenario, mp4, *, output_snapshot, snapshot_step):
        seen["query_step"] = snapshot_step
        for path in (mp4, output_snapshot):
            __import__("pathlib").Path(path).write_bytes(b"gathering")
        return dict(scenario_id=scenario, snapshot_metadata=dict(step=snapshot_step))

    def fake_candidate(_traces, png, *, selection):
        __import__("pathlib").Path(png).write_bytes(b"candidate")
        return dict(output=png, selection=selection, candidates=[])

    monkeypatch.setattr(V, "render_method_gamma_comparison", fake_comparison)
    monkeypatch.setattr(V, "render_margin_gathering_video", fake_gathering)
    monkeypatch.setattr(V, "render_candidate_query_snapshot", fake_candidate)
    manifest = V.render_bundle(
        str(method_path), str(trace_path), 7, str(tmp_path / "rendered"),
        snapshot_trace_path=str(snapshot_path),
    )
    assert manifest["controllers_rerun_by_renderer"] is False
    assert seen == {"method_step": 6, "query_step": 3}
    assert manifest["snapshot_steps"]["method_comparison"]["step"] == 6
    assert manifest["snapshot_steps"]["query_gathering_and_candidates"]["step"] == 3
    assert manifest["query_horizon_contract"] == (
        "every resolved B query is reverified over all H=10 transitions"
    )
    assert (tmp_path / "rendered" / "render_manifest.json").exists()
