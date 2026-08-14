import copy

import numpy as np
import pytest

import sfm_b1_density_diagnostic as D


def _rows(scenario, *, r0, selected, kazuki):
    values = {"safemppi_expert": r0, "arm_a_r10_raw": selected, "default_kazuki": kazuki}
    return [dict(
        method=method, scenario_id=scenario, gamma=gamma, success=bool(success),
        collision=not bool(success), timeout=False, steps=10, min_clearance=.1,
    ) for method, statuses in values.items()
       for gamma, success in zip(D.DISPLAY_GAMMAS, statuses)]


def test_density_bank_is_disjoint_fixed_and_content_hashed():
    left = D.diagnostic_bank(230000, 4, 700000)
    right = D.diagnostic_bank(230000, 4, 700000)
    assert left == right
    assert left["scenarios"] == [230000, 230001, 230002, 230003]
    assert left["gammas"] == [0.1, 0.5, 1.0]
    assert left["environment"]["n_ped"] == 50
    assert left["environment"]["ped_speed_range"] == [0.5, 1.0]
    changed = copy.deepcopy(left)
    changed["scenarios"][0] += 1
    changed.pop("bank_sha256")
    assert D._canonical_sha256(changed) != left["bank_sha256"]


def test_case_selection_prefers_strict_requested_tier_over_fallback():
    rows = []
    rows += _rows(230000, r0=(1, 1, 1), selected=(1, 1, 1), kazuki=(1, 1, 1))
    rows += _rows(230001, r0=(0, 1, 1), selected=(1, 1, 1), kazuki=(1, 0, 1))
    rows += _rows(230002, r0=(0, 0, 0), selected=(0, 0, 0), kazuki=(0, 0, 0))
    chosen, scores = D.choose_scenario(rows, (230000, 230001, 230002))
    assert chosen["scenario_id"] == 230001
    assert chosen["score"]["tier"] == 0
    assert len(scores) == 3


def test_shared_method_step_minimizes_mean_distance_and_ties_earliest():
    methods = {method: {} for method in D.METHODS}
    for method_index, method in enumerate(D.METHODS):
        for gamma in D.DISPLAY_GAMMAS:
            trace = []
            for step, distance in enumerate((3.0, 1.0, 2.0)):
                trace.append(dict(
                    step=step, state=np.array([0.0, 0.0, 0.0, 0.0]),
                    ped_xy=np.array([[distance + .01 * method_index, 0.0]]),
                ))
            methods[method][str(gamma)] = dict(trace=trace)
    value = D.choose_shared_method_step(methods)
    assert value["step"] == 1
    assert not value["fallback_to_t0"]
    assert value["common_steps"] == [0, 1, 2]


def test_method_trace_rerun_must_match_selected_search_cells():
    rows = _rows(230001, r0=(0, 1, 1), selected=(1, 1, 1), kazuki=(1, 0, 1))
    payload = dict(selected_scenario_id=230001, rows=copy.deepcopy(rows))
    assert D.validate_method_rerun(payload, 230001, rows)
    changed = copy.deepcopy(rows)
    changed[0]["success"] = not changed[0]["success"]
    try:
        D.validate_method_rerun(payload, 230001, changed)
    except RuntimeError as error:
        assert "changed" in str(error)
    else:
        raise AssertionError("mismatched rerun was accepted")


def _query(candidate, positive, value):
    controls = np.full((10, 2), value, np.float32)
    return dict(
        candidate_id=candidate, controls=controls,
        result=dict(resolved=True, y=int(positive), full_h=True, terminal_step=10),
    )


def test_snapshot_selection_requires_rejection_then_maximizes_full_control_spread():
    traces = [
        dict(scenario_id=1, gamma=.1, step=0, selected_ids=[0, 1, 2, 3], executed_id=0,
             query_rows=[_query(0, True, 0), _query(1, True, .1), _query(2, False, .2), _query(3, False, .3)]),
        dict(scenario_id=1, gamma=.5, step=1, selected_ids=[0, 1, 2, 3], executed_id=0,
             query_rows=[_query(0, True, -1), _query(1, True, 1), _query(2, False, .2), _query(3, False, .3)]),
        dict(scenario_id=1, gamma=1.0, step=2, selected_ids=[0, 1, 2, 3], executed_id=0,
             query_rows=[_query(0, True, -2), _query(1, True, 2), _query(2, True, 0), _query(3, True, .2)]),
    ]
    selected, scores = D.choose_snapshot(traces)
    assert selected["trace_index"] == 1
    assert selected["tier"] == 1
    assert selected["full_h_positive"] == 2
    assert selected["verifier_rejected"] == 2
    assert selected["spread"]["D_U_mean"] > scores[0]["spread"]["D_U_mean"]
    assert 0.0 <= selected["spread"]["D_U_mean"] <= 1.0
    assert 0.0 <= selected["spread"]["D_1_mean"] <= 1.0


def test_snapshot_strict_order_is_p3n1_then_p2n2_then_p4n0():
    traces = [
        dict(scenario_id=1, gamma=.1, step=0, selected_ids=[0, 1, 2, 3], executed_id=0,
             query_rows=[_query(0, True, -2), _query(1, True, 2),
                         _query(2, True, 0), _query(3, True, .2)]),
        dict(scenario_id=1, gamma=.5, step=1, selected_ids=[0, 1, 2, 3], executed_id=0,
             query_rows=[_query(0, True, 0), _query(1, True, .1),
                         _query(2, False, .2), _query(3, False, .3)]),
        dict(scenario_id=1, gamma=1.0, step=2, selected_ids=[0, 1, 2, 3], executed_id=0,
             query_rows=[_query(0, True, 0), _query(1, True, .1),
                         _query(2, True, .2), _query(3, False, .3)]),
    ]
    selected, scores = D.choose_snapshot(traces)
    assert selected["trace_index"] == 2
    assert [row["tier"] for row in scores] == [2, 1, 0]


def test_partial_query_is_rejected_by_snapshot_scoring():
    trace = dict(
        scenario_id=1, gamma=.1, step=0, selected_ids=[0, 1, 2, 3], executed_id=0,
        query_rows=[_query(0, True, 0), _query(1, True, .1),
                    _query(2, True, .2), _query(3, False, .3)],
    )
    trace["query_rows"][2]["result"]["full_h"] = False
    trace["query_rows"][2]["result"]["terminal_step"] = 3
    with pytest.raises(ValueError, match="full H=10"):
        D.snapshot_score(trace, 0)


def test_verifier_timing_counts_attempts_errors_and_amortized_parallel_wall_time():
    traces = [
        dict(selected_ids=[0, 1, 2, 3], query_rows=[{}, {}, {}, {}]),
        dict(selected_ids=[0, 1, 2, 3], query_rows=[{}, {}, {}]),
    ]
    value = D.verifier_timing(traces, dict(timers=dict(verifier=.016)))
    assert value["queried_attempts"] == 8
    assert value["resolved"] == 7
    assert value["errors"] == 1
    assert value["mean_amortized_verifier_wall_ms_per_query"] == 2.0
    assert value["n_theta"] is None
    assert value["angular_grid"] is False
    assert value["K_artificial"] == 16
    assert "exact 2-D angular-interval" in value["verifier_implementation"]


def test_parser_defaults_to_cuda_and_declared_rbf_preflight_values():
    args = D.build_parser().parse_args([
        "collect", "--checkpoint", "a.pt", "--recent-dir", "recent",
        "--scenario", "230001", "--outdir", "out",
    ])
    assert args.device == "cuda"
    assert args.ell == D.DEFAULT_ELL
    assert args.cap == D.SP.GP_CAP == 512


def test_methods_alias_accepts_search_contract_and_defaults_to_cuda():
    args = D.build_parser().parse_args([
        "render-inputs", "--r0", "r0.pt", "--selected", "a.pt",
        "--search-json", "search.json", "--outdir", "out",
    ])
    assert args.command == "render-inputs"
    assert args.device == "cuda"
