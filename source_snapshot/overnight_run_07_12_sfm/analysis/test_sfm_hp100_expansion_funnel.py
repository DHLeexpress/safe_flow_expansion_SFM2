"""Gate 12: bank disjointness and staged-funnel confusion barriers."""
import argparse
import json

import pytest

import sfm_hp100_expansion_funnel as FUNNEL
import sfm_hp100_expansion_search as SEARCH


def test_declared_banks_are_static_and_disjoint():
    FUNNEL.assert_declared_banks_static()
    ranges = [
        (bank["stage"], *FUNNEL.bank_range(bank))
        for stage in FUNNEL.DECLARED_EVAL_BANKS.values()
        for bank in stage
    ]
    for index, (_, lo_a, hi_a) in enumerate(ranges):
        for _, lo_b, hi_b in ranges[index + 1:]:
            assert hi_a <= lo_b or hi_b <= lo_a


def test_acquisition_scenario_starts_are_declared_per_round():
    assert FUNNEL.acquisition_scenario_start(1) == 860_000
    assert FUNNEL.acquisition_scenario_start(3) == 880_000
    with pytest.raises(ValueError):
        FUNNEL.acquisition_scenario_start(0)


def _stage_args(tmp_path, stage, **overrides):
    values = dict(
        stage=stage,
        checkpoint=["r0=" + str(tmp_path / "missing.pt")],
        output=str(tmp_path / "out"),
        device="cpu", physical_gpu=3, verifier_workers=1,
        search_contract=None, status_json=None, heartbeat_seconds=30.0,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_confirm_m100_refuses_to_run_without_a_locked_contract(tmp_path):
    with pytest.raises(ValueError, match="locked search contract"):
        FUNNEL.run_stage(_stage_args(tmp_path, "confirm-m100"))


def test_confirm_m100_refuses_an_unlocked_or_foreign_checkpoint(tmp_path):
    contract = tmp_path / "contract.json"
    contract.write_text(json.dumps({
        "locked_winner": {"label": "A05-R1", "sha256": "f" * 64},
        "r0_sha256": "e" * 64,
    }))
    junk = tmp_path / "foreign.pt"
    junk.write_bytes(b"foreign")
    args = _stage_args(
        tmp_path, "confirm-m100",
        checkpoint=[f"winner={junk}"],
        search_contract=str(contract),
    )
    with pytest.raises(ValueError, match="refuses unlocked checkpoint"):
        FUNNEL.run_stage(args)


def test_stage_refuses_an_existing_output_directory(tmp_path):
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError, match="existing funnel stage output"):
        FUNNEL.run_stage(_stage_args(tmp_path, "dev-m10"))


def test_search_contract_declares_before_and_locks_once(tmp_path):
    contract_path = tmp_path / "SEARCH_CONTRACT.json"
    contract = SEARCH.write_contract(
        contract_path, r0_sha256="a" * 64,
        sr_guard=2.0 / 70.0, timeout_guard=1.0 / 70.0,
    )
    assert contract["status"] == SEARCH.CONTRACT_STATUS
    assert len(contract["recipes"]) == 8
    assert contract["locked_winner"] is None
    with pytest.raises(FileExistsError, match="redeclare"):
        SEARCH.write_contract(
            contract_path, r0_sha256="a" * 64,
            sr_guard=2.0 / 70.0, timeout_guard=1.0 / 70.0,
        )
    m50 = tmp_path / "m50.json"
    m50.write_text("{}")
    touched = tmp_path / "confirm"
    touched.mkdir()
    with pytest.raises(RuntimeError, match="confirmation bank was touched"):
        SEARCH.lock_winner(
            contract_path, winner_label="A05-R1", winner_sha256="b" * 64,
            m50_results_path=m50, confirm_output=touched,
        )
    locked = SEARCH.lock_winner(
        contract_path, winner_label="A05-R1", winner_sha256="b" * 64,
        m50_results_path=m50, confirm_output=tmp_path / "confirm_untouched",
    )
    assert locked["status"] == SEARCH.LOCK_STATUS
    with pytest.raises(ValueError, match="unlocked declared contract"):
        SEARCH.lock_winner(
            contract_path, winner_label="A0-R1", winner_sha256="c" * 64,
            m50_results_path=m50, confirm_output=tmp_path / "confirm_untouched",
        )


def _dev_row(label, sha, *, cr, validity, clearance, sr=0.6, timeout=0.0,
             id_sr=0.95):
    return {
        "label": label,
        "checkpoint_sha256": sha,
        "ood": {"SR": sr, "CR": cr, "timeout": timeout, "Validity": validity,
                "successful_clearance": clearance},
        "id": {"SR": id_sr, "CR": 1.0 - id_sr, "timeout": 0.0,
               "Validity": 0.8, "successful_clearance": 0.3},
    }


def test_shortlist_ranks_by_cr_validity_clearance_with_guards(tmp_path):
    contract = SEARCH.write_contract(
        tmp_path / "contract.json", r0_sha256="a" * 64,
        sr_guard=2.0 / 70.0, timeout_guard=1.0 / 70.0,
    )
    r0_ood = {"SR": 0.56, "CR": 0.4371, "timeout": 0.0029, "Validity": 0.4769}
    r0_id = {"SR": 0.9571, "CR": 0.0429, "timeout": 0.0, "Validity": 0.7906}
    rows = [
        _dev_row("worse-cr", "1" * 64, cr=0.5, validity=0.5, clearance=0.2),
        _dev_row("best-cr", "2" * 64, cr=0.3, validity=0.5, clearance=0.2),
        _dev_row("tie-cr-better-validity", "3" * 64, cr=0.35, validity=0.6,
                 clearance=0.1),
        _dev_row("tie-cr-worse-validity", "4" * 64, cr=0.35, validity=0.5,
                 clearance=0.9),
        _dev_row("guarded-out", "5" * 64, cr=0.01, validity=0.9, clearance=0.9,
                 sr=0.40),
    ]
    shortlist = SEARCH.select_shortlist(
        contract, rows, r0_ood=r0_ood, r0_id=r0_id,
    )
    assert [row["label"] for row in shortlist] == [
        "best-cr", "tie-cr-better-validity", "tie-cr-worse-validity",
    ]
    ledger = {row["label"]: row for row in shortlist[0]["screen_ledger"]}
    assert not ledger["guarded-out"]["eligible"]
    assert "ood_sr_guard" in ledger["guarded-out"]["reasons"]
