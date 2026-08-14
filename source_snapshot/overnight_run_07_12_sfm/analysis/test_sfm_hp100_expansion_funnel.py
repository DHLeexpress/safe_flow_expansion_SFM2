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


def test_screening_bank_is_m20_per_gamma_on_a_single_crn_bank():
    banks = FUNNEL.DECLARED_EVAL_BANKS["screen_m20"]
    assert {bank["scene_profile"] for bank in banks} == {
        FUNNEL.OOD_PROFILE, FUNNEL.ID_PROFILE,
    }
    for bank in banks:
        assert bank["M"] == 20
        assert bank["noise_seed"] == 20_260_814
    assert FUNNEL.DECLARED_EVAL_BANKS["shortlist_m50"][0]["M"] == 50
    assert FUNNEL.DECLARED_EVAL_BANKS["confirm_m100"][0]["M"] == 100


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
        FUNNEL.run_stage(_stage_args(tmp_path, "screen-m20"))


def test_declared_sweep_is_exposure_only_with_fixed_alpha_lr_rounds():
    recipes = SEARCH.declared_recipes()
    assert [recipe.recipe_id for recipe in recipes] == ["E1", "E4", "E16"]
    assert [recipe.exposure_passes for recipe in recipes] == [1, 4, 16]
    for recipe in recipes:
        assert recipe.alpha == 0.0
        assert recipe.learning_rate == 1.0e-5
        assert recipe.rounds == 5
    qualification = SEARCH.qualification_recipes()
    assert [recipe.recipe_id for recipe in qualification] == [
        "QUAL-A0", "QUAL-A005",
    ]
    assert [recipe.alpha for recipe in qualification] == [0.0, 0.05]
    for recipe in qualification:
        assert recipe.exposure_passes == 1
        assert recipe.learning_rate == 1.0e-5
        assert recipe.rounds == 1


def test_search_contract_declares_before_and_locks_once(tmp_path):
    contract_path = tmp_path / "SEARCH_CONTRACT.json"
    contract = SEARCH.write_contract(contract_path, r0_sha256="a" * 64)
    assert contract["status"] == SEARCH.CONTRACT_STATUS
    assert len(contract["recipes"]) == 3
    assert len(contract["qualification_recipes"]) == 2
    assert contract["alpha_retention_rule"] == SEARCH.ALPHA_RETENTION_RULE
    assert contract["per_gamma_cr_collapse"] == SEARCH.PER_GAMMA_CR_COLLAPSE
    assert contract["time_guard_seconds"] == SEARCH.TIME_GUARD_SECONDS
    assert contract["locked_winner"] is None
    with pytest.raises(FileExistsError, match="redeclare"):
        SEARCH.write_contract(contract_path, r0_sha256="a" * 64)
    m50 = tmp_path / "m50.json"
    m50.write_text("{}")
    touched = tmp_path / "confirm"
    touched.mkdir()
    with pytest.raises(RuntimeError, match="confirmation bank was touched"):
        SEARCH.lock_winner(
            contract_path, winner_label="E4", winner_sha256="b" * 64,
            m50_results_path=m50, confirm_output=touched,
        )
    locked = SEARCH.lock_winner(
        contract_path, winner_label="E4", winner_sha256="b" * 64,
        m50_results_path=m50, confirm_output=tmp_path / "confirm_untouched",
    )
    assert locked["status"] == SEARCH.LOCK_STATUS
    with pytest.raises(ValueError, match="unlocked declared contract"):
        SEARCH.lock_winner(
            contract_path, winner_label="E1", winner_sha256="c" * 64,
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


def test_shortlist_time_guard_and_per_gamma_collapse_guard(tmp_path):
    contract = SEARCH.write_contract(
        tmp_path / "contract.json", r0_sha256="a" * 64,
    )
    r0_ood = {"SR": 0.56, "CR": 0.4371, "timeout": 0.0029, "Validity": 0.4769,
              "successful_time": 7.147}
    r0_per_gamma = {
        "0.1": {"CR": 0.40, "Validity": 0.50},
        "1.0": {"CR": 0.45, "Validity": 0.45},
    }
    healthy_per_gamma = {
        "0.1": {"CR": 0.35, "Validity": 0.55},
        "1.0": {"CR": 0.40, "Validity": 0.50},
    }
    collapsed_per_gamma = {
        # Pooled improves but gamma 0.1 collapses beyond both margins.
        "0.1": {"CR": 0.40 + SEARCH.PER_GAMMA_CR_COLLAPSE + 0.01,
                "Validity": 0.50 - SEARCH.PER_GAMMA_VALIDITY_COLLAPSE - 0.01},
        "1.0": {"CR": 0.10, "Validity": 0.80},
    }
    rows = [
        {**_dev_row("healthy", "1" * 64, cr=0.30, validity=0.55,
                    clearance=0.2),
         "ood_per_gamma": healthy_per_gamma},
        {**_dev_row("gamma-collapse", "2" * 64, cr=0.25, validity=0.60,
                    clearance=0.3),
         "ood_per_gamma": collapsed_per_gamma},
        {**_dev_row("slow", "3" * 64, cr=0.20, validity=0.65, clearance=0.3),
         "ood_per_gamma": healthy_per_gamma},
    ]
    rows[2]["ood"]["successful_time"] = (
        7.147 + SEARCH.TIME_GUARD_SECONDS + 0.1
    )
    shortlist = SEARCH.select_shortlist(
        contract, rows, r0_ood=r0_ood, r0_id=r0_id_baseline(),
        r0_ood_per_gamma=r0_per_gamma,
    )
    assert [row["label"] for row in shortlist] == ["healthy"]
    ledger = {row["label"]: row for row in shortlist[0]["screen_ledger"]}
    assert "gamma_0.1_cr_collapse" in ledger["gamma-collapse"]["reasons"]
    assert "gamma_0.1_validity_collapse" in ledger["gamma-collapse"]["reasons"]
    assert "time_guard" in ledger["slow"]["reasons"]


def r0_id_baseline():
    return {"SR": 0.9571, "CR": 0.0429, "timeout": 0.0, "Validity": 0.7906}


def test_shortlist_without_id_screen_skips_the_id_guard(tmp_path):
    contract = SEARCH.write_contract(
        tmp_path / "contract.json", r0_sha256="a" * 64,
    )
    r0_ood = {"SR": 0.56, "CR": 0.4371, "timeout": 0.0029, "Validity": 0.4769}
    row = _dev_row("ood-only", "1" * 64, cr=0.3, validity=0.5, clearance=0.2)
    row["id"] = None
    shortlist = SEARCH.select_shortlist(contract, [row], r0_ood=r0_ood)
    assert [entry["label"] for entry in shortlist] == ["ood-only"]


def test_declared_recipes_v2_cross_exposure_with_the_two_surfaces():
    recipes = SEARCH.declared_recipes_v2()
    assert [recipe.recipe_id for recipe in recipes] == [
        "E1", "E4", "E16", "E1R", "E4R", "E16R",
    ]
    assert len({recipe.recipe_id for recipe in recipes}) == 6
    assert [recipe.exposure_passes for recipe in recipes] == [1, 4, 16] * 2
    assert [recipe.optimizer_scope for recipe in recipes] == (
        ["trunk_and_head"] * 3 + ["last_two_blocks_and_head"] * 3
    )
    for recipe in recipes:
        assert recipe.alpha == 0.0
        assert recipe.learning_rate == 1.0e-5
        assert recipe.rounds == 5
    # v1 recipes are implicitly the full surface.
    assert all(
        recipe.optimizer_scope == "trunk_and_head"
        for recipe in SEARCH.declared_recipes()
    )


def test_contract_v2_amends_recipes_only_and_still_shortlists(tmp_path):
    v1_path = tmp_path / "SEARCH_CONTRACT.json"
    v1 = SEARCH.write_contract(v1_path, r0_sha256="a" * 64)
    v2_path = tmp_path / "SEARCH_CONTRACT_V2.json"
    with pytest.raises(FileNotFoundError, match="superseded"):
        SEARCH.write_contract_v2(
            v2_path, r0_sha256="a" * 64,
            amendment_of=tmp_path / "missing.json", reason="x",
        )
    with pytest.raises(ValueError, match="recorded reason"):
        SEARCH.write_contract_v2(
            v2_path, r0_sha256="a" * 64, amendment_of=v1_path, reason="   ",
        )
    contract = SEARCH.write_contract_v2(
        v2_path, r0_sha256="a" * 64, amendment_of=v1_path,
        reason="add the reduced last_two_blocks_and_head surface arms",
    )
    assert contract["status"] == SEARCH.CONTRACT_STATUS
    assert len(contract["recipes"]) == 6
    amendment = contract["amendment"]
    assert amendment["supersedes"] == str(v1_path.resolve())
    assert amendment["supersedes_sha256"] == SEARCH.sha256_file(v1_path)
    assert amendment["declared_before_confirmation_read"] is True
    # Guards, banks, ranking, and shortlist size are unchanged from v1.
    for key in (
        "sr_guard", "timeout_guard", "time_guard_seconds",
        "per_gamma_cr_collapse", "per_gamma_validity_collapse",
        "shortlist_size", "ranking_key", "banks", "r0_sha256",
    ):
        assert contract[key] == v1[key]
    # The amended contract feeds select_shortlist unchanged.
    shortlist = SEARCH.select_shortlist(
        contract,
        [_dev_row("E4R_r2", "2" * 64, cr=0.30, validity=0.55, clearance=0.2)],
        r0_ood={"SR": 0.56, "CR": 0.4371, "timeout": 0.0029,
                "Validity": 0.4769},
        r0_id={"SR": 0.9571, "CR": 0.0429, "timeout": 0.0,
               "Validity": 0.7906},
    )
    assert [row["label"] for row in shortlist] == ["E4R_r2"]
    with pytest.raises(FileExistsError, match="redeclare"):
        SEARCH.write_contract_v2(
            v2_path, r0_sha256="a" * 64, amendment_of=v1_path, reason="y",
        )


def test_declared_recipes_v3_drop_reduced_and_add_last_block_arms():
    recipes = SEARCH.declared_recipes_v3()
    assert [recipe.recipe_id for recipe in recipes] == [
        "E1", "E4", "E16", "E1L", "E4L", "E16L",
    ]
    assert len({recipe.recipe_id for recipe in recipes}) == 6
    assert [recipe.exposure_passes for recipe in recipes] == [1, 4, 16] * 2
    assert [recipe.optimizer_scope for recipe in recipes] == (
        ["trunk_and_head"] * 3 + ["last_block_and_head"] * 3
    )
    assert not any("R" == recipe.recipe_id[-1] for recipe in recipes)
    for recipe in recipes:
        assert recipe.alpha == 0.0
        assert recipe.learning_rate == 1.0e-5
        assert recipe.rounds == 5


def test_contract_v3_extends_the_amendment_chain_and_still_shortlists(tmp_path):
    v1_path = tmp_path / "SEARCH_CONTRACT.json"
    v1 = SEARCH.write_contract(v1_path, r0_sha256="a" * 64)
    v2_path = tmp_path / "SEARCH_CONTRACT_V2.json"
    SEARCH.write_contract_v2(
        v2_path, r0_sha256="a" * 64, amendment_of=v1_path,
        reason="add the reduced last_two_blocks_and_head surface arms",
    )
    v3_path = tmp_path / "SEARCH_CONTRACT_V3.json"
    with pytest.raises(ValueError, match="recorded reason"):
        SEARCH.write_contract_v3(
            v3_path, r0_sha256="a" * 64, amendment_of=v2_path, reason=" ",
        )
    contract = SEARCH.write_contract_v3(
        v3_path, r0_sha256="a" * 64, amendment_of=v2_path,
        reason=(
            "add the minimal last_block_and_head arms and drop the v2 "
            "last_two_blocks_and_head arms"
        ),
    )
    assert contract["status"] == SEARCH.CONTRACT_STATUS
    assert [recipe["recipe_id"] for recipe in contract["recipes"]] == [
        "E1", "E4", "E16", "E1L", "E4L", "E16L",
    ]
    amendment = contract["amendment"]
    assert amendment["supersedes"] == str(v2_path.resolve())
    assert amendment["supersedes_sha256"] == SEARCH.sha256_file(v2_path)
    # The chain records v2's own amendment block (which points at v1).
    assert amendment["supersedes_amendment"]["supersedes"] == str(
        v1_path.resolve()
    )
    assert amendment["declared_before_confirmation_read"] is True
    # Guards, banks, ranking, and shortlist size are unchanged from v1.
    for key in (
        "sr_guard", "timeout_guard", "time_guard_seconds",
        "per_gamma_cr_collapse", "per_gamma_validity_collapse",
        "shortlist_size", "ranking_key", "banks", "r0_sha256",
    ):
        assert contract[key] == v1[key]
    shortlist = SEARCH.select_shortlist(
        contract,
        [_dev_row("E4L_r2", "3" * 64, cr=0.30, validity=0.55, clearance=0.2)],
        r0_ood={"SR": 0.56, "CR": 0.4371, "timeout": 0.0029,
                "Validity": 0.4769},
        r0_id={"SR": 0.9571, "CR": 0.0429, "timeout": 0.0,
               "Validity": 0.7906},
    )
    assert [row["label"] for row in shortlist] == ["E4L_r2"]
    with pytest.raises(FileExistsError, match="redeclare"):
        SEARCH.write_contract_v3(
            v3_path, r0_sha256="a" * 64, amendment_of=v2_path, reason="y",
        )
