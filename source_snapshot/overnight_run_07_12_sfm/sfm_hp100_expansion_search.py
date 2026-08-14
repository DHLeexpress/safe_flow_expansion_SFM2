"""Bounded fixed-recipe search contract for SFM2 predictive expansion.

Every recipe is declared in source below and frozen into a written
``SEARCH_CONTRACT.json`` *before* any shortlist or confirmation bank is read.
Shortlisting happens on the development M10 bank, the shortlist is evaluated
once on the fresh M50 bank, and ``lock_winner`` fixes the single confirmation
candidate before the untouched M100 bank may run.  No runner-up substitution
after M50 is read; no reading M100 before locking.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from sfm_hp100_expansion_funnel import (
    DECLARED_EVAL_BANKS,
    assert_declared_banks_static,
    sha256_file,
)


VERSION = "sfm_hp100_expansion_search_v1"
CONTRACT_STATUS = "SFM2_EXPANSION_SEARCH_DECLARED"
LOCK_STATUS = "SFM2_EXPANSION_WINNER_LOCKED"
SHORTLIST_SIZE = 3

RANKING_KEY = (
    "minimize OOD CR, then maximize OOD Validity, then maximize OOD "
    "successful_clearance; guarded by OOD SR >= r0 - sr_guard, ID SR >= "
    "r0_ID - sr_guard, timeout <= r0 + timeout_guard"
)


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    alpha: float
    rounds: int
    learning_rate: float
    replay: str = (
        "single pass over D+, no duplicates, full_set_mean negative mass"
    )


def declared_recipes() -> tuple[Recipe, ...]:
    """The complete frozen search space; edit requires a new declared contract."""
    return (
        Recipe("A0-R1", 0.00, 1, 1.0e-6),
        Recipe("A0-R3", 0.00, 3, 1.0e-6),
        Recipe("A05-R1", 0.05, 1, 1.0e-6),
        Recipe("A05-R3", 0.05, 3, 1.0e-6),
        Recipe("A15-R1", 0.15, 1, 1.0e-6),
        Recipe("A15-R3", 0.15, 3, 1.0e-6),
        Recipe("A05-R1-LR", 0.05, 1, 1.0e-5),
        Recipe("A0-R1-LR", 0.00, 1, 1.0e-5),
    )


def write_contract(
    path: str | Path,
    *,
    r0_sha256: str,
    sr_guard: float,
    timeout_guard: float,
) -> dict:
    """Freeze recipes, banks, ranking, and shortlist size before any run."""
    assert_declared_banks_static()
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(f"refusing to redeclare an existing contract: {path}")
    if len(str(r0_sha256)) != 64:
        raise ValueError("the contract requires the full r0 SHA-256")
    recipes = declared_recipes()
    if len({recipe.recipe_id for recipe in recipes}) != len(recipes):
        raise ValueError("recipe ids must be unique")
    contract = {
        "status": CONTRACT_STATUS,
        "version": VERSION,
        "r0_sha256": str(r0_sha256).lower(),
        "recipes": [asdict(recipe) for recipe in recipes],
        "banks": {
            stage: [dict(bank) for bank in banks]
            for stage, banks in DECLARED_EVAL_BANKS.items()
        },
        "ranking_key": RANKING_KEY,
        "shortlist_size": SHORTLIST_SIZE,
        "sr_guard": float(sr_guard),
        "timeout_guard": float(timeout_guard),
        "locked_winner": None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)
    return contract


def _metrics(entry: dict) -> dict:
    pooled = entry["pooled"]
    for field in ("SR", "CR", "timeout", "Validity"):
        if pooled.get(field) is None:
            raise ValueError(f"funnel result lacks pooled {field}")
    return pooled


def select_shortlist(
    contract: dict,
    dev_results: list[dict],
    *,
    r0_ood: dict,
    r0_id: dict,
) -> list[dict]:
    """Rank guarded dev-M10 candidates by the declared lexicographic key.

    ``dev_results`` rows carry ``label``, ``checkpoint_sha256``, ``ood``, and
    ``id`` pooled metric dicts from the dev-M10 funnel stage.
    """
    if contract.get("status") not in {CONTRACT_STATUS, LOCK_STATUS}:
        raise ValueError("shortlisting requires the declared search contract")
    sr_guard = float(contract["sr_guard"])
    timeout_guard = float(contract["timeout_guard"])
    guarded = []
    ledger = []
    for row in dev_results:
        ood = _metrics({"pooled": row["ood"]})
        matched = _metrics({"pooled": row["id"]})
        reasons = []
        if ood["SR"] < float(r0_ood["SR"]) - sr_guard:
            reasons.append("ood_sr_guard")
        if matched["SR"] < float(r0_id["SR"]) - sr_guard:
            reasons.append("id_sr_guard")
        if ood["timeout"] > float(r0_ood["timeout"]) + timeout_guard:
            reasons.append("timeout_guard")
        ledger.append({
            "label": row["label"], "eligible": not reasons, "reasons": reasons,
        })
        if not reasons:
            guarded.append(row)
    ordered = sorted(guarded, key=lambda row: (
        float(row["ood"]["CR"]),
        -float(row["ood"]["Validity"]),
        -(
            float("-inf") if row["ood"]["successful_clearance"] is None
            else float(row["ood"]["successful_clearance"])
        ),
        str(row["label"]),
    ))
    shortlist = ordered[:int(contract["shortlist_size"])]
    return [
        {
            "label": row["label"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "ood": dict(row["ood"]),
            "id": dict(row["id"]),
            "screen_ledger": ledger,
        }
        for row in shortlist
    ]


def lock_winner(
    contract_path: str | Path,
    *,
    winner_label: str,
    winner_sha256: str,
    m50_results_path: str | Path,
    confirm_output: str | Path,
) -> dict:
    """Lock the single M100 candidate; refuse if M100 already ran."""
    contract_path = Path(contract_path).resolve()
    contract = json.loads(contract_path.read_text())
    if contract.get("status") != CONTRACT_STATUS:
        raise ValueError("lock_winner requires an unlocked declared contract")
    if contract.get("locked_winner") is not None:
        raise ValueError("the search contract already locked a winner")
    if Path(confirm_output).exists():
        raise RuntimeError(
            "refusing to lock a winner after the confirmation bank was touched"
        )
    if len(str(winner_sha256)) != 64:
        raise ValueError("the locked winner requires its full SHA-256")
    contract["locked_winner"] = {
        "label": str(winner_label),
        "sha256": str(winner_sha256).lower(),
        "m50_results": {
            "path": str(Path(m50_results_path).resolve()),
            "sha256": sha256_file(m50_results_path),
        },
    }
    contract["status"] = LOCK_STATUS
    temporary = contract_path.with_name(f".{contract_path.name}.tmp")
    temporary.write_text(
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(contract_path)
    return contract


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    declare = sub.add_parser("declare", help="write the frozen search contract")
    declare.add_argument("--contract", required=True)
    declare.add_argument("--r0-sha256", required=True)
    declare.add_argument("--sr-guard", type=float, default=2.0 / 70.0)
    declare.add_argument("--timeout-guard", type=float, default=1.0 / 70.0)
    lock = sub.add_parser("lock", help="lock the single M100 winner")
    lock.add_argument("--contract", required=True)
    lock.add_argument("--winner-label", required=True)
    lock.add_argument("--winner-sha256", required=True)
    lock.add_argument("--m50-results", required=True)
    lock.add_argument("--confirm-output", required=True)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "declare":
        contract = write_contract(
            args.contract, r0_sha256=args.r0_sha256,
            sr_guard=float(args.sr_guard),
            timeout_guard=float(args.timeout_guard),
        )
        print(json.dumps({
            "status": contract["status"],
            "recipes": len(contract["recipes"]),
        }, sort_keys=True))
    else:
        contract = lock_winner(
            args.contract, winner_label=args.winner_label,
            winner_sha256=args.winner_sha256,
            m50_results_path=args.m50_results,
            confirm_output=args.confirm_output,
        )
        print(json.dumps({
            "status": contract["status"],
            "winner": contract["locked_winner"]["label"],
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
