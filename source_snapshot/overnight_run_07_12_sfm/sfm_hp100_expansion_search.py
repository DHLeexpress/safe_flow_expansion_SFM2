"""Bounded fixed-recipe search contract for SFM2 predictive expansion.

The declared sweep has exactly one primary variable: effective replay/update
exposure ``E`` — complete deterministically reshuffled passes over the eligible
round archive — at the historically stable values ``E in {1, 4, 16}``.  Alpha
is fixed at 0 and the learning rate at 1e-5 for every sweep arm; every arm
runs cumulatively through rounds r1..r5, and every saved checkpoint r0..r5 is
screened on the same fixed disjoint raw M20-per-gamma CRN bank.  Learning rate
and round count are *not* swept.

Qualification (run once, before the sweep) compares alpha=0 against
alpha=0.05 on the identical shared round-1 archive; alpha=0.05 survives into
later arms only under the declared retention rule below.

The amended v2 sweep (``declared_recipes_v2``) crosses the same E values with
the two declared trainable surfaces — full ``trunk_and_head`` (``E{n}``) and
reduced ``last_two_blocks_and_head`` (``E{n}R``, trunk.inp frozen) — with
alpha, learning rate, rounds, guards, banks, and ranking unchanged.

Every recipe is declared in source and frozen into a written
``SEARCH_CONTRACT.json`` *before* any shortlist or confirmation bank is read.
Shortlisting happens on the M20 screening bank, finalists are evaluated once
on the fresh M50 bank, and ``lock_winner`` fixes the single confirmation
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


VERSION = "sfm_hp100_expansion_search_v2"
CONTRACT_STATUS = "SFM2_EXPANSION_SEARCH_DECLARED"
LOCK_STATUS = "SFM2_EXPANSION_WINNER_LOCKED"
SHORTLIST_SIZE = 3

# Fixed non-swept recipe surface.
SWEEP_ALPHA = 0.0
SWEEP_LEARNING_RATE = 1.0e-5
SWEEP_ROUNDS = 5
SWEEP_EXPOSURE_PASSES = (1, 4, 16)

# Declared screening guards.  The M20 OOD screen is 7 gammas x 20 = 140
# rollouts per checkpoint, so one episode moves a pooled rate by 1/140.
# SR may drop at most 4 episodes (~2.9%, matching the historical 2/70 M10
# margin in absolute terms); timeout may rise at most 2 episodes.
SR_GUARD = 4.0 / 140.0
TIMEOUT_GUARD = 2.0 / 140.0
# Successful time-to-goal may not worsen by more than this pooled margin
# (r0 OOD is 7.147 s; 0.5 s is ~7% and well above M20 sampling noise).
TIME_GUARD_SECONDS = 0.5
# Per-gamma collapse guard: pooled improvement bought by wrecking a single
# gamma is a failure.  At M20 per gamma one episode is 5 points, so 10 points
# (= 2 episodes) is the smallest margin distinguishable from noise.
PER_GAMMA_CR_COLLAPSE = 0.10
PER_GAMMA_VALIDITY_COLLAPSE = 0.10

RANKING_KEY = (
    "r0-relative four-metric improvement, lexicographic: minimize OOD CR, "
    "then maximize OOD Validity, then maximize OOD successful_clearance, "
    "then minimize OOD successful_time; guarded by OOD SR >= r0 - sr_guard, "
    "ID SR >= r0_ID - sr_guard when the ID screen ran, timeout <= r0 + "
    "timeout_guard, successful_time <= r0 + time_guard_seconds, and the "
    "per-gamma collapse guard on CR/Validity"
)

ALPHA_RETENTION_RULE = (
    "alpha=0.05 is retained beyond qualification only if, on the identical "
    "shared round-1 archive and the same M20 screen, it improves OOD CR or "
    "Validity beyond the corresponding guard margin relative to the alpha=0 "
    "arm without violating the SR or timeout guard; otherwise every later "
    "arm runs alpha=0"
)


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    alpha: float
    exposure_passes: int
    learning_rate: float
    rounds: int
    # Declared trainable surface; the narrowed scopes leave trunk.inp (and,
    # for last_block_and_head, blocks[0]) frozen alongside the condition
    # encoders.
    optimizer_scope: str = "trunk_and_head"
    replay: str = (
        "E complete reshuffled passes over D+ per round, no oversampling, "
        "full_set_mean negative mass, cumulative rounds with persistent Adam"
    )


def declared_recipes() -> tuple[Recipe, ...]:
    """The complete frozen sweep space; editing requires a new contract."""
    return tuple(
        Recipe(
            f"E{exposure}", SWEEP_ALPHA, exposure,
            SWEEP_LEARNING_RATE, SWEEP_ROUNDS,
        )
        for exposure in SWEEP_EXPOSURE_PASSES
    )


def declared_recipes_v2() -> tuple[Recipe, ...]:
    """The amended sweep space: the E sweep crossed with the two declared
    trainable surfaces.  ``E{n}`` arms train ``trunk_and_head``; ``E{n}R``
    arms train the reduced ``last_two_blocks_and_head`` surface."""
    return tuple(
        Recipe(
            f"E{exposure}{suffix}", SWEEP_ALPHA, exposure,
            SWEEP_LEARNING_RATE, SWEEP_ROUNDS, optimizer_scope=scope,
        )
        for scope, suffix in (
            ("trunk_and_head", ""), ("last_two_blocks_and_head", "R"),
        )
        for exposure in SWEEP_EXPOSURE_PASSES
    )


def declared_recipes_v3() -> tuple[Recipe, ...]:
    """The re-amended sweep space: the E sweep crossed with the two retained
    trainable surfaces.  ``E{n}`` arms train the full ``trunk_and_head``
    surface (327,956 params); ``E{n}L`` arms train the minimal
    ``last_block_and_head`` surface (blocks[1] + head, 137,236 params, with
    trunk.inp and blocks[0] frozen).  The v2 ``E{n}R``
    ``last_two_blocks_and_head`` arms are dropped by this amendment (the
    scope itself stays supported for the recorded v2 artifacts)."""
    return tuple(
        Recipe(
            f"E{exposure}{suffix}", SWEEP_ALPHA, exposure,
            SWEEP_LEARNING_RATE, SWEEP_ROUNDS, optimizer_scope=scope,
        )
        for scope, suffix in (
            ("trunk_and_head", ""),
            ("last_block_and_head", "L"),
        )
        for exposure in SWEEP_EXPOSURE_PASSES
    )


def qualification_recipes() -> tuple[Recipe, ...]:
    """One-shot alpha comparison on the identical shared round-1 archive."""
    return (
        Recipe("QUAL-A0", 0.0, 1, SWEEP_LEARNING_RATE, 1),
        Recipe("QUAL-A005", 0.05, 1, SWEEP_LEARNING_RATE, 1),
    )


def _contract_payload(
    recipes: tuple[Recipe, ...],
    *,
    r0_sha256: str,
    sr_guard: float,
    timeout_guard: float,
    time_guard_seconds: float,
) -> dict:
    assert_declared_banks_static()
    if len(str(r0_sha256)) != 64:
        raise ValueError("the contract requires the full r0 SHA-256")
    if len({recipe.recipe_id for recipe in recipes}) != len(recipes):
        raise ValueError("recipe ids must be unique")
    return {
        "status": CONTRACT_STATUS,
        "version": VERSION,
        "r0_sha256": str(r0_sha256).lower(),
        "recipes": [asdict(recipe) for recipe in recipes],
        "qualification_recipes": [
            asdict(recipe) for recipe in qualification_recipes()
        ],
        "alpha_retention_rule": ALPHA_RETENTION_RULE,
        "banks": {
            stage: [dict(bank) for bank in banks]
            for stage, banks in DECLARED_EVAL_BANKS.items()
        },
        "ranking_key": RANKING_KEY,
        "shortlist_size": SHORTLIST_SIZE,
        "sr_guard": float(sr_guard),
        "timeout_guard": float(timeout_guard),
        "time_guard_seconds": float(time_guard_seconds),
        "per_gamma_cr_collapse": float(PER_GAMMA_CR_COLLAPSE),
        "per_gamma_validity_collapse": float(PER_GAMMA_VALIDITY_COLLAPSE),
        "locked_winner": None,
    }


def _write_contract_file(path: Path, contract: dict) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to redeclare an existing contract: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(contract, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def write_contract(
    path: str | Path,
    *,
    r0_sha256: str,
    sr_guard: float = SR_GUARD,
    timeout_guard: float = TIMEOUT_GUARD,
    time_guard_seconds: float = TIME_GUARD_SECONDS,
) -> dict:
    """Freeze recipes, banks, ranking, guards, and shortlist size before any run."""
    path = Path(path).resolve()
    contract = _contract_payload(
        declared_recipes(), r0_sha256=r0_sha256, sr_guard=sr_guard,
        timeout_guard=timeout_guard, time_guard_seconds=time_guard_seconds,
    )
    _write_contract_file(path, contract)
    return contract


def write_contract_v2(
    path: str | Path,
    *,
    r0_sha256: str,
    amendment_of: str | Path,
    reason: str,
    sr_guard: float = SR_GUARD,
    timeout_guard: float = TIMEOUT_GUARD,
    time_guard_seconds: float = TIME_GUARD_SECONDS,
) -> dict:
    """Amended contract: the 6-recipe surface-crossed sweep, guards unchanged.

    Legal only while no shortlist/confirmation bank has been read for the
    amended recipes; the amendment block records the superseded contract and
    the reason so the change is declared, never silent.
    """
    path = Path(path).resolve()
    amendment_of = Path(amendment_of).resolve()
    if not amendment_of.is_file():
        raise FileNotFoundError(
            f"the amended contract must reference the superseded one: "
            f"{amendment_of}"
        )
    superseded = json.loads(amendment_of.read_text())
    if superseded.get("status") not in {CONTRACT_STATUS, LOCK_STATUS}:
        raise ValueError("amendment_of is not a declared search contract")
    if not str(reason).strip():
        raise ValueError("an amendment requires a recorded reason")
    contract = _contract_payload(
        declared_recipes_v2(), r0_sha256=r0_sha256, sr_guard=sr_guard,
        timeout_guard=timeout_guard, time_guard_seconds=time_guard_seconds,
    )
    contract["amendment"] = {
        "supersedes": str(amendment_of),
        "supersedes_sha256": sha256_file(amendment_of),
        "reason": str(reason),
        "scope": (
            "recipes only: guards, banks, ranking key, and shortlist size "
            "are unchanged from the superseded contract"
        ),
        "declared_before_confirmation_read": True,
    }
    _write_contract_file(path, contract)
    return contract


def write_contract_v3(
    path: str | Path,
    *,
    r0_sha256: str,
    amendment_of: str | Path,
    reason: str,
    sr_guard: float = SR_GUARD,
    timeout_guard: float = TIMEOUT_GUARD,
    time_guard_seconds: float = TIME_GUARD_SECONDS,
) -> dict:
    """Re-amended contract: the 6-recipe two-surface sweep, guards unchanged.

    v3 both adds the minimal ``last_block_and_head`` arms and removes the v2
    ``last_two_blocks_and_head`` arms.  Legal only while no
    shortlist/confirmation bank has been read for the amended recipes; the
    amendment block records the superseded contract and the reason, extending
    the v1 -> v2 -> v3 chain so the change is declared, never silent.
    """
    path = Path(path).resolve()
    amendment_of = Path(amendment_of).resolve()
    if not amendment_of.is_file():
        raise FileNotFoundError(
            f"the amended contract must reference the superseded one: "
            f"{amendment_of}"
        )
    superseded = json.loads(amendment_of.read_text())
    if superseded.get("status") not in {CONTRACT_STATUS, LOCK_STATUS}:
        raise ValueError("amendment_of is not a declared search contract")
    if not str(reason).strip():
        raise ValueError("an amendment requires a recorded reason")
    contract = _contract_payload(
        declared_recipes_v3(), r0_sha256=r0_sha256, sr_guard=sr_guard,
        timeout_guard=timeout_guard, time_guard_seconds=time_guard_seconds,
    )
    contract["amendment"] = {
        "supersedes": str(amendment_of),
        "supersedes_sha256": sha256_file(amendment_of),
        "supersedes_amendment": superseded.get("amendment"),
        "reason": str(reason),
        "scope": (
            "recipes only: guards, banks, ranking key, and shortlist size "
            "are unchanged from the superseded contract"
        ),
        "declared_before_confirmation_read": True,
    }
    _write_contract_file(path, contract)
    return contract


def _metrics(entry: dict) -> dict:
    pooled = entry["pooled"]
    for field in ("SR", "CR", "timeout", "Validity"):
        if pooled.get(field) is None:
            raise ValueError(f"funnel result lacks pooled {field}")
    return pooled


def _per_gamma_collapse_reasons(
    contract: dict, per_gamma: dict, r0_per_gamma: dict,
) -> list[str]:
    reasons = []
    cr_margin = float(contract["per_gamma_cr_collapse"])
    validity_margin = float(contract["per_gamma_validity_collapse"])
    for gamma, r0_row in r0_per_gamma.items():
        row = per_gamma.get(gamma)
        if row is None:
            reasons.append(f"missing_gamma_{gamma}")
            continue
        if float(row["CR"]) > float(r0_row["CR"]) + cr_margin:
            reasons.append(f"gamma_{gamma}_cr_collapse")
        if float(row["Validity"]) < float(r0_row["Validity"]) - validity_margin:
            reasons.append(f"gamma_{gamma}_validity_collapse")
    return reasons


def select_shortlist(
    contract: dict,
    screen_results: list[dict],
    *,
    r0_ood: dict,
    r0_id: dict | None = None,
    r0_ood_per_gamma: dict | None = None,
) -> list[dict]:
    """Rank guarded M20-screen candidates by the declared r0-relative key.

    ``screen_results`` rows carry ``label``, ``checkpoint_sha256``, ``ood``
    pooled metrics, optionally ``id`` pooled metrics (when the ID screen ran)
    and ``ood_per_gamma`` per-gamma metric dicts from the screen-m20 stage.
    """
    if contract.get("status") not in {CONTRACT_STATUS, LOCK_STATUS}:
        raise ValueError("shortlisting requires the declared search contract")
    sr_guard = float(contract["sr_guard"])
    timeout_guard = float(contract["timeout_guard"])
    time_guard = float(contract["time_guard_seconds"])
    guarded = []
    ledger = []
    for row in screen_results:
        ood = _metrics({"pooled": row["ood"]})
        reasons = []
        if ood["SR"] < float(r0_ood["SR"]) - sr_guard:
            reasons.append("ood_sr_guard")
        if row.get("id") is not None:
            if r0_id is None:
                raise ValueError("ID screen results require the r0 ID baseline")
            if _metrics({"pooled": row["id"]})["SR"] < (
                float(r0_id["SR"]) - sr_guard
            ):
                reasons.append("id_sr_guard")
        if ood["timeout"] > float(r0_ood["timeout"]) + timeout_guard:
            reasons.append("timeout_guard")
        r0_time = r0_ood.get("successful_time")
        time = ood.get("successful_time")
        if r0_time is not None and time is not None and (
            float(time) > float(r0_time) + time_guard
        ):
            reasons.append("time_guard")
        if r0_ood_per_gamma is not None and row.get("ood_per_gamma") is not None:
            reasons.extend(_per_gamma_collapse_reasons(
                contract, row["ood_per_gamma"], r0_ood_per_gamma,
            ))
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
        (
            float("inf") if row["ood"].get("successful_time") is None
            else float(row["ood"]["successful_time"])
        ),
        str(row["label"]),
    ))
    shortlist = ordered[:int(contract["shortlist_size"])]
    return [
        {
            "label": row["label"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "ood": dict(row["ood"]),
            "id": (None if row.get("id") is None else dict(row["id"])),
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
    declare.add_argument("--sr-guard", type=float, default=SR_GUARD)
    declare.add_argument("--timeout-guard", type=float, default=TIMEOUT_GUARD)
    declare.add_argument(
        "--time-guard-seconds", type=float, default=TIME_GUARD_SECONDS,
    )
    declare_v2 = sub.add_parser(
        "declare-v2",
        help="write the amended 6-recipe (surface-crossed) search contract",
    )
    declare_v2.add_argument("--contract", required=True)
    declare_v2.add_argument("--r0-sha256", required=True)
    declare_v2.add_argument("--amendment-of", required=True)
    declare_v2.add_argument("--reason", required=True)
    declare_v3 = sub.add_parser(
        "declare-v3",
        help=(
            "write the re-amended 6-recipe contract (adds last_block_and_head "
            "arms, drops the v2 last_two_blocks_and_head arms)"
        ),
    )
    declare_v3.add_argument("--contract", required=True)
    declare_v3.add_argument("--r0-sha256", required=True)
    declare_v3.add_argument("--amendment-of", required=True)
    declare_v3.add_argument("--reason", required=True)
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
            time_guard_seconds=float(args.time_guard_seconds),
        )
        print(json.dumps({
            "status": contract["status"],
            "recipes": len(contract["recipes"]),
        }, sort_keys=True))
    elif args.command in {"declare-v2", "declare-v3"}:
        writer = (
            write_contract_v2 if args.command == "declare-v2"
            else write_contract_v3
        )
        contract = writer(
            args.contract, r0_sha256=args.r0_sha256,
            amendment_of=args.amendment_of, reason=args.reason,
        )
        print(json.dumps({
            "status": contract["status"],
            "recipes": len(contract["recipes"]),
            "amendment": contract["amendment"]["supersedes"],
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
