"""Round orchestrator for SFM2 positive-minus-alpha-negative expansion.

Each arm runs *cumulatively* through its declared rounds (r1..r5 in the
sweep): every round gathers an enriched acquisition archive with the current
sampling policy (round 1 samples from r0), applies one declared update of E
complete reshuffled passes with a persistent Adam optimizer, saves a strict
raw-evaluable checkpoint, and saves a complete resume state (model, optimizer
momentum, RNG, GP/reference support, archive ledger, recipe, round index) so
any saved round can later resume exactly to r6+ via ``--resume-from``.

The acquisition reference representation, RBF calibration support, and every
condition encoder stay pinned to the canonical r0 checkpoint for every round:
there is no round-to-round GP support accumulation, no buffer rule, and no
historical replay role.

The first-round archive may be shared between qualification arms (gather
once, train per alpha) through ``--reuse-archive``; the archive provenance is
then verified against the declared r0 digest fail-closed.

A ``QUALIFICATION_AUDIT.json`` is always written at the end of a recipe run:
D+/D- yield, NVP/retry statistics, exact replay exposure, CFM loss curves,
gradient norms/clipping, parameter drift, and per-gamma controller failures —
the evidence base for a recorded correction and rerun after a failed
qualification.  There is no silent retry logic anywhere in this module.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_ball_launch as BASE
from sfm_hp100_ball_core.expansion import mean_pairwise_lengthscale
import sfm_hp100_exhaustive_hybrid as HYBRID
import sfm_hp100_predictive_execution as PRED

import sfm_hp100_expansion_archive as ARCH
from sfm_hp100_expansion_funnel import acquisition_scenario_start
from sfm_hp100_expansion_status import Heartbeat
import sfm_hp100_expansion_update as UPD


VERSION = "sfm_hp100_expansion_round_v2"
ROUND_STATUS = "SFM2_EXPANSION_ROUND_COMPLETE"
RECIPE_STATUS = "SFM2_EXPANSION_RECIPE_COMPLETE"
RESUME_STATUS = "SFM2_EXPANSION_RESUME_STATE"
HEALTH_STATUS = "SFM2_EXPANSION_ROUND_HEALTH"
FALLBACK_STATUS = "SFM2_EXPANSION_FALLBACK_DIAGNOSIS"

# ---- Declared acquisition health gate ------------------------------------
# Evaluated after every round's archive gather and BEFORE any training, so a
# cumulative arm never trains on a degenerate archive produced by a degraded
# previous-round checkpoint.  A criterion whose statistic is unavailable
# (e.g. a reused archive without block summaries) is recorded as
# ``evaluable: false`` and does not fail the gate.
#
# Absolute per-gamma D+ floor: below this the CFM update mean for that gamma
# is dominated by a handful of windows.
HEALTH_MIN_DPLUS_PER_GAMMA = 5
# Relative floor against the same arm's round-1 per-gamma yield: a collapse
# to under a quarter of the initial yield flags a degraded sampler even when
# the absolute floor still holds.
HEALTH_MIN_DPLUS_BASELINE_FRACTION = 0.25
# Lineages terminating NVP after full retry exhaustion.  Recorded correction
# 001 (2026-08-14): the a-priori 0.50 sat below the measured r0 round-1
# baseline (33/56 = 0.589 NVP lineages), so every arm halted at round 2 while
# every other statistic (per-gamma D+ yield, retry pressure, drift, CFM
# probes, raw M20) stayed healthy.  Recalibrated to that measured baseline
# plus a 0.15 absolute margin; genuine collapse (three quarters of lineages
# starving) still trips the gate.
HEALTH_MAX_NVP_LINEAGE_FRACTION = 0.75
# Replan contexts whose final B block stayed all-negative after retries.
HEALTH_MAX_ZERO_POSITIVE_CONTEXT_FRACTION = 0.50
# Mean acquisition attempts per replan context (retry pressure).
HEALTH_MAX_MEAN_ATTEMPTS = 16.0


def _mean(values):
    cleaned = [float(value) for value in values if value is not None]
    return float(np.mean(cleaned)) if cleaned else None


def acquisition_statistics(
    rows: list[dict], block_summaries: list[dict],
) -> dict:
    """Acquisition/uncertainty-tilting statistics for one round's archive.

    Everything here is controller/training-side accounting; it never counts
    as evaluation.
    """
    positives = [row for row in rows if row["role"] == "positive"]
    gammas = sorted({float(row["gamma"]) for row in rows})
    dplus_per_gamma = {
        f"{gamma:g}": sum(float(row["gamma"]) == gamma for row in positives)
        for gamma in gammas
    }
    outcome_counts: dict[str, int] = {}
    contexts = 0
    attempts_total = 0
    retried_fractions = []
    pooled_blocks = []
    for block in block_summaries:
        summary = block.get("summary", block)
        for outcome, count in summary.get("outcome_counts", {}).items():
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + int(count)
        contexts += int(summary.get("contexts", 0))
        pooled = summary.get("pooled", {})
        attempts_total += int(pooled.get("attempts", 0))
        retried_fractions.append(summary.get("retried_context_fraction"))
        pooled_blocks.append(pooled)
    lineages = sum(outcome_counts.values())
    nvp = int(outcome_counts.get("nvp", 0))
    attempt_histogram: dict[str, int] = {}
    for row in rows:
        key = str(int(row["attempt"]))
        attempt_histogram[key] = attempt_histogram.get(key, 0) + 1
    first16 = _mean([row.get("positive_first16") for row in rows])
    b32 = _mean([row.get("positive_B32") for row in rows])
    return {
        "dplus_total": len(positives),
        "dplus_per_gamma": dplus_per_gamma,
        "sample_counts": ARCH._sample_counts(rows),
        "lineage_outcomes": outcome_counts,
        "nvp_lineage_fraction": (nvp / lineages if lineages else None),
        "contexts": (contexts if block_summaries else None),
        "zero_positive_context_fraction": (
            nvp / contexts if contexts else None
        ),
        "mean_attempts": (
            attempts_total / contexts if contexts else None
        ),
        "retried_context_fraction": _mean(retried_fractions),
        "attempt_histogram": attempt_histogram,
        "mean_base_std": _mean([row.get("base_std") for row in rows]),
        "mean_beta": _mean([row.get("beta") for row in rows]),
        "mean_marginal_ess_over_k": _mean(
            [row.get("marginal_ESS_over_K") for row in rows]
        ),
        "selected_sigma": {
            "mean": _mean([row.get("selected_sigma") for row in rows]),
            "min": (
                min(float(row["selected_sigma"]) for row in rows)
                if rows and rows[0].get("selected_sigma") is not None else None
            ),
            "max": (
                max(float(row["selected_sigma"]) for row in rows)
                if rows and rows[0].get("selected_sigma") is not None else None
            ),
        },
        "mean_positive_first16": first16,
        "mean_positive_B32": b32,
        # Uncertainty-tilting effectiveness: exact-positive density of the
        # acquired B=32 block relative to the first-16 proxy density.
        "uncertainty_tilt_ratio": (
            None if not first16 or b32 is None
            else (b32 / 32.0) / (first16 / 16.0)
        ),
        # Selector-vs-max-margin disagreement and the remaining pooled
        # controller statistics, verbatim per gathered block.
        "block_pooled_summaries": pooled_blocks,
    }


def evaluate_health_gate(
    stats: dict,
    *,
    gammas: tuple[float, ...],
    baseline_dplus_per_gamma: dict | None = None,
) -> dict:
    """Apply the declared thresholds; unavailable statistics never fail."""
    criteria = []

    def criterion(name, observed, threshold, passed, evaluable=True):
        criteria.append({
            "criterion": name, "observed": observed, "threshold": threshold,
            "evaluable": bool(evaluable),
            "passed": bool(passed) if evaluable else True,
        })

    for gamma in gammas:
        key = f"{float(gamma):g}"
        observed = int(stats["dplus_per_gamma"].get(key, 0))
        criterion(
            f"min_dplus_gamma_{key}", observed, HEALTH_MIN_DPLUS_PER_GAMMA,
            observed >= HEALTH_MIN_DPLUS_PER_GAMMA,
        )
        if baseline_dplus_per_gamma is not None:
            baseline = baseline_dplus_per_gamma.get(key)
            floor = (
                None if baseline is None
                else HEALTH_MIN_DPLUS_BASELINE_FRACTION * float(baseline)
            )
            criterion(
                f"min_dplus_baseline_fraction_gamma_{key}", observed, floor,
                floor is None or observed >= floor,
                evaluable=floor is not None,
            )
    for name, observed, threshold in (
        (
            "max_nvp_lineage_fraction",
            stats.get("nvp_lineage_fraction"),
            HEALTH_MAX_NVP_LINEAGE_FRACTION,
        ),
        (
            "max_zero_positive_context_fraction",
            stats.get("zero_positive_context_fraction"),
            HEALTH_MAX_ZERO_POSITIVE_CONTEXT_FRACTION,
        ),
        (
            "max_mean_attempts",
            stats.get("mean_attempts"),
            HEALTH_MAX_MEAN_ATTEMPTS,
        ),
    ):
        criterion(
            name, observed, threshold,
            observed is None or float(observed) <= threshold,
            evaluable=observed is not None,
        )
    return {
        "status": HEALTH_STATUS,
        "version": VERSION,
        "passed": all(row["passed"] for row in criteria),
        "criteria": criteria,
        "stats": stats,
        "thresholds": {
            "min_dplus_per_gamma": HEALTH_MIN_DPLUS_PER_GAMMA,
            "min_dplus_baseline_fraction": HEALTH_MIN_DPLUS_BASELINE_FRACTION,
            "max_nvp_lineage_fraction": HEALTH_MAX_NVP_LINEAGE_FRACTION,
            "max_zero_positive_context_fraction":
                HEALTH_MAX_ZERO_POSITIVE_CONTEXT_FRACTION,
            "max_mean_attempts": HEALTH_MAX_MEAN_ATTEMPTS,
        },
    }


def _trainable_state_drift(adapter, reference_state: dict) -> float:
    """Relative L2 drift of the trunk/head surface against a stored state."""
    numerator = 0.0
    denominator = 0.0
    for key, value in adapter.policy.state_dict().items():
        if not (key.startswith("trunk.") or key.startswith("head.")):
            continue
        other = reference_state[key].to(value.device, value.dtype)
        numerator += float((value - other).square().sum())
        denominator += float(other.square().sum())
    return (numerator ** 0.5) / max(denominator ** 0.5, 1.0e-24)


def _archive_cfm_probe(
    adapter, rows: list[dict], *, seed: int, label: str,
) -> dict | None:
    """Seeded no-grad CFM loss of the current policy over archived D+ rows."""
    positives = [row for row in rows if row["role"] == "positive"]
    if not positives:
        return None
    device = next(adapter.parameters()).device
    contexts, candidates = HYBRID._stack_rows(positives, device)
    HYBRID._set_step_seed(seed, device)
    with torch.no_grad():
        losses = adapter.cfm_loss(contexts, candidates, reduction="none")
    return {
        "label": label,
        "rows": len(positives),
        "cfm_loss_mean": float(losses.mean()),
        "cfm_loss_max": float(losses.max()),
    }


def fallback_diagnosis(
    *,
    adapter,
    reference,
    round_index: int,
    health: dict,
    rows: list[dict],
    archive_ledger: list[dict],
    parent_checkpoint_path: Path | None,
    probe_seed: int,
) -> dict:
    """Automatic discriminating evidence after a failed acquisition gate.

    Gathers the drift/likelihood/typology evidence separating "the updated
    policy broke" from "acquisition broke"; it never retrains, never retries,
    and leaves the arm halted.  Resuming is a recorded evidence-based
    correction under the qualification-failure protocol.
    """
    probes = []
    for record in archive_ledger[:-1]:
        previous = ARCH.load_archive(Path(record["path"]))
        probe = _archive_cfm_probe(
            adapter, previous["rows"], seed=probe_seed,
            label=f"round_{record['round']}_dplus_under_current_policy",
        )
        if probe is not None:
            probes.append(probe)
    current_probe = _archive_cfm_probe(
        adapter, rows, seed=probe_seed,
        label=f"round_{round_index}_dplus_under_current_policy",
    )
    if current_probe is not None:
        probes.append(current_probe)

    drift_vs_r0 = _trainable_state_drift(
        adapter, reference.policy.state_dict(),
    )
    drift_vs_parent = None
    if parent_checkpoint_path is not None and parent_checkpoint_path.is_file():
        parent_state = torch.load(
            parent_checkpoint_path, map_location="cpu", weights_only=False,
        )["state_dict"]
        drift_vs_parent = _trainable_state_drift(adapter, parent_state)

    per_gamma = {}
    for row in rows:
        bucket = per_gamma.setdefault(f"{float(row['gamma']):g}", {
            "rows": 0, "positive_first16": [], "positive_B32": [],
            "base_std": [], "attempts_used": [],
            "negative_reasons": {},
        })
        bucket["rows"] += 1
        bucket["positive_first16"].append(row.get("positive_first16"))
        bucket["positive_B32"].append(row.get("positive_B32"))
        bucket["base_std"].append(row.get("base_std"))
        bucket["attempts_used"].append(row.get("attempts_used"))
        reason = row.get("negative_reason")
        if reason is not None:
            bucket["negative_reasons"][reason] = (
                bucket["negative_reasons"].get(reason, 0) + 1
            )
    for bucket in per_gamma.values():
        for key in (
            "positive_first16", "positive_B32", "base_std", "attempts_used",
        ):
            bucket[key] = _mean(bucket[key])

    counts = ARCH._sample_counts(rows)
    return {
        "status": FALLBACK_STATUS,
        "version": VERSION,
        "halted_round": int(round_index),
        "health": health,
        "policy_drift": {
            "relative_trainable_drift_vs_r0": drift_vs_r0,
            "relative_trainable_drift_vs_previous_round": drift_vs_parent,
        },
        "dplus_cfm_probes": probes,
        "per_gamma_positive_vanishing": per_gamma,
        "failure_typology": {
            "exact_negative_nvp_after_retries": counts["all_negative_nvp"],
            "realized_collision": counts["realized_collision"],
            "realized_oob": counts["realized_oob"],
            "note": (
                "all_negative_nvp counts unsafe generation resolved exactly "
                "negative after the full retry schedule; realized_* count "
                "executed y=1 rows whose live transition failed"
            ),
        },
        "recommended_followup": {
            "action": (
                "schedule a screen-m20 raw evaluation of the current "
                "checkpoint to separate 'policy broke' (raw metrics degrade) "
                "from 'acquisition broke' (raw metrics hold while the "
                "controller starves)"
            ),
            "checkpoint": (
                None if parent_checkpoint_path is None
                else str(parent_checkpoint_path)
            ),
        },
        "protocol": (
            "arm halted at this round; earlier checkpoints remain evaluable; "
            "no automatic retraining or silent recovery — rerun only as a "
            "recorded evidence-based correction"
        ),
    }


def split_roles(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    positives = [row for row in rows if row["role"] == "positive"]
    negatives = [row for row in rows if row["role"] == "negative"]
    if len(positives) + len(negatives) != len(rows):
        raise ValueError("archive rows carry an undeclared role")
    return positives, negatives


def _load_reused_archive(path: Path, *, expected_checkpoint_sha: str) -> dict:
    payload = ARCH.load_archive(path)
    provenance = payload["provenance"]
    if provenance["checkpoint_sha256"] != str(expected_checkpoint_sha).lower():
        raise RuntimeError(
            "reused archive was not gathered from the declared sampling checkpoint"
        )
    if int(payload["config"]["round_index"]) != 1:
        raise RuntimeError("only a round-1 archive may be reused across arms")
    return payload


def _rng_snapshot() -> dict:
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "numpy": np.random.get_state(),
    }


def _rng_restore(snapshot: dict) -> None:
    torch.set_rng_state(snapshot["torch_cpu"].cpu().to(torch.uint8))
    if snapshot.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([
            state.cpu().to(torch.uint8) for state in snapshot["torch_cuda"]
        ])
    np.random.set_state(snapshot["numpy"])


def resume_payload(
    *,
    adapter: PORT.HP100ExpansionPolicy,
    optimizer: torch.optim.Adam,
    round_index: int,
    parent_checkpoint_sha256: str,
    r0_checkpoint_sha256: str,
    recipe_provenance: dict,
    archive_ledger: list[dict],
    gp_state: dict,
    baseline_dplus_per_gamma: dict | None = None,
    optimizer_scope: str = UPD.OPTIMIZER_SCOPE,
) -> dict:
    """Everything needed to continue this arm bitwise at round_index + 1."""
    return {
        "status": RESUME_STATUS,
        "version": VERSION,
        "round_index": int(round_index),
        "baseline_dplus_per_gamma": baseline_dplus_per_gamma,
        "parent_checkpoint_sha256": str(parent_checkpoint_sha256),
        "r0_checkpoint_sha256": str(r0_checkpoint_sha256),
        "recipe_provenance": recipe_provenance,
        "archive_ledger": archive_ledger,
        "model_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in adapter.policy.state_dict().items()
        },
        "model_config": adapter.policy.config(),
        "optimizer_state_dict": optimizer.state_dict(),
        "encoder_state_sha256": UPD.frozen_surface_sha256(
            adapter, optimizer_scope,
        ),
        "rng": _rng_snapshot(),
        "gp_state": gp_state,
    }


def load_resume(path: str | Path, *, expected_r0_sha256: str) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("status") != RESUME_STATUS:
        raise ValueError(f"not an SFM2 expansion resume state: {path}")
    if payload["r0_checkpoint_sha256"] != str(expected_r0_sha256).lower():
        raise RuntimeError("resume state was not produced under the declared r0")
    return payload


def run_recipe(args) -> dict:
    output = Path(args.output).resolve()
    resume_state = None
    if args.resume_from is not None:
        if not output.is_dir() or not (output / "RECIPE_DECLARED.json").is_file():
            raise FileNotFoundError(
                "resume requires the original recipe output directory"
            )
    elif output.exists():
        raise FileExistsError(f"refusing existing recipe output: {output}")
    rounds = int(args.rounds)
    if rounds < 1:
        raise ValueError("declare at least one expansion round")
    gpu_contract = BASE._gpu_contract(args.device, int(args.physical_gpu))
    r0_sha = PRED.sha256_file(args.checkpoint)
    if r0_sha != str(args.expected_checkpoint_sha256).lower():
        raise RuntimeError("r0 checkpoint SHA256 mismatch")
    policy, payload = GPS.load_sfm_hp100_policy(args.checkpoint, device=args.device)
    adapter = PORT.HP100ExpansionPolicy(policy).eval()
    reference_policy, _ = GPS.load_sfm_hp100_policy(
        args.checkpoint, device=args.device,
    )
    reference = PORT.HP100ExpansionPolicy(reference_policy).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    update_config = UPD.UpdateConfig(
        alpha=float(args.alpha),
        learning_rate=float(args.learning_rate),
        batch_size=int(args.batch_size),
        exposure_passes=int(args.exposure_passes),
        grad_clip_norm=float(args.grad_clip_norm),
        max_relative_parameter_drift=float(args.max_relative_parameter_drift),
        train_mode=str(args.train_mode),
        optimizer_scope=str(args.optimizer_scope),
        seed=int(args.update_seed),
    )
    update_config.validate()
    # The asserted-frozen digest is scope-aware: under the reduced scope it
    # also pins trunk.inp bitwise to the r0 state.
    encoder_r0 = UPD.frozen_surface_sha256(
        reference, update_config.optimizer_scope,
    )

    gammas = HYBRID._parse_gammas(args.gammas)
    device = next(reference.parameters()).device
    if args.resume_from is not None:
        resume_state = load_resume(
            args.resume_from, expected_r0_sha256=r0_sha,
        )
        saved_config = dict(resume_state["recipe_provenance"]["update_config"])
        # Resume states written before the scope field existed were implicitly
        # the full trunk_and_head surface; a declared scope mismatch is still
        # refused below.  Recipe identity is checked before the digest so a
        # scope change reports as a recipe mismatch, not encoder drift.
        saved_config.setdefault("optimizer_scope", UPD.OPTIMIZER_SCOPE)
        if saved_config != asdict(update_config):
            raise RuntimeError(
                "resume requires the identical declared update recipe"
            )
        if resume_state["encoder_state_sha256"] != encoder_r0:
            raise RuntimeError("resume state encoders drifted from the r0 digest")
        gp_state = resume_state["gp_state"]
        features = gp_state["features"].to(device)
        calibration = gp_state["calibration"]
        lengthscale = float(gp_state["lengthscale"])
    else:
        features, calibration = BASE.calibration_features(
            reference, dataset_root=args.pretrain_dataset_root,
            expected_manifest_sha256=(
                args.expected_pretrain_dataset_manifest_sha256
            ),
            count=50, seed=int(args.seed), base_std=1.0,
            paired_noised_representation=True,
        )
        lengthscale = mean_pairwise_lengthscale(features)
        gp_state = {
            "rule": (
                "frozen 50-row balanced pretrained r0 preflight for every "
                "round; no round-to-round accumulation"
            ),
            "features": features.detach().cpu().clone(),
            "calibration": calibration,
            "lengthscale": float(lengthscale),
        }
    support = HYBRID._calibration_support_by_gamma(
        features, calibration, gammas, device=device,
    )

    heartbeat = Heartbeat(
        args.status_json, interval_seconds=float(args.heartbeat_seconds),
    )
    recipe_provenance = {
        "recipe_id": str(args.recipe_id),
        "r0_checkpoint": str(Path(args.checkpoint).resolve()),
        "r0_checkpoint_sha256": r0_sha,
        "r0_scientific_status": payload.get("scientific_status"),
        "gpu": gpu_contract,
        "calibration": {**calibration, "lengthscale": float(lengthscale)},
        "update_config": asdict(update_config),
        "rounds": rounds,
        "gp_support_rule": gp_state["rule"] if "rule" in gp_state else (
            "frozen 50-row balanced pretrained r0 preflight for every round"
        ),
        "source_sha256": PRED.sha256_file(__file__),
    }

    trainable_parameters, _ = UPD.configure_trainable(
        adapter, update_config.optimizer_scope,
    )
    optimizer = torch.optim.Adam(
        trainable_parameters, lr=update_config.learning_rate,
    )
    archive_ledger: list[dict] = []
    round_markers: list[dict] = []
    checkpoints: list[dict] = []
    health_markers: list[dict] = []
    baseline_dplus_per_gamma: dict | None = None
    halted_at_round: int | None = None
    first_round = 1
    parent_sha = r0_sha
    if resume_state is not None:
        adapter.policy.load_state_dict(resume_state["model_state_dict"])
        optimizer.load_state_dict(resume_state["optimizer_state_dict"])
        _rng_restore(resume_state["rng"])
        archive_ledger = list(resume_state["archive_ledger"])
        baseline_dplus_per_gamma = resume_state.get("baseline_dplus_per_gamma")
        parent_sha = str(resume_state["parent_checkpoint_sha256"])
        first_round = int(resume_state["round_index"]) + 1
        if first_round > rounds:
            raise ValueError(
                "resume state already completed the declared round count"
            )
        for marker_path in sorted(output.glob("ROUND_*_COMPLETE.json")):
            round_markers.append(json.loads(marker_path.read_text()))
        for health_path in sorted(output.glob("ROUND_*_HEALTH.json")):
            health_markers.append(json.loads(health_path.read_text()))
        checkpoints = [
            {
                "round": marker["round"],
                "sha256": marker["checkpoint"]["sha256"],
            }
            for marker in round_markers
        ]
    else:
        output.mkdir(parents=True)
        HYBRID._write_json(output / "RECIPE_DECLARED.json", recipe_provenance)

    for round_index in range(first_round, rounds + 1):
        round_dir = output / f"round_{round_index:02d}"
        current_sha = PRED._model_state_sha256(adapter)
        archive_config = ARCH.ArchiveConfig(
            gammas=gammas,
            lineages_per_gamma=int(args.lineages_per_gamma),
            blocks=int(args.blocks),
            max_steps=int(args.max_steps),
            max_attempts=int(args.max_attempts),
            ess_target=float(args.ess_target),
            seed=int(args.seed),
            round_index=round_index,
            scene_profile=str(args.scene_profile),
        )
        archive_config.validate()
        if round_index == 1 and args.reuse_archive is not None:
            archive_payload = _load_reused_archive(
                Path(args.reuse_archive), expected_checkpoint_sha=r0_sha,
            )
            rows = archive_payload["rows"]
            archive_record = {
                "reused": True,
                "round": round_index,
                "path": str(Path(args.reuse_archive).resolve()),
                "sha256": PRED.sha256_file(args.reuse_archive),
                "sample_counts": ARCH._sample_counts(rows),
            }
        else:
            scenario_start = acquisition_scenario_start(round_index)
            task = PORT.SFMHP100ExpansionTask(
                scene_profile=archive_config.scene_profile,
                scenario_start=scenario_start,
            ).attach_context_encoder(adapter.policy)
            for parameter in adapter.parameters():
                parameter.requires_grad_(False)
            round_dir.mkdir(parents=True)
            provenance = {
                "checkpoint_sha256": (
                    r0_sha if round_index == 1 else parent_sha
                ),
                "policy_state_sha256": current_sha,
                "reference_checkpoint_sha256": r0_sha,
                "source_sha256": PRED.sha256_file(ARCH.__file__),
                "scenario_start": int(scenario_start),
            }
            with HYBRID._OrderedSidecarVerifier(
                task, int(args.verifier_workers),
            ) as verifier:
                gathered = ARCH.gather_round(
                    adapter, reference, task, config=archive_config,
                    lengthscale=float(lengthscale), support_by_gamma=support,
                    verifier=verifier, provenance=provenance,
                    trace_dir=round_dir, heartbeat=heartbeat,
                )
            rows = gathered["rows"]
            ARCH.assert_bank_disjoint(rows)
            archive_path = round_dir / f"expansion_archive_r{round_index}.pt"
            HYBRID._torch_save(archive_path, {
                "status": ARCH.ARCHIVE_STATUS,
                "version": ARCH.VERSION,
                "semantics": ARCH.SEMANTICS,
                "config": asdict(archive_config),
                "provenance": provenance,
                "rows": rows,
            })
            archive_record = {
                "reused": False,
                "round": round_index,
                "path": str(archive_path),
                "sha256": PRED.sha256_file(archive_path),
                "sample_counts": gathered["sample_counts"],
                "block_summaries": gathered["block_summaries"],
                "trace_shards": gathered["trace_shards"],
            }
        archive_ledger.append(archive_record)

        # Acquisition health gate: declared, evaluated before any training.
        stats = acquisition_statistics(
            rows, archive_record.get("block_summaries", []),
        )
        health = evaluate_health_gate(
            stats, gammas=tuple(gammas),
            baseline_dplus_per_gamma=baseline_dplus_per_gamma,
        )
        health["round"] = round_index
        health["recipe_id"] = str(args.recipe_id)
        HYBRID._write_json(
            output / f"ROUND_{round_index}_HEALTH.json", health,
        )
        health_markers.append(health)
        if baseline_dplus_per_gamma is None:
            baseline_dplus_per_gamma = dict(stats["dplus_per_gamma"])
        if not health["passed"]:
            parent_path = (
                output / f"checkpoint_r{round_index - 1}.pt"
                if round_index > 1 else Path(args.checkpoint)
            )
            diagnosis = fallback_diagnosis(
                adapter=adapter, reference=reference,
                round_index=round_index, health=health, rows=rows,
                archive_ledger=archive_ledger,
                parent_checkpoint_path=parent_path,
                probe_seed=int(args.update_seed),
            )
            HYBRID._write_json(output / "FALLBACK_DIAGNOSIS.json", diagnosis)
            halted_at_round = round_index
            heartbeat.beat(
                status="halted", phase="health_gate", round=round_index,
            )
            break

        positives, negatives = split_roles(rows)
        heartbeat.beat(
            status="running", phase="update", round=round_index,
            samples=ARCH._sample_counts(rows),
        )
        update_metrics = UPD.expansion_update(
            adapter, positives, negatives, update_config,
            round_index=round_index, optimizer=optimizer,
        )
        if update_metrics["encoder_state_sha256_after"] != encoder_r0:
            raise RuntimeError("condition encoders drifted from the r0 digest")
        checkpoint_path = output / f"checkpoint_r{round_index}.pt"
        HYBRID._torch_save(checkpoint_path, UPD.checkpoint_payload(
            adapter,
            parent_checkpoint_sha256=parent_sha,
            pretrained_checkpoint_sha256=r0_sha,
            round_index=round_index,
            alpha=update_config.alpha,
            exposure_passes=update_config.exposure_passes,
            optimizer_scope=update_config.optimizer_scope,
        ))
        checkpoint_sha = PRED.sha256_file(checkpoint_path)
        resume_path = output / f"resume_r{round_index}.pt"
        HYBRID._torch_save(resume_path, resume_payload(
            adapter=adapter, optimizer=optimizer, round_index=round_index,
            parent_checkpoint_sha256=checkpoint_sha,
            r0_checkpoint_sha256=r0_sha,
            recipe_provenance=recipe_provenance,
            archive_ledger=archive_ledger, gp_state=gp_state,
            baseline_dplus_per_gamma=baseline_dplus_per_gamma,
            optimizer_scope=update_config.optimizer_scope,
        ))
        marker = {
            "status": ROUND_STATUS,
            "version": VERSION,
            "recipe_id": str(args.recipe_id),
            "round": round_index,
            "archive": archive_record,
            "acquisition_statistics": stats,
            "health": {
                "passed": health["passed"],
                "path": str(output / f"ROUND_{round_index}_HEALTH.json"),
            },
            "update": update_metrics,
            "checkpoint": {
                "path": str(checkpoint_path), "sha256": checkpoint_sha,
                "policy_state_sha256": PRED._model_state_sha256(adapter),
            },
            "resume_state": {
                "path": str(resume_path),
                "sha256": PRED.sha256_file(resume_path),
            },
            "parent_checkpoint_sha256": parent_sha,
        }
        HYBRID._write_json(output / f"ROUND_{round_index}_COMPLETE.json", marker)
        round_markers.append(marker)
        checkpoints.append({"round": round_index, "sha256": checkpoint_sha})
        parent_sha = checkpoint_sha
        if not update_metrics["accepted"]:
            break

    final = {
        "status": RECIPE_STATUS,
        "version": VERSION,
        "provenance": recipe_provenance,
        "resumed_from": (
            None if args.resume_from is None
            else str(Path(args.resume_from).resolve())
        ),
        "rounds_completed": len(round_markers),
        "rounds_declared": rounds,
        "checkpoints": checkpoints,
        "halted_at_round": halted_at_round,
        "health_gate": [
            {
                "round": health["round"], "passed": health["passed"],
                "failed_criteria": [
                    row["criterion"] for row in health["criteria"]
                    if not row["passed"]
                ],
            }
            for health in health_markers
        ],
        # D+ yield trend across rounds so degradation is visible at a glance.
        "dplus_yield_trend": [
            {
                "round": health["round"],
                "dplus_total": health["stats"]["dplus_total"],
                "dplus_per_gamma": health["stats"]["dplus_per_gamma"],
                "nvp_lineage_fraction":
                    health["stats"]["nvp_lineage_fraction"],
                "uncertainty_tilt_ratio":
                    health["stats"]["uncertainty_tilt_ratio"],
            }
            for health in health_markers
        ],
        "accepted": (
            halted_at_round is None
            and bool(round_markers)
            and all(
                marker["update"]["accepted"] for marker in round_markers
            )
        ),
    }
    HYBRID._write_json(output / "RECIPE_COMPLETE.json", final)
    HYBRID._write_json(
        output / "QUALIFICATION_AUDIT.json",
        qualification_audit(round_markers, recipe_provenance),
    )
    heartbeat.beat(
        status="complete", phase="recipe", rounds=len(round_markers),
    )
    return final


def qualification_audit(round_markers: list[dict], provenance: dict) -> dict:
    """The declared evidence base after any (possibly failed) recipe run.

    Captures D+/D- yield, NVP/retry behaviour, exact replay exposure, CFM
    loss curves, gradient norms/clipping, parameter drift, and per-gamma
    controller failures.  A failed qualification stops the launch; the
    correction that follows must cite this artifact and be recorded — the
    audit itself never retries anything.
    """
    per_round = []
    for marker in round_markers:
        update = marker["update"]
        archive = marker["archive"]
        blocks = archive.get("block_summaries", [])
        per_gamma_controller = {}
        retry = {"retried_context_fraction": [], "terminal_nvp": 0}
        for block in blocks:
            summary = block.get("summary", {})
            retry["retried_context_fraction"].append(
                summary.get("retried_context_fraction")
            )
            outcomes = summary.get("outcome_counts", {})
            retry["terminal_nvp"] += int(outcomes.get("nvp", 0))
            for gamma, row in summary.get("per_gamma", {}).items():
                per_gamma_controller.setdefault(gamma, []).append(row)
        per_round.append({
            "round": marker["round"],
            "sample_counts": archive["sample_counts"],
            "negative_counts_by_reason": update["negative_counts_by_reason"],
            "retry": retry,
            "per_gamma_controller_training_only": per_gamma_controller,
            "exposure": {
                key: update[key]
                for key in (
                    "exposure_passes_declared", "exposure_passes_completed",
                    "unique_positive_samples", "unique_negative_samples",
                    "positive_exposures", "negative_exposures",
                    "duplicate_exposures", "in_archive_duplicate_rows",
                    "adam_steps",
                )
            },
            "loss": {
                key: update[key]
                for key in (
                    "positive_loss_mean", "negative_loss_mean",
                    "objective_mean", "per_pass_positive_loss",
                )
            },
            "gradient": {
                key: update[key]
                for key in (
                    "grad_norm_pre_clip", "grad_norm_post_clip",
                    "per_pass_grad_norm", "clipped_fraction",
                    "positive_grad_norm", "negative_grad_norm", "grad_cosine",
                )
            },
            "drift": {
                "relative_parameter_drift": update["relative_parameter_drift"],
                "drift_gate": update["drift_gate"],
                "finite": update["finite"],
                "abort_reason": update["abort_reason"],
                "accepted": update["accepted"],
            },
        })
    return {
        "status": "SFM2_EXPANSION_QUALIFICATION_AUDIT",
        "version": VERSION,
        "recipe_id": provenance["recipe_id"],
        "update_config": provenance["update_config"],
        "rounds": per_round,
        "note": (
            "controller outcomes are acquisition/training statistics only "
            "and never count as evaluation"
        ),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--checkpoint", required=True)
    value.add_argument("--expected-checkpoint-sha256", required=True)
    value.add_argument("--pretrain-dataset-root", required=True)
    value.add_argument("--expected-pretrain-dataset-manifest-sha256", required=True)
    value.add_argument("--output", required=True)
    value.add_argument("--recipe-id", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--physical-gpu", type=int, default=3)
    value.add_argument("--verifier-workers", type=int, default=32)
    value.add_argument("--rounds", type=int, default=5)
    value.add_argument("--alpha", type=float, default=0.0)
    value.add_argument("--learning-rate", type=float, default=1.0e-5)
    value.add_argument("--batch-size", type=int, default=64)
    value.add_argument("--exposure-passes", type=int, default=1)
    value.add_argument("--grad-clip-norm", type=float, default=1.0)
    value.add_argument(
        "--max-relative-parameter-drift", type=float, default=0.25,
    )
    value.add_argument("--train-mode", default="eval", choices=("eval", "train"))
    value.add_argument(
        "--optimizer-scope", default=UPD.OPTIMIZER_SCOPE,
        choices=(UPD.OPTIMIZER_SCOPE, UPD.REDUCED_OPTIMIZER_SCOPE),
    )
    value.add_argument("--update-seed", type=int, default=2)
    value.add_argument("--scene-profile", default="double_density_velocity_ood")
    value.add_argument("--gammas", default="0.1,0.2,0.3,0.4,0.5,0.7,1.0")
    value.add_argument("--lineages-per-gamma", type=int, default=8)
    value.add_argument("--blocks", type=int, default=1)
    value.add_argument("--max-steps", type=int, default=180)
    value.add_argument("--max-attempts", type=int, default=32)
    value.add_argument("--ess-target", type=float, default=0.1)
    value.add_argument("--seed", type=int, default=41)
    value.add_argument("--reuse-archive", default=None)
    value.add_argument("--resume-from", default=None)
    value.add_argument("--status-json", default=None)
    value.add_argument("--heartbeat-seconds", type=float, default=30.0)
    return value


def main(argv=None) -> int:
    final = run_recipe(parser().parse_args(argv))
    print(json.dumps(
        {"status": final["status"], "rounds": final["rounds_completed"],
         "accepted": final["accepted"]},
        sort_keys=True,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
