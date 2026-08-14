"""Experimental MPC-cost execution rule over the frozen acquisition contract.

This module is an ADDITIVE variant runner of the authoritative no-update
diagnostic in ``sfm_hp100_predictive_execution``.  Everything in the frozen
contract is unchanged: K=64/B=32, ESS/K target, the retry schedule, exact
full-H10 GREEN eligibility, NVP semantics, the archive/trace schema, and the
raw evaluator.  Only the choice AMONG exact positives differs.

Motivation (measured on the round-1 archive): the authoritative lexicographic
selector is effectively pure argmax H10 progress -- continuous first key,
0/2300 ties -- so its clearance/uncertainty keys never act, and lineages burn
clearance monotonically into dead-end states that terminate NVP.  This variant
replaces the dead lexicographic tail with one declared scalar cost inspired by
the historical MPPI min-cost rule:

    cost_j = -progress_j
             + lam * sum_{h=1..10} rho**(10-h) * exp((r_eff - d_hj)/sigma_len)

where ``d_hj`` is the CV-predicted min-over-pedestrians clearance of candidate
``j`` at horizon step ``h``.  ``rho > 1`` weights near-horizon proximity more
than late-horizon proximity; a SMALL ``r_eff`` keeps the exponential inactive
until the plan actually approaches the crowd, preserving the pretrained
aggressiveness; ``sigma_len`` sets how sharply the soft cost turns on.  There
is no hard clearance gate: eligibility stays exact GREEN only.

Reproducing archive_r1 lineages: keep ``--scenario-start 860000 --seed 41
--lineages-per-gamma 8`` and select gammas via ``--gammas``.  A replica-subset
flag is NOT provided because the scene episode id depends on
``lineages_per_gamma`` and the replica index; changing either would silently
change scenarios.  Gamma subsetting is the only safe knob.

Every artifact written here carries ``execution_rule="predictive_mpc_v2"``,
the four cost parameters, and this file's SHA-256, so a v2 trace can never be
mistaken for an authoritative diagnostic.  Traces keep the events/outcomes
layout, so ``sfm_hp100_predictive_execution_viz.py`` renders them unchanged.
Alongside the v2 choice, the OLD rule's choice is recorded per selector call
(``v2_shadow`` on the first prediction audit plus ``selector_shadow.jsonl``)
for before/after selection-change statistics and rescue-case videos.
"""
from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_ball_launch as BASE
from sfm_hp100_ball_core.expansion import mean_pairwise_lengthscale
import sfm_hp100_exhaustive_hybrid as HYBRID
import sfm_hp100_predictive_execution as PRED
import sfm_scene as SS


VERSION = "sfm_hp100_predictive_execution_v2_mpc"
EXECUTION_RULE = "predictive_mpc_v2"
V2_PREFLIGHT_STATUS = "SFM_HP100_PREDICTIVE_EXECUTION_V2_PREFLIGHT_PASSED"
V2_TRACE_STATUS = "SFM_HP100_PREDICTIVE_EXECUTION_V2_TRACE"
V2_STATUS = "SFM_HP100_PREDICTIVE_EXECUTION_V2_COMPLETE"
V2_SAMPLE_STATUS = "SFM_HP100_PREDICTIVE_V2_SAMPLE_ARCHIVE"
HORIZON_KEY = "mpc_horizon_clearances"
COST_KEY = "mpc_cost"
SHADOW_KEY = "v2_shadow"
# Exponent clamp: keeps synthetic/degenerate inputs from overflowing exp();
# real exact-positive clearances are >= 0 so the exponent stays <= r_eff/sigma.
MAX_EXPONENT = 60.0


@dataclass(frozen=True)
class MPCRuleParams:
    """Declared parameters of the v2 scalar execution cost.

    lam:       weight of the summed proximity cost against H10 progress.
    rho:       per-step decay base; rho > 1 makes a violation at h=1 cost
               rho**(H-1)/rho**(H-h) times more than the same violation at h,
               i.e. one-step-ahead proximity dominates.
    r_eff:     effective radius (m) where the exponential activates; small
               values preserve the pretrained aggressiveness away from the
               crowd.
    sigma_len: softness length (m) of the exponential activation.
    """

    lam: float = 1.0
    rho: float = 1.1
    r_eff: float = 0.30
    sigma_len: float = 0.10

    def validate(self) -> None:
        if self.lam < 0.0:
            raise ValueError("lam must be non-negative")
        if self.rho < 1.0:
            raise ValueError("rho must be >= 1 (near-horizon weighting)")
        if self.r_eff <= 0.0:
            raise ValueError("r_eff must be positive")
        if self.sigma_len <= 0.0:
            raise ValueError("sigma_len must be positive")


def horizon_clearances(task, context, candidate) -> list[float]:
    """CV-predicted min-over-pedestrians clearance per horizon step h=1..H.

    Reuses exactly the distance-matrix construction of
    ``PRED.prediction_metrics`` (clipped dynamics, CV pedestrian propagation,
    all pedestrians); ``min(vector)`` equals its ``predicted_min_clearance``.
    """
    robot, ped_xy, ped_vel = task.decode_context(context)
    controls = np.asarray(candidate.detach().cpu(), np.float32).reshape(-1, 2)
    states = PORT.clipped_plan_states(robot, controls)
    segment = states[:, :2]
    pedestrians = PORT.VERIFY.predict_pedestrians(
        ped_xy, ped_vel, H=len(controls),
    )
    if pedestrians.shape[1]:
        distance = np.linalg.norm(
            segment[:, None, :] - pedestrians, axis=2,
        ) - float(SS.R_PED)
        return [float(value) for value in distance[1:].min(axis=1)]
    return [float("inf")] * len(controls)


def mpc_cost_components(task, context, candidate) -> dict:
    """Selector-visible cost inputs for one H10 candidate."""
    clearances = horizon_clearances(task, context, candidate)
    return {
        "horizon_clearances": clearances,
        "min_clearance": float(min(clearances)),
    }


def mpc_cost(
    progress: float,
    clearances: Sequence[float],
    params: MPCRuleParams,
) -> float:
    """-progress + lam * sum_h rho**(H-h) * exp((r_eff - d_h)/sigma_len)."""
    horizon = len(clearances)
    proximity = 0.0
    for index, clearance in enumerate(clearances):
        h = index + 1
        exponent = (params.r_eff - float(clearance)) / params.sigma_len
        proximity += params.rho ** (horizon - h) * math.exp(
            min(exponent, MAX_EXPONENT)
        )
    return -float(progress) + params.lam * proximity


def select_predictive_mpc(
    results: Sequence,
    audits: Sequence[dict],
    selected_sigma: Sequence[float],
    *,
    params: MPCRuleParams,
    cost_inputs: Sequence[Sequence[float]] | None = None,
) -> int | None:
    """Minimum-cost exact positive; eligibility identical to the frozen rule.

    ``cost_inputs`` supplies per-candidate horizon clearance vectors; when
    omitted they are read from ``audit[HORIZON_KEY]`` (installed by the
    patched ``prediction_metrics``).  Deterministic tie-break: cost, then
    higher progress, then lower candidate index.  Per-candidate costs are
    recorded into ``audit[COST_KEY]`` (None for ineligible candidates).
    """
    if not (len(results) == len(audits) == len(selected_sigma)):
        raise ValueError("selector inputs differ in length")
    if cost_inputs is not None and len(cost_inputs) != len(results):
        raise ValueError("cost inputs differ in length")
    params.validate()
    eligible = []
    costs: dict[int, float] = {}
    for index, (result, audit) in enumerate(zip(results, audits)):
        audit[COST_KEY] = None
        if result.error or not result.valid or not result.progress_eligible:
            continue
        if not audit["predicted_collision_free"]:
            raise RuntimeError(
                "exact-positive candidate failed its duplicate CV collision audit"
            )
        if not math.isclose(
            float(result.progress), float(audit["H10_goal_progress"]),
            rel_tol=0.0, abs_tol=2.0e-6,
        ):
            raise RuntimeError("verifier and selector H10 progress differ")
        clearances = (
            cost_inputs[index] if cost_inputs is not None
            else audit.get(HORIZON_KEY)
        )
        if clearances is None:
            raise ValueError(
                "mpc selector requires per-horizon clearances "
                f"(cost_inputs or audit[{HORIZON_KEY!r}])"
            )
        cost = mpc_cost(float(result.progress), clearances, params)
        costs[index] = cost
        audit[COST_KEY] = float(cost)
        eligible.append(index)
    if not eligible:
        return None
    return min(eligible, key=lambda index: (
        float(costs[index]),
        -float(results[index].progress),
        int(index),
    ))


@contextlib.contextmanager
def install_v2_selector(params: MPCRuleParams, shadow: list):
    """Scoped patch of PRED's selector and audit builder.

    ``PRED.gather_predictive`` resolves both ``prediction_metrics`` and
    ``select_predictive_progress`` through module globals at call time, so a
    scoped rebind is sufficient and the frozen file is never edited.  The
    patched audit builder appends the per-horizon clearance vector; the
    patched selector runs BOTH rules, records the old rule's choice, and
    returns the v2 choice.  Trace schema is unchanged apart from additive
    audit keys, so the existing viz renders v2 traces.
    """
    params.validate()
    original_selector = PRED.select_predictive_progress
    original_metrics = PRED.prediction_metrics

    def patched_metrics(task, context, candidate):
        audit = original_metrics(task, context, candidate)
        clearances = horizon_clearances(task, context, candidate)
        recomputed = float(min(clearances))
        recorded = float(audit["predicted_min_clearance"])
        if math.isfinite(recorded) or math.isfinite(recomputed):
            if not math.isclose(
                recomputed, recorded, rel_tol=0.0, abs_tol=1.0e-4,
            ):
                raise RuntimeError(
                    "v2 horizon clearances disagree with the frozen CV audit"
                )
        audit[HORIZON_KEY] = clearances
        return audit

    def patched_selector(results, audits, selected_sigma):
        old_local = original_selector(results, audits, selected_sigma)
        new_local = select_predictive_mpc(
            results, audits, selected_sigma, params=params,
        )
        record = {
            "call": len(shadow),
            "old_rule_local": None if old_local is None else int(old_local),
            "new_rule_local": None if new_local is None else int(new_local),
            "changed": (
                old_local is not None and new_local is not None
                and int(old_local) != int(new_local)
            ),
            "eligible": int(sum(
                audit.get(COST_KEY) is not None for audit in audits
            )),
        }
        shadow.append(record)
        if audits:
            audits[0][SHADOW_KEY] = dict(record)
        return new_local

    PRED.prediction_metrics = patched_metrics
    PRED.select_predictive_progress = patched_selector
    try:
        yield
    finally:
        PRED.select_predictive_progress = original_selector
        PRED.prediction_metrics = original_metrics


def v2_provenance(params: MPCRuleParams) -> dict:
    """Fields stamped into every v2 artifact so it cannot pass as authoritative."""
    return {
        "execution_rule": EXECUTION_RULE,
        "mpc_params": asdict(params),
        "authoritative": False,
        "v2_version": VERSION,
        "v2_source_sha256": PRED.sha256_file(__file__),
        "base_module": {
            "version": PRED.VERSION,
            "source_sha256": PRED.sha256_file(PRED.__file__),
        },
    }


def shadow_summary(shadow: Sequence[dict]) -> dict:
    comparable = [
        row for row in shadow
        if row["old_rule_local"] is not None
        and row["new_rule_local"] is not None
    ]
    return {
        "selector_calls": len(shadow),
        "comparable_calls": len(comparable),
        "changed_calls": int(sum(row["changed"] for row in comparable)),
        "change_fraction": (
            None if not comparable
            else float(np.mean([row["changed"] for row in comparable]))
        ),
    }


def run(args) -> dict:
    """v2 mirror of ``PRED.run``: same preflight/gather/summarize transaction,
    the scoped selector patch installed around the gather, and v2 provenance
    stamped into every artifact."""
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing output: {output}")
    params = MPCRuleParams(
        lam=float(args.mpc_lam), rho=float(args.mpc_rho),
        r_eff=float(args.mpc_r_eff), sigma_len=float(args.mpc_sigma_len),
    )
    params.validate()
    gpu_contract = BASE._gpu_contract(args.device, int(args.physical_gpu))
    checkpoint_sha = PRED.sha256_file(args.checkpoint)
    if checkpoint_sha != str(args.expected_checkpoint_sha256).lower():
        raise RuntimeError("checkpoint SHA256 mismatch")
    policy, payload = GPS.load_sfm_hp100_policy(
        args.checkpoint, device=args.device,
    )
    adapter = PORT.HP100ExpansionPolicy(policy).eval()
    reference_policy, _ = GPS.load_sfm_hp100_policy(
        args.checkpoint, device=args.device,
    )
    reference = PORT.HP100ExpansionPolicy(reference_policy).eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    before = PRED._model_state_sha256(adapter)

    config = PRED.PredictiveConfig(
        gammas=HYBRID._parse_gammas(args.gammas),
        lineages_per_gamma=int(args.lineages_per_gamma),
        max_steps=int(args.max_steps), max_attempts=int(args.max_attempts),
        ess_target=float(args.ess_target), seed=int(args.seed),
    )
    config.validate()
    features, calibration = BASE.calibration_features(
        reference, dataset_root=args.pretrain_dataset_root,
        expected_manifest_sha256=args.expected_pretrain_dataset_manifest_sha256,
        count=50, seed=config.seed, base_std=1.0,
        paired_noised_representation=True,
    )
    lengthscale = mean_pairwise_lengthscale(features)
    support = HYBRID._calibration_support_by_gamma(
        features, calibration, config.gammas,
        device=next(reference.parameters()).device,
    )
    task = PORT.SFMHP100ExpansionTask(
        scene_profile=args.scene_profile,
        scenario_start=int(args.scenario_start),
    ).attach_context_encoder(policy)
    keys = PRED._lineage_keys(config)

    output.mkdir(parents=True)
    preflight = {
        "status": V2_PREFLIGHT_STATUS,
        "version": VERSION,
        **v2_provenance(params),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_scientific_status": payload.get("scientific_status"),
        "config": asdict(config),
        "scene": SS.scene_profile(args.scene_profile),
        "scenario_start": int(args.scenario_start),
        "gpu": gpu_contract,
        "calibration": {**calibration, "lengthscale": float(lengthscale)},
        "prediction_note": (
            "exact GREEN eligibility unchanged; the MPC cost only re-ranks "
            "exact positives; the old rule's choice is shadow-recorded"
        ),
        "policy_trainable_parameters": 0,
    }
    PRED._write_json(output / "PREFLIGHT.json", preflight)

    shadow: list[dict] = []
    with HYBRID._OrderedSidecarVerifier(
        task, int(args.verifier_workers),
    ) as verifier:
        with install_v2_selector(params, shadow):
            gathered = PRED.gather_predictive(
                adapter, reference, task, keys=keys, config=config,
                lengthscale=float(lengthscale), support_by_gamma=support,
                verifier=verifier,
            )
    after = PRED._model_state_sha256(adapter)
    if after != before:
        raise RuntimeError("no-update v2 diagnostic changed the policy")
    summary = PRED.summarize(gathered, config)

    trace_path = output / "predictive_trace.pt"
    torch.save({
        "status": V2_TRACE_STATUS, "version": VERSION,
        **v2_provenance(params),
        "preflight": preflight, **gathered,
    }, trace_path)
    samples_path = output / "predictive_samples.pt"
    torch.save({
        "status": V2_SAMPLE_STATUS,
        **v2_provenance(params),
        "semantics": (
            "v2 MPC-rule diagnostic samples; NOT authoritative expansion "
            "training data unless the v2 rule is separately adopted; one "
            "executed exact-positive per successful context unless its first "
            "action realizes collision/OOB; one nonexecuted resolved "
            "exact-negative counterfactual per exhausted lineage"
        ),
        "samples": gathered["samples"],
    }, samples_path)
    shadow_path = output / "selector_shadow.jsonl"
    with shadow_path.open("w") as stream:
        for row in shadow:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    marker = {
        "status": V2_STATUS, "version": VERSION,
        **v2_provenance(params),
        "preflight": preflight, "summary": summary,
        "selector_shadow": shadow_summary(shadow),
        "outcomes": {
            label: dict(row) for label, row in gathered["outcomes"].items()
        },
        "policy_unchanged": True,
        "policy_state_sha256": before,
        "trace": {
            "path": str(trace_path),
            "sha256": PRED.sha256_file(trace_path),
        },
        "samples": {
            "path": str(samples_path),
            "sha256": PRED.sha256_file(samples_path),
        },
        "shadow": {
            "path": str(shadow_path),
            "sha256": PRED.sha256_file(shadow_path),
        },
        "timers_seconds": gathered["timers_seconds"],
    }
    PRED._write_json(output / "PREDICTIVE_EXECUTION_V2_COMPLETE.json", marker)
    return marker


def parser():
    value = PRED.parser()
    value.description = __doc__
    value.add_argument("--mpc-lam", type=float, default=MPCRuleParams.lam)
    value.add_argument("--mpc-rho", type=float, default=MPCRuleParams.rho)
    value.add_argument("--mpc-r-eff", type=float, default=MPCRuleParams.r_eff)
    value.add_argument(
        "--mpc-sigma-len", type=float, default=MPCRuleParams.sigma_len,
    )
    return value


def main(argv=None) -> int:
    result = run(parser().parse_args(argv))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
