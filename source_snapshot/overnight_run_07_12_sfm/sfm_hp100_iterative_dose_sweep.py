"""Authenticated development sweep over one saved HP100 microcycle view.

Every cell starts from the same preserved selected checkpoint, consumes the
same completed-cycle P1/P2/Ncausal training view, and is evaluated on the same
canonical 16-lineage raw temperature-one CRN gate.  Original Dminus rows are
never selected.  This is a development screen only: its checkpoints do not
enter GP support, an archive, or promotion.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import torch

import _paths  # noqa: F401
import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_ball_launch as BASE
import sfm_hp100_disaster_prefix_continuation as CONT
import sfm_hp100_disaster_prefix_sweep as SWEEP
import sfm_hp100_exhaustive_hybrid as HYBRID
import sfm_hp100_iterative_microcycles as ITER


VERSION = "sfm_hp100_iterative_dose_sweep_v1"
PREFLIGHT_STATUS = "SFM_HP100_ITERATIVE_DOSE_SWEEP_PREFLIGHT_PASSED"
COMPLETE_STATUS = "SFM_HP100_ITERATIVE_DOSE_SWEEP_COMPLETE"
NO_WINNER_STATUS = "SFM_HP100_ITERATIVE_DOSE_SWEEP_NO_ELIGIBLE_CELL"
EXPECTED_EVIDENCE_SOURCE_COMMIT = "15231d30ff4af8427509464dd827490a221b3c78"
SCOPES = ("head_only", "last_block_and_head")
DOSE_FAMILIES = ("original", "p1_dominant", "balanced_depth")


@dataclass(frozen=True)
class Dose:
    name: str
    pair: str
    p1_learning_rate: float
    p1_passes: int
    p2_learning_rate: float
    p2_passes: int
    negative_learning_rate: float
    negative_alpha: float
    negative_passes: int

    @property
    def positive_only(self) -> bool:
        return self.negative_alpha == 0.0


def declared_doses(family: str = "original") -> tuple[Dose, ...]:
    """Return one declared table of paired positive/Ncausal cells."""
    if family == "original":
        rows = (
            ("soft1", 1.0e-5, 1, 3.0e-4, 1, 3.0e-5, 4),
            ("selected1", 1.0e-5, 1, 1.0e-3, 1, 1.0e-4, 4),
            ("repeat4", 1.0e-5, 4, 3.0e-4, 4, 3.0e-5, 16),
            ("sufficient4", 3.0e-5, 4, 1.0e-3, 4, 1.0e-4, 16),
        )
    elif family == "p1_dominant":
        rows = (
            ("balanced1", 1.0e-5, 1, 1.0e-5, 1, 3.0e-5, 4),
            ("retain1", 3.0e-5, 1, 1.0e-5, 1, 3.0e-5, 4),
            ("anchor4", 3.0e-5, 4, 1.0e-6, 1, 3.0e-5, 4),
            ("p1strong", 1.0e-4, 1, 1.0e-6, 1, 3.0e-5, 4),
        )
    elif family == "balanced_depth":
        rows = (
            ("balanced2", 1.0e-5, 2, 1.0e-5, 2, 3.0e-5, 2),
            ("balanced4", 1.0e-5, 4, 1.0e-5, 4, 3.0e-5, 4),
            ("balanced8", 1.0e-5, 8, 1.0e-5, 8, 3.0e-5, 4),
            ("balanced16", 1.0e-5, 16, 1.0e-5, 16, 3.0e-5, 4),
        )
    else:
        raise ValueError(f"dose family must be one of {DOSE_FAMILIES}")
    doses = []
    for pair, p1_lr, p1_passes, p2_lr, p2_passes, negative_lr, negative_passes in rows:
        doses.extend((
            Dose(
                f"{pair}_positive_only", pair,
                p1_lr, p1_passes, p2_lr, p2_passes,
                negative_lr, 0.0, 1,
            ),
            Dose(
                f"{pair}_ncausal", pair,
                p1_lr, p1_passes, p2_lr, p2_passes,
                negative_lr, 0.1, negative_passes,
            ),
        ))
    return tuple(doses)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def _torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _torch_load(path: Path) -> Any:
    return torch.load(path, map_location="cpu", weights_only=False)


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _record(path: Path) -> dict:
    return {"path": str(path), "sha256": HYBRID._sha256(path)}


def _require_hash(path: Path, expected: str, role: str) -> str:
    actual = HYBRID._sha256(path)
    if actual != str(expected).lower():
        raise RuntimeError(f"{role} SHA256 mismatch")
    return actual


def _cycle_row(marker: dict, cycle: int) -> dict:
    matches = [
        row for row in marker.get("cycles", ())
        if int(row.get("microcycle", -1)) == int(cycle)
    ]
    if len(matches) != 1:
        raise ValueError("declared cycle is not unique in iterative marker")
    return matches[0]


def load_completed_cycle(
    iterative_root: str | Path,
    expected_marker_sha256: str,
    cycle: int,
) -> dict:
    """Authenticate one admissible current-cycle view and its selected parent."""
    root = Path(iterative_root).resolve()
    marker_path = root / "ITERATIVE_COMPLETE.json"
    marker_sha = _require_hash(
        marker_path, expected_marker_sha256, "iterative completion marker",
    )
    marker = _json(marker_path)
    if marker.get("status") not in {
        ITER.ALL_CLEAR_STATUS, ITER.PLATEAU_STATUS, ITER.MAX_CYCLES_STATUS,
    }:
        raise ValueError("iterative root has no supported terminal status")
    if marker.get("source_HEAD") != EXPECTED_EVIDENCE_SOURCE_COMMIT:
        raise ValueError("iterative evidence source commit is not 15231d3")
    if marker.get("promotable") is not False:
        raise ValueError("iterative evidence unexpectedly claims promotion")
    if marker.get("GP_updated") is not False or marker.get("archive_committed") is not False:
        raise ValueError("iterative evidence contaminated GP or archive state")

    preflight_path = root / "PREFLIGHT.json"
    _require_hash(preflight_path, marker["preflight_sha256"], "iterative preflight")
    preflight = _json(preflight_path)
    if preflight.get("status") != ITER.PREFLIGHT_STATUS:
        raise ValueError("iterative preflight status is invalid")
    if preflight.get("source_gate", {}).get("HEAD") != EXPECTED_EVIDENCE_SOURCE_COMMIT:
        raise ValueError("iterative preflight source commit drifted")

    selected = dict(marker.get("initial_selected_checkpoint") or {})
    if set(selected) != {"path", "sha256", "policy_state_sha256"}:
        raise ValueError("iterative marker lacks the preserved selected checkpoint")
    selected_path = Path(selected["path"]).resolve()
    _require_hash(selected_path, selected["sha256"], "selected checkpoint")
    preflight_selected = preflight.get("selection", {}).get("checkpoint")
    if preflight_selected != selected:
        raise ValueError("iterative preflight and marker selected checkpoints differ")

    row = _cycle_row(marker, int(cycle))
    cycle_root = root / "microcycles" / f"cycle_{int(cycle):03d}"
    cycle_marker_path = cycle_root / "CYCLE_COMPLETE.json"
    _require_hash(cycle_marker_path, row["marker_sha256"], "cycle marker")
    cycle_marker = _json(cycle_marker_path)
    if cycle_marker.get("status") not in {
        "SFM_HP100_ITERATIVE_MICROCYCLE_ACCEPTED",
        "SFM_HP100_ITERATIVE_MICROCYCLE_REJECTED",
    }:
        raise ValueError("cycle marker is not complete")
    if int(cycle_marker.get("microcycle", -1)) != int(cycle):
        raise ValueError("cycle marker index drifted")
    if cycle_marker.get("parent_policy_state_sha256") != selected["policy_state_sha256"]:
        raise ValueError("declared cycle was not gathered from the selected checkpoint")
    if cycle_marker.get("training_view_admissible") is not True:
        raise ValueError("declared cycle has no admissible training view")
    if cycle_marker.get("GP_updated") is not False or cycle_marker.get("archive_committed") is not False:
        raise ValueError("declared cycle entered GP or archive state")

    training_path = cycle_root / "training_view.pt"
    artifacts = cycle_marker["artifacts"]
    _require_hash(training_path, artifacts["training_view_sha256"], "training view")
    training_payload = _torch_load(training_path)
    if training_payload.get("status") != "SFM_HP100_ITERATIVE_MICROCYCLE_TRAINING_VIEW":
        raise ValueError("training-view payload status is invalid")
    if int(training_payload.get("microcycle", -1)) != int(cycle):
        raise ValueError("training-view cycle drifted")
    if training_payload.get("admissible") is not True:
        raise ValueError("training-view payload is inadmissible")
    if training_payload.get("enters_GP") is not False:
        raise ValueError("training view unexpectedly enters GP support")
    if training_payload.get("source_cycles") != [int(cycle)]:
        raise ValueError("training view is not current-cycle-only")
    semantic = {
        "microcycle": int(cycle),
        "training_view": training_payload["training_view"],
        "admissible": True,
    }
    if ITER.canonical_sha256(semantic) != artifacts["training_view_canonical_sha256"]:
        raise RuntimeError("training-view canonical SHA256 mismatch")
    view = training_payload["training_view"]
    if not view["retained_P1"] or not view["retained_P2"] or not view["Ncausal"]:
        raise ValueError("development sweep requires nonempty P1, P2, and Ncausal rows")
    if cycle_marker["training_counts"]["selected_training_exposures"]["Dminus"] != 0:
        raise ValueError("source cycle exposed original Dminus rows")

    baseline_path = root / "raw_gate_baseline.pt"
    _require_hash(baseline_path, marker["baseline_raw_gate_sha256"], "raw baseline")
    baseline_raw = _torch_load(baseline_path)
    if ITER.canonical_sha256(baseline_raw) != marker["baseline_raw_gate_canonical_sha256"]:
        raise RuntimeError("raw baseline canonical SHA256 mismatch")
    baseline_summary = ITER._raw_gate_summary(
        baseline_raw, HYBRID._lineage_keys(_config_from_preflight(preflight)),
    )
    if baseline_summary != marker.get("baseline_raw_gate"):
        raise ValueError("raw baseline artifact and iterative summary differ")

    reference = {
        "path": str(Path(preflight["reference_checkpoint"]).resolve()),
        "sha256": str(preflight["reference_checkpoint_sha256"]),
    }
    _require_hash(Path(reference["path"]), reference["sha256"], "r0 reference checkpoint")
    return {
        "root": str(root),
        "iterative_marker": {"path": str(marker_path), "sha256": marker_sha},
        "marker": marker,
        "preflight": preflight,
        "selected_checkpoint": selected,
        "cycle": int(cycle),
        "cycle_marker": {"path": str(cycle_marker_path), "sha256": row["marker_sha256"]},
        "training_view_artifact": _record(training_path),
        "training_view": view,
        "training_view_canonical_sha256": artifacts["training_view_canonical_sha256"],
        "baseline_raw_artifact": _record(baseline_path),
        "baseline_raw": baseline_raw,
        "baseline_summary": baseline_summary,
        "reference_checkpoint": reference,
    }


def _config_from_preflight(preflight: dict) -> HYBRID.HybridConfig:
    values = dict(preflight["config"])
    values["gammas"] = tuple(map(float, values["gammas"]))
    config = HYBRID.HybridConfig(**values)
    config.validate()
    if len(HYBRID._lineage_keys(config)) != ITER.EXPECTED_LINEAGES:
        raise ValueError("saved iterative config is not the canonical 16-lineage bank")
    return config


def _configure(adapter: PORT.HP100ExpansionPolicy, scope: str):
    parameters = ITER._configure_trainable(adapter, scope)
    return parameters


def _load_adapter(checkpoint: dict, scope: str, device: str):
    policy, payload = GPS.load_sfm_hp100_policy(checkpoint["path"], device=device)
    adapter = PORT.HP100ExpansionPolicy(policy)
    parameters = _configure(adapter, scope)
    if HYBRID._model_state_sha256(adapter) != checkpoint["policy_state_sha256"]:
        raise RuntimeError("loaded selected policy-state SHA256 mismatch")
    return adapter, payload, parameters


def _assert_update(update: dict, view: dict, dose: Dose) -> None:
    negative = update["negative"]
    if negative["negative_source"] != "Ncausal_only":
        raise RuntimeError("cell used a non-Ncausal negative source")
    selected = negative["selected_source_counts"]
    if int(selected["Dminus"]) != 0:
        raise RuntimeError("original Dminus entered a development cell")
    if int(selected["Ncausal"]) != len(view["Ncausal"]):
        raise RuntimeError("Ncausal row count drifted")
    expected = {
        "P1": (len(view["retained_P1"]), dose.p1_passes),
        "P2": (len(view["retained_P2"]), dose.p2_passes),
        "negative": (len(view["Ncausal"]), dose.negative_passes),
    }
    for phase, (rows, passes) in expected.items():
        report = update[phase]
        if int(report["rows"]) != rows or int(report["passes"]) != passes:
            raise RuntimeError(f"{phase} row/pass accounting drifted")
        if int(report["total_row_exposures"]) != rows * passes:
            raise RuntimeError(f"{phase} exact-pass exposure accounting drifted")
    effective = float(dose.negative_learning_rate * dose.negative_alpha)
    if float(negative["effective_learning_rate"]) != effective:
        raise RuntimeError("negative effective learning rate drifted")


def _raw_base_rows(payload: dict) -> list[dict]:
    return [
        {
            "lineage": str(event["lineage"]),
            "step": int(event["step"]),
            "flow_base": event["flow_base"],
        }
        for event in payload.get("events", ())
    ]


def _register_raw_bases(registry: dict, payload: dict) -> str:
    rows = _raw_base_rows(payload)
    for row in rows:
        key = (row["lineage"], row["step"])
        digest = ITER.canonical_sha256(row["flow_base"])
        previous = registry.setdefault(key, digest)
        if previous != digest:
            raise RuntimeError("raw temperature-one CRN flow base drifted")
    return ITER.canonical_sha256(rows)


def _checkpoint_payload(
    adapter: PORT.HP100ExpansionPolicy,
    *,
    parent: dict,
    scope: str,
    dose: Dose,
    cycle_marker_sha256: str,
) -> dict:
    return {
        "scientific_status": "HP100_ITERATIVE_DOSE_SWEEP_DIAGNOSTIC_ONLY",
        "state_dict": {
            key: value.detach().cpu().clone()
            for key, value in adapter.policy.state_dict().items()
        },
        "config": adapter.policy.config(),
        "parent_checkpoint_sha256": parent["sha256"],
        "parent_policy_state_sha256": parent["policy_state_sha256"],
        "optimizer_scope": scope,
        "dose": asdict(dose),
        "source_cycle_marker_sha256": cycle_marker_sha256,
        "promotable": False,
        "enters_GP": False,
        "enters_archive": False,
    }


def _selection_key(cell: dict) -> tuple:
    return (
        -int(cell["acceptance"]["raw_temperature_one_CLEAR_delta"]),
        float(cell["update"]["relative_parameter_drift"]),
        int(cell["dose_index"]),
        int(cell["scope_index"]),
    )


def evaluate_cells(
    *,
    adapter_factory: Callable[[str], PORT.HP100ExpansionPolicy],
    training_view: dict,
    baseline_summary: dict,
    evaluate_fn: Callable[[PORT.HP100ExpansionPolicy, str], tuple[dict, dict]],
    output: Path,
    parent_checkpoint: dict,
    cycle_marker_sha256: str,
    batch_size: int,
    max_relative_parameter_drift: float,
    seed: int,
    doses: Sequence[Dose] = (),
    scopes: Sequence[str] = SCOPES,
    raw_crn_reference: dict | None = None,
    update_fn: Callable = CONT.apply_disaster_prefix_update,
) -> dict:
    """Evaluate declared cells without sharing model or optimizer state."""
    doses = tuple(doses or declared_doses())
    if tuple(scopes) != SCOPES:
        raise ValueError("development sweep requires the paired declared scopes")
    view_sha = ITER.canonical_sha256(training_view)
    cells = []
    raw_base_registry = {}
    if raw_crn_reference is not None:
        _register_raw_bases(raw_base_registry, raw_crn_reference)
    positive_anchor_shas = {}
    for dose_index, dose in enumerate(doses):
        for scope_index, scope in enumerate(scopes):
            cell_name = f"{scope}__{dose.name}"
            cell_root = output / "cells" / cell_name
            adapter = adapter_factory(scope)
            parent_policy_sha = HYBRID._model_state_sha256(adapter)
            if parent_policy_sha != parent_checkpoint["policy_state_sha256"]:
                raise RuntimeError("cell did not start from the preserved selected policy")
            update = update_fn(
                adapter, training_view,
                p2_learning_rate=dose.p2_learning_rate,
                p1_learning_rate=dose.p1_learning_rate,
                negative_learning_rate=dose.negative_learning_rate,
                alpha=dose.negative_alpha,
                batch_size=int(batch_size),
                max_relative_parameter_drift=float(max_relative_parameter_drift),
                seed=int(seed),
                p1_passes=dose.p1_passes,
                p2_passes=dose.p2_passes,
                negative_passes=dose.negative_passes,
                negative_source="Ncausal_only",
            )
            update.pop("_positive_anchor_state_dict", None)
            _assert_update(update, training_view, dose)
            anchor_key = (scope, dose.pair)
            anchor_sha = str(update["positive_anchor_model_sha256"])
            previous_anchor = positive_anchor_shas.setdefault(anchor_key, anchor_sha)
            if previous_anchor != anchor_sha:
                raise RuntimeError(
                    "paired positive-only/Ncausal positive anchors differ"
                )
            if ITER.canonical_sha256(training_view) != view_sha:
                raise RuntimeError("cell update mutated the authenticated training view")

            raw_payload = None
            raw_summary = None
            decision = {
                "accepted": False,
                "reason": "atomic_parameter_drift_gate_rejected",
                "raw_temperature_one_CLEAR_delta": 0,
            }
            checkpoint = None
            raw_record = None
            raw_base_sha = None
            if update["accepted"]:
                raw_payload, raw_summary = evaluate_fn(adapter, cell_name)
                raw_base_sha = _register_raw_bases(raw_base_registry, raw_payload)
                decision = ITER._acceptance_decision(
                    baseline_summary, raw_summary,
                )
                checkpoint_path = cell_root / "checkpoint_candidate.pt"
                _torch_save(checkpoint_path, _checkpoint_payload(
                    adapter, parent=parent_checkpoint, scope=scope, dose=dose,
                    cycle_marker_sha256=cycle_marker_sha256,
                ))
                checkpoint = {
                    **_record(checkpoint_path),
                    "policy_state_sha256": HYBRID._model_state_sha256(adapter),
                }
                raw_path = cell_root / "raw_gate.pt"
                _torch_save(raw_path, raw_payload)
                raw_record = {
                    **_record(raw_path),
                    "canonical_sha256": ITER.canonical_sha256(raw_payload),
                }

            eligible = bool(update["accepted"] and decision["accepted"])
            cell = {
                "status": (
                    "SFM_HP100_ITERATIVE_DOSE_CELL_ELIGIBLE"
                    if eligible else "SFM_HP100_ITERATIVE_DOSE_CELL_REJECTED"
                ),
                "cell": cell_name,
                "scope": scope,
                "scope_index": scope_index,
                "dose": asdict(dose),
                "dose_index": dose_index,
                "positive_only_control": dose.positive_only,
                "parent_policy_state_sha256": parent_policy_sha,
                "training_view_canonical_sha256": view_sha,
                "update": update,
                "raw_gate": raw_summary,
                "acceptance": decision,
                "eligible": eligible,
                "checkpoint": checkpoint,
                "raw_gate_artifact": raw_record,
                "raw_flow_base_canonical_sha256": raw_base_sha,
                "Dminus_training_exposures": 0,
                "GP_updated": False,
                "archive_committed": False,
                "promotable": False,
            }
            marker_path = cell_root / "CELL_COMPLETE.json"
            _write_json(marker_path, cell)
            cells.append({
                **cell,
                "marker": str(marker_path),
                "marker_sha256": HYBRID._sha256(marker_path),
            })

    eligible = [cell for cell in cells if cell["eligible"]]
    winner = min(eligible, key=_selection_key) if eligible else None
    return {
        "status": COMPLETE_STATUS if winner is not None else NO_WINNER_STATUS,
        "version": VERSION,
        "baseline_raw_gate": baseline_summary,
        "declared_doses": [asdict(dose) for dose in doses],
        "declared_scopes": list(scopes),
        "paired_positive_anchor_sha256": {
            f"{scope}:{pair}": digest
            for (scope, pair), digest in sorted(positive_anchor_shas.items())
        },
        "common_raw_CRN_registry": {
            "rows": len(raw_base_registry),
            "canonical_sha256": ITER.canonical_sha256([
                {"lineage": key[0], "step": key[1], "flow_base_sha256": digest}
                for key, digest in sorted(raw_base_registry.items())
            ]),
        },
        "cells": cells,
        "eligible_cells": len(eligible),
        "selection_rule": (
            "prior CLEAR preservation, at least one new raw temp1 CLEAR, "
            "candidate-non-CLEAR prefix and aggregate progress nondecrease; "
            "maximize new CLEAR then lower relative parameter drift, declared order"
        ),
        "winner": winner,
        "selected_checkpoint": None if winner is None else winner["checkpoint"],
        "promotable": False,
        "GP_updated": False,
        "archive_committed": False,
        "disjoint_evaluation_required_before_promotion": True,
    }


def run(args) -> dict:
    repo = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"refusing existing sweep output: {output}")
    if output == repo or repo in output.parents:
        raise ValueError("sweep output must remain outside the source worktree")
    source_gate = SWEEP.source_gate(repo, args.expected_source_commit)
    doses = declared_doses(args.dose_family)
    inputs = load_completed_cycle(
        args.iterative_root,
        args.expected_iterative_marker_sha256,
        args.cycle,
    )
    config = _config_from_preflight(inputs["preflight"])
    selected = inputs["selected_checkpoint"]
    reference = inputs["reference_checkpoint"]

    reference_adapter, reference_payload, reference_sha, _, _ = (
        HYBRID._load_input_checkpoint(
            reference["path"], reference["sha256"], args.device, "head_only",
        )
    )
    if reference_payload.get("scientific_status") != "canonical_ID_promoted":
        raise ValueError("raw-gate context reference is not canonical HP100 r0")
    for parameter in reference_adapter.parameters():
        parameter.requires_grad_(False)
    reference_adapter.eval()
    task = PORT.SFMHP100ExpansionTask(
        scene_profile=inputs["preflight"]["scene"]["scene_profile"],
        scenario_start=int(inputs["preflight"]["scenario_start"]),
    ).attach_context_encoder(reference_adapter.policy)
    keys = HYBRID._lineage_keys(config)
    gpu = BASE._gpu_contract(args.device, args.physical_gpu)

    def adapter_factory(scope: str):
        adapter, _, _ = _load_adapter(selected, scope, args.device)
        return adapter

    output.mkdir(parents=True)
    with HYBRID._OrderedSidecarVerifier(task, int(args.verifier_workers)) as verifier:
        baseline_adapter = adapter_factory("head_only")
        baseline_rerun = HYBRID.raw_only_recheck(
            baseline_adapter, task, keys=keys, config=config,
            microcycle=0, verifier=verifier,
        )
        baseline_canonical = ITER.canonical_sha256(baseline_rerun)
        expected_baseline = inputs["marker"]["baseline_raw_gate_canonical_sha256"]
        if baseline_canonical != expected_baseline:
            raise RuntimeError("selected checkpoint raw CRN baseline did not reproduce")

        preflight = {
            "status": PREFLIGHT_STATUS,
            "version": VERSION,
            "source_gate": source_gate,
            "source_components": {
                Path(path).name: HYBRID._sha256(path)
                for path in (
                    __file__, GPS.__file__, PORT.__file__, BASE.__file__,
                    CONT.__file__, SWEEP.__file__, HYBRID.__file__, ITER.__file__,
                )
            },
            "iterative_input": {
                key: inputs[key] for key in (
                    "root", "iterative_marker", "selected_checkpoint",
                    "cycle", "cycle_marker", "training_view_artifact",
                    "training_view_canonical_sha256", "baseline_raw_artifact",
                    "reference_checkpoint",
                )
            },
            "config": asdict(config),
            "dose_family": args.dose_family,
            "doses": [asdict(dose) for dose in doses],
            "scopes": list(SCOPES),
            "same_checkpoint_view_and_raw_CRN_for_every_cell": True,
            "baseline_rerun_canonical_sha256": baseline_canonical,
            "negative_source": "Ncausal_only",
            "Dminus_training_exposures": 0,
            "gpu": gpu,
        }
        preflight_path = output / "PREFLIGHT.json"
        _write_json(preflight_path, preflight)

        def evaluate(adapter, cell_name):
            raw = HYBRID.raw_only_recheck(
                adapter, task, keys=keys, config=config,
                microcycle=int(args.cycle), verifier=verifier,
            )
            return raw, ITER._raw_gate_summary(raw, keys)

        result = evaluate_cells(
            adapter_factory=adapter_factory,
            training_view=inputs["training_view"],
            baseline_summary=inputs["baseline_summary"],
            evaluate_fn=evaluate,
            output=output,
            parent_checkpoint=selected,
            cycle_marker_sha256=inputs["cycle_marker"]["sha256"],
            batch_size=int(config.batch_size),
            max_relative_parameter_drift=float(config.max_relative_parameter_drift),
            seed=int(config.seed),
            doses=doses,
            raw_crn_reference=baseline_rerun,
        )
    result.update(
        dose_family=args.dose_family,
        preflight=str(preflight_path),
        preflight_sha256=HYBRID._sha256(preflight_path),
        source_HEAD=source_gate["HEAD"],
    )
    _write_json(output / "SWEEP_COMPLETE.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--repo-root", required=True)
    value.add_argument("--expected-source-commit", required=True)
    value.add_argument("--iterative-root", required=True)
    value.add_argument("--expected-iterative-marker-sha256", required=True)
    value.add_argument("--cycle", required=True, type=int)
    value.add_argument(
        "--dose-family", choices=DOSE_FAMILIES, default="original",
    )
    value.add_argument("--output", required=True)
    value.add_argument("--device", default="cuda:0")
    value.add_argument("--physical-gpu", required=True, type=int)
    value.add_argument("--verifier-workers", type=int, default=32)
    return value


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if int(args.cycle) < 1 or int(args.verifier_workers) < 1:
        raise ValueError("cycle and verifier workers must be positive")
    result = run(args)
    print(json.dumps({"status": result["status"], "output": args.output}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
