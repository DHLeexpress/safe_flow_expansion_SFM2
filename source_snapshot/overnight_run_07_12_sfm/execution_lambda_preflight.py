"""Task-agnostic calibration for ``J - lambda * one_step_margin``.

The input is a JSON or JSONL collection with one row per candidate.  This
module deliberately does not know about a simulator, policy, or verifier: it
only consumes already-audited candidate measurements.  Selection is restricted
to rows for which both ``exact_positive`` and ``execution_eligible`` are true.

The robust reference scale is

    lambda0 = median_context(max(J) - min(J))
              / median_context_nonzero(max(m) - min(m)).

Only contexts with at least two eligible candidates contribute to the scale.
Contexts with no eligible candidate remain NVP; the preflight never invents a
fallback candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_MULTIPLIERS = (0.0, 0.5, 1.0, 2.0, 4.0)


@dataclass(frozen=True)
class FieldMap:
    gamma: str = "gamma"
    context_id: str = "context_id"
    candidate_id: str = "candidate_id"
    exact_positive: str = "exact_positive"
    execution_eligible: str = "execution_eligible"
    native_cost: str = "native_cost"
    step_margin: str = "step_margin"


@dataclass(frozen=True)
class Candidate:
    gamma: float
    context_id: str | int
    candidate_id: str | int
    exact_positive: bool
    execution_eligible: bool
    native_cost: float
    step_margin: float
    raw_record_index: int

    @property
    def eligible(self) -> bool:
        return self.exact_positive and self.execution_eligible


def _strict_json_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _json_loads(text: str, *, source: str) -> Any:
    try:
        return json.loads(text, object_pairs_hook=_strict_json_object)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON in {source}: {exc}") from exc


def load_candidate_records(path: str | os.PathLike[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load flat candidate rows and return them with an exact parsed-input copy."""

    source = Path(path)
    raw_bytes = source.read_bytes()
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"input is not UTF-8: {source}") from exc
    if not text.strip():
        raise ValueError(f"input is empty: {source}")

    if source.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            value = _json_loads(line, source=f"{source}:{line_number}")
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number} must contain one JSON object")
            records.append(value)
        payload: Any = records
        input_format = "jsonl"
    else:
        payload = _json_loads(text, source=str(source))
        input_format = "json"
        if isinstance(payload, list):
            records = payload
        elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
            records = payload["records"]
        else:
            raise ValueError("JSON input must be a list or an object containing a 'records' list")
        if not all(isinstance(row, dict) for row in records):
            raise ValueError("every candidate record must be a JSON object")

    if not records:
        raise ValueError("input contains no candidate records")
    return list(records), {
        "format": input_format,
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "payload": payload,
    }


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite JSON number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _strict_bool(value: Any, *, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a JSON boolean")
    return value


def _scalar_id(value: Any, *, label: str) -> str | int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError(f"{label} must be a string or integer")
    if isinstance(value, str) and not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _id_key(value: str | int) -> tuple[str, str]:
    return type(value).__name__, str(value)


def _validate_declared_gammas(values: Sequence[float] | None, atol: float) -> tuple[float, ...] | None:
    if values is None:
        return None
    result = tuple(_finite_number(value, label="declared gamma") for value in values)
    if not result:
        raise ValueError("declared gamma list must not be empty")
    for index, gamma in enumerate(result):
        for previous in result[:index]:
            if math.isclose(gamma, previous, rel_tol=0.0, abs_tol=atol):
                raise ValueError(f"declared gammas {previous} and {gamma} overlap at atol={atol}")
    return result


def _canonical_gamma(value: Any, declared: tuple[float, ...] | None, atol: float) -> float:
    gamma = _finite_number(value, label="gamma")
    if declared is not None:
        matches = [item for item in declared if math.isclose(gamma, item, rel_tol=0.0, abs_tol=atol)]
        if len(matches) != 1:
            raise ValueError(
                f"gamma {gamma:.12g} matches {len(matches)} declared values at atol={atol}"
            )
        return matches[0]

    digits = max(0, int(math.ceil(-math.log10(atol)))) if atol < 1.0 else 0
    rounded = float(round(gamma, digits))
    if math.isclose(gamma, rounded, rel_tol=0.0, abs_tol=atol):
        return 0.0 if rounded == 0.0 else rounded
    return gamma


def normalize_candidates(
    records: Sequence[Mapping[str, Any]],
    *,
    fields: FieldMap = FieldMap(),
    declared_gammas: Sequence[float] | None = None,
    gamma_atol: float = 1.0e-6,
) -> list[Candidate]:
    if not math.isfinite(gamma_atol) or gamma_atol <= 0.0:
        raise ValueError("gamma_atol must be finite and positive")
    declared = _validate_declared_gammas(declared_gammas, gamma_atol)
    field_names = tuple(asdict(fields).values())
    if len(set(field_names)) != len(field_names):
        raise ValueError("field mapping contains duplicate input field names")

    candidates: list[Candidate] = []
    seen: set[tuple[float, tuple[str, str], tuple[str, str]]] = set()
    seen_gammas: set[float] = set()
    for index, row in enumerate(records):
        missing = [name for name in field_names if name not in row]
        if missing:
            raise ValueError(f"record {index} is missing required fields: {', '.join(missing)}")
        gamma = _canonical_gamma(row[fields.gamma], declared, gamma_atol)
        context_id = _scalar_id(row[fields.context_id], label=f"record {index} context_id")
        candidate_id = _scalar_id(row[fields.candidate_id], label=f"record {index} candidate_id")
        exact_positive = _strict_bool(
            row[fields.exact_positive], label=f"record {index} exact_positive"
        )
        execution_eligible = _strict_bool(
            row[fields.execution_eligible], label=f"record {index} execution_eligible"
        )
        if execution_eligible and not exact_positive:
            raise ValueError(
                f"record {index} is execution-eligible but not exact-positive; refusing inconsistent gate"
            )
        candidate = Candidate(
            gamma=gamma,
            context_id=context_id,
            candidate_id=candidate_id,
            exact_positive=exact_positive,
            execution_eligible=execution_eligible,
            native_cost=_finite_number(row[fields.native_cost], label=f"record {index} native_cost"),
            step_margin=_finite_number(row[fields.step_margin], label=f"record {index} step_margin"),
            raw_record_index=index,
        )
        identity = (gamma, _id_key(context_id), _id_key(candidate_id))
        if identity in seen:
            raise ValueError(
                f"duplicate candidate_id {candidate_id!r} in gamma={gamma}, context={context_id!r}"
            )
        seen.add(identity)
        seen_gammas.add(gamma)
        candidates.append(candidate)

    if declared is not None:
        missing_gammas = [gamma for gamma in declared if gamma not in seen_gammas]
        if missing_gammas:
            raise ValueError(f"no records found for declared gammas: {missing_gammas}")
    return candidates


def _group_contexts(candidates: Iterable[Candidate]) -> list[tuple[float, str | int, list[Candidate]]]:
    grouped: dict[tuple[float, tuple[str, str]], tuple[str | int, list[Candidate]]] = {}
    for candidate in candidates:
        key = candidate.gamma, _id_key(candidate.context_id)
        if key not in grouped:
            grouped[key] = candidate.context_id, []
        grouped[key][1].append(candidate)
    result = [
        (gamma, context_id, sorted(rows, key=lambda row: (_id_key(row.candidate_id), row.raw_record_index)))
        for (gamma, _), (context_id, rows) in grouped.items()
    ]
    return sorted(result, key=lambda item: (item[0], _id_key(item[1])))


def calibrate_lambda(candidates: Sequence[Candidate]) -> dict[str, Any]:
    cost_spans: list[float] = []
    nonzero_margin_spans: list[float] = []
    context_spans: list[dict[str, Any]] = []
    for gamma, context_id, rows in _group_contexts(candidates):
        eligible = [row for row in rows if row.eligible]
        if len(eligible) < 2:
            continue
        cost_span = max(row.native_cost for row in eligible) - min(row.native_cost for row in eligible)
        margin_span = max(row.step_margin for row in eligible) - min(row.step_margin for row in eligible)
        cost_spans.append(cost_span)
        if margin_span > 0.0:
            nonzero_margin_spans.append(margin_span)
        context_spans.append({
            "gamma": gamma,
            "context_id": context_id,
            "eligible_candidates": len(eligible),
            "native_cost_span": cost_span,
            "step_margin_span": margin_span,
        })
    if not cost_spans:
        raise ValueError("lambda calibration needs at least one context with two eligible candidates")
    if not nonzero_margin_spans:
        raise ValueError("lambda calibration found no nonzero eligible step-margin span")
    median_cost_span = float(statistics.median(cost_spans))
    median_margin_span = float(statistics.median(nonzero_margin_spans))
    if median_cost_span <= 0.0:
        raise ValueError("median eligible native-cost span is zero; lambda scale is not identifiable")
    lambda0 = median_cost_span / median_margin_span
    if not math.isfinite(lambda0) or lambda0 <= 0.0:
        raise ValueError("computed lambda0 is not finite and positive")
    return {
        "formula": "median(per_context_native_cost_span) / median(nonzero_per_context_step_margin_span)",
        "lambda0": lambda0,
        "median_native_cost_span": median_cost_span,
        "median_nonzero_step_margin_span": median_margin_span,
        "native_cost_span_contexts": len(cost_spans),
        "nonzero_step_margin_span_contexts": len(nonzero_margin_spans),
        "context_spans": context_spans,
    }


def _select(rows: Sequence[Candidate], *, mode: str, lambda_value: float = 0.0) -> Candidate:
    if not rows:
        raise ValueError("cannot select from an empty candidate set")
    if mode == "cost":
        key = lambda row: (row.native_cost, _id_key(row.candidate_id), row.raw_record_index)
    elif mode == "margin":
        key = lambda row: (-row.step_margin, _id_key(row.candidate_id), row.raw_record_index)
    elif mode == "blend":
        key = lambda row: (
            row.native_cost - lambda_value * row.step_margin,
            _id_key(row.candidate_id),
            row.raw_record_index,
        )
    else:
        raise ValueError(f"unknown selection mode {mode!r}")
    return min(rows, key=key)


def _summary(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "min": None, "max": None}
    return {
        "mean": float(statistics.fmean(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def _gamma_key(gamma: float) -> str:
    return f"{gamma:.12g}"


def evaluate_multipliers(
    candidates: Sequence[Candidate],
    *,
    lambda0: float,
    multipliers: Sequence[float] = DEFAULT_MULTIPLIERS,
) -> list[dict[str, Any]]:
    if not math.isfinite(lambda0) or lambda0 <= 0.0:
        raise ValueError("lambda0 must be finite and positive")
    parsed_multipliers = tuple(_finite_number(value, label="multiplier") for value in multipliers)
    if not parsed_multipliers or any(value < 0.0 for value in parsed_multipliers):
        raise ValueError("multipliers must be a nonempty sequence of nonnegative values")
    if len(set(parsed_multipliers)) != len(parsed_multipliers):
        raise ValueError("multipliers must be unique")

    contexts = _group_contexts(candidates)
    results: list[dict[str, Any]] = []
    for multiplier in parsed_multipliers:
        lambda_value = lambda0 * multiplier
        selections: list[dict[str, Any]] = []
        nvp_contexts: list[dict[str, Any]] = []
        gamma_rows: dict[float, dict[str, Any]] = {}
        cost_agreement = 0
        margin_agreement = 0
        selected_total = 0
        for gamma, context_id, rows in contexts:
            stats = gamma_rows.setdefault(gamma, {
                "contexts": 0,
                "candidate_rows": 0,
                "exact_positive_candidates": 0,
                "execution_eligible_candidates": 0,
                "eligible_candidates": 0,
                "selected": [],
                "nvp": 0,
                "cost_agreement": 0,
                "margin_agreement": 0,
            })
            stats["contexts"] += 1
            stats["candidate_rows"] += len(rows)
            stats["exact_positive_candidates"] += sum(row.exact_positive for row in rows)
            stats["execution_eligible_candidates"] += sum(row.execution_eligible for row in rows)
            eligible = [row for row in rows if row.eligible]
            stats["eligible_candidates"] += len(eligible)
            if not eligible:
                stats["nvp"] += 1
                nvp_contexts.append({"gamma": gamma, "context_id": context_id})
                continue

            pure_cost = _select(eligible, mode="cost")
            pure_margin = _select(eligible, mode="margin")
            selected = _select(eligible, mode="blend", lambda_value=lambda_value)
            agrees_cost = selected.candidate_id == pure_cost.candidate_id
            agrees_margin = selected.candidate_id == pure_margin.candidate_id
            cost_agreement += int(agrees_cost)
            margin_agreement += int(agrees_margin)
            selected_total += 1
            stats["cost_agreement"] += int(agrees_cost)
            stats["margin_agreement"] += int(agrees_margin)
            stats["selected"].append(selected)
            selections.append({
                "gamma": gamma,
                "context_id": context_id,
                "candidate_id": selected.candidate_id,
                "raw_record_index": selected.raw_record_index,
                "eligible_candidates": len(eligible),
                "native_cost": selected.native_cost,
                "step_margin": selected.step_margin,
                "blended_score": selected.native_cost - lambda_value * selected.step_margin,
                "pure_cost_candidate_id": pure_cost.candidate_id,
                "pure_margin_candidate_id": pure_margin.candidate_id,
                "agrees_with_pure_cost": agrees_cost,
                "agrees_with_pure_margin": agrees_margin,
            })

        per_gamma: dict[str, Any] = {}
        for gamma in sorted(gamma_rows):
            stats = gamma_rows[gamma]
            selected_rows: list[Candidate] = stats.pop("selected")
            count = len(selected_rows)
            contexts_count = stats["contexts"]
            per_gamma[_gamma_key(gamma)] = {
                "gamma": gamma,
                **stats,
                "selected_contexts": count,
                "nvp_rate": stats["nvp"] / contexts_count,
                "agreement_with_pure_cost": stats["cost_agreement"] / count if count else None,
                "agreement_with_pure_margin": stats["margin_agreement"] / count if count else None,
                "selected_native_cost": _summary([row.native_cost for row in selected_rows]),
                "selected_step_margin": _summary([row.step_margin for row in selected_rows]),
            }
        results.append({
            "multiplier": multiplier,
            "lambda": lambda_value,
            "score": "native_cost - lambda * step_margin",
            "context_count": len(contexts),
            "selected_contexts": selected_total,
            "nvp_contexts": len(nvp_contexts),
            "nvp_rate": len(nvp_contexts) / len(contexts),
            "agreement_with_pure_cost": cost_agreement / selected_total if selected_total else None,
            "agreement_with_pure_margin": margin_agreement / selected_total if selected_total else None,
            "selections": selections,
            "nvp": nvp_contexts,
            "per_gamma": per_gamma,
        })
    return results


def build_report(
    records: Sequence[Mapping[str, Any]],
    *,
    raw_input: Mapping[str, Any],
    fields: FieldMap = FieldMap(),
    declared_gammas: Sequence[float] | None = None,
    gamma_atol: float = 1.0e-6,
    multipliers: Sequence[float] = DEFAULT_MULTIPLIERS,
) -> dict[str, Any]:
    candidates = normalize_candidates(
        records,
        fields=fields,
        declared_gammas=declared_gammas,
        gamma_atol=gamma_atol,
    )
    calibration = calibrate_lambda(candidates)
    return {
        "status": "EXECUTION_LAMBDA_PREFLIGHT_COMPLETE",
        "protocol": {
            "eligible_gate": "exact_positive AND execution_eligible",
            "nvp_rule": "no eligible candidate; no fallback",
            "selection_score": "native_cost - lambda * step_margin",
            "field_map": asdict(fields),
            "declared_gammas": list(declared_gammas) if declared_gammas is not None else None,
            "gamma_atol": gamma_atol,
            "multipliers": list(multipliers),
        },
        "record_count": len(records),
        "calibration": calibration,
        "sweep": evaluate_multipliers(
            candidates,
            lambda0=calibration["lambda0"],
            multipliers=multipliers,
        ),
        "raw_input": dict(raw_input),
    }


def _parse_csv_floats(value: str, *, label: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be comma-separated floats") from exc
    if not result:
        raise argparse.ArgumentTypeError(f"{label} must not be empty")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="Flat candidate JSON or JSONL")
    parser.add_argument("--output", required=True, help="Output report JSON")
    parser.add_argument("--declared-gammas", help="Comma-separated canonical gamma values")
    parser.add_argument("--gamma-atol", type=float, default=1.0e-6)
    parser.add_argument("--multipliers", default="0,0.5,1,2,4")
    for name, default in asdict(FieldMap()).items():
        parser.add_argument(f"--{name.replace('_', '-')}-field", default=default)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    declared = (
        _parse_csv_floats(args.declared_gammas, label="declared gammas")
        if args.declared_gammas is not None
        else None
    )
    multipliers = _parse_csv_floats(args.multipliers, label="multipliers")
    fields = FieldMap(**{
        name: getattr(args, f"{name}_field") for name in asdict(FieldMap())
    })
    records, raw_input = load_candidate_records(args.input)
    report = build_report(
        records,
        raw_input=raw_input,
        fields=fields,
        declared_gammas=declared,
        gamma_atol=args.gamma_atol,
        multipliers=multipliers,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
