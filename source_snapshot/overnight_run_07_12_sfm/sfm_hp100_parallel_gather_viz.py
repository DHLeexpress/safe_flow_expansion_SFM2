"""Trace-only visualization for one HP100 parallel-gather retry batch.

This module deliberately does not import a verifier, policy, simulator, or
scene generator.  It renders only positions, labels, and certificate faces
that were recorded by the expansion process.  Consequently, a plot cannot
silently re-solve a candidate under different dynamics or verifier settings.

Trace schema (``sfm_hp100_parallel_gather_trace_v1``)
-----------------------------------------------------
The input may be one JSON object with ``metadata`` and ``events`` keys, or
JSONL with one metadata record followed by event records.  Metadata contains
``K=16``, ``B=4``, ``H=10``, ``parallel_episodes=16``, and a non-empty
``verifier_contract``.  Every event is identified by

``(round, gamma, retry_batch, replica, lineage_id, step)``.

An event records ``state``, ``ped_xy``, ``all_K`` candidate segments,
``selected_ids``, four ``query_rows`` with the already-computed exact labels
and faces, ``executed_id`` (or ``null`` for NVP), ``episode_status``, and
``committed_success``.  ``scenario_id`` is required as provenance but is not
part of the lineage identity: all 16 panels may share a diagnostic scenario,
or they may be 16 independently sampled training scenarios.  A query may record
``execution_eligible`` separately from its exact label.  If omitted, exact
positivity is treated as execution eligibility.

The renderer writes a 4x4 MP4, a candidate-specific B=4 PNG/PDF, a committed
summary, and a hash-authenticated ``TRACE_COMPLETE.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
import numpy as np


SCHEMA_VERSION = "sfm_hp100_parallel_gather_trace_v1"
TRACE_COMPLETE_STATUS = "SFM_HP100_PARALLEL_GATHER_TRACE_COMPLETE"
TERMINAL_STATUSES = {"nvp", "success", "collision", "timeout", "cutoff"}

GRAY = "#8C8C8C"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D62728"
BLUE = "#0072B2"
BLACK = "#111111"
GOLD = "#D4A017"

IDENTITY_FIELDS = (
    "round", "gamma", "retry_batch", "replica", "lineage_id", "step",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path):
    with path.open() as stream:
        return json.load(stream)


def load_trace(path: str | Path) -> tuple[dict, list[dict]]:
    """Load the declared JSON or JSONL trace without interpreting geometry."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open() as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSONL record at line {line_number}: {error}"
                    ) from error
        if not rows:
            raise ValueError("empty trace JSONL")
        metadata = None
        events = []
        for row in rows:
            record_type = row.get("type")
            if record_type == "metadata" or (
                    "metadata" in row and not set(IDENTITY_FIELDS) <= set(row)):
                if metadata is not None:
                    raise ValueError("trace JSONL contains multiple metadata records")
                metadata = dict(row.get("metadata", {
                    key: value for key, value in row.items() if key != "type"
                }))
            elif record_type == "event":
                events.append(dict(row.get("event", {
                    key: value for key, value in row.items() if key != "type"
                })))
            elif set(IDENTITY_FIELDS) <= set(row):
                events.append(dict(row))
            else:
                raise ValueError("untyped JSONL record is neither metadata nor event")
        if metadata is None:
            raise ValueError("trace JSONL has no metadata record")
        return metadata, events

    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError("trace JSON must be an object")
    metadata = payload.get("metadata")
    events = payload.get("events")
    if not isinstance(metadata, dict) or not isinstance(events, list):
        raise ValueError("trace JSON requires object metadata and list events")
    return dict(metadata), [dict(event) for event in events]


def _meta(metadata: dict, key: str):
    if key in metadata:
        return metadata[key]
    config = metadata.get("config", {})
    if key in config:
        return config[key]
    raise ValueError(f"trace metadata is missing {key!r}")


def event_identity(event: dict) -> tuple[int, float, int, int, str, int]:
    missing = [key for key in IDENTITY_FIELDS if key not in event]
    if missing:
        raise ValueError(f"trace event is missing identity fields: {missing}")
    values = (
        int(event["round"]), round(float(event["gamma"]), 6),
        int(event["retry_batch"]), int(event["replica"]),
        str(event["lineage_id"]), int(event["step"]),
    )
    if not math.isfinite(values[1]):
        raise ValueError("event gamma must be finite")
    return values


def _segment(value, *, H: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.shape != (H + 1, 2) or not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must be a finite [{H + 1},2] segment")
    return array


def _state(value) -> np.ndarray:
    array = np.asarray(value, dtype=float).reshape(-1)
    if len(array) < 2 or not np.all(np.isfinite(array)):
        raise ValueError("state must contain at least two finite values")
    return array


def _pedestrians(value) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.size == 0:
        return np.empty((0, 2), dtype=float)
    if array.ndim != 2 or array.shape[1] != 2 or not np.all(np.isfinite(array)):
        raise ValueError("ped_xy must be a finite [N,2] array")
    return array


def _candidate_map(event: dict, *, H: int, K: int) -> dict[int, dict]:
    rows = event.get("all_K")
    if not isinstance(rows, list) or len(rows) != K:
        raise ValueError(f"all_K must contain exactly K={K} candidates")
    result = {}
    for row in rows:
        candidate_id = int(row["candidate_id"])
        if candidate_id in result:
            raise ValueError(f"duplicate candidate_id {candidate_id}")
        _segment(row.get("segment"), H=H, label=f"candidate {candidate_id}")
        result[candidate_id] = row
    return result


def _query_map(event: dict, *, H: int, B: int, candidates: dict) -> dict[int, dict]:
    selected = list(map(int, event.get("selected_ids", ())))
    if len(selected) != B or len(set(selected)) != B:
        raise ValueError(f"selected_ids must contain exactly B={B} unique IDs")
    if not set(selected) <= set(candidates):
        raise ValueError("selected_ids contains an ID absent from all_K")
    rows = event.get("query_rows")
    if not isinstance(rows, list) or len(rows) != B:
        raise ValueError(f"query_rows must contain exactly B={B} rows")
    result = {}
    for row in rows:
        candidate_id = int(row["candidate_id"])
        if candidate_id in result or candidate_id not in selected:
            raise ValueError("query_rows does not map one-to-one to selected_ids")
        certificate = row.get("result")
        if not isinstance(certificate, dict) or not bool(certificate.get("resolved")):
            raise ValueError("renderer requires every B query to have a resolved label")
        label = int(certificate.get("y", -1))
        if label not in (0, 1):
            raise ValueError("resolved B query label must be zero or one")
        if (not bool(certificate.get("full_h"))
                or int(certificate.get("terminal_step", -1)) != H):
            raise ValueError("renderer requires exact full-H query records")
        recorded = _segment(
            certificate.get("segment"), H=H,
            label=f"query result {candidate_id}",
        )
        generated = _segment(
            candidates[candidate_id].get("segment"), H=H,
            label=f"candidate {candidate_id}",
        )
        if not np.allclose(recorded, generated, rtol=0.0, atol=1.0e-7):
            raise ValueError("query result segment differs from its generated candidate")
        eligible = bool(row.get("execution_eligible", label == 1))
        if eligible and label != 1:
            raise ValueError("an exact-negative query cannot be execution eligible")
        result[candidate_id] = row
    if set(result) != set(selected):
        raise ValueError("query_rows does not cover selected_ids exactly")
    return result


def _validate_faces(query: dict, *, expected_artificial_faces: int) -> None:
    certificate = query["result"]
    if int(certificate["y"]) != 1:
        return
    faces = certificate.get("faces")
    if not isinstance(faces, list) or not faces:
        raise ValueError("exact-positive query is missing stored verifier faces")
    artificial = 0
    for index, face in enumerate(faces):
        if not isinstance(face, dict):
            raise ValueError("stored verifier faces must be plain dictionaries")
        normal = np.asarray(face.get("a"), dtype=float)
        margin = float(face.get("m", float("nan")))
        if normal.shape != (2,) or not np.all(np.isfinite(normal)):
            raise ValueError(f"face {index} has an invalid 2-D normal")
        if not math.isfinite(margin) or not bool(face.get("feasible", True)):
            raise ValueError("exact-positive query has an infeasible stored face")
        artificial += str(face.get("kind", "")) == "artificial"
    if artificial != expected_artificial_faces:
        raise ValueError(
            "exact-positive query has "
            f"{artificial} artificial faces, expected {expected_artificial_faces}"
        )


def validate_trace(metadata: dict, events: Iterable[dict]) -> list[dict]:
    """Fail closed on identity, K/B, label, execution, and lineage invariants."""
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"trace schema must be {SCHEMA_VERSION!r}")
    K, B, H = (int(_meta(metadata, key)) for key in ("K", "B", "H"))
    parallel = int(_meta(metadata, "parallel_episodes"))
    if (K, B, H, parallel) != (16, 4, 10, 16):
        raise ValueError("renderer contract requires K=16, B=4, H=10, replicas=16")
    if not isinstance(metadata.get("verifier_contract"), dict) \
            or not metadata["verifier_contract"]:
        raise ValueError("trace metadata lacks a verifier contract")
    expected_artificial = int(metadata.get("expected_artificial_faces", 16))
    if expected_artificial != 16:
        raise ValueError("SFM trace requires exactly 16 artificial outer faces")
    goal = metadata.get("goal")
    if goal is not None:
        goal = np.asarray(goal, dtype=float)
        if goal.shape != (2,) or not np.all(np.isfinite(goal)):
            raise ValueError("declared goal must be a finite 2-D point")

    events = [dict(event) for event in events]
    if not events:
        raise ValueError("trace contains no events")
    identities = [event_identity(event) for event in events]
    if len(set(identities)) != len(identities):
        raise ValueError("trace contains a duplicate full event identity")

    groups: dict[tuple[int, float, int, int, str], list[dict]] = {}
    for event in events:
        identity = event_identity(event)
        if identity[0] < 0 or identity[2] < 0 or identity[3] < 0 or identity[5] < 0:
            raise ValueError("round, retry_batch, replica, and step must be nonnegative")
        _state(event.get("state"))
        _pedestrians(event.get("ped_xy", ()))
        if "scenario_id" not in event or str(event["scenario_id"]) == "":
            raise ValueError("each trace event requires a non-empty scenario_id")
        candidates = _candidate_map(event, H=H, K=K)
        queries = _query_map(event, H=H, B=B, candidates=candidates)
        for query in queries.values():
            _validate_faces(query, expected_artificial_faces=expected_artificial)

        eligible = {
            candidate_id for candidate_id, query in queries.items()
            if bool(query.get(
                "execution_eligible", int(query["result"]["y"]) == 1,
            ))
        }
        executed = event.get("executed_id")
        status = str(event.get("episode_status", ""))
        if status not in {"active", *TERMINAL_STATUSES}:
            raise ValueError(f"invalid episode_status {status!r}")
        if status == "cutoff" and not (
            metadata.get("diagnostic_only") is True
            and metadata.get("enters_replay") is False
        ):
            raise ValueError(
                "cutoff is allowed only for diagnostic_only=true, enters_replay=false"
            )
        if executed is None:
            if eligible or status != "nvp":
                raise ValueError(
                    "executed_id is null iff no B query is execution eligible and status=NVP"
                )
        else:
            executed = int(executed)
            if executed not in eligible:
                raise ValueError("executed_id is not an eligible exact-positive B query")
            if status == "nvp":
                raise ValueError("NVP event cannot execute a candidate")
        committed = event.get("committed_success")
        if not isinstance(committed, bool):
            raise ValueError("committed_success must be an explicit boolean")
        if committed and status not in {"active", "success"}:
            raise ValueError("a committed-success lineage has a non-success outcome")
        key = identity[:5]
        groups.setdefault(key, []).append(event)

    by_batch: dict[tuple[int, float, int], set[tuple[int, str]]] = {}
    for key, rows in groups.items():
        steps = sorted(int(row["step"]) for row in rows)
        if steps != list(range(steps[-1] + 1)):
            raise ValueError(f"lineage {key} has non-contiguous or nonzero-based steps")
        ordered = sorted(rows, key=lambda row: int(row["step"]))
        if any(str(row["episode_status"]) != "active" for row in ordered[:-1]):
            raise ValueError(f"lineage {key} terminates before its final event")
        if str(ordered[-1]["episode_status"]) not in TERMINAL_STATUSES:
            raise ValueError(f"lineage {key} lacks a terminal final event")
        flags = {bool(row["committed_success"]) for row in rows}
        if len(flags) != 1:
            raise ValueError(f"lineage {key} changes committed_success mid-trajectory")
        if True in flags and str(ordered[-1]["episode_status"]) != "success":
            raise ValueError("committed_success lineage does not end in success")
        batch = key[:3]
        member = (key[3], key[4])
        if member in by_batch.setdefault(batch, set()):
            raise ValueError("retry batch repeats a replica/lineage identity")
        by_batch[batch].add(member)
    for batch, members in by_batch.items():
        replicas = [replica for replica, _ in members]
        if len(members) != parallel or len(set(replicas)) != parallel:
            raise ValueError(
                f"batch {batch} has {len(members)} lineages; expected {parallel} replicas"
            )
    return sorted(events, key=event_identity)


def select_batch(
    events: Iterable[dict], *, round_index: int, gamma: float, retry_batch: int,
) -> list[dict]:
    selected = [
        event for event in events
        if int(event["round"]) == int(round_index)
        and abs(float(event["gamma"]) - float(gamma)) <= 5.0e-7
        and int(event["retry_batch"]) == int(retry_batch)
    ]
    if not selected:
        raise ValueError("requested round/gamma/retry batch is absent from the trace")
    return selected


def _query_status(query: dict) -> str:
    return "positive" if int(query["result"]["y"]) == 1 else "negative"


def _queries(event: dict) -> dict[int, dict]:
    return {int(row["candidate_id"]): row for row in event["query_rows"]}


def _candidates(event: dict) -> dict[int, dict]:
    return {int(row["candidate_id"]): row for row in event["all_K"]}


def _lineages(events: Iterable[dict]) -> dict[tuple[int, str], list[dict]]:
    groups = {}
    for event in events:
        key = (int(event["replica"]), str(event["lineage_id"]))
        groups.setdefault(key, []).append(event)
    return {
        key: sorted(rows, key=lambda row: int(row["step"]))
        for key, rows in groups.items()
    }


def committed_summary(events: Iterable[dict]) -> dict:
    events = list(events)
    groups = _lineages(events)
    outcomes = {status: 0 for status in sorted(TERMINAL_STATUSES)}
    for rows in groups.values():
        outcomes[str(rows[-1]["episode_status"])] += 1
    positives = sum(
        int(query["result"]["y"]) == 1
        for event in events for query in event["query_rows"]
    )
    queried = sum(len(event["query_rows"]) for event in events)
    committed = sum(
        bool(rows[0]["committed_success"]) for rows in groups.values()
    )
    scenario_ids = sorted({str(event["scenario_id"]) for event in events})
    lineage_rows = []
    for (replica, lineage_id), rows in sorted(groups.items()):
        lineage_rows.append(dict(
            replica=int(replica), lineage_id=str(lineage_id),
            scenario_id=str(rows[0]["scenario_id"]),
            steps=len(rows), outcome=str(rows[-1]["episode_status"]),
            committed_success=bool(rows[0]["committed_success"]),
        ))
    return dict(
        lineages=len(groups), events=len(events), K_generated=len(events) * 16,
        B_queried=queried, exact_positive=positives,
        exact_negative=queried - positives,
        executed=sum(event.get("executed_id") is not None for event in events),
        outcomes=outcomes, committed_success_lineages=committed,
        committed_success_events=sum(
            bool(event["committed_success"]) for event in events
        ),
        scenario_ids=scenario_ids,
        panel_scenario_mode=(
            "fixed_scenario_diagnostic" if len(scenario_ids) == 1
            else "distinct_training_scenarios"
        ),
        lineage_rows=lineage_rows,
    )


def halfspace_polygon(A, b, *, tolerance: float = 1.0e-7):
    """Intersect stored 2-D halfspaces; this performs no verification."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[1] != 2 or b.shape != (len(A),):
        raise ValueError("expected stored A[m,2] and b[m]")
    vertices = []
    for first in range(len(A)):
        for second in range(first + 1, len(A)):
            matrix = np.stack((A[first], A[second]))
            if abs(float(np.linalg.det(matrix))) < 1.0e-10:
                continue
            point = np.linalg.solve(matrix, np.array((b[first], b[second])))
            if np.all(A @ point <= b + tolerance):
                vertices.append(point)
    if len(vertices) < 3:
        return None
    vertices = np.unique(np.round(np.asarray(vertices), decimals=9), axis=0)
    if len(vertices) < 3:
        return None
    center = vertices.mean(axis=0)
    angle = np.arctan2(vertices[:, 1] - center[1], vertices[:, 0] - center[0])
    return vertices[np.argsort(angle)]


def _face_line_segment(normal, offset, bounds):
    """Clip one stored affine face line to the declared world-frame bounds."""
    normal = np.asarray(normal, dtype=float).reshape(2)
    xmin, xmax, ymin, ymax = map(float, bounds)
    points = []
    if abs(float(normal[1])) > 1.0e-12:
        for x in (xmin, xmax):
            y = (float(offset) - float(normal[0]) * x) / float(normal[1])
            if ymin - 1.0e-9 <= y <= ymax + 1.0e-9:
                points.append((x, y))
    if abs(float(normal[0])) > 1.0e-12:
        for y in (ymin, ymax):
            x = (float(offset) - float(normal[1]) * y) / float(normal[0])
            if xmin - 1.0e-9 <= x <= xmax + 1.0e-9:
                points.append((x, y))
    unique = []
    for point in points:
        if not any(np.linalg.norm(np.asarray(point) - np.asarray(other)) < 1.0e-8
                   for other in unique):
            unique.append(point)
    return np.asarray(unique[:2], dtype=float) if len(unique) >= 2 else None


def _active_face_mask(A, b, polygon, *, tolerance=2.0e-6):
    if polygon is None:
        return np.zeros(len(A), dtype=bool)
    residual = np.abs(np.asarray(polygon) @ np.asarray(A).T - np.asarray(b)[None])
    return np.any(residual <= float(tolerance), axis=0)


def _draw_stored_face_audit(axis, event: dict, query: dict, metadata: dict, bounds) -> None:
    """Draw every stored face, anchor/contact point, and the min-affine envelope."""
    certificate = query["result"]
    faces = [face for face in certificate.get("faces", ())
             if bool(face.get("feasible", True)) and float(face.get("m", 0.0)) > 0.0]
    if not faces:
        return
    center = np.asarray(certificate["segment"], dtype=float)[0]
    A = np.asarray([face["a"] for face in faces], dtype=float)
    margins = np.asarray([face["m"] for face in faces], dtype=float)
    offsets = A @ center + margins
    envelope = halfspace_polygon(A, offsets)
    active = _active_face_mask(A, offsets, envelope)

    diagnostics = certificate.get("diagnostics", {})
    sensing = float(diagnostics.get(
        "sensing_radius", diagnostics.get("R_eff", metadata.get("sensing_radius", 2.0)),
    ))
    rho_art = float(diagnostics.get("rho_art", .16))
    artificial_rows = [index for index, face in enumerate(faces)
                       if str(face.get("kind", "")) == "artificial"]
    anchor_margin = sensing * math.cos(math.pi / max(len(artificial_rows), 3))
    artificial_index = 0
    pedestrians = _pedestrians(event.get("ped_xy", ()))
    ped_radius = float(metadata.get("pedestrian_radius", .2))

    for index, face in enumerate(faces):
        kind = str(face.get("kind", ""))
        is_active = bool(active[index])
        segment = _face_line_segment(A[index], offsets[index], bounds)
        if segment is not None:
            line, = axis.plot(
                segment[:, 0], segment[:, 1],
                color=("#006D2C" if is_active else "#74C476"),
                lw=(1.15 if is_active else .55),
                ls=("-" if is_active else "--"),
                alpha=(.92 if is_active else .58), zorder=4,
            )
            line.set_gid(f"{kind}-face-{'active' if is_active else 'redundant'}")

        obstacle_center = None
        obstacle_radius = None
        if kind == "artificial":
            theta = 2.0 * math.pi * artificial_index / max(len(artificial_rows), 1)
            radial = np.array([math.cos(theta), math.sin(theta)], dtype=float)
            obstacle_center = center + (anchor_margin + rho_art) * radial
            obstacle_radius = rho_art
            anchor = Circle(
                obstacle_center, rho_art, facecolor="none",
                edgecolor=("#006D2C" if is_active else "#A1D99B"),
                lw=(.85 if is_active else .45), ls=("-" if is_active else "--"),
                alpha=(.88 if is_active else .58), zorder=4.5,
            )
            anchor.set_gid("artificial-anchor")
            axis.add_patch(anchor)
            axis.plot(*obstacle_center, marker=".", color="#238B45", ms=2.2, zorder=5)
            artificial_index += 1
        elif kind in {"real", "real-moving"}:
            label = str(face.get("label", ""))
            digits = "".join(character for character in label if character.isdigit())
            if digits and int(digits) < len(pedestrians):
                obstacle_center = pedestrians[int(digits)]
                obstacle_radius = ped_radius

        if obstacle_center is not None and obstacle_radius is not None:
            contact = np.asarray(obstacle_center) - float(obstacle_radius) * A[index]
            point, = axis.plot(
                contact[0], contact[1], marker="o", linestyle="None",
                markerfacecolor="#FFD92F", markeredgecolor="#006D2C",
                markeredgewidth=.45, ms=3.2, zorder=6,
            )
            point.set_gid("tangent-contact")

    if envelope is not None:
        closed = np.vstack((envelope, envelope[0]))
        line, = axis.plot(
            closed[:, 0], closed[:, 1], color="#00441B", lw=1.9,
            alpha=.95, zorder=5.5,
        )
        line.set_gid("min-affine-envelope")


def stored_level_polygons(event: dict, query: dict, *, H: int = 10):
    """Reconstruct h=1..H polygons only from the recorded exact faces."""
    certificate = query["result"]
    if int(certificate["y"]) != 1:
        return []
    faces = [face for face in certificate["faces"] if bool(face.get("feasible", True))]
    A = np.asarray([face["a"] for face in faces], dtype=float)
    margins = np.asarray([face["m"] for face in faces], dtype=float)
    center = np.asarray(certificate["segment"], dtype=float)[0]
    gamma = float(event["gamma"])
    polygons = []
    for horizon in range(1, H + 1):
        beta_h = 1.0 - (1.0 - gamma) ** horizon
        polygon = halfspace_polygon(A, A @ center + beta_h * margins)
        if polygon is None:
            raise ValueError(
                f"stored exact-positive faces do not bound level set h={horizon}"
            )
        polygons.append((horizon, polygon))
    return polygons


def _plot_bounds(metadata: dict, events: Iterable[dict]) -> tuple[float, float, float, float]:
    declared = metadata.get("plot_bounds")
    if declared is not None:
        values = tuple(map(float, declared))
        if len(values) != 4 or not all(map(math.isfinite, values)):
            raise ValueError("plot_bounds must be [xmin,xmax,ymin,ymax]")
        if values[0] >= values[1] or values[2] >= values[3]:
            raise ValueError("plot_bounds has non-positive width or height")
        return values
    points = []
    if metadata.get("goal") is not None:
        points.append(np.asarray(metadata["goal"], dtype=float))
    for event in events:
        points.append(_state(event["state"])[:2])
        pedestrians = _pedestrians(event.get("ped_xy", ()))
        if len(pedestrians):
            points.extend(pedestrians)
        for candidate in event["all_K"]:
            points.extend(np.asarray(candidate["segment"], dtype=float))
    points = np.asarray(points, dtype=float)
    lo, hi = points.min(axis=0), points.max(axis=0)
    width = max(float(np.max(hi - lo)), 1.0)
    center = (lo + hi) / 2.0
    pad = 0.08 * width
    return (
        float(center[0] - width / 2 - pad),
        float(center[0] + width / 2 + pad),
        float(center[1] - width / 2 - pad),
        float(center[1] + width / 2 + pad),
    )


def _draw_pedestrians(axis, event: dict, metadata: dict) -> None:
    radius = float(metadata.get("pedestrian_radius", 0.2))
    for position in _pedestrians(event.get("ped_xy", ())):
        axis.add_patch(Circle(
            position, radius=radius, facecolor="#666666", edgecolor=BLACK,
            linewidth=.35, alpha=.62, zorder=5,
        ))


def _draw_goal(axis, metadata: dict) -> None:
    if metadata.get("goal") is None:
        return
    goal = np.asarray(metadata["goal"], dtype=float)
    axis.plot(
        goal[0], goal[1], marker="*", color=GOLD, markeredgecolor=BLACK,
        markeredgewidth=.55, ms=8.0, linestyle="None", zorder=11,
    )


def _set_axis(axis, bounds) -> None:
    axis.set_aspect("equal")
    axis.set_xlim(bounds[0], bounds[1])
    axis.set_ylim(bounds[2], bounds[3])
    axis.set_xticks([])
    axis.set_yticks([])
    axis.grid(alpha=.12)


def _history(rows: list[dict], frame_step: int) -> np.ndarray:
    visible = [row for row in rows if int(row["step"]) <= frame_step]
    points = [_state(row["state"])[:2] for row in visible]
    if visible and visible[-1].get("executed_id") is not None:
        candidate = _candidates(visible[-1])[int(visible[-1]["executed_id"])]
        points.append(np.asarray(candidate["segment"], dtype=float)[1])
    return np.asarray(points, dtype=float)


def draw_lineage_event(
    axis, event: dict, history: np.ndarray, metadata: dict, bounds,
) -> None:
    candidates = _candidates(event)
    queries = _queries(event)
    queried_ids = set(queries)
    for candidate in candidates.values():
        segment = np.asarray(candidate["segment"], dtype=float)
        axis.plot(segment[:, 0], segment[:, 1], color=GRAY, lw=.42,
                  alpha=.24, zorder=1)
    for candidate_id in event["selected_ids"]:
        segment = np.asarray(candidates[int(candidate_id)]["segment"], dtype=float)
        axis.plot(segment[:, 0], segment[:, 1], color=ORANGE, lw=.85,
                  alpha=.78, zorder=2)
    for candidate_id in queried_ids:
        query = queries[candidate_id]
        segment = np.asarray(candidates[candidate_id]["segment"], dtype=float)
        positive = _query_status(query) == "positive"
        color = GREEN if positive else RED
        axis.plot(segment[:, 0], segment[:, 1], color=color, lw=1.05,
                  alpha=.9, zorder=3)
        if not positive:
            axis.plot(segment[-1, 0], segment[-1, 1], marker="x", color=RED,
                      ms=3.5, mew=.9, zorder=6)
    executed = event.get("executed_id")
    if executed is not None:
        segment = np.asarray(candidates[int(executed)]["segment"], dtype=float)
        axis.plot(segment[:, 0], segment[:, 1], color=BLUE, lw=1.65,
                  alpha=.95, zorder=7)
        axis.plot(segment[:2, 0], segment[:2, 1], color=BLUE, lw=3.0, zorder=8)
    if len(history):
        trajectory_color = GOLD if bool(event["committed_success"]) else BLACK
        axis.plot(history[:, 0], history[:, 1], color=trajectory_color,
                  lw=2.0 if trajectory_color == GOLD else 1.3, zorder=9)
    _draw_pedestrians(axis, event, metadata)
    _draw_goal(axis, metadata)
    state = _state(event["state"])
    axis.plot(state[0], state[1], "o", color=BLACK, ms=3.2, zorder=10)
    terminal = str(event["episode_status"])
    suffix = ""
    if terminal != "active":
        suffix = f" | {terminal.upper()}"
    if bool(event["committed_success"]):
        suffix += " | COMMITTED"
        for spine in axis.spines.values():
            spine.set_color(GOLD)
            spine.set_linewidth(1.8)
    axis.set_title(
        f"rep {int(event['replica']):02d} · scene {event['scenario_id']} · "
        f"t={int(event['step']):03d}{suffix}",
        fontsize=7,
    )
    _set_axis(axis, bounds)


def _legend(figure) -> None:
    handles = [
        Line2D([], [], color=GRAY, lw=1, label="K=16 generated"),
        Line2D([], [], color=ORANGE, lw=1.5, label="B=4 queried"),
        Line2D([], [], color=GREEN, lw=2, label="exact positive"),
        Line2D([], [], color=RED, lw=2, marker="x", label="exact negative"),
        Line2D([], [], color=BLUE, lw=2.5, label="selected / executed"),
        Line2D([], [], color=BLACK, lw=2, label="executed trajectory"),
        Line2D([], [], color=GOLD, lw=2.5, label="committed success"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=7, fontsize=7,
                  framealpha=.95)


def render_parallel_video(
    metadata: dict, events: Iterable[dict], output: str | Path, *, fps: int = 8,
) -> Path:
    """Render 16 lineages; a terminated lineage freezes without stopping peers."""
    events = list(events)
    groups = _lineages(events)
    if len(groups) != 16:
        raise ValueError("parallel video requires exactly 16 lineages")
    ordered = sorted(groups.items(), key=lambda item: item[0])
    bounds = _plot_bounds(metadata, events)
    final_step = max(int(rows[-1]["step"]) for _, rows in ordered)
    lookup = {
        key: {int(row["step"]): row for row in rows}
        for key, rows in ordered
    }
    figure, axes = plt.subplots(4, 4, figsize=(12, 12), constrained_layout=False)
    figure.subplots_adjust(left=.035, right=.99, bottom=.065, top=.89,
                           wspace=.07, hspace=.15)
    _legend(figure)

    batch_identity = event_identity(events[0])[:3]
    scenario_ids = sorted({str(event["scenario_id"]) for event in events})
    scenario_label = (
        f"fixed diagnostic scene {scenario_ids[0]}"
        if len(scenario_ids) == 1
        else f"{len(scenario_ids)} recorded training scenes"
    )
    figure.suptitle(
        f"HP100 parallel gather · r{batch_identity[0]} · "
        f"gamma={batch_identity[1]:g} · retry={batch_identity[2]} · "
        f"{scenario_label}\nGold means committed to replay; uncommitted diagnostics are presentation-only.",
        fontsize=12, y=.985,
    )

    def update(frame_step):
        for axis, (key, rows) in zip(axes.flat, ordered):
            axis.clear()
            event = lookup[key].get(frame_step, rows[-1] if frame_step > int(rows[-1]["step"])
                                    else rows[0])
            history = _history(rows, min(frame_step, int(rows[-1]["step"])))
            draw_lineage_event(axis, event, history, metadata, bounds)
        return []

    movie = animation.FuncAnimation(
        figure, update, frames=range(final_step + 1), interval=1000 / fps,
        blit=False, repeat=False,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(
        fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20"],
    )
    movie.save(output, writer=writer, dpi=120)
    plt.close(figure)
    return output


def choose_candidate_event(events: Iterable[dict]) -> dict:
    """Choose deterministically: most positives, then earliest identity."""
    events = list(events)
    return min(
        events,
        key=lambda event: (
            -sum(int(row["result"]["y"]) == 1 for row in event["query_rows"]),
            event_identity(event),
        ),
    )


def _candidate_audit_bounds(metadata: dict, event: dict) -> tuple[float, float, float, float]:
    """Robot-centered bounds that keep all 16 finite-sensing anchors visible."""
    diagnostics = {}
    for query in event.get("query_rows", ()):
        result = query.get("result", {})
        if result.get("resolved"):
            diagnostics = result.get("diagnostics", {})
            break
    sensing = float(diagnostics.get(
        "sensing_radius", diagnostics.get("R_eff", metadata.get("sensing_radius", 2.0)),
    ))
    rho_art = float(diagnostics.get("rho_art", .16))
    center = _state(event["state"])[:2]
    extent = sensing + 2.0 * rho_art + .12
    return (
        float(center[0] - extent), float(center[0] + extent),
        float(center[1] - extent), float(center[1] + extent),
    )


def _draw_stored_levels(axis, event: dict, query: dict) -> None:
    polygons = stored_level_polygons(event, query)
    for horizon, polygon in polygons:
        alpha = .18 + .58 * horizon / len(polygons)
        axis.add_patch(Polygon(
            polygon, closed=True, fill=False, edgecolor=GREEN,
            linewidth=.62, alpha=alpha, zorder=1.5,
        ))


def render_candidate_specific(
    metadata: dict, event: dict, png: str | Path, pdf: str | Path,
) -> tuple[Path, Path]:
    candidates = _candidates(event)
    queries = _queries(event)
    bounds = _candidate_audit_bounds(metadata, event)
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 8.4), constrained_layout=True)
    for axis, candidate_id in zip(axes.flat, map(int, event["selected_ids"])):
        query = queries[candidate_id]
        positive = _query_status(query) == "positive"
        for candidate in candidates.values():
            segment = np.asarray(candidate["segment"], dtype=float)
            axis.plot(segment[:, 0], segment[:, 1], color=GRAY, lw=.38,
                      alpha=.13, zorder=1)
        if positive:
            _draw_stored_levels(axis, event, query)
        _draw_stored_face_audit(axis, event, query, metadata, bounds)
        segment = np.asarray(candidates[candidate_id]["segment"], dtype=float)
        color = GREEN if positive else RED
        axis.plot(segment[:, 0], segment[:, 1], color=color, lw=2.2,
                  marker=".", ms=2.8, zorder=6)
        if not positive:
            axis.plot(segment[-1, 0], segment[-1, 1], "x", color=RED,
                      ms=7, mew=1.5, zorder=7)
        if event.get("executed_id") is not None \
                and int(event["executed_id"]) == candidate_id:
            axis.plot(segment[:2, 0], segment[:2, 1], color=BLUE, lw=4.0,
                      zorder=8)
        _draw_pedestrians(axis, event, metadata)
        _draw_goal(axis, metadata)
        state = _state(event["state"])
        axis.plot(state[0], state[1], "o", color=BLACK, ms=5, zorder=9)
        is_executed = (
            event.get("executed_id") is not None
            and int(event["executed_id"]) == candidate_id
        )
        axis.set_title(
            f"B candidate {candidate_id} · exact {int(query['result']['y'])}"
            + (" · executed" if is_executed else ""),
            fontsize=9,
        )
        _set_axis(axis, bounds)
    figure.suptitle(
        "Stored candidate-specific GREEN geometry · "
        f"scene {event['scenario_id']} · rep {event['replica']} · "
        f"t={event['step']} · gamma={float(event['gamma']):g}\n"
        "Ten level sets are reconstructed from stored exact faces; no verifier is run.",
        fontsize=11,
    )
    figure.legend(handles=[
        Line2D([], [], color="#006D2C", lw=1.3, label="active face"),
        Line2D([], [], color="#74C476", lw=.8, ls="--", label="redundant face"),
        Line2D([], [], color="#238B45", marker="o", markerfacecolor="none",
               lw=0, label="artificial anchor disk"),
        Line2D([], [], color="#006D2C", marker="o", markerfacecolor="#FFD92F",
               lw=0, label="disk tangency"),
        Line2D([], [], color="#00441B", lw=2.0, label="min-affine envelope"),
    ], loc="lower center", ncol=5, fontsize=7, framealpha=.95)
    png, pdf = Path(png), Path(pdf)
    png.parent.mkdir(parents=True, exist_ok=True)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(png, dpi=180)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_trace_complete(
    output_dir: str | Path, *, source_trace: str | Path,
    selection: dict, summary: dict, artifacts: Iterable[str | Path],
) -> Path:
    output_dir = Path(output_dir)
    source_trace = Path(source_trace).resolve()
    artifact_rows = {}
    for artifact in artifacts:
        path = Path(artifact).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        artifact_rows[path.name] = dict(
            path=str(path), bytes=path.stat().st_size, sha256=sha256_file(path),
        )
    marker = output_dir / "TRACE_COMPLETE.json"
    payload = dict(
        status=TRACE_COMPLETE_STATUS, schema_version=SCHEMA_VERSION,
        source_trace=dict(
            path=str(source_trace), bytes=source_trace.stat().st_size,
            sha256=sha256_file(source_trace),
        ),
        selection=selection, summary=summary, artifacts=artifact_rows,
    )
    _write_json(marker, payload)
    return marker


def render_bundle(
    trace_path: str | Path, output_dir: str | Path, *, round_index: int,
    gamma: float, retry_batch: int, fps: int = 8,
) -> dict:
    metadata, events = load_trace(trace_path)
    events = validate_trace(metadata, events)
    selected = select_batch(
        events, round_index=round_index, gamma=gamma, retry_batch=retry_batch,
    )
    # Revalidate the isolated batch so an incomplete filter cannot be rendered.
    selected = validate_trace(metadata, selected)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video = render_parallel_video(
        metadata, selected, output_dir / "parallel_lineages_4x4.mp4", fps=fps,
    )
    diagnostic = choose_candidate_event(selected)
    png, pdf = render_candidate_specific(
        metadata, diagnostic,
        output_dir / "candidate_specific_B.png",
        output_dir / "candidate_specific_B.pdf",
    )
    summary = committed_summary(selected)
    summary_path = output_dir / "committed_summary.json"
    _write_json(summary_path, summary)
    selection = dict(
        round=int(round_index), gamma=float(gamma), retry_batch=int(retry_batch),
        candidate_event=dict(zip(IDENTITY_FIELDS, event_identity(diagnostic))),
    )
    marker = write_trace_complete(
        output_dir, source_trace=trace_path, selection=selection,
        summary=summary, artifacts=(video, png, pdf, summary_path),
    )
    return dict(video=video, candidate_png=png, candidate_pdf=pdf,
                summary=summary_path, marker=marker)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--round", type=int, required=True, dest="round_index")
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--retry-batch", type=int, required=True)
    parser.add_argument("--fps", type=int, default=8)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.fps < 1:
        raise ValueError("fps must be positive")
    outputs = render_bundle(
        args.trace, args.output_dir, round_index=args.round_index,
        gamma=args.gamma, retry_batch=args.retry_batch, fps=args.fps,
    )
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))


if __name__ == "__main__":
    main()
