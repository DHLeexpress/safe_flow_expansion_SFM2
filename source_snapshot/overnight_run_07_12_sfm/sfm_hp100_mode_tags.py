"""Behavior-mode tagging for v2 MPC-rule expansion archives.

Every executed D+ row is classified along two declared axes:

- interaction axis (row-local): the selected candidate's per-horizon minimum
  CV pedestrian clearance dips below the MPC activation radius
  (``min_h d_h < r_eff``) — the collision-cost term was active, so the row is
  an avoidance-context sample rather than open-space goal seeking;
- decision axis (trace join): the v2 selector shadow record
  (``audits[0]["v2_shadow"]``, keyed by ``(lineage, step, attempt)``) says
  the MPC rule chose a different candidate than progress-argmax would have —
  safety actively altered the action, not merely blessed it.

Tags are attached in place as ``row["mode_tags"] = {"interaction": bool,
"changed": bool | None}`` (``None`` = no shadow record for that attempt).
The declared ``mode_gamma_tree`` D+ mass mode consumes these tags and fails
closed on untagged rows; rows collected under the authoritative
progress-argmax rule carry no ``mpc_horizon_clearances`` and are therefore
rejected by ``interaction_flag`` rather than silently mis-tagged.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import torch

# The v2 MPC rule's declared collision-cost activation radius (metres); the
# interaction axis uses the same constant so "avoidance context" means
# exactly "the cost term that shaped the selection was non-negligible".
DEFAULT_R_EFF = 0.45


def interaction_flag(row: dict, r_eff: float = DEFAULT_R_EFF) -> bool:
    """True when the selected candidate's horizon clearance dips below r_eff.

    Fails closed (KeyError) when the row carries no
    ``prediction_audit["mpc_horizon_clearances"]`` — only v2-collected
    archives are taggable.
    """
    clearances = row["prediction_audit"]["mpc_horizon_clearances"]
    return min(float(value) for value in clearances) < float(r_eff)


def changed_map_from_trace(trace_path: str | Path) -> dict:
    """``(lineage, step, attempt) -> changed`` from a v2 trace's shadows.

    Attempts whose first prediction audit carries no ``v2_shadow`` record are
    absent from the map (their rows tag ``changed=None``).
    """
    payload = torch.load(
        Path(trace_path), map_location="cpu", weights_only=False,
    )
    changed: dict[tuple[str, int, int], bool] = {}
    for event in payload["events"]:
        lineage = str(event["lineage"])
        step = int(event["step"])
        for attempt_row in event["attempts"]:
            audits = attempt_row.get("prediction_audits") or []
            if not audits:
                continue
            shadow = audits[0].get("v2_shadow")
            if shadow is None:
                continue
            key = (lineage, step, int(attempt_row["attempt"]))
            changed[key] = bool(shadow["changed"])
    return changed


def tag_rows(rows: list[dict], changed_maps) -> dict:
    """Attach ``mode_tags`` in place and return per-gamma bucket counts.

    ``changed_maps`` is one ``changed_map_from_trace`` result or an iterable
    of them (later maps win on key collisions, which cannot occur between
    distinct collection jobs). Buckets per gamma:

    - ``goal_seeking``: interaction False (open space; ``changed`` ignored);
    - ``safe_pass``: interaction True, shadow says the choice was unchanged;
    - ``hard_avoid``: interaction True, safety changed the choice;
    - ``unknown_changed``: interaction True, no shadow record.
    """
    merged: dict = {}
    if isinstance(changed_maps, dict):
        merged.update(changed_maps)
    else:
        for one in changed_maps:
            merged.update(one)
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {
        "goal_seeking": 0, "safe_pass": 0, "hard_avoid": 0,
        "unknown_changed": 0,
    })
    for row in rows:
        interaction = interaction_flag(row)
        key = (str(row["lineage"]), int(row["step"]), int(row["attempt"]))
        changed = merged.get(key)
        row["mode_tags"] = {
            "interaction": bool(interaction),
            "changed": None if changed is None else bool(changed),
        }
        if not interaction:
            bucket = "goal_seeking"
        elif changed is True:
            bucket = "hard_avoid"
        elif changed is False:
            bucket = "safe_pass"
        else:
            bucket = "unknown_changed"
        stats[f"{float(row['gamma']):g}"][bucket] += 1
    return dict(stats)
