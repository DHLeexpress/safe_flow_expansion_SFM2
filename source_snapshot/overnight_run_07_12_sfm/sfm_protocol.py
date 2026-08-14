"""Frozen protocol constants and disjoint seed banks for SFM Hp10/B1."""
from __future__ import annotations

import sfm_scene as SS

GAMMAS = SS.GAMMAS
K = 16
B = 4
T = 180
H = 10
W = 2
BATCH = 128
LR = 1.0e-5
ESS_TARGET = 0.5
ROUNDS = 20
SCENARIOS_PER_ROUND = 8
N_PED = 20

# Legacy Hp10/B1 study constants remain available because the standalone
# package intentionally preserves and tests that completed lineage alongside
# the canonical HP100 pretraining path. HP100 code does not consume them.
SWEEP_LR = 1.0e-4
OPTIMIZER_CHUNKS = 16
INNER_EPOCHS = (1, 4, 16)
GP_RETAIN_PER_ROUND = 256
GP_CAP = 512
GP_RETAIN_QUANTILE = 0.75
SANITY_M = 10
ADAPTIVE_PROPOSAL_K = 64
ADAPTIVE_QUERY_BATCH = 4
ADAPTIVE_MAX_QUERIES = 64

# Explicit environment metadata.  N_PED remains the legacy B1 expansion
# constant; scientific evaluation selects one of the named profiles instead.
TRAINING_ENVIRONMENT = SS.scene_profile("training")
MATCHED_ID_ENVIRONMENT = SS.scene_profile("matched_id")
ID_ENVIRONMENT = SS.scene_profile("id")
DENSITY_OOD_ENVIRONMENT = SS.scene_profile("density_ood")
REQUESTED_OOD_ENVIRONMENT = SS.scene_profile("requested_ood")
LEGACY_VELOCITY_OOD_ENVIRONMENT = SS.scene_profile("legacy_velocity_ood")

# Demonstrations are below 8,000. Every named bank is mutually disjoint.
PRETRAIN_GATE_EP0 = 12_000
# A second matched-ID bank used only after the M10 pretraining screen.  Its
# episode IDs are disjoint from both the demonstration attempts (< 5,000) and
# the screen above.
PRETRAIN_CONFIRM_EP0 = 14_000
EXPANSION_EP0 = 20_000
SCREEN_EP0 = 50_000
CONFIRM_EP0 = 80_000
KAZUKI_CONFIRM_EP0 = 90_000
SMOKE_EP0 = 110_000
SMOKE_EVAL_EP0 = 130_000
DEPLOY_ID_EP0 = 150_000
DEPLOY_OOD_EP0 = 170_000
QUERY_DIAGNOSTIC_EP0 = 190_000
DEPLOY_DENSITY_OOD_EP0 = 210_000
DEPLOY_DOUBLE_SHIFT_EP0 = 250_000
TEMPERATURE_SELECT_EP0 = 300_000
CURVE_SCREEN_EP0 = 320_000
FINAL_CONFIRM_EP0 = 400_000
ADAPTIVE_CONFIRM_EP0 = 500_000


def expansion_scenarios(round_i, *, smoke=False):
    if int(round_i) < 1:
        raise ValueError("round indices start at one")
    base = SMOKE_EP0 if smoke else EXPANSION_EP0
    start = base + (int(round_i) - 1) * SCENARIOS_PER_ROUND
    return tuple(range(start, start + SCENARIOS_PER_ROUND))


def raw_bank(ep0, m_per_gamma):
    return {str(g): tuple(range(int(ep0), int(ep0) + int(m_per_gamma))) for g in GAMMAS}
