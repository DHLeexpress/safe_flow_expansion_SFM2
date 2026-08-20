"""DPO-CFM pair training: objective math, noise discipline, freeze, harvest."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_dpo_train as DPO
import sfm_hp100_exhaustive_hybrid as HYBRID
import sfm_hp100_expansion_update as UPD
import sfm_hp100_pair_harvest as PH

CTX = GPS.LOW_TOKEN + GPS.VISUAL_TOKEN
DEVICE = torch.device("cpu")


def _adapter(seed=0):
    torch.manual_seed(seed)
    return PORT.HP100ExpansionPolicy(GPS.build_sfm_hp100_policy()).eval()


def _batch(count=4, seed=11):
    generator = torch.Generator().manual_seed(seed)
    contexts = torch.randn(count, CTX + 4, generator=generator)
    pos = torch.randn(count, 10, 2, generator=generator)
    neg = torch.randn(count, 10, 2, generator=generator)
    return contexts, pos, neg


def test_objective_matches_manual_softplus_and_beta_scaling():
    theta = _adapter(1)
    reference = _adapter(2)
    contexts, pos, neg = _batch()
    outs = {}
    for beta in (1.0, 3.0):
        total, audit = DPO.dpo_batch_losses(
            theta, reference, contexts, contexts, pos, neg,
            beta=beta, anchor_weight=0.0, noise_seed=99, device=DEVICE,
        )
        outs[beta] = (total, audit)
    # Manual recomputation with the identical shared-noise discipline.
    losses = {}
    for name, model, actions in (
        ("lp_t", theta, pos), ("ln_t", theta, neg),
        ("lp_r", reference, pos), ("ln_r", reference, neg),
    ):
        HYBRID._set_step_seed(99, DEVICE)
        with torch.no_grad():
            losses[name] = model.cfm_loss(contexts, actions, reduction="none")
    margin = (losses["lp_t"] - losses["lp_r"]) - (losses["ln_t"] - losses["ln_r"])
    for beta in (1.0, 3.0):
        expected = F.softplus(beta * margin).mean()
        assert float(outs[beta][0].detach()) == pytest.approx(
            float(expected), abs=1e-6,
        )
        assert outs[beta][1]["margin_mean"] == pytest.approx(
            float(margin.mean()), abs=1e-6,
        )


def test_identical_theta_and_reference_gives_log2():
    theta = _adapter(3)
    reference = _adapter(3)  # same seed -> identical parameters
    contexts, pos, neg = _batch(seed=5)
    total, audit = DPO.dpo_batch_losses(
        theta, reference, contexts, contexts, pos, neg,
        beta=1.0, anchor_weight=0.0, noise_seed=7, device=DEVICE,
    )
    assert audit["margin_mean"] == pytest.approx(0.0, abs=1e-6)
    assert float(total) == pytest.approx(float(np.log(2.0)), abs=1e-6)


def test_shared_noise_discipline_is_reproducible():
    theta = _adapter(4)
    reference = _adapter(5)
    contexts, pos, neg = _batch(seed=21)
    audits = []
    for _ in range(2):
        _, audit = DPO.dpo_batch_losses(
            theta, reference, contexts, contexts, pos, neg,
            beta=1.0, anchor_weight=0.5, noise_seed=1234, device=DEVICE,
        )
        audits.append(audit)
    assert audits[0] == audits[1]


def test_anchor_weight_adds_scaled_positive_loss():
    theta = _adapter(6)
    reference = _adapter(7)
    contexts, pos, neg = _batch(seed=31)
    base, audit0 = DPO.dpo_batch_losses(
        theta, reference, contexts, contexts, pos, neg,
        beta=1.0, anchor_weight=0.0, noise_seed=55, device=DEVICE,
    )
    weighted, audit1 = DPO.dpo_batch_losses(
        theta, reference, contexts, contexts, pos, neg,
        beta=1.0, anchor_weight=0.7, noise_seed=55, device=DEVICE,
    )
    assert float(weighted) == pytest.approx(
        float(base) + 0.7 * audit1["anchor_loss"], abs=1e-6,
    )
    assert audit0["anchor_loss"] == pytest.approx(audit1["anchor_loss"], abs=1e-9)


def test_one_step_widens_negative_positive_gap_with_frozen_surface():
    theta = _adapter(8)
    reference = _adapter(8)
    contexts, pos, neg = _batch(count=8, seed=41)
    parameters, _ = UPD.configure_trainable(theta, UPD.OPTIMIZER_SCOPE)
    frozen_before = UPD.frozen_surface_sha256(theta, UPD.OPTIMIZER_SCOPE)

    def gap() -> float:
        HYBRID._set_step_seed(77, DEVICE)
        with torch.no_grad():
            lp = theta.cfm_loss(contexts, pos, reduction="none")
        HYBRID._set_step_seed(77, DEVICE)
        with torch.no_grad():
            ln = theta.cfm_loss(contexts, neg, reduction="none")
        return float((ln - lp).mean())

    before = gap()
    optimizer = torch.optim.Adam(parameters, lr=5.0e-3)
    for step in range(3):
        total, _ = DPO.dpo_batch_losses(
            theta, reference, contexts, contexts, pos, neg,
            beta=5.0, anchor_weight=0.0, noise_seed=88 + step, device=DEVICE,
        )
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
    assert gap() > before
    assert UPD.frozen_surface_sha256(theta, UPD.OPTIMIZER_SCOPE) == frozen_before


# ---- harvester -------------------------------------------------------------


def _stub_plans(seed, K=PH.PRED.K):
    generator = torch.Generator().manual_seed(int(seed) % (2**31))
    return torch.randn(K, 10, 2, generator=generator)


def _stub_bases(seed, K=PH.PRED.K):
    generator = torch.Generator().manual_seed((int(seed) + 1) % (2**31))
    return torch.randn(K, 20, generator=generator)


def _stub_sample_blocks(adapter, contexts, seeds, *, K, flow_base_std):
    return (
        [_stub_plans(seed) for seed in seeds],
        [_stub_bases(seed) for seed in seeds],
        None, None,
    )


def _synthetic_trace(block_seed=101, corrupt_segment=False):
    key = HYBRID.LineageKey(gamma=0.3, replica=1)
    step, attempt = 4, 0
    seed = HYBRID._sampling_seed(
        block_seed, "predictive_always_on", key, step,
        microcycle=0, attempt=attempt,
    )
    plans = _stub_plans(seed)
    candidate_ids = list(range(32))
    state_before = np.asarray([0.5, 0.4, 0.2, -0.1], np.float32)
    segments = np.stack([
        PORT.clipped_plan_states(state_before, plans[i].numpy())[:, :2]
        for i in candidate_ids
    ]).astype(np.float32)
    if corrupt_segment:
        segments[3] += 1.0
    verification = []
    audits = []
    for local in range(32):
        # locals 3 and 9 are exact negatives; 9 has the higher progress and
        # must be preferred by the harvester's temptation rule.
        valid = local not in (3, 9)
        verification.append({
            "valid": valid, "hp_eligible": True, "margin": 0.1,
            "native_cost": 1.0,
            "H10_progress": {3: 0.8, 9: 1.4}.get(local, 1.0),
            "progress_eligible": True, "error": False, "step_margin": 0.0,
        })
        audits.append({"predicted_min_clearance": 0.2 + 0.01 * local})
    event = {
        "gamma": 0.3, "replica": 1, "lineage": key.label, "scenario_id": 777,
        "step": step,
        "context": torch.randn(CTX + 4, generator=torch.Generator().manual_seed(9)),
        "state_before": state_before,
        "attempts": [{
            "attempt": attempt, "base_std": 1.0,
            "candidate_ids": candidate_ids,
            "B_segments": segments,
            "verification": verification,
            "prediction_audits": audits,
            "predictive_local": 5,
        }],
        "executed_role": "positive",
        "terminal": None,
    }
    return [event], plans


def test_harvester_recovers_the_tempting_negative(monkeypatch):
    events, plans = _synthetic_trace()
    monkeypatch.setattr(PH.ACQ, "_sample_blocks", _stub_sample_blocks)
    pairs, stats = PH.harvest_trace(
        events, adapter=None, block_seed=101, source="synthetic",
        device=torch.device("cpu"),
    )
    assert stats["pairs"] == 1 and stats["segment_mismatches"] == 0
    pair = pairs[0]
    assert pair["pos_local"] == 5 and pair["neg_local"] == 9
    assert torch.equal(pair["pos_action"], plans[5])
    assert torch.equal(pair["neg_action"], plans[9])
    assert pair["provenance"]["block_seed"] == 101
    assert pair["neg_verification"]["valid"] is False


def test_harvester_group_mode_carries_labels_and_flow_bases(monkeypatch):
    events, plans = _synthetic_trace()
    monkeypatch.setattr(PH.ACQ, "_sample_blocks", _stub_sample_blocks)
    key = HYBRID.LineageKey(gamma=0.3, replica=1)
    seed = HYBRID._sampling_seed(
        101, "predictive_always_on", key, 4, microcycle=0, attempt=0,
    )
    pairs, stats, groups = PH.harvest_trace(
        events, adapter=None, block_seed=101, source="synthetic",
        device=torch.device("cpu"), collect_groups=True,
    )
    assert stats["pairs"] == 1 and len(groups) == 1
    group = groups[0]
    ids = events[0]["attempts"][0]["candidate_ids"]
    assert torch.equal(group["actions"], plans[ids])
    assert torch.equal(group["flow_bases"], _stub_bases(seed)[ids])
    assert group["valid"][9] is False and group["valid"][5] is True
    assert group["executed_local"] == 5
    assert len(group["H10_progress"]) == 32
    assert len(group["predicted_min_clearance"]) == 32


def test_harvester_fails_closed_on_segment_mismatch(monkeypatch):
    events, _ = _synthetic_trace(corrupt_segment=True)
    # Make local 3 the chosen negative so the corrupted segment is exercised.
    events[0]["attempts"][0]["verification"][9]["H10_progress"] = 0.1
    monkeypatch.setattr(PH.ACQ, "_sample_blocks", _stub_sample_blocks)
    with pytest.raises(RuntimeError, match="mismatch rate"):
        PH.harvest_trace(
            events, adapter=None, block_seed=101, source="synthetic",
            device=torch.device("cpu"),
        )


def test_harvester_skips_events_without_exact_negative():
    events, _ = _synthetic_trace()
    for row in events[0]["attempts"][0]["verification"]:
        row["valid"] = True
    assert PH.executed_pair_specs(events) == []
