"""NFT contrastive velocity training: branch optima, CRN, weights, guards."""
import json

import pytest
import torch

import grid_policy_sfm_hp100 as GPS
import sfm_hp100_ball_adapter as PORT
import sfm_hp100_exhaustive_hybrid as HYBRID
import sfm_hp100_expansion_update as UPD
import sfm_hp100_nft_train as NFT
import sfm_hp100_predictive_execution as PRED

CTX = GPS.LOW_TOKEN + GPS.VISUAL_TOKEN
DEVICE = torch.device("cpu")


def _adapter(seed=0):
    torch.manual_seed(seed)
    return PORT.HP100ExpansionPolicy(GPS.build_sfm_hp100_policy()).eval()


def _group(seed, *, gamma=0.3, valid=(True, True, False, False),
           progress=None, scenario_id=900, step=0):
    generator = torch.Generator().manual_seed(seed)
    count = len(valid)
    return {
        "context": torch.randn(CTX + 4, generator=generator),
        "gamma": float(gamma), "replica": 0,
        "scenario_id": int(scenario_id), "step": int(step), "attempt": 0,
        "base_std": 1.0,
        "actions": torch.randn(count, 10, 2, generator=generator),
        "flow_bases": torch.randn(count, 10, 2, generator=generator),
        "valid": list(valid),
        "H10_progress": list(progress) if progress is not None
        else [1.0] * count,
        "predicted_min_clearance": [0.2] * count,
        "executed_local": list(valid).index(True),
        "provenance": {"source": "synthetic", "block_seed": 0},
    }


def _uniform_weights(group):
    count = len(group["valid"])
    return torch.full((count,), 1.0 / count)


def _seeded_draw(count, d, seed):
    """Replicate the trainer's shared (x0, tau) draw for expectations."""
    HYBRID._set_step_seed(seed, DEVICE)
    x0 = torch.randn_like(torch.zeros(count, d))
    tau = torch.rand(count).clamp(1.0e-4, 1.0)
    return x0, tau


class _StubPolicy:
    def __init__(self, fn):
        self.fn = fn
        self.d = 20
        self.u_max = 2.0
        self.calls = []

    def __call__(self, x_tau, tau, token):
        self.calls.append((x_tau.detach().clone(), tau.detach().clone()))
        return self.fn(x_tau, tau, token)


class _StubAdapter:
    def __init__(self, fn):
        self.policy = _StubPolicy(fn)


def test_branch_optima_reach_zero_loss():
    beta = 0.25
    group = _group(1, valid=(True, True, False, False))
    x1 = group["actions"].reshape(4, 20) / 2.0
    x0, _ = _seeded_draw(4, 20, seed=99)
    velocity = x1 - x0
    anchor = torch.full((4, 20), 0.3)
    reference = _StubAdapter(lambda x, t, tok: anchor.clone())
    tokens = torch.zeros(1, 1)

    # theta at the positive-branch optimum v_ref + (v - v_ref)/beta.
    theta = _StubAdapter(
        lambda x, t, tok: anchor + (velocity - anchor) / beta,
    )
    _, audit = NFT.nft_batch_loss(
        theta, reference, tokens, tokens, [group], [_uniform_weights(group)],
        beta=beta, coupled_x0=False, noise_seed=99, device=DEVICE,
    )
    assert audit["pos_branch_loss"] == pytest.approx(0.0, abs=1.0e-10)
    assert audit["neg_branch_loss"] > 1.0e-3

    # theta at the negative-branch optimum v_ref - (v - v_ref)/beta.
    theta = _StubAdapter(
        lambda x, t, tok: anchor - (velocity - anchor) / beta,
    )
    _, audit = NFT.nft_batch_loss(
        theta, reference, tokens, tokens, [group], [_uniform_weights(group)],
        beta=beta, coupled_x0=False, noise_seed=99, device=DEVICE,
    )
    assert audit["neg_branch_loss"] == pytest.approx(0.0, abs=1.0e-10)
    assert audit["pos_branch_loss"] > 1.0e-3


def test_beta_one_positive_branch_is_reference_free():
    theta = _adapter(2)
    group = _group(3, valid=(True, False, True, False))
    tokens = theta._policy_context(group["context"].unsqueeze(0))
    losses = {}
    for seed, name in ((5, "ref_a"), (6, "ref_b")):
        reference = _adapter(seed)
        ref_tokens = reference._policy_context(group["context"].unsqueeze(0))
        _, audit = NFT.nft_batch_loss(
            theta, reference, tokens, ref_tokens,
            [group], [_uniform_weights(group)],
            beta=1.0, coupled_x0=False, noise_seed=11, device=DEVICE,
        )
        losses[name] = audit
    # At beta=1 the positive branch is pure CFM regression on theta alone.
    assert losses["ref_a"]["pos_branch_loss"] == pytest.approx(
        losses["ref_b"]["pos_branch_loss"], abs=1.0e-9,
    )
    assert losses["ref_a"]["neg_branch_loss"] != pytest.approx(
        losses["ref_b"]["neg_branch_loss"], abs=1.0e-6,
    )


def test_shared_noise_discipline_is_reproducible():
    theta = _adapter(4)
    reference = _adapter(5)
    group = _group(7)
    tokens = theta._policy_context(group["context"].unsqueeze(0))
    ref_tokens = reference._policy_context(group["context"].unsqueeze(0))
    audits = []
    for _ in range(2):
        _, audit = NFT.nft_batch_loss(
            theta, reference, tokens, ref_tokens,
            [group], [_uniform_weights(group)],
            beta=1.0, coupled_x0=False, noise_seed=1234, device=DEVICE,
        )
        audits.append(audit)
    assert audits[0] == audits[1]
    _, other = NFT.nft_batch_loss(
        theta, reference, tokens, ref_tokens,
        [group], [_uniform_weights(group)],
        beta=1.0, coupled_x0=False, noise_seed=1235, device=DEVICE,
    )
    assert other["pos_branch_loss"] != pytest.approx(
        audits[0]["pos_branch_loss"], abs=1.0e-9,
    )


def test_group_weights_split_mass_and_follow_progress():
    group = _group(
        8, valid=(True, True, False, False, True),
        progress=(1.0, 2.0, 0.0, 0.0, 3.0),
    )
    weights = NFT.group_candidate_weights(group)
    assert float(weights.sum()) == pytest.approx(1.0, abs=1.0e-6)
    soft = torch.softmax(
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        / NFT.PROGRESS_TEMPERATURE,
        dim=0,
    )
    for local, expected in zip((0, 1, 4), soft):
        assert float(weights[local]) == pytest.approx(
            0.5 * float(expected), abs=1.0e-6,
        )
    assert float(weights[2]) == pytest.approx(0.25, abs=1.0e-6)
    assert float(weights[3]) == pytest.approx(0.25, abs=1.0e-6)
    # Higher progress -> more of the positive mass.
    assert float(weights[4]) > float(weights[1]) > float(weights[0])

    all_positive = _group(9, valid=(True, True))
    weights = NFT.group_candidate_weights(all_positive)
    assert float(weights.sum()) == pytest.approx(1.0, abs=1.0e-6)


def test_dataset_weights_balance_gamma_mass():
    groups = [
        _group(10, gamma=0.3, scenario_id=900),
        _group(11, gamma=0.3, scenario_id=901),
        _group(12, gamma=1.0, scenario_id=902),
    ]
    weights = NFT.dataset_weights(groups)
    low = float(weights[0].sum() + weights[1].sum())
    high = float(weights[2].sum())
    assert low == pytest.approx(0.5, abs=1.0e-6)
    assert high == pytest.approx(0.5, abs=1.0e-6)


def test_coupled_x0_substitutes_stored_bases_for_positives_only():
    group = _group(13, valid=(True, False, True, False))
    x1 = group["actions"].reshape(4, 20) / 2.0
    recorder = _StubAdapter(lambda x, t, tok: torch.zeros_like(x))
    theta = _StubAdapter(lambda x, t, tok: torch.zeros_like(x))
    tokens = torch.zeros(1, 1)
    for coupled in (False, True):
        recorder.policy.calls.clear()
        NFT.nft_batch_loss(
            theta, recorder, tokens, tokens,
            [group], [_uniform_weights(group)],
            beta=1.0, coupled_x0=coupled, noise_seed=321, device=DEVICE,
        )
        x_tau, tau = recorder.policy.calls[-1]
        x0_expected, tau_expected = _seeded_draw(4, 20, seed=321)
        assert torch.allclose(tau, tau_expected)
        if coupled:
            for row in (0, 2):
                x0_expected[row] = group["flow_bases"][row].reshape(20)
        expected = (1.0 - tau_expected)[:, None] * x0_expected \
            + tau_expected[:, None] * x1
        assert torch.allclose(x_tau, expected, atol=1.0e-6)


def test_lambda_safe_guards_conflicting_gradients():
    g_pos = torch.tensor([1.0, 0.0])
    assert NFT.lambda_safe(g_pos, torch.tensor([2.0, 0.0])) == 1.0
    assert NFT.lambda_safe(g_pos, torch.tensor([0.0, 5.0])) == 1.0
    # Full conflict: scale to (1 - mu).
    assert NFT.lambda_safe(g_pos, -g_pos) == pytest.approx(
        1.0 - NFT.LAMBDA_SAFE_MU,
    )
    # Steeper conflict: proportionally smaller.
    assert NFT.lambda_safe(g_pos, -4.0 * g_pos) == pytest.approx(
        (1.0 - NFT.LAMBDA_SAFE_MU) / 4.0,
    )
    # Mild conflict clamps at 1.
    assert NFT.lambda_safe(g_pos, -0.1 * g_pos) == 1.0
    # The guaranteed first-order decrease of the positive branch.
    scale = NFT.lambda_safe(g_pos, -4.0 * g_pos)
    combined = g_pos + scale * (-4.0 * g_pos)
    assert float(torch.dot(combined, g_pos)) == pytest.approx(
        NFT.LAMBDA_SAFE_MU * float(g_pos.square().sum()), abs=1.0e-6,
    )


def test_one_optimizer_step_keeps_frozen_surface_digest():
    theta = _adapter(14)
    reference = _adapter(15)
    scope = UPD.PROJECTION_OPTIMIZER_SCOPE
    parameters, _ = UPD.configure_trainable(theta, scope)
    frozen_before = UPD.frozen_surface_sha256(theta, scope)
    before = [parameter.detach().clone() for parameter in parameters]
    group = _group(16)
    tokens = theta._policy_context(group["context"].unsqueeze(0))
    ref_tokens = reference._policy_context(group["context"].unsqueeze(0))
    optimizer = torch.optim.Adam(parameters, lr=1.0e-3)
    total, _ = NFT.nft_batch_loss(
        theta, reference, tokens, ref_tokens,
        [group], [_uniform_weights(group)],
        beta=1.0, coupled_x0=False, noise_seed=17, device=DEVICE,
    )
    optimizer.zero_grad(set_to_none=True)
    total.backward()
    optimizer.step()
    assert UPD.frozen_surface_sha256(theta, scope) == frozen_before
    assert any(
        not torch.equal(a, b.detach())
        for a, b in zip(before, parameters)
    )


# ---- runner ----------------------------------------------------------------


def _strict_checkpoint(adapter, path):
    payload = UPD.checkpoint_payload(
        adapter,
        parent_checkpoint_sha256="0" * 64,
        pretrained_checkpoint_sha256="0" * 64,
        round_index=0,
        alpha=0.0,
    )
    torch.save(payload, path)
    return PRED.sha256_file(path)


def _runner_fixture(tmp_path, group_count=6):
    sha = _strict_checkpoint(_adapter(seed=20), tmp_path / "r0.pt")
    groups = [
        _group(30 + index, gamma=(0.3 if index % 2 == 0 else 1.0),
               scenario_id=900 + index, step=index)
        for index in range(group_count)
    ]
    archive = tmp_path / "groups.pt"
    torch.save({"status": NFT.GROUP_ARCHIVE_STATUS, "groups": groups}, archive)
    return sha, archive


def _runner_args(tmp_path, sha, archive, output, extra=()):
    return NFT.parser().parse_args([
        "--checkpoint", str(tmp_path / "r0.pt"),
        "--expected-checkpoint-sha256", sha,
        "--reference-checkpoint", str(tmp_path / "r0.pt"),
        "--expected-reference-checkpoint-sha256", sha,
        "--groups", str(archive),
        "--context-path", "stored",
        "--optimizer-scope", UPD.OPTIMIZER_SCOPE,
        "--groups-per-batch", "2",
        "--output", str(output),
        "--device", "cpu", "--physical-gpu", "-1",
        *extra,
    ])


def test_runner_end_to_end_snapshots_and_acceptance(tmp_path):
    sha, archive = _runner_fixture(tmp_path)
    output = tmp_path / "out"
    marker = NFT.run(_runner_args(
        tmp_path, sha, archive, output, extra=("--snapshot-every", "1"),
    ))
    assert marker["accepted"] is True and marker["steps"] == 3
    assert [row["step"] for row in marker["snapshots"]] == [1, 2, 3]
    for row in marker["snapshots"]:
        payload = torch.load(row["path"], map_location="cpu",
                             weights_only=False)
        assert payload["nft"]["beta"] == 1.0
        GPS.load_sfm_hp100_policy(row["path"], device="cpu")
    assert marker["checkpoint"] is not None
    assert (output / "checkpoint_r1.pt").exists()
    GPS.load_sfm_hp100_policy(output / "checkpoint_r1.pt", device="cpu")
    listed = json.loads((output / "NFT_TRAIN_COMPLETE.json").read_text())
    assert listed["status"] == NFT.STATUS
    assert listed["per_pass"][0]["pos_branch_loss"] > 0.0
    assert listed["per_pass"][0]["grad_norm"] > 0.0
    assert marker["relative_parameter_drift"] > 0.0


def test_runner_reverts_when_drift_gate_fails(tmp_path):
    sha, archive = _runner_fixture(tmp_path)
    output = tmp_path / "out_reverted"
    marker = NFT.run(_runner_args(
        tmp_path, sha, archive, output,
        extra=("--max-relative-parameter-drift", "0.0"),
    ))
    assert marker["accepted"] is False
    assert marker["checkpoint"] is None
    assert not (output / "checkpoint_r1.pt").exists()
    assert marker["relative_parameter_drift"] > 0.0
    assert (output / "NFT_TRAIN_COMPLETE.json").exists()


def test_runner_refuses_wide_scope_with_stored_tokens(tmp_path):
    sha, archive = _runner_fixture(tmp_path, group_count=2)
    with pytest.raises(ValueError, match="wider scopes need"):
        NFT.run(_runner_args(
            tmp_path, sha, archive, tmp_path / "out_scope",
            extra=("--optimizer-scope", UPD.PROJECTION_OPTIMIZER_SCOPE),
        ))


def test_runner_refuses_non_group_archives(tmp_path):
    sha, _ = _runner_fixture(tmp_path, group_count=2)
    bad = tmp_path / "bad.pt"
    torch.save({"status": "SFM2_PAIR_ARCHIVE", "groups": []}, bad)
    with pytest.raises(RuntimeError, match="labeled-group archive"):
        NFT.run(_runner_args(tmp_path, sha, bad, tmp_path / "out_bad"))
