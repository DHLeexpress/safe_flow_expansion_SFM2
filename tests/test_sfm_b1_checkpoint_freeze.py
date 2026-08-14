import torch

import grid_policy_sfm as GPS
import sfm_b1_store as BS


def test_strict_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(2)
    policy = GPS.build_sfm_policy(width=32)
    path = tmp_path / "policy.pt"
    GPS.save_sfm_policy(policy, path, extra={"tag": "strict"})
    restored, checkpoint = GPS.load_sfm_policy(path)
    assert checkpoint["tag"] == "strict"
    assert restored.config() == policy.config()
    for name, value in policy.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


def test_residual_trunk_default_path_matches_original_sequential_blocks():
    torch.manual_seed(20)
    policy = GPS.build_sfm_policy(width=32, res_dropout=0.05).eval()
    value = torch.randn(5, policy.d + policy.ctx_dim + policy.t_dim)
    expected = policy.trunk.inp(value)
    for block in policy.trunk.blocks:
        expected = expected + block(expected)
    torch.testing.assert_close(policy.trunk(value), expected, rtol=0, atol=0)


def test_only_enc_grid_frozen_and_sha_unchanged_after_update():
    torch.manual_seed(3)
    policy = GPS.build_sfm_policy(width=32)
    frozen = BS.configure_expansion_trainability(policy)
    assert frozen and all(name.startswith("enc_grid.") for name in frozen)
    assert policy.enc_low[0].weight.requires_grad
    assert policy.gru.weight_ih_l0.requires_grad
    assert policy.trunk.inp[0].weight.requires_grad
    assert policy.head.weight.requires_grad
    before = BS.module_sha256(policy.enc_grid)
    optimizer = torch.optim.SGD([p for p in policy.parameters() if p.requires_grad], lr=1e-3)
    hp10 = torch.randn(4, 10, 16, 12)
    low = torch.randn(4, 5)
    hist = torch.randn(4, 16, 2)
    controls = torch.randn(4, 10, 2).clamp(-2, 2)
    loss = policy.cfm_loss(controls, policy.ctx_from(hp10, low, hist))
    loss.backward(); optimizer.step()
    assert BS.module_sha256(policy.enc_grid) == before
