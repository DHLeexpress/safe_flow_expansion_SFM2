import torch

import sfm_b1_rbf as R


def test_rbf_posterior_matches_direct_linear_algebra():
    torch.manual_seed(4)
    train = torch.randn(12, 9)
    query = torch.randn(5, 9)
    gp = R.RBFGP(0.8, 1e-2)
    gp.set_buffer(train)
    normalized_train = R.l2_normalize(train)
    normalized_query = R.l2_normalize(query)
    K = gp._kernel(normalized_train, normalized_train).double()
    cross = gp._kernel(normalized_query, normalized_train).double()
    expected = 1.0 - (cross * torch.linalg.solve(
        K + gp.lam * torch.eye(len(K), dtype=torch.double), cross.T
    ).T).sum(1)
    torch.testing.assert_close(gp.sigma(query).square().double(), expected.clamp_min(0), rtol=1e-5, atol=1e-6)


def test_pending_point_conditioning_reduces_duplicate_score():
    features = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [-1., 0.]])
    gp = R.RBFGP(0.5, 1e-2)
    vectors = gp.sequential_score_vectors(features, torch.tensor([0, 1, 2, 3]), 2)
    assert vectors[1][0] < vectors[0][1] * 0.1  # remaining duplicate of selected point
    generator = torch.Generator().manual_seed(2)
    selected, trace = gp.sequential_acquire(features, 4, beta=0.05, generator=generator)
    assert len(selected) == len(set(selected)) == 4
    assert all(0 < row["ess_norm"] <= 1 for row in trace)


def test_batched_covariance_and_acquisition_match_independent_contexts():
    torch.manual_seed(12)
    train = torch.randn(17, 7)
    features = torch.randn(3, 6, 7)
    gp = R.RBFGP(.6, 1e-2)
    gp.set_buffer(train)
    expected = torch.stack([gp.posterior_covariance(row) for row in features])
    torch.testing.assert_close(gp.posterior_covariance_batched(features), expected)
    torch.testing.assert_close(
        gp.acquisition_sigma_batched(features),
        torch.stack([gp.acquisition_sigma(row) for row in features]),
    )
    generator = torch.Generator().manual_seed(7)
    selected, traces = gp.sequential_acquire_batched(features, 4, .1, generator=generator)
    assert len(selected) == len(traces) == 3
    assert all(len(values) == len(set(values)) == 4 for values in selected)
    assert all(len(values) == 4 for values in traces)


def test_lengthscale_demands_exactly_fifty():
    try:
        R.mean_pairwise_lengthscale(torch.randn(49, 4))
    except ValueError as error:
        assert "exactly 50" in str(error)
    else:
        raise AssertionError("49 embeddings were accepted")
    assert R.mean_pairwise_lengthscale(torch.randn(50, 4)) > 0


def test_preflight_rejects_no_uplift():
    rows = [dict(ell_multiplier=ell, cap=cap, ess_solved=True, stable_conditioning=True, uplift=-.1)
            for ell in (.5, 1.) for cap in (256, 512)]
    try:
        R.choose_preflight(rows)
    except RuntimeError as error:
        assert "positive" in str(error)
    else:
        raise AssertionError("non-uplifting preflight was selected")
