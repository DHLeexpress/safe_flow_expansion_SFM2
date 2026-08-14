"""Tested exact RBF-GP and pending-conditioned B1 acquisition primitives."""
from __future__ import annotations

import math
from collections import defaultdict
import numpy as np
import torch


def l2_normalize(values, eps=1.0e-9):
    values = torch.as_tensor(values)
    return values / values.norm(dim=-1, keepdim=True).clamp_min(float(eps))


def balanced_exact50(records, *, gamma_key="gamma", scenario_key="scenario_id"):
    """Round-robin exactly 50 records with marginal cell counts differing by at most one."""
    groups = {}
    for record in records:
        key = (round(float(record[gamma_key]), 8), int(record[scenario_key]))
        groups.setdefault(key, []).append(record)
    if not groups:
        raise ValueError("no pretrained embedding records")
    by_gamma = {}
    for key in sorted(groups):
        by_gamma.setdefault(key[0], []).append(key)
    gamma_keys = sorted(by_gamma)
    keys = []
    scenario_index = 0
    while len(keys) < 50:
        progressed = False
        for gamma in gamma_keys:
            candidates = by_gamma[gamma]
            if scenario_index < len(candidates):
                keys.append(candidates[scenario_index])
                progressed = True
                if len(keys) == 50:
                    break
        if not progressed:
            break
        scenario_index += 1
    if len(keys) < 50:
        # Revisit cells only after every available gamma/scenario cell got one slot.
        keys = [key for index in range(50) for key in sorted(groups)
                if index < len(groups[key])][:50]
    chosen = []
    cell_visits = defaultdict(int)
    for key in keys:
        index = cell_visits[key]
        if index >= len(groups[key]):
            raise ValueError("fewer than 50 balanced pretrained embeddings")
        chosen.append(groups[key][index])
        cell_visits[key] += 1
    if len({id(value) for value in chosen}) != 50:
        raise RuntimeError("lengthscale sample contains duplicates")
    return chosen


def mean_pairwise_lengthscale(features):
    values = l2_normalize(torch.as_tensor(features).detach().to(torch.float64))
    if values.ndim != 2 or values.shape[0] != 50:
        raise ValueError("ell0 requires exactly 50 pretrained embeddings")
    lengthscale = float(torch.pdist(values, p=2).mean())
    if not math.isfinite(lengthscale) or lengthscale <= 0.0:
        raise ValueError("invalid pretrained mean pairwise distance")
    return lengthscale


class RBFGP:
    def __init__(self, lengthscale, lam=1.0e-2):
        if not math.isfinite(float(lengthscale)) or float(lengthscale) <= 0.0:
            raise ValueError("lengthscale must be finite and positive")
        if not math.isfinite(float(lam)) or float(lam) <= 0.0:
            raise ValueError("lambda must be finite and positive")
        self.ell = float(lengthscale)
        self.lam = float(lam)
        self.X = None
        self.L = None
        self.jitter = None

    @staticmethod
    def _sqdist(left, right):
        return ((left * left).sum(1, keepdim=True) + (right * right).sum(1)[None]
                - 2.0 * left @ right.T).clamp_min(0.0)

    def _kernel(self, left, right):
        return torch.exp(-self._sqdist(left, right) / (2.0 * self.ell ** 2))

    @torch.no_grad()
    def set_buffer(self, features):
        if features is None or len(features) == 0:
            self.X = self.L = None
            self.jitter = None
            return
        self.X = l2_normalize(torch.as_tensor(features).detach())
        kernel = self._kernel(self.X, self.X).to(torch.float64)
        eye = torch.eye(len(kernel), dtype=kernel.dtype, device=kernel.device)
        jitter = self.lam
        last_error = None
        for _ in range(6):
            try:
                self.L = torch.linalg.cholesky(kernel + jitter * eye)
                self.jitter = float(jitter)
                return
            except RuntimeError as error:
                last_error = error
                jitter *= 10.0
        raise RuntimeError("RBF-GP Cholesky failed") from last_error

    @torch.no_grad()
    def posterior_covariance(self, features, include_observation_noise=True):
        query = l2_normalize(torch.as_tensor(features).detach())
        covariance = self._kernel(query, query)
        if self.X is not None:
            cross = self._kernel(query, self.X)
            solved = torch.cholesky_solve(cross.T.to(torch.float64), self.L)
            covariance = covariance - cross @ solved.to(cross.dtype)
        covariance = 0.5 * (covariance + covariance.T)
        if include_observation_noise:
            covariance = covariance + self.lam * torch.eye(
                len(query), dtype=query.dtype, device=query.device
            )
        return covariance

    @torch.no_grad()
    def posterior_covariance_batched(self, features, include_observation_noise=True):
        """Posterior covariance for independent ``[context,K,D]`` pools.

        The GP memory is shared, but no covariance or pending observation is
        introduced across contexts.  Only the expensive solve against the
        common GP buffer is batched.
        """
        query = l2_normalize(torch.as_tensor(features).detach())
        if query.ndim != 3:
            raise ValueError("batched GP queries must have shape [context,K,D]")
        difference = query[:, :, None, :] - query[:, None, :, :]
        covariance = torch.exp(
            -difference.square().sum(-1) / (2.0 * self.ell ** 2)
        )
        if self.X is not None:
            contexts, candidates, dimension = query.shape
            cross = self._kernel(query.reshape(-1, dimension), self.X).reshape(
                contexts, candidates, len(self.X)
            )
            solved = torch.cholesky_solve(
                cross.reshape(-1, len(self.X)).T.to(torch.float64), self.L
            ).T.to(cross.dtype).reshape_as(cross)
            covariance = covariance - torch.einsum("cim,cjm->cij", cross, solved)
        covariance = 0.5 * (covariance + covariance.transpose(1, 2))
        if include_observation_noise:
            covariance = covariance + self.lam * torch.eye(
                covariance.shape[-1], dtype=query.dtype, device=query.device
            )[None]
        return covariance

    @torch.no_grad()
    def sigma(self, features):
        covariance = self.posterior_covariance(features, include_observation_noise=False)
        return torch.diagonal(covariance).clamp_min(0.0).sqrt()

    @torch.no_grad()
    def acquisition_sigma(self, features):
        """First-step marginal sigma in the same normalized scale as pending acquisition."""
        covariance = self.posterior_covariance(features, include_observation_noise=True)
        return (torch.diagonal(covariance) / (1.0 + self.lam)).clamp(0.0, 1.0).sqrt()

    @torch.no_grad()
    def acquisition_sigma_batched(self, features):
        covariance = self.posterior_covariance_batched(features, include_observation_noise=True)
        return (torch.diagonal(covariance, dim1=1, dim2=2) / (1.0 + self.lam)).clamp(
            0.0, 1.0
        ).sqrt()

    @staticmethod
    def _condition(covariance, remaining, chosen_local):
        keep = torch.ones(len(covariance), dtype=torch.bool, device=covariance.device)
        keep[int(chosen_local)] = False
        if not keep.any():
            return covariance.new_zeros((0, 0)), remaining[keep]
        cross = covariance[keep, int(chosen_local)]
        denominator = covariance[int(chosen_local), int(chosen_local)].clamp_min(1.0e-12)
        updated = covariance[keep][:, keep] - torch.outer(cross, cross) / denominator
        return 0.5 * (updated + updated.T), remaining[keep]

    @torch.no_grad()
    def sequential_score_vectors(self, features, order, steps):
        features = torch.as_tensor(features)
        order = torch.as_tensor(order, dtype=torch.long, device=features.device)
        if sorted(order.tolist()) != list(range(len(features))):
            raise ValueError("order must be a permutation")
        covariance = self.posterior_covariance(features)
        remaining = torch.arange(len(features), device=features.device)
        vectors = []
        for step in range(int(steps)):
            scores = (torch.diagonal(covariance) / (1.0 + self.lam)).clamp(0.0, 1.0)
            vectors.append(scores)
            location = torch.nonzero(remaining == order[step], as_tuple=False).flatten()
            covariance, remaining = self._condition(covariance, remaining, int(location[0]))
        return vectors

    @torch.no_grad()
    def sequential_acquire(self, features, steps, beta, *, generator=None):
        if not math.isfinite(float(beta)) or float(beta) <= 0.0:
            raise ValueError("beta must be finite and positive")
        features = torch.as_tensor(features)
        covariance = self.posterior_covariance(features)
        remaining = torch.arange(len(features), device=features.device)
        selected, trace = [], []
        for _ in range(int(steps)):
            scores = (torch.diagonal(covariance) / (1.0 + self.lam)).clamp(0.0, 1.0)
            weights = torch.exp(((scores - scores.max()) / float(beta)).clamp(-30.0, 30.0))
            probability = weights / weights.sum()
            local = int(torch.multinomial(probability, 1, generator=generator).item())
            chosen = int(remaining[local])
            probability64 = probability.detach().cpu().to(torch.float64)
            ess = float(
                1.0 / (probability64.square().sum() * len(probability64))
            )
            trace.append(dict(
                remaining=remaining.detach().cpu(), scores=scores.detach().cpu(),
                probability=probability.detach().cpu(), chosen=chosen,
                chosen_sigma=float(scores[local].sqrt()), ess_norm=ess,
            ))
            selected.append(chosen)
            covariance, remaining = self._condition(covariance, remaining, local)
        return selected, trace

    @torch.no_grad()
    def sequential_acquire_batched(self, features, steps, beta, *, generator=None):
        """Pending-conditioned acquisition for many independent contexts."""
        if not math.isfinite(float(beta)) or float(beta) <= 0.0:
            raise ValueError("beta must be finite and positive")
        features = torch.as_tensor(features)
        covariance = self.posterior_covariance_batched(features)
        context_count, candidate_count = covariance.shape[:2]
        remaining = torch.arange(candidate_count, device=features.device)[None].expand(
            context_count, -1
        ).clone()
        selected = [[] for _ in range(context_count)]
        traces = [[] for _ in range(context_count)]
        for _ in range(int(steps)):
            scores = (torch.diagonal(covariance, dim1=1, dim2=2) / (1.0 + self.lam)).clamp(
                0.0, 1.0
            )
            weights = torch.exp(
                ((scores - scores.max(dim=1, keepdim=True).values) / float(beta)).clamp(-30.0, 30.0)
            )
            probability = weights / weights.sum(dim=1, keepdim=True)
            local = torch.multinomial(probability, 1, generator=generator).squeeze(1)
            chosen = remaining.gather(1, local[:, None]).squeeze(1)
            probability64 = probability.detach().cpu().to(torch.float64)
            ess = 1.0 / (
                probability64.square().sum(dim=1) * probability64.shape[1]
            )
            for context in range(context_count):
                traces[context].append(dict(
                    remaining=remaining[context].detach().cpu(),
                    scores=scores[context].detach().cpu(),
                    probability=probability[context].detach().cpu(),
                    chosen=int(chosen[context]),
                    chosen_sigma=float(scores[context, local[context]].sqrt()),
                    ess_norm=float(ess[context]),
                ))
                selected[context].append(int(chosen[context]))
            if covariance.shape[1] == 1:
                covariance = covariance.new_zeros((context_count, 0, 0))
                remaining = remaining[:, :0]
                continue
            all_indices = torch.arange(covariance.shape[1], device=features.device)[None].expand(
                context_count, -1
            )
            keep = all_indices != local[:, None]
            indices = all_indices[keep].reshape(context_count, -1)
            reduced = covariance.gather(
                1, indices[:, :, None].expand(-1, -1, covariance.shape[2])
            ).gather(2, indices[:, None, :].expand(-1, indices.shape[1], -1))
            cross = covariance.gather(2, local[:, None, None].expand(-1, covariance.shape[1], 1))
            cross = cross.squeeze(2).gather(1, indices)
            denominator = covariance[
                torch.arange(context_count, device=features.device), local, local
            ].clamp_min(1.0e-12)
            covariance = reduced - cross[:, :, None] * cross[:, None, :] / denominator[:, None, None]
            covariance = 0.5 * (covariance + covariance.transpose(1, 2))
            remaining = remaining.gather(1, indices)
        return selected, traces

    @torch.no_grad()
    def diagnostics(self):
        if self.X is None:
            return dict(n=0, kernel_condition=1.0, kernel_effective_rank=0.0, jitter=None)
        kernel = self._kernel(self.X, self.X).to(torch.float64)
        eigenvalues = torch.linalg.eigvalsh(kernel).clamp_min(0.0)
        condition = float((eigenvalues.max() + self.lam) / (eigenvalues.min() + self.lam))
        effective_rank = float(eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1.0e-12))
        return dict(
            n=len(self.X), kernel_condition=condition,
            kernel_effective_rank=effective_rank, jitter=self.jitter,
        )


def normalized_ess(scores, beta):
    scores = torch.as_tensor(scores).detach().cpu().to(torch.float64)
    probability = torch.softmax((scores - scores.max()) / float(beta), dim=0)
    return float(1.0 / (probability.square().sum() * len(probability)))


def solve_beta(score_vectors, target=0.5, *, lower=1.0e-6, upper=10.0, iterations=80):
    vectors = [
        torch.as_tensor(value).detach().cpu().to(torch.float64)
        for value in score_vectors if len(value)
    ]
    if not vectors:
        raise ValueError("beta calibration has no score vectors")
    def objective(beta):
        return float(np.mean([normalized_ess(value, beta) for value in vectors]))
    low_value, high_value = objective(lower), objective(upper)
    if not (low_value <= float(target) <= high_value):
        raise ValueError(f"ESS target {target} not bracketed by [{low_value}, {high_value}]")
    low, high = float(lower), float(upper)
    for _ in range(int(iterations)):
        middle = math.sqrt(low * high)
        if objective(middle) < float(target):
            low = middle
        else:
            high = middle
    beta = math.sqrt(low * high)
    return beta, objective(beta)


def calibrate_beta(gp, feature_batches, *, B=4, target=0.5, seed=0):
    device = torch.as_tensor(feature_batches[0]).device
    generator = torch.Generator(device=device).manual_seed(int(seed))
    vectors = []
    stacked = torch.stack([torch.as_tensor(features, device=device) for features in feature_batches])
    covariance = gp.posterior_covariance_batched(stacked)
    context_count, candidate_count = covariance.shape[:2]
    remaining = torch.arange(candidate_count, device=device)[None].expand(context_count, -1).clone()
    orders = torch.stack([
        torch.randperm(candidate_count, generator=generator, device=device)
        for _ in range(context_count)
    ])
    for step in range(int(B)):
        scores = (torch.diagonal(covariance, dim1=1, dim2=2) / (1.0 + gp.lam)).clamp(0.0, 1.0)
        vectors.extend(scores[context] for context in range(context_count))
        chosen = orders[:, step]
        local = torch.stack([
            torch.nonzero(remaining[context] == chosen[context], as_tuple=False).flatten()[0]
            for context in range(context_count)
        ])
        if covariance.shape[1] == 1:
            break
        all_indices = torch.arange(covariance.shape[1], device=device)[None].expand(context_count, -1)
        keep = all_indices != local[:, None]
        indices = all_indices[keep].reshape(context_count, -1)
        reduced = covariance.gather(
            1, indices[:, :, None].expand(-1, -1, covariance.shape[2])
        ).gather(2, indices[:, None, :].expand(-1, indices.shape[1], -1))
        cross = covariance.gather(2, local[:, None, None].expand(-1, covariance.shape[1], 1))
        cross = cross.squeeze(2).gather(1, indices)
        denominator = covariance[
            torch.arange(context_count, device=device), local, local
        ].clamp_min(1.0e-12)
        covariance = reduced - cross[:, :, None] * cross[:, None, :] / denominator[:, None, None]
        covariance = 0.5 * (covariance + covariance.transpose(1, 2))
        remaining = remaining.gather(1, indices)
    return solve_beta(vectors, target)


def quantiles(values):
    values = np.asarray(values, float)
    if not len(values):
        return {key: None for key in ("q0", "q25", "q50", "q75", "q100")}
    return dict(zip(("q0", "q25", "q50", "q75", "q100"),
                    map(float, np.quantile(values, [0, .25, .5, .75, 1]))))


def acquisition_diagnostics(all_sigma, selected_sigma):
    all_sigma = np.asarray(all_sigma, float)
    selected_sigma = np.asarray(selected_sigma, float)
    uplift = float(np.median(selected_sigma) - np.median(all_sigma))
    return dict(all_K_sigma=quantiles(all_sigma), selected_B_sigma=quantiles(selected_sigma), uplift=uplift)


def choose_preflight(rows):
    """Choose maximum uplift, then the smallest cap retaining >=90% of cap-512 uplift."""
    stable = [row for row in rows if row.get("ess_solved") and row.get("stable_conditioning")]
    if not stable:
        raise RuntimeError("no solvable/stable RBF preflight candidate")
    by_ell = {}
    for row in stable:
        by_ell.setdefault(float(row["ell_multiplier"]), []).append(row)
    ell_scores = {ell: float(np.median([item["uplift"] for item in values]))
                  for ell, values in by_ell.items()}
    selected_ell = max(sorted(ell_scores), key=lambda ell: ell_scores[ell])
    candidates = by_ell[selected_ell]
    cap512 = next(item for item in candidates if int(item["cap"]) == 512)
    if float(cap512["uplift"]) <= 0.0:
        raise RuntimeError("cap-512 does not produce positive median uncertainty uplift")
    threshold = 0.9 * float(cap512["uplift"])
    retained = [item for item in candidates if float(item["uplift"]) >= threshold]
    selected = min(retained, key=lambda item: int(item["cap"]))
    return dict(selected=selected, ell_median_uplift=ell_scores, cap512_threshold=threshold)
