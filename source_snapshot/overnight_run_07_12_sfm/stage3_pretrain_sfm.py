"""From-scratch, trajectory-disjoint Hp10 SFM pretraining and ID-only promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

import _paths  # noqa: F401
import grid_policy_sfm as GPS
import sfm_hp_history as HH
import sfm_protocol as SP

HERE = os.path.dirname(os.path.abspath(__file__))
INPUT = "/home/dohyun/projects/cfm_mppi/overnight_run_07_12_sfm"
DATASET = os.path.join(INPUT, "dataset_id_v01")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gamma_path(dataset, gamma):
    return os.path.join(dataset, f"sfm_windows_g{gamma}.pt")


def load_split(dataset, gammas=SP.GAMMAS, val_frac=0.1, seed=0):
    """Build Hp10 before splitting and retain trajectory IDs for exact sampler mass."""
    train_parts = [[] for _ in range(6)]
    val_parts = [[] for _ in range(5)]
    split_meta = {}
    for gamma_index, gamma in enumerate(gammas):
        path = _gamma_path(dataset, gamma)
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        if not payload.get("success_only", False):
            raise ValueError(f"dataset is not successful-only: {path}")
        episodes = payload["episode"].to(torch.int64)
        steps = payload["step"].to(torch.int64)
        hp10 = HH.build_hp10(payload["grid"], episodes, steps)
        unique = torch.unique(episodes, sorted=True)
        if len(unique) < 10:
            raise ValueError(f"gamma {gamma} has fewer than ten trajectories")
        generator = torch.Generator().manual_seed(int(seed) + 1009 * gamma_index + int(1000 * gamma))
        shuffled = unique[torch.randperm(len(unique), generator=generator)]
        n_val = max(1, int(round(len(unique) * float(val_frac))))
        val_episodes, train_episodes = shuffled[:n_val], shuffled[n_val:]
        val_mask = torch.isin(episodes, val_episodes)
        train_mask = ~val_mask
        source = (hp10, payload["low5"], payload["hist"], payload["U"])
        for index, tensor in enumerate(source):
            train_parts[index].append(tensor[train_mask].clone())
            val_parts[index].append(tensor[val_mask].clone())
        train_parts[4].append(episodes[train_mask].clone())
        train_parts[5].append(torch.full((int(train_mask.sum()),), gamma_index, dtype=torch.int64))
        val_parts[4].append(torch.full((int(val_mask.sum()),), gamma_index, dtype=torch.int64))
        split_meta[str(gamma)] = dict(
            file=os.path.abspath(path), sha256=sha256_file(path),
            train_episodes=sorted(map(int, train_episodes.tolist())),
            val_episodes=sorted(map(int, val_episodes.tolist())),
            train_windows=int(train_mask.sum()), val_windows=int(val_mask.sum()),
        )
    train = tuple(torch.cat(parts) for parts in train_parts)
    val = tuple(torch.cat(parts) for parts in val_parts)
    return train, val, split_meta


def hierarchical_sampler_weights(episodes, gamma_indices):
    """Uniform objective mass gamma -> successful trajectory -> window."""
    episodes = torch.as_tensor(episodes, dtype=torch.int64)
    gamma_indices = torch.as_tensor(gamma_indices, dtype=torch.int64)
    weights = torch.zeros(len(episodes), dtype=torch.float64)
    gammas = torch.unique(gamma_indices, sorted=True)
    for gamma in gammas:
        gamma_mask = gamma_indices == gamma
        trajectories = torch.unique(episodes[gamma_mask], sorted=True)
        for trajectory in trajectories:
            mask = gamma_mask & (episodes == trajectory)
            weights[mask] = 1.0 / (len(gammas) * len(trajectories) * int(mask.sum()))
    if not torch.isclose(weights.sum(), torch.tensor(1.0, dtype=weights.dtype), atol=1.0e-10):
        raise RuntimeError("pretraining hierarchical mass does not sum to one")
    return weights


@contextmanager
def preserve_rng():
    python_state, numpy_state, torch_state = random.getstate(), np.random.get_state(), torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


@torch.no_grad()
def validation(policy, val, gammas, device, batch=1024, seed=41017):
    policy.eval()
    totals = np.zeros(len(gammas), np.float64)
    counts = np.zeros(len(gammas), np.int64)
    loader = DataLoader(TensorDataset(*val), batch_size=int(batch), shuffle=False, num_workers=0)
    with preserve_rng():
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        for hp10, low, hist, controls, gamma_index in loader:
            hp10, low, hist, controls = [value.to(device) for value in (hp10, low, hist, controls)]
            for index in torch.unique(gamma_index).tolist():
                mask = gamma_index == int(index)
                count = int(mask.sum())
                loss = policy.cfm_loss(
                    controls[mask], policy.ctx_from(hp10[mask], low[mask], hist[mask])
                )
                totals[index] += float(loss) * count
                counts[index] += count
    per_gamma = {str(gamma): float(totals[index] / counts[index]) for index, gamma in enumerate(gammas)}
    return float(np.mean(list(per_gamma.values()))), per_gamma


def atomic_save(payload, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + ".tmp"
    torch.save(payload, temporary)
    os.replace(temporary, path)


def checkpoint_payload(policy, epoch, history, args, split_meta):
    return dict(
        state_dict={name: value.detach().cpu() for name, value in policy.state_dict().items()},
        config=policy.config(), epoch=int(epoch), history=history, args=vars(args), split_meta=split_meta,
        initialization="from_scratch", partial_transplant=False,
    )


def id_raw_gate(policy, *, m_per_gamma, ep0, device, nfe=8):
    """Fixed ID raw temp=1 gate; this function imports no OOD bank."""
    from sfm_b1_eval import raw_rollout
    rows = []
    policy.eval()
    for gamma in SP.GAMMAS:
        for episode in range(int(ep0), int(ep0) + int(m_per_gamma)):
            rows.append(raw_rollout(
                policy, episode, gamma, device=device, nfe=int(nfe), temp=1.0,
                ped_speed_range=(0.5, 1.0), sample_seed=710_000,
            ))
    collision = sum(row["collision"] for row in rows) / len(rows)
    success = sum(row["success"] for row in rows) / len(rows)
    return dict(
        bank="fixed trajectory-disjoint ID raw temp=1", ep0=int(ep0), M_per_gamma=int(m_per_gamma),
        nfe=int(nfe), pooled_CR=collision, pooled_SR=success,
        per_gamma={str(gamma): dict(
            CR=float(np.mean([row["collision"] for row in rows if row["gamma"] == gamma])),
            SR=float(np.mean([row["success"] for row in rows if row["gamma"] == gamma])),
        ) for gamma in SP.GAMMAS},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--policy-out", required=True)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--val-batch", type=int, default=1024)
    parser.add_argument("--samples-per-epoch", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20260720)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--ckpt-every", type=int, default=10)
    parser.add_argument("--gate-m", type=int, default=10)
    parser.add_argument("--gate-top", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train, val, split_meta = load_split(args.dataset, SP.GAMMAS, args.val_frac, args.seed)
    weights = hierarchical_sampler_weights(train[4], train[5])
    samples_per_epoch = int(args.samples_per_epoch) or len(train[0])
    policy = GPS.build_sfm_policy(grid_shape=(10, 16, 12), device=args.device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: ((epoch + 1) / max(1, args.warmup) if epoch < args.warmup else
                       0.5 * (1.0 + math.cos(math.pi * min(
                           1.0, (epoch - args.warmup) / max(1, args.epochs - args.warmup)
                       )))),
    )
    history = []
    candidates = []
    for epoch in range(args.epochs):
        generator = torch.Generator().manual_seed(args.seed * 100003 + epoch)
        sampler = WeightedRandomSampler(
            weights, samples_per_epoch, replacement=True, generator=generator
        )
        loader = DataLoader(
            TensorDataset(*train[:4]), batch_size=args.batch, sampler=sampler,
            drop_last=True, num_workers=args.num_workers,
        )
        policy.train()
        losses = []
        for hp10, low, hist, controls in loader:
            hp10, low, hist, controls = [value.to(args.device) for value in (hp10, low, hist, controls)]
            loss = policy.cfm_loss(controls, policy.ctx_from(hp10, low, hist))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        scheduler.step()
        macro, per_gamma = validation(policy, val, SP.GAMMAS, args.device, args.val_batch)
        row = dict(epoch=epoch + 1, train_cfm=float(np.mean(losses)), val_macro_cfm=macro,
                   val_per_gamma=per_gamma, lr=optimizer.param_groups[0]["lr"])
        history.append(row)
        if (epoch + 1) % args.ckpt_every == 0 or epoch + 1 == args.epochs:
            path = os.path.join(args.outdir, f"ckpt_{epoch + 1}.pt")
            atomic_save(checkpoint_payload(policy, epoch + 1, history, args, split_meta), path)
            candidates.append((macro, path))
        print(json.dumps(row), flush=True)
    # Validation defines candidates; the fixed ID raw gate promotes among only the best validation candidates.
    gates = []
    for macro, path in sorted(candidates)[:max(1, int(args.gate_top))]:
        candidate, _ = GPS.load_sfm_policy(path, device=args.device)
        gate = id_raw_gate(candidate, m_per_gamma=args.gate_m, ep0=SP.PRETRAIN_GATE_EP0, device=args.device)
        gate.update(path=os.path.abspath(path), val_macro_cfm=float(macro))
        gates.append(gate)
    selected = min(gates, key=lambda row: (
        max(value["CR"] for value in row["per_gamma"].values()), row["pooled_CR"],
        -min(value["SR"] for value in row["per_gamma"].values()), -row["pooled_SR"],
        row["val_macro_cfm"],
    ))
    promoted, checkpoint = GPS.load_sfm_policy(selected["path"], device="cpu")
    GPS.save_sfm_policy(promoted, args.policy_out, extra=dict(
        selected_by="trajectory-disjoint ID validation + fixed ID raw temp=1 gate",
        selected_gate=selected, all_id_gates=gates, split_meta=split_meta,
        dataset=os.path.abspath(args.dataset), dataset_manifest_sha256=sha256_file(os.path.join(args.dataset, "manifest.json")),
        pretrained_from_scratch=True, partial_transplant=False,
    ))
    report = dict(status="PRETRAIN_COMPLETE", policy=os.path.abspath(args.policy_out),
                  policy_sha256=sha256_file(args.policy_out), selected=selected, gates=gates,
                  split_meta=split_meta, sampler_mass="gamma -> successful trajectory -> window")
    with open(os.path.join(args.outdir, "pretraining_report.json"), "w") as stream:
        json.dump(report, stream, indent=2)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
