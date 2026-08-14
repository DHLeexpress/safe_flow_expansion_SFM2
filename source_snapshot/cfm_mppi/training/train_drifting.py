from __future__ import annotations

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cfm_mppi.data import CanonicalDataset, canonical_collate
from cfm_mppi.models.contextual_transformer import count_parameters
from cfm_mppi.models.drifting_generator import DriftingGenerator
from cfm_mppi.training.drift_loss_torch import drifting_loss


def _write_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def _controls(batch):
    return batch["controls_si"].float().transpose(1, 2).contiguous()


def _loss_step(model, batch, device):
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    target = _controls(batch)
    noise = torch.randn_like(target)
    gen = model.forward_batch(noise, batch)
    # Fallback negative: the initial noise sequence, documented in conflicts.
    return drifting_loss(gen, target, fixed_neg=noise), gen


def _eval(model, loader, device):
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            loss, _ = _loss_step(model, batch, device)
            total += float(loss.detach().cpu())
            n += 1
    return total / max(n, 1)


def get_parser():
    p = argparse.ArgumentParser(description="Train Drifting-style one-step generator.")
    p.add_argument("--train-data", default="dataset/canonical/train.pt")
    p.add_argument("--val-data", default="dataset/canonical/val.pt")
    p.add_argument("--output-dir", default="output_dir/drifting_generator")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--history-len", type=int, default=10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--test-run", action="store_true")
    return p


def main():
    args = get_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)
    if not Path(args.train_data).exists():
        raise FileNotFoundError(f"Missing canonical train split: {args.train_data}.")
    if not Path(args.val_data).exists():
        raise FileNotFoundError(f"Missing canonical val split: {args.val_data}.")
    collate = partial(canonical_collate, random_truncate=True, min_horizon=10)
    train_loader = DataLoader(
        CanonicalDataset(args.train_data),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        CanonicalDataset(args.val_data),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(canonical_collate, random_truncate=False),
    )
    device = torch.device(args.device)
    model = DriftingGenerator.from_mizuta_defaults(history_len=args.history_len).to(device)
    param_count = count_parameters(model)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = float("inf")
    log_path = out / "train_log.jsonl"
    for epoch in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        total = 0.0
        n = 0
        grad_norm_value = 0.0
        for batch in train_loader:
            loss, _ = _loss_step(model, batch, device)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            grad_norm_value = float(grad_norm.detach().cpu())
            opt.step()
            total += float(loss.detach().cpu())
            n += 1
        train_loss = total / max(n, 1)
        val_loss = _eval(model, val_loader, device)
        ckpt = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "val_loss": val_loss,
            "parameter_count": param_count,
            "nfe": 1,
        }
        latest = out / "checkpoint_latest.pth"
        torch.save(ckpt, latest)
        if val_loss < best:
            best = val_loss
            torch.save(ckpt, out / "checkpoint_best.pth")
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "gradient_norm": grad_norm_value,
            "lr": opt.param_groups[0]["lr"],
            "epoch_time": time.perf_counter() - t0,
            "checkpoint_path": str(latest),
            "best_checkpoint_path": str(out / "checkpoint_best.pth"),
            "parameter_count": param_count,
            "nfe": 1,
        }
        _write_jsonl(log_path, record)
        print(json.dumps(record), flush=True)
        if args.test_run:
            break


if __name__ == "__main__":
    main()
