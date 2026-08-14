from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from cfm_mppi.data import CanonicalDataset, canonical_collate
from cfm_mppi.models.contextual_transformer import ContextualTransformerModel, count_parameters
from cfm_mppi.training.train_loop_safe_cfm import evaluate_safe_cfm, train_one_epoch_safe_cfm


def _write_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()


def _save_checkpoint(path: Path, model, optimizer, epoch: int, args, val_loss: float, param_count: int) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "val_loss": val_loss,
            "parameter_count": param_count,
        },
        path,
    )


def get_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train proposed safe contextual CFM.")
    p.add_argument("--train-data", default="dataset/canonical/train.pt")
    p.add_argument("--val-data", default="dataset/canonical/val.pt")
    p.add_argument("--output-dir", default="output_dir/safe_contextual_cfm")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--history-len", type=int, default=10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--test-run", action="store_true")
    # convergence-based termination + experiment tracking
    p.add_argument("--patience", type=int, default=40,
                   help="early stop after this many epochs without val-loss improvement")
    p.add_argument("--min-delta", type=float, default=1e-4,
                   help="minimum relative val-loss improvement to reset patience")
    p.add_argument("--lr-schedule", default="cosine", choices=["none", "cosine"],
                   help="cosine annealing LR over epochs")
    p.add_argument("--wandb-mode", default="offline", choices=["offline", "online", "disabled"])
    p.add_argument("--wandb-project", default="cfm-mppi-safe")
    p.add_argument("--wandb-name", default=None)
    return p


def main() -> None:
    args = get_parser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "args.json").open("w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    train_path = Path(args.train_data)
    val_path = Path(args.val_data)
    if not train_path.exists():
        raise FileNotFoundError(f"Missing canonical train split: {train_path}. Run scripts/build_canonical_dataset.sh first.")
    if not val_path.exists():
        raise FileNotFoundError(f"Missing canonical val split: {val_path}. Run scripts/build_canonical_dataset.sh first.")

    train_ds = CanonicalDataset(train_path)
    val_ds = CanonicalDataset(val_path)
    collate = partial(canonical_collate, random_truncate=True, min_horizon=10)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=partial(canonical_collate, random_truncate=False),
        drop_last=False,
    )
    device = torch.device(args.device)
    model = ContextualTransformerModel.from_mizuta_defaults(history_len=args.history_len).to(device)
    param_count = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
                 if args.lr_schedule == "cosine" else None)

    # Weights & Biases (offline by default: no API key needed; `wandb sync` later to upload)
    run = None
    try:
        import os
        import wandb
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)
        run = wandb.init(project=args.wandb_project, name=args.wandb_name,
                         mode=args.wandb_mode, dir=str(out), config=vars(args))
    except Exception as exc:  # wandb missing/broken -> fall back to jsonl-only
        print(f"[wandb] disabled ({exc})", flush=True)

    best = float("inf")
    patience_left = args.patience
    log_path = out / "train_log.jsonl"
    for epoch in range(args.epochs):
        train_stats = train_one_epoch_safe_cfm(model, train_loader, optimizer, device, grad_clip=args.grad_clip)
        val_stats = evaluate_safe_cfm(model, val_loader, device)
        val_loss = val_stats["loss"]
        cur_lr = optimizer.param_groups[0]["lr"]
        if scheduler is not None:
            scheduler.step()
        latest = out / "checkpoint_latest.pth"
        _save_checkpoint(latest, model, optimizer, epoch, args, val_loss, param_count)
        improved = val_loss < best * (1.0 - args.min_delta)
        if improved:
            best = val_loss
            patience_left = args.patience
            _save_checkpoint(out / "checkpoint_best.pth", model, optimizer, epoch, args, val_loss, param_count)
        else:
            patience_left -= 1
        record = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_loss,
            "best_val_loss": best,
            "gradient_norm": train_stats["gradient_norm"],
            "lr": cur_lr,
            "patience_left": patience_left,
            "epoch_time": train_stats["epoch_time"],
            "checkpoint_path": str(latest),
            "best_checkpoint_path": str(out / "checkpoint_best.pth"),
            "parameter_count": param_count,
        }
        _write_jsonl(log_path, record)
        if run is not None:
            run.log({k: v for k, v in record.items() if isinstance(v, (int, float))}, step=epoch)
        print(json.dumps(record), flush=True)
        if args.test_run:
            break
        if patience_left <= 0:
            print(f"[early-stop] no val-loss improvement for {args.patience} epochs "
                  f"(best={best:.6f}); terminating at epoch {epoch}.", flush=True)
            break
    if run is not None:
        run.summary["best_val_loss"] = best
        run.finish()


if __name__ == "__main__":
    main()
