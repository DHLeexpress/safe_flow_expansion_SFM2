"""Train the reward-tilted conditional flow proposal q_θ(U|o,γ) by the
energy/reward-weighted CFM loss (EFM, arXiv:2503.04975).

Per training step we draw ONE control sequence from the stored top-K with
probability ∝ its MPPI weight w ∝ exp(-S/λ); regressing the rectified-flow target
on these reward-weighted draws makes the terminal marginal equal the tilt
p(U|o,γ) ∝ 1[U∈F] exp(-S/λ). Context is normalized and translation-invariant.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import torch
import torch.nn.functional as F
from cfm_mppi.models.tilted_flow import TiltedFlowProposal


def _load(paths):
    ctx, ctr, w, g = [], [], [], []
    H = None
    for p in paths:
        d = torch.load(p, weights_only=False)
        ctx.append(d["context"]); ctr.append(d["controls"]); w.append(d["weights"]); g.append(d["gamma"])
        H = d["meta"]["horizon"]
    return (torch.cat(ctx), torch.cat(ctr), torch.cat(w), torch.cat(g), H)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=["dataset/tilted/tilt_a.pt", "dataset/tilted/tilt_b.pt"])
    p.add_argument("--output-dir", default="output_dir/tilted_flow")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=384)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = p.parse_args()
    dev = torch.device(cli.device)

    ctx, ctr, w, g, H = _load(cli.data)
    S, K = w.shape
    mu = ctx.mean(0); sd = ctx.std(0).clamp_min(1e-3)
    ctx_n = ((ctx - mu) / sd).to(dev)
    ctr = ctr.to(dev); w = w.to(dev); g = g.to(dev)
    # NOVELTY: EXPLICIT PERTURBATION TARGET. Per step the nominal control is the
    # reward-weighted MPPI mean ū_s = Σ_k w_{s,k} U_{s,k}; the generator learns the
    # PERTURBATION δU = U − ū about that nominal (the "maximum perturbation
    # sequences we can apply"), not the absolute control. At inference we add ū back.
    nominal = (w.unsqueeze(-1).unsqueeze(-1) * ctr).sum(1) / w.sum(1).clamp_min(1e-8).view(S, 1, 1)  # [S,H,2]
    # REFINEMENT: ONE gamma-agnostic generator. Condition ONLY on obs/polytope
    # geometry (the nominal control + geometric constraints); gamma is NOT a
    # condition -- it only moves the rejection threshold (1-gamma)^i downstream.
    cond_dim = ctx.shape[1]
    model = TiltedFlowProposal(horizon=H, cond_dim=cond_dim, hidden=cli.hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cli.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cli.epochs)
    out = Path(cli.output_dir); out.mkdir(parents=True, exist_ok=True)
    # train/val split for the learning curve
    nval = max(1, S // 10)
    rp = torch.randperm(S, device=dev)
    vidx, tidx = rp[:nval], rp[nval:]
    Strain = tidx.numel()
    print(f"tilted data: {S} steps ({Strain} train / {nval} val) x {K} samples, H={H}, cond_dim={cond_dim} (gamma-agnostic)", flush=True)

    udim = H * 2

    def _fm_loss(idx, train=True):
        B = idx.numel()
        cond = ctx_n[idx]                                                # [B, cond_dim] obs/polytope geometry only
        j = torch.multinomial(w[idx].clamp_min(1e-8), 1).squeeze(1)      # ∝ tilt weight exp(-S/λ)
        U = ctr[idx, j]                                                  # [B,H,2]
        dU = U - nominal[idx]                                            # EXPLICIT perturbation target δU = U − ū
        x1 = dU.reshape(B, udim)
        x0 = torch.randn_like(x1)
        t = torch.rand(B, device=dev)
        xt = t.view(B, 1) * x1 + (1 - t.view(B, 1)) * x0
        pred = model(xt, t, cond)
        return F.mse_loss(pred, x1 - x0)

    nb = max(1, Strain // cli.batch_size)
    hist = []
    for epoch in range(cli.epochs):
        perm = tidx[torch.randperm(Strain, device=dev)]
        model.train(); tot = 0.0
        for bi in range(nb):
            idx = perm[bi * cli.batch_size:(bi + 1) * cli.batch_size]
            if idx.numel() == 0:
                continue
            loss = _fm_loss(idx, train=True)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += float(loss.detach())
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = float(sum(float(_fm_loss(vidx)) for _ in range(4)) / 4.0)  # avg over noise/weight draws
        rec = {"epoch": epoch, "train_loss": tot / nb, "val_loss": vl, "lr": sched.get_last_lr()[0]}
        hist.append(rec)
        if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == cli.epochs - 1:
            print(json.dumps(rec), flush=True)
    (out / "loss_history.json").write_text(json.dumps(hist))
    torch.save({"model": model.state_dict(), "ctx_mu": mu, "ctx_sd": sd, "horizon": H,
                "cond_dim": cond_dim, "hidden": cli.hidden, "gamma_agnostic": True,
                "perturbation_target": True}, out / "checkpoint.pth")
    print(f"saved {out}/checkpoint.pth  (+loss_history.json, {len(hist)} epochs)")


if __name__ == "__main__":
    main()
