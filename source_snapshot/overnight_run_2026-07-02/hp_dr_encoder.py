"""PHASE-DR side-quest (user 2026-07-06): adapt the H_P CNN+AAP encoder on domain-randomized-start data,
with the velocity field FROZEN — the mirror image of expansion (there we freeze the encoder and move the field).

Train: only enc_grid.* params get gradients; trunk/head frozen -> the encoder must map NEW off-diagonal H_P
patterns into representations the frozen field already knows how to act on (CFM loss on DR windows).
Report per epoch: val-cfm on DR split AND val-cfm on the ORIGINAL dataset (compatibility gauge — must not rot).
Outputs:
  results/hp_arch/enc_hp_dr.pt       adapted encoder (enc_grid.* state_dict)
  results/hp_arch/res2w256_dr.pt     res2w256_ft with the encoder SPLICED in (loader-compatible ckpt)
(The untouched original encoder is archived at results/hp_arch/enc_hp_original.pt.)
"""
from __future__ import annotations

import argparse
import math
import os

import torch
from torch.utils.data import DataLoader, TensorDataset

import _paths  # noqa: F401
import hp_arch_sweep as ARCH

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = os.path.join(HERE, "results", "hp_arch")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_pool(prefix):
    G, L, Hh, U = [], [], [], []
    for g in ("0.1", "0.5", "1.0"):
        d = torch.load(os.path.join(HERE, "dataset", f"{prefix}windows_g{g}.pt"))
        G.append(d["grid"]); L.append(d["low5"]); Hh.append(d["hist"]); U.append(d["U"])
    return tuple(torch.cat(x) for x in (G, L, Hh, U))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(OUTD, "res2w256_ft.pt"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--mix-orig", type=float, default=0.0,
                    help="fraction of each batch drawn from the ORIGINAL windows (fallback lever)")
    ap.add_argument("--mode", choices=("enc", "full"), default="enc",
                    help="enc = THE METHOD (encoder only, trunk/head frozen). full = ORACLE baseline: train "
                         "everything on DR data — saved for reference lines ONLY, never the deployed method "
                         "(trunk trained on expert demos of new modes = unfaithful to safe expansion).")
    args = ap.parse_args()

    pol, meta = ARCH.load_arch(args.base, device=DEV)
    enc_only = args.mode == "enc"
    for n, p in pol.named_parameters():
        p.requires_grad_(True if not enc_only else n.startswith("enc_grid"))
    enc_pars = [p for n, p in pol.named_parameters() if n.startswith("enc_grid")]
    train_pars = enc_pars if enc_only else list(pol.parameters())
    print(f"[dr-{args.mode}] trainable {sum(p.numel() for p in train_pars)/1e3:.1f}k / "
          f"total {sum(p.numel() for p in pol.parameters())/1e3:.1f}k params", flush=True)

    G, L, Hh, U = load_pool("dr_")
    n = G.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(0))
    G, L, Hh, U = G[perm], L[perm], Hh[perm], U[perm]
    nval = max(2048, n // 10)
    tr = TensorDataset(G[nval:], L[nval:], Hh[nval:], U[nval:])
    va = (G[:nval].to(DEV), L[:nval].to(DEV), Hh[:nval].to(DEV), U[:nval].to(DEV))
    dl = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True)

    oG, oL, oH, oU = load_pool("")                      # original data: compatibility val + optional mix
    og = torch.Generator().manual_seed(1); oidx = torch.randperm(oG.shape[0], generator=og)[:4096]
    ova = (oG[oidx].to(DEV), oL[oidx].to(DEV), oH[oidx].to(DEV), oU[oidx].to(DEV))
    print(f"[dr-enc] DR windows {n} ({n-nval} train) · orig-val 4096 · mix_orig {args.mix_orig}", flush=True)

    opt = torch.optim.AdamW(train_pars, lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda ep: (ep + 1) / args.warmup if ep < args.warmup else
        0.5 * (1 + math.cos(math.pi * (ep - args.warmup) / max(1, args.epochs - args.warmup))))
    best = (float("inf"), None, -1)
    for ep in range(args.epochs):
        pol.train(); tot = nb = 0
        for gb, lb, hb, ub in dl:
            if args.mix_orig > 0:
                k = int(round(args.mix_orig * gb.shape[0]))
                if k:
                    j = torch.randint(0, oG.shape[0], (k,))
                    gb = torch.cat([gb[k:], oG[j]]); lb = torch.cat([lb[k:], oL[j]])
                    hb = torch.cat([hb[k:], oH[j]]); ub = torch.cat([ub[k:], oU[j]])
            loss = pol.cfm_loss(ub.to(DEV), pol.ctx_from(gb.to(DEV), lb.to(DEV), hb.to(DEV)))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        sched.step()
        pol.eval()
        with torch.no_grad():
            torch.manual_seed(0)
            vdr = float(pol.cfm_loss(va[3], pol.ctx_from(va[0], va[1], va[2])))
            vor = float(pol.cfm_loss(ova[3], pol.ctx_from(ova[0], ova[1], ova[2])))
        if vdr < best[0]:
            keep = (lambda k: k.startswith("enc_grid")) if enc_only else (lambda k: True)
            best = (vdr, {k: v.detach().cpu().clone() for k, v in pol.state_dict().items() if keep(k)}, ep)
        print(f"ep {ep:03d}  train {tot/nb:.4f}  val-DR {vdr:.4f}  val-ORIG {vor:.4f}"
              + ("  *best" if best[2] == ep else ""), flush=True)

    base = torch.load(args.base, map_location="cpu", weights_only=False)
    if enc_only:
        enc_out = os.path.join(OUTD, "enc_hp_dr.pt")
        torch.save(dict(enc_state=best[1], best_val_dr=best[0], best_ep=best[2],
                        base=os.path.basename(args.base), mix_orig=args.mix_orig), enc_out)
        sd = dict(base["state_dict"]); sd.update(best[1])
        spliced = dict(base); spliced["state_dict"] = sd
        spliced["dr_adapted"] = True; spliced["dr_val"] = best[0]; spliced["enc_source"] = "enc_hp_dr.pt"
        spl_out = os.path.join(OUTD, "res2w256_dr.pt")
        torch.save(spliced, spl_out)
        print(f"[dr-enc] saved encoder -> {enc_out}\n[dr-enc] saved SPLICED model -> {spl_out}", flush=True)
    else:
        trunk = {k: v for k, v in best[1].items() if k.startswith(("trunk", "head"))}
        tr_out = os.path.join(OUTD, "trunk_hp_dr.pt")
        torch.save(dict(trunk_head_state=trunk, best_val_dr=best[0], best_ep=best[2],
                        note="ORACLE trunk (trained on DR expert demos) — reference only, NOT the method"), tr_out)
        oracle = dict(base); oracle["state_dict"] = dict(best[1])
        oracle["oracle_dr_full"] = True; oracle["dr_val"] = best[0]
        or_out = os.path.join(OUTD, "res2w256_drfull.pt")
        torch.save(oracle, or_out)
        print(f"[dr-full] saved ORACLE trunk -> {tr_out}\n[dr-full] saved ORACLE model -> {or_out}\n"
              f"[dr-full] use for reference lines only (its it0 coverage = the ceiling)", flush=True)


if __name__ == "__main__":
    main()
