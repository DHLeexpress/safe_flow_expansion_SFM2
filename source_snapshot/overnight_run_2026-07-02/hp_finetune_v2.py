"""OVERNIGHT v2 pretrain (user 2026-07-06 'hopeless version' — OOD data folded into pretraining):
fine-tune res2w256_ft on the MERGED 4002-traj pool (2001 origin-start + 2001 off-diagonal-start, already
appended into dataset/windows_g*.pt) -> results/hp_arch/res2w256_ft_v2.pt (load_arch-compatible)."""
import os

import torch

import _paths  # noqa: F401
import hp_arch_sweep as ARCH

OUT = ARCH.OUT

pol, _ = ARCH.load_arch(os.path.join(OUT, "res2w256_ft.pt"), device=ARCH.DEV)
pol, bv = ARCH.train(pol, "res2w256_ft_v2", epochs=60, lr=1e-4)
ck = torch.load(os.path.join(OUT, "res2w256_ft_v2.pt"), map_location="cpu", weights_only=False)
ck["variant"] = "res2w256"
ck["finetuned_from"] = "res2w256_ft"
ck["data"] = "4002 trajs (2001 origin + 2001 offdiag |y-x|>=0.5)"
torch.save(ck, os.path.join(OUT, "res2w256_ft_v2.pt"))
print(f"[ft_v2] DONE best_val {bv:.4f} -> {os.path.join(OUT, 'res2w256_ft_v2.pt')}", flush=True)
