"""Fine-tune res2w256 on the enlarged ~2000-traj dataset (user 2026-07-05): continue from the arch-sweep winner
at lower lr (1e-4, 60 ep, cosine) — dataset picked up automatically (windows_g*.pt now includes seeds 0-666).
Saves results/hp_arch/res2w256_ft.pt (variant key kept loader-compatible) and runs the standard eval
(validity2@it0 n=25/γ + mm-splits)."""
import os

import torch

import hp_arch_sweep as ARCH

OUT = ARCH.OUT

pol, _ = ARCH.load_arch(os.path.join(OUT, "res2w256.pt"), device=ARCH.DEV)
pol, bv = ARCH.train(pol, "res2w256_ft", epochs=60, lr=1e-4)
ck = torch.load(os.path.join(OUT, "res2w256_ft.pt"), map_location="cpu", weights_only=False)
ck["variant"] = "res2w256"          # keep load_arch-compatible (same architecture)
ck["finetuned_from"] = "res2w256"
torch.save(ck, os.path.join(OUT, "res2w256_ft.pt"))
val, mm = ARCH.evaluate(pol, "res2w256_ft")
print(f"[ft] DONE best_val {bv:.4f}", flush=True)
