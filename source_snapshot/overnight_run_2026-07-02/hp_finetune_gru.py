"""Fine-tune res2w256_gru (user 2026-07-06): continue the GRU variant at lower lr on the 4002-traj set ->
results/hp_arch/res2w256_gru_ft.pt (variant='res2w256_gru' so load_arch rebuilds the GRU arch). Re-eval."""
import os
import torch

import _paths  # noqa: F401
import hp_arch_sweep as ARCH

OUT = ARCH.OUT
pol, _ = ARCH.load_arch(os.path.join(OUT, "res2w256_gru.pt"), device=ARCH.DEV)
pol, bv = ARCH.train(pol, "res2w256_gru_ft", epochs=60, lr=1e-4)
ck = torch.load(os.path.join(OUT, "res2w256_gru_ft.pt"), map_location="cpu", weights_only=False)
ck["variant"] = "res2w256_gru"                 # buildable variant for load_arch (train saved the ft name)
ck["finetuned_from"] = "res2w256_gru"
torch.save(ck, os.path.join(OUT, "res2w256_gru_ft.pt"))
ARCH.evaluate(pol, "res2w256_gru_ft")
print(f"[gru_ft] DONE best val-cfm {bv:.4f} -> results/hp_arch/res2w256_gru_ft.pt", flush=True)
