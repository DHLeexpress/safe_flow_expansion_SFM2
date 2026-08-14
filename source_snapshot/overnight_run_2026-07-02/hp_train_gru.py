"""Pretrain res2w256_GRU on the 4002-traj dataset (user 2026-07-06): same ResTrunk as v2 but with the GRU over
past executed controls ENABLED (ctx 37->53). Trains from scratch (GRU can't warm-start from the no-GRU v2).
Saves results/hp_arch/res2w256_gru.pt (variant-tagged, load_arch-compatible) + curve; runs the standard eval.
Compare val-cfm and validity2 to res2w256_ft_v2 (val-cfm 0.810, validity2@it0 n=50 = 77% [76/78/76])."""
import _paths  # noqa: F401
import hp_arch_sweep as ARCH

pol = ARCH.build("res2w256_gru")
print(f"[gru] params {sum(p.numel() for p in pol.parameters())/1e3:.1f}k · ctx_dim {pol.ctx_dim} · use_gru {pol.use_gru}", flush=True)
pol, bv = ARCH.train(pol, "res2w256_gru", epochs=120, lr=3e-4)
ARCH.evaluate(pol, "res2w256_gru")
print(f"[gru] DONE best val-cfm {bv:.4f} -> results/hp_arch/res2w256_gru.pt", flush=True)
