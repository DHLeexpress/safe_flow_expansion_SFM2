# SafeFlow Exploration — overnight_run_today

> **한 줄 요약 (KR).** 보수적인 deterministic polytope(`polytope.py`)에 갇힌 SafeMPPI를 *씨앗(seed)* 으로 삼아,
> ACTFLOW (arXiv:2606.08802) 의 *Generable-Set Expansion* 원리를 제어/DTCBF 세팅에 이식한다.
> Flow-Matching(FM) policy 가 **delta-U 가 아니라 control sequence 통째로**를 생성하고,
> **verifier(=polytope+DTCBF certificate 를 찾는 최적화)** 가 safe 라고 판정한 sample 만 대량으로 모아
> FM 을 재학습 → **DTCBF-safe 하지만 덜 보수적(less conservative)** 인 *multi-modal* policy 로 확장한다.
> 오늘의 목표: single-obstacle / two-obstacle-narrow-gap 두 fixed 환경에서 shallow FM 을 overfit 시켜
> left/right, left/mid/right multi-modality 와 **coverage–validity** 곡선(ACTFLOW Fig 재현)을 그린다.

This folder realizes the **SafeFlow Exploration Loop** (the Figma board concept). The board is JS/auth-gated so
it could not be machine-read; this plan is reconstructed from the written spec + the ACTFLOW paper + the repo.

---

## 1. The architecture (the Figma layers, as a diagram)

```mermaid
flowchart TB
    subgraph L0["TOP LAYER — Problem instance (context c)"]
        ctx["start x0 · goal g · obstacles {position p_o, velocity v_o, radius r}"]
    end

    subgraph L1["CONTROLLER LAYER — Safe MPPI (the data engine / seed)"]
        mppi["MPPI rollout (double integrator)"]
        dtcbf["Polytope-based DTCBF rejection / mirror-sampling\n(candidate polytope P_cand from polytope.py — DETERMINISTIC, conservative)"]
        safeU["safe MPPI control SEQUENCES  U = (u_0..u_{T-1})"]
        mppi --> dtcbf --> safeU
    end

    subgraph L2["POLICY LAYER — Flow-Matching policy q_theta(U | c)"]
        fm["FM velocity net v_theta(U_tau, tau, c)\nlearns WHOLE control sequences (multi-modal), NOT delta-U"]
        phis["phi_s : hidden feature @ noise level s  ->  representation Z_s  (for Eq.10)"]
        fm --- phis
    end

    subgraph L3["CERTIFICATE LAYER — candidate polytope <-> verifier"]
        cand["candidate polytope P_cand (reference)"]
        ver["VERIFIER = optimization:\n find a polytope P* + DTCBF (gamma) certifying a sampled U\n  (1) verified polytope + its level sets {h>=(1-gamma)^i}"]
        cand <--> ver
    end

    subgraph L4["LEARNING LOOP — Safe Flow Expansion (ACTFLOW)"]
        explore["Eq.9 active exploration: tilt FM samples by uncertainty sigma\n q* ∝ q_theta · exp(sigma/beta)"]
        sigma["Eq.10 sigma_t over phi_s (GP / Bayesian-linear posterior var)"]
        query["query verifier -> label y ∈ {safe, unsafe}"]
        buffer["buffer D_t += (U, y)   (bulk-collect SAFE samples)"]
        update["UpdateFlow: CFM loss on SAFE (+) minus alpha_t * UNSAFE (-)\n=> (3) new FM policy respecting NEW (less-conservative) polytope"]
        explore --> query --> buffer --> update
        sigma --> explore
        update -->|theta_{t+1} -> phi_s, sigma re-fit| sigma
    end

    ctx --> mppi
    safeU -->|distill: pretrain theta_0| fm
    fm --> explore
    ver -->|"(1) judge safeMPPI-trained FM output safe?"| query
    cand -->|reference geometry| ver
    update -->|expanded q_theta| fm
    fm -->|"(2) sample under surrounding constraints"| ver
```

**Reading the loop (matches the spec's (1)(2)(3)):**

- **(1) Judge safety.** The verifier finds a *verified polytope and its level sets* and uses it to decide whether a
  control sequence emitted by the (SafeMPPI-seeded) FM policy is safe.
- **(2) Safe Flow Expansion.** Under the surrounding obstacle constraints, FM samples are judged by the verifier;
  the SAFE ones are bulk-collected and used to update the FM policy.
- **(3) Less-conservative policy.** The result is a *new* FM policy that respects a *new* polytope shape (not the
  deterministic `polytope.py` corridor) and therefore generates **DTCBF-safe but less conservative** trajectories
  that hug the true free space (e.g. thread a narrow gap).

> **Note on "/figma 로 그려봐".** I cannot push nodes to your Figma *board* — there is no Figma write tool in this
> environment (the `DesignSync` tool targets `claude.ai/design` design-systems, a different product). The mermaid
> diagram above + `design/` docs are the faithful machine-readable version. If you want it as an interactive
> artifact I can render it into a `claude.ai/design` project via DesignSync, or export an SVG — tell me which.

---

## 2. Why this is the right reframe of ACTFLOW

| ACTFLOW (molecules/proteins) | SafeFlow Exploration (control)                                              |
|------------------------------|----------------------------------------------------------------------------|
| design `x`                   | control sequence `U=(u_0..u_{T-1})` → trajectory `xi(U;x0)` for a fixed env |
| valid space `Omega*`         | **DTCBF-certifiable safe** sequences (less conservative than `P_cand`)      |
| generable set `Omega^tau_θ`  | trajectories the FM policy actually samples for that env                    |
| verifier `v(x)∈{0,1}`        | **optimization**: ∃ polytope `P*`+`gamma` certifying `U`?  (see `design/VERIFIER.md`) |
| `phi_s` (noised-flow repr.)  | hidden feature of FM velocity net at noise level `s`                        |
| Eq.9 active exploration      | uncertainty-tilted sampling from FM policy                                  |
| Eq.10 sigma                  | GP / Bayesian-linear posterior variance over `phi_s`                        |
| coverage / validity / Vendi  | mode/descriptor coverage / verifier pass-rate / Vendi (see `design/METRICS.md`) |

Full term-by-term definitions: **`design/SAFEFLOW_GLOSSARY.md`** (with the extra-care Eq.9 & Eq.10 treatment the
spec asked for, plus an iterative self-diagnosis checklist).

---

## 3. Today's concrete experiment (spec 3.3)

Two **fixed** 2-D double-integrator environments (overfit, shallow model):

- **ENV-A `single`** — one obstacle on the start→goal line ⇒ *left/right dilemma* (bimodal Omega*).
- **ENV-B `gap`** — two stacked obstacles with a narrow but DTCBF-passable gap (when `gamma` is non-conservative)
  ⇒ *left / middle / right trilemma* (trimodal Omega*).

Pipeline (`run_today.py`):
1. **Seed/pretrain** `theta_0`: distill SafeMPPI/mirror-sampled SAFE sequences that respect the conservative
   `P_cand`. Deliberately collapse to ONE leaf (e.g. "go right") to mimic ACTFLOW's narrow pretrained mode.
2. **Safe Flow Expansion**: run the ACTFLOW loop (Eq.9 sampling → verifier → buffer → UpdateFlow) for `T` rounds.
3. **Evaluate & plot**:
   - multi-modal trajectory overlays (left/right; left/mid/right) at rounds {0, mid, final},
   - **coverage–validity curve** vs round (ACTFLOW-style),
   - **diversity (Vendi)** vs round,
   - **self-diagnosis** of Eq.9/Eq.10 (`results/diagnostics.json`).

## 4. File map

```
overnight_run_today/
├── PLAN.md                        # this file
├── design/
│   ├── SAFEFLOW_GLOSSARY.md       # term-by-term defs + Eq.9 & Eq.10 deep-dive + self-diagnosis  ← core
│   ├── VERIFIER.md                # verifier-as-optimization, complexity, 3D scaling
│   └── METRICS.md                 # coverage / validity / diversity for trajectories
├── src/
│   ├── dynamics.py                # double integrator + ENV-A/ENV-B definitions
│   ├── dtcbf.py                   # affine DTCBF barrier + polytope + VERIFIER (LP)
│   ├── flow_policy.py             # conditional FM over control SEQUENCES + phi_s hook
│   ├── uncertainty.py             # Eq.10 sigma (GP posterior var; linear & RBF kernels)
│   ├── descriptors.py             # trajectory descriptor + coverage / Vendi
│   ├── safeflow.py                # the ACTFLOW loop (Eq.9, UpdateFlow, diagnostics)
│   └── plots.py                   # overlays + coverage-validity curve
├── run_today.py                   # entrypoint: pretrain -> expand -> plot (ENV-A & ENV-B)
├── results/                       # metrics json, diagnostics, checkpoints
└── figures/                       # png/gif outputs
```

## 5. How to run

```bash
cd /home/dohyun/projects/cfm_mppi
python overnight_run_today/run_today.py --env single --rounds 60 --device cuda
python overnight_run_today/run_today.py --env gap    --rounds 80 --device cuda
# smoke test:
python overnight_run_today/run_today.py --env single --rounds 4 --smoke
```
