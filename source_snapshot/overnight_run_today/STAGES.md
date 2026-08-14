# The stages, in plain language (with every symbol defined)

This explains the SafeFlow Exploration pipeline the way the GIFs show it, defining each symbol the first
time it appears. Figures are in `figures/<env>_stageN_*.{png,gif}` for `env ∈ {single, gap}`.

## Notation (defined once, plain words)
- **environment / scene** `c` — the fixed problem: where the robot **starts**, where the **goal** is, and the
  **obstacles** (each a circle: center + radius). `single` = one obstacle on the straight line; `gap` = two
  stacked obstacles with a narrow passable slot between them.
- **control sequence** `U` — the robot's planned accelerations for the whole horizon, `U = (u_0, …, u_{T-1})`,
  `T = 40` steps. This is the *thing the generative model outputs* (a whole plan at once, not one step).
- **trajectory** — the path you get by simulating the robot (a double integrator) forward under `U`. `p_t` is the
  robot position at step `t`.
- **clearance barrier** `h` — how safe a position is: `h = (distance to the nearest obstacle) − (its radius +
  robot radius)`. `h > 0` means safe (outside the obstacle), `h = 0` is touching, `h < 0` is a collision.
  `h_0` = clearance at the start.
- **γ (gamma)** — the **conservativeness knob** (0–1). The safety rule (DTCBF) says clearance may shrink over
  time but no faster than the schedule `h(step i) ≥ (1−γ)^i · h_0`. Small γ = must keep a wide berth
  (conservative); larger γ = allowed to pass closer (less conservative). We cap it at `γ_max = 0.7`.
- **the "ruler"** — that schedule `(1−γ)^i · h_0` drawn as nested level-set curves (orange in the GIFs). A plan is
  **certified safe** if its clearance stays above the ruler at every step.
- **verified polytope** — at each step the safe side of the nearest obstacle is a half-plane (a flat wall tangent to
  the obstacle); the collection of these walls is the local polytope, and the ruler curves are its level sets. The
  **verifier** is the check "does a valid polytope+γ certify this plan?" (here a fast closed-form test).
- **FM policy** `q(U | c)` — the **flow-matching generative model** that produces control sequences. "Flow
  matching" = a generative model that turns random noise into a sample by following a learned velocity field; it can
  represent **multi-modal** distributions (e.g. left *and* right).
- **mode (homotopy class)** — which way the plan goes around the obstacles: `single` → {LEFT, RIGHT};
  `gap` → {LEFT, GAP(through the middle), RIGHT}.
- **coverage** — what fraction of the truly-reachable safe behaviors the FM policy actually produces (0–100%).
- **validity** — what fraction of the FM's sampled plans are certified safe **and** reach the goal (0–100%).
- **mode-coverage** — how many of the modes (LEFT/RIGHT/[GAP]) the policy generates with non-negligible
  probability, as a fraction.
- **Vendi** — a diversity score (think "effective number of distinct behaviors").

---

## Stage 0 — seed → expanded  (`stage0_seed_vs_expanded.png`, static)
Two panels on the clearance-field background (blue = far from obstacle, dark line = `h=0` boundary). **Left:** the
**seed** FM — conservative, only one leaf (e.g. always pass right). **Right:** the **expanded** FM — all certified
modes, hugging the free space (and threading the gap). Red dotted = the deterministic **conservative candidate
polytope** (`polytope.py`); the expanded policy deliberately goes *beyond* it (= less conservative). This is the
before/after summary of the whole method.

## Stage 1 — SafeMPPI with the ruler  (`stage1_safemppi_ruler.gif`, the data engine)
The classic sample-then-reject controller, animated. Each step it samples many candidate plans, **rejects** any that
break the ruler `h(x_i) ≥ (1−γ)^i h_0` (**red ✗**), keeps the rest (**green**), and averages the survivors into the
executed path (red line). Orange curves = the ruler. This produces the **conservative, one-leaf safe plans** that we
distill into the seed FM. *This is the "do SafeMPPI, put the ruler there" stage.*

## Stage 2 — the FM field under the ruler  (`stage2_fm_field_certified.gif`, generative policy)
Now the FM *itself* generates plans (no MPPI). One subplot per mode. As the robot moves along a generated plan, we
overlay the **ruler** (orange level set, which loosens over the horizon) and the **verified-polytope faces** (red
tangent walls + outward normal arrow) that certify it. The text box shows the live clearance `h`, the ruler
threshold `(1−γ)^t h_0`, and the certifying `γ_req`. *This shows "how the generative policy works, with the ruler."*

## Stage 3 — Safe Flow Expansion  (`stage3_safeflow_expansion.gif`, the learning loop)
The headline animation over expansion **rounds**. **Left panel:** the FM's sampled plans, colored by mode, with the
fixed conservative candidate polytope (red dotted) for reference — you watch the seed's single leaf **open into all
modes** and grow *beyond* the conservative polytope (the effective safe region changes / "the polytope changes").
**Right panel:** coverage / validity / mode-coverage filling in round by round. *This shows "after applying Safe
Flow Expansion, it fits the safe set better."*

---

## How to read success
- `single`: seed = right-only → expanded = both sides; coverage and mode-coverage rise to full; validity recovers.
- `gap`: seed = one side → expanded = left / through-gap / right; all three modes present; the gap mode is the
  narrow one, so it sits at a few percent but is real.

(Next phase, not done here: swap the toy scene for pedestrian data and compare against Mizuta — the same four
stages, same metrics.)
