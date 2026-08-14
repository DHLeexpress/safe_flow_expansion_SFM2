# VERIFIER_POLYTOPE.md - Compact SOCP Polytope Verifier

This note defines the Pillar 3 verifier polytope for a local MPPI 10-step forward rollout. The verifier is designed to certify a **class of local answers**, not one nominal path. The current target is a compact verifier polytope whose level sets contain the queried rollout and whose faces are valid tangent/separating supports for sensed circular obstacles and artificial sensing-boundary anchors.

**Core statement:** the fixed-normal verifier is an LP, while the variable circular tangent-face verifier is an SOCP.
**Practical statement:** positive weights alone do not reshape independent faces; explicit margin bounds such as `m_min <= m_i <= m_max` do.

---

## 1. Local trajectory and level-set certificate

Let

\[
q_0,q_1,\ldots,q_H,\qquad H=10,\qquad c=q_0
\]

be one local MPPI forward rollout from the current robot state. For a CBF parameter \(\gamma\), define

\[
\alpha_t=(1-\gamma)^t,\qquad \beta_t=1-\alpha_t.
\]

A verifier face is written in robot-centered form

\[
a_i^\top(x-c)\le m_i.
\]

The full verifier polytope is

\[
P=\{x:a_i^\top(x-c)\le m_i,\ i=1,\ldots,N\}.
\]

For one face, the normalized level-set value is

\[
h_i(x)=\frac{m_i-a_i^\top(x-c)}{m_i}.
\]

The recursive/level-set certificate requires

\[
h_i(q_t)\ge \alpha_t,\qquad t=1,\ldots,H.
\]

This is equivalent to the linear constraint

\[
a_i^\top(q_t-c)\le \beta_t m_i.
\]

**If every face satisfies this inequality, then \(H_P(q_t)=\min_i h_i(q_t)\ge \alpha_t\) for the whole polytope.**

---

## 2. Circular obstacle tangent/separation condition

Assume obstacle \(i\) is a disk

\[
O_i=\{x:\|x-o_i\|_2\le r_i\}.
\]

To place the disk outside the verifier polytope face, require

\[
a_i^\top(x-c)\ge m_i\qquad\forall x\in O_i.
\]

The minimum support of a disk in direction \(a_i\) is

\[
\min_{x\in O_i}a_i^\top(x-c)=a_i^\top(o_i-c)-r_i\|a_i\|_2.
\]

Therefore the exact disk-separation condition is

\[
a_i^\top(o_i-c)-r_i\|a_i\|_2\ge m_i,
\]

or

\[
r_i\|a_i\|_2\le a_i^\top(o_i-c)-m_i.
\]

**Equality is geometric tangency; strict inequality is a separating support with extra clearance.**

---

## 3. Compact SOCP verifier formulation

For one variable face per sensed obstacle or artificial boundary anchor, solve

\[
\begin{aligned}
\max_{\{a_i,m_i\}}\quad
&\sum_i w_i m_i\\
\text{s.t.}\quad
&a_i^\top(q_t-c)\le \beta_t m_i,
&&\forall i,\ t=1,\ldots,H,\\
&r_i\|a_i\|_2\le a_i^\top(o_i-c)-m_i,
&&\forall i,\\
&\|a_i\|_2\le 1,
&&\forall i,\\
&m_i\ge m_{\min},
&&\forall i.
\end{aligned}
\]

This is an SOCP because:

- \(a_i^\top(q_t-c)\le\beta_t m_i\) is linear.
- \(r_i\|a_i\|_2\le a_i^\top(o_i-c)-m_i\) is a second-order cone constraint.
- \(\|a_i\|_2\le1\) is a second-order cone constraint.
- \(m_i\ge m_{\min}\) is linear.
- The objective \(\sum_i w_i m_i\) is linear.

The SOC ball constraint

\[
\|a_i\|_2\le1
\]

replaces the nonconvex equality

\[
\|a_i\|_2=1.
\]

For the positive max-margin case \(w_i>0\), this relaxation is not loose when the margin is not capped above. If a feasible solution has \(m_i>0\) and \(0<\|a_i\|_2<1\), scaling \((a_i,m_i)\mapsto(s a_i,s m_i)\) preserves all homogeneous constraints until \(\|a_i\|_2=1\), and improves the objective.

**When an upper bound \(m_i\le m_{\max}\) is active, the optimizer may hit the margin cap before the unit-normal ball saturates. The SOCP remains valid, but the no-looseness argument must be read with this cap in mind.**

---

## 4. Margin-bounded SOCP for tube-like verifier polytopes

To explicitly cap the half-width of verifier faces, add the linear upper bound

\[
m_i\le m_{\max}.
\]

The compact margin-bounded verifier is

\[
\begin{aligned}
\max_{\{a_i,m_i\}}\quad
&\sum_i w_i m_i\\
\text{s.t.}\quad
&a_i^\top(q_t-c)\le \beta_t m_i,\\
&r_i\|a_i\|_2\le a_i^\top(o_i-c)-m_i,\\
&\|a_i\|_2\le 1,\\
&m_{\min}\le m_i\le m_{\max}.
\end{aligned}
\]

**The upper bound \(m_{\max}\) is the direct geometry knob that turns a pointy max-margin wedge into a capped, tube-like verifier polytope.**

This bound is linear, so the problem remains an SOCP. It is different from changing positive weights. Positive weights rescale independent face objectives; \(m_{\max}\) changes the feasible set.

---

## 5. Fixed-normal LP as a special case

If every \(a_i\) is fixed, then the only decision variables are \(m_i\). The trajectory constraint

\[
a_i^\top(q_t-c)\le\beta_t m_i
\]

is linear in \(m_i\), and the obstacle tangent bound for a fixed unit normal becomes

\[
m_i\le a_i^\top(o_i-c)-r_i.
\]

So the fixed-normal verifier is an LP. **The variable-normal circular verifier is the SOCP above.**

---

## 6. Positive weights do not shape independent faces

If each obstacle or artificial anchor gets its own independent face, then the feasible set decomposes:

\[
\mathcal C=\mathcal C_1\times\cdots\times\mathcal C_N.
\]

Therefore

\[
\max_{(a_i,m_i)\in\mathcal C_i}\sum_i w_i m_i
=\sum_i\max_{(a_i,m_i)\in\mathcal C_i}w_i m_i.
\]

For any positive weight \(w_i>0\),

\[
\arg\max w_i m_i=\arg\max m_i.
\]

**Therefore positive weight ratios do not change the geometry in the separable one-face-per-anchor max-margin SOCP. They only rescale the scalar objective value.**

To make weights change shape, the problem needs coupling, signed penalties, shared margin budgets, target tube-width objectives, fixed parallel tube faces, or obstacle-to-face assignment variables.

---

## 7. Artificial boundary obstacles

The artificial boundary anchors use the same circular-obstacle SOCP template.

Given sensing range \(R\) and \(K\) artificial directions,

\[
n_\ell=
\begin{bmatrix}
\cos(2\pi\ell/K)\\
\sin(2\pi\ell/K)
\end{bmatrix},
\qquad \ell=0,\ldots,K-1.
\]

The inscribed \(K\)-gon apothem is

\[
M_K=R\cos(\pi/K).
\]

Choose an artificial obstacle radius \(\rho_{\rm art}\), and place

\[
o_\ell^{\rm art}=c+(M_K+\rho_{\rm art})n_\ell,
\qquad
r_\ell^{\rm art}=\rho_{\rm art}.
\]

Then the radial tangent face has margin

\[
m_\ell=n_\ell^\top(o_\ell^{\rm art}-c)-\rho_{\rm art}=M_K.
\]

**Artificial obstacles let the verifier solve the outer sensing envelope through the same tangent-face SOCP instead of using a separate hard-coded polygon.**

---

## 8. Demonstration settings

Canonical narrow-gap setup:

```text
robot/reference center c = (0, 0)
sensing range R = 2.0
horizon H = 10
trajectory q_t = (1.28 * t/H, 0.035 * sin(pi*t/H)), t=0,...,H
real obstacle 0 = center (0.78,  0.55), radius 0.35
real obstacle 1 = center (0.78, -0.55), radius 0.35
```

Artificial boundary anchors for the margin-bound demo:

```text
K values: 4, 8, 16
gamma values: 0.3, 0.5, 0.8
rho_art = 0.16
M_K = R*cos(pi/K)
o_art[ell] = (M_K + rho_art) * [cos(2*pi*ell/K), sin(2*pi*ell/K)]
r_art[ell] = rho_art
```

Positive-weight invariance demo:

```text
weight mode A: w_real = 0.01,  w_art = 100.0
weight mode B: w_real = 100.0, w_art = 0.01
```

Margin-bound demo:

```text
loose m-bounds: m_min = 0.03, m_max = 5.00
tight m-bounds: m_min = 0.03, m_max = 0.40
```

**In the generated 6x3 figure, the tight bound \(m_{\max}=0.40\) clips both real and artificial faces and produces the expected tube-like verifier geometry.**

---

## 9. Demonstration result summary

The margin-bound figure uses 6 rows and 3 columns:

- columns: \(\gamma=0.3,0.5,0.8\);
- rows: \(K=4,8,16\), each repeated for loose and tight margin bounds.

Mean values over the three gamma columns:

| K | mode | mean area | mean solve time (ms) | mean real margin | mean artificial margin |
|---:|---|---:|---:|---:|---:|
| 4 | loose | 2.559 | 178.08 | 0.426 | 1.414 |
| 4 | tight | 0.863 | 172.93 | 0.400 | 0.400 |
| 8 | loose | 3.363 | 297.29 | 0.426 | 1.848 |
| 8 | tight | 0.808 | 292.99 | 0.400 | 0.400 |
| 16 | loose | 3.497 | 540.94 | 0.426 | 1.962 |
| 16 | tight | 0.798 | 565.22 | 0.400 | 0.400 |

The implementation provided for the 2-D demo uses a dense angular feasibility scan to visualize the same circular tangent-face conditions. A production implementation can call a conic solver directly for the SOCP.

---

## 10. Code files

- `src/demo_verifier_polytope.py`: earlier narrow-gap and weight-invariance demonstrations.
- `src/pillar3_m_bounds_6x3.py`: margin-bound 6x3 demonstration with artificial obstacles shown explicitly.
- `paper/compact_polytope_verifier.tex`: IEEE-style compact note.
- `tex/verifier_polytope_socp_proofs.tex`: standalone proof resource.
- `paper/figures/*.png`: self-contained image assets used by the TeX source.

Run the margin-bound figure:

```bash
python src/pillar3_m_bounds_6x3.py
```
