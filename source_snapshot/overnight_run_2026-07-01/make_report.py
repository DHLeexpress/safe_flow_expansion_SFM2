"""Render a two-column IEEE-style implementation report (PDF) explaining Safe Flow Expansion end-to-end.
No LaTeX engine on this box -> equations are rendered with matplotlib mathtext and embedded; body via reportlab.
The CFM-details section is written in Kazuki style (bold Remarks).
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER, TA_LEFT
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, Image,
                                Table, TableStyle, FrameBreak, NextPageTemplate, KeepTogether)

import _paths
import config as C

HERE = os.path.dirname(os.path.abspath(__file__))
EQDIR = os.path.join(C.FIGURES, "_eq"); os.makedirs(EQDIR, exist_ok=True)
FIG = os.path.join(HERE, "figures")
PAPERIMG = os.path.join(C.ROOT, "Dohyun_ICRA2026_SafeFlowExpansion", "images")
OUT = os.path.join(C.FIGURES, "SafeFlowExpansion_report.pdf")
COL = 504.0  # single-column content width (pt) — keeps the detailed grid/feature figures legible

# ---------------- equation rendering (matplotlib mathtext -> png) ----------------
_eqn = [0]
def eqpng(tex, fs=13, pad=0.04):
    tex = tex.replace(r"\big", "").replace(r"\Big", "")   # matplotlib mathtext lacks \big/\Big
    _eqn[0] += 1
    p = os.path.join(EQDIR, f"eq{_eqn[0]}.png")
    fig = plt.figure(figsize=(0.1, 0.1))
    fig.text(0.5, 0.5, f"${tex}$", fontsize=fs, ha="center", va="center")
    fig.savefig(p, dpi=240, bbox_inches="tight", pad_inches=pad, facecolor="white")
    plt.close(fig)
    return p

def imgflow(path, frac=0.98, maxw=COL, align="CENTER"):
    ir = ImageReader(path); iw, ih = ir.getSize()
    w = min(maxw * frac, iw * 72.0 / 240.0)  # 240 dpi -> pt
    im = Image(path, width=w, height=w * ih / iw); im.hAlign = align
    return im

# ---------------- fonts (DejaVu Serif has full Greek + math glyphs; base-14 Times does not) ----------------
import matplotlib as _mpl
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
_fdir = os.path.join(os.path.dirname(_mpl.__file__), "mpl-data", "fonts", "ttf")
try:
    pdfmetrics.registerFont(TTFont("DV", os.path.join(_fdir, "DejaVuSerif.ttf")))
    pdfmetrics.registerFont(TTFont("DV-B", os.path.join(_fdir, "DejaVuSerif-Bold.ttf")))
    pdfmetrics.registerFont(TTFont("DV-I", os.path.join(_fdir, "DejaVuSerif-Italic.ttf")))
    pdfmetrics.registerFont(TTFont("DV-BI", os.path.join(_fdir, "DejaVuSerif-BoldItalic.ttf")))
    pdfmetrics.registerFont(TTFont("DVM", os.path.join(_fdir, "DejaVuSansMono.ttf")))
    pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")
    FONT, FB, FI, FBI, FMONO = "DV", "DV-B", "DV-I", "DV-BI", "DVM"
except Exception:
    FONT, FB, FI, FBI, FMONO = "Times-Roman", "Times-Bold", "Times-Italic", "Times-BoldItalic", "Courier"

# ---------------- styles ----------------
ss = getSampleStyleSheet()
TITLE = ParagraphStyle("T", parent=ss["Title"], fontName=FB, fontSize=16, leading=19, alignment=TA_CENTER)
SUBT = ParagraphStyle("ST", parent=ss["Normal"], fontName=FI, fontSize=10.5, leading=12.5, alignment=TA_CENTER)
AUTH = ParagraphStyle("AU", parent=ss["Normal"], fontName=FONT, fontSize=9, leading=11.5, alignment=TA_CENTER)
ABS = ParagraphStyle("AB", parent=ss["Normal"], fontName=FONT, fontSize=9, leading=11.8, alignment=TA_JUSTIFY)
H1 = ParagraphStyle("H1", parent=ss["Normal"], fontName=FB, fontSize=12, leading=14.5, spaceBefore=11, spaceAfter=3)
H2 = ParagraphStyle("H2", parent=ss["Normal"], fontName=FBI, fontSize=10, leading=12.5, spaceBefore=6, spaceAfter=1)
BODY = ParagraphStyle("B", parent=ss["Normal"], fontName=FONT, fontSize=9.3, leading=12.3, alignment=TA_JUSTIFY, spaceAfter=4)
CAP = ParagraphStyle("C", parent=ss["Normal"], fontName=FI, fontSize=8.2, leading=10, alignment=TA_JUSTIFY, spaceAfter=6, spaceBefore=1)

def P(t): return Paragraph(t, BODY)
def RMK(label, t): return Paragraph(f"<b>Remark ({label}).</b> {t}", BODY)
def fig_block(path, caption, frac=0.98):
    if not os.path.exists(path):
        return P(f"[missing figure: {os.path.basename(path)}]")
    return KeepTogether([imgflow(path, frac=frac), Paragraph(caption, CAP)])
def eq_block(tex, fs=12.5, frac=0.9):
    return KeepTogether([Spacer(1, 2), imgflow(eqpng(tex, fs=fs), frac=frac), Spacer(1, 2)])

story = [NextPageTemplate("later")]

# ===================== TITLE / ABSTRACT (full-width frame) =====================
story += [Paragraph("Safe Flow Expansion: A Detailed Implementation Report", TITLE),
          Paragraph("Windowed &#947;-conditioned flow-matching control with a compact-polytope certificate "
                    "and certified generable-set expansion", SUBT),
          Spacer(1, 3),
          Paragraph("Implementation notes for <font face='Courier'>overnight_run_2026-07-01</font>. "
                    "Notation follows ActiveFlowExpansion (AFE, arXiv:2606.08802); our own symbols are defined on first use.", AUTH),
          Spacer(1, 5),
          Paragraph("<b>Abstract —</b> We turn a sampling-based planner (SafeMPPI) into a safe, safety-tunable "
                    "<i>data engine</i>, distil it into a real-time conditional flow-matching (CFM) policy, attach a "
                    "second-order-cone (SOCP) <i>polytope certificate</i> so that a single certified sample proves a "
                    "trajectory safe, and then <i>actively expand</i> the policy's certified generable set following the "
                    "AFE recipe. This report defines every abbreviation, states the problem, and explains — with the "
                    "actual equations and figures — how the flow feature &#966;<sub>s</sub> is computed, what the kernel / "
                    "&#963;-histogram / certified-coverage panels of the 2&#215;2 video mean, what the model parameters "
                    "are, how the polytope maps to the conditioning context, and why validity can fall as coverage rises.", ABS),
          FrameBreak()]

# ===================== I. GLOSSARY =====================
story += [Paragraph("I.&nbsp;&nbsp;Abbreviations and Symbols", H1)]
gloss = [
    ("FM / CFM", "Flow Matching / Conditional FM: learn a velocity field that transports noise to data (here, to control windows), conditioned on observations."),
    ("MPPI", "Model Predictive Path Integral control: sampling-based planner; the action is a reward-weighted average of noisy rollouts."),
    ("DCBF / DTCBF", "(Discrete-Time) Control Barrier Function: a scalar h whose one-step decay h(x_{k+1}) &#8805; (1&#8722;&#947;)h(x_k) keeps the safe set forward-invariant."),
    ("SafeMPPI / MPPI-DCBF", "Our planner: MPPI that rejects any rollout violating the polytope DCBF, so only certified-safe samples are averaged."),
    ("SOCP", "Second-Order Cone Program: the convex problem that fits the max-margin safe polytope (the certificate)."),
    ("AFE", "ActiveFlowExpansion (arXiv:2606.08802). <b>AFE-9 / AFE-10 = Equations (9) / (10) of that paper</b> (defined in &#167;V)."),
    ("GP", "Gaussian Process (here Bayesian linear regression in a feature space) used to estimate verifier uncertainty &#963;."),
    ("ESS", "Effective Sample Size of a weighted set: ESS = 1/&#931;w&#772;&#178;; ESS&#8776;N means the weights are ~uniform."),
    ("OT / ODE", "Optimal-Transport probability path / Ordinary Differential Equation integrated at inference to turn noise into a control window."),
    ("H_pred / H_exec", "Prediction window length (the FM outputs H_pred controls) / executed controls per step (receding horizon)."),
    ("&#947;", "Safety knob &#8712;(0,1]: small &#947; = conservative (small safe set), large &#947; = permissive."),
    ("&#956;<sub>&#952;</sub>, p<sub>1</sub><sup>&#952;</sup>", "The flow (velocity) network with parameters &#952;, and its generation law (samples drawn by the ODE)."),
    ("o, c, &#966;<sub>s</sub>", "o = polar occupancy grid [3,16,12]; c = low-dim context [7]; &#966;<sub>s</sub> = noised-flow feature (&#167;IV)."),
    ("&#937;*", "The set of certifiable trajectories (estimated by broad proposals); its cells are the coverage denominator."),
    ("k, K<sub>t</sub>, &#963;<sub>t</sub>", "Kernel k(x,x')=&#10216;&#966;<sub>s</sub>(x),&#966;<sub>s</sub>(x')&#10217;; Gram matrix K<sub>t</sub>; verifier-uncertainty &#963;<sub>t</sub>."),
]
rows = [[Paragraph(f"<b>{a}</b>", ParagraphStyle('g', parent=BODY, fontSize=8, leading=9.5)),
         Paragraph(d, ParagraphStyle('g2', parent=BODY, fontSize=8, leading=9.5, spaceAfter=0))] for a, d in gloss]
tbl = Table(rows, colWidths=[54, COL - 60])
tbl.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.grey),
                         ("TOPPADDING", (0, 0), (-1, -1), 1.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 1.5)]))
story += [tbl]

# ===================== II. PROBLEM STATEMENT =====================
story += [Paragraph("II.&nbsp;&nbsp;Problem Statement", H1),
          P("A robot with discrete-time control-affine dynamics must reach a goal through a cluttered scene. We want a "
            "<b>single generative policy</b> that (i) is <b>safe</b> (collision-free with a certificate, not just a soft "
            "penalty), (ii) is <b>tunable</b> online by one scalar &#947; trading safety against reach, and (iii) is "
            "<b>multi-modal</b> — it can produce the qualitatively distinct safe maneuvers around obstacles (e.g. pass "
            "below, weave through the gap, or go over the top). A sampling planner (MPPI) is unsafe because it averages "
            "samples; a plain generative policy inherits safety only statistically and tends to collapse to one "
            "conservative mode. Our pipeline (Fig.&nbsp;1) fixes both: a certified data engine, a distilled policy, a "
            "per-trajectory certificate, and an expansion loop that grows the set of certified modes."),
          fig_block(os.path.join(FIG, "schematic_overview.png"),
                    "Fig. 1. End-to-end pipeline: Stage&nbsp;2 SafeMPPI data engine &#8594; Stage&nbsp;3 &#947;-flow policy "
                    "&#8594; Stage&nbsp;4 SOCP certificate &#8594; Stage&nbsp;5 Safe Flow Expansion loop.", frac=1.0)]

# ===================== III. WHY A CLEAN RESTART (WINDOWED) =====================
story += [Paragraph("III.&nbsp;&nbsp;Why a Clean Restart: Windowed Generation", H1),
          P("<b>The problem with one-shot generation.</b> The previous build generated an entire ~80&#215;2 trajectory in "
            "one shot &#8212; a ~160-dimensional design. The SOCP verifier then discarded about <b>96%</b> of samples "
            "(raw validity &#8776;4%): a single long sample rarely satisfies the safety geometry everywhere at once."),
          P("<b>The fix (SOTA diffusion/flow-policy practice).</b> The FM predicts a short <b>control window</b> "
            "U<sub>t:t+H_pred</sub> instead of the whole path, deployed receding-horizon (execute H_exec=1, re-plan). "
            "Each of 55 episodes becomes 55&#215;80 low-dimensional windows, so the per-sample dimension is small and the "
            "certifiable fraction rises sharply. The <b>observation is what the SafeMPPI expert actually sees</b>: a "
            "robot-centered, goal-aligned polar polytope-occupancy grid [3,16,12] plus a low-dim local-frame state, and "
            "both the state and the target controls are expressed in a <b>goal-aligned local frame</b> (rotate "
            "world&#8594;local to train, local&#8594;world at inference)."),
          P("<b>Staged, approval-gated rollout.</b> (1) narrow-gap sanity &#8594; (2) windowed dataset (&#8805;100 "
            "episodes &#215; &#947;&#8712;{0.1,0.5,1.0}) &#8594; (3) policy + safe expansion &#8594; (4) clutter &#8594; "
            "(5) pedestrians. Coverage and validity are kept as <b>swappable modules</b> (the input space is infinite, so "
            "coverage is inherently a choice of metric). Reported live to Weights &amp; Biases.")]

# ===================== IV. CFM POLICY (KAZUKI STYLE) =====================
story += [Paragraph("IV.&nbsp;&nbsp;The Generative Policy (CFM details)", H1),
          P("We learn a velocity field &#956;<sub>&#952;</sub> that, integrated from Gaussian noise, produces a control "
            "window. Training uses the conditional flow-matching loss along the straight optimal-transport path "
            "x<sub>&#964;</sub>=(1&#8722;&#964;)&#949;+&#964;U:"),
          eq_block(r"\mathcal{L}(\theta)=\mathbb{E}_{\tau,\epsilon,(U,o,\gamma,c)}\big\|\,u^{\theta}_{\tau}(x_\tau\mid o,\gamma,c)-(U-\epsilon)\,\big\|^2", fs=12),
          RMK("what is learned", "the network regresses the <b>velocity</b> (U&#8722;&#949;) that carries a noise sample "
              "to the expert control window U; at inference we integrate this field with an ODE, so <b>no online sampling</b> "
              "of rollouts is needed."),
          Paragraph("A.&nbsp;Model parameters &#952; (defined cleanly)", H2),
          P("The policy is <font face='Courier'>GridLowFlowPolicy</font>. Its parameters &#952; are four sub-networks:"),
          P("&#8226; <b>grid encoder</b> E<sub>g</sub>: flatten o&#8712;&#8477;<sup>3&#215;16&#215;12</sup> (576) "
            "&#8594; 256 &#8594; 96 (SiLU). &#8226; <b>low encoder</b> E<sub>&#8467;</sub>: [7] &#8594; 64 &#8594; 48. "
            "The context is c&#772; = [E<sub>&#8467;</sub>(&#8467;); E<sub>g</sub>(o)] &#8712; &#8477;<sup>144</sup>. "
            "&#8226; <b>velocity trunk+head</b> u<sup>&#952;</sup>: input [flattened window U/u_max (20) &#8853; noise-time "
            "&#964; &#8853; c&#772;] &#8594; MLP width 256, depth 3 (SiLU) &#8594; head &#8594; &#8477;<sup>H_pred&#215;2</sup>. "
            "&#8226; <b>safety decoder</b> D<sub>g</sub>: 96 &#8594; 256 &#8594; 576 (reconstructs o; &#167;IV-C)."),
          P("<b>Window length.</b> H_pred = 10 controls per window (the FM output and the MPPI plan length); H_exec = 1 "
            "executed per step; the verifier window is 10. &#947;&#8712;{0.1,0.5,1.0}."),
          Paragraph("B.&nbsp;How &#966;<sub>s</sub> (the flow feature) is evaluated", H2),
          P("&#966;<sub>s</sub>(U) is the <b>noised-flow representation</b>: we noise the window to level s along the OT "
            "path, U<sub>s</sub>=(1&#8722;s)&#949;+sU, pass it through the velocity trunk at noise-time &#964;=s, and read "
            "the <b>penultimate hidden activation</b> (the width-256 vector just before the output head), averaged over a "
            "few fixed noise seeds. We use s=0.9."),
          eq_block(r"\phi_s(U)=\mathrm{trunk}_\theta\big((1-s)\epsilon+sU,\;\tau{=}s,\;\bar c\big)\in\mathbb{R}^{256}", fs=12),
          RMK("what the feature is", "&#966;<sub>s</sub> is <b>how the trained network internally represents a window</b> "
              "at a moderately-noised level &#8212; not the raw controls. Two windows the policy treats as the same "
              "maneuver map to nearby &#966;<sub>s</sub>; two different modes map far apart. This is the feature the AFE "
              "kernel and uncertainty (&#167;V) operate on."),
          fig_block(os.path.join(FIG, "slalom", "diag_feature.png"),
                    "Fig. 2. <b>Flow feature made concrete.</b> Three queried control windows (steer down / straight / up) "
                    "at the SAME conditioning c&#772; &#8594; their noised-flow input U<sub>s</sub> (dashed) &#8594; the "
                    "feature &#966;<sub>s</sub>(U)&#8712;&#8477;<sup>256</sup> (bars, visibly different) &#8594; the kernel "
                    "k=&#10216;&#966;<sub>s</sub>,&#966;<sub>s</sub>'&#10217; (right): down&#8596;up are least similar "
                    "(0.88), straight sits between (0.93/0.97).", frac=1.0),
          Paragraph("C.&nbsp;How the polytope maps to the context (the &#8216;polytope&#8594;context&#8217; map)", H2),
          P("The grid o encodes the free-space polytope the SafeMPPI expert sees: channel&nbsp;1 = obstacle occupancy, "
            "channel&nbsp;2 = polytope face mask, channel&nbsp;3 = clipped barrier value H<sub>P</sub>, on a robot-centered "
            "polar grid (16 angles &#215; 12 radii) aligned to the goal. The grid encoder E<sub>g</sub> compresses o to a "
            "96-d token that conditions the velocity field. To <b>force</b> that token to carry the safety geometry (not "
            "whatever the CFM loss happens to need), we add an auxiliary reconstruction head D<sub>g</sub> and train"),
          eq_block(r"\mathcal{L}_{\mathrm{aux}}(\theta)=\big\|\,D_g\big(E_g(o)\big)-o\,\big\|^2,\qquad \mathcal{L}_{\mathrm{tot}}=\mathcal{L}+0.3\,\mathcal{L}_{\mathrm{aux}}", fs=12),
          RMK("does it work?", "yes &#8212; the aux reconstruction error falls to <b>&#8776;0.01 (~1% MSE)</b> in both "
              "scenes (&#167;VII), i.e. the 96-d token reconstructs the safety grid almost perfectly. The encoder "
              "genuinely captures the polytope; it is not ignored by the CFM head."),
          fig_block(os.path.join(FIG, "slalom", "diag_context.png"),
                    "Fig. 3. <b>Context encoding made concrete.</b> Three environments (top) &#8594; their polar safety "
                    "grid o[H_P] (middle; green=safe, red=blocked) &#8594; the context vector "
                    "c&#772;=[E<sub>low</sub>(&#8467;)<sub>48</sub> | E<sub>grid</sub>(o)<sub>96</sub>]&#8712;&#8477;<sup>144</sup> "
                    "(bottom, split at 48). Different environments yield visibly different context vectors.", frac=1.0),
          fig_block(os.path.join(FIG, "slalom", "diag_safety_recon.png"),
                    "Fig. 4. <b>The context truly stores the safety geometry.</b> Input safety grid (left 3 columns) vs. "
                    "its reconstruction from the 96-d grid token via D<sub>g</sub> (right 3), for states of increasing "
                    "danger; aux MSE &#8776;1%.", frac=1.0)]

# ===================== V. ACTIVE FLOW EXPANSION =====================
story += [Paragraph("V.&nbsp;&nbsp;Active Flow Expansion (AFE-9/10, kernel, &#963;-tilt)", H1),
          P("The pretrained policy is conservative (often one mode). <b>Safe Flow Expansion</b> grows its <i>certified</i> "
            "generable set by continued pre-training on self-generated, verifier-accepted trajectories &#8212; the AFE "
            "recipe with the verifier instantiated as our SOCP certificate. The verifier is"),
          Paragraph("<i>&#7811;(x)=1</i> &#8660; x is collision-free &#8743; goal-reaching &#8743; the SOCP is feasible "
                    "(a certifying polytope exists).",
                    ParagraphStyle("vc", parent=BODY, alignment=TA_CENTER, fontName=FI, spaceBefore=3, spaceAfter=4)),
          P("<b>AFE-9 (Equation 9 of the AFE paper) &#8212; the active query.</b> Draw the next batch to be informative "
            "about the verifier while staying near the current policy:"),
          eq_block(r"x_{t+1}\sim\mathrm{argmax}_{q}\;\mathbb{E}_{x\sim q}[\sigma_t(\phi_s(x))]-\beta\,\mathrm{KL}(q\,\|\,p_1^{\theta_t})", fs=11.5, frac=1.0),
          P("Its closed-form optimum is q*&#8733;p<sub>1</sub><sup>&#952;</sup>&#183;exp(&#963;<sub>t</sub>/&#946;). "
            "<b>The &#8216;&#963;-tilt&#8217; is exactly this reweighting</b>: sample windows from the policy, multiply their "
            "probabilities by exp(&#963;/&#946;) (up-weight high-uncertainty ones), and resample. &#946;&#8594;&#8734; "
            "recovers ordinary sampling. <b>&#8216;Acquisition space&#8217;</b> = the space in which we decide <i>which "
            "query is worth asking the verifier</i>, i.e. the ranking by &#963; (a term from Bayesian optimization / active "
            "learning)."),
          P("<b>AFE-10 (Equation 10) &#8212; the uncertainty.</b> &#963;<sub>t</sub> is the GP (Bayesian-linear) posterior "
            "standard deviation over the already-queried designs X<sub>t</sub>, in the feature space &#966;<sub>s</sub>:"),
          eq_block(r"\sigma_t^2(x)=k(x,x)-k(x,X_t)\big(K_t+\lambda I\big)^{-1}k(X_t,x),\;\; k(x,x')=\langle\phi_s(x),\phi_s(x')\rangle", fs=10.5, frac=1.0),
          RMK("our instantiation", "in the <i>windowed</i> setting a design is a window at a fixed conditioning, so the "
              "&#963;-tilt is near-uniform (early vs. late windows are not comparable across a trajectory). We therefore "
              "drive exploration with the <b>finite-&#946; realization</b> of AFE-9 &#8212; sampling at raised temperature "
              "plus certified <b>broad proposals</b> &#8212; and keep AFE-10/the kernel as a live <b>diagnostic</b>."),
          Paragraph("A.&nbsp;The algorithm", H2),
          P("Seed D<sub>0</sub> with certified broad proposals. For each round: <b>(1) Generate</b> closed-loop rollouts "
            "(raised temperature) + broad proposals; <b>(2) Verify</b> with &#7811; (keep collision-free &#8743; goal "
            "&#8743; SOCP-feasible windows); <b>(3) UpdateFlow</b> with the signed objective g<sub>t</sub>=&#8711;L&#770;<sup>+</sup>"
            "&#8722;&#945;&#8711;L&#770;<sup>&#8722;</sup>, using &#945;=0 (standard CFM on accepted windows) with "
            "<b>inverse-frequency mode-balanced replay</b>; <b>(4) evaluate</b> certified coverage and keep the peak round. "
            "Hyperparameters: rounds T&#8804;40, 24 rollouts + 140 broad per &#947;, 200 SGD steps/round, lr 2e-4, batch 128.")]

# ===================== VI. THE 2x2 DIAGNOSTICS =====================
story += [Paragraph("VI.&nbsp;&nbsp;The 2&#215;2 Video Panels &#8212; Disambiguated", H1),
          P("The three panels of <font face='Courier'>expansion_2x2.mp4</font> measure <b>different things</b>; this is the "
            "common source of confusion."),
          P("&#8226; <b>Live kernel</b> (top-right): the Gram matrix K<sub>t</sub> of ~90 candidate windows sampled at one "
            "pre-obstacle state, rows/cols sorted by window direction. It shows <i>feature-space geometry</i>: uniformly "
            "bright &#8658; the policy is uni-modal; <b>block structure</b> &#8658; windows have split into distinct modes."),
          P("&#8226; <b>&#963;-histogram</b> (bottom-left): the distribution of &#963;<sub>t</sub> (AFE-10) over those "
            "candidates, plus ESS = 1/&#931;w&#772;&#178; of the AFE-9 tilt weights. It shows <i>acquisition "
            "informativeness</i>: ESS&#8776;N &#8658; tilt ~uniform (uninformative); ESS&#8595; &#8658; &#963; "
            "discriminates novel candidates."),
          P("&#8226; <b>Certified coverage</b> (bottom-right): the fraction of reachable-safe cells near the obstacles "
            "covered by the policy's certified trajectories &#8212; the <b>task objective</b>:"),
          eq_block(r"\mathrm{Cov}_t=\frac{|\,\mathcal{C}_{\mathrm{acc}}(t)\cap \mathcal{C}^*\,|}{|\,\mathcal{C}^*\,|},\qquad \mathcal{C}^*=\mathrm{cells}(\Omega^*)", fs=11.5, frac=1.0),
          P("<b>Key.</b> coverage is task-space (cells filled, the thing optimized); the kernel is feature-space; &#963;/ESS "
            "is acquisition informativeness. The kernel and &#963; are <b>signatures of <i>when</i> a new mode emerges</b>, "
            "not the objective. Empirically the coverage climbs monotonically while ESS stays high/noisy &#8212; confirming "
            "the tilt is not the driver."),
          fig_block(os.path.join(FIG, "slalom", "expansion_2x2.png"),
                    "Fig. 5. Slalom expansion (T=27). Certified trajectories (a new mode flashed at its discovery round) | "
                    "live kernel K<sub>t</sub> | &#963;-histogram + ESS | certified coverage vs. round.", frac=1.0)]

# ===================== VII. RESULTS / W&B =====================
story += [Paragraph("VII.&nbsp;&nbsp;What W&amp;B Shows about Learning", H1),
          P("<b>Pretrain.</b> CFM loss falls 1.32&#8594;0.68 (gap) / 1.33&#8594;0.66 (slalom); validation tracks training "
            "(no overfit). It converges to the <b>irreducible multi-modal-target floor &#8776;0.66</b>: at a fixed "
            "conditioning the target window is multi-modal (down/weave/up all valid), so the velocity MSE cannot reach 0 "
            "&#8212; a floor, not a failure. The aux grid-reconstruction loss falls 0.20&#8594;0.01 (both scenes): the "
            "polytope&#8594;context map is learned essentially perfectly."),
          P("<b>Expansion.</b> Certified coverage rises and mode-coverage &#8594; 1.0 at all &#947; (slalom coverage "
            "0.38&#8594;0.79 over 27 rounds; down+weave discovered at round&nbsp;1, the hard over-the-top at round&nbsp;27). "
            "&#947; modulates diversity: at &#947;=0.1 the policy strongly prefers the safe down mode; at &#947;&#8805;0.5 "
            "all three modes come readily."),
          Paragraph("A.&nbsp;Why validity can decrease during expansion", H2),
          P("Coverage and validity can move in <b>opposite directions</b>. Expansion pushes probability mass toward "
            "<i>new, harder</i> modes (weave, over-the-top). At a conservative &#947; (e.g. 0.1) those modes sit close to "
            "the obstacles, so the policy's attempts at them are more often collision-prone or fail the SOCP &#8212; "
            "raising coverage of the <i>space</i> while lowering the <i>fraction</i> of samples that are fully certified "
            "(slalom &#947;=0.1 validity 0.53&#8594;0.28). Three concrete causes: (i) the reward for diversity trades "
            "against the conservative single-mode optimum; (ii) mode-balanced replay up-weights rare, harder modes; "
            "(iii) the certificate is strict (collision &#8743; goal &#8743; SOCP), so a near-miss counts as invalid. We "
            "mitigate with peak-round saving and mode-balancing, but the trade is real &#8212; it is the honest cost of "
            "expanding into hard modes."),
          fig_block(os.path.join(FIG, "slalom", "exploration_modes.png"),
                    "Fig. 6. The trained policy generating its three certified modes (around-down / weave / over-the-top) "
                    "for each &#947;.", frac=1.0),
          Paragraph("VIII.&nbsp;&nbsp;Caveats", H1),
          P("&#8226; The AFE-9 &#963;-tilt is near-uniform in the windowed setting (diagnostic only). &#8226; &#937;* (the "
            "coverage denominator) is a broad-proposal estimate, so coverage is relative to that proxy. &#8226; The CFM "
            "floor &#8776;0.66 is expected (multi-modal target). &#8226; Over-the-top needed a dedicated high-arc proposal "
            "and 27 rounds &#8212; it is genuinely the hardest mode. &#8226; Moving pedestrians will need a velocity-aware "
            "(higher-order) polytope; the current polytope is position-only.")]

# ===================== build =====================
def build():
    from reportlab.platypus import SimpleDocTemplate
    doc = SimpleDocTemplate(OUT, pagesize=letter, topMargin=54, bottomMargin=48, leftMargin=54, rightMargin=54,
                            title="Safe Flow Expansion — Implementation Report")
    flow = [f for f in story if type(f).__name__ not in ("NextPageTemplate", "FrameBreak")]  # single column
    doc.build(flow)
    print(f"report -> {OUT}")


if __name__ == "__main__":
    build()
