# IEEE-style Compact Polytope Verifier Package

Compile:

```bash
cd paper
pdflatex compact_polytope_verifier.tex
pdflatex compact_polytope_verifier.tex
```

The package is self-contained: all images referenced by the TeX file are included under `paper/figures/`.

Key source files:

- `compact_polytope_verifier.tex`: IEEE-style compact note.
- `../VERIFIER_POLYTOPE.md`: updated markdown note.
- `../tex/verifier_polytope_socp_proofs.tex`: standalone proof resource.
- `../src/pillar3_m_bounds_6x3.py`: margin-bound demo code.
- `../results/narrow_gap_m_bounds_K_gamma_6x3.csv`: numerical output used in the table.
