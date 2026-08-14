# Local six-row SFM comparison

> **Legacy Hp10 tool.** This renderer is preserved for the July Hp10 lineage.
> Its policy loader cannot admit the canonical HP100 checkpoint. Use the
> bundled HP100 branch/mechanism videos or port the renderer explicitly before
> claiming HP100 compatibility.

This tool collects controller traces once, then renders both an MP4 and
frame-indexed vector PDF snapshots without rerunning any controller. The
bundled pretrained checkpoint is authenticated as:

```text
checkpoints/hp10_pretrained_r0.pt
SHA-256 1b5179c935d3eeff8824967d707d64cc9bab273949ee1f0e4f190172bab1b215
```

## What the six rows mean

| row | environment | controller and stopping rule |
|---:|---|---|
| 1 | matched ID: 20 pedestrians, 0.5–1.0 m/s | faithful SafeMPPI demonstration expert; show the first exact full-\(H=10\) negative plan, but do not execute it |
| 2 | double-shift OOD: 40 pedestrians, 1.0–2.0 m/s | the same expert and verifier gate |
| 3 | double-shift OOD | pretrained B1 collection; max one-step nominal-\(H_P\)-margin selector |
| 4 | double-shift OOD | same proposals and verifier; native SafeMPPI-cost selector |
| 5 | double-shift OOD | same proposals and verifier; balanced safety/performance rank |
| 6 | double-shift OOD | Kazuki guidance on the same pretrained flow prior; cyan is integrated goal guidance and magenta is integrated safety guidance |

Rows 3–5 use the same \(K=16\), RBF-selected \(B=4\), exact moving-pedestrian
full-\(H=10\) verifier, episode, gamma, and keyed proposal noise. Only the
execution ranking differs. Their collector is diagnostic/offline: after a
finite-\(B\) NVP it labels and executes an independent raw continuation instead
of pretending that the rollout is a certified deployment.

The display uses true blue for positive executed-window branches and true red
for negative branches. It has no robot direction arrow and no temporary
candidate inset. The goal appears in every panel. All row and gamma labels,
plus the frame index and simulator step, appear in the right-hand column.

## Recommended episodes

- `--id-episode 150000`: start of the canonical matched-ID deployment bank.
- `--ood-episode 250000`: the exact double-density/double-velocity episode used
  by the earlier pre-expansion comparison.
- `--gammas 0.1 0.5 1.0`: the established paper columns.

For the same pedestrian RNG identity under two scene profiles, set both
episode arguments to `250000`. This is a paired visualization, not a member of
the canonical ID evaluation bank.

Declared banks can be printed with:

```bash
/Users/dhl/anaconda3/bin/python3 scripts/sfm_compare.py episode-banks
```

The important ranges are demonstrations `0–7999`, expansion `20000–20159`,
matched-ID deployment from `150000`, double-shift OOD deployment from `250000`,
and the current raw M50 bank `260000–260049` per gamma.

## Generate traces and video

Run from `/Users/dhl/Documents/safe_flow_expansion_SFM`:

```bash
/Users/dhl/anaconda3/bin/python3 scripts/sfm_compare.py all \
  --checkpoint checkpoints/hp10_pretrained_r0.pt \
  --output-dir outputs/six_row_pretrained_ep150000_250000 \
  --id-episode 150000 \
  --ood-episode 250000 \
  --gammas 0.1 0.5 1.0 \
  --device mps \
  --verifier-workers 8 \
  --fps 5 \
  --frame-stride 2
```

The command refuses to reuse an existing output directory. Controller traces
are saved in `six_row_traces.pt`; `six_row_comparison.mp4` is only a rendering
of that immutable bundle. On a CUDA host, replace `--device mps` with
`--device cuda`.

For a future expanded model, change only:

```text
--checkpoint /absolute/path/to/expanded_checkpoint.pt
```

The known filename `hp10_pretrained_r0.pt` is pinned to its promoted SHA.
Other checkpoint names are recorded and hashed but are not incorrectly forced
to equal the pretrained SHA. An explicit pin remains available through
`--expected-checkpoint-sha256`.

## Export one frame as a vector PDF

Read the frame index printed at the right of the video, then run:

```bash
/Users/dhl/anaconda3/bin/python3 scripts/sfm_snapshot.py \
  --trace-bundle outputs/six_row_pretrained_ep150000_250000/six_row_traces.pt \
  --render-json outputs/six_row_pretrained_ep150000_250000/RENDER_COMPLETE.json \
  --frame-index 42 \
  --output-pdf outputs/six_row_pretrained_ep150000_250000/frame_042.pdf
```

This rerenders frame 42 from the stored traces as vector graphics; it does not
extract a raster image from the MP4 and does not rerun sampling, guidance, or
verification.

## Prompt template

> In `/Users/dhl/Documents/safe_flow_expansion_SFM`, run the authenticated
> six-row SFM comparison using checkpoint `<CHECKPOINT>`, matched-ID episode
> `<ID_EPISODE>`, double-shift-OOD episode `<OOD_EPISODE>`, and gamma columns
> `<GAMMAS>`. Use `scripts/sfm_compare.py all`, preserve the immutable
> `six_row_traces.pt`, and report `RENDER_COMPLETE.json`, the MP4 path, frame
> count, checkpoint SHA, and any NVP row/step. Do not select or curate an
> episode based on the rendered outcome. After I provide a frame index, use
> `scripts/sfm_snapshot.py` to produce the corresponding vector PDF from the
> same trace SHA; do not rerun controllers.
