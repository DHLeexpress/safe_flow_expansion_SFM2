#!/usr/bin/env bash
# Assemble the 3x3 grid: rows PRE / Champion / Kazuki, columns gamma 0.1/0.5/1.0.
# Shorter panels are frozen on their last frame (tpad clone) to the longest
# duration, then xstacked. Usage: assemble_grid.sh <panel_dir> <out_mp4>
set -euo pipefail
D=$1; OUT=$2
first_dur=0
inputs=()
for name in pre_g0.1 pre_g0.5 pre_g1 champion_g0.1 champion_g0.5 champion_g1 kazuki_g0.1 kazuki_g0.5 kazuki_g1; do
  f=$D/panel_${name}.mp4
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
  inputs+=("$f")
done
max=0
for f in "${inputs[@]}"; do
  d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f")
  max=$(python3 -c "print(max($max, $d))")
done
pad=$(python3 -c "print($max + 1.0)")
FI=""; FC=""
i=0
for f in "${inputs[@]}"; do
  FI="$FI -i $f"
  FC="$FC[$i:v]tpad=stop_mode=clone:stop_duration=$pad,trim=duration=$max,setpts=PTS-STARTPTS[v$i];"
  i=$((i+1))
done
FC="$FC[v0][v1][v2][v3][v4][v5][v6][v7][v8]xstack=inputs=9:layout=0_0|w0_0|w0+w3_0|0_h0|w0_h0|w0+w3_h0|0_h0+h3|w0_h0+h3|w0+w3_h0+h3[grid]"
ffmpeg -y $FI -filter_complex "$FC" -map "[grid]" -c:v libx264 -pix_fmt yuv420p -movflags +faststart "$OUT"
ffmpeg -y -sseof -0.2 -i "$OUT" -frames:v 1 "${OUT%.mp4}_last_frame.png"
echo ASSEMBLED
