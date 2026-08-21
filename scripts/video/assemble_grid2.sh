#!/usr/bin/env bash
# Assemble the compact 3x3 grid and wrap it in the outside title/legend frame.
# Usage: assemble_grid2.sh <panel_dir> <out_mp4> <frame_png> <left> <top> <W> <H>
set -euo pipefail
D=$1; OUT=$2; FRAME=$3; LEFT=$4; TOP=$5; W=$6; H=$7

inputs=()
for name in pre_g0.1 pre_g0.5 pre_g1 champion_g0.1 champion_g0.5 champion_g1 \
            kazuki_g0.1 kazuki_g0.5 kazuki_g1; do
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
FC="$FC[v0][v1][v2][v3][v4][v5][v6][v7][v8]xstack=inputs=9:layout=0_0|w0_0|w0+w3_0|0_h0|w0_h0|w0+w3_h0|0_h0+h3|w0_h0+h3|w0+w3_h0+h3[grid];"
FC="$FC[grid]pad=${W}:${H}:${LEFT}:${TOP}:white[bg];[bg][9:v]overlay=0:0,scale=1900:-2[out]"

ffmpeg -y $FI -i "$FRAME" -filter_complex "$FC" -map "[out]" \
  -c:v libx264 -crf 20 -pix_fmt yuv420p -movflags +faststart "$OUT"
ffmpeg -y -sseof -0.3 -i "$OUT" -frames:v 1 "${OUT%.mp4}_last_frame.png"
echo ASSEMBLED "$OUT"
