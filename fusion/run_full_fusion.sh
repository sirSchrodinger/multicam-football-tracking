#!/usr/bin/env bash
# Tam 2-kamera füzyon pipeline — cam1 full tracking bitince tek komut.
# Kullanim: bash fusion/run_full_fusion.sh [CAM1_TRACKS] [CAM2_TRACKS]
set -euo pipefail
cd "$(dirname "$0")/.."
CAM1="${1:-raw/tracks_cankaya_cam1.parquet}"
CAM2="${2:-raw/tracks_cankaya_cam2_fullgame.parquet}"
PY=venv/bin/python
BASE=scratchpad/cam2_baseline/tracks_cankaya_cam2_fullgame_stitched.parquet
mkdir -p docs

echo "== 1/3 füzyon motoru =="
$PY fusion/fuse_engine.py "$CAM1" "$CAM2" --out scratchpad/fused_full.parquet

echo "== 2/3 çaprazlaşma analizi =="
$PY fusion/crossing_analysis.py "$CAM1" "$CAM2" --out scratchpad/crossings_full.json

echo "== 3/3 rapor + 2D replay =="
ARGS=(scratchpad/fused_full.parquet --crossings scratchpad/crossings_full.json
      --video docs/fusion_replay.mp4 --report docs/FUSION_REPORT.md)
[ -f "$BASE" ] && ARGS+=(--baseline "$BASE")
$PY fusion/fusion_report.py "${ARGS[@]}"

echo "BITTI -> docs/FUSION_REPORT.md + docs/fusion_replay.mp4"
