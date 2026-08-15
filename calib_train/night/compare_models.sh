#!/usr/bin/env bash
# seg2_hr_crop.pth inince TEK KOMUT: baseline vs crop(full) vs crop(TILED) — aynı 121-havuz, dürüst.
# crop-aware'in asıl testi TILED (far-çizgi crop'u yüksek-res). Çıktı cand/_cov_*.json + özet.
cd <repo-root>
PY=venv/bin/python
run(){ echo "=== $4 ==="; env MODEL="$1" BASE=32 TW=1024 TH=576 PRE=none THR=0.5 TTA="$2" TAG="$3" \
  $PY -m calib_train.eval_coverage 2>/dev/null | grep -E "KAPSAMA|MODEL"; }
run calib_train/seg2_hr2.pth      none base121      "BASELINE seg2_hr2 (full)"
[ -f calib_train/seg2_hr_crop.pth ] && {
  run calib_train/seg2_hr_crop.pth none cropfull121  "CROP-AWARE (full)"
  run calib_train/seg2_hr_crop.pth tiled cropTILE121 "CROP-AWARE (TILED far)"
} || echo "seg2_hr_crop.pth henüz yok"
echo "=== ÖZET ==="; for t in base121 cropfull121 cropTILE121; do
  f=calib_train/cand/_cov_$t.json
  [ -f "$f" ] && $PY -c "import json;d=json.load(open('$f'));print(f\"  {d['tag']:14s} kapsama %{d['coverage_pct']} ({d['accept']}/{d['n']}) med-res {d['median_res']}m\")"
done