#!/usr/bin/env bash
# NOKTA-clicker indirmesini yakala → solve_points. Tarayıcı '(1)' ekine/çakışmaya
# dayanıklı: her tur ~/Downloads/clicks_cankaya_cam2*.json içinden EN YENİyi seçer,
# bayat calib/ kopyasını siler (solver indirileni okusun).
set -u
cd <repo-root>
GLOB="$HOME/Downloads/clicks_cankaya_cam2*.json"
newest(){ ls -t $GLOB 2>/dev/null | head -1; }

base=0; f=$(newest); [ -n "${f:-}" ] && base=$(stat -c %Y "$f")
echo "izliyorum: $GLOB (en yeni mtime=$base) — yeni indirme bekleniyor"
for _ in $(seq 1 5400); do          # ~90 dk
  f=$(newest)
  if [ -n "${f:-}" ]; then
    cur=$(stat -c %Y "$f")
    if [ "$cur" != "$base" ]; then
      sleep 1
      rm -f calib/clicks_cankaya_cam2.json
      cp -f "$f" calib/clicks_cankaya_cam2.json
      n=$(venv/bin/python -c "import json;print(len(json.load(open('calib/clicks_cankaya_cam2.json')).get('points',{})))" 2>/dev/null)
      echo "=== yeni clicks geldi ($n nokta) [$(basename "$f")] -> solver ==="
      venv/bin/python calib/solve_points.py
      exit 0
    fi
  fi
  sleep 1
done
echo "zaman aşımı — yeni indirme gelmedi"
