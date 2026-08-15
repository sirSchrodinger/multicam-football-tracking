#!/usr/bin/env bash
# PAINT json indirildiğinde solver çalıştır. Tarayıcı '(1)' eki/çakışmasına dayanıklı:
# her tur ~/Downloads/paint_cankaya_cam2*.json içinden EN YENİyi seçer; bayat
# calib/ kopyasını siler (solver onu değil indirileni okusun).
set -u
cd <repo-root>
GLOB="$HOME/Downloads/paint_cankaya_cam2*.json"
newest(){ ls -t $GLOB 2>/dev/null | head -1; }

base=0; f=$(newest); [ -n "${f:-}" ] && base=$(stat -c %Y "$f")
echo "izliyorum: $GLOB (en yeni mtime=$base) — paint indirme bekleniyor"
for _ in $(seq 1 5400); do          # ~90 dk
  f=$(newest)
  if [ -n "${f:-}" ]; then
    cur=$(stat -c %Y "$f")
    if [ "$cur" != "$base" ]; then
      sleep 1
      rm -f calib/paint_cankaya_cam2.json
      cp -f "$f" calib/paint_cankaya_cam2.json
      n=$(venv/bin/python -c "import json;print(len(json.load(open('calib/paint_cankaya_cam2.json')).get('lines',{})),'çizgi +',len(json.load(open('calib/paint_cankaya_cam2.json')).get('points',{})),'nokta')" 2>/dev/null)
      echo "=== yeni paint geldi ($n) [$(basename "$f")] -> solver ==="
      venv/bin/python calib/solve_paint.py
      exit 0
    fi
  fi
  sleep 1
done
echo "zaman aşımı — paint indirme gelmedi"
