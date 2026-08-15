#!/bin/bash
# rapor.html -> PDF (headless brave, GUI YOK). Brave-yükü olmadan hafif açılır.
cd <repo-root>
venv/bin/python -c "
c=open('calib_train/cand/rapor.html').read()
open('calib_train/cand/rapor_full.html','w').write('<!doctype html><html><head><meta charset=\"utf-8\"><style>*{margin:0;padding:0;box-sizing:border-box}@page{margin:0}body{-webkit-print-color-adjust:exact;print-color-adjust:exact}</style></head><body>'+c+'</body></html>')
"
/opt/brave.com/brave/brave --headless --disable-gpu --no-sandbox --no-pdf-header-footer \
  --print-to-pdf=calib_train/cand/halisaha_rapor.pdf --virtual-time-budget=8000 \
  "file://$PWD/calib_train/cand/rapor_full.html" 2>&1 | tail -1
ls -la calib_train/cand/halisaha_rapor.pdf
