#!/usr/bin/env python3
"""make_standard_v3 — INCE SHIM. Kanonik mantik pitch/scale_snap.py'dedir.

Eskiden burada ayri (lsq) bir generator vardi; bu, diskteki v3 JSON ile
docs/SCALE_NOTES.md arasinda olcek uyusmazligi yaratiyordu. Tek-yontem-tek-sayi
icin generator artik pitch/scale_snap.make_v3 (izotropik geomean). Bu dosya
geriye-donuk CLI uyumlulugu icin ona delege eder.

Kullanim:
  python -m calib.make_standard_v3 calib/cankaya_cam2_v2.json calib/cankaya_cam2_v3.json
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pitch.scale_snap import (  # noqa: E402,F401
    STD_CATALOG, DEFAULT_BY_FORMAT, snap_size, make_v3, _main,
)

if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
