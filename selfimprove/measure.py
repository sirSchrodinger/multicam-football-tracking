"""measure.py — birleşik ScoreCard (proxy eksenler + held-out recall).

İki sinyal AYRI tutulur (dürüstlük):
  proxy  : match_quality.assess_match -> calib/identity/motion/detection eksenleri.
           UCUZ, GT'siz, her saha/cycle. SADECE yönlendirme için (routing).
  recall_heldout : recall_score 3-kör-sayıcı reconcile (recall_val/score_summary.json).
           İnsan-GT. recall İDDİASI ve detector-gate SADECE buradan.

ScoreCard = {
  "target": str,                       # ölçülen koşu (saha/pencere)
  "proxy_axes": {calib,detection,identity,motion: {score,...}},
  "proxy_limiting_factor": str,        # match_quality argmin (semptom)
  "recall_heldout": {base, base_tile, far_base, far_base_tile, precision_fp,
                     n_gt_confident, n_venues_with_gt} | None,
  "gate_ready": bool,                  # LOSO mümkün mü (>=2 saha blind-GT)?
}
CPU-only.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.match_quality import assess_match  # noqa: E402

# Drift_guard leave-one-stadium-out için gereken MINIMUM blind-GT saha sayısı.
# Tek saha QA hiçbir şey kanıtlamaz (drift_guard docstring) -> gate >=2 ister.
MIN_GT_VENUES_FOR_GATE = 2


def recall_from_summary(summary_path: str | Path = ROOT / "recall_val/score_summary.json",
                        n_venues_with_gt: int = 1) -> dict | None:
    """recall_score reconcile aggregate'inden held-out recall metriklerini çıkar.

    aggregate.far/rest: {H(confident), A(ambiguous), base, tile, unc, fp}.
    recall = (bir tespitle örtülen confident-GT) / (confident-GT). FP=0 => precision %100.
    Döndürür None eğer summary yoksa.
    """
    p = Path(summary_path)
    if not p.exists():
        return None
    agg = json.loads(p.read_text()).get("aggregate", {})
    far, rest = agg.get("far", {}), agg.get("rest", {})
    Hc = far.get("H", 0) + rest.get("H", 0)
    if Hc == 0:
        return None
    base = far.get("base", 0) + rest.get("base", 0)
    tile = far.get("tile", 0) + rest.get("tile", 0)
    fp = far.get("fp", 0) + rest.get("fp", 0)
    farH = far.get("H", 0) or 1
    return {
        "base": round(base / Hc, 4),
        "base_tile": round((base + tile) / Hc, 4),
        "far_base": round(far.get("base", 0) / farH, 4),
        "far_base_tile": round((far.get("base", 0) + far.get("tile", 0)) / farH, 4),
        "precision_fp": int(fp),
        "n_gt_confident": int(Hc),
        "n_gt_far": int(far.get("H", 0)),
        "n_venues_with_gt": int(n_venues_with_gt),
        "source": str(p),
    }


def measure(target: str,
            tracks_path: str,
            state_path: str | None = None,
            calib_path: str | None = None,
            recall_summary: str | None = None,
            gt_venues: list[str] | None = None) -> dict:
    """Bir koşunun birleşik ScoreCard'ını üret.

    target        : etiket (örn. 'cankaya_active_win66905').
    tracks_path   : stitched/bridged per-frame parquet (proxy eksenler için).
    state_path    : sürekli konum parquet (motion/off-field için, ops).
    calib_path    : kalibrasyon json (qa.median_px için, ops).
    recall_summary: recall_val/score_summary.json (held-out recall; bu target'ın
                    blind-GT'si varsa). Yoksa recall_heldout=None.
    gt_venues     : blind-GT'si olan saha listesi (gate_ready hesabı).
    """
    proxy = assess_match(tracks_path=tracks_path, state_path=state_path,
                         calib_path=calib_path)
    gt_venues = gt_venues or []
    rh = None
    if recall_summary:
        rh = recall_from_summary(recall_summary, n_venues_with_gt=len(gt_venues) or 1)
    return {
        "target": target,
        "proxy_axes": proxy.get("axes", {}),
        "proxy_overall": proxy.get("overall_score"),
        "proxy_limiting_factor": proxy.get("limiting_factor"),
        "proxy_manual_fix": proxy.get("manual_fix", []),
        "recall_heldout": rh,
        "gate_ready": len(gt_venues) >= MIN_GT_VENUES_FOR_GATE,
    }


if __name__ == "__main__":
    d = ROOT / "stats_out/active_game"
    sc = measure(
        target="cankaya_active",
        tracks_path=str(d / "tracks_active_game_presplit_player_bridged.parquet"),
        state_path=str(d / "state14_clean.parquet"),
        calib_path=str(ROOT / "calib/cankaya_cam2_FINAL.json"),
        recall_summary=str(ROOT / "recall_val/score_summary.json"),
        gt_venues=["cankaya_cam2"],
    )
    print(json.dumps(sc, indent=2, ensure_ascii=False, default=str))
