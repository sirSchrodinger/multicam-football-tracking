"""gate.py — drift_guard sarmalayıcı: LOSO-hazırlık + promote/rollback kararı.

DÜRÜSTLÜK KAPISI (sistemin tek gerçek başarı ölçütü):
  - Promosyon ANCAK leave-one-stadium-out held-out'ta ölçülen iyileşme + regresyon-yok
    ile mümkün. Tek-saha QA hiçbir şey kanıtlamaz -> >=2 blind-GT saha şart.
  - gate_or_rollback muhafazakâr: held-out AP/recall gerilerse, HERHANGİ bir sahada
    AP çökerse veya reproj artarsa REDDET (B2B güven > marjinal kazanç).

İki giriş:
  evaluate(model_fn, loso_val) : aday modeli held-out'ta ölç (GPU; aday inference'i
        çağırana ait — bu modül torch import etmez). Genelde RunPod'da koşar.
  decide(new_metrics, baseline) : ölçülmüş metrikleri kıyasla -> keep/rollback +
        gate_ready ön-koşulu. CPU-only.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from self_training import drift_guard as dg  # noqa: E402

MIN_GT_VENUES = 2  # measure.MIN_GT_VENUES_FOR_GATE ile aynı sözleşme


def evaluate(model_fn, loso_val, iou_thr: float = 0.5) -> dict:
    """Aday modeli leave-one-stadium-out held-out sette değerlendir (drift_guard)."""
    return dg.evaluate_held_out(model_fn, loso_val, iou_thr=iou_thr)


def decide(new_metrics: dict, baseline_metrics: dict | None,
           gt_venues: list[str], **gate_kwargs) -> dict:
    """promote/rollback kararı + LOSO-hazırlık ön-koşulu.

    Döndürür {decision: 'keep'|'rollback'|'blocked', reasons, deltas, gate_ready}.
    'blocked' = LOSO mümkün değil (yeterli blind-GT saha yok) ya da baseline yok;
    bu durumda HİÇBİR promosyon yapılmaz (sahte ilerleme önlenir).
    """
    gate_ready = len(set(gt_venues)) >= MIN_GT_VENUES
    if not gate_ready:
        return {"decision": "blocked", "gate_ready": False,
                "reasons": [f"loso_not_ready(blind_gt_venues={len(set(gt_venues))}<{MIN_GT_VENUES})"],
                "deltas": {},
                "note": "ikinci sahada blind-GT toplanana dek detector promosyonu YASAK"}
    if baseline_metrics is None:
        return {"decision": "blocked", "gate_ready": True,
                "reasons": ["no_baseline_metrics"], "deltas": {},
                "note": "ilk LOSO baseline ölçülmeli (promote ile state'e yazılır)"}
    v = dg.gate_or_rollback(new_metrics, baseline_metrics, **gate_kwargs)
    return {"decision": "keep" if v["keep"] else "rollback",
            "gate_ready": True, "reasons": v["reasons"], "deltas": v["deltas"]}


def vibration_ok(reproj_resid_px, max_resid_px: float = 4.0) -> bool:
    """Ucuz titreşim nöbetçisi (sabit-kamera H hâlâ geçerli mi)."""
    return dg.vibration_sentinel_ok(reproj_resid_px, max_resid_px=max_resid_px)


# =========================================================================== #
# Track A config gate  (frozen insan-GT; drift_guard ruhu, recall/precision ekseni)
# =========================================================================== #
def config_gate(cand_test: dict, base_test: dict, *,
                min_far_gain: float = 0.0,
                max_rest_drop: float = 0.0,
                max_fp_delta: int = 0,
                eps: float = 1e-9) -> dict:
    """recover_far/post-proc CONFIG promosyon kapısı (frozen insan-GT TEST dilimi).

    gate_or_rollback ile aynı muhafazakâr ruh: ŞÜPHEDE REDDET. KEEP ancak ve ancak:
      - far_recall ARTMIŞ (cand.far - base.far > min_far_gain), VE
      - rest_recall DÜŞMEMİŞ (base.rest - cand.rest <= max_rest_drop), VE
      - FP KÖTÜLEŞMEMİŞ (cand.fp - base.fp <= max_fp_delta).
    (Track B'nin LOSO/AP eksenine karşılık burada count-GT recall/precision ekseni.)

    cand_test / base_test : recovery_search._recall_on çıktısı (TEST dilimi).
    Döndürür {decision:'keep'|'rollback', reasons, deltas}.
    """
    reasons: list[str] = []
    cf = cand_test.get("far_recall"); bf = base_test.get("far_recall")
    cr = cand_test.get("rest_recall"); br = base_test.get("rest_recall")
    cfp = int(cand_test.get("fp") or 0); bfp = int(base_test.get("fp") or 0)

    d_far = (cf - bf) if (cf is not None and bf is not None) else None
    d_rest = (cr - br) if (cr is not None and br is not None) else None
    d_fp = cfp - bfp

    if d_far is None:
        reasons.append("no_far_signal")
    elif d_far <= min_far_gain + eps:
        reasons.append(f"no_far_gain(d_far={d_far:+.4f} <= {min_far_gain})")
    if d_rest is not None and (-d_rest) > max_rest_drop + eps:
        reasons.append(f"rest_recall_drop(d_rest={d_rest:+.4f} < -{max_rest_drop})")
    if d_fp > max_fp_delta:
        reasons.append(f"fp_worse(d_fp={d_fp:+d} > {max_fp_delta})")

    return {
        "decision": "keep" if not reasons else "rollback",
        "reasons": reasons,
        "deltas": {
            "far_recall": (round(d_far, 4) if d_far is not None else None),
            "rest_recall": (round(d_rest, 4) if d_rest is not None else None),
            "fp": d_fp,
        },
        "gt_n": cand_test.get("gt_n"),
        "test_frames": cand_test.get("n_frames"),
    }
