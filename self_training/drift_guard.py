"""drift_guard.py — fine-tune regresyon kapisi + rollback (DEFERRED).

Self-training'in tek dogru basari olcutu: GORULMEMIS sahada bozulmamak.
Bu modul iki seyi yapar:

  1) Insan-dogrulamali, KUCUK ve DONMUS bir validasyon seti uzerinde modeli olcer
     (detection mAP/recall + kalibrasyon tarafinda median reproj_px / auto-accept).
     Validasyon seti leave-one-stadium-out olmali: egitimde GORULMEYEN sahalardan.

  2) Yeni tur metrikleri onceki (baseline) tura gore regresyon gosteriyorsa
     fine-tune'u REDDET ve eski agirliga geri don (rollback). Sadece held-out
     hata ARTMADIYSA yeni agirligi tut.

Anti-overpromise: tek-saha QA hicbir sey kanitlamaz. Gate yalniz held-out metrige bakar.

Bu modul agirlik/torch yuklemez; model'i bir callable olarak alir, boylece
GPU-suz import edilebilir. Gercek inference cagrisini cagiran saglar.
"""
from __future__ import annotations

from typing import Any, Callable, Iterable

import numpy as np


# ---------------------------------------------------------------------------
# detection metrigi (basit, bagimsiz mAP@0.5 + recall)
# ---------------------------------------------------------------------------
def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4) a ve (M,4) b arasinda IoU matrisi (N,M). xyxy ham piksel."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    area_a = np.clip(ax2 - ax1, 0, None) * np.clip(ay2 - ay1, 0, None)
    area_b = np.clip(bx2 - bx1, 0, None) * np.clip(by2 - by1, 0, None)
    union = area_a + area_b - inter + 1e-6
    return (inter / union).astype(np.float32)


def detection_ap_recall(preds_per_image: list[dict], gts_per_image: list[np.ndarray],
                        iou_thr: float = 0.5) -> dict:
    """Tek-sinif (person) AP@iou_thr + recall hesapla.

    preds_per_image: her goruntu icin {"boxes": (N,4) xyxy, "scores": (N,)}
    gts_per_image:   her goruntu icin (M,4) xyxy insan-dogrulamali GT kutular
    Donen: {"ap": float, "recall": float, "precision": float, "n_gt": int, "n_pred": int}

    Klasik VOC-tarzi: skora gore sirala, greedy GT-eslesme, PR egrisi altinda alan
    (101-nokta yerine tum-nokta interpolasyon — kucuk val seti icin yeterli).
    """
    all_scores: list[float] = []
    all_tp: list[int] = []
    n_gt_total = 0
    for pred, gt in zip(preds_per_image, gts_per_image):
        boxes = np.asarray(pred.get("boxes", np.zeros((0, 4))), dtype=np.float32)
        scores = np.asarray(pred.get("scores", np.zeros(len(boxes))), dtype=np.float32)
        gt = np.asarray(gt, dtype=np.float32).reshape(-1, 4)
        n_gt_total += len(gt)
        if len(boxes) == 0:
            continue
        order = np.argsort(-scores)
        boxes, scores = boxes[order], scores[order]
        iou = _iou_xyxy(boxes, gt) if len(gt) else np.zeros((len(boxes), 0))
        matched = np.zeros(len(gt), dtype=bool)
        for i in range(len(boxes)):
            tp = 0
            if len(gt):
                j = int(np.argmax(iou[i])) if iou.shape[1] else -1
                if j >= 0 and iou[i, j] >= iou_thr and not matched[j]:
                    matched[j] = True
                    tp = 1
            all_scores.append(float(scores[i]))
            all_tp.append(tp)

    n_pred = len(all_scores)
    if n_pred == 0:
        return {"ap": 0.0, "recall": 0.0, "precision": 0.0,
                "n_gt": n_gt_total, "n_pred": 0}
    order = np.argsort(-np.asarray(all_scores))
    tp = np.asarray(all_tp)[order]
    fp = 1 - tp
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recall = tp_cum / (n_gt_total + 1e-6)
    precision = tp_cum / (tp_cum + fp_cum + 1e-6)
    # tum-nokta interpolasyon AP
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 0.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))
    return {
        "ap": round(ap, 4),
        "recall": round(float(recall[-1]) if len(recall) else 0.0, 4),
        "precision": round(float(precision[-1]) if len(precision) else 0.0, 4),
        "n_gt": n_gt_total, "n_pred": n_pred,
    }


# ---------------------------------------------------------------------------
# held-out degerlendirme
# ---------------------------------------------------------------------------
def evaluate_held_out(model: Callable[[Any], dict],
                      leave_one_stadium_out_val: Iterable[dict],
                      iou_thr: float = 0.5) -> dict:
    """Modeli leave-one-stadium-out donmus val seti uzerinde degerlendir.

    model: bir goruntu (np.ndarray BGR ya da yol) alip
           {"boxes": (N,4) xyxy, "scores": (N,)} donduren callable.
           (Boylece bu modul torch yuklemeden import edilebilir; cagiran
            RF-DETR sarmalayicisini callable olarak verir.)
    leave_one_stadium_out_val: her ogesi:
        {
          "stadium": str,            # held-out saha kimligi
          "image": <model girdisi>,  # frame (ndarray) veya yol
          "gt_boxes": (M,4) xyxy,    # insan-dogrulamali oyuncu kutulari
          # opsiyonel kalibrasyon tarafi:
          "reproj_px": float|None,       # bu sahada median pitch-reprojection hatasi
          "auto_accepted": bool|None,    # auto_calib self-verify gate sonucu
        }

    Donen: {
      "overall": {"ap":..,"recall":..,"precision":..},
      "per_stadium": {stadium: {"ap":..,"recall":..,"n_gt":..}},
      "calib": {"median_reproj_px": float|None, "auto_accept_rate": float|None,
                "n_venues": int},
      "n_images": int,
    }
    Generalization SADECE buradan okunur (tek-saha QA degil).
    """
    items = list(leave_one_stadium_out_val)
    preds_all: list[dict] = []
    gts_all: list[np.ndarray] = []
    by_stad: dict[str, dict] = {}
    reproj_vals: list[float] = []
    auto_flags: list[bool] = []

    for it in items:
        out = model(it["image"])
        pred = {"boxes": np.asarray(out.get("boxes", np.zeros((0, 4))), dtype=np.float32),
                "scores": np.asarray(out.get("scores", np.zeros(0)), dtype=np.float32)}
        gt = np.asarray(it.get("gt_boxes", np.zeros((0, 4))), dtype=np.float32).reshape(-1, 4)
        preds_all.append(pred)
        gts_all.append(gt)
        st = it.get("stadium", "unknown")
        by_stad.setdefault(st, {"preds": [], "gts": []})
        by_stad[st]["preds"].append(pred)
        by_stad[st]["gts"].append(gt)
        if it.get("reproj_px") is not None:
            reproj_vals.append(float(it["reproj_px"]))
        if it.get("auto_accepted") is not None:
            auto_flags.append(bool(it["auto_accepted"]))

    overall = detection_ap_recall(preds_all, gts_all, iou_thr)
    per_stadium = {}
    for st, d in by_stad.items():
        m = detection_ap_recall(d["preds"], d["gts"], iou_thr)
        per_stadium[st] = {"ap": m["ap"], "recall": m["recall"], "n_gt": m["n_gt"]}

    calib = {
        "median_reproj_px": (round(float(np.median(reproj_vals)), 3)
                             if reproj_vals else None),
        "auto_accept_rate": (round(float(np.mean(auto_flags)), 3)
                             if auto_flags else None),
        "n_venues": len(by_stad),
    }
    return {"overall": overall, "per_stadium": per_stadium,
            "calib": calib, "n_images": len(items)}


# ---------------------------------------------------------------------------
# regresyon kapisi / rollback
# ---------------------------------------------------------------------------
def gate_or_rollback(new_metrics: dict, baseline_metrics: dict,
                     *,
                     min_ap_gain: float = 0.0,
                     max_ap_drop: float = 0.005,
                     max_recall_drop: float = 0.01,
                     max_reproj_increase_px: float = 0.5,
                     max_per_stadium_ap_drop: float = 0.03) -> dict:
    """Yeni tur held-out metriklerini baseline'a kiyasla; tut ya da rollback karari.

    Karar True (KEEP) ancak ve ancak:
      - overall AP, baseline'dan en az min_ap_gain artmis (yani gerilememis), VE
      - overall AP dususu <= max_ap_drop (gurultu tolerasi), VE
      - overall recall dususu <= max_recall_drop, VE
      - HICBIR held-out sahada AP > max_per_stadium_ap_drop dusmemis
        (ortalama iyiyken tek sahada cokme = gizli overfit), VE
      - kalibrasyon varsa median_reproj_px artisi <= max_reproj_increase_px.

    Aksi halde False => eski agirliga rollback. Bilincli muhafazakar:
    suphede iken yeni modeli REDDET (B2B guven > marjinal kazanc).

    Donen: {"keep": bool, "reasons": [...], "deltas": {...}}
    """
    reasons: list[str] = []

    def _ov(m, k):
        return float(m.get("overall", {}).get(k, 0.0))

    d_ap = _ov(new_metrics, "ap") - _ov(baseline_metrics, "ap")
    d_recall = _ov(new_metrics, "recall") - _ov(baseline_metrics, "recall")

    if d_ap < min_ap_gain:
        if d_ap < -max_ap_drop:
            reasons.append(f"overall_ap_drop({d_ap:+.4f} < -{max_ap_drop})")
        # min_ap_gain>0 ile "ilerleme yok" da reddedilebilir:
        elif min_ap_gain > 0:
            reasons.append(f"insufficient_ap_gain({d_ap:+.4f} < {min_ap_gain})")
    if -d_recall > max_recall_drop:
        reasons.append(f"recall_drop({d_recall:+.4f} < -{max_recall_drop})")

    # per-stadium cokme kontrolu
    new_ps = new_metrics.get("per_stadium", {})
    base_ps = baseline_metrics.get("per_stadium", {})
    worst = None
    for st, bm in base_ps.items():
        if st in new_ps:
            drop = float(bm.get("ap", 0.0)) - float(new_ps[st].get("ap", 0.0))
            if worst is None or drop > worst[1]:
                worst = (st, drop)
            if drop > max_per_stadium_ap_drop:
                reasons.append(f"stadium_collapse[{st}](-{drop:.3f})")

    # kalibrasyon reproj
    new_rp = new_metrics.get("calib", {}).get("median_reproj_px")
    base_rp = baseline_metrics.get("calib", {}).get("median_reproj_px")
    d_reproj = None
    if new_rp is not None and base_rp is not None:
        d_reproj = float(new_rp) - float(base_rp)
        if d_reproj > max_reproj_increase_px:
            reasons.append(f"reproj_worse(+{d_reproj:.3f}px > {max_reproj_increase_px})")

    return {
        "keep": len(reasons) == 0,
        "reasons": reasons,
        "deltas": {
            "ap": round(d_ap, 4), "recall": round(d_recall, 4),
            "worst_stadium": (worst[0] if worst else None),
            "worst_stadium_ap_drop": (round(worst[1], 4) if worst else None),
            "reproj_px": (round(d_reproj, 3) if d_reproj is not None else None),
        },
    }


# ---------------------------------------------------------------------------
# Mean-Teacher EMA onerisi (yardimci + rehber)
# ---------------------------------------------------------------------------
def ema_update(teacher_state: dict, student_state: dict, decay: float = 0.999) -> dict:
    """Mean-Teacher EMA: teacher = decay*teacher + (1-decay)*student.

    Self-training'de pseudo-label gurultusunu bastirmak icin teacher'i student'in
    yavas hareketli ortalamasi olarak tut; pseudo-label'lari teacher uretsin,
    gradyan student'a aksin. Bu yardimci torch-bagimsizdir: state_dict benzeri
    {ad: np.ndarray} sozlugu bekler (gercek egitimde torch tensor'leriyle
    train.py icinde cagrilir).

    Donen: guncellenmis teacher_state (yeni sozluk).
    Not: Bu CEKIRDEK referanstir; gercek torch EMA train.py'de uygulanir.
    """
    if not (0.0 < decay < 1.0):
        raise ValueError("decay (0,1) araliginda olmali")
    out: dict = {}
    for k, t in teacher_state.items():
        s = student_state.get(k)
        if s is None:
            out[k] = t
            continue
        t = np.asarray(t, dtype=np.float64)
        s = np.asarray(s, dtype=np.float64)
        out[k] = decay * t + (1.0 - decay) * s
    return out


def vibration_sentinel_ok(reproj_resid_px: Iterable[float],
                          max_resid_px: float = 4.0) -> bool:
    """Ucuz titresim nobetcisi: 3-4 yeniden-yansitilmis sabit landmark'in artigi.

    Sabit kamerada H mac boyunca degismemeli. Bu landmark'lar (kose, ceza sahasi
    T-kesisimi) her N frame'de yeniden-yansitilip GT piksel konumlariyla karsilastirilir.
    Hepsi <= max_resid_px ise H gecerli (yeniden-cozme YOK); degilse RECALIBRATE.
    Donen: True = H hala gecerli.
    """
    r = np.asarray(list(reproj_resid_px), dtype=np.float64)
    if r.size == 0:
        return True
    return bool(np.all(r <= max_resid_px))
