#!/usr/bin/env python3
"""recall_eval.py — DONDURULMUS insan-GT'ye karsi tespit recall'i OLC.

Projenin #1 durustluk acigi: amator gece footage'inda RF-DETR recall'i HIC
insan-GT'ye karsi olculmedi (recall_qc.py docstring: "hicbir recall % iddiasi
DONDURULMUS insan-GT olmadan yapilmaz"). Bu harness o GT'yi kurar ve olcer:

  1) `frames`  : cam2.mp4'ten 8 frozen frame + ust-bant zoom crop'lari uretir
                 (GT sayimi icin; tespitten BAGIMSIZ/blind).
  2) `detect`  : her frozen frame'de
                   - base detektor (thr=0.30, MIN_H=25, NMS=0.6)  [export_tracks ile birebir]
                   - far-band tiling recovery (crop ust-bant -> upscale -> re-detect)
                 -> recall_val/detections.json + recall_val/anno/*.png (base=yesil, tiled=sari)
  3) `score`   : el-sayimi GT json'unu (recall_val/gt.json) tespitlere esle ->
                 per-zone recall (far/mid/near), base vs base+tiling, precision/FP,
                 conf-esik surgusu. GT belirsizliginden DURUST hata bandi.

DURUSTLUK: GT counters ham goruntuye bakar, tespit kutularini GORMEZ (blind).
recall = (bir tespitle eslesen GT kisi) / (GT kisi). Far-band tiling'in gercekten
recall artirip artirmadigini olcer (recall_qc.FarBandRecovery zaten yazili ama
GT'ye karsi hic dogrulanmadi).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "recall_val"
# Detektor agirligi. Varsayilan = regular (3-kor-sayici GT'nin olculdugu model;
# committed baseline'i bozmamak icin). ft (fine-tuned, yeni TR sahalarda generalize)
# ile olcmek icin:  HALISAHA_WEIGHTS=ft ./venv/bin/python eval/recall_eval.py detect
# GT-anchored olcum (1 Tem): ft, far-band base tespitini reg 74 -> 82 (+8/92 far
# oyuncu, precision temiz/gorsel-dogrulandi) tasidi; CLAHE ft ustune +0, tile +2.
import os as _os
_wsel = _os.environ.get("HALISAHA_WEIGHTS", "regular").lower()
if _wsel in ("ft_v2", "ftv2", "v2"):
    WEIGHTS = ROOT / "models/weights/checkpoint_ft_v2.pth"   # dark-aug far-dark fine-tune
elif _wsel in ("ft", "finetuned"):
    WEIGHTS = ROOT / "models/weights/checkpoint_ft.pth"
else:
    WEIGHTS = ROOT / "models/weights/checkpoint_best_regular.pth"

# export_tracks.py ile BIREBIR ayni base sabitler
THRESH = 0.30
MIN_H = 25
NMS_THRESH = 0.6

# far-band tiling (recall_qc.py ile ayni felsefe)
FAR_BAND = (110, 365)   # ust-bant satir araligi (far third; cam2 geometrisi)
TILE_UP = 2.0
TILE_THRESH = 0.40      # recall_qc olculmus deger (0.15-0.20 FP seli)
DEDUP_PX = 28.0         # base ile ayni-oyuncu piksel yaricapi (foot mesafe)

# zone esikleri (foot_y, full-frame): far<360, mid<560, near>=560.
# Zorluk-yansitir (kucuk/uzak = far). Ham foot_y saklanir; bucketing post-hoc.
ZONE_T1, ZONE_T2 = 360.0, 560.0

FROZEN = [37307, 44769, 52230, 59692, 67153, 74615, 82076, 85807]


def zone_of(foot_y: float) -> str:
    return "far" if foot_y < ZONE_T1 else ("mid" if foot_y < ZONE_T2 else "near")


# --------------------------------------------------------------------------- #
# 1) frames
# --------------------------------------------------------------------------- #
def cmd_frames(_args):
    import cv2
    (VAL / "frames").mkdir(parents=True, exist_ok=True)
    (VAL / "crops").mkdir(parents=True, exist_ok=True)
    c = cv2.VideoCapture(str(ROOT / "raw/cankaya_cam2.mp4"))
    fps = c.get(cv2.CAP_PROP_FPS)
    meta = {}
    for fi in FROZEN:
        c.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = c.read()
        if not ok:
            print("FAIL", fi); continue
        cv2.imwrite(str(VAL / f"frames/f{fi}.png"), fr)
        # far-band zoom (rows 90-380) x2.5 -> kucuk/uzak oyuncular sayilabilir
        up = fr[90:380, :].copy()
        up = cv2.resize(up, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(str(VAL / f"crops/f{fi}_farzoom.png"), up)
        meta[str(fi)] = {"frame": fi, "t_sec": round(fi / fps, 1)}
    c.release()
    (VAL / "frozen_meta.json").write_text(json.dumps(meta, indent=2))
    print("frames+crops yazildi:", list(meta.keys()))


# --------------------------------------------------------------------------- #
# 2) detect  (GPU)
# --------------------------------------------------------------------------- #
def _load_model():
    from rfdetr import RFDETRLargeDeprecated
    return RFDETRLargeDeprecated(pretrain_weights=str(WEIGHTS), device="cuda",
                                 num_classes=4)


def _predict(model, frame_bgr, thr):
    from PIL import Image
    det = model.predict(Image.fromarray(frame_bgr[:, :, ::-1]), threshold=thr)
    xy = np.asarray(det.xyxy, float).reshape(-1, 4)
    cf = np.asarray(det.confidence, float).reshape(-1)
    return xy, cf


def _base_detect(model, frame_bgr):
    """export_tracks.py ile birebir: thr=0.30, MIN_H height filtre, class-agnostic NMS."""
    import supervision as sv
    xy, cf = _predict(model, frame_bgr, THRESH)
    if len(xy):
        keep = (xy[:, 3] - xy[:, 1]) > MIN_H
        xy, cf = xy[keep], cf[keep]
    if not len(xy):
        return np.empty((0, 4)), np.empty((0,))
    d = sv.Detections(xyxy=xy, confidence=cf, class_id=np.zeros(len(xy), int))
    d = d.with_nms(threshold=NMS_THRESH, class_agnostic=True)
    return np.asarray(d.xyxy, float).reshape(-1, 4), np.asarray(d.confidence, float)


def _tile_far(model, frame_bgr, base_xy):
    """Far-bandi upscale edip re-detect; kutulari ham-frame px'e dondur,
    base ile ayni olanlari (foot mesafe < DEDUP_PX) ele. recall_qc._tile_detect mantigi."""
    import cv2
    y0, y1 = FAR_BAND
    crop = frame_bgr[y0:y1].copy()
    big = cv2.resize(crop, None, fx=TILE_UP, fy=TILE_UP, interpolation=cv2.INTER_CUBIC)
    xy, cf = _predict(model, big, TILE_THRESH)
    if not len(xy):
        return np.empty((0, 4)), np.empty((0,))
    xy = xy.copy()
    xy[:, [0, 2]] /= TILE_UP
    xy[:, [1, 3]] = xy[:, [1, 3]] / TILE_UP + y0
    keep = (xy[:, 3] - xy[:, 1]) > MIN_H
    xy, cf = xy[keep], cf[keep]
    if not len(xy):
        return np.empty((0, 4)), np.empty((0,))
    # base ile dedup: foot-noktasi (28px) + KUTU-ORTUSME (IoU>0.3 ya da merkez-icinde).
    # Sadece foot-noktasi yetersizdi: tiling, base'in bulduklarini tekrar uretiyordu
    # (sarilarin ~%80'i kopya). Kutu-ortusme bunu eler.
    bfoot = (np.stack([(base_xy[:, 0] + base_xy[:, 2]) / 2, base_xy[:, 3]], 1)
             if len(base_xy) else np.empty((0, 2)))
    tfoot = np.stack([(xy[:, 0] + xy[:, 2]) / 2, xy[:, 3]], 1)

    def _dup_box(tb):
        for b in base_xy:
            tcx, tcy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
            if b[0] <= tcx <= b[2] and b[1] <= tcy <= b[3]:
                return True
            ix1, iy1 = max(tb[0], b[0]), max(tb[1], b[1])
            ix2, iy2 = min(tb[2], b[2]), min(tb[3], b[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            ua = ((tb[2] - tb[0]) * (tb[3] - tb[1])
                  + (b[2] - b[0]) * (b[3] - b[1]) - inter)
            if ua > 0 and inter / ua > 0.3:
                return True
        return False

    new = []
    for i in range(len(xy)):
        if len(bfoot) and np.min(np.linalg.norm(bfoot - tfoot[i], axis=1)) < DEDUP_PX:
            continue
        if len(base_xy) and _dup_box(xy[i]):
            continue
        new.append(i)
    new = np.array(new, int)
    return xy[new], cf[new]


def cmd_detect(_args):
    import cv2
    (VAL / "anno").mkdir(parents=True, exist_ok=True)
    model = _load_model()
    out = {}
    for fi in FROZEN:
        fr = cv2.imread(str(VAL / f"frames/f{fi}.png"))
        bxy, bcf = _base_detect(model, fr)
        txy, tcf = _tile_far(model, fr, bxy)
        # kayit
        def pack(xy, cf):
            return [dict(x1=float(a[0]), y1=float(a[1]), x2=float(a[2]), y2=float(a[3]),
                         foot_x=float((a[0] + a[2]) / 2), foot_y=float(a[3]),
                         conf=float(c), zone=zone_of(float(a[3])))
                    for a, c in zip(xy, cf)]
        out[str(fi)] = {"base": pack(bxy, bcf), "tiled": pack(txy, tcf)}
        # annotated overlay (MY verification; GT'den ayri)
        vis = fr.copy()
        for a, c in zip(bxy, bcf):
            cv2.rectangle(vis, (int(a[0]), int(a[1])), (int(a[2]), int(a[3])), (0, 255, 0), 2)
            cv2.putText(vis, f"{c:.2f}", (int(a[0]), int(a[1]) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        for a, c in zip(txy, tcf):
            cv2.rectangle(vis, (int(a[0]), int(a[1])), (int(a[2]), int(a[3])), (0, 255, 255), 2)
            cv2.putText(vis, f"T{c:.2f}", (int(a[0]), int(a[1]) - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.rectangle(vis, (0, FAR_BAND[0]), (1919, FAR_BAND[1]), (255, 0, 0), 1)
        cv2.imwrite(str(VAL / f"anno/f{fi}_anno.png"), vis)
        print(f"f{fi}: base={len(bxy)} tiled+={len(txy)} "
              f"(far base={sum(1 for a in bxy if a[3]<ZONE_T1)})")
    (VAL / "detections.json").write_text(json.dumps(out, indent=2))
    print("detections.json yazildi")


# --------------------------------------------------------------------------- #
# 3) score  (GT json -> recall)
# --------------------------------------------------------------------------- #
def _match(gt_pts, det_foots, radius=55.0):
    """Greedy en-yakin esleme (foot piksel). gt_pts:[(x,y)], det_foots:[(x,y)].
    Doner: matched bool[len(gt)], her GT icin eslesen det idx (veya -1)."""
    matched = np.zeros(len(gt_pts), bool)
    used = np.zeros(len(det_foots), bool)
    if not len(gt_pts) or not len(det_foots):
        return matched, [-1] * len(gt_pts)
    D = np.linalg.norm(np.asarray(gt_pts)[:, None, :] - np.asarray(det_foots)[None, :, :], axis=2)
    assign = [-1] * len(gt_pts)
    order = np.dstack(np.unravel_index(np.argsort(D.ravel()), D.shape))[0]
    for gi, dj in order:
        if D[gi, dj] > radius:
            break
        if matched[gi] or used[dj]:
            continue
        matched[gi] = True; used[dj] = True; assign[gi] = int(dj)
    return matched, assign


def cmd_score(args):
    gt = json.loads((VAL / "gt.json").read_text())
    det = json.loads((VAL / "detections.json").read_text())
    radius = args.radius
    zones = ["far", "mid", "near"]
    # accumulators: per zone, confident GT count + matched-by-base + matched-by-tiled
    acc = {z: dict(gt_conf=0, gt_amb=0, base=0, tiled=0) for z in zones}
    fp_base = 0; fp_tiled = 0; det_base_total = 0
    per_frame = []
    for fi in FROZEN:
        k = str(fi)
        gframe = gt.get(k)
        if gframe is None:
            print(f"[uyari] GT yok f{fi}, atlandi"); continue
        people = gframe["people"]  # [{x,y,conf:'confident'|'ambiguous',note}]
        gt_pts = [(p["x"], p["y"]) for p in people]
        gt_conf = [p.get("conf", "confident") == "confident" for p in people]
        base = det[k]["base"]; tiled = det[k]["tiled"]
        base_foot = [(d["foot_x"], d["foot_y"]) for d in base]
        all_foot = base_foot + [(d["foot_x"], d["foot_y"]) for d in tiled]
        m_base, _ = _match(gt_pts, base_foot, radius)
        m_all, _ = _match(gt_pts, all_foot, radius)
        # precision: tespitlerden GT'ye eslesmeyen = FP
        # (base)
        mb_det, _ = _match(base_foot, gt_pts, radius)
        fp_base += int((~mb_det).sum()); det_base_total += len(base_foot)
        ma_det, _ = _match(all_foot, gt_pts, radius)
        fp_tiled += int((~ma_det).sum())
        for i, p in enumerate(people):
            z = zone_of(p["y"])
            if gt_conf[i]:
                acc[z]["gt_conf"] += 1
                acc[z]["base"] += int(m_base[i])
                acc[z]["tiled"] += int(m_all[i])
            else:
                acc[z]["gt_amb"] += 1
        per_frame.append(dict(frame=fi, gt=len(people),
                              gt_conf=int(sum(gt_conf)),
                              base_det=len(base_foot), tiled_extra=len(tiled),
                              base_hit=int(m_base[np.array(gt_conf)].sum()) if len(gt_conf) else 0,
                              tiled_hit=int(m_all[np.array(gt_conf)].sum()) if len(gt_conf) else 0))

    def rec(n, d):
        return None if d == 0 else round(100.0 * n / d, 1)

    print("\n================  RECALL  (confident-GT bazli)  ================")
    print(f"{'zone':6} {'GT':>4} {'amb':>4} {'base_hit':>9} {'base_rec%':>10} "
          f"{'+tile_hit':>10} {'tile_rec%':>10}")
    tot = dict(gt_conf=0, gt_amb=0, base=0, tiled=0)
    for z in zones:
        a = acc[z]
        for kk in tot: tot[kk] += a[kk]
        print(f"{z:6} {a['gt_conf']:>4} {a['gt_amb']:>4} {a['base']:>9} "
              f"{str(rec(a['base'],a['gt_conf'])):>10} {a['tiled']:>10} "
              f"{str(rec(a['tiled'],a['gt_conf'])):>10}")
    print(f"{'ALL':6} {tot['gt_conf']:>4} {tot['gt_amb']:>4} {tot['base']:>9} "
          f"{str(rec(tot['base'],tot['gt_conf'])):>10} {tot['tiled']:>10} "
          f"{str(rec(tot['tiled'],tot['gt_conf'])):>10}")
    # belirsizlik bandi: tum ambiguous GT'yi de payda say -> alt-sinir recall
    gt_all = tot["gt_conf"] + tot["gt_amb"]
    print(f"\nrecall bandi (base): {rec(tot['base'],gt_all)}% (tum-amb dahil, alt-sinir) "
          f"– {rec(tot['base'],tot['gt_conf'])}% (sadece-confident, ust-sinir)")
    print(f"precision (base): {rec(det_base_total-fp_base, det_base_total)}%  "
          f"(FP={fp_base}/{det_base_total})")
    print(f"tiling net etki: +{tot['tiled']-tot['base']} confident-GT yakalandi, "
          f"FP-artis {fp_tiled-fp_base}")
    (VAL / "score_out.json").write_text(json.dumps(
        dict(acc=acc, totals=tot, fp_base=fp_base, fp_tiled=fp_tiled,
             det_base_total=det_base_total, radius=radius, per_frame=per_frame),
        indent=2))
    print("\nscore_out.json yazildi")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("frames")
    sub.add_parser("detect")
    sp = sub.add_parser("score"); sp.add_argument("--radius", type=float, default=55.0)
    args = ap.parse_args()
    {"frames": cmd_frames, "detect": cmd_detect, "score": cmd_score}[args.cmd](args)


if __name__ == "__main__":
    main()
