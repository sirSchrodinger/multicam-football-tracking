#!/usr/bin/env python3
"""static_fp_filter — SABİT-kamera avantajıyla statik yanlış-pozitifleri ele.

İçgörü (literatür değil, GÖZLEM): kamera sabit. Reklam panosu, file, direk, duvar
afişi ASLA hareket etmez; oyuncu hareket eder. Bir tespit-kutusu, bir tesisin
KARELERİNİN büyük kısmında NEREDEYSE AYNI pikselde (çok küçük varyansla) tekrar
ediyorsa = statik yapı = FP. Hareket eden / yalnız bazı karelerde görünen = oyuncu.

Çankaya pipeline'ındaki spurious-filtre 'static' kuralının pseudo-label sürümü;
burada per-venue COCO üstünde, calib gerektirmeden (görüntü-uzayı) çalışır.
"""
from __future__ import annotations
import numpy as np


def flag_static_fp(boxes, n_frames_total, recur_frac=0.45, pos_std_px=14.0,
                   min_frames=4):
    """boxes: list of dict(frame, cx, foot_y, w, h). Doner: set(box_index) = statik FP.

    Kümeleme: ayak-noktasi (cx, foot_y) yakinligina gore greedy. Bir kume
    KARELERIN >= recur_frac'inde gorunuyor VE konum std < pos_std_px ise (piksel-kilitli)
    -> statik yapi (pano/file/direk), oyuncu degil -> tum uyeleri FP.
    Hareketli ya da seyrek kume = oyuncu, korunur.
    """
    n = len(boxes)
    if n == 0:
        return set()
    cx = np.array([b["cx"] for b in boxes], float)
    fy = np.array([b["foot_y"] for b in boxes], float)
    fr = np.array([b["frame"] for b in boxes], int)
    P = np.column_stack([cx, fy])
    used = np.zeros(n, bool)
    flagged = set()
    R = 26.0  # ayni-yapi yaricapi (px)
    order = np.argsort(fy)  # ust-bant (uzak/pano) once
    for i in order:
        if used[i]:
            continue
        d = np.hypot(P[:, 0] - P[i, 0], P[:, 1] - P[i, 1])
        members = np.where((d < R) & (~used))[0]
        used[members] = True
        if len(members) < min_frames:
            continue
        frames_hit = np.unique(fr[members])
        frac = len(frames_hit) / max(1, n_frames_total)
        pstd = float(np.hypot(P[members, 0].std(), P[members, 1].std()))
        # piksel-kilitli + cogu karede = statik yapi (pano/file/direk)
        if frac >= recur_frac and pstd < pos_std_px:
            flagged.update(int(m) for m in members)
    return flagged


def clean_coco_static(coco, recur_frac=0.45, pos_std_px=14.0):
    """COCO dict'i per-venue statik-FP'den temizle. Doner: (yeni_coco, rapor)."""
    imgs = {im["id"]: im for im in coco["images"]}
    by_venue = {}
    for im in coco["images"]:
        by_venue.setdefault(im.get("venue", "?"), []).append(im["id"])
    ann_by_img = {}
    for a in coco["annotations"]:
        ann_by_img.setdefault(a["image_id"], []).append(a)

    drop_ids = set()
    report = {}
    for venue, img_ids in by_venue.items():
        nft = len(img_ids)
        boxes, idx = [], []
        for iid in img_ids:
            for a in ann_by_img.get(iid, []):
                x, y, w, h = a["bbox"]
                boxes.append(dict(frame=iid, cx=x + w / 2, foot_y=y + h, w=w, h=h))
                idx.append(a["id"])
        flagged = flag_static_fp(boxes, nft, recur_frac, pos_std_px)
        for j in flagged:
            drop_ids.add(idx[j])
        report[venue] = dict(n_boxes=len(boxes), n_static_fp=len(flagged))

    new_anns = [a for a in coco["annotations"] if a["id"] not in drop_ids]
    new = dict(images=coco["images"], annotations=new_anns,
               categories=coco["categories"])
    return new, dict(total_dropped=len(drop_ids), per_venue=report)
