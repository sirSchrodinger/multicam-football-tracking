#!/usr/bin/env python3
"""Halısaha çizgi tespiti — turf-maskeli, tekrar-kullanılabilir modül.

Amaç: bir kareden BOYALI saha-çizgilerini (touchline, kale-çizgisi, orta-çizgi,
orta-yuvarlak) çıkar. Sadece OpenCV (BSD) + numpy, GPU yok.

Boru hattı:
  turf_mask  -> yeşil zemin (en büyük bileşen), file/tribün/reklam dışarıda
  white_mask -> düşük-doygunluk + yüksek-parlaklık + tophat (ince beyaz şerit)
  LSD        -> çizgi-segmentleri, uzunluk filtresi
  merge      -> eş-doğrusal segmentleri tek uzun çizgiye birleştir
Çıktı: DetectedLines(segments, merged, prob_map). `prob_map` [0,1] float —
homography.drift_check ve metre-domeni çizgi-artığı (calib QA) için kullanılır.

Bu modül calib DEĞİL; kalibrasyonun HAM sinyalidir. İyi çizgi tespiti + doğru H
=> reprojekte template çizgileri tespit-edilen çizgilere oturur (görsel + sayısal QA).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


# --------------------------------------------------------------- maskeler ----
def _turf_band(img, hlo, hhi, slo, vlo):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    m = ((H >= hlo) & (H <= hhi) & (S > slo) & (V > vlo)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    nc, lbl, st, _ = cv2.connectedComponentsWithStats(m)
    if nc > 1:
        big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
        m = (lbl == big).astype(np.uint8) * 255
    return m


def turf_mask(img: np.ndarray, dilate: int = 9) -> np.ndarray:
    """Zemin maskesi (uint8 0/255). En büyük bağlı bileşen = saha.

    ADAPTİF: önce dar yeşil bandı (H30-95); saha kapsamı <%12 ise mavi/cyan-cast
    sahalar için geniş banda (H25-140) düş (Cengiz gibi mavi-tint tesisler). Böyle
    normal yeşil sahalar etkilenmez, sadece renk-kaymalı sahalar kurtarılır.
    """
    m = _turf_band(img, 30, 95, 30, 30)
    if m.mean() / 255.0 < 0.12:  # yeşil bulunamadı -> geniş band (mavi-green cast)
        m2 = _turf_band(img, 25, 140, 25, 35)
        if m2.mean() > m.mean():
            m = m2
    if dilate > 0:
        m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    return m


def white_mask(img: np.ndarray, turf: np.ndarray | None = None,
               tophat_k: int = 25, sat_max: int = 90, val_min: int = 140) -> np.ndarray:
    """Boyalı beyaz-çizgi maskesi (uint8 0/255), turf ile sınırlı.

    (düşük-S & yüksek-V)  AND  tophat-Otsu (ince açık şerit)  AND  turf.
    tophat çizgi genişliğinden büyük yapıyı bastırır -> geniş beyaz yüzey
    (kaleci forması, reklam panosu) elenir, ince çizgi kalır.
    """
    if turf is None:
        turf = turf_mask(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    S, V = hsv[:, :, 1], hsv[:, :, 2]
    white = ((S < sat_max) & (V > val_min)).astype(np.uint8) * 255
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT,
        cv2.getStructuringElement(cv2.MORPH_RECT, (tophat_k, tophat_k)))
    _, tm = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = cv2.bitwise_and(cv2.bitwise_and(white, tm), turf)
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


# ------------------------------------------------------- segment yardımcı ----
def _seg_angle(s) -> float:
    """Segment açısı [0,180) derece (yatay=0)."""
    x1, y1, x2, y2 = s
    a = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
    return float(a)


def _seg_len(s) -> float:
    x1, y1, x2, y2 = s
    return float(np.hypot(x2 - x1, y2 - y1))


def _point_line_dist(px, py, x1, y1, x2, y2) -> float:
    """(px,py) noktasının (x1,y1)-(x2,y2) SONSUZ doğrusuna dik mesafesi."""
    dx, dy = x2 - x1, y2 - y1
    n = np.hypot(dx, dy)
    if n < 1e-6:
        return float(np.hypot(px - x1, py - y1))
    return float(abs(dy * px - dx * py + x2 * y1 - y2 * x1) / n)


def merge_collinear(segments, angle_tol: float = 6.0, offset_tol: float = 14.0,
                    gap_tol: float = 60.0):
    """Eş-doğrusal + yakın segmentleri tek uzun çizgiye birleştir.

    İki segment: açı farkı < angle_tol VE birinin uç-noktalarının diğerinin
    doğrusuna dik mesafesi < offset_tol VE eksenel boşluk < gap_tol ise birleşir.
    Birleşik çizgi, kümedeki tüm uç-noktaların ana-yöne izdüşümünün uçlarıdır.
    Döner: [(x1,y1,x2,y2), ...] azalan-uzunluk sıralı.
    """
    segs = [tuple(map(float, s)) for s in segments]
    n = len(segs)
    if n == 0:
        return []
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    ang = [_seg_angle(s) for s in segs]
    for i in range(n):
        for j in range(i + 1, n):
            da = abs(ang[i] - ang[j])
            da = min(da, 180 - da)
            if da > angle_tol:
                continue
            # j'nin uçlarının i-doğrusuna diki
            x1, y1, x2, y2 = segs[i]
            d1 = _point_line_dist(segs[j][0], segs[j][1], x1, y1, x2, y2)
            d2 = _point_line_dist(segs[j][2], segs[j][3], x1, y1, x2, y2)
            if max(d1, d2) > offset_tol:
                continue
            # eksenel boşluk: i ve j uçlarının ana-yön izdüşüm aralıkları örtüşür/yakın mı
            theta = np.radians(ang[i])
            ux, uy = np.cos(theta), np.sin(theta)
            pi = [(segs[i][0] * ux + segs[i][1] * uy),
                  (segs[i][2] * ux + segs[i][3] * uy)]
            pj = [(segs[j][0] * ux + segs[j][1] * uy),
                  (segs[j][2] * ux + segs[j][3] * uy)]
            lo_i, hi_i = min(pi), max(pi)
            lo_j, hi_j = min(pj), max(pj)
            gap = max(lo_i - hi_j, lo_j - hi_i, 0.0)
            if gap > gap_tol:
                continue
            union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    merged = []
    for members in groups.values():
        pts = []
        wts = []
        for m in members:
            x1, y1, x2, y2 = segs[m]
            pts += [(x1, y1), (x2, y2)]
            wts += [_seg_len(segs[m])] * 2
        pts = np.array(pts)
        wts = np.array(wts)
        c = np.average(pts, axis=0, weights=wts)
        # ağırlıklı ana yön (SVD)
        d = (pts - c) * np.sqrt(wts)[:, None]
        _, _, vt = np.linalg.svd(d, full_matrices=False)
        u = vt[0]
        proj = (pts - c) @ u
        a = c + proj.min() * u
        b = c + proj.max() * u
        merged.append((float(a[0]), float(a[1]), float(b[0]), float(b[1])))
    merged.sort(key=_seg_len, reverse=True)
    return merged


@dataclass
class DetectedLines:
    segments: list          # ham LSD segmentleri (uzunluk-filtreli)
    merged: list            # eş-doğrusal birleştirilmiş uzun çizgiler
    prob_map: np.ndarray    # [0,1] float, beyaz-çizgi olasılığı (drift/residual)
    turf: np.ndarray = field(repr=False, default=None)

    @property
    def n(self) -> int:
        return len(self.merged)


def detect(img: np.ndarray, min_len: int = 55, merge: bool = True,
           **mask_kw) -> DetectedLines:
    """Ana giriş: BGR kare -> DetectedLines.

    min_len: LSD segment min piksel uzunluğu. merge=False ham segmentleri döner.
    mask_kw: white_mask parametreleri (tophat_k, sat_max, val_min).
    """
    turf = turf_mask(img)
    m = white_mask(img, turf, **mask_kw)
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    raw = lsd.detect(m)[0]
    segs = []
    if raw is not None:
        for l in raw:
            x1, y1, x2, y2 = l[0]
            if np.hypot(x2 - x1, y2 - y1) >= min_len:
                segs.append((float(x1), float(y1), float(x2), float(y2)))
    merged = merge_collinear(segs) if merge else list(segs)
    prob = (m.astype(np.float32) / 255.0)
    return DetectedLines(segments=segs, merged=merged, prob_map=prob, turf=turf)


def draw(img: np.ndarray, dl: DetectedLines, color=(0, 235, 255),
         thickness: int = 2, use_merged: bool = True) -> np.ndarray:
    """Tespit edilen çizgileri kopya görüntü üzerine çiz."""
    vis = img.copy()
    lines = dl.merged if use_merged else dl.segments
    for x1, y1, x2, y2 in lines:
        cv2.line(vis, (int(round(x1)), int(round(y1))),
                 (int(round(x2)), int(round(y2))), color, thickness, cv2.LINE_AA)
    return vis


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else "calib/cankaya_cam2_frame_raw.png"
    im = cv2.imread(p)
    if im is None:
        raise SystemExit(f"okunamadı: {p}")
    dl = detect(im)
    print(f"{p}: {len(dl.segments)} ham segment -> {dl.n} birleşik çizgi "
          f"(turf %{100*dl.turf.mean()/255:.0f})")
    out = "scratchpad/line_detect_selftest.jpg"
    cv2.imwrite(out, draw(im, dl))
    print("yazıldı", out)
