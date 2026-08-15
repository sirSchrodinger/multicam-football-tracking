#!/usr/bin/env python3
"""Çizgi-tabanlı KABA otomatik kalibrasyon — yeni sahalar için (manuel-clicksiz).

DÜRÜST SINIR: tek kareden, bilinmeyen lens-içsel parametreleriyle tam-metrik
kalibrasyon halısaha balıkgözünde çözülemez (cankaya GOLD = 4-tık manuel). Bu
modül EN-İYİ-ÇABA homografisi üretir:

  1. turf-maske -> saha dörtgeni (approxPolyDP / minAreaRect)
  2. köşeleri en yakın beyaz-çizgi kesişimine SNAP et (varsa) -> keskinleştir
  3. (L,W) ızgara + X/Y yönelim araması: her hipotez için viz.calib_metric
     line_residual_m HESAPLA, EN DÜŞÜK artığı seç (öz-geliştirme hedefi)
  4. status="auto_lines", düşük-güven bayrağı + residual raporla

Çıktı GOLD DEĞİL; "AUTO (kaba)" etiketli. Loop bunu iyileştirir (çizgi kalitesi,
köşe-snap, k1-undistort). Sadece OpenCV + numpy.
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

sys.path.insert(0, ".")
from pitch.homography import PitchHomography  # noqa: E402
from viz import line_detect, calib_metric  # noqa: E402
from viz.quad_panel import _canvas_M  # noqa: E402


def _warp_green(frame, homo, L, W):
    """Warp'ta orta+uzak bantların YEŞİL (turf) oranı. Domed/kafesli sahalarda
    turf-maske çatıyı/duvarı yutunca warp'a yeşil-olmayan (çatı) girer -> düşük.
    Adversarial-audit'in 'warp çatıyı gösteriyor = dejenere' bulgusunun sayısı.
    Döner: (far_green, mid_green) [0,1]."""
    M, (CW, CH), pad, sc = _canvas_M(L, W, scale=max(10.0, 900.0 / L))
    Hc = M @ homo.H_img2pitch
    warp = cv2.warpPerspective(frame, Hc, (CW, CH))
    hsv = cv2.cvtColor(warp, cv2.COLOR_BGR2HSV)
    g = ((hsv[:, :, 0] >= 25) & (hsv[:, :, 0] <= 140) &
         (hsv[:, :, 1] > 30) & (hsv[:, :, 2] > 30))
    x0, x1 = pad, pad + int(L * sc)
    yf0, yf1 = pad, pad + int(W * sc / 3)                    # uzak üçte-bir
    ym0, ym1 = pad + int(W * sc / 3), pad + int(2 * W * sc / 3)  # orta
    far = g[yf0:yf1, x0:x1].mean() if yf1 > yf0 else 0.0
    mid = g[ym0:ym1, x0:x1].mean() if ym1 > ym0 else 0.0
    return float(far), float(mid)


def _order_corners(pts):
    """4 nokta -> [TL, TR, BR, BL] (görüntü: sol-üst, sağ-üst, sağ-alt, sol-alt)."""
    pts = np.asarray(pts, float)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    order = np.argsort(ang)
    p = pts[order]
    # en-sol-üst'ü başa al: en küçük (x+y)
    s = p.sum(axis=1)
    k = int(np.argmin(s))
    p = np.roll(p, -k, axis=0)
    # saat-yönü olmasını garanti et (TL,TR,BR,BL) — 2D çapraz çarpım
    a, b = p[1] - p[0], p[2] - p[0]
    if (a[0] * b[1] - a[1] * b[0]) < 0:
        p = p[[0, 3, 2, 1]]
    return p


def turf_quad(turf):
    """Turf maskesinden saha dörtgeni (4,2). approxPolyDP 4 vermezse minAreaRect."""
    cnts, _ = cv2.findContours(turf, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    peri = cv2.arcLength(c, True)
    for eps in (0.02, 0.03, 0.05, 0.08):
        ap = cv2.approxPolyDP(c, eps * peri, True)
        if len(ap) == 4:
            return _order_corners(ap.reshape(-1, 2))
    box = cv2.boxPoints(cv2.minAreaRect(c))
    return _order_corners(box)


def _line_intersections(merged, img_shape):
    """Birleşik çizgi çiftlerinin kesişim noktaları (görüntü içi)."""
    H, W = img_shape[:2]
    pts = []
    for i in range(len(merged)):
        x1, y1, x2, y2 = merged[i]
        for j in range(i + 1, len(merged)):
            x3, y3, x4, y4 = merged[j]
            d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(d) < 1e-6:
                continue
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
            if -0.1 * W <= px <= 1.1 * W and -0.1 * H <= py <= 1.1 * H:
                pts.append((px, py))
    return np.array(pts) if pts else np.empty((0, 2))


def _snap(corners, inter, radius=60.0):
    """Her köşeyi radius içindeki en yakın çizgi-kesişimine çek."""
    if len(inter) == 0:
        return corners
    out = corners.copy()
    for k in range(len(corners)):
        d = np.linalg.norm(inter - corners[k], axis=1)
        j = int(np.argmin(d))
        if d[j] < radius:
            out[k] = inter[j]
    return out


def _line_from_seg(x1, y1, x2, y2):
    """Segment -> (a,b,c) genel doğru (a*x+b*y+c=0, normalize)."""
    a, b = y2 - y1, x1 - x2
    c = -(a * x1 + b * y1)
    n = np.hypot(a, b) + 1e-9
    return a / n, b / n, c / n


def _intersect(l1, l2):
    a1, b1, c1 = l1; a2, b2, c2 = l2
    d = a1 * b2 - a2 * b1
    if abs(d) < 1e-9:
        return None
    return np.array([(b1 * c2 - b2 * c1) / d, (a2 * c1 - a1 * c2) / d])


def refine_quad_with_lines(quad, merged, ang_tol=22.0, perp_tol=45.0):
    """Turf-dörtgeni köşelerini BOYALI çizgilere oturt.

    Her kenar için (üst/sağ/alt/sol) en paralel + en yakın uzun çizgiyi bul; komşu
    kenarların çizgi-kesişiminden köşeyi yeniden türet. Turf-kenarı (çim/file) yerine
    gerçek painted-line köşesi => magenta template gerçek çizgilere oturur. Çizgi
    bulunamayan kenar için turf-köşesi korunur. Döner: (rafine_quad, n_kenar_bulundu).
    """
    if len(merged) == 0:
        return quad, 0
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]  # TL-TR, TR-BR, BR-BL, BL-TL
    edge_lines = [None] * 4
    for ei, (i, j) in enumerate(edges):
        p0, p1 = quad[i], quad[j]
        ev = p1 - p0
        ea = np.degrees(np.arctan2(ev[1], ev[0])) % 180.0
        emid = (p0 + p1) / 2.0
        best = None
        for m in merged:
            x1, y1, x2, y2 = m
            if np.hypot(x2 - x1, y2 - y1) < 0.5 * np.linalg.norm(ev):
                continue  # kenardan çok kısa çizgileri atla
            ma = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180.0
            da = abs(ma - ea); da = min(da, 180 - da)
            if da > ang_tol:
                continue
            line = _line_from_seg(x1, y1, x2, y2)
            dist = abs(line[0] * emid[0] + line[1] * emid[1] + line[2])
            if dist > perp_tol:
                continue
            if best is None or dist < best[0]:
                best = (dist, line)
        if best is not None:
            edge_lines[ei] = best[1]

    out = quad.copy().astype(float)
    n_found = 0
    for ci in range(4):
        e_prev = (ci - 1) % 4  # bu köşeye giren kenar
        e_next = ci            # bu köşeden çıkan kenar
        if edge_lines[e_prev] is not None and edge_lines[e_next] is not None:
            p = _intersect(edge_lines[e_prev], edge_lines[e_next])
            if p is not None and np.linalg.norm(p - quad[ci]) < 120:
                out[ci] = p; n_found += 1
    return out, n_found


# DÜRÜST SINIR: tek kareden ne mutlak ölçek ne de en-boy oranı çizgi-artığını
# MINIMIZE ederek çıkarılamaz (sparse iç-çizgi => en ince/küçük saha dejenere
# kazanır). Bu yüzden şablon NOMİNAL sabittir (tipik halısaha 40x20, 2:1). Auto
# yalnız 4 sınır-köşesini + yönelimi bulur; residual bu fit'in KALİTE-SKORUdur
# (aranan serbest parametre YOK => oyunlanamaz).
L_NOMINAL, W_NOMINAL = 40.0, 20.0


def autocalib(frame, camera_id="auto", dl=None):
    """Kaba otomatik kalibrasyon (şablon nominal 40x20; sadece köşe+yönelim fit).

    Döner: (PitchHomography|None, report dict). residual = fit kalite-skoru.
    """
    if dl is None:
        dl = line_detect.detect(frame)
    quad = turf_quad(dl.turf)
    rep = dict(camera_id=camera_id, ok=False, reason="", residual_m=float("nan"),
               L=L_NOMINAL, W=W_NOMINAL, orient=None, n_lines=dl.n, snapped=0,
               scale="nominal 40x20 (cizgiden mutlak olcek/en-boy cikmaz)")
    if quad is None:
        rep["reason"] = "turf dörtgeni bulunamadı"
        return None, rep

    # BOYALI-çizgi rafinesi: turf-kenarı köşelerini gerçek painted-line'lara oturt
    refined, n_edge = refine_quad_with_lines(quad, dl.merged)
    inter = _line_intersections(dl.merged, frame.shape)
    snapped = _snap(refined, inter)
    rep["snapped"] = int((np.linalg.norm(snapped - quad, axis=1) > 1).sum())
    rep["edge_refined"] = int(n_edge)
    quad = snapped

    # şablon köşe sırası: görüntü [TL,TR,BR,BL]; saha near=alt(Y=0) far=üst(Y=W)
    # NOMİNAL 40x20 sabit; yalnız X-yönelim aranır. Residual = saf fit kalite-skoru.
    L, W = L_NOMINAL, W_NOMINAL
    best = None
    for orient in ("normal", "flipx"):
        if orient == "normal":
            dst = np.float32([[0, W], [L, W], [L, 0], [0, 0]])  # TL,TR,BR,BL
        else:
            dst = np.float32([[L, W], [0, W], [0, 0], [L, 0]])
        H, _ = cv2.findHomography(quad.astype(np.float32), dst)
        if H is None:
            continue
        homo = PitchHomography(camera_id, None)
        homo.H_img2pitch = H.astype(np.float64)
        homo.H_pitch2img = np.linalg.inv(H)
        homo._fallback_dims = (float(L), float(W))
        homo.status = "auto_lines"
        r = calib_metric.line_residual_m(homo, dl, L=float(L), W=float(W))
        score = r["template_support_m"]
        if np.isnan(score):
            continue
        score = score + 1.5 * (1.0 - r["coverage_0p5"])  # kapsam ödülü
        if best is None or score < best[0]:
            best = (score, homo, orient, r)
    if best is None:
        rep["reason"] = "geçerli homografi hipotezi yok"
        return None, rep
    _, homo, orient, r = best

    # DEPTH-COLLAPSE GUARD (adversarial-audit bulgusu): düşük çizgi-artığı, çökük
    # derinlik-eksenini YAKALAMAZ. Turf-dörtgeni yan-kenar orta-noktaları metriğe
    # eşlenir; sağlıklı perspektifte ~W/2'ye düşmeli. Grazing/dejenere fit'te
    # uzak-zon çöker => orta-nokta W'ye yapışır (skew>~0.72). Bunu quality'ye kat.
    ml = homo.pixel_to_pitch(((quad[3] + quad[0]) / 2.0).reshape(1, 2))[0, 1]
    mr = homo.pixel_to_pitch(((quad[2] + quad[1]) / 2.0).reshape(1, 2))[0, 1]
    depth_skew = float(np.mean([ml, mr]) / W_NOMINAL)
    far_collapse = depth_skew > 0.72 or depth_skew < 0.28
    rep["depth_skew"] = round(depth_skew, 3)
    rep["far_collapse"] = bool(far_collapse)

    # WARP-GREEN GUARD (empirik doğrulandı): mid<0.70 => turf-maske çatı/duvar
    # yutmuş, warp dejenere (Atakum/Ayazma). far_green = uzak-zon güven proxy'si.
    far_green, mid_green = _warp_green(frame, homo, L_NOMINAL, W_NOMINAL)
    warp_degenerate = mid_green < 0.70 or far_green < 0.45
    rep["far_green"] = round(far_green, 2)
    rep["mid_green"] = round(mid_green, 2)
    rep["warp_degenerate"] = bool(warp_degenerate)

    res_m = r["template_support_m"]
    if warp_degenerate:
        quality = "zayıf(çatı/duvar warp'a giriyor — manuel gerek)"
    elif far_collapse:
        quality = "zayıf(uzak-zon çökük — manuel gerek)"
    elif res_m < 0.8:
        quality = "iyi (uzak-zon nominal/düşük-hassas)" if far_green < 0.7 else "iyi"
    elif res_m < 1.5:
        quality = "orta"
    else:
        quality = "zayıf(manuel gerek)"

    homo.calib_method = "auto_lines_nominal_fit"
    homo._qa = dict(line_residual=r, depth_skew=depth_skew, far_collapse=far_collapse,
                    far_green=far_green, mid_green=mid_green,
                    method="turf_quad+snap+nominal40x20+orient+depth+warpgreen_guard")
    rep.update(ok=True, residual_m=res_m, orient=orient,
               coverage_0p5=r["coverage_0p5"], aspect=2.0, quality=quality)
    return homo, rep


def autocalib_hull(ref_frame, feet_cloud, camera_id="auto", dl=None):
    """OYUNCU-HULL kalibrasyonu — domed/kafesli sahalar için (turf-maske çatıyı
    yutunca fallback). feet_cloud: (N,2) çok-kareden toplanmış ham ayak-pikselleri.

    Oyuncular çatıda yürümez => ayak-bulutunun min-alan dörtgeni GERÇEK sahayı verir.
    Ayazma'da warp mid_green 0.45→0.99 (çatı warp'tan gitti). Döner: (homo, report).
    """
    feet_cloud = np.asarray(feet_cloud, float).reshape(-1, 2)
    rep = dict(camera_id=camera_id, ok=False, method="player_hull", n_feet=len(feet_cloud))
    if len(feet_cloud) < 30:
        rep["reason"] = f"yetersiz ayak-bulutu ({len(feet_cloud)})"
        return None, rep
    quad = _order_corners(cv2.boxPoints(
        cv2.minAreaRect(cv2.convexHull(feet_cloud.astype(np.float32)))))
    if dl is None:
        dl = line_detect.detect(ref_frame)
    L, W = L_NOMINAL, W_NOMINAL
    best = None
    for orient in ("normal", "flipx"):
        dst = (np.float32([[0, W], [L, W], [L, 0], [0, 0]]) if orient == "normal"
               else np.float32([[L, W], [0, W], [0, 0], [L, 0]]))
        H, _ = cv2.findHomography(quad.astype(np.float32), dst)
        if H is None:
            continue
        homo = PitchHomography(camera_id, None)
        homo.H_img2pitch = H.astype(np.float64); homo.H_pitch2img = np.linalg.inv(H)
        homo._fallback_dims = (L, W)
        homo.status = "auto_hull"
        fg, mg = _warp_green(ref_frame, homo, L, W)
        r = calib_metric.line_residual_m(homo, dl, L=L, W=W)
        if best is None or mg + 0.3 * fg > best[0]:
            best = (mg + 0.3 * fg, homo, fg, mg, orient, r)
    if best is None:
        rep["reason"] = "homografi yok"
        return None, rep
    _, homo, fg, mg, orient, r = best
    homo.calib_method = "auto_player_hull"
    homo._qa = dict(far_green=fg, mid_green=mg, line_residual=r, method="player_hull_minarearect")
    rep.update(ok=True, orient=orient, far_green=round(fg, 2), mid_green=round(mg, 2),
               residual_m=round(r["template_support_m"], 3),
               quality=("iyi(hull)" if mg > 0.85 else "orta(hull)" if mg > 0.65 else "zayıf(hull)"))
    return homo, rep


if __name__ == "__main__":
    import json
    p = sys.argv[1] if len(sys.argv) > 1 else "self_training/data/raw/YeşilÇimenHalıSa.mp4"
    fidx = int(sys.argv[2]) if len(sys.argv) > 2 else 900
    cap = cv2.VideoCapture(p); cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ok, fr = cap.read(); cap.release()
    assert ok, f"okunamadı {p}"
    homo, rep = autocalib(fr, camera_id="selftest")
    print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v)
                      for k, v in rep.items()}, ensure_ascii=False, indent=1))
