#!/usr/bin/env python3
"""4'lü panel üretici — DETECT · ÇİZGİ · WARP · 2D.

Alperen'in istediği tek-bakış saha görselleştirmesi. Her saha/pencere için:
  [1] DETECT  — ham kare + oyuncu kutuları + ayak noktaları (dedektörün gördüğü)
  [2] ÇİZGİ   — tespit-edilen saha çizgileri (cyan) + reprojekte template (magenta)
                => kalibrasyonun görsel + sayısal QA'sı (çizgiler oturuyor mu?)
  [3] WARP    — kuş-bakışı (homography ile metrik-tuval); template referans (beyaz)
  [4] 2D      — şematik radar + saha-metre ayak noktaları (takım renkli)

homo=None ise (kalibrasyonsuz saha) 3/4 panel "calib yok" gösterir; 1/2 çalışır.
Bu modül öz-geliştirmenin GÖZÜ: panel-2/3 kalibrasyon hatasını görünür kılar,
viz.calib_metric.line_residual_m onu SAYIYA çevirir (loop bunu küçültür).
"""
from __future__ import annotations

import cv2
import numpy as np

from viz import line_detect

# takım/rol renk paleti (BGR)
TEAM_COLORS = [(60, 60, 235), (235, 180, 60), (60, 220, 235), (235, 60, 200)]
FOOT_COLOR = (70, 240, 70)
FONT = cv2.FONT_HERSHEY_SIMPLEX

# cv2 Hershey fontları yalnız ASCII çizer; unicode -> ??? olur. Sanitize et.
_MAP = {"—": "-", "–": "-", "·": "|", "×": "x", "≈": "~", "✓": "OK",
        "ı": "i", "İ": "I", "ş": "s", "Ş": "S", "ğ": "g", "Ğ": "G",
        "ç": "c", "Ç": "C", "ö": "o", "Ö": "O", "ü": "u", "Ü": "U"}


def _ascii(s: str) -> str:
    for k, v in _MAP.items():
        s = s.replace(k, v)
    return s.encode("ascii", "ignore").decode()


def _put(img, text, org, scale, color, thick=1):
    cv2.putText(img, _ascii(text), org, FONT, scale, color, thick, cv2.LINE_AA)


# ------------------------------------------------------------ tuval yardım ---
def _canvas_M(L, W, scale=18.0, pad=40):
    """metre -> kuş-bakışı tuval piksel afin (3x3). Y ters (uzak=üst)."""
    Sx = int(round(L * scale))
    Sy = int(round(W * scale))
    M = np.array([[scale, 0.0, pad],
                  [0.0, -scale, pad + Sy],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    return M, (Sx + 2 * pad, Sy + 2 * pad), pad, scale


def _m2c(M, pts_m):
    p = np.c_[np.asarray(pts_m, float), np.ones(len(pts_m))] @ M.T
    return p[:, :2]


def _label(vis, name, tag=None, tag_col=(120, 255, 120)):
    h, w = vis.shape[:2]
    bar = max(24, h // 22)
    cv2.rectangle(vis, (0, 0), (w, bar), (0, 0, 0), -1)
    _put(vis, name, (6, bar - 7), bar / 44.0, (255, 255, 255), 1)
    if tag:
        (tw, _), _ = cv2.getTextSize(_ascii(tag), FONT, bar / 48.0, 1)
        _put(vis, tag, (w - tw - 8, bar - 7), bar / 48.0, tag_col, 1)
    return vis


def _draw_field(im, M, L, W, col=(90, 150, 100), circle_r=3.0):
    """Şematik saha çizgileri (metre template) tuvale."""
    def L2(a, b):
        p = _m2c(M, [a, b])
        cv2.line(im, tuple(np.int32(p[0])), tuple(np.int32(p[1])), col, 2, cv2.LINE_AA)
    L2((0, 0), (L, 0)); L2((L, 0), (L, W)); L2((L, W), (0, W)); L2((0, W), (0, 0))
    L2((L / 2, 0), (L / 2, W))
    c = _m2c(M, [(L / 2, W / 2)])[0]
    r = int(circle_r * abs(M[0, 0]))
    cv2.circle(im, tuple(np.int32(c)), r, col, 2, cv2.LINE_AA)
    # kaleler (kırmızı) — orta ±1.5m
    for gx in (0.0, L):
        p = _m2c(M, [(gx, W / 2 - 1.5), (gx, W / 2 + 1.5)])
        cv2.line(im, tuple(np.int32(p[0])), tuple(np.int32(p[1])), (0, 0, 210), 4, cv2.LINE_AA)


# --------------------------------------------------------------- paneller ----
def panel_detect(frame, feet_px, boxes=None, teams=None):
    vis = frame.copy()
    if boxes is not None:
        for i, b in enumerate(boxes):
            c = TEAM_COLORS[teams[i] % len(TEAM_COLORS)] if teams is not None else (200, 200, 200)
            cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), c, 2)
    for i, (fx, fy) in enumerate(feet_px):
        cv2.circle(vis, (int(fx), int(fy)), 5, FOOT_COLOR, -1)
        cv2.circle(vis, (int(fx), int(fy)), 5, (255, 255, 255), 1)
    return _label(vis, f"1 · DETECT — {len(feet_px)} oyuncu ayak-tespiti")


def panel_lines(frame, homo, dl=None):
    if dl is None:
        dl = line_detect.detect(frame)
    vis = line_detect.draw(frame, dl, color=(0, 235, 255), thickness=2)
    tag = f"oto: {dl.n} cizgi"
    tcol = (120, 255, 120)
    if homo is not None:
        # reprojekte template (magenta) — HAM piksel uzayında (distorted=True)
        for (p0, p1) in homo._template_segments_m():
            a = homo.pitch_to_pixel(np.array([p0]), distorted=True)[0]
            b = homo.pitch_to_pixel(np.array([p1]), distorted=True)[0]
            cv2.line(vis, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])),
                     (230, 80, 230), 2, cv2.LINE_AA)
        tag = "cyan=tespit · magenta=calib"
        tcol = (230, 120, 230)
    return _label(vis, "2 · CIZGI tespiti", tag, tcol)


def panel_warp(frame, homo, L, W):
    if homo is None or homo.H_img2pitch is None:
        blank = np.full((frame.shape[0], frame.shape[1], 3), 45, np.uint8)
        _put(blank, "WARP: kalibrasyon yok", (30, blank.shape[0] // 2),
             1.0, (120, 120, 120), 2)
        return _label(blank, "3 · WARP (kus-bakisi)", "calib yok", (120, 120, 120))
    M, (CW, CH), pad, scale = _canvas_M(L, W, scale=max(10.0, 900.0 / L))
    # undistorted kare -> tuval
    src = frame
    if homo.K is not None and homo.dist is not None and homo.undistort_applied:
        src = cv2.undistort(frame, homo.K, homo.dist)
    Himg2canvas = M @ homo.H_img2pitch
    warped = cv2.warpPerspective(src, Himg2canvas, (CW, CH),
                                 flags=cv2.INTER_LINEAR, borderValue=(20, 30, 22))
    # saha dikdörtgeni + kenar payı dışını karart (file/tribün streak'ini sınırla)
    corners = _m2c(M, [(-1, -1), (L + 1, -1), (L + 1, W + 1), (-1, W + 1)])
    mask = np.zeros((CH, CW), np.uint8)
    cv2.fillConvexPoly(mask, np.int32(corners), 255)
    warped[mask == 0] = (20, 30, 22)
    _draw_field(warped, M, L, W, col=(255, 255, 255))
    return _label(warped, "3 · WARP (kus-bakisi, homography)", "beyaz=template")


def panel_2d(feet_m, L, W, teams=None):
    M, (CW, CH), pad, scale = _canvas_M(L, W, scale=max(10.0, 900.0 / L))
    im = np.full((CH, CW, 3), (30, 46, 34), np.uint8)
    _draw_field(im, M, L, W, col=(120, 175, 130))
    n = 0
    for i, (x, y) in enumerate(feet_m):
        if not (-2 <= x <= L + 2 and -2 <= y <= W + 2):
            continue
        c = _m2c(M, [(x, y)])[0]
        col = TEAM_COLORS[teams[i] % len(TEAM_COLORS)] if teams is not None else (60, 60, 235)
        cv2.circle(im, tuple(np.int32(c)), 9, col, -1)
        cv2.circle(im, tuple(np.int32(c)), 9, (255, 255, 255), 1)
        n += 1
    cv2.circle(im, (pad // 2 + 4, CH - pad // 2 - 4), 6, (0, 200, 255), -1)
    _put(im, "KAM", (pad // 2 + 14, CH - pad // 2), 0.5, (0, 200, 255), 1)
    return _label(im, f"4 · 2D radar - {n} oyuncu")


# ------------------------------------------------------------------ montaj ---
def make_quad(frame, homo, feet_px, out_path, title, boxes=None, teams=None,
              L=None, W=None, residual_m=None, subtitle=None):
    """4 paneli tek montaj görüntüde birleştir ve kaydet.

    frame: BGR ham kare. homo: PitchHomography|None. feet_px: (N,2) ham-px ayak.
    residual_m: verilirse başlığa çizgi-artığı (calib QA) yazılır.
    Döner: montaj görüntü (BGR).
    """
    feet_px = np.asarray(feet_px, float).reshape(-1, 2)
    if L is None or W is None:
        if homo is not None:
            L, W = homo._dims_m()
        else:
            L, W = 40.0, 20.0
    feet_m = homo.pixel_to_pitch(feet_px) if (homo is not None and len(feet_px)) else np.empty((0, 2))

    dl = line_detect.detect(frame)
    p1 = panel_detect(frame, feet_px, boxes, teams)
    p2 = panel_lines(frame, homo, dl)
    p3 = panel_warp(frame, homo, L, W)
    p4 = panel_2d(feet_m, L, W, teams)

    # panelleri ortak boya getir (P1/P2 kare-oranı; P3/P4 tuval-oranı) -> 2x2 grid
    TW = 960
    def fit(p):
        h, w = p.shape[:2]
        return cv2.resize(p, (TW, int(round(h * TW / w))))
    p1, p2, p3, p4 = map(fit, (p1, p2, p3, p4))
    hL = max(p1.shape[0], p3.shape[0]); hR = max(p2.shape[0], p4.shape[0])
    rowh = max(hL, hR)
    def pad_to(p, h):
        if p.shape[0] < h:
            p = np.vstack([p, np.full((h - p.shape[0], p.shape[1], 3), 18, np.uint8)])
        return p
    top = np.hstack([pad_to(p1, rowh), pad_to(p2, rowh)])
    bot = np.hstack([pad_to(p3, rowh), pad_to(p4, rowh)])
    grid = np.vstack([top, bot])

    # üst başlık şeridi
    head = np.full((70, grid.shape[1], 3), (18, 18, 20), np.uint8)
    _put(head, title, (16, 34), 0.95, (255, 255, 255), 2)
    sub = subtitle or ""
    if residual_m is not None:
        q = f"cizgi-artigi (calib QA): {residual_m:.2f} m"
        qcol = (90, 235, 90) if residual_m < 0.6 else (60, 200, 235) if residual_m < 1.2 else (60, 60, 235)
        _put(head, q, (16, 60), 0.62, qcol, 1)
        if sub:
            (tw, _), _ = cv2.getTextSize(_ascii(sub), FONT, 0.55, 1)
            _put(head, sub, (grid.shape[1] - tw - 16, 60), 0.55, (170, 170, 170), 1)
    elif sub:
        _put(head, sub, (16, 60), 0.55, (170, 170, 170), 1)
    out = np.vstack([head, grid])
    if out_path:
        cv2.imwrite(out_path, out)
    return out
