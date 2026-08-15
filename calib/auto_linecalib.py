#!/usr/bin/env python3
"""Tam-otomatik PENTAGON çizgi-kalibrasyon denemesi (Alperen teşhisi: yakın köşe
occluded→hesaplanmalı, ovallik→plumb-line, orta-saha→halfway kısıtı).

Beyaz-çizgi maskesinden adlı çizgilerin nokta-setlerini bölge+yön ile çıkar →
pitch.line_calib.build_homography_from_lines (occluded köşe = çizgi kesişimi,
dürüstlük-kapılı). Gate RuntimeError verirse veya overlay kötüyse → manuel clicker.
"""
import cv2, numpy as np, json, sys
sys.path.insert(0, ".")
from pitch.template import PitchTemplate
from pitch import line_calib

d = json.load(open("calib/cankaya_cam2_v2.json"))
K = np.array(d["K"], float); D = np.array(d["dist"], float)
L, W = d["template"]["dims_m"]                       # 34, 18
cap = cv2.VideoCapture("raw/cankaya_cam2.mp4"); cap.set(1, 67153); ok, im = cap.read()
und = cv2.undistort(im, K, D)
white = cv2.imread("scratchpad/white_mask.png", 0)

def region_pts(xr, yr, ang_lo, ang_hi, max_n=400, min_len=70):
    """bölge+yön bandındaki beyaz çizgi noktalarını Hough segment uçlarından topla."""
    m = np.zeros_like(white); m[yr[0]:yr[1], xr[0]:xr[1]] = 255
    sub = cv2.bitwise_and(white, m)
    lines = cv2.HoughLinesP(sub, 1, np.pi/180, 50, minLineLength=min_len, maxLineGap=40)
    pts = []
    for l in (lines if lines is not None else []):
        x1, y1, x2, y2 = l[0]; ang = np.degrees(np.arctan2(y2-y1, x2-x1)) % 180
        if ang_lo <= ang <= ang_hi:
            # segment boyunca örnekle
            for t in np.linspace(0, 1, 8):
                pts.append((x1+(x2-x1)*t, y1+(y2-y1)*t))
    pts = np.array(pts, float)
    return pts[:max_n] if len(pts) else pts

# adlı çizgi bölgeleri (undistorted 1920x1080; kamera yakın-kalenin SOLunda)
line_clicks = {}
defs = {
    "touchline_yW": ((300, 1900), (120, 360), 0, 30),    # UZAK touchline (üst band, ~yatay)
    "touchline_y0": ((250, 1700), (430, 1080), 8, 45),   # YAKIN touchline (alt, diyagonal)
    "endline_xL":   ((1380, 1900), (110, 320), 50, 130), # UZAK end-line (sağ, ~dik)
    "endline_x0":   ((40, 560), (340, 1120), 50, 130),   # YAKIN goal-line (sol, ~dik)
    "halfway":      ((600, 1500), (180, 720), 50, 130),  # ORTA çizgi (merkez, ~dik)
}
for name, (xr, yr, a0, a1) in defs.items():
    p = region_pts(xr, yr, a0, a1)
    if len(p) >= 4:
        line_clicks[name] = p
    print(f"  {name}: {len(p)} nokta {'OK' if len(p)>=4 else 'YETERSİZ'}")

# kale direkleri = ayna-kırıcı + ölçek tohumu (v2 tıklamalarından, world 3m ağız)
ip = np.array(d["_img_pts_und"], float); wp = np.array(d["_world_pts"], float)
point_lms = []
for w, p in zip(wp, ip):
    if abs(w[1]-7.5) < 0.1 or abs(w[1]-10.5) < 0.1:      # goal mouth noktaları
        point_lms.append((p, w))

tmpl = PitchTemplate.from_dict(d["template"])
try:
    homo, corners, qa = line_calib.build_homography_from_lines(
        "cankaya_cam2", tmpl, line_clicks, K=K, dist=D, point_lms=point_lms, ransac_px=5.0)
    print("\n=== PENTAGON FIT BAŞARILI ===")
    print("köşeler (occluded dahil, hesaplanan):")
    for cid, c in corners.items():
        p = c["img_und"]
        print(f"  {cid}: world={c['world']} img={None if p is None else p.round(0)} "
              f"ill={c['ill']} extrap={c['extrap']:.2f} sin={c['sin_angle']:.2f}")
    print("line QA (metre):", {k: round(v, 2) for k, v in qa.items() if isinstance(v, (int, float))})
    homo.save("calib/cankaya_cam2_v5_auto.json")
    print("yazildi calib/cankaya_cam2_v5_auto.json")
except RuntimeError as e:
    print("\n=== GATE REFUSED ===", e)
    print("-> manuel line-clicker gerek (auto night-detection yetersiz).")
