#!/usr/bin/env python3
"""calib/recalib.py — pentagon-farkında re-kalibrasyon yardımcıları.

İŞ AKIŞI (29 Haz handoff fix):
  1. undist_grid: temiz UNDISTORTED kare + etiketli piksel grid (landmark okuma için).
  2. solve: etiketli çizgi/nokta korespondanslarından line_calib ile H kur + QA.
  3. overlay: H'yi gerçek karenin üstüne çiz (beyaz çizgilere oturuyor mu doğrula).

Tüm tıklamalar/koordinatlar UNDISTORTED piksel uzayında (cv2.undistort(K,dist), P=K).
"""
import sys, json, argparse
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, ".")

NPY = "calib/cankaya_cam2_distortion.npy"


def load_intrinsics(npy=NPY):
    k1, fx, cx, cy = (float(v) for v in np.load(npy))
    K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], float)
    dist = np.array([k1, 0.0, 0.0, 0.0, 0.0], float)
    return K, dist


def grab_frame(video, t_sec):
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t_sec * fps)))
    ok, fr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"frame okunamadi t={t_sec}")
    return fr


def undistort(fr, K, dist):
    return cv2.undistort(fr, K, dist)


def draw_grid(img, step=100, major=200):
    out = img.copy()
    h, w = out.shape[:2]
    for x in range(0, w, step):
        c = (0, 200, 255) if x % major == 0 else (60, 90, 100)
        th = 2 if x % major == 0 else 1
        cv2.line(out, (x, 0), (x, h), c, th)
        if x % major == 0:
            cv2.putText(out, str(x), (x + 2, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
    for y in range(0, h, step):
        c = (0, 200, 255) if y % major == 0 else (60, 90, 100)
        th = 2 if y % major == 0 else 1
        cv2.line(out, (0, y), (w, y), c, th)
        if y % major == 0:
            cv2.putText(out, str(y), (4, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2)
    return out


def draw_grid_roi(img, x0, y0, x1, y1, scale, step=50, major=100):
    """ROI'yi kes, scale ile büyüt, ORİJİNAL undistorted koordinatla grid bas."""
    crop = img[y0:y1, x0:x1].copy()
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    H, W = crop.shape[:2]
    # dikey çizgiler (orijinal x = x0..x1)
    xs0 = ((x0 + step - 1) // step) * step
    for xo in range(xs0, x1 + 1, step):
        xc = int(round((xo - x0) * scale))
        c = (0, 200, 255) if xo % major == 0 else (70, 100, 110)
        th = 2 if xo % major == 0 else 1
        cv2.line(crop, (xc, 0), (xc, H), c, th)
        if xo % major == 0:
            cv2.putText(crop, str(xo), (xc + 2, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    ys0 = ((y0 + step - 1) // step) * step
    for yo in range(ys0, y1 + 1, step):
        yc = int(round((yo - y0) * scale))
        c = (0, 200, 255) if yo % major == 0 else (70, 100, 110)
        th = 2 if yo % major == 0 else 1
        cv2.line(crop, (0, yc), (W, yc), c, th)
        if yo % major == 0:
            cv2.putText(crop, str(yo), (4, yc + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 2)
    return crop


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["undist_grid", "crop"])
    ap.add_argument("--box", default="", help="x0,y0,x1,y1 (undistorted)")
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--src", default="calib/undist_clean.png")
    ap.add_argument("--video", default="raw/_active_game.mp4")
    ap.add_argument("--t", type=float, default=110.0)
    ap.add_argument("--out", default="calib/undist_grid.png")
    ap.add_argument("--raw-out", default="calib/raw_frame_t.png")
    a = ap.parse_args()
    if a.cmd == "crop":
        und = cv2.imread(a.src)
        x0, y0, x1, y1 = (int(v) for v in a.box.split(","))
        out = draw_grid_roi(und, x0, y0, x1, y1, a.scale)
        cv2.imwrite(a.out, out)
        print("yazildi:", a.out, "roi", (x0, y0, x1, y1), "->", out.shape)
        sys.exit(0)
    K, dist = load_intrinsics()
    fr = grab_frame(a.video, a.t)
    cv2.imwrite(a.raw_out, fr)
    und = undistort(fr, K, dist)
    cv2.imwrite("calib/undist_clean.png", und)
    cv2.imwrite(a.out, draw_grid(und))
    print("yazildi:", a.out, "+ calib/undist_clean.png +", a.raw_out, "shape", und.shape)
