#!/usr/bin/env python3
"""3-panel senkron örnek klip: cam1 görüntü | cam2 görüntü | füzyon 2D kuş-bakışı.
Aynı maçın iki kamerası + ortak sahada birleşik izler. Hem demo hem PDF figürü.

Kullanim: example_clip.py --t0 55.1 --t1 60.1 --fused scratchpad/fused_smoke3.parquet \
          --cam1-tracks ... --cam2-tracks ... --out scratchpad/example.mp4
"""
import json, argparse, sys, os
import numpy as np, pandas as pd, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pitch_map as pm

CFG = "calib/fusion_config.json"


def draw_boxes(frame, g, color, label):
    for _, r in g.iterrows():
        x1 = int(r.foot_x - r.box_w / 2); y1 = int(r.foot_y - r.box_h)
        x2 = int(r.foot_x + r.box_w / 2); y2 = int(r.foot_y)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (int(r.foot_x), int(r.foot_y)), 3, color, -1)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0", type=float, required=True, help="oyun-zamani basi (s)")
    ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--fused", required=True)
    ap.add_argument("--cam1-tracks", default="raw/tracks_cankaya_cam1_smoke.parquet")
    ap.add_argument("--cam2-tracks", default="raw/tracks_cankaya_cam2_fullgame.parquet")
    ap.add_argument("--out", default="scratchpad/example.mp4")
    ap.add_argument("--fps", type=int, default=15)
    a = ap.parse_args()

    cfg = json.load(open(CFG))
    OFF = cfg["time_sync"]["offset_cam1_to_cam2_s"]
    FPS1 = cfg["cameras"]["cam1"]["fps"]; FPS2 = cfg["cameras"]["cam2"]["fps"]
    Lm = cfg["pitch_dims_m"]["L"]; Wm = cfg["pitch_dims_m"]["W"]

    t1d = pd.read_parquet(a.cam1_tracks); t2d = pd.read_parquet(a.cam2_tracks)
    fused = pd.read_parquet(a.fused)
    cap1 = cv2.VideoCapture("raw/cankaya_cam1.mp4")
    cap2 = cv2.VideoCapture("raw/cankaya_cam2.mp4")

    PANEL_H = 360
    SCALE = 16; PADX, PADY = 30, 30
    Wp2 = int(Lm * SCALE + 2 * PADX); Hp2 = int(Wm * SCALE + 2 * PADY)
    def topx(X, Y): return int(PADX + X * SCALE), int(PADY + (Wm - Y) * SCALE)
    pids = sorted(fused.pid.unique())
    rng = np.linspace(0, 179, max(len(pids), 1)).astype(int)
    pcol = {p: tuple(int(c) for c in cv2.cvtColor(np.uint8([[[h, 200, 255]]]),
            cv2.COLOR_HSV2BGR)[0, 0]) for p, h in zip(pids, rng)}
    fg = {p: gg.sort_values("t") for p, gg in fused.groupby("pid")}

    times = np.arange(a.t0, a.t1, 1.0 / a.fps)
    # panel boyutlari: cam1 (1280x720)->PANEL_H, cam2 (1920x1080)->PANEL_H
    def fit(img, h):
        w = int(img.shape[1] * h / img.shape[0]); return cv2.resize(img, (w, h))
    vw = None
    for t in times:
        f1 = int(round((t - OFF) * FPS1)); f2 = int(round(t * FPS2))
        cap1.set(cv2.CAP_PROP_POS_FRAMES, max(f1, 0)); ok1, fr1 = cap1.read()
        cap2.set(cv2.CAP_PROP_POS_FRAMES, max(f2, 0)); ok2, fr2 = cap2.read()
        if not ok1 or not ok2:
            continue
        g1 = t1d[t1d.frame == max(f1, 0)]; g2 = t2d[t2d.frame == f2]
        p1 = fit(draw_boxes(fr1, g1, (80, 220, 80), f"cam1  t={t-OFF:.1f}s"), PANEL_H)
        p2 = fit(draw_boxes(fr2, g2, (40, 150, 230), f"cam2  t={t:.1f}s"), PANEL_H)
        # 2D panel
        d2 = np.full((Hp2, Wp2, 3), 30, np.uint8)
        cv2.rectangle(d2, topx(0, 0), topx(Lm, Wm), (90, 120, 90), 2)
        cv2.line(d2, topx(Lm / 2, 0), topx(Lm / 2, Wm), (90, 120, 90), 1)
        cv2.circle(d2, topx(Lm / 2, Wm / 2), int(3 * 1.3252 * SCALE), (90, 120, 90), 1)
        for p in pids:
            gg = fg[p]
            if t < gg.t.iloc[0] - 0.3 or t > gg.t.iloc[-1] + 0.3:
                continue
            m = (gg.t >= t - 1.5) & (gg.t <= t)
            pts = [topx(x, y) for x, y in zip(gg.X[m], gg.Y[m])]
            for i in range(1, len(pts)):
                cv2.line(d2, pts[i - 1], pts[i], pcol[p], 2)
            x = np.interp(t, gg.t, gg.X); y = np.interp(t, gg.t, gg.Y)
            cv2.circle(d2, topx(x, y), 6, pcol[p], -1)
            cv2.circle(d2, topx(x, y), 6, (240, 240, 240), 1)
        cv2.rectangle(d2, (0, 0), (Wp2, 30), (0, 0, 0), -1)
        cv2.putText(d2, "FUZYON 2D (ortak saha)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
        d2 = fit(d2, PANEL_H)
        gap = np.full((PANEL_H, 6, 3), 60, np.uint8)
        row = np.hstack([p1, gap, p2, gap, d2])
        if vw is None:
            vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps,
                                 (row.shape[1], row.shape[0]))
        vw.write(row)
    if vw: vw.release()
    cap1.release(); cap2.release()
    print("yazildi:", a.out)


if __name__ == "__main__":
    main()
