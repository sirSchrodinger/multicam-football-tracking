#!/usr/bin/env python3
"""stats/match2d.py — izlenebilir 2D maç replay (top-down) — Roadmap Faz 2.

continuous_state (player_state_continuous.parquet: player_id,frame,x,y,status,conf)
+ kalibrasyon -> kuş-bakışı 2D animasyon (MP4). DÜRÜSTLÜK kodlu:
  observed   = dolu disk
  interpolated = halka (içi boş)
  predicted  = soluk + iz (gözlem değil; uydurma değil, model-tahmini işaretli)
Takım rengi GÜVENİLMEZ (gece/dinamik yelek) -> oyuncu kimliğine göre ayrık renk
(anonim), sahte 7-7 YOK. Ölçek "rel_m (approx ±N%)" diye etiketlenir.

Lisans: OpenCV(BSD) + numpy; MP4 için sistem-ffmpeg (LGPL, çağrılır, gömülü değil).
"""
from __future__ import annotations
import json, subprocess, tempfile
from pathlib import Path
import numpy as np
import cv2


def _palette(n):
    cols = []
    for i in range(n):
        h = int(180 * (i * 0.61803398875 % 1.0))
        bgr = cv2.cvtColor(np.uint8([[[h, 200, 255]]]), cv2.COLOR_HSV2BGR)[0, 0]
        cols.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return cols


def _pitch_canvas(L, W, ppm, pad, band_note):
    Hc = int(W * ppm + 2 * pad); Wc = int(L * ppm + 2 * pad)
    Hc += Hc % 2; Wc += Wc % 2                                # libx264 -> çift boyut
    img = np.full((Hc, Wc, 3), (35, 70, 35), np.uint8)        # koyu yeşil
    # alternatif şerit (görsel)
    for i in range(int(L)):
        if i % 2 == 0:
            x0 = int(pad + i * ppm); x1 = int(pad + (i + 1) * ppm)
            img[pad:Hc - pad, x0:x1] = (40, 80, 40)
    def m2px(x, y):
        return int(pad + x * ppm), int(Hc - pad - y * ppm)   # y yukarı
    white = (235, 235, 235)
    cv2.rectangle(img, m2px(0, 0), m2px(L, W), white, 2)
    cv2.line(img, m2px(L / 2, 0), m2px(L / 2, W), white, 2)
    cv2.circle(img, m2px(L / 2, W / 2), int(4.0 * ppm), white, 2)
    cv2.putText(img, band_note, (pad, Hc - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (200, 200, 200), 1)
    return img, m2px, (Hc, Wc)


def build_replay(state_parquet, calib_json, out_mp4,
                 t0=None, t1=None, fps_out=12, trail=10, ppm=26, pad=28):
    import pandas as pd
    df = pd.read_parquet(state_parquet)
    calib = json.loads(Path(calib_json).read_text())
    L, W = calib["pitch_dims_m"]["L"], calib["pitch_dims_m"]["W"]
    sa = calib.get("scale_anchor") or {}
    band = sa.get("label", "rel_m (kalibre, ölçek doğrulanmadı)")
    # gözlem-gürültüsü jitter'ı kır: oyuncu-içi rolling-median (gerçek hareketi bozmaz)
    df = df.sort_values(["player_id", "frame"])
    for col in ("x", "y"):
        df[col] = df.groupby("player_id")[col].transform(
            lambda s: s.rolling(7, center=True, min_periods=1).median())
    # saha-dışı sızıntıyı clamp et (foot/calib gürültüsü oyuncuyu çizgi dışına atıyor)
    df["x"] = df["x"].clip(0, L); df["y"] = df["y"].clip(0, W)
    if t0 is not None:
        df = df[df.t_sec >= t0]
    if t1 is not None:
        df = df[df.t_sec <= t1]
    pids = sorted(df.player_id.unique())
    pal = {p: c for p, c in zip(pids, _palette(len(pids)))}
    # playback frame'lerini t_sec'e göre örnekle
    tmin, tmax = float(df.t_sec.min()), float(df.t_sec.max())
    times = np.arange(tmin, tmax, 1.0 / fps_out)
    frames_by = {f: g for f, g in df.groupby("frame")}
    all_frames = sorted(frames_by)
    fnum = np.array(all_frames); ftime = np.array([frames_by[f].t_sec.iloc[0] for f in all_frames])

    tmp = Path(tempfile.mkdtemp())
    base, m2px, (Hc, Wc) = _pitch_canvas(L, W, ppm, pad, band)
    hist = {}  # player_id -> list of recent (px,py)
    n_written = 0
    for k, t in enumerate(times):
        fi = all_frames[int(np.argmin(np.abs(ftime - t)))]
        g = frames_by[fi]
        img = base.copy()
        on = 0
        for _, r in g.iterrows():
            x, y, st = float(r.x), float(r.y), str(r.status)
            px, py = m2px(np.clip(x, 0, L), np.clip(y, 0, W))
            col = pal.get(r.player_id, (200, 200, 200))
            h = hist.setdefault(r.player_id, [])
            h.append((px, py)); h[:] = h[-trail:]
            # iz (fade)
            for j in range(1, len(h)):
                a = j / len(h)
                cv2.line(img, h[j - 1], h[j], tuple(int(c * a) for c in col), 1)
            if st == "observed":
                cv2.circle(img, (px, py), 7, col, -1); cv2.circle(img, (px, py), 7, (15, 15, 15), 1); on += 1
            elif st == "interpolated":
                cv2.circle(img, (px, py), 7, col, 2)
            else:  # predicted
                cv2.circle(img, (px, py), 5, tuple(int(c * 0.5) for c in col), 1)
        cv2.putText(img, f"t={t:6.1f}s   sahada(gozlenen)={on}", (pad, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1)
        cv2.putText(img, "dolu=gozlenen  halka=interp  soluk=tahmin",
                    (pad, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 220, 180), 1)
        cv2.imwrite(str(tmp / f"f{k:05d}.png"), img); n_written += 1

    r = subprocess.run(["ffmpeg", "-y", "-framerate", str(fps_out), "-i",
                        str(tmp / "f%05d.png"), "-c:v", "libx264", "-pix_fmt",
                        "yuv420p", "-crf", "23", str(out_mp4)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg rc={r.returncode}: {r.stderr[-800:]}")
    for p in tmp.glob("*.png"):
        p.unlink()
    tmp.rmdir()
    return dict(out=str(out_mp4), frames=n_written, players=len(pids),
                window=[tmin, tmax], canvas=[Hc, Wc])


def build_sidebyside(state_parquet, calib_json, video_path, out_mp4,
                     t0, t1, fps_out=10, ppm=22, pad=22):
    """SOL: gerçek (undistort) maç görüntüsü | SAĞ: 2D replay — senkron.
    Doğrulama: replay gerçekle örtüşüyor mu (warmup mı maç mı, kümeler doğru mu)."""
    import pandas as pd
    df = pd.read_parquet(state_parquet)
    calib = json.loads(Path(calib_json).read_text())
    L, W = calib["pitch_dims_m"]["L"], calib["pitch_dims_m"]["W"]
    K = np.array(calib["K"]) if calib.get("K") else None
    dist = np.array(calib["dist"]) if calib.get("dist") else None
    band = (calib.get("scale_anchor") or {}).get("label", "rel_m")
    df = df.sort_values(["player_id", "frame"])
    for c in ("x", "y"):
        df[c] = df.groupby("player_id")[c].transform(lambda s: s.rolling(7, center=True, min_periods=1).median())
    df["x"] = df.x.clip(0, L); df["y"] = df.y.clip(0, W)
    win = df[(df.t_sec >= t0) & (df.t_sec <= t1)]
    pids = sorted(df.player_id.unique()); pal = {p: c for p, c in zip(pids, _palette(len(pids)))}
    base, m2px, (Hc, Wc) = _pitch_canvas(L, W, ppm, pad, band)
    cap = cv2.VideoCapture(str(video_path)); fps = cap.get(cv2.CAP_PROP_FPS)
    frames_by = {f: g for f, g in win.groupby("frame")}
    all_f = sorted(frames_by); ftime = np.array([frames_by[f].t_sec.iloc[0] for f in all_f])
    tmp = Path(tempfile.mkdtemp()); hist = {}
    rh = Hc  # gerçek görüntü yüksekliği = canvas
    for k, t in enumerate(np.arange(t0, t1, 1.0 / fps_out)):
        fi = all_f[int(np.argmin(np.abs(ftime - t)))]
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, real = cap.read()
        if not ok:
            continue
        if K is not None:
            real = cv2.undistort(real, K, dist)
        rw = int(rh * real.shape[1] / real.shape[0]); rw += rw % 2
        realr = cv2.resize(real, (rw, rh))
        cv2.putText(realr, "GERCEK (undistort)", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        img = base.copy()
        on = 0
        for _, r in frames_by[fi].iterrows():
            px, py = m2px(np.clip(r.x, 0, L), np.clip(r.y, 0, W)); col = pal.get(r.player_id, (200, 200, 200))
            h = hist.setdefault(r.player_id, []); h.append((px, py)); h[:] = h[-10:]
            for j in range(1, len(h)):
                cv2.line(img, h[j - 1], h[j], tuple(int(c * j / len(h)) for c in col), 1)
            if r.status == "observed":
                cv2.circle(img, (px, py), 6, col, -1); on += 1
            elif r.status == "interpolated":
                cv2.circle(img, (px, py), 6, col, 2)
            else:
                cv2.circle(img, (px, py), 4, tuple(int(c * .5) for c in col), 1)
        cv2.putText(img, f"2D REPLAY  t={t:5.1f}s gozlenen={on}", (pad, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1)
        combo = np.hstack([realr, img]); combo = combo[:, :combo.shape[1] - combo.shape[1] % 2]
        cv2.imwrite(str(tmp / f"f{k:05d}.png"), combo)
    cap.release()
    r = subprocess.run(["ffmpeg", "-y", "-framerate", str(fps_out), "-i", str(tmp / "f%05d.png"),
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(out_mp4)],
                       capture_output=True, text=True)
    for p in tmp.glob("*.png"):
        p.unlink()
    tmp.rmdir()
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg: {r.stderr[-400:]}")
    return dict(out=str(out_mp4), window=[t0, t1])


if __name__ == "__main__":
    import sys
    r = build_replay(
        sys.argv[1] if len(sys.argv) > 1 else "stats_out/cankaya_cam2_fullgame/player_state_continuous.parquet",
        sys.argv[2] if len(sys.argv) > 2 else "stats_out/cankaya_cam2_fullgame/cankaya_cam2_v2_vheight.json",
        sys.argv[3] if len(sys.argv) > 3 else "stats_out/cankaya_cam2_fullgame/replay.mp4",
        t0=float(sys.argv[4]) if len(sys.argv) > 4 else None,
        t1=float(sys.argv[5]) if len(sys.argv) > 5 else None)
    print(json.dumps(r, indent=2))
