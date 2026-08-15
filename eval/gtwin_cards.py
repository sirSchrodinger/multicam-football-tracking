"""Tracklet/grup KARTLARI — agent-gorsel adjudication malzemesi.

Her tracklet (veya on-gruplama sonrasi grup) icin tek JPG:
  [header: id, sure, n_det, box_h, zone]  [crop x N (t-etiketli, x3 buyutulmus)]
  [pitch yorunge mini-plot (FINAL koordinat)]

Kullanim:
  venv/bin/python eval/gtwin_cards.py            # tracklet kartlari
  venv/bin/python eval/gtwin_cards.py groups.json # grup kartlari (tid listesi/grup)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

DET = 'scratchpad/gtwin_tracklets.parquet'
VID = 'raw/cankaya_cam2.mp4'
OUTDIR = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('scratchpad/gtwin_cards')
CROP_H = 200
N_SAMPLES = 5
FPS_ACT = 24.8717


def pick_samples(g: pd.DataFrame, n=N_SAMPLES) -> pd.DataFrame:
    g = g.sort_values('t')
    if len(g) <= n:
        return g
    qs = np.linspace(0, 1, n)
    idx = []
    for q in qs:
        # zaman-dilimi icinde en buyuk kutu (net crop)
        lo, hi = g.t.quantile(max(0, q - 0.1)), g.t.quantile(min(1, q + 0.1))
        cand = g[(g.t >= lo) & (g.t <= hi)]
        if not len(cand):
            cand = g
        pick = cand.loc[(cand.y1 - cand.y0).idxmax()]
        idx.append(pick.name)
    return g.loc[sorted(set(idx))]


def extract_crops(df_samples: pd.DataFrame):
    """frame -> crop tek gecis (sirali seek)."""
    cap = cv2.VideoCapture(VID)
    crops = {}
    for fr, grp in sorted(df_samples.groupby('frame')):
        cap.set(1, int(fr))
        ok, im = cap.read()
        if not ok:
            continue
        H, W = im.shape[:2]
        for ridx, r in grp.iterrows():
            w = r.x1 - r.x0; h = r.y1 - r.y0
            mx, my = 0.25 * w, 0.15 * h
            x0, y0 = int(max(0, r.x0 - mx)), int(max(0, r.y0 - my))
            x1, y1 = int(min(W, r.x1 + mx)), int(min(H, r.y1 + my))
            c = im[y0:y1, x0:x1]
            if c.size == 0:
                continue
            s = CROP_H / c.shape[0]
            crops[ridx] = cv2.resize(c, None, fx=s, fy=s,
                                     interpolation=cv2.INTER_CUBIC)
    cap.release()
    return crops


def traj_plot(g: pd.DataFrame, L=48.4, Wp=22.5, w=520, h=260) -> np.ndarray:
    img = np.full((h, w, 3), (30, 60, 30), np.uint8)
    cv2.rectangle(img, (10, 10), (w - 10, h - 10), (200, 200, 200), 1)
    cv2.line(img, (w // 2, 10), (w // 2, h - 10), (160, 160, 160), 1)
    pts = g.sort_values('t')[['px', 'py']].values
    ts = g.sort_values('t').t.values
    if len(pts):
        t0, t1 = ts.min(), max(ts.max(), ts.min() + 1e-6)
        for k in range(len(pts)):
            x = int(10 + (w - 20) * np.clip(pts[k, 0] / L, 0, 1))
            y = int(10 + (h - 20) * np.clip(pts[k, 1] / Wp, 0, 1))
            c = int(255 * (ts[k] - t0) / (t1 - t0))
            cv2.circle(img, (x, y), 3, (0, c, 255 - c), -1)
        cv2.putText(img, 'yesil=son / kirmizi=ilk', (14, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
    return img


def make_card(name: str, gfull: pd.DataFrame, g: pd.DataFrame, crops: dict,
              note: str = '') -> np.ndarray:
    samp = [crops[i] for i in g.index if i in crops]
    ts = [g.loc[i].t for i in g.index if i in crops]
    if not samp:
        samp = [np.zeros((CROP_H, 60, 3), np.uint8)]
        ts = [0]
    labeled = []
    for c, t in zip(samp, ts):
        c = c.copy()
        cv2.putText(c, f"t={t:.1f}", (2, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (0, 255, 255), 1)
        labeled.append(c)
    strip = np.hstack([np.pad(c, ((0, 0), (0, 6), (0, 0)), constant_values=15)
                       for c in labeled])
    tp = traj_plot(gfull)
    W = max(strip.shape[1], tp.shape[1], 700)
    header = np.full((46, W, 3), 15, np.uint8)
    span = gfull.t.max() - gfull.t.min()
    hmean = float((gfull.y1 - gfull.y0).mean())
    zones = ','.join(sorted(set(
        'far' if y < 360 else ('mid' if y < 560 else 'near') for y in gfull.y1)))
    cv2.putText(header, f"{name} | {gfull.t.min():.1f}-{gfull.t.max():.1f}s "
                        f"(span {span:.1f}s, n={len(gfull)}) | box_h~{hmean:.0f}px "
                        f"| zone:{zones} {note}",
                (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    def padw(a):
        return np.pad(a, ((0, 0), (0, W - a.shape[1]), (0, 0)), constant_values=15)
    return np.vstack([header, padw(strip), padw(tp)])


def main():
    df = pd.read_parquet(DET)
    OUTDIR.mkdir(exist_ok=True)
    groups = None
    if len(sys.argv) > 1:
        groups = json.load(open(sys.argv[1]))  # {"G0": [tid, ...], ...}
    if groups is None:
        groups = {f"T{t}": [int(t)] for t in sorted(df.tracklet_id.unique())}

    # tum orneklem satirlarini topla -> tek gecis crop
    sel = []
    per_group = {}
    for gname, tids in groups.items():
        g = df[df.tracklet_id.isin(tids)]
        if not len(g):
            continue
        s = pick_samples(g)
        per_group[gname] = (g, s)
        sel.append(s)
    allsamp = pd.concat(sel)
    print(f"{len(per_group)} kart, {len(allsamp)} crop cekiliyor...", flush=True)
    crops = extract_crops(allsamp)
    for gname, (g, s) in per_group.items():
        card = make_card(gname, g, s, crops)
        cv2.imwrite(str(OUTDIR / f"{gname}.jpg"), card,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    idx = {g: {'tids': tids, 'n': int(len(per_group[g][0])),
               't0': float(per_group[g][0].t.min()),
               't1': float(per_group[g][0].t.max())}
           for g, tids in groups.items() if g in per_group}
    json.dump(idx, open(OUTDIR / '_index.json', 'w'), indent=1)
    print(f"kartlar -> {OUTDIR}/ ({len(per_group)})", flush=True)


if __name__ == '__main__':
    main()
