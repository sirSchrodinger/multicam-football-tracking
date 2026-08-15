#!/usr/bin/env python3
"""player_state — SUREKLI saha-ici oyuncu konum haritasi (projenin EN ONEMLI kismi, Alperen).

Dedektor her frame ~12/14 goruyor (recall acigi, uzak-uc); oyuncular kadraj-disina cikar,
algilanmaz. Bu modul her CORE oyuncu icin AKTIF SPAN'inde HER frame'de plausible konum tutar:
  observed      : gercek tespit (conf=1)
  interpolated  : iki-yanli bosluk, Hermite (uc-hizlarini koruyan) -> egri yola sadik
  predicted     : tek-yanli (span-kenari) sabit-hiz extrapolasyon, <=max_extrap_s, conf hizla duser

DURUSTLUK: 'net olmasina gerek yok ama mantikli olmali' (Alperen). interpolated/predicted
TESPIT DEGIL -> conf<1 + status isaretli; mesafe/hiz toplamasina GIRMEZ (sadece observed).
Konum HARITASI / occupancy / gorsellestirme icin conf-agirlikli kullanilir. Span DISINA
(ilk tespitten once / son sonra) ASLA uydurma konum yazilmaz (oyuncu o an sahada degil/yedek).

Plausibilite: konum saha+margin'e kirpilir; uzun bosluk -> dusuk conf (guvenme sinyali),
konum yine de last-known+hareket ile 'kabaca' tutulur. Sayim-priori (roster sabit) core
oyuncularin span'inde yasamasiyla saglanir (kaybolup yok-sayilmaz).

Lisans: numpy + pandas (BSD). GPU yok.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:               # CLI/pytest farkli cwd'lerde import icin
    sys.path.insert(0, _ROOT)


def _hermite(p0, v0, p1, v1, t):
    """Cubic Hermite (uc konumlar p ve uc hizlar v; t in [0,1]). p,v: (2,) array."""
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
    h11 = t**3 - t**2
    return h00 * p0 + h10 * v0 + h01 * p1 + h11 * v1


def _local_vel(frames_sorted, P, fr, win=8):
    """fr civarinda yerel hiz (m/frame), +-win icindeki en yakin gozlemden."""
    near = [x for x in frames_sorted if 0 < abs(x - fr) <= win]
    if not near:
        return np.zeros(2)
    a = min(near, key=lambda x: abs(x - fr))
    return (np.asarray(P[fr]) - np.asarray(P[a])) / float(fr - a)


def continuous_state(df_clustered, homo, fps, core_ids=None,
                     id_col="player_id", max_extrap_s=2.0, vel_win=8):
    """Her core oyuncu icin AKTIF SPAN'inde frame-bazli surekli konum.

    df_clustered: player_id/tid + frame + foot_x/foot_y. homo: PitchHomography (pitch icin).
    Doner: (state_df, coverage). state_df kolonlari:
      player_id, frame, t_sec, x, y, status (observed|interpolated|predicted), conf.
    coverage: {player_id: {n_span, observed, interpolated, predicted, observed_frac}}.
    """
    key = id_col if id_col in df_clustered.columns else "tid"
    foot = df_clustered[["foot_x", "foot_y"]].to_numpy(np.float64)
    pm = homo.pixel_to_pitch(foot)
    d = df_clustered.assign(pmX=pm[:, 0], pmY=pm[:, 1])
    L, W = homo._dims_m()
    max_extrap = int(round(max_extrap_s * fps))

    if core_ids is None:
        core_ids = [int(p) for p in d[key].unique() if int(p) >= 0]
    core_ids = [int(p) for p in core_ids]

    rows = []
    coverage = {}
    for pid in core_ids:
        g = d[d[key] == pid].sort_values("frame")
        if len(g) < 2:
            continue
        P = {int(r.frame): (float(r.pmX), float(r.pmY)) for r in g.itertuples()}
        fs = sorted(P)
        f0, f1 = fs[0], fs[-1]
        nobs = nint = npred = 0
        for f in range(f0, f1 + 1):
            if f in P:
                x, y = P[f]; status, conf = "observed", 1.0
                nobs += 1
            else:
                prev = [x for x in fs if x < f]
                nxt = [x for x in fs if x > f]
                if prev and nxt:                       # iki-yanli -> Hermite
                    fa, fb = prev[-1], nxt[0]
                    pa, pb = np.array(P[fa]), np.array(P[fb])
                    span = fb - fa
                    va = _local_vel(fs, P, fa, vel_win) * span
                    vb = _local_vel(fs, P, fb, vel_win) * span
                    xy = _hermite(pa, va, pb, vb, (f - fa) / span)
                    conf = float(np.exp(-span / (2.0 * fps)))
                    status = "interpolated"; nint += 1
                else:                                   # tek-yanli (span-kenari) -> extrapol
                    fa = prev[-1] if prev else nxt[0]
                    dt = f - fa
                    if abs(dt) > max_extrap:
                        continue                        # cok uzak -> uydurma, ATLA
                    xy = np.array(P[fa]) + _local_vel(fs, P, fa, vel_win) * dt
                    conf = float(np.exp(-abs(dt) / (0.7 * fps)))
                    status = "predicted"; npred += 1
                x = float(np.clip(xy[0], -1.0, L + 1.0))
                y = float(np.clip(xy[1], -1.0, W + 1.0))
            rows.append((pid, f, f / fps, round(x, 3), round(y, 3), status, round(conf, 3)))
        span_n = f1 - f0 + 1
        coverage[pid] = dict(n_span=span_n, observed=nobs, interpolated=nint,
                             predicted=npred, observed_frac=round(nobs / span_n, 3))

    state_df = pd.DataFrame(rows, columns=["player_id", "frame", "t_sec",
                                           "x", "y", "status", "conf"])
    return state_df, coverage


def occupancy_heatmap(state_df, dims_m, bin_m=1.0, conf_weight=True, smooth=True):
    """SUREKLI durumdan equal-area occupancy heatmap. Tespit-only'den FARKI: recall-acigi
    boyunca dolgu konumlar da sayilir, AMA conf-agirlikli (dolgu < gozlenen) -> dururst.

    state_df: continuous_state ciktisi. Doner (grid (ny,nx), (xedges,yedges)).
    grid = bin'deki toplam conf-agirlik / sure (saniye-esdeger); occupancy yogunlugu.
    """
    L, W = float(dims_m[0]), float(dims_m[1])
    nx = max(1, int(np.ceil(L / bin_m))); ny = max(1, int(np.ceil(W / bin_m)))
    xe = np.linspace(0, L, nx + 1); ye = np.linspace(0, W, ny + 1)
    x = state_df["x"].to_numpy(float); y = state_df["y"].to_numpy(float)
    w = state_df["conf"].to_numpy(float) if conf_weight else np.ones(len(state_df))
    inb = (x >= 0) & (x <= L) & (y >= 0) & (y <= W)
    H, _, _ = np.histogram2d(x[inb], y[inb], bins=[xe, ye], weights=w[inb])  # (nx,ny)
    grid = H.T.astype(float)
    if smooth:
        try:
            from scipy.ndimage import gaussian_filter
            grid = gaussian_filter(grid, max(0.5, 1.0 / bin_m))
        except Exception:
            pass
    if grid.max() > 0:
        grid = grid / grid.max()
    return grid, (xe, ye)


def map_at_frame(state_df, frame):
    """Tek karede sahadaki tum oyuncularin konumu (surekli harita anlik-goruntusu)."""
    return state_df[state_df["frame"] == int(frame)].copy()


def coverage_summary(coverage):
    """Insan-okur ozet: kac oyuncu, ortalama observed_frac, toplam interp/pred."""
    if not coverage:
        return dict(n_players=0)
    fr = np.array([c["observed_frac"] for c in coverage.values()])
    return dict(
        n_players=len(coverage),
        mean_observed_frac=round(float(fr.mean()), 3),
        min_observed_frac=round(float(fr.min()), 3),
        total_interpolated=int(sum(c["interpolated"] for c in coverage.values())),
        total_predicted=int(sum(c["predicted"] for c in coverage.values())),
        note=("interpolated/predicted TESPIT DEGIL (conf<1) -> mesafe/hiz HARIC, sadece "
              "konum-haritasi/occupancy icin. Span disina uydurma yok."))


if __name__ == "__main__":
    import sys
    import json
    from pitch.homography import PitchHomography
    pq = sys.argv[1]
    calib = sys.argv[2] if len(sys.argv) > 2 else "calib/cankaya_cam2_v2.json"
    df = pd.read_parquet(pq)
    homo = PitchHomography.load(calib)
    fps = float(df["frame"].max()) / float(df["t_sec"].max())
    # core: presence sidecar varsa kullan
    core = None
    sr = Path(pq).with_name(Path(pq).stem.replace("_stitched", "").replace("_player", "")
                            + "_stitch_report.json")
    if sr.exists():
        rep = json.loads(sr.read_text())
        pr = rep.get("presence", {}).get("per_player", {})
        core = [int(p) for p, v in pr.items() if v.get("role") == "core"]
    st, cov = continuous_state(df, homo, fps, core_ids=core)
    print(json.dumps(coverage_summary(cov), ensure_ascii=False, indent=2))
