#!/usr/bin/env python3
"""2-kamera saha-koordinat fuzyonu — parcalanmayi KAYNAKTA azaltir.

Tek-kamera re-ID duvari: caprazlasan oyuncular (~2m + benzer gece formasi) =
BILGI-limiti. Cozum: zit-uctaki ikinci kamera ayni ani baska acidan gorur.
Cekirdek yeni yetenek: KAMERALAR-ARASI ZAMAN-ORTUSME = MUST-link (ayni oyuncu
iki kez gorulur). Tek-kamerada zaman-ortusme = cannot-link (ByteTrack ayirdi).
cam2 bir oyuncuyu carpisma aninda kaybederse, cam1'in ortusen parcasi kimligi
tasir -> birlesik track kopmaz.

Kullanim:
  fuse_engine.py CAM1_TRACKS CAM2_TRACKS [--config calib/fusion_config.json]
                 [--out scratchpad/fused.parquet]
"""
import json, argparse, numpy as np, pandas as pd, sys, os
import cv2
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pitch_map as pm


def load_cfg(p):
    return json.load(open(p))


def to_shared(df, cal, flip, Lm, Wm):
    """foot piksel -> ortak metrik (Xm,Ym). flip: cam1 180 don."""
    P = pm.to_pitch(cal, df[["foot_x", "foot_y"]].values)
    X, Y = P[:, 0], P[:, 1]
    if flip:
        X, Y = Lm - X, Wm - Y
    return X, Y


def reliability(X, Y, cam_xy, D0=22.0):
    """Kameraya yakinlik agirligi: uzak-alan sikismasi -> uzakta dusuk guven."""
    d = np.hypot(X - cam_xy[0], Y - cam_xy[1])
    return 1.0 / (1.0 + (d / D0) ** 2)


def build_tracklets(c1tr, c2tr, cfg):
    """Iki parquet -> ortak-cerceve tracklet listesi (her biri zaman-sirali polyline)."""
    cam = cfg["cameras"]
    c1 = pm.load(cam["cam1"]["calib"]); c2 = pm.load(cam["cam2"]["calib"])
    Lm = c1["L"] * c1["SC"]; Wm = c1["W"] * c1["SC"]
    OFF = cfg["time_sync"]["offset_cam1_to_cam2_s"]      # cam2_t = cam1_t + OFF
    cam1_xy = (Lm, Wm / 2.0); cam2_xy = (0.0, Wm / 2.0)  # kamera saha-konumlari

    tracklets = []  # dict(src, tid, t, X, Y, w)
    for src, df, cal, flip, fps, camxy in [
        ("cam1", c1tr, c1, cam["cam1"]["flip180"], cam["cam1"]["fps"], cam1_xy),
        ("cam2", c2tr, c2, cam["cam2"]["flip180"], cam["cam2"]["fps"], cam2_xy),
    ]:
        X, Y = to_shared(df, cal, flip, Lm, Wm)
        df = df.copy(); df["Xm"] = X; df["Ym"] = Y
        # ortak oyun-zamani (cam2 saati referans)
        df["tg"] = df["t_sec"] + (OFF if src == "cam1" else 0.0)
        for tid, g in df.groupby("tid"):
            g = g.sort_values("tg")
            if len(g) < 2:
                continue
            # saha-ici makul nokta filtresi
            m = (g.Xm > -3) & (g.Xm < Lm + 3) & (g.Ym > -3) & (g.Ym < Wm + 3)
            g = g[m]
            if len(g) < 2:
                continue
            w = reliability(g.Xm.values, g.Ym.values, camxy)
            tracklets.append(dict(
                src=src, tid=int(tid), camxy=camxy,
                t=g.tg.values, X=g.Xm.values, Y=g.Ym.values, w=w,
                t0=float(g.tg.min()), t1=float(g.tg.max())))
    return tracklets, (Lm, Wm), (cam1_xy, cam2_xy)


def _coreg_pairs(tracklets, dims, dt=0.12, gate=1.5, band=(0.25, 0.75)):
    """Ortak zaman-binlerinde cam1 vs cam2 eslesen mid-saha ciftleri topla."""
    from collections import defaultdict
    Lm, Wm = dims
    b1 = defaultdict(list); b2 = defaultdict(list)
    for tr in tracklets:
        tgt = b1 if tr["src"] == "cam1" else b2
        for t, x, y in zip(tr["t"], tr["X"], tr["Y"]):
            tgt[round(t / dt)].append((x, y))
    A = []; B = []
    for k in set(b1) & set(b2):
        P = np.array(b1[k]); Q = np.array(b2[k])
        D = np.linalg.norm(P[:, None] - Q[None], axis=2)
        ri, ci = linear_sum_assignment(np.minimum(D, gate * 4))
        for r, c in zip(ri, ci):
            if D[r, c] <= gate:
                mx = (P[r, 0] + Q[c, 0]) / 2
                if band[0] * Lm < mx < band[1] * Lm:   # sadece mid-saha (iki kam guvenilir)
                    A.append(P[r]); B.append(Q[c])
    return np.array(A), np.array(B)


def coregister(tracklets, dims, verbose=True):
    """flip+sync sonrasi ARTIK rigid kaymayi kaldir (cam1 -> cam2 cercevesi).
    Robust: medyan-translation + capraz-dogrulanmis opsiyonel rotasyon. (R,t) doner."""
    A, B = _coreg_pairs(tracklets, dims, gate=1.5)
    if len(A) < 30:
        if verbose: print(f"[coreg] yetersiz cift ({len(A)}) -> duzeltme YOK")
        return np.eye(2), np.zeros(2), dict(n=len(A), applied=False)
    # 1) robust translation (komponent medyan), bir kez sik gate ile yenile
    t = np.median(B - A, axis=0)
    A2, B2 = _coreg_pairs(tracklets, dims, gate=1.2)
    if len(A2) >= 30:
        t = np.median(B2 - (A2), axis=0)
        A, B = A2, B2
    pre = np.median(np.linalg.norm(A - B, axis=1))
    post_t = np.median(np.linalg.norm(A + t - B, axis=1))
    # 2) opsiyonel rotasyon: capraz-dogrulama (Kabsch train/test)
    n = len(A); idx = np.arange(n)
    tr_m = idx % 5 != 0; te_m = ~tr_m
    At, Bt = A[tr_m], B[tr_m]
    ca, cb = At.mean(0), Bt.mean(0)
    H = (At - ca).T @ (Bt - cb)
    U, S, Vt = np.linalg.svd(H)
    dsign = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, dsign]) @ U.T
    tR = cb - R @ ca
    ang = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    # held-out RMS: rotasyon mu sadece-translation mi?
    te_t = np.sqrt((np.linalg.norm(A[te_m] + t - B[te_m], axis=1) ** 2).mean())
    te_R = np.sqrt((np.linalg.norm((R @ A[te_m].T).T + tR - B[te_m], axis=1) ** 2).mean())
    use_rot = (te_R < te_t - 0.02) and (abs(ang) < 5.0)
    if use_rot:
        Rf, tf = R, tR; post = np.median(np.linalg.norm((Rf @ A.T).T + tf - B, axis=1))
    else:
        Rf, tf = np.eye(2), t; post = post_t
    if verbose:
        print(f"[coreg] cift={n} | once medyan={pre:.3f}m | translation={t.round(3)} "
              f"-> {post_t:.3f}m | rot {ang:+.2f}deg held-out(t={te_t:.3f} R={te_R:.3f}) "
              f"-> {'ROTASYON+t' if use_rot else 'sadece-t'} | sonra medyan={post:.3f}m")
    return Rf, tf, dict(n=n, applied=True, pre_m=float(pre), post_m=float(post),
                        angle_deg=float(ang) if use_rot else 0.0, t=tf.tolist(), use_rot=bool(use_rot))


def apply_coreg(tracklets, R, t):
    """cam1 tracklet'lerine (R,t) uygula + agirligi yeniden hesapla."""
    for tr in tracklets:
        if tr["src"] != "cam1":
            continue
        P = np.c_[tr["X"], tr["Y"]]
        P2 = (R @ P.T).T + t
        tr["X"], tr["Y"] = P2[:, 0], P2[:, 1]
        cam = np.array(tr["camxy"]); cam2 = R @ cam + t
        tr["camxy"] = (float(cam2[0]), float(cam2[1]))
        tr["w"] = reliability(tr["X"], tr["Y"], tr["camxy"])
        tr["t0"], tr["t1"] = float(tr["t"].min()), float(tr["t"].max())
    return tracklets


def interp_at(tr, ts):
    """tracklet'i verilen zaman ornek noktalarinda interpolasyon (kapsam disi=nan)."""
    X = np.interp(ts, tr["t"], tr["X"], left=np.nan, right=np.nan)
    Y = np.interp(ts, tr["t"], tr["Y"], left=np.nan, right=np.nan)
    return X, Y


def overlap_dist(a, b, dt=0.08):
    """iki tracklet zaman-ortusmesinde medyan saha mesafesi (yoksa None)."""
    lo = max(a["t0"], b["t0"]); hi = min(a["t1"], b["t1"])
    if hi - lo < 0.4:
        return None, 0.0
    ts = np.arange(lo, hi, dt)
    ax, ay = interp_at(a, ts); bx, by = interp_at(b, ts)
    d = np.hypot(ax - bx, ay - by)
    good = np.isfinite(d)
    if good.sum() < 5:
        return None, 0.0
    return float(np.median(d[good])), float(hi - lo)


class DSU:
    def __init__(s, n): s.p = list(range(n))
    def find(s, x):
        while s.p[x] != x:
            s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def union(s, a, b): s.p[s.find(a)] = s.find(b)


def fuse(tracklets, dims, D_OVERLAP=2.5, MAXGAP=8.0, GAP_GATE=3.5, SPRINT=8.0):
    """Tracklet grafi -> birlesik oyuncu bilesenleri (cannot-link saygili)."""
    n = len(tracklets)
    dsu = DSU(n)
    # bilesen basina (kamera -> [araliklar]) cannot-link kontrolu icin
    comp_intervals = [{tr["src"]: [(tr["t0"], tr["t1"])]} for tr in tracklets]

    def can_merge(i, j):
        ci, cj = dsu.find(i), dsu.find(j)
        if ci == cj:
            return False
        A, B = comp_intervals[ci], comp_intervals[cj]
        for src in set(A) & set(B):
            for (a0, a1) in A[src]:
                for (b0, b1) in B[src]:
                    if min(a1, b1) - max(a0, b0) > 0.3:   # ayni kamera, zaman ortusmesi -> YASAK
                        return False
        return True

    def do_merge(i, j):
        ci, cj = dsu.find(i), dsu.find(j)
        for src, ivs in comp_intervals[cj].items():
            comp_intervals[ci].setdefault(src, []).extend(ivs)
        dsu.union(cj, ci)
        comp_intervals[dsu.find(i)] = comp_intervals[ci]

    # --- 1) KAMERALAR-ARASI ORTUSME MUST-link (wall-breaker) ---
    cross = []
    for i in range(n):
        for j in range(i + 1, n):
            if tracklets[i]["src"] == tracklets[j]["src"]:
                continue
            md, dur = overlap_dist(tracklets[i], tracklets[j])
            if md is not None and md <= D_OVERLAP:
                cross.append((md, i, j))
    cross.sort()                       # en yakin once
    n_cross = 0
    for md, i, j in cross:
        if can_merge(i, j):
            do_merge(i, j); n_cross += 1

    # --- 2) DEVAM (gap) link: zaman-disjoint + hareket-tutarli ---
    order = sorted(range(n), key=lambda k: tracklets[k]["t0"])
    cont = []
    for a in range(n):
        ta = tracklets[a]
        for b in range(n):
            if a == b:
                continue
            tb = tracklets[b]
            gap = tb["t0"] - ta["t1"]
            if gap < 0.0 or gap > MAXGAP:
                continue
            # a-bitis hizi ile b-baslangici ongor
            ka = max(1, np.searchsorted(ta["t"], ta["t1"] - 0.5))
            if len(ta["t"]) - ka < 1: ka = len(ta["t"]) - 2
            dtv = ta["t"][-1] - ta["t"][ka]
            vel = np.array([0.0, 0.0])
            if dtv > 0.05:
                vel = np.array([ta["X"][-1] - ta["X"][ka], ta["Y"][-1] - ta["Y"][ka]]) / dtv
            pred = np.array([ta["X"][-1], ta["Y"][-1]]) + vel * gap
            d = np.hypot(pred[0] - tb["X"][0], pred[1] - tb["Y"][0])
            raw = np.hypot(ta["X"][-1] - tb["X"][0], ta["Y"][-1] - tb["Y"][0])
            if raw / max(gap, 0.04) > SPRINT:
                continue
            if d <= GAP_GATE:
                cont.append((d + 0.3 * gap, a, b))
    cont.sort()
    n_cont = 0
    for c, a, b in cont:
        if can_merge(a, b):
            do_merge(a, b); n_cont += 1

    # --- bilesenler ---
    comps = {}
    for k in range(n):
        comps.setdefault(dsu.find(k), []).append(k)
    return comps, dict(n_cross=n_cross, n_cont=n_cont)


def player_tracks(tracklets, comps, dt=0.08, SPRINT=8.0):
    """Her birlesik oyuncu icin: guven-agirlikli fuzyon + duzgunlestirme + kosu mesafesi."""
    rows = []; summ = []
    for pid, members in comps.items():
        t0 = min(tracklets[k]["t0"] for k in members)
        t1 = max(tracklets[k]["t1"] for k in members)
        if t1 - t0 < 1.0:
            continue
        ts = np.arange(t0, t1 + dt, dt)
        accX = np.zeros(len(ts)); accY = np.zeros(len(ts)); accW = np.zeros(len(ts))
        ncam = np.zeros(len(ts))
        for k in members:
            tr = tracklets[k]
            X, Y = interp_at(tr, ts)
            wv = np.interp(ts, tr["t"], tr["w"], left=0, right=0)
            ok = np.isfinite(X)
            accX[ok] += (X * wv)[ok]; accY[ok] += (Y * wv)[ok]; accW[ok] += wv[ok]
            ncam[ok] += 1
        cov = accW > 0
        if cov.sum() < 8:
            continue
        Xf = accX / np.maximum(accW, 1e-9); Yf = accY / np.maximum(accW, 1e-9)
        Xf[~cov] = np.nan; Yf[~cov] = np.nan
        # kapsanan en uzun surekli sureyi al, kisa bosluklari kopru ile doldur (<=0.6s)
        idx = np.where(cov)[0]
        Xi = pd.Series(Xf).interpolate(limit=int(0.6/dt)).values
        Yi = pd.Series(Yf).interpolate(limit=int(0.6/dt)).values
        m = np.isfinite(Xi) & np.isfinite(Yi)
        if m.sum() < 8:
            continue
        # Savitzky-Golay benzeri: kisa pencere medyan + yumusatma
        xs = pd.Series(Xi[m]).rolling(7, center=True, min_periods=1).median().values
        ys = pd.Series(Yi[m]).rolling(7, center=True, min_periods=1).median().values
        tm = ts[m]
        d = np.hypot(np.diff(xs), np.diff(ys)); dts = np.diff(tm)
        v = d / np.maximum(dts, 1e-3)
        dist = d[v <= SPRINT].sum()
        dur = tm[-1] - tm[0]
        nsrc = len(set(tracklets[k]["src"] for k in members))
        summ.append(dict(pid=int(pid), dur_s=float(dur), dist_m=float(dist),
                         m_per_min=float(dist/(dur/60)) if dur > 0 else 0.0,
                         n_tracklets=len(members), n_cameras=nsrc,
                         t0=float(tm[0]), t1=float(tm[-1])))
        for tt, xx, yy in zip(tm, xs, ys):
            rows.append(dict(pid=int(pid), t=float(tt), X=float(xx), Y=float(yy)))
    return pd.DataFrame(rows), pd.DataFrame(summ).sort_values("dur_s", ascending=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cam1_tracks"); ap.add_argument("cam2_tracks")
    ap.add_argument("--config", default="calib/fusion_config.json")
    ap.add_argument("--out", default="scratchpad/fused.parquet")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    c1 = pd.read_parquet(a.cam1_tracks); c2 = pd.read_parquet(a.cam2_tracks)
    print(f"cam1 tracks: {len(c1)} satir {c1.tid.nunique()} tid  "
          f"t[{c1.t_sec.min():.0f},{c1.t_sec.max():.0f}]")
    print(f"cam2 tracks: {len(c2)} satir {c2.tid.nunique()} tid  "
          f"t[{c2.t_sec.min():.0f},{c2.t_sec.max():.0f}]")
    tracklets, dims, camxy = build_tracklets(c1, c2, cfg)
    print(f"ortak-cerceve tracklet: {len(tracklets)} "
          f"(cam1={sum(t['src']=='cam1' for t in tracklets)} "
          f"cam2={sum(t['src']=='cam2' for t in tracklets)})")
    R, t, creg = coregister(tracklets, dims)
    if creg["applied"]:
        apply_coreg(tracklets, R, t)
    comps, stats = fuse(tracklets, dims)
    print(f"FUZYON: kameralar-arasi must-link={stats['n_cross']}  devam-link={stats['n_cont']}")
    big = [c for c in comps.values() if len(c) >= 1]
    print(f"birlesik bilesen: {len(comps)}")
    df, summ = player_tracks(tracklets, comps)
    if len(summ) == 0:
        print("(yeterli kapsamli birlesik oyuncu yok)"); return
    df.to_parquet(a.out)
    summ.to_parquet(a.out.replace(".parquet", "_summary.parquet"))
    meta = dict(coregistration=creg, fuse_links=stats,
                n_tracklets=len(tracklets), n_players=int(len(summ)),
                cam1_tracks=a.cam1_tracks, cam2_tracks=a.cam2_tracks,
                offset_s=cfg["time_sync"]["offset_cam1_to_cam2_s"])
    json.dump(meta, open(a.out.replace(".parquet", "_meta.json"), "w"), indent=2)
    dual = summ[summ.n_cameras == 2]
    print(f"\nbirlesik oyuncu (>=1s): {len(summ)}  iki-kamerali: {len(dual)}")
    print(f">=60s: {(summ.dur_s>=60).sum()}  >=180s: {(summ.dur_s>=180).sum()}  "
          f">=360s: {(summ.dur_s>=360).sum()}")
    top = summ.head(12)
    print("\nen uzun 12 birlesik oyuncu:")
    for _, r in top.iterrows():
        print(f"  pid={r.pid:>3} sure={r.dur_s:6.1f}s mesafe={r.dist_m:6.0f}m "
              f"{r.m_per_min:5.0f}m/dk  parca={r.n_tracklets:>3} kam={r.n_cameras}")
    print(f"\nyazildi: {a.out} + _summary")


if __name__ == "__main__":
    main()
