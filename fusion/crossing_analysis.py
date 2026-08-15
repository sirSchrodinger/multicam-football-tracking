#!/usr/bin/env python3
"""Çaprazlaşma ayrımı — füzyonun re-ID duvarını kırma kazancını ÖLÇER (iddia değil).

Tek-kamera duvarı: cam2'de iki oyuncu ~1m içine girip ayrıldığında (çaprazlaşma),
ByteTrack hangisinin hangisi olduğunu karıştırabilir = kimlik-takası riski. cam1
zıt uçtan baktığı için cam2'nin derinlik-ekseninde çakışan ikiliyi AYRI görür.

Ölçüm: cam2 çaprazlaşma olaylarını bul. Her biri için cam1 o pencereyi SÜREKLİ
kapsayan bir parça sağlıyor mu (= kimliği çaprazlaşma boyunca sabitler)?
Çözülebilir-oran = M/N. Dürüst: bu cam1'in *çözebileceği* üst-sınır (kapsama
gerekli-koşul); gerçek takas-önleme kapsama + ayrılabilirlik ister.
"""
import json, argparse, sys, os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fuse_engine as fe


def _interp(tr, t):
    if t < tr["t0"] or t > tr["t1"]:
        return None
    return np.array([np.interp(t, tr["t"], tr["X"]), np.interp(t, tr["t"], tr["Y"])])


def detect_crossings(cam2, d_close=1.2, d_sep=2.5, dt=0.1, win=0.8):
    """cam2 tracklet ciftlerinde: yakinlasip(d_close) sonra ayrilan(d_sep) anlar."""
    cross = []
    n = len(cam2)
    for a in range(n):
        ta = cam2[a]
        for b in range(a + 1, n):
            tb = cam2[b]
            lo = max(ta["t0"], tb["t0"]); hi = min(ta["t1"], tb["t1"])
            if hi - lo < 2 * win:
                continue
            ts = np.arange(lo, hi, dt)
            ax = np.interp(ts, ta["t"], ta["X"]); ay = np.interp(ts, ta["t"], ta["Y"])
            bx = np.interp(ts, tb["t"], tb["X"]); by = np.interp(ts, tb["t"], tb["Y"])
            d = np.hypot(ax - bx, ay - by)
            k = int(np.argmin(d))
            if d[k] > d_close:
                continue
            # cevre: oncesi VE sonrasi ayrilmis mi (win kadar)
            kb = ts[k] - win; ka = ts[k] + win
            pre = d[(ts < kb)]; post = d[(ts > ka)]
            if len(pre) and len(post) and pre.max() > d_sep and post.max() > d_sep:
                p = np.array([(ax[k] + bx[k]) / 2, (ay[k] + by[k]) / 2])
                cross.append(dict(t=float(ts[k]), p=p, dmin=float(d[k]),
                                  i=ta["tid"], j=tb["tid"]))
    # zaman+konum yakin olaylari tekille (ayni carpisma iki kez sayilmasin)
    cross.sort(key=lambda c: c["t"])
    uniq = []
    for c in cross:
        if uniq and abs(c["t"] - uniq[-1]["t"]) < 1.0 and \
           np.linalg.norm(c["p"] - uniq[-1]["p"]) < 2.5:
            continue
        uniq.append(c)
    return uniq


def cam1_resolves(cam1, ev, win=0.8, gate=4.0):
    """cam1, carpisma penceresini SUREKLI kapsayan ve bolgeye yakin parca saglar mi?"""
    t = ev["t"]; p = ev["p"]
    near = 0
    for tr in cam1:
        if tr["t0"] <= t - win and tr["t1"] >= t + win:
            x = _interp(tr, t)
            if x is not None and np.linalg.norm(x - p) <= gate:
                near += 1
    return near  # >=1: en az bir crosser cam1'de sabit; >=2: ikisi de


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cam1_tracks"); ap.add_argument("cam2_tracks")
    ap.add_argument("--config", default="calib/fusion_config.json")
    ap.add_argument("--out", default="scratchpad/crossings.json")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    c1 = pd.read_parquet(a.cam1_tracks); c2 = pd.read_parquet(a.cam2_tracks)
    tr, dims, _ = fe.build_tracklets(c1, c2, cfg)
    R, t, creg = fe.coregister(tr, dims, verbose=False)
    if creg["applied"]:
        fe.apply_coreg(tr, R, t)
    cam1 = [x for x in tr if x["src"] == "cam1"]
    cam2 = [x for x in tr if x["src"] == "cam2"]
    print(f"tracklet: cam1={len(cam1)} cam2={len(cam2)}")
    ev = detect_crossings(cam2)
    print(f"cam2 capraz-lasma olayi: {len(ev)}")
    if not ev:
        print("(carpisma yok)"); return
    res = [cam1_resolves(cam1, e) for e in ev]
    res = np.array(res)
    one = int((res >= 1).sum()); two = int((res >= 2).sum())
    # cam1'in zaman-kapsamasi olan olaylar (cam1 o anda kayittaysa)
    t_lo = min(x["t0"] for x in cam1) if cam1 else 0
    t_hi = max(x["t1"] for x in cam1) if cam1 else 0
    in_cov = [e for e in ev if t_lo <= e["t"] <= t_hi]
    print(f"cam1 zaman-kapsamindaki carpisma: {len(in_cov)}")
    print(f"cam1 EN AZ BIR crosser'i sabitler: {one}/{len(ev)} ({100*one/len(ev):.0f}%)")
    print(f"cam1 IKISINI de sabitler: {two}/{len(ev)} ({100*two/len(ev):.0f}%)")
    json.dump(dict(n_crossings=len(ev), in_cam1_coverage=len(in_cov),
                   resolved_ge1=one, resolved_ge2=two,
                   frac_ge1=float(one/len(ev)), frac_ge2=float(two/len(ev))),
              open(a.out, "w"), indent=2)
    print("yazildi:", a.out)


if __name__ == "__main__":
    main()
