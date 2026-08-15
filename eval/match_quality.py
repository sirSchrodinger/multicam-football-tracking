#!/usr/bin/env python3
"""eval/match_quality.py — ÖZ-DEĞERLENDİRME / FARKINDALIK sistemi.

Amaç (Alperen): her saha / her maç / her görüntü işlendikten sonra çıktının
kalitesini OTOMATİK + DÜRÜST ölç; "izlenebilir mi, değilse neden, neyi manuel
düzeltmek gerek" diye söyle. Bunu otomatize edeceğiz -> manuel-düzeltme hedefleri
buradan çıkar. Her tesise/maça yeniden uygulanır (camera-agnostik).

Eksenler (her biri 0-100 + dürüst not):
  CALIB     : reproj median_px, per-zone, saha-DIŞI pozisyon %, kapsama.
  DETECTION : kare-başı tespit (gözlenen) / beklenen(14) -> recall tahmini.
  IDENTITY  : n_id vs 14, fragman, fill-rate, TELEPORT (ID-takas proxy), concurrency.
  MOTION    : kare-kare metre adım, fiziksel-dışı (>0.6 m/frame) %.
Sonuç: overall_score + LIMITING_FACTOR + watchable(bool,neden) + manual_fix listesi.

Tasarım: GT gerektirmez (proxy'ler); recall_eval GT'si varsa daha kesin.
Lisans: pandas+numpy. GPU yok.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

EXPECTED = 14
MAX_MPS = 12.0   # fiziksel-dışı hız eşiği


def _score(val, good, bad):
    """val good->100, bad->0 (lineer, clamp). good>bad ya da good<bad olabilir."""
    if good == bad:
        return 100.0
    t = (val - bad) / (good - bad)
    return float(np.clip(t, 0, 1) * 100)


def assess_match(tracks_path=None, state_path=None, calib_path=None,
                 expected=EXPECTED, fps=None):
    rep = {"axes": {}, "manual_fix": [], "limiting_factor": None}

    # ---- CALIB ----
    calib = json.loads(Path(calib_path).read_text()) if calib_path else {}
    qa = calib.get("qa") or {}
    L = (calib.get("pitch_dims_m") or {}).get("L", 34.0)
    W = (calib.get("pitch_dims_m") or {}).get("W", 18.0)
    med_px = float(qa.get("median_px", np.nan))
    calib_score = _score(med_px, 6.0, 25.0) if not np.isnan(med_px) else 50.0

    # ---- yükle ----
    tr = pd.read_parquet(tracks_path) if tracks_path else None
    st = pd.read_parquet(state_path) if state_path else None
    if fps is None and tr is not None and tr.t_sec.max() > 0:
        fps = float(tr.frame.max() / tr.t_sec.max())
    fps = fps or 24.86

    # ---- DETECTION / RECALL (raw stitched tracks: kare-başı in_pitch tespit) ----
    det_score = obs_med = est_recall = np.nan
    if tr is not None:
        idc = "player_id" if "player_id" in tr.columns else "tid"
        perf = tr.groupby("frame")[idc].nunique()
        obs_med = float(perf.median())
        est_recall = obs_med / expected
        det_score = _score(est_recall, 0.95, 0.50)
        rep["axes"]["detection"] = dict(
            score=round(det_score, 0), median_per_frame=round(obs_med, 1),
            est_recall_pct=round(100 * est_recall, 0),
            note=f"kare başı medyan {obs_med:.0f}/{expected} tespit (recall ~%{100*est_recall:.0f})")
        if est_recall < 0.85:
            rep["manual_fix"].append("recall düşük: far-band tiling/CLAHE/self-training")

    # ---- IDENTITY ----
    id_score = np.nan
    if tr is not None:
        idc = "player_id" if "player_id" in tr.columns else "tid"
        maxf = tr.frame.max()
        sp = tr.groupby(idc).agg(n=("frame", "count"), f0=("frame", "min"), f1=("frame", "max"))
        sp["span"] = sp.f1 - sp.f0 + 1
        sp["fill"] = sp.n / sp.span
        n_ids = len(sp)
        n_core = int((sp.span > 0.55 * maxf).sum())
        frag_ratio = n_ids / expected
        fill_med = float(sp[sp.span > 0.3 * maxf].fill.median())
        # TELEPORT (ID-takas proxy): aynı id ardışık-tespit metre-adımı > MAX
        tel = np.nan
        if {"pitch_x", "pitch_y"}.issubset(tr.columns):
            t2 = tr.dropna(subset=["pitch_x", "pitch_y"]).sort_values([idc, "frame"])
            dx = t2.groupby(idc).pitch_x.diff(); dy = t2.groupby(idc).pitch_y.diff()
            dfr = t2.groupby(idc).frame.diff().clip(lower=1)
            step = np.hypot(dx, dy) / dfr * fps  # m/s
            tel = float(np.nanmean(step > MAX_MPS))
        # skor: fragman(14->100, 33->0) + fill(0.8->100,0.4->0) + teleport(0->100,0.1->0)
        frag_s = _score(n_ids, expected, expected * 2.5)
        fill_s = _score(fill_med, 0.8, 0.4)
        tel_s = _score(tel, 0.0, 0.08) if not np.isnan(tel) else 60.0
        id_score = float(np.mean([frag_s, fill_s, tel_s]))
        rep["axes"]["identity"] = dict(
            score=round(id_score, 0), n_ids=n_ids, n_core=n_core,
            fragment_ratio=round(frag_ratio, 2), fill_rate=round(fill_med, 2),
            teleport_frac=None if np.isnan(tel) else round(tel, 3),
            note=f"{n_ids} kimlik (gerçek ~{expected}); fill %{100*fill_med:.0f}; "
                 f"teleport %{100*tel:.1f}" if not np.isnan(tel) else f"{n_ids} kimlik")
        if frag_ratio > 1.4:
            rep["manual_fix"].append(f"kimlik fragman: {n_ids}->{expected} (roster-çapalı stitch / pencere-stitch)")
        if not np.isnan(tel) and tel > 0.03:
            rep["manual_fix"].append("ID-takas (teleport) yüksek: crossing-aware re-assoc")

    # ---- MOTION + saha-DIŞI (state varsa) ----
    mot_score = pos_score = np.nan
    src = st if st is not None else tr
    if src is not None and {"x", "y"}.issubset(src.columns):
        x, y = src.x.values, src.y.values
    elif src is not None and {"pitch_x", "pitch_y"}.issubset(src.columns):
        x, y = src.pitch_x.values, src.pitch_y.values
    else:
        x = y = None
    if x is not None:
        off = np.mean((x < -0.5) | (x > L + 0.5) | (y < -0.5) | (y > W + 0.5))
        pos_score = _score(off, 0.0, 0.15)
        idc = "player_id" if (src is not None and "player_id" in src.columns) else "tid"
        s2 = src.assign(_x=x, _y=y).sort_values([idc, "frame"]) if idc in src.columns else None
        if s2 is not None:
            dx = s2.groupby(idc)._x.diff(); dy = s2.groupby(idc)._y.diff()
            dfr = s2.groupby(idc).frame.diff().clip(lower=1)
            step = np.hypot(dx, dy) / dfr
            imposs = float(np.nanmean(step > MAX_MPS / fps))
            mot_score = _score(imposs, 0.0, 0.05)
            rep["axes"]["motion"] = dict(
                score=round(mot_score, 0), off_field_pct=round(100 * off, 1),
                impossible_step_pct=round(100 * imposs, 1),
                note=f"saha-dışı %{100*off:.0f}; fiziksel-dışı adım %{100*imposs:.0f}")
            if off > 0.05:
                rep["manual_fix"].append("pozisyon saha-dışı sızıyor: kalibrasyon/foot-nokta")

    rep["axes"]["calibration"] = dict(
        score=round(calib_score, 0), median_px=None if np.isnan(med_px) else round(med_px, 1),
        note=f"reproj median {med_px:.1f}px" if not np.isnan(med_px) else "calib QA yok")

    # ---- OVERALL + LIMITING ----
    scores = {k: v["score"] for k, v in rep["axes"].items() if v.get("score") is not None}
    rep["overall_score"] = round(float(np.mean(list(scores.values()))), 0) if scores else None
    if scores:
        lim = min(scores, key=scores.get)
        rep["limiting_factor"] = lim
        rep["watchable"] = bool(rep["overall_score"] >= 70 and scores.get(lim, 0) >= 55)
        rep["verdict"] = (f"İZLENEBİLİR" if rep["watchable"]
                          else f"İZLENEBİLİR DEĞİL — sınırlayan: {lim} ({scores[lim]:.0f}/100)")
    return rep


def print_report(rep):
    print("=" * 56)
    print(f"  MAÇ KALİTE KARNESİ — overall {rep.get('overall_score')}/100")
    print(f"  {rep.get('verdict','')}")
    print("=" * 56)
    for ax, d in rep["axes"].items():
        print(f"  [{ax:11}] {d['score']:>3.0f}/100  {d.get('note','')}")
    if rep["manual_fix"]:
        print("\n  MANUEL-DÜZELTME HEDEFLERİ (otomatize edilecek):")
        for m in rep["manual_fix"]:
            print(f"    - {m}")


if __name__ == "__main__":
    import sys
    d = "stats_out/cankaya_cam2_fullgame/"
    rep = assess_match(
        tracks_path=sys.argv[1] if len(sys.argv) > 1 else d + "tracks_cankaya_cam2_fullgame_presplit_player_bridged.parquet",
        state_path=sys.argv[2] if len(sys.argv) > 2 else d + "state14.parquet",
        calib_path=sys.argv[3] if len(sys.argv) > 3 else "calib/cankaya_cam2_FINAL.json")
    print_report(rep)
    Path(d + "quality_scorecard.json").write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
