#!/usr/bin/env python3
"""gk_anchor — KALECI capa (derin-rezidans + advance-return) + pencere-track konsolidasyonu.

CEPHE 2 (Alperen 29 Haz): onceki KALECI tanimi YANLISTI. Iki kok hata:
  (a) "anlik en-arka oyuncu" / min-max-X mantigi GK'yi saha-oyuncusuyla karistiriyordu;
  (b) UZAK-uc GK'si yanlis kumeyi (x~25, kale-agzinin YANI cy~12.5) etiketliyordu cunku
      far-band reproj-gurultusu med_spd'yi sisirip gercek GK'yi (cy~9, cizgiye <7m)
      gk_speed_max kapisiyla ELIYORDU.

Alperen'in dogru tanimi: "kaleci = arada one cikip tekrar KALEYE donen oyuncu". Bu modul:

  1) detect_goalkeepers(): Her kale-ucu icin KALECI = (i) DERIN-REZIDANS skoru = kale
     CIZGISINE <gk_depth (~6-8m) VE kale-AGZINA (Y=W/2) <mouth_hw (~4m) gecirilen sure orani
     (kose/kanat oyuncusu mouth_hw ile elenir); (ii) ADVANCE-RETURN = X'in ara sira cizgiden
     uzaklasip GERI donmesi (statik defansi ayirir, RAPORLANIR). Aday-siralama BIRINCIL
     derin-rezidans (min/max-X DEGIL). Far-band aday icin hiz-kapisi GEVSER. Her uca EN FAZLA
     BIR zaman-zinciri (frame-disjoint fragmentler). GK saha-oyuncusuyla ASLA birlesmez.
     Her uc icin per-uc GUVEN etiketi (HIGH/MODERATE/LOW) DURUST raporlanir.

  2) consolidate_window(): track_stitch'in over-merge-GUVENLI cekirdegini (_DSU cannot-link,
     _clique_floor sert alt-sinir, forward-only LSAP path-cover, VMAX motion-gate) METRIK
     saha-uzayinda bot_raw_tracks'a uygular. GK zincirleri ON-BIRLESTIRILIR (handoff-ortusmesi
     BUDANIR -> frame-disjoint), GK<->saha YASAK. Sonuc: tid sayisi duser, over-merge=0.

  3) bridge_short_gaps(): konsolide player_id ICINDE <=gap_bridge_s (~0.5s, replay2d
     bosluk-kapisiyla UYUMLU) gozlem bosluklarini lineer interp ile doldurur (spawn azalt).
     UZUN bosluk KOPRULENMEZ -> sahte cross-field iz yok (replay2d'nin onledigi bug).

DURUSTLUK: 14'e ZORLAMA yok; clique_floor sert alt-sinir; eszamanli co-located AYRI oyuncu
ASLA birlesmez (frame-disjoint cannot-link); GK handoff-ortusmesi budanir ve RAPORLANIR;
far-uc GK'si SEYREK gozlemli -> MODERATE guven (gizlenmez); aktif-sayim VARYANSI buyuk olcude
recall-bagimli (CEPHE 3), kimlik degil -> over-claim YOK; mutlak metre/hiz iddiasi yok (~+-13%).

Lisans: numpy + scipy.optimize + pandas + cv2 (replay2d.project icin) — hepsi BSD. GPU yok.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from stats.replay2d import load_calib, project          # metrik projeksiyon
from detect.track_stitch import _DSU, _clique_floor      # over-merge-guvenli cekirdek (REUSE)

_BIG = 1.0e9


# ============================================================ projeksiyon ===
def project_df(df: pd.DataFrame, cal: dict) -> pd.DataFrame:
    """foot_x/foot_y -> metrik X (boy), Y (en). df'e X,Y ekler (kopya doner)."""
    P = project(cal, df[["foot_x", "foot_y"]].values)
    out = df.copy()
    out["X"] = P[:, 0]
    out["Y"] = P[:, 1]
    return out


def far_threshold(df_xy: pd.DataFrame) -> float:
    """Piksel foot_y'nin alt-uctebiri = uzak-uc esigi (altinda=kameradan uzak)."""
    return float(np.quantile(df_xy["foot_y"].values, 1 / 3))


# ====================================================== advance-return =======
def _return_events(d: np.ndarray, deep: float, out: float) -> int:
    """d = cizgiye uzaklik dizisi (m). "Donus" = cizgiye <deep gel, sonra >out cik,
    sonra TEKRAR <deep don. Saf-resident (hic >out cikmaz) -> 0. Patrol eden GK -> >=1.
    'Anlik en-arka' DEGIL; oyuncunun KALEYE donus desenini sayar (Alperen tanimi)."""
    events = 0
    armed = False
    for v in d:
        if v < deep:
            if armed:
                events += 1
                armed = False
        elif v > out:
            armed = True
    return events


# ====================================================== per-tid ozellikler ===
def tid_features(df_xy: pd.DataFrame, fps: float, L: float, W: float,
                 gk_depth: float = 7.0, mouth_hw: float = 4.0,
                 adv_deep: float = 3.0, adv_out: float = 6.0,
                 far_thr: float | None = None) -> pd.DataFrame:
    """Her tid (>=0) icin per-uc DERIN-REZIDANS + ADVANCE-RETURN + konum/hiz/omur.

    res_endA/res_endL = (dist-to-line < gk_depth) VE (|Y - W/2| < mouth_hw) frame-orani.
      mouth_hw=4 -> kale-agzina merkezli; kanat/kose oyuncusu (cy uctan) ELENIR.
    ret_endA/ret_endL = advance-return donus sayisi (adv_deep/adv_out esikleriyle).
    far_frac = uzak-bandda (foot_y < far_thr) gozlem orani (far-band hiz-gevsetmesi icin).
    med_spd = komsu-frame (dt<0.3s) medyan metrik hiz (m/s); far-band'da reproj-gurultusu sisik.
    """
    gy = W / 2.0
    if far_thr is None:
        far_thr = far_threshold(df_xy)
    rows = []
    for tid, g in df_xy.groupby("tid"):
        if int(tid) < 0:
            continue
        g = g.sort_values("frame")
        X = g["X"].to_numpy(float); Y = g["Y"].to_numpy(float); fr = g["frame"].to_numpy(int)
        fy = g["foot_y"].to_numpy(float)
        dt = np.diff(fr) / fps
        dd = np.hypot(np.diff(X), np.diff(Y))
        m = (dt > 0) & (dt < 0.3)
        med_spd = float(np.median(dd[m] / dt[m])) if m.any() else np.nan
        ymouth = np.abs(Y - gy) < mouth_hw
        dA = np.abs(X - 0.0); dL = np.abs(X - L)
        rows.append(dict(
            tid=int(tid), n=len(g), f0=int(fr.min()), f1=int(fr.max()),
            cx=float(np.median(X)), cy=float(np.median(Y)),
            sx=float(np.std(X)), sy=float(np.std(Y)), med_spd=med_spd,
            far_frac=float(np.mean(fy < far_thr)),
            res_endA=float(np.mean((dA < gk_depth) & ymouth)),
            res_endL=float(np.mean((dL < gk_depth) & ymouth)),
            ret_endA=int(_return_events(dA, adv_deep, adv_out)),
            ret_endL=int(_return_events(dL, adv_deep, adv_out)),
            xmin=float(X.min()), xmax=float(X.max()),
        ))
    return pd.DataFrame(rows).sort_values("cx").reset_index(drop=True)


# ============================================================ GK tespiti =====
def _frame_set(df_xy, tid):
    return set(df_xy.loc[df_xy["tid"] == tid, "frame"].astype(int).tolist())


def _co_distance(df_xy, a, b):
    """a,b ortak frame'lerinde medyan metrik mesafe (yoksa np.inf)."""
    ga = df_xy[df_xy["tid"] == a].set_index("frame")[["X", "Y"]]
    gb = df_xy[df_xy["tid"] == b].set_index("frame")[["X", "Y"]]
    common = ga.index.intersection(gb.index)
    if len(common) == 0:
        return np.inf, 0
    d = np.hypot(ga.loc[common, "X"].values - gb.loc[common, "X"].values,
                 ga.loc[common, "Y"].values - gb.loc[common, "Y"].values)
    return float(np.median(d)), int(len(common))


def _build_chain(ordered_tids, df_xy, feats, handoff_overlap_max, handoff_max_m):
    """Bir uctaki GK adaylarindan TEK zaman-zinciri kur (persistent GK).

    ordered_tids: ONCELIK sirali aday tid'ler (BIRINCIL: derin-rezidans skoru). Zincire
    eklenme: her zincir uyesiyle frame-ortusmesi <= handoff_overlap_max VE (ortusme>0 ise)
    ortak-frame mesafesi <= handoff_max_m. Uzun-eszamanli aday DISLANIR (farkli oyuncu);
    kisa handoff / disjoint fragment KABUL (ayni GK)."""
    chain = []
    excluded = []
    for t in ordered_tids:
        t = int(t)
        ok = True
        reason = ""
        for mm in chain:
            ov = len(_frame_set(df_xy, t) & _frame_set(df_xy, mm))
            if ov > handoff_overlap_max:
                ok = False; reason = f"long concurrent overlap {ov}f with tid {mm}"
                break
            if ov > 0:
                d, _ = _co_distance(df_xy, t, mm)
                if d > handoff_max_m:
                    ok = False; reason = f"handoff {ov}f too far {d:.2f}m vs tid {mm}"
                    break
        if ok:
            chain.append(t)
        else:
            excluded.append(dict(tid=t, reason=reason))
    chain.sort(key=lambda t: feats.loc[feats["tid"] == t, "f0"].iloc[0])
    return chain, excluded


def detect_goalkeepers(df: pd.DataFrame, cal: dict, fps: float,
                       gk_depth: float = 7.0, mouth_hw: float = 4.0,
                       res_min: float = 0.55, min_life: int = 40,
                       gk_speed_max: float = 3.0, far_relax: float = 3.0,
                       adv_deep: float = 3.0, adv_out: float = 6.0,
                       handoff_overlap_max: int = 10,
                       handoff_max_m: float = 5.0) -> dict:
    """Her kale-ucu (A: x~0, L: x~L) icin EN FAZLA BIR persistent GK zinciri.

    Aday = (derin-rezidans res_end >= res_min) VE (omur >= min_life) VE
           (med_spd <= gk_speed_max * (far_relax if far-band else 1)).
    Far-band aday (far_frac>0.5) icin hiz-kapisi GEVSER -> reproj-gurultulu uzak-uc GK'si
    YANLISLIKLA ELENMEZ (CEPHE 2 kok hatasi). Adaylar BIRINCIL derin-rezidansa gore siralanir
    (anlik min/max-X DEGIL), tek zincir kurulur, advance-return + GUVEN raporlanir.
    Doner: dict(endA, endL, features, params) — endX = dict(members, ...).
    """
    L, W = cal["L"], cal["W"]
    df_xy = project_df(df, cal)
    far_thr = far_threshold(df_xy)
    feats = tid_features(df_xy, fps, L, W, gk_depth, mouth_hw, adv_deep, adv_out, far_thr)

    out = dict(features=feats.to_dict("records"),
               params=dict(gk_depth=gk_depth, mouth_hw=mouth_hw, res_min=res_min,
                           min_life=min_life, gk_speed_max=gk_speed_max, far_relax=far_relax,
                           adv_deep=adv_deep, adv_out=adv_out,
                           handoff_overlap_max=handoff_overlap_max, handoff_max_m=handoff_max_m,
                           L=round(L, 2), W=round(W, 2), goal_y=round(W / 2, 2)))
    wf0, wf1 = int(df["frame"].min()), int(df["frame"].max())
    win = max(1, wf1 - wf0 + 1)

    for end, res_col, line_x in (("endA", "res_endA", 0.0), ("endL", "res_endL", L)):
        f = feats.copy()
        # far-band-aware hiz kapisi (uzak-uc GK'sini reproj-gurultusu elemesin)
        cap = gk_speed_max * np.where(f["far_frac"].values > 0.5, far_relax, 1.0)
        spd_ok = ~(f["med_spd"].values > cap)         # NaN-guvenli (NaN>cap -> False -> ok)
        cand = f[(f[res_col] >= res_min) & (f["n"] >= min_life) & spd_ok].copy()
        # BIRINCIL siralama: derin-rezidans (yuksek), sonra hiz (dusuk). min/max-X YOK.
        cand = cand.sort_values([res_col, "med_spd"], ascending=[False, True])
        end_tids = [int(t) for t in cand["tid"].tolist()]
        chain, excluded = _build_chain(end_tids, df_xy, feats,
                                       handoff_overlap_max, handoff_max_m)

        # zincir kapsama + far-band + advance-return (zincir-konkat dist-to-line uzerinden)
        cov_frames = set()
        for t in chain:
            cov_frames |= _frame_set(df_xy, t)
        coverage = round(len(cov_frames) / win, 3)
        chain_far = bool(np.mean([feats.loc[feats["tid"] == t, "far_frac"].iloc[0]
                                  for t in chain]) > 0.5) if chain else False
        if chain:
            sub = df_xy[df_xy["tid"].isin(chain)].sort_values("frame")
            d_line = np.abs(sub["X"].to_numpy(float) - line_x)
            ret_events = int(_return_events(d_line, adv_deep, adv_out))
            max_exc = float(d_line.max())
            res_mean = float(np.mean([feats.loc[feats["tid"] == t, res_col].iloc[0]
                                      for t in chain]))
        else:
            ret_events, max_exc, res_mean = 0, 0.0, 0.0

        # DURUST guven: yakin-uc bol gozlem -> HIGH; far-uc seyrek -> MODERATE; zayif -> LOW
        if not chain or res_mean < res_min:
            conf = "LOW"
        elif coverage >= 0.7 and res_mean >= 0.8:
            conf = "HIGH"
        elif res_mean >= 0.7 or (chain_far and coverage >= 0.3):
            conf = "MODERATE"
        else:
            conf = "LOW"

        out[end] = dict(
            members=chain, n_members=len(chain),
            candidates=end_tids, excluded=excluded,
            coverage=coverage, residence_mean=round(res_mean, 3),
            advance_return_events=ret_events, max_excursion_m=round(max_exc, 2),
            far_band=chain_far, confidence=conf,
            span=[min((int(feats.loc[feats["tid"] == t, "f0"].iloc[0]) for t in chain),
                      default=None),
                  max((int(feats.loc[feats["tid"] == t, "f1"].iloc[0]) for t in chain),
                      default=None)] if chain else None,
        )
    return out


# ============================================ ZAMAN-DEGISKEN GK ROL TIMELINE =
def gk_role_timeline(df: pd.DataFrame, cal: dict, fps: float,
                     win_s: float = 8.0, mouth_depth_m: float = 7.0,
                     mouth_halfwidth_m: float = 4.0, min_resident_frac: float = 0.45,
                     hysteresis: float = 1.3, min_presence: float = 0.12) -> dict:
    """KALECI = ZAMAN-DEGISKEN ROL. Her frame icin {'near': pid|None, 'far': pid|None}.

    Sabit-tek-pid GK (eski gk_ids) amator kaleci ROTASYONUNU (farkli kisiler sirayla kaleye
    gecer) ve KIMLIK PARCALANMASINI (re-ID yok -> tek pid 12dk tutmuyor) yakalayamaz. Bu modul
    GK'yi her kale-ucu icin pencere-bazli atar: belirli bir anda gercekten gol-agzinda DERIN
    duran (deep-residency) konsolide oyuncu. Anlik en-arka oyuncu DEGIL.

    Koordinat: origin sol-alt, X=boy[0,L], Y=en[0,W]. near kale x=0, far kale x=L; agiz y=W/2.
    Gol-agzi bolgesi (per uc): |X-gx| < mouth_depth_m VE |Y-W/2| < mouth_halfwidth_m.

    REZIDANS = GOZLEM-BASINA derin-rezidans: trailing pencere [f-win_f+1, f] icinde
      res = (agiz-frame sayisi) / (gozlenen-frame sayisi). Pencere-UZUNLUGUNA bolmek
      fragmentasyonu CEZALANDIRIR (uzak-uc recall'i dusuk -> gercek kaleci seyrek gozlenir);
      gozleme bolmek bunu duzeltir. Saf-fragment hijack'ini ONLEMEK icin PRESENCE kapisi:
      presence = gozlenen-frame / etkin-pencere >= min_presence (2-3 frame'lik hayalet elenir).

    Algoritma (her uc, her frame f):
      - Aday = (res >= min_resident_frac) VE (presence >= min_presence) VE (agiz-frame > 0).
        En yuksek res = GK adayi.
      - HISTEREZIS (flicker engelle): mevcut GK hala aday-esigini gecerken yalniz baska pid onu
        hysteresis (1.3x) katindan fazla asarsa degisir; aksi halde mevcut GK korunur.
      - Mevcut GK esigin altina duserse en iyi aday alinir (rotasyon takibi).
      - Bos/yetersiz pencere (kimse esigi gecmez): onceki GK penceresinde hala agiz-frame'i
        varsa TASINIR (kisa bosluk; win_s'e kadar), yoksa None.

    Doner: {frame:int -> {'near': pid|None, 'far': pid|None}} (replay2d frame-bazli okur).
    df: konsolide (player_id,frame,foot_x,foot_y) VEYA zaten pitch-koord (X,Y). cal: load_calib.
    """
    L, W = cal["L"], cal["W"]
    gy = W / 2.0
    win_f = max(1, int(round(win_s * fps)))

    # koord: foot pikselleri varsa cal ile YENIDEN projekte et (postprocess ile tutarli);
    # yoksa (sentetik test) hazir X,Y kullan.
    if {"foot_x", "foot_y"}.issubset(df.columns) and "K" in cal:
        d = project_df(df, cal)
    else:
        d = df.copy()
    if "player_id" in d.columns:
        d = d[d["player_id"] >= 0]
        id_col = "player_id"
    else:
        id_col = "tid"
    d = d[[id_col, "frame", "X", "Y"]].copy()
    d["frame"] = d["frame"].astype(int)
    if len(d) == 0:
        return {}
    f0, f1 = int(d["frame"].min()), int(d["frame"].max())

    # seen = her pid'in gozlendigi DISTINCT frame'ler (uc-bagimsiz; presence kapisi icin)
    seen = {int(pid): np.unique(g["frame"].to_numpy(int))
            for pid, g in d.groupby(id_col)}

    # per uc: her pid -> agizda gorulen DISTINCT frame'lerin sirali dizisi
    def _mouth_frames(gx):
        m = (np.abs(d["X"].to_numpy(float) - gx) < mouth_depth_m) & \
            (np.abs(d["Y"].to_numpy(float) - gy) < mouth_halfwidth_m)
        sub = d.loc[m, [id_col, "frame"]]
        return {int(pid): np.unique(g["frame"].to_numpy(int))
                for pid, g in sub.groupby(id_col)}

    ends = {"near": _mouth_frames(0.0), "far": _mouth_frames(L)}

    def _win_cnt(arr, lo, f):
        return int(np.searchsorted(arr, f, "right") - np.searchsorted(arr, lo, "left"))

    def _assign(prev, mf, f):
        eff = float(min(win_f, f - f0 + 1))
        lo = f - win_f + 1
        res = {}           # gozlem-basina agiz-rezidansi (siralama/esik)
        mouth_in = {}      # penceredeki agiz-frame sayisi (carry kapisi)
        for pid, fr in mf.items():
            mc = _win_cnt(fr, lo, f)
            if mc == 0:
                continue
            mouth_in[pid] = mc
            sc = _win_cnt(seen.get(pid, fr), lo, f)
            presence = sc / eff
            if sc > 0 and presence >= min_presence:
                res[pid] = mc / sc
        cand = {p: v for p, v in res.items() if v >= min_resident_frac}
        if not cand:
            # bos/yetersiz pencere -> onceki GK penceresinde hala agiz-frame'i varsa TASI
            if prev is not None and mouth_in.get(prev, 0) > 0:
                return prev
            return None
        best = max(cand, key=cand.get)
        if prev is not None and prev in cand:
            # mevcut GK hala gecerli -> yalniz histerezisi asan rakip degistirir
            if best != prev and cand[best] > hysteresis * cand[prev]:
                return best
            return prev
        return best

    timeline = {}
    prev = {"near": None, "far": None}
    for f in range(f0, f1 + 1):
        cur = {e: _assign(prev[e], ends[e], f) for e in ("near", "far")}
        timeline[f] = cur
        prev = cur
    return timeline


# ================================================ GK handoff overlap-budama ==
def trim_gk_overlaps(df: pd.DataFrame, gk: dict) -> tuple[pd.DataFrame, list]:
    """GK zinciri uyeleri arasi handoff frame-ortusmesini BUDA -> frame-disjoint.

    Zincir f0'a gore sirali; sonraki uyenin, onceki uyelerde GORULEN frame'lerdeki satirlari
    DUSURULUR (erken-baslayan tutulur). GK kumesi frame-disjoint olur -> cannot-link DSU temiz
    birlestirir + over-merge=0 GK icin de SERT kalir. Budanan satirlar RAPORLANIR."""
    drop_idx = []
    report = []
    for end in ("endA", "endL"):
        chain = gk.get(end, {}).get("members", [])
        seen = set()
        for t in chain:
            sub = df[df["tid"] == t]
            dup = sub[sub["frame"].astype(int).isin(seen)]
            if len(dup):
                report.append(dict(end=end, tid=int(t),
                                   trimmed_frames=int(len(dup)),
                                   frames=[int(x) for x in dup["frame"].tolist()]))
                drop_idx.extend(dup.index.tolist())
            seen |= set(sub["frame"].astype(int).tolist())
    if drop_idx:
        df = df.drop(index=drop_idx).reset_index(drop=True)
    return df, report


# ============================================== metrik tracklet ozetleri =====
def _metric_summaries(df_xy: pd.DataFrame, k_end: int = 5) -> list[dict]:
    """Her tid (>=0) -> head/tail (metrik), frame_set, f0/f1/t0/t1 (track_stitch deseni)."""
    summaries = []
    for tid, g in df_xy.groupby("tid"):
        if int(tid) < 0:
            continue
        g = g.sort_values("frame")
        XY = g[["X", "Y"]].to_numpy(float)
        fr = g["frame"].to_numpy(int)
        tt = g["t_sec"].to_numpy(float)
        c = len(g); k = min(k_end, c)
        summaries.append(dict(
            tid=int(tid), n=c, f0=int(fr.min()), f1=int(fr.max()),
            t0=float(tt.min()), t1=float(tt.max()),
            head=np.median(XY[:k], axis=0), tail=np.median(XY[-k:], axis=0),
            frame_set=frozenset(int(x) for x in fr)))
    return summaries


# ================================================ pencere konsolidasyonu =====
def consolidate_window(df: pd.DataFrame, cal: dict, fps: float,
                       vmax: float = 6.5, max_gap_s: float = 8.0,
                       far_relax: float = 2.0, gap_pen: float = 0.15,
                       no_link: float = 0.9, k_end: int = 5,
                       detect_gk: bool = True) -> dict:
    """bot_raw_tracks -> konsolide player_id. track_stitch cekirdegi METRIK uzayda.

    Adimlar: (1) GK tespit (derin-rezidans+advance-return) + handoff-budama (frame-disjoint).
    (2) metrik ozetler + clique_floor (sert alt-sinir). (3) forward-only LSAP path-cover,
    VMAX motion-gate + frame-disjoint cannot-link + GK<->saha YASAK. (4) _DSU (transitive
    cannot-link) + GK zinciri on-birlestirme. (5) over-merge=0 ASSERT.

    Doner: dict(consolidated_df, mapping, report, gk).
    """
    from scipy.optimize import linear_sum_assignment

    n_tid_before = int(df["tid"].nunique())                       # -1 dahil ham tid
    n_real_tid_before = int(df[df["tid"] >= 0]["tid"].nunique())  # gercek tid (>=0)
    gk = detect_goalkeepers(df, cal, fps) if detect_gk else {"endA": {"members": []},
                                                             "endL": {"members": []}}
    df, trim_report = trim_gk_overlaps(df, gk)
    gk_members = {t for end in ("endA", "endL") for t in gk.get(end, {}).get("members", [])}

    df_xy = project_df(df, cal)
    summaries = _metric_summaries(df_xy, k_end)
    N = len(summaries)
    idx_of = {summaries[i]["tid"]: i for i in range(N)}

    # GK uc etiketleri (hangi tid hangi uca) — GK<->GK ayni-uc serbest, capraz YASAK
    gk_end = {}
    for end in ("endA", "endL"):
        for t in gk.get(end, {}).get("members", []):
            gk_end[t] = end

    clique_floor = _clique_floor(summaries)

    far_thr = float(np.quantile(df_xy["foot_y"].values, 1 / 3))
    def _far(i):
        sub = df_xy[df_xy["tid"] == summaries[i]["tid"]]
        return float(np.mean(sub["foot_y"].values < far_thr)) > 0.5

    # maliyet matrisi C[i,j]: i.tail -> j.head, SERT kapilar (track_stitch ile ayni)
    C = np.full((N, N), _BIG, float)
    for i in range(N):
        Ai = summaries[i]; ti_far = _far(i)
        ti = Ai["tid"]
        for j in range(N):
            if i == j:
                continue
            Bj = summaries[j]; tj = Bj["tid"]
            if not (Ai["t1"] < Bj["t0"]):          # forward-only
                continue
            if not Ai["frame_set"].isdisjoint(Bj["frame_set"]):  # cannot-link (eszaman)
                continue
            dt = Bj["t0"] - Ai["t1"]
            if not (0.0 < dt <= max_gap_s):
                continue
            # GK<->saha YASAK; GK<->GK yalniz AYNI uc serbest
            gi, gj = ti in gk_end, tj in gk_end
            if gi != gj:
                continue
            if gi and gj and gk_end[ti] != gk_end[tj]:
                continue
            link_far = ti_far or _far(j)
            cap = vmax * (far_relax if link_far else 1.0)   # uzak-ucte motion-gate gevser
            speed = float(np.linalg.norm(Bj["head"] - Ai["tail"]) / dt)
            if speed > cap:                         # VMAX motion gate
                continue
            C[i, j] = speed / vmax + gap_pen * (dt / max_gap_s)

    # GLOBAL one-to-one path-cover (2N x 2N, no_link kosegen). GREEDY DEGIL.
    M = np.full((2 * N, 2 * N), _BIG, float)
    M[:N, :N] = C
    for i in range(N):
        M[i, N + i] = no_link
        M[N + i, i] = no_link
    M[N:, N:] = 0.0
    rows, cols = linear_sum_assignment(M)
    links = []
    for r, cc in zip(rows, cols):
        if r < N and cc < N and C[r, cc] < _BIG / 2:
            links.append((float(C[r, cc]), int(r), int(cc)))
    links.sort()

    # _DSU (transitive cannot-link). GK zincirleri ONCE on-birlestir (frame-disjoint).
    dsu = _DSU([summaries[i]["frame_set"] for i in range(N)])
    gk_preseed = 0
    for end in ("endA", "endL"):
        ch = [idx_of[t] for t in gk.get(end, {}).get("members", []) if t in idx_of]
        for a, b in zip(ch[:-1], ch[1:]):
            if dsu.union(a, b):
                gk_preseed += 1
    n_links = 0
    for _cost, i, j in links:
        if dsu.union(i, j):
            n_links += 1

    roots = [dsu.find(i) for i in range(N)]
    members_of = {}
    for i, r in enumerate(roots):
        members_of.setdefault(r, []).append(i)
    clusters = list(members_of.values())

    # over-merge ASSERT + olcum: kume-ici frame-cakismasi (strict cannot-link)
    over_merge_pairs = []
    for mem in clusters:
        fs_prev = []
        for i in mem:
            fsi = summaries[i]["frame_set"]
            for (jp, fp) in fs_prev:
                inter = fsi & fp
                if inter:
                    d, _ = _co_distance(df_xy, summaries[i]["tid"], summaries[jp]["tid"])
                    over_merge_pairs.append(dict(
                        tid_a=summaries[jp]["tid"], tid_b=summaries[i]["tid"],
                        shared=len(inter), co_dist_m=round(d, 2)))
            fs_prev.append((i, fsi))
    n_over_merge = len(over_merge_pairs)

    # player_id: toplam gozleme gore azalan; GK kumeleri sabit etiket
    cl_obs = sorted(([sum(summaries[i]["n"] for i in mem), mem] for mem in clusters),
                    key=lambda x: -x[0])
    tid2pid = {}
    pid_meta = []
    for pid, (tot, mem) in enumerate(cl_obs):
        member_tids = sorted(summaries[i]["tid"] for i in mem)
        gk_lab = next((gk_end[t] for t in member_tids if t in gk_end), None)
        for i in mem:
            tid2pid[summaries[i]["tid"]] = pid
        f0 = min(summaries[i]["f0"] for i in mem)
        f1 = max(summaries[i]["f1"] for i in mem)
        span = max(1, f1 - f0 + 1)
        covered = len(set().union(*[summaries[i]["frame_set"] for i in mem]))
        pid_meta.append(dict(player_id=pid, role=("GK_" + gk_lab[-1]) if gk_lab else "field",
                             n_tracklets=len(mem), obs=int(tot),
                             member_tids=member_tids, f0=int(f0), f1=int(f1),
                             coverage=round(covered / span, 3)))

    # konsolide df (gurultu tid<0 -> player_id=-1, isaretli)
    out = df_xy.copy()
    out["player_id"] = out["tid"].map(tid2pid).fillna(-1).astype(int)
    out["is_gk"] = out["tid"].isin(gk_members)
    out["bridged"] = False

    wf0, wf1 = int(df["frame"].min()), int(df["frame"].max())
    longest = max(pid_meta, key=lambda m: m["coverage"]) if pid_meta else None
    report = dict(
        n_tid_before=n_tid_before, n_real_tid_before=n_real_tid_before,
        n_real_tid_in=N, n_clusters=len(clusters),
        clique_floor=clique_floor, clique_floor_basis="span-overlap sweep (hard lower bound)",
        n_links=n_links, gk_preseed_unions=gk_preseed,
        over_merge_violations=n_over_merge, over_merge_pairs=over_merge_pairs,
        gk_handoff_trims=trim_report,
        gk_summary={end: dict(
            player_id=tid2pid.get(gk[end]["members"][0]) if gk[end].get("members") else None,
            members=gk[end].get("members", []),
            coverage=gk[end].get("coverage"),
            residence_mean=gk[end].get("residence_mean"),
            advance_return_events=gk[end].get("advance_return_events"),
            max_excursion_m=gk[end].get("max_excursion_m"),
            far_band=gk[end].get("far_band"),
            confidence=gk[end].get("confidence"))
                    for end in ("endA", "endL")},
        longest_track_coverage=longest["coverage"] if longest else None,
        window_frames=int(wf1 - wf0 + 1),
        vmax_mps=vmax, max_gap_s=max_gap_s, unit="metre (vheight ~+-13%)",
        per_cluster=pid_meta,
        caveats=[
            "14'e ZORLAMA yok; clique_floor=%d sert alt-sinir." % clique_floor,
            "eszamanli co-located AYRI oyuncu birlesmez (frame-disjoint cannot-link).",
            "GK handoff frame-ortusmesi BUDANDI (gk_handoff_trims) -> frame-disjoint.",
            "GK = derin-rezidans + advance-return (anlik en-arka/min-max-X DEGIL).",
            "far-uc GK seyrek gozlemli -> MODERATE guven (gizlenmez); aktif-varyans recall-bagimli.",
            "metre vheight oyuncu-boyu olcek ~+-13%; mutlak hiz iddiasi yok.",
        ],
    )
    assert n_over_merge == 0, f"OVER-MERGE: {over_merge_pairs}"
    assert len(clusters) >= clique_floor, f"floor ihlali {len(clusters)}<{clique_floor}"
    return dict(consolidated_df=out, mapping=tid2pid, report=report, gk=gk)


# ============================================== kisa-gap koprulemesi =========
def bridge_short_gaps(df: pd.DataFrame, cal: dict, fps: float,
                      gap_bridge_s: float = 0.5, id_col: str = "player_id") -> tuple[pd.DataFrame, dict]:
    """Konsolide player_id ICINDE <=gap_bridge_s gozlem-bosluklarini lineer interp ile doldur.

    Spawn azaltma: kisa okluzyonda oyuncu kaybolup tekrar dogmasini (aktif-sayim dip/spike)
    onler. UZUN bosluk KOPRULENMEZ (replay2d gap-kapisi gap_max_s=0.5 ile UYUMLU; >gap_bridge_s
    -> sahte cross-field iz YOK). Koprulenen satirlar bridged=True; foot_x/foot_y lineer, X,Y
    yeniden projeksiyonla tutarli. tid=-2 sentinel (gercek tid degil, izlenebilir).
    Doner: (df_bridged, stats). over_merge'a DOKUNMAZ (konsolidasyon sonrasi calisir)."""
    gap_f = max(1, int(round(gap_bridge_s * fps)))
    if "bridged" not in df.columns:
        df = df.assign(bridged=False)
    base = df.copy()
    new = []
    n_gaps = 0
    for pid, g in base[base[id_col] >= 0].groupby(id_col):
        g = g.sort_values("frame")
        fr = g["frame"].to_numpy(int)
        fx = g["foot_x"].to_numpy(float); fy = g["foot_y"].to_numpy(float)
        bh = g["box_h"].to_numpy(float); bw = g["box_w"].to_numpy(float)
        isgk = bool(g["is_gk"].iloc[0]) if "is_gk" in g.columns else False
        # ayni-frame coklu satir olabilir; tekil frame'lere indir (ilk satir)
        seen = set(); order = []
        for k in range(len(fr)):
            if fr[k] not in seen:
                seen.add(fr[k]); order.append(k)
        for a, b in zip(order[:-1], order[1:]):
            gap = fr[b] - fr[a]
            if 1 < gap <= gap_f:
                n_gaps += 1
                for f in range(fr[a] + 1, fr[b]):
                    w = (f - fr[a]) / gap
                    new.append(dict(tid=-2, frame=int(f), t_sec=float(f / fps),
                                    foot_x=float(fx[a] + w * (fx[b] - fx[a])),
                                    foot_y=float(fy[a] + w * (fy[b] - fy[a])),
                                    box_h=float(bh[a]), box_w=float(bw[a]),
                                    player_id=int(pid), is_gk=isgk, bridged=True))
    stats = dict(gap_bridge_s=gap_bridge_s, gap_bridge_frames=gap_f,
                 n_gaps_bridged=int(n_gaps), n_rows_added=int(len(new)))
    if not new:
        return base, stats
    nd = pd.DataFrame(new)
    P = project(cal, nd[["foot_x", "foot_y"]].values)
    nd["X"] = P[:, 0]; nd["Y"] = P[:, 1]
    cols = [c for c in base.columns]
    for c in cols:
        if c not in nd.columns:
            nd[c] = np.nan
    merged = pd.concat([base, nd[cols]], ignore_index=True).sort_values(
        ["player_id", "frame"]).reset_index(drop=True)
    return merged, stats


# ===================================================================== CLI ===
if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/bot_raw_tracks.parquet"
    calib = sys.argv[2] if len(sys.argv) > 2 else "calib/cankaya_cam2_v4.json"
    outdir = sys.argv[3] if len(sys.argv) > 3 else "scratchpad/wf2_gk"
    os.makedirs(outdir, exist_ok=True)
    cal = load_calib(calib)
    df = pd.read_parquet(raw)
    fps = 88294 / 3549.97
    res = consolidate_window(df, cal, fps)
    con = res["consolidated_df"]
    con_bridged, bstats = bridge_short_gaps(con, cal, fps, gap_bridge_s=0.5)
    res["report"]["bridge"] = bstats
    con_bridged.to_parquet(os.path.join(outdir, "consolidated.parquet"))
    json.dump(res["report"], open(os.path.join(outdir, "consolidate_report.json"), "w"),
              indent=2, default=float)
    json.dump(res["gk"], open(os.path.join(outdir, "gk_detect.json"), "w"),
              indent=2, default=float)
    r = res["report"]
    print("calib %s  saha %.1fx%.1fm" % (Path(calib).name, cal["L"], cal["W"]))
    print("tid_before=%s real_in=%d -> clusters=%d (floor=%d)  over_merge=%d  n_links=%d"
          % (r["n_tid_before"], r["n_real_tid_in"], r["n_clusters"], r["clique_floor"],
             r["over_merge_violations"], r["n_links"]))
    for end in ("endA", "endL"):
        g = r["gk_summary"][end]
        print("%s pid=%s members=%s cov=%.2f res=%.2f adv-ret=%s max-exc=%.1fm far=%s conf=%s"
              % (end, g["player_id"], g["members"], g["coverage"] or 0,
                 g["residence_mean"] or 0, g["advance_return_events"],
                 g["max_excursion_m"] or 0, g["far_band"], g["confidence"]))
    print("bridge: %d gaps, +%d rows (<=%.1fs, replay2d-uyumlu)"
          % (bstats["n_gaps_bridged"], bstats["n_rows_added"], bstats["gap_bridge_s"]))
