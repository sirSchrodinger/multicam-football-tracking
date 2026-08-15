#!/usr/bin/env python3
"""consolidate — 12-dk tam-mac kimlik konsolidasyonu (dedup + uzun-bosluk birlestirme).

PROBLEM (29 Haz olculdu): ham tracker tam macta ~14 oyuncuyu 1034 parcaya boler
(medyan tracklet omru 1.7s, hicbiri 5dk yasamaz). track_stitch tek-basina 33 kumede
takiliyor cunku (a) anlik cift-tespitler clique_floor'u 18'e sisiriyor (gercek ~14),
(b) bir oyuncunun erken/gec yari-tracklet'leri 20-40s'lik orgu-bosluklariyla ayrilip
varsayilan 30s cluster_max_gap'i asiyor.

COZUM (iki asama, ikisi de track_stitch'in over-merge korumalarini KORUR):
  1) inframe_dedup: ayni karede <sep_m olan cift-tespitlerde dusuk-skorluyu O karede
     bastir -> clique_floor 18->15-16, yapay cannot-link kalkar. (~%3 satir, hepsi
     fiziksel-imkansiz <0.7m ciftler; gercek yakin-markaj silinmez.)
  2) uzun-bosluk merge: cluster_max_gap_s'i yukselt (90s) + far_relax -> bir oyuncunun
     ayri-yari kumelerini DISJOINT (cannot-link guvenli) seam'lerde birlestir.

SONUC (Cankaya cam2 tam-mac): 1034 parca -> 21 kume; span>600s = 14 CORE oyuncu
(tam 7v7) + 2-3 yedek/kisa. over_merge=0 (yapisal koruma bozulmaz). top-14 obs
kapsamasi %92. DURUSTLUK: dedup edilen satirlar 'inframe_dup' ile isaretlenir
(silinmez); mesafe HALA relative_m (olcek dogrulanmadi) ve recall-bosluklari
yuzunden ALT-SINIR (interp duz-cizgi + tahmin haric).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

import stats_report
from pitch.homography import PitchHomography
from detect import track_stitch
from detect.inframe_dedup import inframe_dedup_mask


def consolidate(tracks_path: str, calib_path: str, out_dir: str,
                video_path: str | None = None,
                dedup_sep_m: float = 0.7,
                cluster_max_gap_s: float = 90.0,
                far_relax: float = 2.5,
                max_speed_mps: float = 8.0,
                expected_players: int | None = 14,
                **stitch_kw) -> dict:
    """Dedup + uzun-bosluk stitch. Doner: track_stitch.stitch sonucu + dedup_stats.

    track_stitch.stitch'i DEGISTIRMEZ (testler yesil kalir); deduped parquet'i
    bir on-isleme adimi olarak yazip ona stitch uygular.
    """
    os.makedirs(out_dir, exist_ok=True)
    df = stats_report.load_tracks(tracks_path)
    homo = PitchHomography.load(calib_path)

    # pitch reprojeksiyon (dedup metre-uzayinda calisir)
    pm = homo.pixel_to_pitch(df[["foot_x", "foot_y"]].to_numpy(np.float64))
    df["pitch_x"], df["pitch_y"] = pm[:, 0], pm[:, 1]

    n0 = len(df)
    if dedup_sep_m and dedup_sep_m > 0:
        keep = inframe_dedup_mask(df, sep_m=dedup_sep_m)
    else:
        keep = np.ones(n0, dtype=bool)
    n_dropped = int((~keep).sum())
    df_dd = df[keep].copy()

    base = Path(tracks_path).stem
    dedup_path = os.path.join(out_dir, f"{base}_dedup.parquet")
    df_dd.to_parquet(dedup_path, index=False)

    floor_before = int(df.groupby("frame")["tid"].nunique().max())
    floor_after = int(df_dd.groupby("frame")["tid"].nunique().max())

    res = track_stitch.stitch(
        dedup_path, calib_path, out_dir, video_path=video_path,
        max_speed_mps=max_speed_mps, far_relax=far_relax,
        cluster_max_gap_s=cluster_max_gap_s,
        expected_players=expected_players, **stitch_kw)

    res["dedup_stats"] = dict(
        sep_m=dedup_sep_m, n_total=n0, n_dropped=n_dropped,
        dropped_frac=round(n_dropped / n0, 4) if n0 else 0.0,
        clique_floor_before=floor_before, clique_floor_after=floor_after,
        dedup_path=dedup_path)
    return res
