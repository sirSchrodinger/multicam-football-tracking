#!/usr/bin/env python3
"""inframe_dedup — ayni karede AYNI oyuncunun cift-tespitini bastir (pitch-NMS).

PROBLEM: 12-dk tam macta tracker bazen tek oyuncuya ANLIK iki kutu uretir
(kismi-orgu, govde-bolunmesi). Bu cift-tespitler:
  * per-frame es-zamanliligi sisirir -> clique_floor (oyuncu-sayisi sert alt
    siniri) yapay yuksek (18 olcusduk, gercek ~14-15);
  * yapay cannot-link uretir -> dogru tracklet'lerin birlesmesini engeller.

COZUM: her karede pitch-uzayinda <sep_m olan tespit ciftlerinde DUSUK-skorlu
(conf x box_h) olani O KAREDE bastir. Tracklet'i SILMEZ -- yalniz o karedeki
fazla satiri isaretler. Tipik gercek-oyuncu ayak-mesafesi >=0.6-0.8m oldugundan
sep_m=0.7 konservatiftir (gercek yakin-markaji silmez).

DURUSTLUK: yalnizca COK-yakin (fiziksel-olarak-imkansiz) ciftler bastirilir;
mesafe esigi raporlanir. Asla oyuncu UYDURMAZ, yalniz fazla-kutu DUSURUR.
Lisans: numpy + scipy.spatial (BSD). GPU yok.
"""
from __future__ import annotations

import numpy as np


def inframe_dedup_mask(df, x_col="pitch_x", y_col="pitch_y",
                       sep_m: float = 0.7) -> np.ndarray:
    """Doner: keep mask (bool, len(df)) -- False = ayni-kare cift-tespit fazlasi.

    df sirasi korunur (mask df.index sirasinda DEGIL, df satir-sirasinda).
    Skor = conf * box_h (yoksa 1.0); cift icinde dusuk-skorlu dusurulur.
    """
    from scipy.spatial import cKDTree

    n = len(df)
    keep = np.ones(n, dtype=bool)
    frame = df["frame"].to_numpy(np.int64)
    X = df[x_col].to_numpy(np.float64)
    Y = df[y_col].to_numpy(np.float64)
    conf = (df["conf"].to_numpy(np.float64) if "conf" in df.columns
            else np.ones(n))
    boxh = (df["box_h"].to_numpy(np.float64) if "box_h" in df.columns
            else np.ones(n))
    score = conf * boxh

    # kare-bazli gruplama (satir-pozisyon indeksleri)
    order = np.argsort(frame, kind="stable")
    fr_s = frame[order]
    bounds = np.flatnonzero(np.diff(fr_s)) + 1
    groups = np.split(order, bounds)

    for gi in groups:
        if gi.size < 2:
            continue
        P = np.column_stack([X[gi], Y[gi]])
        good = np.isfinite(P).all(axis=1)
        if good.sum() < 2:
            continue
        gi_g = gi[good]
        Pg = P[good]
        tree = cKDTree(Pg)
        pairs = tree.query_pairs(r=sep_m)
        for a, b in pairs:
            ra, rb = gi_g[a], gi_g[b]
            if not (keep[ra] and keep[rb]):
                continue
            lo = ra if score[ra] < score[rb] else rb
            keep[lo] = False
    return keep
