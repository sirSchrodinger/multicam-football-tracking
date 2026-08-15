#!/usr/bin/env python3
"""GSI — Gaussian-Smoothed Interpolation for track gap-filling.

Kaynak / citation
-----------------
StrongSORT: Make DeepSORT Great Again, Du et al., arXiv:2202.13514
(IEEE TMM 2023), "Gaussian-smoothed interpolation (GSI)" bileseni. Orada
amac, tespit boslugu sonrasi olusan track parcalarini, BUSAYLI (linear)
interpolasyon yerine bir Gauss-Sureci (GP) regresyonu ile baglamak; boylece
gerek gurultu yumusatilir gerek de bosluk icindeki yol EGRI olarak tahmin
edilir.

Bu repodaki problem (neden GSI?)
--------------------------------
Mevcut topdown pipeline (bkz. replay2d.py / track_robust.bridge_gaps) tespit
bosluklarini DUZ-CIZGI (np.interp) ile doldurur. Egri bir kosunun (yon
degistiren oyuncu) gercek YAY UZUNLUGU, bosluk boyunca cekilen KIRISTEN her
zaman >= uzundur. Dolayisiyla duz-cizgi dolgu, kat edilen mesafeyi SISTEMATIK
OLARAK DUSUK SAYAR (under-count). Olcumlerimizdeki mesafe under-count'unun bir
kismi tam olarak budur.

GSI cozumu: oyuncunun bosluk CEVRESINDEKI gozlenen (frame -> x) ve
(frame -> y) orneklerine RBF cekirdekli bir GP oturt; bosluk frame'lerini bu
GP'nin POSTERIOR ORTALAMA egrisi ile doldur. GP, bosluk kenarlarindaki konum
VE egim bilgisini tasidigi icin kirisi "yuvarlayarak" egriyi takip eder; bu
egrinin yay uzunlugu gercege duz-cizgiden daha yakindir.

Yontem (saf numpy RBF GP)
-------------------------
Her koordinat (x ve y) BAGIMSIZ olarak frame'in fonksiyonu gibi modellenir:
    f(t) ~ GP(mean=ybar, k),   k(t,t') = exp(-0.5 (t-t')^2 / l^2)
Sinyal varyansi 1'e sabitlenir (olcek tahmininin posterior ortalamada
gurultu/sinyal ORANI uzerinden girmesi yeter), gozlem gurultusu `noise`^2.
Gozlenen (t_o, y_o) icin:
    alpha = (K_oo + noise^2 I)^{-1} (y_o - ybar)
    f_*   = K_*o alpha + ybar
Gozlenen frame'ler dolguda DEGISTIRILMEZ (orijinal degerleri korunur); yalniz
gap_mask=True frame'leri GP ortalamasiyla yazilir.

length_scale (l)
----------------
RBF uzunluk-olcegi tahminin ne kadar "esnek/egri" olacagini belirler:
  * cok kucuk  -> GP bosluk ortasinda ortalamaya cokuyor (kotu),
  * cok buyuk  -> asiri yumusar, neredeyse duz-cizgiye dejenere olur.
Verilmezse, gozlenen frame araliginin ~1/4'u sezgisel default olarak alinir
(tek baskin bukulme icin iyi calisir). Yuksek frekansli salinim icin daha
kucuk bir l elle verilmelidir.

Lisans: yalniz numpy (BSD). GPU/dis-model/GPL bagimliligi YOK.
"""
from __future__ import annotations

import numpy as np

__all__ = ["gsi_fill", "linear_fill", "arc_length", "rbf_gp_mean"]


def rbf_gp_mean(t_obs: np.ndarray, y_obs: np.ndarray, t_query: np.ndarray,
                length_scale: float, noise: float = 1e-3) -> np.ndarray:
    """RBF-cekirdekli GP posterior ORTALAMASI (saf numpy).

    f_*(t) = ybar + k(t_*, t_o) (K_oo + noise^2 I)^{-1} (y_o - ybar)

    Sinyal varyansi 1; ybar = ortalama(y_obs). Cozumde Cholesky/solve;
    tekillige karsi kucuk bir jitter eklenir.
    """
    t_obs = np.asarray(t_obs, dtype=float).ravel()
    y_obs = np.asarray(y_obs, dtype=float).ravel()
    t_query = np.asarray(t_query, dtype=float).ravel()
    if t_obs.size == 0:
        return np.full(t_query.shape, np.nan)
    if t_obs.size == 1:
        # tek nokta: GP ortalamasi sabit; o degeri dondur
        return np.full(t_query.shape, y_obs[0])

    ls = float(length_scale)
    if not np.isfinite(ls) or ls <= 0:
        ls = 1.0
    ybar = float(np.mean(y_obs))
    yc = y_obs - ybar

    # gozlem-gozlem korelasyon matrisi (sinyal var = 1)
    d2 = (t_obs[:, None] - t_obs[None, :]) ** 2
    K = np.exp(-0.5 * d2 / (ls * ls))
    K[np.diag_indices_from(K)] += float(noise) ** 2 + 1e-9  # gurultu + jitter

    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, yc))
    except np.linalg.LinAlgError:  # pragma: no cover - savunmaci
        alpha = np.linalg.lstsq(K, yc, rcond=None)[0]

    d2s = (t_query[:, None] - t_obs[None, :]) ** 2
    Ks = np.exp(-0.5 * d2s / (ls * ls))
    return Ks @ alpha + ybar


def _auto_length_scale(t_obs: np.ndarray) -> float:
    span = float(np.max(t_obs) - np.min(t_obs))
    return max(span / 4.0, 1.0)


def gsi_fill(frames, xs, ys, gap_mask, length_scale=None, noise: float = 1e-3):
    """Tespit bosluklarini GSI (RBF-GP) ile doldur.

    Parameters
    ----------
    frames : (N,) frame indeksleri (artma SARTI yok; ic mantik sirasiz calisir)
    xs, ys : (N,) konumlar (pitch metre veya px farketmez). gap_mask=True
             yerlerinde icerik NaN/cop olabilir, kullanilmaz.
    gap_mask : (N,) bool; True = doldurulacak (eksik) frame.
    length_scale : RBF uzunluk-olcegi (frame biriminde). None ise gozlenen
                   frame araliginin ~1/4'u.
    noise : GP gozlem gurultusu (sinyal-var=1'e gore oran). Kucuk => gozlenen
            kenarlara siki oturur.

    Returns
    -------
    (xs_filled, ys_filled) : (N,) float diziler. Gozlenen frame'ler orijinal
    degerleriyle KORUNUR; yalniz gap_mask frame'leri GP ortalamasiyla yazilir.
    Yeterli gozlem (>=2) yoksa duz-cizgi (np.interp) dolguya geri duser.
    """
    frames = np.asarray(frames, dtype=float).ravel()
    xs = np.asarray(xs, dtype=float).ravel().copy()
    ys = np.asarray(ys, dtype=float).ravel().copy()
    gap = np.asarray(gap_mask, dtype=bool).ravel()
    if not (frames.shape == xs.shape == ys.shape == gap.shape):
        raise ValueError("frames, xs, ys, gap_mask ayni uzunlukta olmali")

    obs = (~gap) & np.isfinite(xs) & np.isfinite(ys)
    fill = gap & np.isfinite(frames)

    if obs.sum() < 2 or fill.sum() == 0:
        # GP icin yetersiz: gurevsiz duz-cizgi geri-dusus (hala doldurur)
        return linear_fill(frames, xs, ys, gap)

    t_obs = frames[obs]
    ls = _auto_length_scale(t_obs) if length_scale is None else float(length_scale)
    t_q = frames[fill]

    xs[fill] = rbf_gp_mean(t_obs, xs[obs], t_q, ls, noise=noise)
    ys[fill] = rbf_gp_mean(t_obs, ys[obs], t_q, ls, noise=noise)
    return xs, ys


def linear_fill(frames, xs, ys, gap_mask):
    """Duz-cizgi (np.interp) bosluk dolgusu — baseline / geri-dusus.

    Mevcut pipeline'in (replay2d) yaptiginin sade hali. GSI ile karsilastirma
    ve yetersiz-gozlem geri-dususu icin. Gozlenen frame'ler korunur.
    """
    frames = np.asarray(frames, dtype=float).ravel()
    xs = np.asarray(xs, dtype=float).ravel().copy()
    ys = np.asarray(ys, dtype=float).ravel().copy()
    gap = np.asarray(gap_mask, dtype=bool).ravel()

    obs = (~gap) & np.isfinite(xs) & np.isfinite(ys)
    fill = gap & np.isfinite(frames)
    if obs.sum() < 2 or fill.sum() == 0:
        return xs, ys

    order = np.argsort(frames[obs])
    fo = frames[obs][order]
    xo = xs[obs][order]
    yo = ys[obs][order]
    # np.interp artan xp ister; sirali fo bunu saglar. Aralik disinda uc
    # degerlere clamp eder (np.interp default davranisi).
    xs[fill] = np.interp(frames[fill], fo, xo)
    ys[fill] = np.interp(frames[fill], fo, yo)
    return xs, ys


def arc_length(xs, ys) -> float:
    """Bir yolun yay uzunlugu: ardisik noktalar arasi Euclidean mesafe toplami.

    NaN iceren segmentler atlanir (sonlu segment toplami dondurulur). Girdi
    konumlarinin biriminde sonuc verir (m veya px).
    """
    xs = np.asarray(xs, dtype=float).ravel()
    ys = np.asarray(ys, dtype=float).ravel()
    if xs.size != ys.size:
        raise ValueError("xs ve ys ayni uzunlukta olmali")
    if xs.size < 2:
        return 0.0
    seg = np.hypot(np.diff(xs), np.diff(ys))
    seg = seg[np.isfinite(seg)]
    return float(np.sum(seg))
