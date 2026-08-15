#!/usr/bin/env python3
"""GSI gap-fill kabul testi — StrongSORT (arXiv:2202.13514) GSI bileseni.

Sentetik, ANALITIK olarak bilinen yer-gercegi (ground-truth) ile dogrular:

  1. Egri yol (cember yayi) uzerinde, ortadaki bir bosluk silindiginde GSI
     (RBF-GP) ile doldurulan yolun YAY UZUNLUGU, duz-cizgi (np.interp) dolguya
     gore gercek analitik yay uzunluguna belirgin sekilde DAHA YAKIN. Boylece
     duz-cizgi dolgunun mesafe-undercount problemini GSI'nin azalttigi olculur.
  2. Duz bir segmentte GSI ~= duz-cizgi (ZARAR YOK).
  3. arc_length dogrulugu (bilinen poligon) + NaN dayanikliligi.
  4. GP gercekten gozlenen noktalardan gecer (sahte/ezber degil).
  5. gsi_fill gozlenen frame'leri DEGISTIRMEZ; yetersiz gozlemde cokmeden
     duz-cizgiye geri duser.

Beklenen cevap MODULE icine GOMULU DEGIL: yer-gercegi analitik (R*aci),
baseline ise bagimsiz np.interp ile hesaplanir.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from gsi_gapfill import (  # noqa: E402
    gsi_fill, linear_fill, arc_length, rbf_gp_mean,
)


def _circle(N, turn_deg, R=10.0, theta0=0.0):
    """Tek-parametreli (frame) cember yayi; analitik yay = R*aci."""
    frames = np.arange(N, dtype=float)
    theta = theta0 + np.linspace(0.0, np.deg2rad(turn_deg), N)
    X = R * np.cos(theta)
    Y = R * np.sin(theta)
    true_arc = R * np.deg2rad(turn_deg)
    return frames, X, Y, true_arc


def _independent_linear_fill(frames, xs, ys, gap):
    """Modulden BAGIMSIZ duz-cizgi baseline (dogrudan np.interp)."""
    obs = ~gap
    fo = frames[obs]
    out_x = xs.copy()
    out_y = ys.copy()
    out_x[gap] = np.interp(frames[gap], fo, xs[obs])
    out_y[gap] = np.interp(frames[gap], fo, ys[obs])
    return out_x, out_y


def test_gsi_recovers_curve_better_than_linear():
    # Tam cemberin 90-derecelik yayini kapsayan bir bosluk: kiris yayi ciddi
    # sekilde kisaltir -> duz-cizgi belirgin undercount yapar.
    frames, X, Y, true_arc = _circle(N=240, turn_deg=360.0, R=10.0)
    gap = (frames >= 90) & (frames <= 150)

    xo = X.copy(); yo = Y.copy()
    xo[gap] = np.nan; yo[gap] = np.nan   # bosluk: gercek konum bilinmiyor

    # GSI (default = auto length_scale heuristigini de sinar)
    gx, gy = gsi_fill(frames, xo, yo, gap)
    # Bagimsiz duz-cizgi baseline
    lx, ly = _independent_linear_fill(frames, X.copy(), Y.copy(), gap)

    g_arc = arc_length(gx, gy)
    l_arc = arc_length(lx, ly)
    g_err = abs(g_arc - true_arc)
    l_err = abs(l_arc - true_arc)

    # (a) duz-cizgi GERCEKTEN undercount yapiyor (kiris < yay)
    assert l_arc < true_arc - 1.0, f"baseline undercount beklenir: {l_arc} vs {true_arc}"
    assert l_err > 1.0, f"baseline hatasi anlamli olmali: {l_err}"

    # (b) GSI belirgin marjla daha yakin (en az 5x daha kucuk hata)
    assert g_err < l_err, f"GSI >= linear hata: g={g_err} l={l_err}"
    assert g_err < 0.2 * l_err, f"GSI marji yetersiz: g={g_err} l={l_err}"

    # (c) GSI mutlak olarak gercege cok yakin (%2'den iyi)
    assert g_err < 0.02 * true_arc, f"GSI mutlak hata buyuk: {g_err}"

    # (d) GSI dolgusu GERCEKTEN EGRI: bosluk noktalari duz-cizgi kirisinden
    #     anlamli sapar (yoksa linear'dan farksiz olurdu)
    max_dev = np.max(np.hypot(gx[gap] - lx[gap], gy[gap] - ly[gap]))
    assert max_dev > 1.0, f"GSI yolu egri degil (linear'a cok yakin): {max_dev}"


def test_gsi_no_harm_on_straight_segment():
    # Duz cizgi: GSI duz-cizgiye ~ esit olmali (zarar yok), ikisi de gercek
    # uzunlugu vermeli.
    N = 200
    frames = np.arange(N, dtype=float)
    X = 0.3 * frames + 2.0
    Y = -0.7 * frames + 5.0
    true_len = float(np.hypot(X[-1] - X[0], Y[-1] - Y[0]))

    gap = (frames >= 70) & (frames <= 130)
    xo = X.copy(); yo = Y.copy()
    xo[gap] = np.nan; yo[gap] = np.nan

    gx, gy = gsi_fill(frames, xo, yo, gap)
    lx, ly = _independent_linear_fill(frames, X.copy(), Y.copy(), gap)

    g_arc = arc_length(gx, gy)
    l_arc = arc_length(lx, ly)

    # GSI ile linear nerdeyse ayni (zarar yok) ve gercek uzunluga cok yakin
    assert abs(g_arc - l_arc) < 1e-3 * true_len, f"duz segmentte sapma: {abs(g_arc - l_arc)}"
    assert abs(g_arc - true_len) < 1e-3 * true_len, f"duz uzunluk hatasi: {g_arc} vs {true_len}"


def test_arc_length_known_polyline_and_nan():
    # 3-4-3 = 10 birim poligon
    xs = np.array([0.0, 3.0, 3.0, 0.0])
    ys = np.array([0.0, 0.0, 4.0, 4.0])
    assert abs(arc_length(xs, ys) - 10.0) < 1e-9

    # NaN iceren segmentler atlanir: yalniz [0->1] = 1.0 sayilir
    assert abs(arc_length([0.0, 1.0, np.nan, 3.0], [0, 0, 0, 0]) - 1.0) < 1e-9

    # <2 nokta -> 0
    assert arc_length([1.0], [2.0]) == 0.0


def test_gp_interpolates_observed_points():
    # Gozlenen noktalarda GP ortalamasi (dusuk gurultu) ~ veriye esit olmali:
    # bu, gercek bir GP regresyonu oldugunu (sahte sabit/ezber degil) gosterir.
    t = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    y = np.array([0.0, 1.0, 4.0, 9.0, 16.0, 25.0])  # parabol
    pred = rbf_gp_mean(t, y, t, length_scale=2.0, noise=1e-5)
    assert np.max(np.abs(pred - y)) < 1e-3, f"GP gozlemden gecmiyor: {np.max(np.abs(pred - y))}"

    # ve ARALARDA makul interpolasyon (sabit degil): t=2.5 degeri 4 ile 9 arasi
    mid = float(rbf_gp_mean(t, y, np.array([2.5]), length_scale=2.0, noise=1e-5)[0])
    assert 4.0 < mid < 9.0, f"GP ara-deger makul degil: {mid}"


def test_observed_preserved_and_fallback():
    # Gozlenen frame'ler dolguda AYNEN korunur
    frames, X, Y, _ = _circle(N=120, turn_deg=180.0)
    gap = (frames >= 40) & (frames <= 70)
    xo = X.copy(); yo = Y.copy()
    xo[gap] = np.nan; yo[gap] = np.nan
    gx, gy = gsi_fill(frames, xo, yo, gap)
    assert np.allclose(gx[~gap], X[~gap]), "gozlenen x degisti"
    assert np.allclose(gy[~gap], Y[~gap]), "gozlenen y degisti"
    # bosluk gercekten dolduruldu (NaN kalmadi)
    assert np.all(np.isfinite(gx)) and np.all(np.isfinite(gy))

    # Yetersiz gozlem (<2): cokmeden duz-cizgi geri-dususe gider
    fr = np.arange(5.0)
    g2 = np.array([False, True, True, True, False])  # yalniz uctaki 2 gozlem yetersiz? -> 2 gozlem var
    xx = np.array([0.0, 9, 9, 9, 4.0]); yy = np.array([0.0, 9, 9, 9, 4.0])
    rx, ry = gsi_fill(fr, xx, yy, g2)
    assert np.all(np.isfinite(rx)) and np.all(np.isfinite(ry))
    # 2 uc gozlem (0,0)->(4,4) arasi lineer: orta nokta (2,2)
    assert abs(rx[2] - 2.0) < 1e-6 and abs(ry[2] - 2.0) < 1e-6

    # Gercekten <2 gozlem: tek gozlem -> cokmez
    one = np.array([False, True, True, True, True])
    rx2, ry2 = gsi_fill(fr, xx, yy, one)
    assert rx2.shape == fr.shape
