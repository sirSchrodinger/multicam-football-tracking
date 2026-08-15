#!/usr/bin/env python3
"""temporal_mine sentetik testleri — bilinen yer-gercegiyle (ground truth).

Dayanak: arXiv:1808.04285 (Unsupervised Hard Example Mining from Videos).
Testler MODULE icine cevabi gomMEZ: yer-gercegi BAGIMSIZ bir dogrusal
yorungeyle uretilir, modulun interpolasyonu o yorungeye karsi dogrulanir.
"""
import numpy as np
import pandas as pd

from detect.temporal_mine import mine_false_negatives, mine_orphan_fp


# Bagimsiz yer-gercegi yorungesi: kareye gore DOGRUSAL hareket.
# (Egim != 0 -> her ara kare farkli, kusatan degerler arasinda KESIN.)
def _gt_x(f): return 1.0 + 1.5 * f      # uzunluk ekseni
def _gt_y(f): return 30.0 - 0.8 * f     # genislik ekseni (ters egim)


def _rows(tid, frames):
    return pd.DataFrame({
        "tid":     [tid] * len(frames),
        "frame":   list(frames),
        "pitch_x": [_gt_x(f) for f in frames],
        "pitch_y": [_gt_y(f) for f in frames],
    })


# ---------------------------------------------------------------------------
# (1) 5-kare delik -> tam o 5 kare, interpolasyon yer-gercegine esit
# ---------------------------------------------------------------------------
def test_five_frame_hole_yields_exactly_five_interpolated_fn():
    # tid=1: 0,1,2 VAR ; 3,4,5,6,7 YOK (5-kare delik) ; 8,9,10 VAR
    present = [0, 1, 2, 8, 9, 10]
    df = _rows(1, present)

    fn = mine_false_negatives(df, max_gap_frames=5)

    # Tam olarak kayip 5 kare, sira ile
    got_frames = [f for (_t, f, _x, _y) in fn]
    assert got_frames == [3, 4, 5, 6, 7]
    assert len(fn) == 5

    # Her hedef BAGIMSIZ dogrusal yer-gercegine esit (interpolasyon dogru)
    for tid, f, x, y in fn:
        assert tid == 1
        assert abs(x - _gt_x(f)) < 1e-9, (f, x, _gt_x(f))
        assert abs(y - _gt_y(f)) < 1e-9, (f, y, _gt_y(f))

    # Interpolasyon kusatan degeri KOPYALAMIYOR: ara degerler strictly arasinda
    xs = [x for (_t, _f, x, _y) in fn]
    assert all(_gt_x(2) < x < _gt_x(8) for x in xs)
    assert xs == sorted(xs)  # egim>0 -> artan


# ---------------------------------------------------------------------------
# (2) 1-2 kare izole blip -> orphan FP olarak isaretlenir
# ---------------------------------------------------------------------------
def test_isolated_blip_flagged_as_orphan_fp():
    long_track = _rows(1, range(0, 30))   # 30 kare -> saglam
    blip1 = _rows(2, [12])                # 1 kare blip
    blip2 = _rows(3, [20, 21])            # 2 kare blip
    df = pd.concat([long_track, blip1, blip2], ignore_index=True)

    orphans = mine_orphan_fp(df, min_len=3)

    assert set(orphans) == {2, 3}         # iki blip yetim
    assert 1 not in orphans               # uzun iz yetim DEGIL

    # Esik sinir davranisi: min_len=2 -> sadece 1-kare blip yetim
    assert set(mine_orphan_fp(df, min_len=2)) == {2}


# ---------------------------------------------------------------------------
# (3) Temiz surekli iz -> ne FN ne FP
# ---------------------------------------------------------------------------
def test_clean_continuous_track_yields_neither():
    df = _rows(7, range(0, 40))           # bosluksuz, uzun

    assert mine_false_negatives(df, max_gap_frames=5) == []
    assert mine_orphan_fp(df, min_len=5) == []


# ---------------------------------------------------------------------------
# (4) Cok uzun bosluk (gercek cikis/giris) MAYINLANMAZ
# ---------------------------------------------------------------------------
def test_gap_larger_than_max_not_mined():
    # 10-kare delik, esik 5 -> doldurulmaz
    df = _rows(1, [0, 1, 2, 13, 14, 15])
    assert mine_false_negatives(df, max_gap_frames=5) == []
    # ayni delik esik 10 ile -> 10 hedef
    fn = mine_false_negatives(df, max_gap_frames=10)
    assert [f for (_t, f, _x, _y) in fn] == list(range(3, 13))


# ---------------------------------------------------------------------------
# (5) Birden cok bosluk + birden cok iz birlikte
# ---------------------------------------------------------------------------
def test_multiple_gaps_and_tracks():
    # tid=1: iki ayri kucuk bosluk
    t1 = _rows(1, [0, 1, 3, 4, 6, 7])     # 2 ve 5 eksik (her biri 1-kare)
    # tid=2: temiz
    t2 = _rows(2, range(0, 8))
    df = pd.concat([t1, t2], ignore_index=True)
    fn = mine_false_negatives(df, max_gap_frames=3)
    pairs = sorted((t, f) for (t, f, _x, _y) in fn)
    assert pairs == [(1, 2), (1, 5)]
    # konumlar yine yer-gercegine esit
    for tid, f, x, y in fn:
        assert abs(x - _gt_x(f)) < 1e-9 and abs(y - _gt_y(f)) < 1e-9


# ---------------------------------------------------------------------------
# (6) Kusatan konum NaN -> o bosluk guvenle atlanir
# ---------------------------------------------------------------------------
def test_nan_bracket_skipped():
    df = _rows(1, [0, 1, 2, 6, 7, 8])
    # f=2'deki konumu NaN yap -> 3,4,5 boslugu interpole edilemez
    df.loc[(df.tid == 1) & (df.frame == 2), ["pitch_x", "pitch_y"]] = np.nan
    fn = mine_false_negatives(df, max_gap_frames=5)
    assert fn == []


# ---------------------------------------------------------------------------
# (7) Ozel sutun adlari (foot_x/foot_y) ile de calismali
# ---------------------------------------------------------------------------
def test_custom_columns():
    df = pd.DataFrame({
        "tid":    [1, 1, 1, 1],
        "frame":  [0, 1, 5, 6],
        "foot_x": [0.0, 1.0, 5.0, 6.0],
        "foot_y": [0.0, 2.0, 10.0, 12.0],
    })
    fn = mine_false_negatives(df, max_gap_frames=5,
                              x_col="foot_x", y_col="foot_y")
    # f=1 (x=1,y=2) ile f=5 (x=5,y=10) arasi dogrusal: f=2,3,4
    got = sorted((f, round(x, 6), round(y, 6))
                 for (_t, f, x, y) in fn)
    assert got == [(2, 2.0, 4.0), (3, 3.0, 6.0), (4, 4.0, 8.0)]


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items()
               if k.startswith("test_") and callable(v)]:
        fn(); print("ok", fn.__name__)
