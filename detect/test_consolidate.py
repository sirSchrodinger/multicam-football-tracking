#!/usr/bin/env python3
"""inframe_dedup + consolidate sentetik testleri (over-merge korumasi + dogru bastirma)."""
import numpy as np
import pandas as pd

from detect.inframe_dedup import inframe_dedup_mask


def _make(rows):
    return pd.DataFrame(rows, columns=["tid", "frame", "pitch_x", "pitch_y",
                                       "conf", "box_h"])


def test_suppresses_close_lowscore_keeps_highscore():
    # ayni karede iki tespit 0.3m -> dusuk-skorlu (conf*box_h) bastirilir
    df = _make([
        [1, 0, 5.0, 5.0, 0.9, 100.0],   # yuksek skor -> KAL
        [2, 0, 5.2, 5.0, 0.4, 50.0],    # dusuk skor, 0.2m -> DUS
        [3, 0, 20.0, 10.0, 0.8, 90.0],  # uzak -> KAL
    ])
    keep = inframe_dedup_mask(df, sep_m=0.7)
    assert keep[0] and keep[2] and not keep[1]


def test_far_apart_both_kept():
    # gercek iki oyuncu 3m -> ikisi de KAL (yanlis bastirma yok)
    df = _make([
        [1, 0, 5.0, 5.0, 0.9, 100.0],
        [2, 0, 8.0, 5.0, 0.9, 100.0],
    ])
    keep = inframe_dedup_mask(df, sep_m=0.7)
    assert keep.all()


def test_near_marking_above_threshold_kept():
    # 0.9m (yakin markaj) > 0.7m esik -> ikisi de KAL
    df = _make([
        [1, 0, 5.0, 5.0, 0.9, 100.0],
        [2, 0, 5.9, 5.0, 0.9, 100.0],
    ])
    keep = inframe_dedup_mask(df, sep_m=0.7)
    assert keep.all()


def test_per_frame_only():
    # ayni tid'in FARKLI karelerdeki tespitleri etkilenmez
    df = _make([
        [1, 0, 5.0, 5.0, 0.9, 100.0],
        [1, 1, 5.0, 5.0, 0.9, 100.0],
        [2, 0, 5.1, 5.0, 0.3, 40.0],   # frame0'da yakin -> DUS
    ])
    keep = inframe_dedup_mask(df, sep_m=0.7)
    assert keep[0] and keep[1] and not keep[2]


def test_nan_safe():
    df = _make([
        [1, 0, np.nan, np.nan, 0.9, 100.0],
        [2, 0, 5.0, 5.0, 0.9, 100.0],
    ])
    keep = inframe_dedup_mask(df, sep_m=0.7)
    assert keep[1]  # gecerli olan korunur; NaN cikmaz


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn(); print("ok", fn.__name__)
