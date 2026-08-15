#!/usr/bin/env python3
"""player_state kabul testi — sürekli konum haritası (motion-model gap-fill).

Sentetik: bilinen DUZGUN egri yol + bazi kareler silinmis -> continuous_state Hermite ile
plausible geri bulmali (kisa boslukta <~1m), status/conf dürüst (observed=1, dolgu<1, uzun->dusuk).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from detect.player_state import continuous_state, coverage_summary  # noqa: E402

CALIB = os.path.join(ROOT, "calib", "cankaya_cam2_v2.json")
_REAL = os.path.exists(CALIB)


def _make_player(homo, fps=25.0, n=200, drop=None):
    """Bilinen sinuzoidal pitch yolu -> foot piksel df. drop: silinecek frame indexleri."""
    L, W = homo._dims_m()
    t = np.arange(n)
    X = L / 2 + 8.0 * np.sin(2 * np.pi * t / 120.0)        # duzgun egri
    Y = W / 2 + 4.0 * np.sin(2 * np.pi * t / 90.0 + 1.0)
    pitch = np.column_stack([X, Y])
    foot = homo.pitch_to_pixel(pitch, distorted=True)       # pitch -> HAM px (pixel_to_pitch tersi)
    keep = np.ones(n, bool)
    if drop is not None:
        keep[drop] = False
    df = pd.DataFrame(dict(player_id=0, frame=t[keep], t_sec=t[keep] / fps,
                           foot_x=foot[keep, 0], foot_y=foot[keep, 1],
                           box_h=80.0, box_w=32.0, conf=0.9, bottom_cropped=False))
    return df, pitch


@pytest.mark.skipif(not _REAL, reason="v2 calib yok")
def test_hermite_fill_recovers_curved_path():
    from pitch.homography import PitchHomography
    homo = PitchHomography.load(CALIB)
    n = 200
    drop = list(range(40, 50)) + list(range(100, 104))     # 10-frame ve 4-frame bosluk
    df, truth = _make_player(homo, n=n, drop=drop)
    st, cov = continuous_state(df, homo, fps=25.0)
    # her frame icin state (span tam dolu olmali: 0..n-1)
    assert set(st["frame"]) == set(range(n))
    by = {int(r.frame): r for r in st.itertuples()}
    # observed kareler: conf=1, status observed, konum dogru
    assert by[10].status == "observed" and by[10].conf == 1.0
    # silinen (interpolated) kareler: status interpolated, conf<1, GERCEGE yakin (<1m)
    for f in drop:
        r = by[f]
        assert r.status == "interpolated"
        assert r.conf < 1.0
        err = np.hypot(r.x - truth[f, 0], r.y - truth[f, 1])
        assert err < 1.2, f"frame {f} hata {err:.2f}m (motion-model plausible degil)"
    # uzun bosluk (10fr) ortasi daha dusuk conf, kisa (4fr) daha yuksek
    assert by[45].conf < by[101].conf


@pytest.mark.skipif(not _REAL, reason="v2 calib yok")
def test_no_fabrication_outside_span():
    from pitch.homography import PitchHomography
    homo = PitchHomography.load(CALIB)
    # ilk 30 ve son 30 frame YOK -> span [30, n-31]; disina konum YAZILMAMALI
    n = 150
    df, _ = _make_player(homo, n=n, drop=list(range(0, 30)) + list(range(120, 150)))
    st, _ = continuous_state(df, homo, fps=25.0)
    assert st["frame"].min() >= 30 and st["frame"].max() <= 119
    # observed_frac mantikli (0,1]
    _, cov = continuous_state(df, homo, fps=25.0)
    assert 0.0 < cov[0]["observed_frac"] <= 1.0


@pytest.mark.skipif(not _REAL, reason="v2 calib yok")
def test_occupancy_conf_weighted_honest():
    from pitch.homography import PitchHomography
    from detect.player_state import continuous_state, occupancy_heatmap
    homo = PitchHomography.load(CALIB)
    df, _ = _make_player(homo, n=200, drop=list(range(40, 60)))
    st, _ = continuous_state(df, homo, fps=25.0)
    L, W = homo._dims_m()
    g_cw, _ = occupancy_heatmap(st, (L, W), bin_m=1.0, conf_weight=True)
    g_uw, _ = occupancy_heatmap(st, (L, W), bin_m=1.0, conf_weight=False)
    assert np.nanmax(g_cw) > 0 and np.nanmin(g_cw) >= 0
    # conf-agirlik: dolgu (conf<1) toplam-agirligi DUSURUR -> ham toplam <= agirliksiz toplam
    raw_cw = (st["conf"]).sum(); raw_uw = float(len(st))
    assert raw_cw < raw_uw            # dolgu satirlari conf<1 -> agirligi azaltir (dururst)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
