#!/usr/bin/env python3
"""Taraftar/oyuncu-disi filtresi (roster.flag_spurious RULE-4/5) — sentetik kanit.

Saha-DISI kenar-taraftarlari (STATIK ve GEZINEN ikisi de) isaretlenmeli; saha-ICI
hareketli oyuncular ve cizgi-dibindeki kaleci KORUNMALI. Gercek klipte regresyon yok.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CALIB = os.path.join(ROOT, "calib", "cankaya_cam2_v2.json")
_REAL = os.path.exists(CALIB)


def _build(homo, specs, fps=25.0):
    """specs: list of dict(kind, path_fn). path_fn(k)->(X,Y) pitch. -> df, summaries, pm."""
    import pandas as pd
    from detect import track_stitch as TS
    rows = []
    for tid, sp in enumerate(specs):
        n = sp["n"]
        for k in range(n):
            X, Y = sp["path"](k)
            u, v = homo.pitch_to_pixel(np.array([[X, Y]]))[0]
            rows.append(dict(tid=tid, frame=k, t_sec=k / fps,
                             foot_x=float(u), foot_y=float(v),
                             box_h=80.0, box_w=32.0, conf=0.9,
                             bottom_cropped=False, pitch_x=np.nan, pitch_y=np.nan,
                             in_pitch=False))
    df = pd.DataFrame(rows)
    summaries, _ = TS._summaries(df, homo, 5)
    pm = homo.pixel_to_pitch(df[["foot_x", "foot_y"]].to_numpy(float))
    return df, summaries, pm


@pytest.mark.skipif(not _REAL, reason="v2 calib yok")
def test_spectators_flagged_players_kept():
    from pitch.homography import PitchHomography
    from detect import roster
    homo = PitchHomography.load(CALIB)
    L, W = homo._dims_m()
    rng = np.random.default_rng(0)

    specs = []
    # 8 saha-ICI hareketli oyuncu (random-walk, saha icinde kalir)
    for p in range(8):
        x0 = rng.uniform(5, L - 5); y0 = rng.uniform(4, W - 4)
        walk = rng.normal(0, 0.15, (120, 2)).cumsum(0)
        def mk(x0, y0, walk):
            return lambda k: (np.clip(x0 + walk[k, 0], 1, L - 1),
                              np.clip(y0 + walk[k, 1], 1, W - 1))
        specs.append(dict(kind="player", n=120, path=mk(x0, y0, walk)))
    # cizgi-dibi STATIK kaleci (saha ICI, out_dist~0) -> KORUNMALI
    specs.append(dict(kind="keeper", n=120,
                      path=lambda k: (0.5, W / 2 + 0.05 * np.sin(k / 10))))
    # STATIK taraftar (saha DISI, yan) -> RULE-4/5 ile isaretlenmeli
    specs.append(dict(kind="spec_static", n=120,
                      path=lambda k: (12.0, W + 3.0 + 0.05 * np.sin(k / 8))))
    # GEZINEN taraftar (saha DISI yan boyunca yuruyor) -> RULE-5 ile isaretlenmeli
    specs.append(dict(kind="spec_pacing", n=120,
                      path=lambda k: (5.0 + 0.12 * k, W + 2.6)))

    df, summaries, pm = _build(homo, specs)
    flags = roster.flag_spurious(df, summaries, pm, homo)

    kinds = [s["kind"] for s in specs]
    spec_tids = [i for i, k in enumerate(kinds) if k.startswith("spec_")]
    player_tids = [i for i, k in enumerate(kinds) if k in ("player", "keeper")]

    for t in spec_tids:
        assert t in flags, f"taraftar tid={t} ({kinds[t]}) ISARETLENMEDI"
    for t in player_tids:
        assert t not in flags, f"oyuncu/kaleci tid={t} ({kinds[t]}) YANLIS isaretlendi: {flags.get(t)}"


@pytest.mark.skipif(
    not (_REAL and os.path.exists(
        os.path.join(ROOT, "raw", "tracks_cankaya_cam2_clip2400.parquet"))),
    reason="gercek parquet yok")
def test_real_clip_no_player_dropped_as_spectator():
    """Gercek klipte (taraftarsiz) yeni RULE-5 hicbir gercek oyuncuyu off_field SILMEMELI."""
    import pandas as pd
    from pitch.homography import PitchHomography
    from detect import track_stitch as TS, roster
    homo = PitchHomography.load(CALIB)
    df = pd.read_parquet(os.path.join(ROOT, "raw", "tracks_cankaya_cam2_clip2400.parquet"))
    summaries, _ = TS._summaries(df, homo, 5)
    pm = homo.pixel_to_pitch(df[["foot_x", "foot_y"]].to_numpy(float))
    flags = roster.flag_spurious(df, summaries, pm, homo)
    off = [t for t, r in flags.items() if r == "off_field_persistent"]
    # uzun-yasamli (gercek oyuncu olabilecek) hicbir track off_field ile silinmemeli
    life = {s["tid"]: s["n"] for s in summaries}
    long_off = [t for t in off if life.get(t, 0) > 200]
    assert not long_off, f"uzun-track'ler off_field silindi (regresyon): {long_off}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
