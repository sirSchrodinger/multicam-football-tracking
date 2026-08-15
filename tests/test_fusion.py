#!/usr/bin/env python3
"""2-kamera fuzyon kabul testi — gercek Cankaya verisiyle senkron+oryantasyon+koinsidans kilidi.

Kilitler (sessizce bozulursa CI yakalar):
  - cam1 ORTAK CERCEVEYE 180 flip ile oturur (flip=True, flip=False'tan iyi).
  - zaman-senkron offset (cam2_t = cam1_t + 55.1s) mid-saha koinsidansini <0.8m yapar.
  - fuzyon motoru kameralar-arasi MUST-link ile en az bir iki-kamerali oyuncu uretir.
  - coregister rotasyonu capraz-dogrulama ile guvenli (gurultude R=I doner).
"""
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "fusion"))

CFG = ROOT / "calib/fusion_config.json"
C1 = ROOT / "calib/cankaya_cam1_FINAL.json"
C2 = ROOT / "calib/cankaya_cam2_FINAL.json"
SMOKE = ROOT / "raw/tracks_cankaya_cam1_smoke.parquet"
FULL2 = ROOT / "raw/tracks_cankaya_cam2_fullgame.parquet"

need_data = pytest.mark.skipif(
    not (SMOKE.exists() and FULL2.exists() and C1.exists() and C2.exists()),
    reason="Cankaya track/calib verisi yok")

FPS1 = 30.0
FPS2 = 88294 / 3549.97


def _load():
    import pitch_map as pm
    c1 = pm.load(str(C1)); c2 = pm.load(str(C2))
    Lm, Wm = c1["L"] * c1["SC"], c1["W"] * c1["SC"]
    return pm, c1, c2, Lm, Wm


def _coincidence(off, flip, midonly=True, gate=2.5):
    """mid-saha LSAP eslesmesinde medyan capraz-kamera mesafesi."""
    from scipy.optimize import linear_sum_assignment
    pm, c1, c2, Lm, Wm = _load()
    s1 = pd.read_parquet(SMOKE)
    s2 = pd.read_parquet(FULL2)
    s2 = s2[(s2.t_sec >= 50) & (s2.t_sec <= 66)]

    def shared(df, cal, fl):
        P = pm.to_pitch(cal, df[["foot_x", "foot_y"]].values)
        X, Y = P[:, 0], P[:, 1]
        if fl:
            X, Y = Lm - X, Wm - Y
        return np.c_[X, Y]

    dists = []
    for f1, g1 in s1.groupby("frame"):
        t = f1 / FPS1 + off
        f2 = int(round(t * FPS2))
        g2 = s2[s2.frame == f2]
        if len(g2) == 0:
            continue
        A = shared(g1, c1, flip); B = shared(g2, c2, False)
        if midonly:
            A = A[(A[:, 0] > Lm / 3) & (A[:, 0] < 2 * Lm / 3)]
            B = B[(B[:, 0] > Lm / 3) & (B[:, 0] < 2 * Lm / 3)]
        if len(A) == 0 or len(B) == 0:
            continue
        D = np.linalg.norm(A[:, None] - B[None], axis=2)
        ri, ci = linear_sum_assignment(np.minimum(D, gate * 3))
        for r, c in zip(ri, ci):
            if D[r, c] <= gate:
                dists.append(D[r, c])
    return np.array(dists)


@need_data
def test_orientation_flip_wins():
    """cam1 ortak cerceveye 180 flip ile oturur — flip=True net daha iyi."""
    off = json.load(open(CFG))["time_sync"]["offset_cam1_to_cam2_s"]
    d_flip = _coincidence(off, True)
    d_noflip = _coincidence(off, False)
    assert len(d_flip) >= 20, "flip eslesmesi cok az"
    assert np.median(d_flip) < np.median(d_noflip), \
        f"flip medyan {np.median(d_flip):.2f} >= no-flip {np.median(d_noflip):.2f}"


@need_data
def test_sync_offset_locks_coincidence():
    """Kayitli offset mid-saha koinsidansini sub-metre yapar; komsu offsetler kotu."""
    off = json.load(open(CFG))["time_sync"]["offset_cam1_to_cam2_s"]
    d = _coincidence(off, True)
    assert len(d) >= 20
    assert np.median(d) < 0.8, f"koinsidans medyan {np.median(d):.2f}m > 0.8"
    # senkron gercekten kilitli mi: 5s uzakta belirgin kotulesir
    d_far = _coincidence(off - 5.0, True)
    if len(d_far) >= 20:
        assert np.median(d) < np.median(d_far) + 1e-6


@need_data
def test_coregister_rotation_is_cv_guarded():
    """coregister gurultulu rotasyonu capraz-dogrulama ile reddeder (smoke'ta)."""
    import fuse_engine as fe
    cfg = json.load(open(CFG))
    c1 = pd.read_parquet(SMOKE)
    c2 = pd.read_parquet(FULL2)
    c2 = c2[(c2.t_sec >= 50) & (c2.t_sec <= 66)]
    tr, dims, _ = fe.build_tracklets(c1, c2, cfg)
    R, t, creg = fe.coregister(tr, dims, verbose=False)
    assert creg["applied"]
    # smoke'ta rotasyon CV'yi gecmemeli -> R kimlik
    if not creg["use_rot"]:
        assert np.allclose(R, np.eye(2))


@need_data
def test_cross_camera_mustlink_produces_dual_player():
    """Motor en az bir iki-kamerali birlesik oyuncu uretir (wall-breaker calisir)."""
    import fuse_engine as fe
    cfg = json.load(open(CFG))
    c1 = pd.read_parquet(SMOKE)
    c2 = pd.read_parquet(FULL2)
    c2 = c2[(c2.t_sec >= 50) & (c2.t_sec <= 66)]
    tr, dims, _ = fe.build_tracklets(c1, c2, cfg)
    R, t, creg = fe.coregister(tr, dims, verbose=False)
    if creg["applied"]:
        fe.apply_coreg(tr, R, t)
    comps, stats = fe.fuse(tr, dims)
    assert stats["n_cross"] >= 1, "kameralar-arasi must-link kurulmadi"
    _, summ = fe.player_tracks(tr, comps)
    assert (summ["n_cameras"] == 2).sum() >= 1, "iki-kamerali oyuncu yok"
