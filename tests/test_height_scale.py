#!/usr/bin/env python3
"""height_scale kabul testi — oyuncu-boyu metroloji, gercek Cankaya verisiyle.

Dogrular: f decompose makul; player_height_scale sanity (kamera 2-8m, alan makul,
band sonlu, c makul); rejection yolu (CV/kamera kapilari) calisir.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pitch.height_scale import (  # noqa: E402
    estimate_focal, decompose_ground_homography, player_height_scale,
)

CALIB = os.path.join(ROOT, "calib", "cankaya_cam2_v2.json")
TRACKS = os.path.join(ROOT, "raw", "tracks_cankaya_cam2_clip2400.parquet")
_REAL = os.path.exists(CALIB) and os.path.exists(TRACKS)


def _load_H():
    import json
    d = json.load(open(CALIB))
    return np.array(d["H_img2pitch"])


@pytest.mark.skipif(not _REAL, reason="gercek calib/parquet yok")
def test_focal_plausible():
    f = estimate_focal(_load_H())
    assert 400.0 < f < 4000.0, f"odak makul degil: {f}"


@pytest.mark.skipif(not _REAL, reason="gercek calib/parquet yok")
def test_decompose_camera_above_ground():
    f, R, t, C, lam = decompose_ground_homography(_load_H())
    assert C[2] > 0, "kamera zemin altinda cikti (isaret hatasi)"
    # R sutunlari makul birim-buyuklukte (tam ortonormal degil ama yakin)
    n1 = np.linalg.norm(R[:, 0]); n2 = np.linalg.norm(R[:, 1])
    assert 0.3 < n1 < 3.0 and 0.3 < n2 < 3.0


@pytest.mark.skipif(not _REAL, reason="gercek calib/parquet yok")
def test_player_height_scale_sane():
    r = player_height_scale(CALIB, TRACKS)
    assert r["ok"], f"olcek reddedildi: {r['reasons']}"
    assert 2.0 <= r["camera_height_m"] <= 8.0, r["camera_height_m"]
    assert 20.0 <= r["implied_field_m"]["L"] <= 60.0, r["implied_field_m"]
    assert 10.0 <= r["implied_field_m"]["W"] <= 35.0, r["implied_field_m"]
    assert 0.5 <= r["scale_factor"] <= 2.0, r["scale_factor"]
    assert r["height_cv"] < 0.20, r["height_cv"]
    assert 0.0 < r["band_pct"] < 35.0, r["band_pct"]
    assert r["n_detections"] > 1000


@pytest.mark.skipif(not _REAL, reason="gercek calib/parquet yok")
def test_height_assumption_scales_linearly():
    # varsayilan boyu degistirmek olcegi orantili degistirmeli (uydurma yok, fiziksel)
    r175 = player_height_scale(CALIB, TRACKS, assumed_height_m=1.75)
    r185 = player_height_scale(CALIB, TRACKS, assumed_height_m=1.85)
    ratio = r185["scale_factor"] / r175["scale_factor"]
    # scale_factor 4-ondalik yuvarlanir -> 1e-6 degil, yuvarlama-toleransi
    assert abs(ratio - 1.85 / 1.75) < 3e-3, ratio
    # medyan-boy (geometri) varsayimdan BAGIMSIZ olmali
    assert r185["median_height_relm"] == r175["median_height_relm"]


def _synthetic_scene(L=40.0, W=20.0, f=1100.0, cam_h=4.0, n=2500, seed=0):
    """Bilinen kamera + saha + 1.75m insanlar -> (calib_dict, df). Ground truth testi."""
    import cv2
    import pandas as pd
    cx, cy = 960.0, 540.0
    C = np.array([-3.0, W / 2, cam_h]); target = np.array([L / 2, W / 2, 0.0]); up = np.array([0, 0, 1.0])
    fwd = (target - C) / np.linalg.norm(target - C)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd]); t = -R @ C
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    P = K @ np.column_stack([R, t])

    def proj(Xw):
        h = (P @ np.column_stack([Xw, np.ones(len(Xw))]).T).T
        return h[:, :2] / h[:, 2:3]

    corners = np.array([[0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0.0]])
    H, _ = cv2.findHomography(proj(corners).astype(np.float64),
                             corners[:, :2].astype(np.float64), 0)
    rng = np.random.default_rng(seed)
    X = rng.uniform(2, L - 2, n); Y = rng.uniform(2, W - 2, n); Z = 1.75 + rng.normal(0, 0.07, n)
    foot = proj(np.column_stack([X, Y, np.zeros(n)]))
    head = proj(np.column_stack([X, Y, Z]))
    bh = foot[:, 1] - head[:, 1]
    df = pd.DataFrame(dict(tid=np.arange(n), frame=np.arange(n), t_sec=np.arange(n) / 25.0,
                           foot_x=foot[:, 0], foot_y=foot[:, 1], box_h=bh, box_w=bh / 2.5,
                           conf=np.full(n, 0.9), bottom_cropped=np.zeros(n, bool),
                           pitch_x=np.nan, pitch_y=np.nan, in_pitch=np.ones(n, bool)))
    calib = dict(H_img2pitch=H.tolist(), K=K.tolist(), dist=[0, 0, 0, 0, 0.0],
                 pitch_dims_m=dict(L=L, W=W))
    return calib, df, dict(L=L, W=W, cam_h=cam_h, f=f)


def test_synthetic_ground_truth_exact():
    """KUSURSUZ kalibrasyonda yontem GERCEK sahayi ~0 hatayla geri bulmali (matematik kaniti)."""
    import json
    import tempfile
    calib, df, gt = _synthetic_scene()
    with tempfile.TemporaryDirectory() as td:
        cp = os.path.join(td, "c.json"); tp = os.path.join(td, "t.parquet")
        with open(cp, "w") as fh:
            json.dump(calib, fh)
        df.to_parquet(tp)
        r = player_height_scale(cp, tp)
    assert r["ok"], r["reasons"]
    assert abs(r["implied_field_m"]["L"] - gt["L"]) / gt["L"] < 0.02, r["implied_field_m"]
    assert abs(r["implied_field_m"]["W"] - gt["W"]) / gt["W"] < 0.02, r["implied_field_m"]
    assert abs(r["camera_height_m"] - gt["cam_h"]) / gt["cam_h"] < 0.02, r["camera_height_m"]
    assert abs(r["focal_px"] - gt["f"]) / gt["f"] < 0.05, r["focal_px"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
