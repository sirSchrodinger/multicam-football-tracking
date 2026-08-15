#!/usr/bin/env python3
"""recall_qc.py kabul testi — GPU-suz. venv/bin/python detect/test_recall_qc.py

Gercek parquet (raw/tracks_cankaya_cam2_clip2400.parquet) + gercek calib
(calib/cankaya_cam2_v2.json) uzerinde; recovery mantigi sahte predict_fn ile
(GPU/torch yok). Her assert basarisizsa AssertionError ile patlar.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detect.recall_qc import (  # noqa: E402
    load_calibrated_homography, parity_qc_from_parquet, derive_qc_columns,
    zone_of, FarBandRecovery, THRESH, EXPECTED_PLAYERS,
)

PARQUET = ROOT / "raw/tracks_cankaya_cam2_clip2400.parquet"
CALIB_GOOD = ROOT / "calib/cankaya_cam2_v2.json"
CALIB_BAD = ROOT / "calib/cankaya_cam2.json"


def test_calib_qa_gate():
    """Bozuk kanonik calib REDDEDILIR, saglikli v2 KABUL edilir."""
    homo_bad, qa_bad = load_calibrated_homography(str(CALIB_BAD))
    assert homo_bad is None, "bozuk calib (65px) reddedilmeliydi"
    assert qa_bad.get("rejected"), qa_bad
    homo_ok, qa_ok = load_calibrated_homography(str(CALIB_GOOD))
    assert homo_ok is not None, "saglikli v2 calib kabul edilmeliydi"
    assert qa_ok["median_px"] < 20.0, qa_ok
    print("OK calib QA gate: bad reject (median=%.1f) / good accept (median=%.1f)"
          % (qa_bad["median_px"], qa_ok["median_px"]))


def test_parity_qc_numbers():
    """GPU-suz parite QC olculmus dagilimi uretir."""
    df = pd.read_parquet(PARQUET)
    r = parity_qc_from_parquet(df)
    assert r["n_frames"] == 2985, r["n_frames"]
    assert 14.0 <= r["pct_complete"] <= 17.0, r["pct_complete"]
    assert 80.0 <= r["pct_deficit"] <= 86.0, r["pct_deficit"]
    assert r["deficit_player_frames"] == 6086, r["deficit_player_frames"]
    # deficit flag'leri far_count tasimali; surplus flag'leri 'FP' notu
    defs = [f for f in r["flags"] if f["state"] == "deficit"]
    surs = [f for f in r["flags"] if f["state"] == "surplus"]
    assert defs and all("far_count" in f for f in defs)
    assert all("FP" in f["note"] for f in surs)
    print("OK parity QC: complete=%.1f%% deficit=%.1f%% deficit_pf=%d (def=%d sur=%d)"
          % (r["pct_complete"], r["pct_deficit"], r["deficit_player_frames"],
             len(defs), len(surs)))


def test_derive_columns_schema():
    """zone/recovered/low_conf turetilir; conf<THRESH -> recovered; far+dusuk -> low_conf."""
    df = pd.read_parquet(PARQUET)
    out = derive_qc_columns(df)
    for c in ("zone", "recovered", "low_conf"):
        assert c in out.columns, c
    # recovered tam olarak conf<THRESH (parquet'te recovered kolonu yok)
    assert (out["recovered"].to_numpy() == (df["conf"].to_numpy() < THRESH)).all()
    # low_conf, recovered'i kapsar (superset)
    assert out["low_conf"][out["recovered"]].all()
    # far zone (0) gercekten en kucuk kutular
    z = out["zone"].to_numpy()
    assert out[z == 0]["box_h"].median() < out[z == 2]["box_h"].median()
    print("OK derive cols: recovered=%d low_conf=%d far_box_h=%.0f near_box_h=%.0f"
          % (int(out["recovered"].sum()), int(out["low_conf"].sum()),
             out[z == 0]["box_h"].median(), out[z == 2]["box_h"].median()))


def _fake_dets(xyxy, conf):
    import supervision as sv
    xyxy = np.asarray(xyxy, float).reshape(-1, 4)
    return sv.Detections(xyxy=xyxy, confidence=np.asarray(conf, float),
                         class_id=np.zeros(len(xyxy), int))


def test_recovery_identity_when_not_tiling():
    """should_tile False iken (frame_idx % tile_every != 0) ciktinin kutu sayisi degismez."""
    homo, _ = load_calibrated_homography(str(CALIB_GOOD))
    calls = {"n": 0}

    def predict_fn(img_rgb, thr):
        calls["n"] += 1
        return np.empty((0, 4)), np.empty((0,))

    rec = FarBandRecovery(predict_fn, homo, band=(150, 360), tile_every=6)
    base = _fake_dets([[100, 100, 130, 200]] * 8,
                      [0.9] * 8)  # deficit (8<14)
    out = rec.process(frame_idx=1, t_sec=0.04, frame_bgr=np.zeros((1080, 1920, 3), np.uint8),
                      base_dets=base)
    assert len(out) == 8, "tile_every disinda recovery olmamali"
    assert calls["n"] == 0, "predict cagrilmamaliydi"
    assert out.data["recovered"].sum() == 0
    print("OK identity-when-not-tiling: base 8 -> 8, predict calls=0")


def test_recovery_budget_and_dedup():
    """Deficit frame'de uzak adaylar in_pitch+dedup'tan gecip eklenir; full frame'de eklenmez."""
    homo, _ = load_calibrated_homography(str(CALIB_GOOD))
    # far-band crop (150..360) icinde, in_pitch olacak ayak noktalari uret.
    # upscale=2 oldugundan predict_fn crop+upscale koordinati doner; recall_qc
    # bunu /2 + y0 ile tam-frame'e cevirir. Biz tam-frame hedef noktayi secip
    # crop-upscale koordinatina ters cevirelim.
    band = (150, 360)
    up = 2.0
    # tam-frame uzak ayak noktalari (mid-x, foot_y band icinde)
    targets = [(900, 300), (1000, 320), (700, 280)]
    def to_crop_coords(fx, fy):
        cy = (fy - band[0]) * up
        cx = fx * up
        return [cx - 12, cy - 120, cx + 12, cy]  # kutu (foot=alt orta); /up=60px > MIN_H

    def predict_fn(img_rgb, thr):
        boxes = [to_crop_coords(fx, fy) for fx, fy in targets]
        return np.array(boxes, float), np.array([0.5, 0.45, 0.42])

    rec = FarBandRecovery(predict_fn, homo, band=band, up=up, tile_every=1)

    # deficit frame: base 8 in-pitch oyuncu -> budget=6 -> 3 uzak aday eklenir
    bx = [[800 + 20 * i, 380, 820 + 20 * i, 470] for i in range(8)]
    base = _fake_dets(bx, [0.8] * 8)
    out = rec.process(0, 0.0, np.zeros((1080, 1920, 3), np.uint8), base)
    n_rec = int(out.data["recovered"].sum())
    assert n_rec >= 1, f"deficit frame'de uzak aday eklenmeliydi (n_rec={n_rec})"
    assert len(out) == 8 + n_rec
    # recovered olanlar base'in conf floor'unun ALTINDA (presence-only sinyali)
    rec_conf = out.confidence[out.data["recovered"]]
    assert (rec_conf < THRESH).all() or (rec_conf < 0.6).all()

    # DEDUP: ayni adaylari TEKRAR ver ama base zaten o noktalarda -> 0 eklenmeli
    base_full = _fake_dets(
        bx + [to_crop_coords(fx, fy) for fx, fy in targets],  # not: bunlar crop-coord, sadece dedup testte base pitch'i lazim
        [0.8] * 8 + [0.8] * 3)
    # daha temiz dedup testi: base'e tam-frame uzak noktalari koy
    bx2 = bx + [[fx - 8, fy - 40, fx + 8, fy] for fx, fy in targets]
    base2 = _fake_dets(bx2, [0.8] * 11)
    rec2 = FarBandRecovery(predict_fn, homo, band=band, up=up, tile_every=1)
    out2 = rec2.process(0, 0.0, np.zeros((1080, 1920, 3), np.uint8), base2)
    n_rec2 = int(out2.data["recovered"].sum())
    assert n_rec2 == 0, f"base zaten o noktalarda -> dedup 0 eklemeli (n_rec2={n_rec2})"
    print("OK budget+dedup: deficit base8 -> +%d recovered; dup base11 -> +%d" % (n_rec, n_rec2))


def test_data_carry_through_bytetrack():
    """sv 0.28 data['recovered'] with_nms + ByteTrack uzerinden korunur (entegrasyon on-kosulu)."""
    import supervision as sv
    d = sv.Detections(xyxy=np.array([[10, 10, 30, 70], [200, 50, 220, 120]], float),
                      confidence=np.array([0.9, 0.5]),
                      class_id=np.zeros(2, int),
                      data={"recovered": np.array([False, True])})
    d = d.with_nms(threshold=0.6, class_agnostic=True)
    tr = sv.ByteTrack(frame_rate=25, track_activation_threshold=0.25)
    d = tr.update_with_detections(d)
    assert "recovered" in d.data, "data['recovered'] ByteTrack sonrasi kaybolmamali"
    print("OK data carry: recovered survives nms+ByteTrack ->", d.data["recovered"].tolist())


if __name__ == "__main__":
    fns = [test_calib_qa_gate, test_parity_qc_numbers, test_derive_columns_schema,
           test_recovery_identity_when_not_tiling, test_recovery_budget_and_dedup,
           test_data_carry_through_bytetrack]
    fails = 0
    for fn in fns:
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            fails += 1
            print(f"FAIL {fn.__name__}: {e}")
    print(f"\n{len(fns)-fails}/{len(fns)} test gecti")
    sys.exit(1 if fails else 0)
