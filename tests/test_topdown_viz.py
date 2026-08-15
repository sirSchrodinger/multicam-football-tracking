import os, struct
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKS = os.path.join(REPO, "raw", "tracks_cankaya_cam2_clip2400.parquet")
GOOD   = os.path.join(REPO, "calib", "cankaya_cam2_v2.json")   # 9.87px, gate PASS
BAD    = os.path.join(REPO, "calib", "cankaya_cam2.json")      # 65px, gate REJECT

pytestmark = pytest.mark.skipif(
    not (os.path.exists(TRACKS) and os.path.exists(GOOD) and os.path.exists(BAD)),
    reason="real-data fixtures missing")

def _png_size(path):
    with open(path, "rb") as f:
        head = f.read(33)
    assert head[:8] == b"\x89PNG\r\n\x1a\n", "not a PNG"
    w, h = struct.unpack(">II", head[16:24])
    return w, h

def test_good_calib_produces_one_honest_figure(tmp_path):
    from stats.topdown_viz import render_report_figure
    out = tmp_path / "report.png"
    p = render_report_figure(TRACKS, GOOD, str(out))
    # 1) single shareable artifact actually written, non-trivial size
    assert os.path.exists(p) and os.path.getsize(p) > 50_000
    w, h = _png_size(p)
    assert w > 1500 and h > 1500          # multi-panel report figure
    # 2) the audited JSON was emitted alongside (single source of honesty)
    rep_json = tmp_path / "report_topdown.json"
    assert rep_json.exists()
    import json
    rep = json.loads(rep_json.read_text())
    # 3) honesty invariants the figure text is sourced from
    assert rep["coordinate"]["unit"] == "relative_m"   # NEVER meters
    assert rep["coordinate"]["scale_anchor"] in (None,)  # scale lock OFF
    assert rep["honesty"]["calib_gate"]["ok"] is True
    assert "14" in rep["honesty"]["identity"]          # tracklets != ~14 players caveat
    assert rep["honesty"]["acceleration"].startswith("NOT")

def test_correct_fps_via_load_tracks():
    # regression: must load via stats_report.load_tracks (fps 24.87), not bare read_parquet (25.0)
    import stats_report as SR
    df = SR.load_tracks(TRACKS)
    fps = float(df.attrs["meta"]["fps"])
    assert abs(fps - 24.872) < 0.01

def test_broken_calib_is_fail_closed(tmp_path):
    # 65px H must NOT yield a perspective-biased PNG dressed as top-down
    from stats.topdown_viz import render_report_figure
    out = tmp_path / "should_not_exist.png"
    with pytest.raises(RuntimeError):
        render_report_figure(TRACKS, BAD, str(out))
    assert not out.exists()

def test_trajectory_confidence_is_structural():
    # Design #3 honesty: far-zone (pixel-poor) confidence < near-zone -> faded trails
    import stats_report as SR
    from pitch.homography import PitchHomography
    from stats.topdown_stats import prepare, accept_calib_qa
    df = SR.load_tracks(TRACKS)
    homo = PitchHomography.load(GOOD)
    assert accept_calib_qa(homo._qa)[0]
    prep = prepare(df, homo, float(df.attrs["meta"]["fps"]))
    near = prep.w_conf[prep.zone == 0]
    far  = prep.w_conf[prep.zone == 2]
    assert prep.w_conf.min() >= 0.0 and prep.w_conf.max() <= 1.0
    assert np.nanmedian(far) < np.nanmedian(near)   # far end honestly lower-confidence
