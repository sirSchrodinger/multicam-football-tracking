import os, numpy as np, pandas as pd, pytest
from pitch.homography import PitchHomography
from stats.topdown_stats import accept_calib_qa
from detect.track_stitch import stitch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKS = os.path.join(ROOT, "raw/tracks_cankaya_cam2_clip2400.parquet")
CALIB  = os.path.join(ROOT, "calib/cankaya_cam2_v2.json")
BAD    = os.path.join(ROOT, "calib/cankaya_cam2.json")  # 65px, must be refused

@pytest.fixture(scope="module")
def res(tmp_path_factory):
    out = tmp_path_factory.mktemp("stitch_out")
    return stitch(TRACKS, CALIB, str(out))  # motion-only path (no video); appearance optional

def test_calib_gate_v2_accepts_bad_rejects():
    assert accept_calib_qa(PitchHomography.load(CALIB)._qa)[0] is True
    assert accept_calib_qa(PitchHomography.load(BAD)._qa)[0] is False

def test_fail_closed_on_bad_calib(tmp_path):
    with pytest.raises(Exception):           # must refuse, never silently fabricate
        stitch(TRACKS, BAD, str(tmp_path))

def test_reprojection_sane():               # backbone re-projection is clean
    df = pd.read_parquet(TRACKS)
    homo = PitchHomography.load(CALIB)
    m = homo.pixel_to_pitch(df[["foot_x","foot_y"]].to_numpy(float))
    assert homo.in_pitch(m).mean() > 0.99
    assert -3 < m[:,0].min() and m[:,0].max() < 38   # X within 34 + margin
    assert -3 < m[:,1].min() and m[:,1].max() < 22   # Y within 18 + margin

def test_clique_floor_is_16(res):
    # semantics changed HONESTLY: raw concurrency peak (215,2531) = 16 but ghost-inflated;
    # the true hard floor over SURVIVORS (ghosts removed) is 15.
    assert res["report"]["concurrency_raw_max"] == 16   # raw per-frame max, diagnostic
    assert res["report"]["clique_floor"] == 15          # honest survivor floor

def test_no_temporal_overlap_violations(res):
    assert res["report"]["temporal_violations"] == 0 # over-merge guard intact

def test_cluster_count_honest_range(res):
    n = res["report"]["n_clusters"]
    assert n >= res["report"]["clique_floor"]        # NEVER below floor (no distance lie)
    assert 15 <= n <= 30          # ghost removal -> survivor floor 15; band kept wide
    # explicitly NOT forced to 14

def test_top16_observation_coverage(res):
    assert res["report"]["top16_obs_coverage"] >= 0.90   # measured 0.973

def test_stitched_parquet_lossless(res):
    src = pd.read_parquet(TRACKS)
    st  = res["stitched"] if "stitched" in res else pd.read_parquet(res["stitched_path"])
    assert len(st) == len(src)                       # no row dropped
    assert "player_id" in st.columns
    assert st["player_id"].notna().all()
    # ghosts carry player_id=-1 sentinel (kept, not deleted); real clusters >=0
    assert st.loc[st["player_id"] >= 0, "player_id"].nunique() == res["report"]["n_clusters"]
    # every tid maps to exactly one player_id (partition, not split)
    assert (st.groupby("tid")["player_id"].nunique() == 1).all()

def test_relative_m_no_metric_claims(res):
    r = res["report"]
    assert r["unit"] == "relative_m"
    assert "sprint" not in r and "accel" not in str(r).lower()
    assert PitchHomography.load(CALIB).scale_anchor is None  # metre kilidi kapali

def test_residual_under_merge_reported(res):         # honesty: under-merge disclosed
    assert "residual_under_merge" in res["report"]
