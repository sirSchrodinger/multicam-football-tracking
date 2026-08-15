## tests/test_roster_stitch.py  — REAL-DATA acceptance test (raw/tracks_cankaya_cam2_clip2400.parquet + calib/cankaya_cam2_v2.json)
## Run: venv/bin/python -m pytest tests/test_roster_stitch.py -q
## All numeric targets below were MEASURED on the real parquet, not invented.

import os, numpy as np, pandas as pd, pytest
from pitch.homography import PitchHomography
from detect.track_stitch import stitch

ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACKS = os.path.join(ROOT, "raw/tracks_cankaya_cam2_clip2400.parquet")
CALIB  = os.path.join(ROOT, "calib/cankaya_cam2_v2.json")

@pytest.fixture(scope="module")
def res(tmp_path_factory):
    out = tmp_path_factory.mktemp("roster_out")
    # expected_players=14 declared (sosyalhalisaha knows the 7v7 booking); motion-only path
    return stitch(TRACKS, CALIB, str(out), expected_players=14)

# --- 1. zero-row-loss + ghost sentinel (honesty: nothing silently dropped) ---
def test_zero_row_loss_and_ghost_sentinel(res):
    src = pd.read_parquet(TRACKS)
    st  = res["stitched"] if "stitched" in res else pd.read_parquet(res["stitched_path"])
    assert len(st) == len(src)                       # no row dropped
    assert st["player_id"].notna().all()             # every row keyed
    assert "spurious_reason" in st.columns
    # ghosts are kept in the parquet but flagged -1, never deleted
    g = st[st["player_id"] == -1]
    assert (g["spurious_reason"].str.len() > 0).all()
    # each tid maps to exactly one player_id (partition, not split)
    assert (st.groupby("tid")["player_id"].nunique() == 1).all()

# --- 2. spurious removal is conservative and reasoned ---
def test_spurious_conservative_and_labeled(res):
    r = res["report"]
    assert 1 <= r["n_spurious"] <= 30               # measured ghost candidates ~9-28
    assert r["obs_lost_frac"] <= 0.02               # measured 0.0012 for n<=2 blips
    vocab = {"double_detection", "off_pitch_stationary", "orphan_blip", "static_foreign"}
    for s in r["spurious"]:
        assert s["reason"] in vocab                  # multi-criteria, no blanket-by-lifetime
    assert isinstance(r["spurious_by_reason"], dict)

# --- 3. hard floor is the HONEST survivor floor 15, not the ghost-inflated 16 ---
def test_hard_floor_is_15_not_ghost_inflated_16(res):
    r = res["report"]
    assert r["concurrency_raw_max"] == 16           # raw peak (frames 215,2531) = ghosts, diagnostic
    assert r["clique_floor"] == 15                  # true per-frame max over SURVIVORS
    assert r["clique_floor"] < r["concurrency_raw_max"]   # removing ghosts lowered the bound honestly

# --- 4. over-merge structurally impossible ---
def test_over_merge_zero(res):
    r = res["report"]
    assert r["temporal_violations"] == 0
    assert r.get("over_merge_violations", 0) == 0

# --- 5. cluster count never below floor, and improved over the old 18 ---
def test_never_below_floor_and_improved(res):
    r = res["report"]
    assert r["n_clusters"] >= r["clique_floor"]     # NEVER below floor (no distance lie)
    assert 15 <= r["n_clusters"] <= 18              # was 18; ghost removal + 1 merge -> <=17 expected
    assert r["n_clusters"] <= 18                    # strictly improved-or-equal, never worse

# --- 6. roster headline 14 from 3 independent signals, NOT forced ---
def test_roster_headline_14_not_forced(res):
    rost = res["report"]["roster"]
    assert rost["roster_on_field"] == 14
    assert rost["confidence"] == "high"
    assert rost["p95"] == 14 and rost["dedup_p99"] == 14 and rost["format_roster"] == 14
    assert rost["agreement"] is True
    # CRITICAL: headline (14) and hard floor (15) are SEPARATE honest fields, not collapsed
    assert res["report"]["clique_floor"] == 15
    assert res["report"]["clique_floor"] != rost["roster_on_field"]

# --- 7. format inference is aspect-based (no absolute-metre claim) ---
def test_format_aspect_7v7(res):
    rost = res["report"]["roster"]
    assert abs(rost["format_aspect"] - 1.889) < 0.02   # 34/18
    assert res["report"]["roster"]["format_roster"] == 14

# --- 8. player-keyed parquet excludes ghosts (downstream per-player stats clean) ---
def test_player_keyed_excludes_ghosts(res):
    pk = pd.read_parquet(res["player_keyed_path"])
    assert (pk["player_id"] >= 0).all()
    assert (pk["tid"] >= 0).all()                   # player-keyed tid := player_id

# --- 9. honesty: relative_m only, no fabricated metric, no metre lock ---
def test_relative_m_no_metric_claims(res):
    r = res["report"]
    assert r["unit"] == "relative_m"
    assert PitchHomography.load(CALIB).scale_anchor is None
    blob = str(r).lower()
    assert "sprint" not in blob and "accel" not in blob and "km/h" not in blob

# --- 10. distinct vs on-field reported separately (substitution rotation, e.g. pid16 n=912) ---
def test_distinct_vs_onfield_separately_reported(res):
    r = res["report"]
    assert "n_distinct_identities" in r
    assert "on_field_vs_distinct_note" in r
    assert r["n_distinct_identities"] >= r["roster"]["roster_on_field"]  # distinct may exceed headline, never forced down

# --- 11. residual under-merge still disclosed (honesty kept) ---
def test_residual_under_merge_reported(res):
    assert "residual_under_merge" in res["report"]
    assert "n_cluster_merges" in res["report"]      # frame-disjoint under-merge fix surfaced
