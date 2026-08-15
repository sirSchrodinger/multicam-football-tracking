#!/usr/bin/env python3
"""REAL-DATA kabul testi: detect/team_split.assign_teams.

Sentetik fixture YOK. Gercek Cankaya cam2 clip2400 verisi + v2 calib + video uzerinde
calisir. Repo kokunden: venv/bin/python -m pytest detect/test_team_split.py -q
"""
import json
import os

import pandas as pd
import pytest

from detect.team_split import assign_teams

PLAYER = "stats_out/pipeline_demo/tracks_cankaya_cam2_clip2400_player.parquet"
CALIB_GOOD = "calib/cankaya_cam2_v2.json"
CALIB_BAD = "calib/cankaya_cam2.json"
VIDEO = "raw/cankaya_cam2_clip2400.mp4"
STITCH_REPORT = ("stats_out/pipeline_demo/"
                 "tracks_cankaya_cam2_clip2400_stitch_report.json")

MIN_CONF = 0.35


@pytest.fixture(scope="module")
def res(tmp_path_factory):
    out = str(tmp_path_factory.mktemp("team_out"))
    return assign_teams(PLAYER, CALIB_GOOD, out, VIDEO, min_conf=MIN_CONF), out


def test_1_returns_dict_with_keys(res):
    r, _ = res
    for k in ("report", "mapping", "team_parquet_path", "report_path"):
        assert k in r, f"eksik anahtar: {k}"
    assert os.path.exists(r["team_parquet_path"])
    assert os.path.exists(r["report_path"])


def test_2_all_players_assigned_both_teams_nonempty(res):
    r, _ = res
    df_in = pd.read_parquet(PLAYER)
    pids = sorted(int(p) for p in df_in["player_id"].unique())
    assert len(pids) == 18
    mapping = r["mapping"]
    for p in pids:
        assert int(mapping[p]) in (0, 1), f"player {p} team_id gecersiz"
    sizes = r["report"]["team_sizes"]
    s0, s1 = int(sizes["0"]) if "0" in sizes else sizes[0], \
        int(sizes["1"]) if "1" in sizes else sizes[1]
    assert s0 > 0 and s1 > 0, "bir takim bos"
    # fixed-yellow-band 16-vs-2 hatasina karsi guard; data-driven 2-means dengeli ayirir
    assert min(s0, s1) >= 3, f"asiri-dengesiz split {s0}-{s1}"


def test_3_report_method_and_keys(res):
    r, _ = res
    rep = r["report"]
    assert rep["method_primary"] == "jersey_color_kmeans2_data_driven"
    for k in ("balance_ok", "balance_note", "proxy_pass_intra_rate",
              "proxy_pass_total", "graph_used", "color_separation_confidence",
              "per_player", "caveats"):
        assert k in rep, f"raporda eksik anahtar: {k}"
    assert isinstance(rep["per_player"], list) and len(rep["per_player"]) == 18
    assert isinstance(rep["caveats"], list) and len(rep["caveats"]) >= 1


def test_4_additive_non_destructive(res):
    r, _ = res
    out_df = pd.read_parquet(r["team_parquet_path"])
    in_df = pd.read_parquet(PLAYER)
    assert "team_id" in out_df.columns
    assert "team_conf" in out_df.columns
    assert out_df["team_id"].dtype == "int64"
    assert out_df["team_conf"].dtype == "float32"
    assert len(out_df) == len(in_df), "satir sayisi degisti"
    assert (out_df["tid"] == out_df["player_id"]).all(), "tid==player_id bozuldu"
    for c in in_df.columns:
        assert c in out_df.columns, f"orijinal kolon kayboldu: {c}"


def test_5_honesty_no_forced_balance_and_residual_flagged(res):
    r, _ = res
    rep = r["report"]
    sizes = rep["team_sizes"]
    s0 = int(sizes["0"]) if "0" in sizes else sizes[0]
    s1 = int(sizes["1"]) if "1" in sizes else sizes[1]
    balanced = abs(s0 - s1) <= 2
    # balance_note is None iff balanced; balance_ok matches
    assert (rep["balance_note"] is None) == balanced
    assert bool(rep["balance_ok"]) == balanced
    # residual_under_merge oyunculari dusuk-guven (flagged VEYA conf<min_conf)
    sr = json.load(open(STITCH_REPORT))
    residual = [int(x["player_id"]) for x in sr.get("residual_under_merge", [])]
    assert len(residual) == 2
    conf = r["conf"]
    flagged = set(rep["flagged_low_conf"])
    for p in residual:
        assert (p in flagged) or (float(conf[p]) < MIN_CONF), \
            f"residual oyuncu {p} dusuk-guven isaretlenmedi"


def test_6_fail_closed_bad_calib(tmp_path):
    out = str(tmp_path / "bad")
    with pytest.raises(ValueError) as ei:
        assign_teams(PLAYER, CALIB_BAD, out, VIDEO)
    msg = str(ei.value).lower()
    assert ("calib" in msg) or ("qa" in msg)


def test_7_consumers_unaffected(res):
    import stats_report
    r, _ = res
    df = stats_report.load_tracks(r["team_parquet_path"])
    assert "player_id" in df.columns
    assert "tid" in df.columns
    assert (df["tid"] == df["player_id"]).all()


def test_8_graph_self_disables(res):
    r, _ = res
    rep = r["report"]
    assert isinstance(rep["graph_used"], bool)
    if rep["proxy_pass_intra_rate"] <= 0.55 or rep["proxy_pass_total"] < 30:
        assert rep["graph_used"] is False
