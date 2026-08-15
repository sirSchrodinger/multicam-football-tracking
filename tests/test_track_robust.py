#!/usr/bin/env python3
"""GERCEK-VERI kabul testi: detect/track_robust.py (takip-saglamligi).

Girdiler CANLI dogrulanmis dosyalar (synthetic/uydurma YOK):
  raw/tracks_cankaya_cam2_clip2400.parquet (146 tracklet, 35738 satir)
  calib/cankaya_cam2_v2.json (QA gecer, median 9.87px)
  raw/cankaya_cam2_clip2400.mp4 (jersey renk / appearance)

Test 4 (interpolasyon metre/hiz disi) DELIVERABLE'in durustluk kapisi: guard
eklenmeden KIRMIZI olmali, eklenince yesil.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import detect.track_robust as TR
import detect.track_stitch as TS
from pitch.homography import PitchHomography
from stats_report import load_tracks
from stats.topdown_stats import prepare, kinematics_all

RAW = str(ROOT / "raw" / "tracks_cankaya_cam2_clip2400.parquet")
CALIB = str(ROOT / "calib" / "cankaya_cam2_v2.json")
VIDEO = str(ROOT / "raw" / "cankaya_cam2_clip2400.mp4")


@pytest.fixture(scope="module")
def scratch(tmp_path_factory):
    return tmp_path_factory.mktemp("track_robust")


def _fps(df):
    return float(df["frame"].max()) / float(df["t_sec"].max())


def _dist_by_player(df):
    """distance/speed entrypoint: prepare + kinematics_all -> {player: dist_raw}."""
    homo = PitchHomography.load(CALIB)
    prep = prepare(df, homo, _fps(df))
    kin = kinematics_all(prep)
    return {int(u): float(d) for u, d in zip(kin["uniq"], kin["dist_raw"])}


# 1) diagnose olculen modlari yeniden-uretir -----------------------------------
def test_diagnose_reproduces_measured_modes():
    rep = TR.diagnose(RAW, CALIB)
    assert rep["n_rows"] == 35738 and rep["n_tracklets"] == 146
    assert abs(rep["fps"] - 24.872) < 0.01
    assert rep["mode1a_intra_swaps"]["n"] > 0                      # swap sinyali var
    assert rep["mode1b_encounters"]["n_pairs"] == 133
    assert rep["mode1b_encounters"]["seam_risk"] == 67
    assert rep["mode2a_intra_holes"]["n_tids"] == 110
    assert rep["mode2a_intra_holes"]["n_holes"] == 802
    assert abs(rep["mode3_4_far_border"]["far_row_frac"] - 0.333) < 0.01
    assert rep["mode3_4_far_border"]["literal_border_deaths"] == 0  # MODE4 durust


# 2) presplit kayipsiz + konservatif -------------------------------------------
def test_presplit_lossless_and_conservative(scratch):
    p = TR.presplit(RAW, CALIB, str(scratch / "ps.parquet"))
    d = load_tracks(p)
    assert len(d) == 35738                                         # SIFIR satir kaybi
    assert d["tid"].nunique() > 146                                # crossing-coincident bolme oldu
    assert "frag_split" in d.columns
    # MIN-SEG GUARD: bolme ile uretilen yeni tid'ler >= MIN_SEG_OBS
    raw = load_tracks(RAW)
    new_tids = set(d["tid"].unique()) - set(raw["tid"].unique())
    assert new_tids, "yeni tid uretilmedi (bolme yok)"
    sizes = d[d["tid"].isin(new_tids)].groupby("tid").size()
    assert int(sizes.min()) >= TR.MIN_SEG_OBS                      # gurultu splinter yok


# 3) bridge isaretler + sinirlar durust ----------------------------------------
def test_bridge_marks_and_bounds(scratch):
    res = TS.stitch(RAW, CALIB, str(scratch))                       # video YOK (hiz)
    pk = res["player_keyed_path"]
    out, meta = TR.bridge_gaps(pk, CALIB, str(scratch / "br.parquet"))
    b = load_tracks(out)
    interp = b[b["interpolated"] == True]                           # noqa: E712
    assert meta["bridged_rows"] == len(interp) and meta["bridged_rows"] > 0
    assert interp["conf"].isna().all()                             # kopru satiri tespit DEGIL
    assert meta["bridged_skipped_long"] > 0                        # uzun bosluk durustce doldurulMADI
    assert b["interpolated"].notna().all()
    # hicbir kesintisiz interp dizisi BRIDGE_MAX_FR'yi asmaz
    for _pid, g in b.sort_values("frame").groupby("tid"):
        fr = g.loc[g["interpolated"] == True, "frame"].to_numpy(np.int64)  # noqa: E712
        if fr.size == 0:
            continue
        runs = np.split(fr, np.where(np.diff(fr) > 1)[0] + 1)
        for run in runs:
            assert len(run) <= TR.BRIDGE_MAX_FR


# 4) DURUSTLUK: interpolasyon metre/hiz toplamasindan HARIC (load-bearing) ------
def test_honesty_interpolated_excluded_from_metrics(scratch):
    res = TS.stitch(RAW, CALIB, str(scratch / "h"))
    out, meta = TR.bridge_gaps(res["player_keyed_path"], CALIB,
                               str(scratch / "h" / "br.parquet"))
    b = load_tracks(out)
    assert int((b["interpolated"] == True).sum()) > 0              # gercekten interp satir var # noqa: E712
    stats_with = _dist_by_player(b)
    stats_without = _dist_by_player(b[b["interpolated"] != True])  # noqa: E712
    assert set(stats_with) == set(stats_without)
    for pid in stats_with:
        # interp satir 0 metre katkilar -> with == without (mesafe yalani YOK)
        assert stats_with[pid] == pytest.approx(stats_without[pid], abs=1e-6)


# 5) team_label zorunlu cikti ---------------------------------------------------
def test_team_label_required_output(scratch):
    res = TS.stitch(RAW, CALIB, str(scratch / "t"))
    out, _ = TR.bridge_gaps(res["player_keyed_path"], CALIB,
                            str(scratch / "t" / "br.parquet"))
    b = load_tracks(out)
    teams = TR.team_label(b, CALIB, video_path=VIDEO)
    assert set(teams.keys()) == set(int(x) for x in b["tid"].unique())
    for _pid, t in teams.items():
        assert t["team_id"] in (0, 1)
        assert 0.0 <= t["conf"] <= 1.0
    assert len({t["team_id"] for t in teams.values()}) == 2        # iki takim da var
    # en az bir uzak-uc-baskin / dusuk-kalite oyuncu DUSUK conf tasir (soft, sert-zorlama yok)
    confs = [t["conf"] for t in teams.values()]
    assert min(confs) < max(confs)
    assert any(t["far_dominant"] or t["color_quality"] < 0.3 for t in teams.values())


# NOT: eski test_track_stitch_untouched (SHA256-pin) KALDIRILDI. Workflow-donemi
# izolasyon guard'iydi (track_robust track_stitch'i degistirmesin); artik track_stitch
# aktif gelistirilen sahip modul (presence/pitch/player-keyed eklendi) -> hash-pin her
# mesru duzenlemede kirilir, davranis test etmez. Ayrim kod-incelemeyle saglanir.
