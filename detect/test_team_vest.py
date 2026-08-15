#!/usr/bin/env python3
"""Birim testler: detect/team_vest (CEPHE B yelek-kuralli takim ayrimi).

Hizli/deterministik birim testler (video gerekmez) + opsiyonel real-data smoke.
Repo kokunden: venv/bin/python -m pytest detect/test_team_vest.py -q
"""
import os

import numpy as np
import pytest

from detect.team_vest import (DEFAULTS, classify, detect_gk_candidates,
                              first_goal_gate, torso_features)


# ----------------------------------------------------- ilk-gol kapisi (kural 1) --
def test_gate_unknown_refuses():
    for gt in (None, "unknown", "", "none"):
        g = first_goal_gate(gt)
        assert g["mode"] == "pre_goal_unsplittable"
        assert g["can_split"] is False


def test_gate_post_goal_allows():
    g = first_goal_gate(123.4)
    assert g["mode"] == "post_goal" and g["can_split"] is True
    assert g["goal_time"] == pytest.approx(123.4)


# --------------------------------------------------- tek-yonlu siniflama ----------
def _feat(band="nearmid", fYel=0.0, fDark=0.0, disp=2.0, yness=0.0):
    return dict(band=band, fYel=fYel, fDark=fDark, yness_med=yness,
                yfrac_med=fYel, yness_std=0.05, within_hue_disp=disp,
                n_samples=20, box_h_med=80.0)


def test_classify_vest_consistent_yellow():
    per = classify({1: _feat(fYel=0.7, fDark=0.0, yness=0.15)})
    assert per[1]["label"] == "vest" and per[1]["conf"] > 0.5


def test_classify_no_vest_never_yellow():
    per = classify({2: _feat(fYel=0.0, fDark=0.7, yness=-0.07)})
    assert per[2]["label"] == "no_vest"


def test_classify_contaminated_abstains():
    # hem sari hem koyu kare -> ID-swap kirli -> abstain (sahte takim YOK)
    per = classify({3: _feat(fYel=0.35, fDark=0.41)})
    assert per[3]["label"] == "abstain" and per[3]["contaminated"] is True


def test_classify_far_abstains():
    per = classify({4: _feat(band="far", fYel=np.nan, fDark=np.nan)})
    assert per[4]["label"] == "abstain"


def test_gk_never_vest():
    # sari-baskin olsa bile GK yelek-takimina ASLA atanmaz (kural 2)
    per = classify({5: _feat(fYel=0.8, fDark=0.0, yness=0.2)}, gk_ids=[5])
    assert per[5]["label"] == "gk" and per[5]["gk"] is True


def test_fener_jersey_not_counted_vest():
    # sari-baskin AMA govde-ici hue yayilimi yuksek -> Fener-formasi suphesi, yelek DEGIL (kural 4)
    per = classify({6: _feat(fYel=0.7, fDark=0.0, disp=40.0)})
    assert per[6]["label"] == "abstain" and per[6]["fener_suspect"] is True


def test_no_forced_balance():
    # 5 sari-baskin + 1 koyu: 9-9'a/dengeye ZORLANMAZ
    feats = {i: _feat(fYel=0.7, fDark=0.0, yness=0.2) for i in range(5)}
    feats[99] = _feat(fYel=0.0, fDark=0.6, yness=-0.07)
    per = classify(feats)
    nv = sum(1 for p in per.values() if p["label"] == "vest")
    assert nv == 5  # denge uydurmasi yok


# --------------------------------------------------- govde-ozellik (sentetik) -----
def _patch(bgr, h=60, w=30):
    img = np.zeros((200, 100, 3), np.uint8)
    # foot at (50,160), box_h=120 -> torso ROI ~ y[160-108 .. 160-48]=[52..112]
    img[40:130, 35:65] = bgr
    return img


def test_torso_yellow_patch_reads_yellow():
    yellow = (40, 220, 230)  # BGR ~ sari
    f = torso_features(_patch(yellow), fx=50, fy=160, bh=120, bw=40)
    assert f is not None and f["yfrac"] > 0.5 and f["yness"] > 0.1


def test_torso_blue_patch_reads_not_yellow():
    blue = (200, 90, 40)     # BGR ~ mavi gomlek
    f = torso_features(_patch(blue), fx=50, fy=160, bh=120, bw=40)
    assert f is not None and f["yfrac"] < 0.1 and f["yness"] < 0.0


# --------------------------------------------------- GK heuristik (sentetik) ------
def test_detect_gk_parked_at_goal():
    import pandas as pd
    rows = []
    # p0: kale cizgisinde park (X~33.5, az dolasir) -> GK aday
    # p1: orta sahada gezer -> degil
    for fr in range(60):
        rows.append(dict(player_id=0, frame=fr, pitch_x=33.5 + 0.3 * np.sin(fr / 5),
                         pitch_y=9.0))
        rows.append(dict(player_id=1, frame=fr, pitch_x=5 + 0.4 * fr, pitch_y=9.0))
    df = pd.DataFrame(rows)
    gk = detect_gk_candidates(df, L=34.0)
    assert 0 in gk and 1 not in gk


# --------------------------------------------------- real-data smoke (opsiyonel) --
REAL = ("stats_out/full_demo/"
        "tracks_cankaya_cam2_clip2400_presplit_player.parquet")
VIDEO = "raw/cankaya_cam2_clip2400.mp4"


@pytest.mark.skipif(not (os.path.exists(REAL) and os.path.exists(VIDEO)),
                    reason="real veri yok")
def test_real_pregoal_refuses(tmp_path):
    from detect.team_vest import assign_teams_vest
    res = assign_teams_vest(REAL, str(tmp_path), VIDEO, goal_time=None)
    assert res["report"]["gate"]["mode"] == "pre_goal_unsplittable"
    assert res["report"]["counts"]["vest"] == 0


@pytest.mark.skipif(not (os.path.exists(REAL) and os.path.exists(VIDEO)),
                    reason="real veri yok")
def test_real_postgoal_additive_and_gk_invariant(tmp_path):
    import pandas as pd
    from detect.team_vest import assign_teams_vest
    res = assign_teams_vest(REAL, str(tmp_path), VIDEO, goal_time=0.0)
    src = pd.read_parquet(REAL)
    out = pd.read_parquet(res["parquet_path"])
    assert len(src) == len(out)                       # additive: satir korunur
    assert (src["player_id"].values == out["player_id"].values).all()
    rep = res["report"]
    gk = set(int(k) for k in rep["gk_candidates"])
    vest = set(p["player_id"] for p in rep["per_player"] if p["label"] == "vest")
    assert len(gk & vest) == 0                         # GK ASLA vest
