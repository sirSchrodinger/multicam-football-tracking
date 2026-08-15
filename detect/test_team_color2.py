#!/usr/bin/env python3
"""Birim testler: detect/team_color2 (KIST-GSR / SoccerNet GSR 2025 renk-kumeleme + ban).

Hepsi SENTETIK + bilinen ground-truth uzerinde GERCEK sayisal davranis dogrular
(beklenen cevap modulde HARDCODE EDILMEZ; saflik/etiket GT'den olculur).
Repo kokunden: venv/bin/python -m pytest detect/test_team_color2.py -q
"""
import numpy as np
import pytest

from detect.team_color2 import (
    GK_LABEL, UNKNOWN,
    cluster_jersey, cluster_purity, cross_team_cost, cross_team_cost_matrix,
    fit_teams, fit_uniform_anchor, order_labels_by_x, team_label,
)


# --------------------------------------------------------------- veri uretici ----
def _two_color(n=40, sep=6.0, std=0.5, seed=0):
    """Iki ayrik tek-duze renk populasyonu (or. sari vs koyu), bilinen GT."""
    rng = np.random.default_rng(seed)
    a = rng.normal([0, 0, 0], std, size=(n, 3))
    b = rng.normal([sep, sep, 0], std, size=(n, 3))
    X = np.vstack([a, b])
    gt = np.r_[np.zeros(n), np.ones(n)].astype(int)
    return X, gt


def _uniform_plus_scatter(n_uni=8, n_oth=10, seed=1):
    """SIKI yelek kumesi + DAGILMIS karisik kiyafet (Alperen domain). GT: 0=uni,1=other."""
    rng = np.random.default_rng(seed)
    uni = rng.normal([5, 5, 5], 0.22, size=(n_uni, 3))         # cok siki
    # other: genis dagilmis, yelek kosesinden uzak (karisik sahsi renkler)
    oth = rng.uniform(-4, 1.5, size=(n_oth, 3))
    X = np.vstack([uni, oth])
    gt = np.r_[np.zeros(n_uni), np.ones(n_oth)].astype(int)
    return X, gt


# ============================================================ (a) iki-renk kumeleme
def test_two_color_kmeans_pure():
    X, gt = _two_color(seed=3)
    labels = team_label(X, method="kmeans")
    assert cluster_purity(labels, gt) > 0.90


def test_two_color_gmm_pure():
    X, gt = _two_color(seed=4)
    labels = team_label(X, method="gmm")
    assert cluster_purity(labels, gt) > 0.90


def test_two_color_two_distinct_labels():
    X, _ = _two_color(seed=5)
    labels = team_label(X, method="kmeans")
    assert set(np.unique(labels)) == {0, 1}


# ============================================ (b) tek-siki-yelek + dagilmis other ==
def test_uniform_anchor_recovers_tight_cluster():
    X, gt = _uniform_plus_scatter(seed=1)
    r = fit_uniform_anchor(X)
    labels = r["labels"]
    # yelek takimi (0) TEK SINIF olarak geri gelir, dagilmis other (1) ayrilir
    assert cluster_purity(labels, gt) > 0.90
    # bulunan siki kume gercekten daha SIKI olmali (spread_other > spread_uniform)
    assert r["spread_other"] > r["spread_uniform"]
    # uniform takim sayisi GT ile yakin (tek-sinif capa, dagilmis degil)
    assert abs(r["n_uniform"] - int((gt == 0).sum())) <= 1


def test_uniform_anchor_uniform_is_one_class_not_scattered():
    # yelek takiminin tahmin saflari, scattered'lara KARISMAMALI
    X, gt = _uniform_plus_scatter(n_uni=8, n_oth=12, seed=7)
    labels = team_label(X, method="uniform_anchor")
    uni_pred = labels[gt == 0]
    # yelek oyuncularinin cogunlugu ayni etikette (tek sinif)
    vals, counts = np.unique(uni_pred, return_counts=True)
    assert counts.max() / len(uni_pred) >= 0.875


def test_auto_picks_uniform_anchor_for_asymmetric():
    X, gt = _uniform_plus_scatter(seed=2)
    r = fit_teams(X, method="auto")
    assert r["method"] == "uniform_anchor"
    assert cluster_purity(r["labels"], gt) > 0.90


def test_auto_picks_kmeans_for_symmetric_two_color():
    X, _ = _two_color(seed=9)
    r = fit_teams(X, method="auto")
    assert r["method"] == "kmeans"


# ============================================ capraz-takim CANNOT-LINK (ban) ======
def test_cross_team_cost_inf_across_finite_within():
    assert cross_team_cost(0, 1) == float("inf")     # farkli takim -> yasak
    assert cross_team_cost(1, 0) == float("inf")
    assert cross_team_cost(0, 0) == 0.0              # ayni takim -> serbest
    assert cross_team_cost(1, 1) == 0.0
    assert np.isfinite(cross_team_cost(0, 0))


def test_cross_team_cost_gk_banned_from_field():
    # kaleci farkli sinif -> saha takimlariyla cannot-link
    assert cross_team_cost(GK_LABEL, 0) == float("inf")
    assert cross_team_cost(GK_LABEL, GK_LABEL) == 0.0


def test_cross_team_cost_unknown_no_constraint():
    # belirsiz (-1) -> ZORLA yasaklamayiz (yanlis-negatif tracking kirilmasi yok)
    assert cross_team_cost(UNKNOWN, 0) == 0.0
    assert cross_team_cost(1, UNKNOWN) == 0.0


def test_cross_team_cost_matrix_structure():
    a = np.array([0, 0, 1, 1, UNKNOWN])
    M = cross_team_cost_matrix(a)
    # kosegen (ayni nokta) hep 0
    assert np.all(np.diag(M) == 0.0)
    # 0 vs 1 -> inf
    assert M[0, 2] == float("inf") and M[2, 0] == float("inf")
    # 0 vs 0 -> 0
    assert M[0, 1] == 0.0
    # UNKNOWN satiri/sutunu -> hic inf yok
    assert np.all(np.isfinite(M[4, :])) and np.all(np.isfinite(M[:, 4]))


def test_cross_team_cost_matrix_rectangular():
    a = np.array([0, 1, 0])
    b = np.array([1, 1])
    M = cross_team_cost_matrix(a, b)
    assert M.shape == (3, 2)
    assert M[0, 0] == float("inf")   # 0 vs 1
    assert M[1, 0] == 0.0            # 1 vs 1


# ============================================ saha-x ile kimlik kanoniklestir =====
def test_order_labels_by_x_sorts_left_team_to_zero():
    # kume-kimligi keyfi gelmis olsun; sol takim (kucuk x) 0 olmali
    labels = np.array([1, 1, 0, 0])
    pitch_x = np.array([2.0, 3.0, 28.0, 30.0])   # etiket-1 solda, etiket-0 sagda
    out = order_labels_by_x(labels, pitch_x)
    # solda olan (eski 1) -> 0, sagda (eski 0) -> 1
    assert list(out) == [0, 0, 1, 1]


def test_order_labels_preserves_gk_and_unknown():
    labels = np.array([0, 1, GK_LABEL, UNKNOWN])
    px = np.array([5.0, 30.0, 15.0, 15.0])
    out = order_labels_by_x(labels, px)
    assert out[2] == GK_LABEL and out[3] == UNKNOWN


# ============================================ opsiyonel GK (3. sinif) ==============
def test_optional_gk_recovered_as_separate_class():
    rng = np.random.default_rng(11)
    teamA = rng.normal([0, 0, 0], 0.4, size=(12, 3))
    teamB = rng.normal([6, 6, 0], 0.4, size=(12, 3))
    gk = rng.normal([0, 6, 6], 0.3, size=(2, 3))     # az kisi, farkli renk
    X = np.vstack([teamA, teamB, gk])
    gt = np.r_[np.zeros(12), np.ones(12), np.full(2, 2)].astype(int)
    r = cluster_jersey(X, n_teams=2, with_gk=True, method="kmeans")
    labels = r["labels"]
    # GK_LABEL var ve gercek GK noktalarina denk geliyor
    assert GK_LABEL in set(labels)
    gk_pred = labels[gt == 2]
    assert np.all(gk_pred == GK_LABEL)
    # alan oyuncularinin saflik > 0.90
    field = gt != 2
    assert cluster_purity(labels[field], gt[field]) > 0.90


# ============================================ uctan uca: ban kumeleme ciktisinda ===
def test_end_to_end_cluster_then_ban():
    X, gt = _two_color(seed=21)
    labels = team_label(X, method="kmeans")
    M = cross_team_cost_matrix(labels)
    # ayni-takim ciftleri sonlu, farkli-takim ciftleri inf
    same = labels.reshape(-1, 1) == labels.reshape(1, -1)
    assert np.all(np.isfinite(M[same]))
    assert np.all(np.isinf(M[~same]))


# ============================================ girdi dogrulama =====================
def test_rejects_nan_features():
    X = np.array([[0.0, 0.0], [1.0, np.nan], [2.0, 2.0]])
    with pytest.raises(ValueError):
        team_label(X, method="kmeans")


def test_accepts_1d_features_as_column():
    x = np.array([0.0, 0.1, 5.0, 5.1])      # 1D -> (4,1)
    labels = team_label(x, method="kmeans")
    assert len(labels) == 4
    assert cluster_purity(labels, np.array([0, 0, 1, 1])) > 0.90
