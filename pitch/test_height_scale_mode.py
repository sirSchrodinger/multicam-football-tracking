#!/usr/bin/env python3
"""height_scale_mode kabul testi — sentetik carpik karisimda MODE-vs-MEAN ve
imkansiz-katalog dedektoru (bilinen ground-truth, hicbir cevap modulde gomulu degil).

Kanit hedefleri (BMVC 2011 yaya-boyu oto-kalibrasyon fikrine sadik):
  1. Sola-carpik karisimda (standing ~1.75 + crouch ~1.3 kuyrugu) MODE ~1.75'i bulur,
     MEAN belirgin dusuktur ve MODE mean'den GERCEKTEN cok daha yakindir.
  2. GMM yolu da duruslu merkezi (~1.75) bulur.
  3. Mesafe-agirligi MOD'u tasir: cömelme sayica cogunluk ama duruslu olcumler agir
     tartilirsa weighted-mode duruslu boyu kurtarir (unweighted ise cömelmeye duser).
  4. Standing-gate (foot velocity ~0 + dik aspect) carpik kuyrugu eler.
  5. Imkansiz katalog (46x24 -> 2.4 m insan / olcek celiskisi) FLAGGED; tutarli olan DEGIL.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from pitch.height_scale_mode import (  # noqa: E402
    height_mode_scale, impossible_catalog_flag, ASSUMED_STANDING_M,
)

TRUE_STANDING = 1.75
TRUE_CROUCH = 1.30


def _skewed_mixture(n_standing=900, n_crouch=300, seed=0):
    """Sola-carpik karisim: duruslu cogunluk + cömelme alt-kuyrugu (rel_m=m)."""
    rng = np.random.default_rng(seed)
    standing = rng.normal(TRUE_STANDING, 0.05, n_standing)
    crouch = rng.normal(TRUE_CROUCH, 0.08, n_crouch)
    return np.concatenate([standing, crouch])


# --------------------------------------------------------------------------- #
#  1. MODE carpitmaya direnir, MEAN dusuk
# --------------------------------------------------------------------------- #
def test_mode_recovers_standing_mean_is_biased_low():
    H = _skewed_mixture()
    r = height_mode_scale(H)  # method='kde' varsayilan

    # mean'i BAGIMSIZ hesapla (modul cevabini gomme yok)
    plain_mean = float(np.mean(H))
    mode = r["mode_height"]

    mode_err = abs(mode - TRUE_STANDING)
    mean_err = abs(plain_mean - TRUE_STANDING)

    # MODE gercek duruslu boya yakin
    assert mode_err < 0.08, f"mode {mode:.3f} duruslu boya uzak"
    # MEAN sistematik olarak DUSUK (carpitma kaniti)
    assert plain_mean < TRUE_STANDING - 0.08, f"mean {plain_mean:.3f} beklenenden yuksek"
    # MODE, MEAN'den GERCEKTEN cok daha yakin (en az 2x)
    assert mode_err < 0.5 * mean_err, (
        f"mode_err={mode_err:.3f} mean_err={mean_err:.3f} -> mode yeterince yakin degil")
    # modul kendi mean'ini de raporluyor ve plain-mean ile uyumlu (uniform agirlik)
    assert abs(r["mean_height"] - plain_mean) < 1e-9
    assert r["n_used"] == len(H)
    # scale = assumed / mode ; mode ~1.75 ve assumed 1.75 -> ~1.0
    assert abs(r["scale"] - ASSUMED_STANDING_M / mode) < 1e-9
    assert 0.9 < r["scale"] < 1.1


# --------------------------------------------------------------------------- #
#  2. GMM yolu da duruslu merkezi bulur
# --------------------------------------------------------------------------- #
def test_gmm_finds_standing_center():
    H = _skewed_mixture()
    r = height_mode_scale(H, method="gmm")
    assert abs(r["mode_height"] - TRUE_STANDING) < 0.08, r["mode_height"]
    # yine mean'den belirgin yuksek (carpitmaya direndi)
    assert r["mode_height"] > float(np.mean(H)) + 0.08


# --------------------------------------------------------------------------- #
#  3. Mesafe-agirligi MOD'u tasir
# --------------------------------------------------------------------------- #
def test_distance_weights_steer_the_mode():
    rng = np.random.default_rng(1)
    # cömelme SAYICA cogunluk (uzak/guvenilmez), duruslu azinlik ama YAKIN (agir)
    crouch = rng.normal(TRUE_CROUCH, 0.06, 700)
    standing = rng.normal(TRUE_STANDING, 0.05, 300)
    H = np.concatenate([crouch, standing])
    w = np.concatenate([np.full(700, 0.1), np.full(300, 1.0)])  # yakin=agir

    r_unw = height_mode_scale(H)            # agirliksiz -> cömelme cogunluguna duser
    r_w = height_mode_scale(H, weights=w)   # agirlikli -> duruslu boya cikar

    assert abs(r_unw["mode_height"] - TRUE_CROUCH) < 0.10, r_unw["mode_height"]
    assert abs(r_w["mode_height"] - TRUE_STANDING) < 0.10, r_w["mode_height"]
    # agirlik MOD'u acikca yukari tasidi
    assert r_w["mode_height"] > r_unw["mode_height"] + 0.25


# --------------------------------------------------------------------------- #
#  4. Standing-gate carpik kuyrugu eler
# --------------------------------------------------------------------------- #
def test_standing_gate_removes_nonstanding():
    rng = np.random.default_rng(2)
    # cömelme SAYICA cogunluk -> gate OLMADAN mod cömelmeye duser; gate duruslu kurtarir
    n_s, n_c = 400, 1000
    standing = rng.normal(TRUE_STANDING, 0.05, n_s)
    crouch = rng.normal(TRUE_CROUCH, 0.05, n_c)
    H = np.concatenate([standing, crouch])
    # duruslu: dik aspect (yuksek) + yavas/stabil ayak ; cömelme: tersi
    aspect = np.concatenate([rng.normal(2.6, 0.2, n_s), rng.normal(1.2, 0.15, n_c)])
    speed = np.concatenate([rng.uniform(0.0, 0.5, n_s), rng.uniform(2.0, 4.0, n_c)])

    r_nogate = height_mode_scale(H)
    r_gate = height_mode_scale(H, aspect=aspect, foot_speed=speed,
                               speed_max=1.0, aspect_min=1.8)

    assert r_gate["gated"] is True
    # gate sonrasi neredeyse yalniz duruslu olcumler kaldi
    assert r_gate["n_used"] < r_nogate["n_used"]
    # gate OLMADAN mod carpik cogunluga (cömelme) dusmus
    assert abs(r_nogate["mode_height"] - TRUE_CROUCH) < 0.10, r_nogate["mode_height"]
    # gate ILE mod gercek duruslu boyu kurtardi
    assert abs(r_gate["mode_height"] - TRUE_STANDING) < 0.06, r_gate["mode_height"]
    # gate'li mod, gate'siz moddan gercege belirgin daha yakin
    assert abs(r_gate["mode_height"] - TRUE_STANDING) < abs(
        r_nogate["mode_height"] - TRUE_STANDING) - 0.25


def test_too_few_used_returns_nan():
    r = height_mode_scale(np.array([1.7, 1.8]), min_used=5)
    assert np.isnan(r["mode_height"]) and np.isnan(r["scale"])
    assert r["n_used"] == 2


# --------------------------------------------------------------------------- #
#  5. Imkansiz-katalog dedektoru
# --------------------------------------------------------------------------- #
def test_impossible_catalog_is_flagged_consistent_is_not():
    # IMKANSIZ: katalog 46x24 -> boy-capasiyla %15 sapan olcek + boyut araligi disinda
    bad = impossible_catalog_flag(scale_from_height=1.00, scale_from_goal=1.15,
                                  implied_L=46.0, implied_W=24.0)
    assert bad["flagged"] is True
    assert "futsal" in bad["reason"] or "celis" in bad["reason"]

    # TUTARLI: olcekler ~esit (%2) + boyutlar futsal araliginda
    good = impossible_catalog_flag(scale_from_height=1.00, scale_from_goal=1.02,
                                   implied_L=38.0, implied_W=20.0)
    assert good["flagged"] is False
    assert good["reason"] == "tutarli"
    assert abs(good["ratio"] - (1.00 / 1.02)) < 1e-9


def test_catalog_flag_branches_are_independent():
    # SADECE olcek celiskisi (boyut araliginda ama insanlar ~2.4 m -> olcek sapar)
    # scale_from_height=1.0 (1.75 m capa), scale_from_goal -> 1.75/2.4 anlaminda 0.73
    ratio_only = impossible_catalog_flag(scale_from_height=1.0,
                                         scale_from_goal=1.0 / (1.75 / 2.4),
                                         implied_L=40.0, implied_W=20.0)
    assert ratio_only["flagged"] is True
    assert "celis" in ratio_only["reason"]
    assert "futsal" not in ratio_only["reason"]  # boyutlar makul

    # SADECE boyut disinda (olcekler tutarli)
    dim_only = impossible_catalog_flag(scale_from_height=1.0, scale_from_goal=1.0,
                                       implied_L=50.0, implied_W=20.0)
    assert dim_only["flagged"] is True
    assert "futsal" in dim_only["reason"]
    assert "celis" not in dim_only["reason"]  # olcekler ozdes


def test_catalog_zero_scale_guard():
    r = impossible_catalog_flag(1.0, 0.0, 40.0, 20.0)
    assert r["flagged"] is True
    assert np.isinf(r["ratio"])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
