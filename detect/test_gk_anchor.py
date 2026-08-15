#!/usr/bin/env python3
"""Birim testler: detect/gk_anchor.gk_role_timeline (ZAMAN-DEGISKEN GK ROL).

Sentetik df'lerle (video/calib gerekmez) GK timeline'in 3 davranisini kanitlar:
  (a) STABILITE  : tek-kaleci -> tum frame ayni pid.
  (b) ROTASYON   : iki-kaleci (ilk yari A, ikinci yari B) -> gecis takip edilir (histerezis-gecikmeli).
  (c) HISTEREZIS : tek-frame disari cikan kaleci GK kalir (flicker yok).
Repo kokunden: venv/bin/python -m pytest detect/test_gk_anchor.py -q
"""
import numpy as np
import pandas as pd

from detect.gk_anchor import gk_role_timeline

# sentetik saha: cal yalniz L,W tasir (foot/K yok -> X,Y dogrudan kullanilir)
CAL = {"L": 34.0, "W": 20.0}
FPS = 25.0
GY = CAL["W"] / 2.0          # agiz merkezi


def _df(rows):
    """rows: list of (player_id, frame, X, Y) -> DataFrame."""
    return pd.DataFrame(rows, columns=["player_id", "frame", "X", "Y"])


def _near_seq(tl, n):
    return [tl[f]["near"] for f in range(n)]


# ----------------------------------------------------- (a) STABILITE ------------
def test_stability_single_keeper_constant_pid():
    # pid 7 tum mac near-agizda (X~1, Y=W/2); pid 1 orta sahada gezer.
    rows = []
    for f in range(400):
        rows.append((7, f, 1.0, GY))                 # kaleci: agizda sabit
        rows.append((1, f, 17.0, GY + 3.0))          # saha oyuncusu: orta saha
    tl = gk_role_timeline(_df(rows), CAL, FPS)
    seq = _near_seq(tl, 400)
    assert set(seq) == {7}, f"tek-kaleci timeline sabit degil: {set(seq)}"
    # far uc bos -> hep None
    assert all(tl[f]["far"] is None for f in range(400))


# ----------------------------------------------------- (b) ROTASYON -------------
def test_rotation_keeper_handoff_followed():
    # ilk yari pid A=10 agizda, ikinci yari pid B=20 agizda (amator rotasyon).
    N = 600
    half = N // 2
    rows = []
    for f in range(N):
        if f < half:
            rows.append((10, f, 1.0, GY))            # A kalede
            rows.append((20, f, 17.0, GY))           # B orta saha
        else:
            rows.append((10, f, 17.0, GY))           # A orta sahaya cikti
            rows.append((20, f, 1.0, GY))            # B kaleye gecti
    tl = gk_role_timeline(_df(rows), CAL, FPS)
    # erken frame -> A; gec frame -> B (gecis takip edildi)
    assert tl[100]["near"] == 10, f"erken GK A degil: {tl[100]['near']}"
    assert tl[N - 1]["near"] == 20, f"gec GK B'ye gecmedi: {tl[N - 1]['near']}"
    seq = _near_seq(tl, N)
    # gecis A->B tam olarak bir kez ve yari-noktadan SONRA (histerezis-gecikmeli) olmali
    switches = [f for f in range(1, N) if seq[f] != seq[f - 1] and seq[f] is not None and seq[f - 1] is not None]
    assert switches, "hic gecis yok"
    assert all(s >= half for s in switches), f"gecis yari-noktadan once: {switches}"
    assert seq[switches[-1]] == 20


# ----------------------------------------------------- (c) HISTEREZIS -----------
def test_hysteresis_single_frame_out_keeps_gk():
    # pid 5 near-agizda; SADECE 1 frame disari cikar (flicker). GK degismemeli.
    rows = []
    for f in range(400):
        X = 9.0 if f == 200 else 1.0                 # tek frame agiz-disi (X=9 > depth ile sinir)
        rows.append((5, f, X, GY))
        rows.append((2, f, 17.0, GY))                # saha oyuncusu
    tl = gk_role_timeline(_df(rows), CAL, FPS)
    seq = _near_seq(tl, 400)
    assert set(seq) == {5}, f"tek-frame flicker GK'yi degistirdi: {set(seq)}"


def test_hysteresis_blocks_brief_challenger():
    # yerlesik GK=5 agizda; rakip=8 KISA sure (histerezis pencere-payindan kisa) agiza dalip cikar.
    # Histerezis: rakip mevcut GK'nin 1.3x'ini gecemez -> GK 5 kalir.
    rows = []
    for f in range(500):
        rows.append((5, f, 1.0, GY))                 # yerlesik kaleci surekli agizda
        # rakip 8 yalniz 40 frame agizda (pencere ~200f -> frac<<1, GK 5'i asamaz)
        X8 = 1.5 if 250 <= f < 290 else 17.0
        rows.append((8, f, X8, GY))
    tl = gk_role_timeline(_df(rows), CAL, FPS)
    seq = _near_seq(tl, 500)
    assert set(seq) == {5}, f"kisa rakip GK'yi flip etti (histerezis kirildi): {set(seq)}"


# --------------------------------------------- gap-carry (kisa bosluk tasima) ----
def test_short_gap_carries_previous_gk():
    # GK=3 agizda; ortada KISA gozlem boslugu (agizda kimse) -> onceki GK tasinir.
    rows = []
    for f in range(300):
        if 150 <= f < 160:
            continue                                 # 10-frame gozlem boslugu (agizda kimse)
        rows.append((3, f, 1.0, GY))
    tl = gk_role_timeline(_df(rows), CAL, FPS)
    # bosluk icinde onceki GK (3) tasinmali (win_s rezidansi pozitif)
    assert tl[155]["near"] == 3, f"kisa boslukta GK tasinmadi: {tl[155]['near']}"
