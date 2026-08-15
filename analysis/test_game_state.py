#!/usr/bin/env python3
"""game_state kabul testleri — pozisyon-only KICKOFF + defansif WALL.

Sentetik, BİLİNEN-GERÇEK formasyonlar üzerinde anlamlı davranışı kanıtlar:
  * temiz bir kickoff formasyonu (takımlar ayrık, biri orta noktada, duruyor) ATEŞLER,
    ve ateşleme zamanı centroid'in çizgiyi geçtiği gerçek kareye yakındır;
  * açık/rastgele oyun ATEŞLEMEZ;
  * doğrusal-eşit-aralıklı-dik 3-oyunculu duvar ATEŞLER, dağınık savunma ATEŞLEMEZ;
  * aynalanmış kickoff/wall BİREBİR AYNI skoru verir (yansıma-simetri).

Repo kökünden: venv/bin/python -m pytest analysis/test_game_state.py -q
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.game_state import (  # noqa: E402
    detect_kickoff, detect_wall, kickoff_readiness, best_wall,
    reflect_x, reflect_tracks_df,
)

L, W = 40.0, 20.0          # saha (uzunluk, genişlik) metre
HALF = L / 2.0
FPS = 25.0


# --------------------------------------------------------------- kickoff data


def _kickoff_df(n_ready=24, n_cross=24, cross_vx=0.6, fps=FPS):
    """Sentetik kickoff: hazır faz (ayrık + orta + duruyor) -> sol takım çizgiyi geçer.

    team0 = sol/vuruşu kullanan (4 sol oyuncu + 1 orta-nokta oyuncusu), team1 = sağ (5).
    Hazır fazda herkes sabit; cross fazında team0 +X yönünde kayar -> centroid HALF'i geçer.
    Döner: (df, expected_cross_t). expected_cross_t veriden ölçülür (modüle gömülmez).
    """
    # sabit hazır-formasyon konumları
    team0_left = np.array([[6, 6], [8, 14], [10, 8], [12, 12]], float)   # 4 sol
    team0_center = np.array([[HALF, W / 2]], float)                       # orta nokta
    team1_right = np.array([[26, 7], [28, 13], [30, 9], [32, 11], [34, 10]], float)
    team0_xy = np.vstack([team0_left, team0_center])                      # 5 oyuncu
    team1_xy = team1_right                                                # 5 oyuncu

    rows = []
    frame = 0
    # --- hazır faz: tamamen sabit ---
    for _ in range(n_ready):
        t = frame / fps
        for j, (x, y) in enumerate(team0_xy):
            rows.append((j, frame, t, x, y, 0))
        for j, (x, y) in enumerate(team1_xy):
            rows.append((100 + j, frame, t, x, y, 1))
        frame += 1

    # --- cross faz: team0 (orta dahil) +X kayar, team1 sabit ---
    cross_t = None
    base0 = team0_xy.copy()
    for k in range(1, n_cross + 1):
        t = frame / fps
        shifted = base0.copy()
        shifted[:, 0] = base0[:, 0] + cross_vx * k          # her kare +cross_vx m
        cen_x = shifted[:, 0].mean()
        for j, (x, y) in enumerate(shifted):
            rows.append((j, frame, t, x, y, 0))
        for j, (x, y) in enumerate(team1_xy):
            rows.append((100 + j, frame, t, x, y, 1))
        if cross_t is None and cen_x > HALF:
            cross_t = t                                     # centroid'in çizgiyi geçtiği ilk kare
        frame += 1

    df = pd.DataFrame(rows, columns=["tid", "frame", "t_sec",
                                     "pitch_x", "pitch_y", "team_id"])
    return df, cross_t


def _open_play_df(n=80, seed=7, fps=FPS):
    """Rastgele açık oyun: yapısız konumlar + yüksek hareket; ayrık/orta/durağan DEĞİL."""
    rng = np.random.default_rng(seed)
    rows = []
    # her oyuncu pitch içinde rastgele yürüyüş (her kare büyük adım -> yüksek hız)
    pos = np.column_stack([rng.uniform(2, L - 2, 10), rng.uniform(2, W - 2, 10)])
    teams = np.array([0] * 5 + [1] * 5)
    for frame in range(n):
        t = frame / fps
        pos = pos + rng.uniform(-1.5, 1.5, pos.shape)       # ~30-50 m/s, set DEĞİL
        pos[:, 0] = np.clip(pos[:, 0], 1, L - 1)
        pos[:, 1] = np.clip(pos[:, 1], 1, W - 1)
        for j in range(10):
            rows.append((j, frame, t, pos[j, 0], pos[j, 1], int(teams[j])))
    return pd.DataFrame(rows, columns=["tid", "frame", "t_sec",
                                       "pitch_x", "pitch_y", "team_id"])


# -------------------------------------------------------------- kickoff tests


def test_kickoff_fires_on_set_then_cross():
    df, cross_t = _kickoff_df()
    assert cross_t is not None, "sentetik kurulum hatası: centroid hiç geçmedi"
    fires = detect_kickoff(df, (L, W))
    assert len(fires) >= 1, "temiz kickoff formasyonu ATEŞLEMEDİ"
    # ateşleme, centroid'in çizgiyi gerçekten geçtiği kareye yakın olmalı (~2 kare)
    nearest = min(fires, key=lambda t: abs(t - cross_t))
    assert abs(nearest - cross_t) <= 2.0 / FPS + 1e-6, \
        f"ateşleme {nearest:.3f}s, beklenen crossing {cross_t:.3f}s'e uzak"


def test_kickoff_readiness_high_during_set_low_during_play():
    df, _ = _kickoff_df()
    ser = kickoff_readiness(df, (L, W))
    # hazır fazın ilk yarısında readiness ~1 olmalı
    early = ser[ser["t_sec"] < 0.3]["readiness"].to_numpy()
    assert early.size > 0 and early.min() > 0.9, "set hâli yüksek readiness vermedi"
    # split/center/speed bileşenleri de hazır fazda ~1
    e0 = ser.iloc[0]
    assert e0["split"] > 0.95 and e0["center"] > 0.95 and e0["speed"] > 0.95


def test_open_play_does_not_fire():
    df = _open_play_df()
    fires = detect_kickoff(df, (L, W))
    assert fires == [], f"açık/rastgele oyun yanlışlıkla ateşledi: {fires}"
    # readiness her karede düşük (ayrık+orta+durağan değil)
    ser = kickoff_readiness(df, (L, W))
    assert ser["readiness"].max() < 0.5


def test_kickoff_mirror_same_score_and_timestamps():
    df, _ = _kickoff_df()
    df_m = reflect_tracks_df(df, L, swap_team=True)     # orta çizgiye göre ayna + takas

    ser = kickoff_readiness(df, (L, W)).sort_values("frame").reset_index(drop=True)
    ser_m = kickoff_readiness(df_m, (L, W)).sort_values("frame").reset_index(drop=True)
    # readiness serisi BİREBİR aynı (yansıma-değişmez skorlama)
    np.testing.assert_allclose(ser["readiness"].to_numpy(),
                               ser_m["readiness"].to_numpy(), atol=1e-9)
    np.testing.assert_allclose(ser["split"].to_numpy(),
                               ser_m["split"].to_numpy(), atol=1e-9)

    f0 = detect_kickoff(df, (L, W))
    f1 = detect_kickoff(df_m, (L, W))
    assert len(f0) == len(f1) and len(f0) >= 1
    np.testing.assert_allclose(sorted(f0), sorted(f1), atol=1e-9)


# ----------------------------------------------------------------- wall data


def _wall_positions():
    """Defansif duvar senaryosu. Kale X=0'da; restart (10,10).

    rg = goal-restart = (-10,0) -> X ekseni. Duvar buna DİK = Y ekseni boyunca dizilir,
    restart'tan ~7 m (X=3) ileride, hatta ortalı (Y~10). 3 oyuncu 1.5 m aralıkla.
    """
    restart = np.array([10.0, 10.0])
    goal = np.array([0.0, 10.0])
    wall = np.array([[3.0, 8.5], [3.0, 10.0], [3.0, 11.5]])     # doğrusal, eşit, dik
    others = np.array([[1.0, 10.0],      # kaleci (kalede)
                       [9.0, 4.0]])       # alakasız savunan
    positions = np.vstack([wall, others])
    return positions, restart, goal


def _scattered_positions():
    """Dağınık savunma: doğrusal/eşit/dik/bant koşullarını sağlamayan konumlar."""
    restart = np.array([10.0, 10.0])
    goal = np.array([0.0, 10.0])
    positions = np.array([
        [2.0, 3.0],
        [6.0, 16.0],
        [12.0, 9.0],
        [4.0, 14.0],
        [8.0, 6.5],
    ])
    return positions, restart, goal


# ------------------------------------------------------------------ wall tests


def test_wall_fires_on_collinear_perpendicular_triplet():
    positions, restart, goal = _wall_positions()
    fired, score = detect_wall(positions, restart, goal)
    assert fired, f"doğrusal-dik-bantlı 3'lü duvar ATEŞLEMEDİ (score={score:.3f})"
    assert score > 0.6
    # en iyi alt-küme GERÇEK duvar oyuncularından seçilmeli (kaleci/alakasız değil)
    _, _, members = best_wall(positions, restart, goal)
    assert len(members) >= 2 and set(members).issubset({0, 1, 2}), \
        f"duvar yanlış oyunculara kilitlendi: {members}"


def test_scattered_defense_does_not_fire():
    positions, restart, goal = _scattered_positions()
    fired, score = detect_wall(positions, restart, goal)
    assert not fired, f"dağınık savunma yanlışlıkla duvar saydı (score={score:.3f})"
    assert score < 0.6


def test_wall_score_is_reflection_symmetric():
    positions, restart, goal = _wall_positions()
    f0, s0 = detect_wall(positions, restart, goal)
    # her şeyi orta çizgiye göre aynala (X -> L-X)
    pm = reflect_x(positions, L)
    rm = reflect_x(restart, L)
    gm = reflect_x(goal, L)
    f1, s1 = detect_wall(pm, rm, gm)
    assert f0 == f1
    assert abs(s0 - s1) < 1e-9, f"aynalanmış duvar farklı skor: {s0} vs {s1}"


def test_wall_needs_perpendicularity():
    """Aynı doğrusal-eşit oyuncular ama (restart->kale) hattına PARALEL -> duvar değil."""
    restart = np.array([10.0, 10.0])
    goal = np.array([0.0, 10.0])
    # X ekseni (rg yönü) boyunca dizilmiş = paralel, dik değil
    parallel = np.array([[5.0, 10.0], [6.5, 10.0], [8.0, 10.0]])
    fired, score = detect_wall(parallel, restart, goal)
    assert not fired, f"hatta-paralel dizilim duvar saydı (score={score:.3f})"


def test_wall_needs_distance_band():
    """Doğru geometri ama restart'a ÇOK yakın (bant dışı) -> skor düşmeli."""
    restart = np.array([10.0, 10.0])
    goal = np.array([0.0, 10.0])
    near = np.array([[9.4, 8.5], [9.4, 10.0], [9.4, 11.5]])   # ~0.6 m, banttan çok yakın
    fired, score = detect_wall(near, restart, goal)
    assert not fired, f"bant-dışı (çok yakın) dizilim duvar saydı (score={score:.3f})"
