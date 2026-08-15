"""test_possession — top-bağımsız sahiplenme + pas/turnover çıkarımının
sentetik-GT testleri.

Tüm yer-doğrulukları (ground truth) sentetik ve BİLİNİR. Beklenen cevap modüle
gömülü DEĞİLDİR; modül ham konumlardan oyun-odağını ve Viterbi yolunu hesaplar,
test yalnız gerçek sayısal davranışı doğrular:

  * convergence_point  : bilinen bir noktada kesişen ışınlar => o nokta geri
                         kazanılır; paralel ışınlar => ok=False (yedek).
  * pass               : taşıyan A açıkça odağı tutar, sonra takım arkadaşı B'ye
                         devreder => sahip A->B'yi izler, devirde bir 'pass' olayı.
  * viterbi flicker    : 1-kare çapraz-saha titremesi (savunanlar bir an uzak
                         köşeye baskı yapar) => açgözlü sahip sıçrar AMA Viterbi
                         bastırır (sahip A kalır, hiç olay yok).
  * turnover           : karşı-takım oyuncusuna geçiş + takım etiketi => 'turnover';
                         aynı dizi etiketsiz => 'pass' (turnover ayrımı etiket ister).

Senaryo üretici "savunan baskısı" mekaniği: pres yapan oyuncular her kare
taşıyana doğru sabit küçük adımla yaklaşır => geri-fark hızları taşıyanı
gösterir => hız-ışınları taşıyanda kesişir => odak ~ taşıyan. Taşıyanın kendisi
sabit durur (hız ~0, ışına oy vermez) => odağa en yakın oyuncu olarak seçilir.
"""
import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis import possession as P  # noqa: E402


# --------------------------------------------------------------------------- #
# Senaryo yardımcıları
# --------------------------------------------------------------------------- #
DEF_ANGLES = np.deg2rad([30.0, 150.0, 270.0])  # 3 pres-eden, farklı açılarda
R0 = 8.0       # başlangıç baskı yarıçapı (m)
DELTA = 0.30   # kare-başına içe sürünme (m), v_min'in üstünde


def _press_ring(carrier, t_local):
    """Taşıyanın etrafında 3 pres-eden oyuncunun konumu (içe sürünerek)."""
    r = R0 - DELTA * t_local
    out = []
    for a in DEF_ANGLES:
        out.append(carrier + r * np.array([np.cos(a), np.sin(a)]))
    return out  # list of 3 (2,) arrays


def _toward(p_from, p_to, t_steps, step=DELTA):
    """p_from'dan p_to'ya doğru sabit adımlı sürünme dizisi (t_steps konum)."""
    d = p_to - p_from
    u = d / (np.linalg.norm(d) + 1e-9)
    return [p_from + u * step * k for k in range(t_steps)]


def _build_pass_scene():
    """A (id0, takım0) 0..30 taşır, sonra B (id1, takım0)'ye pas; C (id2) uzak.

    Oyuncular: 0=A, 1=B, 2=C (hepsi takım0), 3/4/5 = pres-edenler (takım1).
    Pres-edenler 0..30 A'ya, 31.. B'ye sürünür (31'de konum sürekli).
    """
    A = np.array([12.0, 10.0])
    B = np.array([22.0, 13.0])
    C = np.array([35.0, 5.0])
    T = 60
    SW = 31  # pas karesi
    K = 6
    pos = np.full((T, K, 2), np.nan)

    # sabit taşıyıcı/uzak oyuncular
    pos[:, 0] = A
    pos[:, 1] = B
    pos[:, 2] = C

    # pres-edenler: faz A (0..30) A'ya sürün
    end_ph1 = [None, None, None]
    for t in range(SW):
        ring = _press_ring(A, t)
        for k in range(3):
            pos[t, 3 + k] = ring[k]
        end_ph1 = ring
    # faz B (31..): 30. kareden süreklilikle B'ye sürün
    for k in range(3):
        seq = _toward(end_ph1[k], B, T - SW)
        for j, xy in enumerate(seq):
            pos[SW + j, 3 + k] = xy

    teams = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
    return pos, teams, A, B, SW


def _build_flicker_scene():
    """A (id0) tüm dizi boyunca taşır; t=FL'de savunanlar TEK kare uzak C'yi hedefler.

    Pres-edenler her kare "aim" noktasına sabit DM adımıyla yaklaşır (integrate-
    toward-aim). aim = A her zaman, AMA t=FL'de aim = C: o tek karede hız-ışınları
    (yayılmış konumlardan C'ye) C'de kesişir => odak C'ye sıçrar; FL+1'de aim
    yeniden A olur => odak A'ya temiz döner (tam 1-kare titreme). Açgözlü sahip
    o karede çapraz-sahadaki C'ye titrer; Viterbi bastırmalı (sahip A kalır).
    """
    A = np.array([12.0, 10.0])
    C = np.array([34.0, 4.0])  # uzak köşe (>15 m)
    T = 40
    FL = 20
    DM = 0.15  # kare-başına adım (v_min=0.05'in üstünde)
    K = 5      # 0=A, 1=C(uzak takım-arkadaşı), 2/3/4 = pres-edenler
    pos = np.full((T, K, 2), np.nan)
    pos[:, 0] = A
    pos[:, 1] = C

    # pres-edenleri A çevresinde yay (farklı açılar => iyi-koşullu kesişim)
    for k in range(3):
        pos[0, 2 + k] = A + R0 * np.array([np.cos(DEF_ANGLES[k]), np.sin(DEF_ANGLES[k])])
    for t in range(1, T):
        aim = C if t == FL else A
        for k in range(3):
            prev = pos[t - 1, 2 + k]
            u = aim - prev
            u = u / (np.linalg.norm(u) + 1e-9)
            pos[t, 2 + k] = prev + DM * u
    return pos, A, C, FL


def _build_turnover_scene():
    """A (id0, takım0) 0..30 taşır, sonra top E (id1, takım1)'e geçer (turnover).

    Pres-edenler (id 2/3/4) faz1 A'ya, faz2 E'ye sürünür.
    """
    A = np.array([12.0, 10.0])
    E = np.array([24.0, 12.0])
    T = 60
    SW = 31
    K = 5  # 0=A(t0), 1=E(t1), 2/3/4 pres
    pos = np.full((T, K, 2), np.nan)
    pos[:, 0] = A
    pos[:, 1] = E

    end_ph1 = [None, None, None]
    for t in range(SW):
        ring = _press_ring(A, t)
        for k in range(3):
            pos[t, 2 + k] = ring[k]
        end_ph1 = ring
    for k in range(3):
        seq = _toward(end_ph1[k], E, T - SW)
        for j, xy in enumerate(seq):
            pos[SW + j, 2 + k] = xy

    teams = {0: 0, 1: 1, 2: 0, 3: 1, 4: 0}
    return pos, teams, SW


# --------------------------------------------------------------------------- #
# convergence_point birim testi (bilinen kesişim)
# --------------------------------------------------------------------------- #
def test_convergence_point_recovers_known_focus():
    rng = np.random.default_rng(0)
    target = np.array([5.0, 7.0])
    pts = np.array([[0.0, 0.0], [12.0, 1.0], [3.0, 14.0], [11.0, 13.0]])
    dirs = []
    for p in pts:
        d = target - p
        d = d / np.linalg.norm(d)
        d = d + rng.normal(0, 0.01, size=2)  # küçük gürültü
        dirs.append(d)
    x, ok = P.convergence_point(pts, np.array(dirs))
    assert ok
    assert np.linalg.norm(x - target) < 0.3


def test_convergence_point_parallel_is_degenerate():
    pts = np.array([[0.0, 0.0], [0.0, 5.0], [0.0, 9.0]])
    dirs = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])  # hepsi paralel
    x, ok = P.convergence_point(pts, dirs)
    assert ok is False
    # yedek = nokta ortalaması
    assert np.allclose(x, pts.mean(axis=0))


# --------------------------------------------------------------------------- #
# Odak gerçekten taşıyanı buluyor mu?
# --------------------------------------------------------------------------- #
def test_focus_tracks_the_carrier():
    pos, teams, A, B, SW = _build_pass_scene()
    det = P.infer_possession(pos, team_labels=teams, return_details=True)
    focus = det["focus"]
    # faz1 ortasında odak A'ya, faz2 ortasında B'ye yakın olmalı
    assert np.linalg.norm(focus[15] - A) < 2.0
    assert np.linalg.norm(focus[45] - B) < 2.0


# --------------------------------------------------------------------------- #
# PAS: A -> B (aynı takım)
# --------------------------------------------------------------------------- #
def test_pass_event_fires_at_handoff():
    pos, teams, A, B, SW = _build_pass_scene()
    possessors, events = P.infer_possession(pos, team_labels=teams)

    # faz1 sahip = A(0), faz2 sahip = B(1)
    assert possessors[10] == 0
    assert possessors[20] == 0
    assert possessors[50] == 1
    assert possessors[58] == 1

    # tam olarak bir 0->1 değişimi, 'pass', devir penceresinde
    changes = [e for e in events if e[1] == 0 and e[2] == 1]
    assert len(changes) == 1
    frame, fr, to, kind = changes[0]
    assert kind == "pass"
    assert abs(frame - SW) <= 3
    # hiç turnover olmamalı (hepsi aynı takım)
    assert all(e[3] != "turnover" for e in events)


# --------------------------------------------------------------------------- #
# VITERBI: 1-kare çapraz-saha titremesini bastır
# --------------------------------------------------------------------------- #
def test_viterbi_suppresses_cross_pitch_flicker():
    pos, A, C, FL = _build_flicker_scene()
    det = P.infer_possession(pos, return_details=True)

    greedy = det["greedy"]
    possessors = det["possessors"]

    # açgözlü (yumuşatmasız) sahip titreme karesinde çapraz-sahadaki C'ye sıçramalı
    # (Viterbi'nin gerçekten iş gördüğünü kanıtlar)
    assert greedy[FL] != 0
    flick_pid = greedy[FL]
    fi = det["pids"].index(flick_pid)
    assert np.linalg.norm(det["pos"][FL, fi] - A) > 15.0  # çapraz-saha
    # açgözlü TEK kare titremeli; komşu kareler temiz A olmalı
    assert greedy[FL - 1] == 0 and greedy[FL + 1] == 0

    # Viterbi titremeyi yutmalı: sahip her karede A(0), hiç olay yok
    assert all(p == 0 for p in possessors)
    _, events = P.infer_possession(pos)
    assert events == []

    # KONTROL: bastırma gerçekten geçiş cezasından gelir, veriden değil.
    # Cezalar ~0 ise Viterbi açgözlüye iner ve titreme HAYATTA kalır.
    poss_nopen, _ = P.infer_possession(
        pos, switch_penalty=0.0, switch_dist_weight=0.0
    )
    assert poss_nopen[FL] == flick_pid  # cezasız => titreme geçer

    # KONTROL: sabit ceza TEK BAŞINA yetmez (çapraz-saha kazancı büyük);
    # mesafe-ölçekli ceza ("anında saha geçemez") yükü taşıyan terimdir.
    poss_baseonly, _ = P.infer_possession(
        pos, switch_penalty=8.0, switch_dist_weight=0.0
    )
    assert poss_baseonly[FL] == flick_pid  # yalnız base => hâlâ titrer
    poss_dist, _ = P.infer_possession(
        pos, switch_penalty=0.0, switch_dist_weight=0.6
    )
    assert poss_dist[FL] == 0  # mesafe cezası => bastırılır


# --------------------------------------------------------------------------- #
# TURNOVER: karşı takıma geçiş
# --------------------------------------------------------------------------- #
def test_turnover_with_team_labels():
    pos, teams, SW = _build_turnover_scene()
    possessors, events = P.infer_possession(pos, team_labels=teams)

    assert possessors[10] == 0   # A (takım0)
    assert possessors[55] == 1   # E (takım1)

    changes = [e for e in events if e[1] == 0 and e[2] == 1]
    assert len(changes) == 1
    frame, fr, to, kind = changes[0]
    assert kind == "turnover"
    assert abs(frame - SW) <= 3


def test_same_change_without_labels_is_pass_not_turnover():
    pos, teams, SW = _build_turnover_scene()
    # etiket YOK => aynı sahip değişimi 'pass' (turnover ayrımı etiket gerektirir)
    _, events = P.infer_possession(pos, team_labels=None)
    changes = [e for e in events if e[1] == 0 and e[2] == 1]
    assert len(changes) == 1
    assert changes[0][3] == "pass"
    assert all(e[3] == "pass" for e in events)


# --------------------------------------------------------------------------- #
# Girdi-biçimi esnekliği: list[dict] aynı sonucu vermeli
# --------------------------------------------------------------------------- #
def test_list_of_dict_input_matches_array():
    pos, teams, A, B, SW = _build_pass_scene()
    T, K, _ = pos.shape
    frames = []
    for t in range(T):
        frames.append({k: pos[t, k].copy() for k in range(K)})
    p_arr, _ = P.infer_possession(pos, team_labels=teams)
    p_lst, _ = P.infer_possession(frames, team_labels=teams)
    assert p_arr == p_lst
