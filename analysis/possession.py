#!/usr/bin/env python3
"""possession — top-bağımsız (ball-free) POSSESSION + pass/turnover çıkarımı.

PathCRF'in (Mughal et al., "PathCRF: structured path inference for sports
tracking", arXiv:2602.12080) basitleştirilmiş, TOP-İZSİZ bir uyarlamasıdır.
PathCRF'in temel fikri: ayrık sahiplenme (possession) etiketlerinin bir
zaman-dizisini, her-kare birli (unary) bir uyum skoru ile ardışık-kareler arası
ikili (pairwise) bir geçiş cezasını birleştiren bir zincir-CRF / Viterbi yolu
olarak çözmek. Orijinal yöntem TOP konumunu gözlem olarak kullanır.

DÜRÜST UYARI (kör nokta): Burada bir top izi YOKTUR. Sahibi, oyuncuları bir
graf olarak modelleyip her karede "oyun-odağı" (play-focus / convergence point)
adını verdiğimiz bir PROXY noktaya en yakın makul oyuncuyu seçerek TAHMİN ederiz.
Oyun-odağı, hareket eden oyuncuların hız ışınlarının en-küçük-kareler kesişim
noktasıdır (savunanlar topu taşıyana baskı yapar => ışınları taşıyanda kesişir).
Yeterli hareket yoksa odak, oyuncu kütle-merkezine düşer. Bu bir TAHMİNDİR, yer
gerçeği (ground truth) değildir; gerçek bir top izi gelirse `estimate_focus_track`
çıktısını doğrudan top konumlarıyla değiştirmek tek entegrasyon dikişidir
(geri kalan birli/ikili + Viterbi mantığı aynı kalır).

Zaman-yumuşatma (Viterbi/DP): bir sahip kareler arasında sahanın bir ucundan
diğerine ANINDA atlayamaz. Geçiş cezası iki bileşenlidir:
  base `switch_penalty`        — her sahip değişimine sabit ceza (1-kare titreme/
                                  flicker'ı bastırır: titreme 2 değişim ister),
  `switch_dist_weight` * mesafe — eski sahibin konumu ile yeni sahibin konumu
                                  arası saha-mesafesi (çapraz-saha sıçramayı
                                  ağır cezalandırır => "anında atlayamaz").

Bir sahip DEĞİŞİMİ = bir pas (aynı takım) ya da top kaybı/turnover (karşı takım,
takım etiketleri verilmişse). Takım etiketi yoksa değişim varsayılan 'pass'
olarak işaretlenir (turnover ayrımı takım etiketi gerektirir).

Koordinat sözleşmesi (repo geneli): saha metrik X = uzunluk (length),
Y = genişlik (width). Birim metre. Bağımlılık: yalnızca numpy (BSD). GPU yok,
harici model yok.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

PositionsInput = Union[np.ndarray, Sequence]
TeamLabels = Union[None, Dict, Sequence]


# --------------------------------------------------------------------------- #
# Girdi normalizasyonu
# --------------------------------------------------------------------------- #
def _normalize_positions(
    positions_per_frame: PositionsInput,
) -> Tuple[List, np.ndarray, np.ndarray]:
    """positions_per_frame -> (pids, pos[T,K,2], present[T,K]).

    Kabul edilen biçimler:
      * np.ndarray (T, P, 2): oyuncu kimlikleri 0..P-1; NaN => o karede yok.
      * list[dict]          : her kare {pid: (x, y)}; eksik pid => o karede yok.
      * list[array (m,2)]   : her kare m oyuncu, kimlik = satır indeksi 0..m-1.

    pos saha-metre cinsindendir; yok olan girişler NaN. present bool maskedir.
    """
    if isinstance(positions_per_frame, np.ndarray) and positions_per_frame.ndim == 3:
        arr = np.asarray(positions_per_frame, dtype=float)
        if arr.shape[2] != 2:
            raise ValueError("ndarray girdisi (T, P, 2) olmalı")
        present = np.isfinite(arr).all(axis=2)
        pids = list(range(arr.shape[1]))
        return pids, arr, present

    frames = list(positions_per_frame)
    T = len(frames)
    if T == 0:
        raise ValueError("positions_per_frame boş")

    parsed: List[Dict] = []
    pidset = set()
    for fr in frames:
        d: Dict = {}
        if isinstance(fr, dict):
            for k, v in fr.items():
                d[k] = np.asarray(v, dtype=float)
        else:
            a = np.asarray(fr, dtype=float)
            if a.ndim != 2 or a.shape[1] != 2:
                raise ValueError("kare biçimi dict ya da (m, 2) dizisi olmalı")
            for i, xy in enumerate(a):
                d[i] = xy
        parsed.append(d)
        pidset.update(d.keys())

    try:
        pids = sorted(pidset)
    except TypeError:  # karışık tip kimlikler => stringe göre sırala
        pids = sorted(pidset, key=lambda p: (str(type(p)), str(p)))
    idx = {p: i for i, p in enumerate(pids)}
    K = len(pids)

    pos = np.full((T, K, 2), np.nan, dtype=float)
    present = np.zeros((T, K), dtype=bool)
    for t, d in enumerate(parsed):
        for p, xy in d.items():
            if xy.shape == (2,) and np.isfinite(xy).all():
                i = idx[p]
                pos[t, i] = xy
                present[t, i] = True
    return pids, pos, present


def _velocities(pos: np.ndarray, present: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Kare-başına anlık hız (geri-fark; t=0 için ileri-fark) ve hız büyüklüğü.

    Hız konum farkıdır (metre/kare). Komşu karelerden biri eksikse NaN.
    """
    T = pos.shape[0]
    vel = np.full_like(pos, np.nan)
    if T >= 2:
        vel[0] = pos[1] - pos[0]
        vel[1:] = pos[1:] - pos[:-1]
    speed = np.linalg.norm(vel, axis=2)  # NaN otomatik yayılır
    return vel, speed


# --------------------------------------------------------------------------- #
# Oyun-odağı (play-focus) tahmini
# --------------------------------------------------------------------------- #
def convergence_point(
    points: np.ndarray,
    directions: np.ndarray,
    weights: Optional[np.ndarray] = None,
    cond_thresh: float = 1e8,
) -> Tuple[np.ndarray, bool]:
    """Ağırlıklı en-küçük-kareler ışın-kesişim noktası.

    Her oyuncu, `points[k]` noktasından `directions[k]` yönünde bir DOĞRU (ışın)
    tanımlar. Aranan x noktası, tüm doğrulara olan ağırlıklı dik-uzaklık
    karelerinin toplamını minimize eder:
        min_x  sum_k w_k * || (I - u_k u_k^T) (x - p_k) ||^2 ,   u_k = birim yön.
    Normal denklem (sum_k w_k M_k) x = sum_k w_k M_k p_k,  M_k = I - u_k u_k^T.

    Dönüş: (x, ok). Işınlar ~paralel (sistem kötü-koşullu) ise ok=False ve x =
    noktaların ağırlıklı ortalaması (çağıran tarafça yedek olarak kullanılır).
    """
    P = np.asarray(points, dtype=float)
    D = np.asarray(directions, dtype=float)
    m = P.shape[0]
    if weights is None:
        w = np.ones(m)
    else:
        w = np.asarray(weights, dtype=float)

    centroid = np.average(P, axis=0, weights=w) if m else np.zeros(2)
    if m < 2:
        return centroid, False

    nrm = np.linalg.norm(D, axis=1)
    good = nrm > 1e-12
    if good.sum() < 2:
        return centroid, False
    U = D[good] / nrm[good, None]
    Pg = P[good]
    wg = w[good]

    I2 = np.eye(2)
    A = np.zeros((2, 2))
    b = np.zeros(2)
    for u, p, wk in zip(U, Pg, wg):
        M = I2 - np.outer(u, u)
        A += wk * M
        b += wk * (M @ p)

    ev = np.linalg.eigvalsh(A)  # artan sırada, >=0 (A pozitif yarı-tanımlı)
    if not np.all(np.isfinite(ev)) or ev[1] <= 0 or ev[0] <= ev[1] / cond_thresh:
        # rank-eksik: tüm yönler ~aynı (paralel ışınlar) => kesişim belirsiz
        return centroid, False
    x = np.linalg.solve(A, b)
    if not np.all(np.isfinite(x)):
        return centroid, False
    return x, True


def estimate_focus_track(
    pos: np.ndarray,
    present: np.ndarray,
    vel: np.ndarray,
    speed: np.ndarray,
    v_min: float = 0.05,
    clip_margin: float = 3.0,
    focus_smooth: float = 0.0,
) -> np.ndarray:
    """Kare-başına oyun-odağı izi (T, 2).

    Her karede hareket eden (speed >= v_min) oyuncuların hız ışınlarının
    kesişimi. Yetersiz hareket / kötü-koşul => mevcut oyuncuların kütle-merkezi.
    Odak, mevcut oyuncuların sınırlayıcı kutusu + clip_margin içine kırpılır
    (paralel-ışın kaynaklı kaçak çözümleri engeller). focus_smooth>0 ise hafif
    EMA zaman-yumuşatması (varsayılan 0 = kapalı; Viterbi asıl yumuşatıcıdır).
    """
    T = pos.shape[0]
    focus = np.full((T, 2), np.nan)
    prev = None
    for t in range(T):
        pmask = present[t]
        if not pmask.any():
            focus[t] = prev if prev is not None else np.zeros(2)
            continue
        pts_all = pos[t, pmask]
        centroid = pts_all.mean(axis=0)

        mv = pmask & np.isfinite(speed[t]) & (speed[t] >= v_min)
        if mv.sum() >= 2:
            f, ok = convergence_point(pos[t, mv], vel[t, mv], speed[t, mv])
            if not ok:
                f = centroid
        else:
            f = centroid

        lo = pts_all.min(axis=0) - clip_margin
        hi = pts_all.max(axis=0) + clip_margin
        f = np.clip(f, lo, hi)

        if focus_smooth > 0.0 and prev is not None:
            f = (1.0 - focus_smooth) * f + focus_smooth * prev
        focus[t] = f
        prev = f
    return focus


def _unary_costs(pos: np.ndarray, present: np.ndarray, focus: np.ndarray) -> np.ndarray:
    """Birli maliyet U[t,k] = oyuncu k'nin odağa saha-uzaklığı (metre).

    Yok olan oyuncuya +inf (o karede sahip olamaz). Düşük maliyet = makul sahip.
    """
    T, K = present.shape
    U = np.full((T, K), np.inf)
    for t in range(T):
        pmask = present[t]
        if pmask.any():
            U[t, pmask] = np.linalg.norm(pos[t, pmask] - focus[t], axis=1)
    return U


# --------------------------------------------------------------------------- #
# Viterbi yol-çözümü (zaman-yumuşatma)
# --------------------------------------------------------------------------- #
def _viterbi(
    U: np.ndarray,
    pos: np.ndarray,
    switch_penalty: float,
    switch_dist_weight: float,
) -> np.ndarray:
    """Toplam maliyeti minimize eden sahip-dizisini çöz (durum indeksleri).

    maliyet = sum_t U[t, s_t] + sum_t Trans(s_{t-1} -> s_t),
    Trans(i->j) = 0 (i==j) veya switch_penalty + switch_dist_weight *
                  || pos[t-1, i] - pos[t, j] ||  (i != j).
    """
    T, K = U.shape
    dp = np.full((T, K), np.inf)
    back = np.full((T, K), -1, dtype=int)
    dp[0] = U[0]

    for t in range(1, T):
        prev = pos[t - 1]  # (K, 2)
        cur = pos[t]       # (K, 2)
        D = np.linalg.norm(prev[:, None, :] - cur[None, :, :], axis=2)  # (K,K)
        Trans = switch_penalty + switch_dist_weight * D
        np.fill_diagonal(Trans, 0.0)
        Trans = np.where(np.isfinite(Trans), Trans, np.inf)
        cand = dp[t - 1][:, None] + Trans  # (i, j)
        best_i = np.argmin(cand, axis=0)
        dp[t] = U[t] + cand[best_i, np.arange(K)]
        back[t] = best_i

    states = np.zeros(T, dtype=int)
    states[T - 1] = int(np.argmin(dp[T - 1]))
    for t in range(T - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    return states


# --------------------------------------------------------------------------- #
# Takım etiketi yardımcıları
# --------------------------------------------------------------------------- #
def _team_lookup(team_labels: TeamLabels, pids: List):
    """team_labels -> (pid -> team) okuyucu fonksiyonu (None = bilinmiyor)."""
    if team_labels is None:
        return lambda p: None
    if isinstance(team_labels, dict):
        return lambda p: team_labels.get(p, None)
    seq = list(team_labels)  # indeks = pid varsayımı (0..K-1 kimlikler için)
    idx = {p: i for i, p in enumerate(pids)}

    def lookup(p):
        i = idx.get(p, None)
        if i is None or i >= len(seq):
            return None
        return seq[i]

    return lookup


# --------------------------------------------------------------------------- #
# Ana API
# --------------------------------------------------------------------------- #
def infer_possession(
    positions_per_frame: PositionsInput,
    team_labels: TeamLabels = None,
    switch_penalty: float = 4.0,
    switch_dist_weight: float = 0.6,
    v_min: float = 0.05,
    focus_smooth: float = 0.0,
    clip_margin: float = 3.0,
    return_details: bool = False,
):
    """Top-bağımsız sahiplenme + pas/turnover çıkarımı (PathCRF-basit).

    Parametreler
    ----------
    positions_per_frame : (T,P,2) dizi | list[dict{pid:(x,y)}] | list[(m,2)].
    team_labels         : {pid: team} | indeksli dizi | None. None ise değişim
                          'pass' varsayılır (turnover ayrımı etiket gerektirir).
    switch_penalty      : sahip değişimine sabit ceza (metre cinsi maliyet).
    switch_dist_weight  : pas mesafesine bağlı ek ceza katsayısı.
    v_min               : ışın oyu için min hız (metre/kare).
    focus_smooth        : odak EMA katsayısı (0 = kapalı).
    return_details      : True ise ara büyüklükleri de döndürür.

    Dönüş
    -----
    return_details=False: (possessors, events)
      possessors : uzunluk T liste; her kare için sahip pid ya da None.
      events     : list[(frame, from_pid, to_pid, kind)], kind in {'pass',
                   'turnover'}; sahip değiştiği karede üretilir.
    return_details=True: yukarıdakileri + focus/unary/greedy/pids içeren dict.
    """
    pids, pos, present = _normalize_positions(positions_per_frame)
    T, K = present.shape
    vel, speed = _velocities(pos, present)
    focus = estimate_focus_track(
        pos, present, vel, speed, v_min=v_min,
        clip_margin=clip_margin, focus_smooth=focus_smooth,
    )
    U = _unary_costs(pos, present, focus)

    states = _viterbi(U, pos, switch_penalty, switch_dist_weight)

    # durum indeksi -> pid (o karede oyuncu yoksa None)
    possessors: List = []
    for t in range(T):
        s = states[t]
        if present[t, s]:
            possessors.append(pids[s])
        else:
            possessors.append(None)

    team_of = _team_lookup(team_labels, pids)
    events: List[Tuple] = []
    prev_p = None
    for t in range(T):
        p = possessors[t]
        if p is None:
            continue
        if prev_p is not None and p != prev_p:
            ta, tb = team_of(prev_p), team_of(p)
            kind = "turnover" if (ta is not None and tb is not None and ta != tb) else "pass"
            events.append((t, prev_p, p, kind))
        prev_p = p

    if not return_details:
        return possessors, events

    # açgözlü (greedy, yumuşatmasız) sahip — Viterbi'nin işini görünür kılar
    greedy: List = []
    for t in range(T):
        if np.isfinite(U[t]).any():
            greedy.append(pids[int(np.argmin(U[t]))])
        else:
            greedy.append(None)

    return {
        "possessors": possessors,
        "events": events,
        "focus": focus,
        "unary": U,
        "greedy": greedy,
        "states": states,
        "pids": pids,
        "pos": pos,
        "present": present,
    }
