#!/usr/bin/env python3
"""game_state — pozisyon-temelli oyun-durumu olayları: KICKOFF/restart + defansif WALL.

Bu modül takım-pozisyonlarından (saha-metresi) iki taktik olayı YALNIZCA geometriyle
tespit eder; renk/forma/top tespiti gerektirmez (o yüzden re-ID/top-bağımsız, sahada
sağlam). İki primitif:

  KICKOFF (başlangıç vuruşu / gol-sonrası restart):
    Kısa bir pencerede her iki takım orta çizgiyle TEMİZ ayrılmış (her takım kendi
    yarısına hapsolmuş), >=1 oyuncu orta noktanın r yarıçapı içinde, genel hız düşük
    ("set" / hazır hâli). Ardından bir takımın AĞIRLIK MERKEZİ orta çizgiyi geçer ->
    o zaman damgasında kickoff ateşlenir. Bu, SoccerCPD'nin (arXiv:2206.10926) formasyon
    bölütleme fikrinin pozisyon-only, etiketsiz bir özel hâli: formasyonun simetrik
    "iki yarı" snapshot'ını ararız.

  WALL (serbest vuruş duvarı):
    Savunan kümede 2-4 oyuncu ki (a) DOĞRUSAL (düşük çizgi-uyum artığı), (b) ~eşit
    aralıklı, (c) (restart_noktası -> savunulan_kale) hattına ~DİK, (d) restart
    noktasından ~5-11 m bandında. TacticAI'nin (Nature 2024) takım-sporu simetrilerini
    kullanması fikrinden ilham: skorlama YANSIMA-SİMETRİK kurulur, yani aynalanmış bir
    formasyon AYNI skoru verir (reflection equivariance).

Konvansiyon: saha koordinatı metre; X=uzunluk (length), Y=genişlik (width). Orta çizgi
X = L/2, orta nokta (L/2, W/2). Tracks şeması: tid, frame, t_sec, pitch_x, pitch_y,
team_id (0/1). Bağımlılık: yalnızca numpy/pandas (BSD). GPU yok, harici model yok.

Yansıma-simetri (tasarım sözü): tüm skor bileşenleri (split-margini, orta-noktaya
uzaklık, hız, doğrusallık artığı, aralık-tekdüzeliği, diklik, mesafe-bandı) bir
yansıma X -> L-X (ve takım etiketlerinin takası) altında DEĞİŞMEZ büyüklüklerden
türetilir; bu yüzden aynalanmış bir kickoff/wall birebir aynı skoru üretir.
"""
from __future__ import annotations

import itertools
import math
from typing import Iterable

import numpy as np
import pandas as pd

# --------------------------------------------------------------------- KICKOFF


def _prep_tracks(df: pd.DataFrame) -> pd.DataFrame:
    """Tracks df -> temiz (frame, t_sec, pitch_x, pitch_y, team_id, tid, speed).

    in_pitch varsa sadece saha-içi satırlar; NaN pitch koordinatları atılır; her tid
    için ardışık-kare anlık hızı (m/s) t_sec farkından hesaplanır.
    """
    need = {"pitch_x", "pitch_y", "t_sec", "team_id"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"detect_kickoff: eksik kolon(lar): {sorted(miss)}")
    d = df.copy()
    if "frame" not in d.columns:
        # t_sec'i kareye çevir (sentetik/akış için sağlam fallback)
        d["frame"] = pd.factorize(np.round(d["t_sec"].to_numpy(), 6))[0]
    if "tid" not in d.columns:
        d["tid"] = np.arange(len(d))
    if "in_pitch" in d.columns:
        d = d[d["in_pitch"].astype(bool)]
    d = d[np.isfinite(d["pitch_x"]) & np.isfinite(d["pitch_y"])]
    d = d.sort_values(["tid", "t_sec"]).reset_index(drop=True)
    # per-tid anlık hız
    gx = d.groupby("tid")
    dt = gx["t_sec"].diff()
    dx = gx["pitch_x"].diff()
    dy = gx["pitch_y"].diff()
    with np.errstate(invalid="ignore", divide="ignore"):
        sp = np.hypot(dx, dy) / dt
    sp = sp.replace([np.inf, -np.inf], np.nan)
    # her tid'in ilk karesi: bir sonraki hızla doldur (yoksa 0)
    d["speed"] = sp.to_numpy()
    d["speed"] = d.groupby("tid")["speed"].transform(
        lambda s: s.bfill().fillna(0.0))
    return d


def _two_team_ids(team_ids: np.ndarray) -> tuple:
    """En kalabalık iki takım etiketini döndür (hakem/3.-etiket gürültüsünü ele)."""
    vals, cnt = np.unique(team_ids, return_counts=True)
    order = np.argsort(-cnt)
    top = vals[order][:2]
    return tuple(sorted(top.tolist()))


def _snapshot_score(px, py, team, speed, pitch_dims, p):
    """Tek kare için (readiness, split, center, speed, team_centroid_x, sides) hesapla.

    readiness = split_score * center_score * speed_score, hepsi [0,1] ve yansıma-değişmez.
    sides: {team_id: 'L'|'R'} (orta-nokta-dışı oyuncuların takım centroid X'ine göre).
    team_cx: {team_id: o takımın TÜM oyuncuları üzerinden centroid X} (crossing için).
    """
    L, W = pitch_dims
    half = 0.5 * L
    cx0, cy0 = 0.5 * L, 0.5 * W
    r = p["center_radius_m"]

    px = np.asarray(px, float)
    py = np.asarray(py, float)
    team = np.asarray(team)
    speed = np.asarray(speed, float)

    tids = _two_team_ids(team)
    if len(tids) < 2:
        return dict(readiness=0.0, split=0.0, center=0.0, speed=0.0,
                    team_cx={}, sides={})

    # --- center_score: orta noktaya en yakın oyuncu ---
    dcen = np.hypot(px - cx0, py - cy0)
    d_min = float(dcen.min())
    if d_min <= r:
        center_score = 1.0
    else:
        center_score = math.exp(-(d_min - r) / max(r, 1e-6))
    is_center = dcen <= r  # orta noktadaki oyuncu(lar): vuruşu kullanan, çizgide durabilir

    # --- speed_score: genel düşük hız ---
    v = float(np.nanmean(speed)) if speed.size else 0.0
    vmax = p["max_speed_mps"]
    speed_score = 1.0 if v <= vmax else math.exp(-(v - vmax) / max(vmax, 1e-6))

    # --- split_score: her takım kendi yarısında mı (orta-nokta oyuncusu hariç) ---
    # takım centroid X'i (orta-nokta-dışı) ile L/R ata, sonra yanlış-yarı ihlali ölç.
    masks = {t: (team == t) for t in tids}
    cx_noncenter = {}
    for t in tids:
        m = masks[t] & (~is_center)
        cx_noncenter[t] = float(px[m].mean()) if m.any() else float(px[masks[t]].mean())
    # küçük centroid X = sol (L), büyük = sağ (R)
    left_t = min(tids, key=lambda t: cx_noncenter[t])
    right_t = max(tids, key=lambda t: cx_noncenter[t])
    sides = {left_t: "L", right_t: "R"}

    tol = p["split_tol_m"]
    scale = p["split_scale_m"]
    margins = []
    n_noncenter = 0
    for t in tids:
        m = masks[t] & (~is_center)
        if not m.any():
            continue
        n_noncenter += int(m.sum())
        xs = px[m]
        if sides[t] == "L":
            margin = (half - xs)        # sol takım: half'in solunda olmalı (>0 doğru)
        else:
            margin = (xs - half)        # sağ takım: half'in sağında olmalı
        margins.append(margin + tol)    # tol kadar çizgiyi aşma affı
    if n_noncenter < 2 or left_t == right_t:
        split_score = 0.0
    else:
        m_min = float(np.min(np.concatenate(margins)))
        split_score = 1.0 if m_min >= 0.0 else math.exp(m_min / max(scale, 1e-6))

    # --- takım centroid X (TÜM oyuncular) crossing için ---
    team_cx = {t: float(px[masks[t]].mean()) for t in tids}

    readiness = split_score * center_score * speed_score
    return dict(readiness=readiness, split=split_score, center=center_score,
                speed=speed_score, team_cx=team_cx, sides=sides)


def _kickoff_snapshots(df: pd.DataFrame, pitch_dims, p) -> list:
    """Her kare için snapshot skoru (zaman-sıralı liste)."""
    d = _prep_tracks(df)
    snaps = []
    for fr, g in d.groupby("frame", sort=True):
        t = float(g["t_sec"].mean())
        s = _snapshot_score(g["pitch_x"].to_numpy(), g["pitch_y"].to_numpy(),
                            g["team_id"].to_numpy(), g["speed"].to_numpy(),
                            pitch_dims, p)
        s["t"] = t
        s["frame"] = int(fr)
        snaps.append(s)
    snaps.sort(key=lambda s: s["t"])
    return snaps


_KICK_DEFAULTS = dict(
    center_radius_m=3.5,    # orta nokta yarıçapı (vuruşu kullanan oyuncu)
    max_speed_mps=1.0,      # "düşük hız" eşiği (set hâli ~ duruyor)
    split_tol_m=1.0,        # çizgiyi aşma affı (oyuncu çizgide durabilir)
    split_scale_m=2.0,      # yanlış-yarı ihlali yumuşaklığı
    ready_thresh=0.5,       # readiness >= bu -> "hazır" kare
    min_ready_sec=0.4,      # hazır hâl bu kadar sürmeli (set onayı)
    disarm_sec=6.0,         # armed olduktan sonra crossing için zaman penceresi
    cross_margin_m=0.0,     # centroid çizgiyi bu kadar geçince crossing sayılır
    cooldown_sec=8.0,       # bir kickoff sonrası tekrar ateşleme yasağı
)


def kickoff_readiness(df: pd.DataFrame, pitch_dims, **kw) -> pd.DataFrame:
    """Kare-başına kickoff "hazır-ol" (set) skor serisi.

    Döner: DataFrame[frame, t_sec, readiness, split, center, speed]. readiness ve tüm
    bileşenleri yansıma X->L-X (+ takım takası) altında DEĞİŞMEZ; bu fonksiyon
    simetri-testinin de giriş noktasıdır.
    """
    p = {**_KICK_DEFAULTS, **kw}
    snaps = _kickoff_snapshots(df, pitch_dims, p)
    return pd.DataFrame([
        dict(frame=s["frame"], t_sec=s["t"], readiness=s["readiness"],
             split=s["split"], center=s["center"], speed=s["speed"])
        for s in snaps
    ])


def detect_kickoff(df: pd.DataFrame, pitch_dims, **kw) -> list:
    """Pozisyon-only kickoff/restart tespiti -> ateşlenen zaman damgaları (saniye).

    Akış: (1) snapshot readiness'i hesapla; (2) readiness >= ready_thresh kesintisiz
    min_ready_sec sürünce "armed" ol ve set-yanlarını (hangi takım L/R) kaydet; (3) armed
    iken bir takımın AĞIRLIK MERKEZİ orta çizgiyi set-yanından karşıya geçtiği ilk karede
    o t_sec'i ateşle; (4) cooldown_sec boyunca yeniden ateşleme yok.

    Parametreler kw ile geçilebilir (bkz. _KICK_DEFAULTS).
    """
    p = {**_KICK_DEFAULTS, **kw}
    L, _ = pitch_dims
    half = 0.5 * L
    snaps = _kickoff_snapshots(df, pitch_dims, p)
    if not snaps:
        return []

    fires = []
    ready_start_t = None
    armed = False
    armed_sides = None
    armed_t = None
    cooldown_until = -np.inf

    for s in snaps:
        t = s["t"]
        if t < cooldown_until:
            ready_start_t = None
            armed = False
            continue

        # --- hazır-ol takibi ---
        if s["readiness"] >= p["ready_thresh"] and s["sides"]:
            if ready_start_t is None:
                ready_start_t = t
            if (t - ready_start_t) >= p["min_ready_sec"]:
                armed = True
                armed_sides = dict(s["sides"])     # en güncel set-yanları
                armed_t = t
        else:
            ready_start_t = None

        # --- armed iken crossing ara ---
        if armed and armed_sides is not None:
            if (t - armed_t) > p["disarm_sec"]:
                armed = False
                armed_sides = None
                continue
            crossed = False
            for tid, side in armed_sides.items():
                cx = s["team_cx"].get(tid)
                if cx is None:
                    continue
                if side == "L" and cx > half + p["cross_margin_m"]:
                    crossed = True
                elif side == "R" and cx < half - p["cross_margin_m"]:
                    crossed = True
            if crossed and t > armed_t:
                fires.append(float(t))
                armed = False
                armed_sides = None
                ready_start_t = None
                cooldown_until = t + p["cooldown_sec"]

    return fires


# ------------------------------------------------------------------------ WALL


_WALL_DEFAULTS = dict(
    dist_min_m=5.0,         # restart noktasından duvara min mesafe (futsal ~5 m)
    dist_max_m=11.0,        # max (futbol ~9.15 m'yi de kapsar)
    dist_margin_m=3.0,      # bant dışı yumuşak çürüme
    collinear_rms_m=0.6,    # doğrusallık: dik artık RMS bunda 0'a iner
    spacing_cv_tol=0.6,     # aralık tekdüzeliği: gap'lerin CV'si bunda 0'a iner
    gap_min_m=0.4,          # ardışık duvar oyuncuları arası makul boşluk
    gap_max_m=3.0,
    gap_margin_m=1.5,
    perp_tol_deg=35.0,      # (restart->kale) hattına diklikten sapma toleransı
    align_tol_m=3.0,        # duvar merkezinin restart->kale hattına yanal kayması
    search_radius_m=14.0,   # restart çevresinde aday oyuncu yarıçapı
    min_wall=2,
    max_wall=4,
    score_thresh=0.6,       # geometrik-ortalama skoru bunun üstündeyse duvar var
)


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _score_wall_subset(pts: np.ndarray, restart_pt, goal_pt, p) -> float:
    """Tek aday alt-kümenin duvar skoru [0,1] (yansıma-değişmez bileşenlerin geo-ort.).

    Bileşenler: doğrusallık, aralık-tekdüzeliği, aralık-büyüklüğü, diklik, mesafe-bandı,
    hat-hizalama. Hepsi mesafe/açı gibi yansıma altında korunan büyüklüklerden türer.
    """
    pts = np.asarray(pts, float)
    restart = np.asarray(restart_pt, float)
    goal = np.asarray(goal_pt, float)

    # --- doğrusal uyum (PCA / total least squares) ---
    c = pts.mean(axis=0)
    Q = pts - c
    # 2x2 kovaryans -> öz-vektörler
    cov = Q.T @ Q
    evals, evecs = np.linalg.eigh(cov)           # artan sırada
    direction = evecs[:, -1]                      # en büyük öz-değer = çizgi yönü
    normal = evecs[:, 0]                          # dik yön
    perp_res = Q @ normal                         # her noktanın dik artığı
    rms = float(np.sqrt(np.mean(perp_res ** 2)))
    collinear = _clip01(1.0 - rms / p["collinear_rms_m"])

    # --- çizgi üzerinde projeksiyon -> ardışık aralıklar ---
    # k>=2 olduğundan en az bir aralık var; k=2'de tek aralık -> CV=0 (trivial tekdüze).
    proj = np.sort(Q @ direction)
    gaps = np.diff(proj)
    mean_gap = float(gaps.mean())
    cv = float(gaps.std() / mean_gap) if mean_gap > 1e-6 else 1.0
    spacing = _clip01(1.0 - cv / p["spacing_cv_tol"])
    # ardışık boşluk büyüklüğü makul mü (duvar oyuncuları omuz-omuza ~0.5-2 m)
    g = abs(mean_gap)
    if p["gap_min_m"] <= g <= p["gap_max_m"]:
        gap_size = 1.0
    else:
        deficit = max(p["gap_min_m"] - g, g - p["gap_max_m"], 0.0)
        gap_size = _clip01(1.0 - deficit / p["gap_margin_m"])

    # --- diklik: çizgi yönü (restart->kale) hattına dik mi ---
    rg = goal - restart
    nrg = np.linalg.norm(rg)
    if nrg < 1e-6:
        return 0.0
    rg_hat = rg / nrg
    cos_to_rg = abs(float(direction @ rg_hat))    # 0 -> tam dik
    cmax = math.sin(math.radians(p["perp_tol_deg"]))
    perp = _clip01(1.0 - cos_to_rg / max(cmax, 1e-6))

    # --- mesafe bandı: duvar merkezi restart'tan ~5-11 m ---
    dist = float(np.linalg.norm(c - restart))
    if p["dist_min_m"] <= dist <= p["dist_max_m"]:
        distance = 1.0
    else:
        deficit = max(p["dist_min_m"] - dist, dist - p["dist_max_m"], 0.0)
        distance = _clip01(1.0 - deficit / p["dist_margin_m"])

    # --- hizalama: duvar merkezi restart->kale hattının üstünde (yanal kayma küçük) ---
    rel = c - restart
    along = float(rel @ rg_hat)                    # hat yönünde ilerleme
    lateral = float(np.linalg.norm(rel - along * rg_hat))
    align_lat = _clip01(1.0 - lateral / p["align_tol_m"])
    # duvar restart ile kale ARASINDA olmalı (along pozitif ve segmenti aşmamalı)
    if along <= 0.0:
        between = 0.0
    elif along <= nrg:
        between = 1.0
    else:
        between = _clip01(1.0 - (along - nrg) / p["align_tol_m"])
    align = align_lat * between

    factors = np.array([collinear, spacing, gap_size, perp, distance, align],
                       dtype=float)
    if np.any(factors <= 0.0):
        return 0.0
    return float(np.exp(np.mean(np.log(factors))))   # geometrik ortalama


def detect_wall(positions, restart_pt, goal_pt, **kw):
    """Savunan kümede serbest-vuruş duvarı var mı? -> (fired: bool, score: float).

    positions: (N,2) savunan oyuncu konumları (saha-metresi). restart_pt: serbest vuruş
    noktası (2,). goal_pt: savunulan kale merkezi (2,). restart çevresindeki
    search_radius_m içindeki oyuncuların 2..max_wall'lık tüm alt-kümeleri puanlanır;
    en yüksek skor eşik üstündeyse duvar (fired=True) döner.

    Skor yansıma-simetrik: positions/restart/goal birlikte aynalanırsa skor aynı kalır.
    Ek olarak en iyi alt-kümeyi de görmek için best_wall(...) kullanılabilir.
    """
    fired, score, _ = best_wall(positions, restart_pt, goal_pt, **kw)
    return fired, score


def best_wall(positions, restart_pt, goal_pt, **kw):
    """detect_wall'ın açık hâli -> (fired, best_score, best_members_idx)."""
    p = {**_WALL_DEFAULTS, **kw}
    pos = np.asarray(positions, float)
    if pos.ndim != 2 or pos.shape[1] != 2:
        raise ValueError("positions (N,2) olmalı")
    restart = np.asarray(restart_pt, float)
    goal = np.asarray(goal_pt, float)

    ok = np.isfinite(pos).all(axis=1)
    idx_all = np.where(ok)[0]
    if idx_all.size < p["min_wall"]:
        return False, 0.0, ()

    # restart çevresindeki adaylar
    d = np.linalg.norm(pos[idx_all] - restart, axis=1)
    cand = idx_all[d <= p["search_radius_m"]]
    if cand.size < p["min_wall"]:
        return False, 0.0, ()

    best_s = 0.0
    best_m = ()
    kmax = min(p["max_wall"], cand.size)
    for k in range(p["min_wall"], kmax + 1):
        for combo in itertools.combinations(cand.tolist(), k):
            s = _score_wall_subset(pos[list(combo)], restart, goal, p)
            if s > best_s:
                best_s = s
                best_m = combo
    fired = best_s >= p["score_thresh"]
    return fired, float(best_s), best_m


# ------------------------------------------------------- yansıma yardımcıları


def reflect_x(arr, L: float):
    """X -> L - X yansıması (saha-metresi); (...,2) son ekseni (x,y) kabul eder."""
    a = np.array(arr, float)
    a[..., 0] = L - a[..., 0]
    return a


def reflect_tracks_df(df: pd.DataFrame, L: float, swap_team: bool = True) -> pd.DataFrame:
    """Tracks df'i orta çizgiye göre aynala (pitch_x -> L-pitch_x), opsiyonel takım takası.

    Simetri testleri için: aynalanmış formasyon AYNI kickoff/skor üretmelidir.
    """
    d = df.copy()
    d["pitch_x"] = L - d["pitch_x"].to_numpy()
    if swap_team and "team_id" in d.columns:
        tids = _two_team_ids(d["team_id"].to_numpy())
        if len(tids) == 2:
            a, b = tids
            tcol = d["team_id"].to_numpy().copy()
            swapped = tcol.copy()
            swapped[tcol == a] = b
            swapped[tcol == b] = a
            d["team_id"] = swapped
    return d
