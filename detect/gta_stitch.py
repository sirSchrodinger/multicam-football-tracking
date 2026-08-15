#!/usr/bin/env python3
"""gta_stitch — GTA-Link-vari OFFLINE tracklet SPLIT + CONNECT (saha-uzayinda).

Paper: "GTA: Global Tracklet Association for Multi-Object Tracking in Sports"
(Y. Hsu et al., arXiv:2411.08216). GTA-Link iki offline-asama uygular:
  (1) SPLITTER  — bir tracklet'in icinde >1 kimlik varsa (ID-switch) tracklet'i
      per-frame OZELLIK uzerinde DBSCAN ile kumeleyip BOLER;
  (2) CONNECTOR — ayni kimligin zaman-bosluguyla ayrilmis parcalarini, AYNI
      ANDA gorunmeyenler arasinda, benzerlik esigiyle BIRLESTIRIR.

Orijinal GTA appearance-embedding kullanir. Bizde gece-halisaha goruntusunde
appearance SANS-SEVIYESINDE (olculdu: intra ~0.59 vs inter ~0.58, detect/
track_stitch.py notu). Bu yuzden ozellik olarak GORUNUM YERINE saha-konumu +
HIZ-SUREKSIZLIGI (velocity discontinuity) kullaniyoruz. Kamera SABIT oldugundan
saha-metresi domeninde hiz-kapisi fiziksel anlam tasir (broadcast'te tasimaz).

Bu modul detect/consolidate.py + detect/track_stitch.py fikrini saf, tracker-
bagimsiz, test-edilebilir bir API'ye urunlestirir. Tracklet = (id, frames[],
pitch positions[]); konumlar METRE (saha X=uzunluk, Y=genislik) varsayilir.

YAPISAL DEGISMEZLER (track_stitch ile ayni felsefe):
  * CONNECTOR asla ZAMANDA ORTUSEN (concurrent) iki tracklet'i birlestirmez
    (cannot-link; frame-kumesi kesisimi = sert yasak), transitif olarak da
    (union-find frame-kumesi ayrikligi). over-merge = 0 hedefi yapisaldir.
  * CONNECTOR asla fiziksel-imkansiz hizi asarak baglamaz: gerekli_hiz =
    mesafe / bosluk-suresi <= max_speed_mps olmak ZORUNDA.
  * SPLITTER yalnizca hiz-sureksizligi (teleport) VE DBSCAN'in ayri uzaysal
    kume onaylamasi durumunda boler; tek bir jitter/glitch'i (ayni DBSCAN
    kumesi) yeniden-birlestirir -> yanlis-bolme'ye karsi koruma.

Lisans: numpy + scipy + scikit-learn (BSD/MIT). GPU yok, AGPL yok.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np

try:  # sklearn ortamda mevcut; yine de saf-numpy fallback tutuyoruz
    from sklearn.cluster import DBSCAN as _SK_DBSCAN
except Exception:  # pragma: no cover
    _SK_DBSCAN = None


# ============================================================ veri yapisi ====
@dataclass
class Tracklet:
    """Tek tracklet: id + zaman-sirali frames/konumlar (saha metre).

    frames : (N,) int    — kare indeksleri (artan)
    xy     : (N,2) float  — saha konumu (x=uzunluk, y=genislik), METRE
    t      : (N,) float   — saniye (yoksa frames/fps'ten turetilir)
    team   : opsiyonel takim etiketi (None = bilinmiyor)
    """
    tid: object
    frames: np.ndarray
    xy: np.ndarray
    t: np.ndarray
    team: Optional[int] = None

    def __len__(self) -> int:
        return int(len(self.frames))

    @property
    def t0(self) -> float:
        return float(self.t[0])

    @property
    def t1(self) -> float:
        return float(self.t[-1])

    def frame_set(self) -> frozenset:
        return frozenset(int(f) for f in self.frames)


def make_tracklet(tid, frames, xy, t=None, team=None, fps: float = 25.0) -> Tracklet:
    """Diziler/listelerden zaman-sirali Tracklet kurar (t yoksa frames/fps)."""
    frames = np.asarray(frames, dtype=np.int64).reshape(-1)
    xy = np.asarray(xy, dtype=np.float64).reshape(len(frames), 2)
    if t is None:
        t = frames.astype(np.float64) / float(fps)
    else:
        t = np.asarray(t, dtype=np.float64).reshape(-1)
    order = np.argsort(frames, kind="stable")  # kare-artan garanti
    return Tracklet(tid=tid, frames=frames[order].copy(), xy=xy[order].copy(),
                    t=t[order].copy(), team=team)


def _coerce(tr: Union[Tracklet, dict], fps: float) -> Tracklet:
    """Tracklet ya da dict({tid,frames,xy[,t,team]}) -> normalize Tracklet."""
    if isinstance(tr, Tracklet):
        # zaten sirali kurulmus varsayilir; yine de hizalı kopya don
        return tr
    if isinstance(tr, dict):
        return make_tracklet(tr["tid"], tr["frames"], tr["xy"],
                             tr.get("t"), tr.get("team"), fps=fps)
    raise TypeError(f"Tracklet ya da dict bekleniyor, {type(tr)} geldi")


def tracklets_from_df(df, fps: float = 25.0, x_col: str = "pitch_x",
                      y_col: str = "pitch_y", id_col: str = "tid",
                      t_col: str = "t_sec", team_col: Optional[str] = None
                      ) -> list:
    """Repo tracks-parquet df'inden (tid,frame,t_sec,pitch_x,pitch_y) tracklet
    listesi kurar. consolidate/track_stitch ile ayni giris sozlesmesi; saf,
    homography GEREKTIRMEZ (pitch_x/pitch_y onceden reprojekte edilmis olmali).
    NaN konumlu satirlar atilir.
    """
    out = []
    for tid, g in df.groupby(id_col, sort=True):
        xy = g[[x_col, y_col]].to_numpy(np.float64)
        ok = np.isfinite(xy).all(axis=1)
        if ok.sum() < 1:
            continue
        g = g[ok]
        xy = xy[ok]
        frames = g["frame"].to_numpy(np.int64) if "frame" in g else \
            np.arange(len(g), dtype=np.int64)
        t = g[t_col].to_numpy(np.float64) if t_col in g.columns else None
        team = None
        if team_col is not None and team_col in g.columns:
            vals = g[team_col].dropna().unique()
            team = int(vals[0]) if len(vals) else None
        out.append(make_tracklet(tid, frames, xy, t, team, fps=fps))
    return out


# ============================================================ union-find =====
class _DSU:
    """Cannot-link yeniden-kontrollu union-find (track_stitch._DSU deseni).

    Her kok icin birlesik frame-kumesi tutulur; union(a,b) eger birlesik kumeler
    ZAMANDA kesisiyorsa REDDEDILIR -> transitif over-merge korumasi.
    """

    def __init__(self, frame_sets: Sequence[frozenset]):
        n = len(frame_sets)
        self.parent = list(range(n))
        self.rank = [0] * n
        self.frames = [set(fs) for fs in frame_sets]

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        sa, sb = self.frames[ra], self.frames[rb]
        small, big = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
        if not small.isdisjoint(big):   # ZAMAN-ORTUSME -> birlestirme yasak
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.frames[ra].update(self.frames[rb])
        self.frames[rb] = set()
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


# ============================================================ DBSCAN yard. ===
def _dbscan_labels(P: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Per-nokta DBSCAN etiketleri (sklearn varsa onu, yoksa saf-numpy)."""
    if _SK_DBSCAN is not None:
        return _SK_DBSCAN(eps=eps, min_samples=min_samples).fit_predict(P)
    return _dbscan_numpy(P, eps, min_samples)  # pragma: no cover


def _dbscan_numpy(P: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Minimal DBSCAN (sklearn yoksa). O(n^2); kucuk tracklet'ler icin yeter."""
    n = len(P)
    D = np.linalg.norm(P[:, None, :] - P[None, :, :], axis=2)
    neigh = [np.flatnonzero(D[i] <= eps) for i in range(n)]
    core = np.array([len(neigh[i]) >= min_samples for i in range(n)])
    labels = np.full(n, -1, dtype=int)
    cid = 0
    for i in range(n):
        if labels[i] != -1 or not core[i]:
            continue
        stack = [i]
        labels[i] = cid
        while stack:
            j = stack.pop()
            for k in neigh[j]:
                if labels[k] == -1:
                    labels[k] = cid
                    if core[k]:
                        stack.append(k)
        cid += 1
    return labels


def _fill_noise(labels: np.ndarray) -> np.ndarray:
    """DBSCAN gurultu (-1) noktalarini zamansal komsu etiketine yapistir."""
    labels = labels.copy()
    if np.all(labels == -1):
        return np.zeros(len(labels), dtype=int)
    last = -1
    for i in range(len(labels)):           # ileri-doldur
        if labels[i] != -1:
            last = labels[i]
        elif last != -1:
            labels[i] = last
    nxt = -1
    for i in range(len(labels) - 1, -1, -1):  # bastaki -1'ler icin geri-doldur
        if labels[i] != -1:
            nxt = labels[i]
        elif nxt != -1:
            labels[i] = nxt
    return labels


def _dominant(labels_slice: np.ndarray) -> int:
    """Bir segmentteki baskin (en sik) DBSCAN etiketi."""
    vals, counts = np.unique(labels_slice, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _absorb_short(segs: list, min_len: int) -> list:
    """min_len'den kisa segmentleri komsu segmente yutturur (sliver onleme)."""
    if len(segs) <= 1:
        return segs
    out = [list(segs[0])]
    for a, b in segs[1:]:
        if (b - a) < min_len:
            out[-1][1] = b              # onceki segmente yut
        else:
            out.append([a, b])
    if len(out) >= 2 and (out[0][1] - out[0][0]) < min_len:
        out[1][0] = out[0][0]           # bastaki kisa segmenti sonrakine yut
        out.pop(0)
    return [tuple(x) for x in out]


# ============================================================ SPLITTER =======
def split_tracklets(tracklets: Sequence, max_speed_mps: float = 8.0,
                    eps_m: float = 2.0, min_samples: int = 3,
                    min_seg_frames: int = 2, fps: float = 25.0) -> list:
    """GTA-Link SPLITTER: ID-switch iceren tracklet'leri BOLER.

    Her tracklet icin (saha-uzayinda):
      1) HIZ-SUREKSIZLIGI: ardisik ornekler arasi gap-duyarli hiz
         (mesafe/dt) > max_speed_mps ise -> aday kesik (teleport = imkansiz hiz).
      2) DBSCAN: per-frame konumlari kumeler. Bir aday kesigin iki yanindaki
         segmentler FARKLI DBSCAN kumesindeyse kesik ONAYLANIR (gercek ID-jump);
         AYNI kumedeyse (jitter/glitch) yeniden-birlestirilir (yanlis-bolme yok).
    Doner: bolunmus alt-tracklet listesi (id = "<tid>.<j>"); bolunme yoksa
    orijinal tracklet AYNEN (tek elemanli olarak) doner.
    """
    out = []
    for tr in tracklets:
        out.extend(_split_one(_coerce(tr, fps), max_speed_mps, eps_m,
                              min_samples, min_seg_frames))
    return out


def _child_id(parent, j: int):
    return f"{parent}.{j}"


def _split_one(tr: Tracklet, max_speed_mps: float, eps_m: float,
               min_samples: int, min_seg_frames: int) -> list:
    n = len(tr)
    if n <= 2:
        return [tr]
    P = tr.xy.astype(np.float64)
    t = tr.t.astype(np.float64)

    # 1) hiz-sureksizligi (gap-duyarli): teleport -> aday kesik
    d = np.linalg.norm(np.diff(P, axis=0), axis=1)       # (n-1,)
    dt = np.diff(t)
    safe_dt = np.where(dt > 0, dt, np.inf)               # dt<=0 (dup) -> hiz 0
    v = d / safe_dt
    hard = v > max_speed_mps
    if not hard.any():
        return [tr]                                      # surekli hareket = tek kimlik

    # 2) DBSCAN per-frame uzaysal kumeleme + gurultu doldurma
    labels = _fill_noise(_dbscan_labels(P, eps_m, max(2, min_samples)))

    cut_idx = list(np.flatnonzero(hard) + 1)
    bounds = [0] + cut_idx + [n]
    segs = [(bounds[k], bounds[k + 1]) for k in range(len(bounds) - 1)]

    # her kesigi DBSCAN ile onayla: ayni baskin etiket -> yeniden-birlestir
    final = [segs[0]]
    for k in range(1, len(segs)):
        a0, b0 = final[-1]
        a1, b1 = segs[k]
        if _dominant(labels[a0:b0]) == _dominant(labels[a1:b1]):
            final[-1] = (a0, b1)        # ayni uzaysal kume -> glitch, birlestir
        else:
            final.append((a1, b1))      # farkli kume -> gercek ID-jump, kes

    final = _absorb_short(final, max(1, min_seg_frames))
    if len(final) <= 1:
        return [tr]

    subs = []
    for j, (a, b) in enumerate(final):
        subs.append(Tracklet(tid=_child_id(tr.tid, j),
                            frames=tr.frames[a:b].copy(),
                            xy=P[a:b].copy(), t=t[a:b].copy(), team=tr.team))
    return subs


# ============================================================ CONNECTOR ======
@dataclass
class ConnectResult:
    tracklets: list          # birlestirilmis oyuncu-seviyesi Tracklet'ler
    labels: dict             # orijinal tid -> player_id (int)
    n_input: int
    n_output: int
    edges: list              # kabul edilen (a_tid, b_tid, gap_s, req_speed)


def connect_tracklets(tracklets: Sequence, max_speed_mps: float = 7.0,
                     max_gap_s: float = 5.0, use_team: bool = True,
                     min_dt_s: Optional[float] = None, fps: float = 25.0
                     ) -> ConnectResult:
    """GTA-Link CONNECTOR: ayni kimligin zaman-boslugu icindeki parcalarini birlestirir.

    Yonlu aday kenar A->B (A biter, sonra B baslar) yalnizca:
      * frame-kumeleri AYRIK (zaman-ortusmesi = cannot-link, ASLA birlestirilmez),
      * 0 <= bosluk(gap) = t0_B - t1_A <= max_gap_s,
      * gerekli_hiz = mesafe(son_A, ilk_B) / max(gap, min_dt) <= max_speed_mps,
      * (use_team) iki takim etiketi de biliniyorsa ESLESIYOR.
    Affinity (dusuk maliyet = daha olasi) = (gerekli_hiz, gap). Greedy: en olasi
    kenardan baslayarak her tracklet'e en fazla BIR ardil/oncul; union-find
    frame-ayrikligiyla transitif over-merge engellenir.

    Doner: ConnectResult(birlestirilmis tracklet'ler, tid->player_id, ...).
    """
    trs = [_coerce(tr, fps) for tr in tracklets]
    n = len(trs)
    if min_dt_s is None:
        min_dt_s = 1.0 / float(fps)

    fsets = [tr.frame_set() for tr in trs]
    t0 = np.array([tr.t0 for tr in trs], dtype=np.float64)
    t1 = np.array([tr.t1 for tr in trs], dtype=np.float64)
    p0 = np.array([tr.xy[0] for tr in trs], dtype=np.float64).reshape(n, 2)
    p1 = np.array([tr.xy[-1] for tr in trs], dtype=np.float64).reshape(n, 2)
    teams = [tr.team for tr in trs]

    edges = []
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            if t1[a] > t0[b]:                       # B, A'dan SONRA baslamali
                continue
            if not fsets[a].isdisjoint(fsets[b]):   # ZAMAN-ORTUSME -> yasak
                continue
            gap = float(t0[b] - t1[a])
            if gap < 0.0 or gap > max_gap_s:
                continue
            if use_team and teams[a] is not None and teams[b] is not None \
                    and teams[a] != teams[b]:       # takim uyusmazligi -> yasak
                continue
            dist = float(np.linalg.norm(p1[a] - p0[b]))
            req = dist / max(gap, min_dt_s)
            if req > max_speed_mps:                 # imkansiz hiz -> yasak
                continue
            edges.append((req, gap, a, b, dist))

    edges.sort(key=lambda e: (e[0], e[1]))          # en olasi (en yavas/yakin) once
    dsu = _DSU(fsets)
    succ = [False] * n
    pred = [False] * n
    accepted = []
    for req, gap, a, b, dist in edges:
        if succ[a] or pred[b]:                       # zincir: tek ardil/oncul
            continue
        if not dsu.union(a, b):                      # transitif zaman-ortusme
            continue
        succ[a] = True
        pred[b] = True
        accepted.append((trs[a].tid, trs[b].tid, gap, req))

    # gruplari topla (kok bazli) -> player_id ata, parcalari birlestir
    roots: dict = {}
    for i in range(n):
        roots.setdefault(dsu.find(i), []).append(i)

    labels: dict = {}
    merged = []
    for pid, (_root, members) in enumerate(sorted(roots.items(),
                                                 key=lambda kv: min(t0[i] for i in kv[1]))):
        ms = sorted(members, key=lambda i: t0[i])
        for i in members:
            labels[trs[i].tid] = pid
        F = np.concatenate([trs[i].frames for i in ms])
        X = np.concatenate([trs[i].xy for i in ms])
        T = np.concatenate([trs[i].t for i in ms])
        order = np.argsort(F, kind="stable")
        team = next((trs[i].team for i in ms if trs[i].team is not None), None)
        merged.append(Tracklet(tid=pid, frames=F[order], xy=X[order],
                              t=T[order], team=team))

    return ConnectResult(tracklets=merged, labels=labels,
                         n_input=n, n_output=len(merged), edges=accepted)


# ============================================================ degerlendirme ==
def over_merge_count(labels: dict, tracklets: Sequence, fps: float = 25.0) -> int:
    """Bir etiketlemede over-merge sayisi: ayni player_id'ye atanmis AMA zamanda
    ORTUSEN tracklet ciftleri (yapisal olarak 0 olmali). Bagimsiz dogrulama.
    """
    trs = [_coerce(tr, fps) for tr in tracklets]
    by_pid: dict = {}
    for tr in trs:
        by_pid.setdefault(labels[tr.tid], []).append(tr.frame_set())
    bad = 0
    for fs_list in by_pid.values():
        for i in range(len(fs_list)):
            for j in range(i + 1, len(fs_list)):
                if not fs_list[i].isdisjoint(fs_list[j]):
                    bad += 1
    return bad
