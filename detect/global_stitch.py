"""OFFLINE GLOBAL tracklet-stitch — OSNet medoid + sert kisitlar (AUDIT_ROADMAP fix#1+4).

Tracklet'leri (konservatif, yuksek-saflik) kimliklere birlestirir:
- ZAMAN-ORTUSME = HARD cannot-link (ayni anda iki yerde olunamaz; tol ile)
- MAKS-HIZ koprusu: gap boyunca gereken pitch-hizi VMAX'i asarsa birlesme YASAK
- GORUNUM kapisi: tracklet-medoid OSNet cos-dist < app_gate
- MAXGAP: bundan uzun boslukta birlesme yok (surukleme riski)
- Opsiyonel roster hedefi YOK — zorlama yerine keep-best (GT ile olculur)

Agglomerative: her adimda en dusuk gorunum-mesafeli GECERLI cift birlesir.
Cluster zaman-araliklari kume olarak tutulur; cannot-link dogal yayilir.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _medoid(embs: np.ndarray) -> np.ndarray:
    if len(embs) == 1:
        return embs[0]
    sim = embs @ embs.T
    return embs[int(np.argmax(sim.sum(1)))]


class _Cluster:
    __slots__ = ('tids', 'intervals', 'emb', 'ends')

    def __init__(self, tid, t0, t1, emb, p0, p1):
        self.tids = [tid]
        self.intervals = [(t0, t1)]
        self.emb = emb
        # ends: (t_start, pos_start, t_end, pos_end) — hiz-kapisi icin uclar
        self.ends = [(t0, p0, t1, p1)]

    def overlaps(self, other, tol=0.15) -> bool:
        for a0, a1 in self.intervals:
            for b0, b1 in other.intervals:
                if min(a1, b1) - max(a0, b0) > tol:
                    return True
        return False

    def min_gap_and_speed(self, other):
        """En yakin uc cifti arasindaki (gap_s, gereken_hiz_mps)."""
        best = (np.inf, np.inf)
        for (a0, ap0, a1, ap1) in self.ends:
            for (b0, bp0, b1, bp1) in other.ends:
                if b0 >= a1:      # other sonra
                    gap = b0 - a1
                    dist = float(np.hypot(*(bp0 - ap1)))
                elif a0 >= b1:    # self sonra
                    gap = a0 - b1
                    dist = float(np.hypot(*(ap0 - bp1)))
                else:
                    continue
                v = dist / max(gap, 0.2)
                if gap < best[0]:
                    best = (gap, v)
        return best


def stitch(df: pd.DataFrame, emb: np.ndarray,
           app_gate: float = 0.16, maxgap: float = 10.0,
           vmax: float = 6.5, overlap_tol: float = 0.15,
           verbose: bool = False) -> np.ndarray:
    """df: t, px, py, tracklet_id kolonlari. Doner: satir-bazi identity_id."""
    df = df.reset_index(drop=True)
    tids = df.tracklet_id.values
    clusters: dict[int, _Cluster] = {}
    for tid in np.unique(tids):
        m = np.where(tids == tid)[0]
        ts = df.t.values[m]
        o = np.argsort(ts)
        m = m[o]; ts = ts[o]
        pos = df[['px', 'py']].values[m]
        clusters[tid] = _Cluster(tid, float(ts[0]), float(ts[-1]),
                                 _medoid(emb[m]), pos[0], pos[-1])

    keys = sorted(clusters.keys())
    # aday ciftler: gorunum-mesafesine gore sirali, tekrar-degerlendirmeli greedy
    merged_into = {}

    def find(k):
        while k in merged_into:
            k = merged_into[k]
        return k

    while True:
        keys_now = sorted({find(k) for k in keys})
        best = None
        for i in range(len(keys_now)):
            a = clusters[keys_now[i]]
            for j in range(i + 1, len(keys_now)):
                b = clusters[keys_now[j]]
                dc = 1.0 - float(np.dot(
                    a.emb / (np.linalg.norm(a.emb) + 1e-9),
                    b.emb / (np.linalg.norm(b.emb) + 1e-9)))
                if dc > app_gate:
                    continue
                if a.overlaps(b, tol=overlap_tol):
                    continue
                gap, v = a.min_gap_and_speed(b)
                if gap > maxgap or v > vmax:
                    continue
                if best is None or dc < best[0]:
                    best = (dc, keys_now[i], keys_now[j])
        if best is None:
            break
        _, ka, kb = best
        a, b = clusters[ka], clusters[kb]
        a.tids += b.tids
        a.intervals += b.intervals
        a.ends += b.ends
        # medoid guncelle: iki medoidin ortalamasi (ucuz yaklastirma)
        a.emb = (a.emb + b.emb)
        a.emb = a.emb / (np.linalg.norm(a.emb) + 1e-9)
        merged_into[kb] = ka
        del clusters[kb]
        if verbose:
            print(f"merge {kb}->{ka} (cos={best[0]:.3f})")

    tid2id = {}
    for k, c in clusters.items():
        for tid in c.tids:
            tid2id[tid] = k
    return np.array([tid2id[t] for t in tids])


def stitch_fast(df: pd.DataFrame, emb: np.ndarray,
                app_gate: float = 0.16, maxgap: float = 10.0,
                vmax: float = 6.5, overlap_tol: float = 0.15,
                verbose: bool = False) -> np.ndarray:
    """stitch() ile ayni mantik, buyuk-N (tam-mac, binlerce tracklet) icin:
    medoid-cos matrisi + oncelik-kuyrugu + merge'te sadece etkilenen satir guncelle.
    """
    import heapq
    df = df.reset_index(drop=True)
    tids = df.tracklet_id.values
    uniq = np.unique(tids)
    n = len(uniq)
    tpos = {}
    meds = np.zeros((n, 512), np.float32)
    clusters = {}
    for k, tid in enumerate(uniq):
        m = np.where(tids == tid)[0]
        ts = df.t.values[m]
        o = np.argsort(ts); m = m[o]; ts = ts[o]
        pos = df[['px', 'py']].values[m]
        meds[k] = _medoid(emb[m])
        clusters[k] = {'tids': [tid],
                       'intervals': [(float(ts[0]), float(ts[-1]))],
                       'ends': [(float(ts[0]), pos[0], float(ts[-1]), pos[-1])],
                       'emb': meds[k].copy(), 'ver': 0}
        tpos[tid] = k

    def pair_ok(a, b):
        """(gecerli_mi, cos). cluster dict'leriyle."""
        ea = a['emb'] / (np.linalg.norm(a['emb']) + 1e-9)
        eb = b['emb'] / (np.linalg.norm(b['emb']) + 1e-9)
        dc = 1.0 - float(np.dot(ea, eb))
        if dc > app_gate:
            return False, dc
        for a0, a1 in a['intervals']:
            for b0, b1 in b['intervals']:
                if min(a1, b1) - max(a0, b0) > overlap_tol:
                    return False, dc
        best_gap, best_v = np.inf, np.inf
        for (a0, ap0, a1, ap1) in a['ends']:
            for (b0, bp0, b1, bp1) in b['ends']:
                if b0 >= a1:
                    gap = b0 - a1; dist = float(np.hypot(*(bp0 - ap1)))
                elif a0 >= b1:
                    gap = a0 - b1; dist = float(np.hypot(*(ap0 - bp1)))
                else:
                    continue
                if gap < best_gap:
                    best_gap, best_v = gap, dist / max(gap, 0.2)
        if best_gap > maxgap or best_v > vmax:
            return False, dc
        return True, dc

    # ilk aday kumesi: cos-matristen app_gateical ciftler (vektorize on-filtre)
    M = meds / (np.linalg.norm(meds, axis=1, keepdims=True) + 1e-9)
    D = 1.0 - (M @ M.T)
    cand_i, cand_j = np.where((D <= app_gate) & (np.triu(np.ones_like(D), 1) > 0))
    heap = []
    for i, j in zip(cand_i, cand_j):
        ok, dc = pair_ok(clusters[i], clusters[j])
        if ok:
            heapq.heappush(heap, (dc, int(i), int(j), 0, 0))
    live = set(clusters.keys())
    merges = 0
    while heap:
        dc, i, j, vi, vj = heapq.heappop(heap)
        if i not in live or j not in live:
            continue
        if clusters[i]['ver'] != vi or clusters[j]['ver'] != vj:
            continue  # bayat kayit; guncel hali merge sirasinda yeniden eklendi
        ok, _ = pair_ok(clusters[i], clusters[j])
        if not ok:
            continue
        a, b = clusters[i], clusters[j]
        a['tids'] += b['tids']
        a['intervals'] += b['intervals']
        a['ends'] += b['ends']
        a['emb'] = a['emb'] + b['emb']
        a['emb'] = a['emb'] / (np.linalg.norm(a['emb']) + 1e-9)
        a['ver'] += 1
        live.discard(j)
        del clusters[j]
        merges += 1
        # yeni birlesik kume vs tum canli kumeler
        for k in live:
            if k == i:
                continue
            ok2, dc2 = pair_ok(a, clusters[k])
            if ok2:
                x, y = (i, k) if i < k else (k, i)
                heapq.heappush(heap, (dc2, x, y,
                                      clusters[x]['ver'], clusters[y]['ver']))
    if verbose:
        print(f"stitch_fast: {n} tracklet -> {len(live)} kimlik ({merges} merge)")
    tid2id = {}
    for k in live:
        for tid in clusters[k]['tids']:
            tid2id[tid] = k
    return np.array([tid2id[t] for t in tids])


if __name__ == '__main__':
    # sentetik self-test: 3 oyuncu, her biri 3 parcaya bolunmus, farkli embedding yonleri
    rng = np.random.default_rng(0)
    rows, embs = [], []
    base = rng.normal(size=(3, 512)); base /= np.linalg.norm(base, axis=1, keepdims=True)
    tid = 0
    for pid in range(3):
        for seg in range(3):
            t0 = seg * 12.0 + pid * 0.0
            for k in range(20):
                t = t0 + k * 0.5
                rows.append({'t': t, 'px': 10.0 * pid + 0.1 * k, 'py': 5.0,
                             'tracklet_id': tid})
                # dikkat: 512-dim'de scale=0.05 gurultu normu ~1.13 = sinyal kadar!
                # gercek OSNet ayni-oyuncu cos-dist ~0.065 -> scale=0.01 (~0.05 dist)
                e = base[pid] + rng.normal(scale=0.01, size=512)
                embs.append(e / np.linalg.norm(e))
            tid += 1
    df = pd.DataFrame(rows); emb = np.stack(embs)
    ids = stitch(df, emb, app_gate=0.3, maxgap=5.0, vmax=8.0)
    # ayni oyuncunun 3 parcasi ayni id olmali; farkli oyuncular farkli
    got = [len(np.unique(ids[df.tracklet_id.isin([p * 3, p * 3 + 1, p * 3 + 2])]))
           for p in range(3)]
    n_ids = len(np.unique(ids))
    assert n_ids == 3 and got == [1, 1, 1], (n_ids, got)
    # zaman-ortusme cannot-link: ayni anda yasayan iki parca birlesmemeli
    df2 = df.copy()
    over = df2.tracklet_id.isin([0, 1])
    df2.loc[df2.tracklet_id == 1, 't'] = df2.loc[df2.tracklet_id == 0, 't'].values
    ids2 = stitch(df2, emb, app_gate=0.5, maxgap=5.0, vmax=8.0)
    assert (ids2[df2.tracklet_id == 0][0] != ids2[df2.tracklet_id == 1][0]), 'overlap merge!'
    # stitch_fast esdegerlik (ayni partition)
    idsf = stitch_fast(df, emb, app_gate=0.3, maxgap=5.0, vmax=8.0)
    import itertools
    def part(ids_):
        m = {}
        for tid, i in zip(df.tracklet_id.values, ids_):
            m.setdefault(i, set()).add(tid)
        return sorted(frozenset(v) for v in m.values())
    assert part(ids) == part(idsf), 'fast != slow partition'
    print('selftest OK: 3 oyuncu 9 parca -> 3 kimlik; overlap cannot-link; fast==slow')
