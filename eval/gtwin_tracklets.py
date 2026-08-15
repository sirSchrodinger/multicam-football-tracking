"""GT-penceresi KONSERVATIF tracklet uretici.

Amac: kimlik-GT insasi icin YUKSEK-SAFLIK tracklet'ler (over-segment OK, impurity YASAK).
- Ardisik kare Hungarian eslemesi, SIKI kapilar (goruntu-mesafe + OSNet cos + olcek)
- Gap koprusu YOK (kayip -> yeni tracklet)
- Gorunum sicramasi -> BOL (running-medoid cos > SPLIT_COS)

Girdi:  scratchpad/gtwin_det.parquet + gtwin_emb.npy  (export_gtwin.py ciktisi)
Cikti:  scratchpad/gtwin_tracklets.parquet (ayni satirlar + tracklet_id kolonu)
        scratchpad/gtwin_tracklet_summary.json
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

DET = 'scratchpad/gtwin_det.parquet'
EMB = 'scratchpad/gtwin_emb.npy'
OUT = 'scratchpad/gtwin_tracklets.parquet'
SUM = 'scratchpad/gtwin_tracklet_summary.json'

# Kapilar (12.5fps, dt=0.08s):
MAX_IMG_DIST = 80.0    # px / kare — hizli oyuncu near'da ~40px/kare, pay birakildi
MAX_EMB_COS = 0.30     # ayni-oyuncu ort 0.065 / farkli 0.311 -> arada sinir
MAX_H_RATIO = 1.6      # kutu-boyu orani (olcek sicramasi = farkli kisi/FP)
SPLIT_COS = 0.45       # running-medoid'e cos-dist bunu asarsa tracklet BOLUNUR
MAX_DT = 0.13          # sadece ardisik ornek-kare (0.08s adim; timing toleransi)


def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - float(np.dot(a, b)))


def build_tracklets(df: pd.DataFrame, emb: np.ndarray,
                    max_img_dist: float = MAX_IMG_DIST,
                    max_emb_cos: float = MAX_EMB_COS,
                    max_h_ratio: float = MAX_H_RATIO,
                    split_cos: float = SPLIT_COS,
                    max_dt: float = MAX_DT):
    df = df.reset_index(drop=True)
    tid = np.full(len(df), -1, dtype=int)
    next_id = 0
    # aktif tracklet: son satir-idx, son t, running-mean emb (normalize)
    active: dict[int, dict] = {}

    frames = sorted(df.t.unique())
    idx_by_t = {t: np.where(df.t.values == t)[0] for t in frames}

    for fi, t in enumerate(frames):
        idxs = idx_by_t[t]
        # suresi gecen aktifleri kapat
        stale = [k for k, v in active.items() if t - v['t'] > max_dt + 1e-9]
        for k in stale:
            del active[k]

        if not active:
            for i in idxs:
                active[next_id] = {'t': t, 'i': i, 'e': emb[i].copy(), 'n': 1}
                tid[i] = next_id
                next_id += 1
            continue

        akeys = list(active.keys())
        cost = np.full((len(akeys), len(idxs)), 1e6)
        for ai, ak in enumerate(akeys):
            av = active[ak]
            pi = av['i']
            pf = np.array([(df.x0.values[pi] + df.x1.values[pi]) / 2, df.y1.values[pi]])
            ph = df.y1.values[pi] - df.y0.values[pi]
            for di, i in enumerate(idxs):
                cf = np.array([(df.x0.values[i] + df.x1.values[i]) / 2, df.y1.values[i]])
                d_img = float(np.hypot(*(cf - pf)))
                if d_img > max_img_dist:
                    continue
                ch = df.y1.values[i] - df.y0.values[i]
                hr = max(ch, ph) / max(min(ch, ph), 1e-6)
                if hr > max_h_ratio:
                    continue
                ae = av['e'] / (np.linalg.norm(av['e']) + 1e-9)
                dc = cos_dist(ae, emb[i])
                if dc > max_emb_cos:
                    continue
                cost[ai, di] = d_img / max_img_dist + dc / max_emb_cos
        ri, ci = linear_sum_assignment(cost)
        matched_d = set()
        for a, d in zip(ri, ci):
            if cost[a, d] >= 1e6:
                continue
            ak = akeys[a]
            i = idxs[d]
            av = active[ak]
            ae = av['e'] / (np.linalg.norm(av['e']) + 1e-9)
            if cos_dist(ae, emb[i]) > split_cos:
                continue  # gorunum sicramasi: eslestirme yerine yeni tracklet
            tid[i] = ak
            av['t'] = t
            av['i'] = i
            av['e'] = av['e'] + emb[i]
            av['n'] += 1
            matched_d.add(d)
        for di, i in enumerate(idxs):
            if di not in matched_d:
                active[next_id] = {'t': t, 'i': i, 'e': emb[i].copy(), 'n': 1}
                tid[i] = next_id
                next_id += 1
    return tid


def main():
    df = pd.read_parquet(DET)
    emb = np.load(EMB)
    assert len(df) == len(emb), f"det {len(df)} != emb {len(emb)}"
    tid = build_tracklets(df, emb)
    df = df.reset_index(drop=True)
    df['tracklet_id'] = tid
    df.to_parquet(OUT)

    g = df.groupby('tracklet_id')
    spans = (g.t.max() - g.t.min())
    summ = {
        'n_det': int(len(df)),
        'n_tracklets': int(df.tracklet_id.nunique()),
        'span_median_s': float(spans.median()),
        'span_p90_s': float(spans.quantile(0.9)),
        'n_span_ge_5s': int((spans >= 5).sum()),
        'n_span_ge_20s': int((spans >= 20).sum()),
        'n_singleton': int((g.size() == 1).sum()),
        'params': {'MAX_IMG_DIST': MAX_IMG_DIST, 'MAX_EMB_COS': MAX_EMB_COS,
                   'SPLIT_COS': SPLIT_COS, 'MAX_H_RATIO': MAX_H_RATIO},
    }
    json.dump(summ, open(SUM, 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
