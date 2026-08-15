"""Adjudication GIRDI hazirlik: on-gruplama (guvenli-esik stitch) + aday ciftler.

1. stitch(app_gate=0.10) -> AUTO-GRUPLAR (farkli-oyuncu p10=0.187, 0.10 guvenli taraf)
2. Grup ciftleri arasi medoid cos<0.32 + zaman-ortusme-yok + hiz-uyumlu -> ADAY CIFT
3. Gruplarin kartlarini uret (gtwin_cards ile grup modu)
Cikti: scratchpad/gtwin_adj_input.json  (workflow args'i icin)
"""
from __future__ import annotations
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from detect.global_stitch import stitch, _Cluster, _medoid  # noqa: E402

DET = 'scratchpad/gtwin_tracklets.parquet'
EMB = 'scratchpad/gtwin_emb.npy'
OUT = 'scratchpad/gtwin_adj_input.json'
GROUPS_JSON = 'scratchpad/gtwin_groups.json'

AUTO_GATE = 0.16      # OSNet ayni-oyuncu ort 0.065 / farkli p10 0.187 — guvenli alt-bant
PAIR_MAX_COS = 0.32
PAIR_MAXGAP = 60.0
PAIR_VMAX = 7.0
OVERLAP_TOL = 0.15
MIN_TRACKLET_DET = 4  # kucuk gurultu-parcalari GT evreninden cikar (det kapsami ~%99 kalir)
KNN_APP = 4           # grup basina en-yakin gorunum komsusu
KNN_MOT = 2           # grup basina en-iyi hareket-devamlilik komsusu


def main():
    df = pd.read_parquet(DET).reset_index(drop=True)
    emb = np.load(EMB)
    sz = df.groupby('tracklet_id').tracklet_id.transform('size')
    df = df[sz >= MIN_TRACKLET_DET].reset_index(drop=True)
    emb = emb[sz.values >= MIN_TRACKLET_DET]
    ids = stitch(df, emb, app_gate=AUTO_GATE, maxgap=20.0, vmax=6.5,
                 overlap_tol=OVERLAP_TOL)
    df['group_id'] = ids
    uniq = sorted(df.group_id.unique())
    gname = {g: f"G{k}" for k, g in enumerate(uniq)}
    groups = {gname[g]: sorted(int(t) for t in
                               df[df.group_id == g].tracklet_id.unique())
              for g in uniq}
    json.dump(groups, open(GROUPS_JSON, 'w'), indent=1)

    # grup ozellikleri
    feats = {}
    for g in uniq:
        m = np.where(df.group_id.values == g)[0]
        ts = df.t.values[m]
        o = np.argsort(ts); m = m[o]; ts = ts[o]
        pos = df[['px', 'py']].values[m]
        c = _Cluster(g, float(ts[0]), float(ts[-1]), _medoid(emb[m]),
                     pos[0], pos[-1])
        # araliklari tracklet bazinda geri kur (dogru overlap icin)
        c.intervals = []
        c.ends = []
        for tid in df[df.group_id == g].tracklet_id.unique():
            mm = np.where(df.tracklet_id.values == tid)[0]
            tt = df.t.values[mm]; oo = np.argsort(tt); mm = mm[oo]; tt = tt[oo]
            pp = df[['px', 'py']].values[mm]
            c.intervals.append((float(tt[0]), float(tt[-1])))
            c.ends.append((float(tt[0]), pp[0], float(tt[-1]), pp[-1]))
        feats[g] = c

    # tum gecerli ciftleri hesapla, sonra grup-basi k-NN buda (ajan-butcesi)
    cand = {}
    for i, ga in enumerate(uniq):
        for gb in uniq[i + 1:]:
            a, b = feats[ga], feats[gb]
            if a.overlaps(b, tol=OVERLAP_TOL):
                continue
            dc = 1.0 - float(np.dot(
                a.emb / (np.linalg.norm(a.emb) + 1e-9),
                b.emb / (np.linalg.norm(b.emb) + 1e-9)))
            gap, v = a.min_gap_and_speed(b)
            if gap > PAIR_MAXGAP or v > PAIR_VMAX:
                continue
            app_cand = dc <= PAIR_MAX_COS
            mot_cand = (gap <= 4.0) and (v <= 5.0)
            if not (app_cand or mot_cand):
                continue
            cand[(ga, gb)] = {'cos': dc, 'gap': gap, 'v': v,
                              'app': app_cand, 'mot': mot_cand}
    # k-NN secim: her grubun en-yakin KNN_APP gorunum + KNN_MOT hareket komsusu.
    # (gorunum degisen ayni-oyuncu parcalari icin hareket-yolu acik kalir;
    # transitiflik union-find'da kalan linkleri tamamlar.)
    keep_pairs = set()
    for g in uniq:
        mine = [(k, c) for k, c in cand.items() if g in k]
        app = sorted([x for x in mine if x[1]['app']], key=lambda x: x[1]['cos'])
        mot = sorted([x for x in mine if x[1]['mot']], key=lambda x: x[1]['gap'])
        for k, _ in app[:KNN_APP]:
            keep_pairs.add(k)
        for k, _ in mot[:KNN_MOT]:
            keep_pairs.add(k)
    pairs = [{'a': gname[ga], 'b': gname[gb],
              'cos': round(c['cos'], 3), 'gap_s': round(c['gap'], 1),
              'v_mps': round(c['v'], 2),
              'why': ('app+mot' if c['app'] and c['mot']
                      else ('app' if c['app'] else 'mot'))}
             for (ga, gb), c in cand.items() if (ga, gb) in keep_pairs]
    pairs.sort(key=lambda p: p['cos'])

    span = df.groupby('group_id').t.agg(['min', 'max'])
    ginfo = {gname[g]: {
        'n_tracklets': int(df[df.group_id == g].tracklet_id.nunique()),
        'n_det': int((df.group_id == g).sum()),
        't0': round(float(span.loc[g, 'min']), 1),
        't1': round(float(span.loc[g, 'max']), 1),
        'card': f"scratchpad/gtwin_cards/{gname[g]}.jpg",
        'intervals': [[round(a, 2), round(b, 2)] for a, b in feats[g].intervals],
    } for g in uniq}

    out = {'groups': ginfo, 'pairs': pairs,
           'params': {'AUTO_GATE': AUTO_GATE, 'PAIR_MAX_COS': PAIR_MAX_COS,
                      'PAIR_MAXGAP': PAIR_MAXGAP, 'PAIR_VMAX': PAIR_VMAX}}
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"{len(uniq)} auto-grup, {len(pairs)} aday cift -> {OUT}")
    print(f"grup kartlari icin: venv/bin/python eval/gtwin_cards.py {GROUPS_JSON}")


if __name__ == '__main__':
    main()
