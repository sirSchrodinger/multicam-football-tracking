"""Round-2 prep: saf-olmayan gruplari tracklet'e geri boz, 0.08'de yeniden grupla.

Girdi: gtwin_adj_result.json (round-1 purity) + gtwin_groups.json + tracklets
Cikti: gtwin_units.json  (unit -> tracklet listesi; P* = round-1 pure grup,
       S* = impure gruptan 0.08-altgrup) + gtwin_spurious_units.json
       + gtwin_adj_args2.json (workflow args: units+pairs)
Kartlar: eval/gtwin_cards.py gtwin_units.json
"""
from __future__ import annotations
import json
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, '.')
from detect.global_stitch import stitch, _Cluster, _medoid  # noqa: E402

STRICT_GATE = 0.08
PAIR_MAX_COS = 0.32
PAIR_MAXGAP = 60.0
PAIR_VMAX = 7.0
OVERLAP_TOL = 0.15
KNN_APP = 4
KNN_MOT = 2
MIN_TRACKLET_DET = 4


def main():
    res = json.load(open('scratchpad/gtwin_adj_result.json'))
    groups = json.load(open('scratchpad/gtwin_groups.json'))
    df = pd.read_parquet('scratchpad/gtwin_tracklets.parquet').reset_index(drop=True)
    emb = np.load('scratchpad/gtwin_emb.npy')
    sz = df.groupby('tracklet_id').tracklet_id.transform('size')
    df = df[sz >= MIN_TRACKLET_DET].reset_index(drop=True)
    emb = emb[sz.values >= MIN_TRACKLET_DET]

    pur = res['purity']
    units: dict[str, list[int]] = {}
    spurious_units = []
    for g, tids in groups.items():
        p = pur.get(g, {'pure': False, 'is_player': True})
        if not p.get('is_player', True):
            if p.get('pure', False):
                spurious_units.append({'unit': f'X{g}', 'tids': tids,
                                       'note': p.get('note', '')})
                continue
            # impure + non-player: tracklet'lere boz, round-2'ye gonder
        if p.get('pure', False) and p.get('is_player', True):
            units[f'P{g}'] = tids
        else:
            # impure -> 0.08 strict altgruplar
            sub = df[df.tracklet_id.isin(tids)].reset_index(drop=True)
            sube = emb[df.tracklet_id.isin(tids).values]
            if not len(sub):
                continue
            ids = stitch(sub, sube, app_gate=STRICT_GATE, maxgap=15.0, vmax=6.5,
                         overlap_tol=OVERLAP_TOL)
            for k, u in enumerate(sorted(set(ids))):
                utids = sorted(int(t) for t in
                               sub[ids == u].tracklet_id.unique())
                units[f'S{g}_{k}'] = utids
    json.dump(units, open('scratchpad/gtwin_units.json', 'w'), indent=1)
    json.dump(spurious_units, open('scratchpad/gtwin_spurious_units.json', 'w'),
              indent=1)
    print(f"{len(units)} unit ({sum(1 for u in units if u.startswith('P'))} pure-grup, "
          f"{sum(1 for u in units if u.startswith('S'))} altgrup), "
          f"{len(spurious_units)} spurious")

    # unit ozellikleri + pairs (adjprep mantigi)
    feats = {}
    for uname, tids in units.items():
        m = df.tracklet_id.isin(tids).values
        idx = np.where(m)[0]
        ts = df.t.values[idx]
        o = np.argsort(ts); idx = idx[o]; ts = ts[o]
        c = _Cluster(uname, float(ts[0]), float(ts[-1]), _medoid(emb[idx]),
                     df[['px', 'py']].values[idx][0],
                     df[['px', 'py']].values[idx][-1])
        c.intervals = []; c.ends = []
        for tid in tids:
            mm = np.where(df.tracklet_id.values == tid)[0]
            tt = df.t.values[mm]; oo = np.argsort(tt); mm = mm[oo]; tt = tt[oo]
            pp = df[['px', 'py']].values[mm]
            c.intervals.append((float(tt[0]), float(tt[-1])))
            c.ends.append((float(tt[0]), pp[0], float(tt[-1]), pp[-1]))
        feats[uname] = c

    uniq = sorted(units.keys())
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
            app_c = dc <= PAIR_MAX_COS
            mot_c = (gap <= 4.0) and (v <= 5.0)
            if not (app_c or mot_c):
                continue
            cand[(ga, gb)] = {'cos': dc, 'gap': gap, 'v': v,
                              'app': app_c, 'mot': mot_c}
    keep = set()
    for g in uniq:
        mine = [(k, c) for k, c in cand.items() if g in k]
        app = sorted([x for x in mine if x[1]['app']], key=lambda x: x[1]['cos'])
        mot = sorted([x for x in mine if x[1]['mot']], key=lambda x: x[1]['gap'])
        for k, _ in app[:KNN_APP]:
            keep.add(k)
        for k, _ in mot[:KNN_MOT]:
            keep.add(k)
    span = {u: (feats[u].intervals and
                (min(a for a, _ in feats[u].intervals),
                 max(b for _, b in feats[u].intervals))) for u in uniq}
    # skip_purity: P* round-1'de kanitli; tek-tracklet S* konservatif-uretimle ~saf.
    # Istisna: non-player-impure gruptan (G20) gelen S* birimleri purity'ye girer.
    nonplayer_impure = {g for g, p in pur.items()
                        if not p.get('is_player', True) and not p.get('pure', False)}
    def _skip(u):
        if u.startswith('P'):
            return True
        src = u[1:].split('_')[0]
        if src in nonplayer_impure:
            return False
        return len(units[u]) == 1
    args = {'groups': {u: {'card': f'scratchpad/gtwin_cards2/{u}.jpg',
                           't0': round(span[u][0], 1), 't1': round(span[u][1], 1),
                           'skip_purity': bool(_skip(u))}
                       for u in uniq},
            'pairs': sorted([{'a': a, 'b': b, 'cos': round(c['cos'], 3),
                              'gap_s': round(c['gap'], 1),
                              'v_mps': round(c['v'], 2),
                              'why': ('app+mot' if c['app'] and c['mot']
                                      else ('app' if c['app'] else 'mot'))}
                             for (a, b), c in cand.items() if (a, b) in keep],
                            key=lambda p: p['cos'])}
    json.dump(args, open('scratchpad/gtwin_adj_args2.json', 'w'),
              separators=(',', ':'))
    print(f"{len(args['pairs'])} aday cift -> gtwin_adj_args2.json")
    # intervals'i lokal birlestirme icin ayri kaydet
    json.dump({u: [[round(a, 2), round(b, 2)] for a, b in feats[u].intervals]
               for u in uniq}, open('scratchpad/gtwin_unit_intervals.json', 'w'))


if __name__ == '__main__':
    main()
