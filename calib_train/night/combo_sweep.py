#!/usr/bin/env python3
"""Bir idx için TÜM top-K combo'ları çöz, her birinin gf_at_far (far-goal grounding) + sanity'sini
göster. Census'un seçtiği combo vs gf-max combo. Amaç: census yamuk seçtiyse, daha iyi (far-goal
gerçekten oturan) bir combo VAR MI? Varsa re-selection kurtarır."""
import os, sys, json, numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))
from calib_train import auto_calib as AC
from calib_train.multihyp_calib import groups_topk, ROLES, _sample_prob
from calib_train.auto_clean2d import sanity
from itertools import product
L,Wp=AC.L,18.0
PC=os.path.join(HERE,"probcache_v3")
CEN=json.load(open(os.path.join(ROOT,"cand","_census_hr3.json")))
MAN={r['idx']:r for r in json.load(open(os.path.join(PC,"manifest.json")))}

def sweep(idx):
    d=np.load(os.path.join(PC,f"{idx:03d}.npz")); prob=d['prob'].astype(np.float32); w,h=int(d['w']),int(d['h'])
    cands,_=groups_topk(prob,w,h,K=3)
    if cands is None: print(f"idx{idx}: aday yok"); return
    ce=CEN.get(str(idx),{}); picked=tuple(ce.get('combo',[]) or [])
    rows=[]
    for combo in product(*[range(len(cands[r])) for r in ROLES]):
        g={r:cands[r][ci] for r,ci in zip(ROLES,combo)}
        rec=AC.solve_calib(g,w/2,h/2,w/2)
        if rec is None: continue
        ok,why=sanity(rec,w,h,img=None)  # img yok -> karanlık-red atlanır, geometri-red kalır
        gf,_=_sample_prob(prob[1],AC.project_metric(np.linspace([L,0],[L,Wp],60),rec['k1'],rec['k2'],rec['H'],w/2,h/2,w/2),w,h)
        gn,_=_sample_prob(prob[0],AC.project_metric(np.linspace([0,0],[0,Wp],60),rec['k1'],rec['k2'],rec['H'],w/2,h/2,w/2),w,h)
        rows.append((combo,rec['fit'],ok,gf or 0.0,gn or 0.0,why))
    rows.sort(key=lambda r:-r[3])
    print(f"\n=== idx{idx} {MAN[idx]['file'][:28]} census-combo={list(picked)} ===")
    print(f"{'combo':>12} {'fit':>5} {'san':>4} {'gf@far':>6} {'gn@near':>7}  reason")
    for combo,fit,ok,gf,gn,why in rows[:8]:
        mk="<PICKED" if combo==picked else ("<GF-MAX" if combo==rows[0][0] else "")
        print(f"{str(list(combo)):>12} {fit:>5.2f} {'ok' if ok else 'X':>4} {gf:>6.2f} {gn:>7.2f}  {why[:30]:30} {mk}")

if __name__=="__main__":
    for idx in ([int(x) for x in sys.argv[1:]] if len(sys.argv)>1 else [13,35,59,61]):
        sweep(idx)
