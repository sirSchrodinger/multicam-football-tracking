#!/usr/bin/env python3
"""Hafif self-audit metriği (cache'ten, GPU'suz, ~30s): kaç idx çözülüyor + medyan fit +
far-goal-grounding dağılımı. Regresyon-guard için _audit_baseline.json ile karşılaştırır.
SELF_AUDIT.log'a tek-satır özet basar (cron çağırır)."""
import os, sys, json, numpy as np, time
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))
from calib_train import auto_calib as AC
from calib_train.multihyp_calib import groups_topk, ROLES, _sample_prob
L,Wp=AC.L,18.0
PC=os.path.join(HERE,"probcache_v3")
CEN=json.load(open(os.path.join(ROOT,"cand","_census_hr3.json")))

def solve(idx):
    ce=CEN.get(str(idx))
    if not ce or ce.get('src') is None: return None
    d=np.load(os.path.join(PC,f"{idx:03d}.npz")); prob=d['prob'].astype(np.float32); w,h=int(d['w']),int(d['h'])
    combo=ce.get('combo',[0,0,0,0]) or [0,0,0,0]
    cands,_=groups_topk(prob,w,h,K=3)
    if cands is None: return None
    try: g={r:cands[r][ci] for r,ci in zip(ROLES,combo)}
    except (IndexError,KeyError): g={r:cands[r][0] for r in ROLES}
    rec=AC.solve_calib(g,w/2,h/2,w/2)
    if rec is None: return None
    gf,_=_sample_prob(prob[1],AC.project_metric(np.linspace([L,0],[L,Wp],60),rec['k1'],rec['k2'],rec['H'],w/2,h/2,w/2),w,h)
    return dict(idx=idx,fit=float(rec['fit']),gf=float(gf or 0.0))

def main():
    t0=time.time(); rows=[r for r in (solve(i) for i in range(121)) if r]
    fits=[r['fit'] for r in rows]; gfs=[r['gf'] for r in rows]
    m=dict(n_solved=len(rows), med_fit=round(float(np.median(fits)),3),
           gf_ge05=sum(1 for g in gfs if g>=0.5), gf_lt02=sum(1 for g in gfs if g<0.2),
           dt=round(time.time()-t0,1))
    base_p=os.path.join(HERE,"_audit_baseline.json")
    warn=""
    if os.path.exists(base_p):
        b=json.load(open(base_p))
        if m['n_solved'] < b.get('n_solved',0)-3: warn=f" ⚠REGRESYON solved {b['n_solved']}->{m['n_solved']}"
    else:
        json.dump(m,open(base_p,"w"))
    print(json.dumps(m,ensure_ascii=False)+warn)

if __name__=="__main__": main()
