#!/usr/bin/env python3
"""F2-karar: TAZE agent-judge verdict'leriyle gf@far'ın (far-goal grounding) GOOD+DECENT vs BAD
ayrımını ölç. Ayırıyorsa rejection-gate eşiği öner. Girdi: cand/_judge_results.json."""
import os, sys, json, numpy as np, statistics as st
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))
from calib_train import auto_calib as AC
from calib_train.multihyp_calib import groups_topk, ROLES, _sample_prob
L,Wp=AC.L,18.0
PC=os.path.join(HERE,"probcache_v3")
CEN=json.load(open(os.path.join(ROOT,"cand","_census_hr3.json")))
JUD=json.load(open(os.path.join(ROOT,"cand","_judge_results.json")))
rows_j=JUD.get("per_idx",JUD) if isinstance(JUD,dict) else JUD
FINAL={r['idx']:r.get('final',r.get('judge')) for r in rows_j}

def gf_of(idx):
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
    gn,_=_sample_prob(prob[0],AC.project_metric(np.linspace([0,0],[0,Wp],60),rec['k1'],rec['k2'],rec['H'],w/2,h/2,w/2),w,h)
    return float(gf or 0.0), float(gn or 0.0), float(rec['fit'])

def main():
    G,B=[],[]; rowout=[]
    for idx,verd in FINAL.items():
        r=gf_of(idx)
        if r is None: continue
        gf,gn,fit=r
        cls='GOOD' if verd in ('GOOD','DECENT') else ('BAD' if verd in ('BAD','ROUGH','UNSOLVABLE') else None)
        rowout.append((idx,verd,round(gf,3),round(gn,3),round(max(gf,gn),3),round(fit,3)))
        if cls=='GOOD': G.append((gf,gn,max(gf,gn)))
        elif cls=='BAD': B.append((gf,gn,max(gf,gn)))
    rowout.sort(key=lambda x:-x[2])
    print(f"{'idx':>4} {'verdict':>10} {'gf@far':>6} {'gn@nr':>6} {'max':>6} {'fit':>5}")
    for idx,verd,gf,gn,mx,fit in rowout:
        print(f"{idx:>4} {str(verd):>10} {gf:>6} {gn:>6} {mx:>6} {fit:>5}")
    for name,key,i in [("gf@far",'gf',0),("max(gf,gn)",'mx',2)]:
        gv=[x[i] for x in G]; bv=[x[i] for x in B]
        if not gv or not bv: continue
        print(f"\n{name}: GOOD+DEC n={len(gv)} mean={st.mean(gv):.3f} med={st.median(gv):.3f} | BAD-family n={len(bv)} mean={st.mean(bv):.3f} med={st.median(bv):.3f}")
        # Youden en iyi eşik
        allv=sorted(set(gv+bv)); best=None
        for t in allv:
            tp=sum(1 for v in gv if v>=t); tn=sum(1 for v in bv if v<t)
            j=tp/len(gv)+tn/len(bv)-1
            if best is None or j>best[0]: best=(j,t,tp/len(gv),tn/len(bv))
        print(f"   en iyi eşik={best[1]:.3f} Youden J={best[0]:.2f} (GOOD-keep={best[2]:.2f} BAD-reject={best[3]:.2f})")

if __name__=="__main__": main()
