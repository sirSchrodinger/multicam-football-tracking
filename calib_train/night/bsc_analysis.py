#!/usr/bin/env python3
"""F1: Boundary Self-Consistency (BSC) — her boundary rolü KENDİ reprojekte çizgisi boyunca
kendi kanal-prob'una oturuyor mu. Golden visual-verdict ile çapraz-doğrula (ayrım temiz mi?).
BSC = mean(gn@near, gf@far, tn@touchN, tf@touchF). Non-gameable: solver residual'ı düz-duvarla
kandırılabilir ama boundary'nin GERÇEK kanal-aktivasyonuna oturması kandırılamaz.
"""
import os, sys, json, numpy as np
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))
from calib_train import auto_calib as AC
from calib_train.multihyp_calib import groups_topk, ROLES, _sample_prob, interior_support
L,Wp=AC.L,18.0
PC=os.path.join(HERE,"probcache_v3")
MAN={r['idx']:r for r in json.load(open(os.path.join(PC,"manifest.json")))}
CEN=json.load(open(os.path.join(ROOT,"cand","_census_hr3.json")))
GOLD=json.load(open(os.path.join(ROOT,"cand","_mh_visual_verdicts.json")))
GOOD={"good","decent"}; BAD={"bad","rough"}  # bad-rotation(...) -> bad-ailesi

def gold_class(idx):
    v=GOLD.get(str(idx))
    if v is None: return None
    v=v.lower()
    if any(g in v for g in GOOD): return "GOOD"
    if "bad" in v or "rough" in v or "rotation" in v: return "BAD"
    return None

def load_prob(idx):
    d=np.load(os.path.join(PC,f"{idx:03d}.npz")); return d['prob'].astype(np.float32), int(d['w']), int(d['h'])

def solve_for(idx):
    ce=CEN.get(str(idx))
    if not ce or ce.get('src') is None: return None,None,None
    prob,w,h=load_prob(idx)
    combo=ce.get('combo',[0,0,0,0]) or [0,0,0,0]
    cands,_=groups_topk(prob,w,h,K=3)
    if cands is None: return None,None,None
    try: groups={r:cands[r][ci] for r,ci in zip(ROLES,combo)}
    except (IndexError,KeyError): groups={r:cands[r][0] for r in ROLES}
    rec=AC.solve_calib(groups,w/2,h/2,w/2)
    return rec,prob,(w,h)

def boundary_scores(rec,prob,w,h):
    k1,k2,H=rec['k1'],rec['k2'],rec['H']
    P=lambda a,b: AC.project_metric(np.linspace(a,b,60),k1,k2,H,w/2,h/2,w/2)
    lines={'gn':(prob[0],[0,0],[0,Wp]),'gf':(prob[1],[L,0],[L,Wp]),
           'tn':(prob[2],[0,0],[L,0]),'tf':(prob[3],[0,Wp],[L,Wp])}
    sc={}
    for k,(ch,a,b) in lines.items():
        m,cov=_sample_prob(ch,P(a,b),w,h); sc[k]=round(m if m else 0.0,3); sc[k+'_cov']=round(cov,2)
    sc['BSC']=round(np.mean([sc['gn'],sc['gf'],sc['tn'],sc['tf']]),3)
    sc['BSCmin']=round(min(sc['gn'],sc['gf'],sc['tn'],sc['tf']),3)
    return sc

def main():
    rows=[]
    for idx in range(121):
        rec,prob,wh=solve_for(idx)
        if rec is None: continue
        w,h=wh; bs=boundary_scores(rec,prob,w,h)
        isup,_=interior_support(rec,prob,w,h)
        rows.append(dict(idx=idx,src=CEN[str(idx)]['src'],fit=round(float(rec['fit']),3),
                         gold=gold_class(idx),interior=round(isup,3),**bs))
    rows.sort(key=lambda r:-r['BSC'])
    print(f"{'idx':>4} {'gold':>5} {'fit':>5} {'gn':>5} {'gf':>5} {'tn':>5} {'tf':>5} {'BSC':>5} {'BSCmn':>5} {'int':>5}")
    print("-"*66)
    for r in rows:
        g={'GOOD':'✓','BAD':'✗'}.get(r['gold'],' ·')
        print(f"{r['idx']:>4} {g:>5} {r['fit']:>5} {r['gn']:>5} {r['gf']:>5} {r['tn']:>5} {r['tf']:>5} {r['BSC']:>5} {r['BSCmin']:>5} {r['interior']:>5}")
    json.dump(rows,open(os.path.join(HERE,"_bsc_scores.json"),"w"),ensure_ascii=False,indent=1)
    # separasyon istatistiği (golden'lı alt-küme)
    G=[r for r in rows if r['gold']=='GOOD']; B=[r for r in rows if r['gold']=='BAD']
    def stat(name,arr,key):
        v=[r[key] for r in arr]; import statistics as st
        return f"{name} n={len(v)} {key} mean={st.mean(v):.3f} med={st.median(v):.3f} min={min(v):.3f} max={max(v):.3f}" if v else f"{name} n=0"
    print("\n=== SEPARASYON (golden alt-küme) ===")
    for key in ('BSC','gf','BSCmin','interior','fit'):
        print(stat("GOOD",G,key)); print(stat("BAD ",B,key)); print()
    # en iyi tek-eşik BSC (Youden)
    if G and B:
        allv=sorted(set([r['BSC'] for r in G+B]))
        best=None
        for t in allv:
            tp=sum(1 for r in G if r['BSC']>=t); fn=len(G)-tp
            tn=sum(1 for r in B if r['BSC']<t);  fp=len(B)-tn
            tpr=tp/len(G); tnr=tn/len(B); j=tpr+tnr-1
            if best is None or j>best[0]: best=(j,t,tpr,tnr)
        print(f"En iyi BSC eşiği={best[1]:.3f} Youden J={best[0]:.2f} (GOOD-recall={best[2]:.2f}, BAD-reject={best[3]:.2f})")
    print(f"-> {os.path.join(HERE,'_bsc_scores.json')}")

if __name__=="__main__": main()
