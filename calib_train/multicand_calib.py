#!/usr/bin/env python3
"""ÇOK-ADAY kalibrasyon recovery: largest-component far-çizgi YANLIŞ olabilir (fence/duvar'a kilit).
calib_multicand: önce standart calib_from_pred; sanity FAIL ise far-rol (goalF,touchF) top-K component
combo'larını dene, convex+residual<TAU geçen EN İYİ'yi seç. Confirmation-bias'a karşı: residual-GATE
(sadece convexity yetmez — Playdrome 1.17 reddedilir). Döner (rec, method, ok).
"""
import os, numpy as np, cv2
from itertools import product
from calib_train import auto_calib as AC

def _role_cands(pc,w,h,thr=0.5,minpts=25,topk=3,maxpts=180,seed=0):
    pcr=cv2.resize(pc,(w,h)); mk=(pcr>thr).astype(np.uint8)
    mk=cv2.morphologyEx(mk,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    ncc,lbl,st,_=cv2.connectedComponentsWithStats(mk)
    if ncc<=1: return []
    order=np.argsort(-st[1:,cv2.CC_STAT_AREA])+1
    rs=np.random.RandomState(seed); out=[]
    for li in order[:topk]:
        pts=np.column_stack(np.where(lbl==li))[:,::-1].astype(float)
        if len(pts)<minpts: continue
        if len(pts)>maxpts: pts=pts[rs.choice(len(pts),maxpts,replace=False)]
        out.append(pts)
    return out

def calib_multicand(prob,w,h,sanity_fn,img=None,thr=0.5,tau=0.45,topk=3):
    """sanity_fn(rec,w,h,img) -> (ok,why). önce standart; FAIL ise multi-cand far recovery."""
    cx,cy,s=w/2,h/2,w/2
    rec=AC.calib_from_pred(prob,w,h,thr=thr)
    if rec.get("ok"):
        ok,why=sanity_fn(rec,w,h,img=img)
        if ok and rec["fit"]<tau: return rec,"largest",True
    # recovery: far-rol top-K combo
    gn=_role_cands(prob[0],w,h,thr=thr,topk=1); tn=_role_cands(prob[2],w,h,thr=thr,topk=1)
    gf=_role_cands(prob[1],w,h,thr=thr,topk=topk); tf=_role_cands(prob[3],w,h,thr=thr,topk=topk)
    if not(gn and tn and gf and tf): return rec,"largest",False
    best=None
    for gi,ti in product(range(len(gf)),range(len(tf))):
        if gi==0 and ti==0: continue   # largest zaten denendi
        groups={"goalN":gn[0],"goalF":gf[gi],"touchN":tn[0],"touchF":tf[ti]}
        r=AC.solve_calib(groups,cx,cy,s)
        if r is None: continue
        rr=dict(ok=True,groups=groups,**r); ok,_=sanity_fn(rr,w,h,img=img)
        if ok and r["fit"]<tau and (best is None or r["fit"]<best["fit"]):
            best=rr
    if best is not None: return best,"multicand",True
    return rec,"largest",False
