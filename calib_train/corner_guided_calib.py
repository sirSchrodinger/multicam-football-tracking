#!/usr/bin/env python3
"""SENTEZ: corner-net(robust rough-init) + seg2(çizgi-detay) -> corner-rehberli çizgi-seçimi -> accurate joint_calib.
Bowtie kökü = YANLIŞ far-line-component (fence-lock). corner-net'in beklediği çizgiye en yakın seg-component'i seç
-> solve_calib (per-saha lens+H) -> temiz warp. corner-net(generalize) + seg2(Alperen-etiketli detay) birleşimi.
"""
import numpy as np, cv2
from calib_train import auto_calib as AC

def _components(pc, w, h, thr=0.4, minpts=20, topk=4, maxpts=200):
    pcr=cv2.resize(pc,(w,h)); mk=(pcr>thr).astype(np.uint8)
    mk=cv2.morphologyEx(mk,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    ncc,lbl,st,_=cv2.connectedComponentsWithStats(mk)
    if ncc<=1: return []
    order=np.argsort(st[1:,cv2.CC_STAT_AREA])[::-1]+1; out=[]; rs=np.random.RandomState(0)
    for li in order[:topk]:
        pts=np.column_stack(np.where(lbl==li))[:,::-1].astype(float)
        if len(pts)<minpts: continue
        if len(pts)>maxpts: pts=pts[rs.choice(len(pts),maxpts,replace=False)]
        out.append(pts)
    return out

def _line_dist(pts, a, b):
    """pts'in a-b çizgisine ortalama mesafesi (component'in beklenen çizgiye yakınlığı)."""
    a=np.array(a,float); b=np.array(b,float); d=b-a; ln=np.hypot(*d)+1e-9; nrm=np.array([-d[1],d[0]])/ln
    return float(np.abs((pts-a)@nrm).mean())

def corner_guided(prob, corners, w, h, Wp=18.0):
    """corners [BL,TL,BR,TR] (image px, corner-net). prob seg2 (>=4 rol). döner accurate rec veya None."""
    cx,cy,s=w/2,h/2,w/2
    BL,TL,BR,TR=[np.array(c,float) for c in corners]
    expect={"goalN":(BL,TL),"goalF":(BR,TR),"touchN":(BL,BR),"touchF":(TL,TR)}  # corner-net beklenen çizgiler
    groups={}
    for ci,role in enumerate(["goalN","goalF","touchN","touchF"]):
        cands=_components(prob[ci],w,h)
        if not cands: return None
        a,b=expect[role]
        best=min(cands,key=lambda p:_line_dist(p,a,b))  # beklenen çizgiye en yakın component
        # aşırı-uzaksa (yanlış) reddet
        if _line_dist(best,a,b) > 0.15*np.hypot(w,h): return None
        groups[role]=best
    r=AC.solve_calib(groups,cx,cy,s,Wp=Wp)
    if r is None: return None
    r["ok"]=True; r["L"]=AC.L; r["W"]=Wp; return r
