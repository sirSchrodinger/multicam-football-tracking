#!/usr/bin/env python3
"""YÖNTEM: multicomp_interior.
Far-rol (goalF=1, touchF=3) için EN BÜYÜK bileşen fence/duvar olabilir -> top-K bileşeni
aday olarak dene. Near-rol (goalN=0, touchN=2) ve yakın çizgiler güvenilir -> largest.
Her (goalF_cand x touchF_cand) kombinasyonu için joint_calib çöz, sanity uygula,
SONRA interior_consistency (center+box+circle reprojeksiyonu, prob[4,5,6] iç-kanala örtüşme)
hesapla. KABUL = sanity OK AND (interior >= thr  VEYA  iç-sinyal yok ise düşük-residual fallback).
Adaylar arasında: önce interior skoru (yüksek), sonra düşük residual ile sırala.

calib_from_pred_mc(prob,w,h,img=None) -> rec (auto_calib.calib_from_pred ile aynı şema +
  meta: 'interior','n_cand','from_largest').
ORİJİNAL auto_calib BOZULMAZ; sadece import edilip sarılır.
"""
import os, numpy as np, cv2
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calib_train import auto_calib as AC
from calib_train import auto_clean2d as AC2
from calib_train import interior_check as IC

NEAR = {0:"goalN", 2:"touchN"}
FAR  = {1:"goalF", 3:"touchF"}
ROLE_ORDER = ["goalN","goalF","touchN","touchF"]

def _components(pc, thr=0.5, minpts=25, maxpts=180, topk=3, rs=None):
    """role prob (h,w full-res) -> list of point-arrays, en büyük 'topk' bileşen (alan azalan)."""
    mk=(pc>thr).astype(np.uint8)
    mk=cv2.morphologyEx(mk,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    ncc,lbl,st,_=cv2.connectedComponentsWithStats(mk)
    if ncc<=1: return []
    order=np.argsort(st[1:,cv2.CC_STAT_AREA])[::-1]+1
    out=[]
    for li in order[:topk]:
        pts=np.column_stack(np.where(lbl==li))[:,::-1].astype(float)
        if len(pts)<minpts: continue
        if len(pts)>maxpts and rs is not None: pts=pts[rs.choice(len(pts),maxpts,replace=False)]
        out.append(pts)
    return out

def _largest_group(pc, thr=0.5, minpts=25, maxpts=180, rs=None):
    c=_components(pc,thr,minpts,maxpts,topk=1,rs=rs)
    return c[0] if c else None

def calib_from_pred_mc(prob, w, h, img=None, thr=0.5, minpts=25,
                       topk=3, interior_thr=0.45):
    cx,cy,s=w/2,h/2,w/2
    rs=np.random.RandomState(0)
    pcf={ci:cv2.resize(prob[ci],(w,h)) for ci in range(4)}
    # near roller: largest (güvenilir)
    near_pts={}
    for ci,role in NEAR.items():
        g=_largest_group(pcf[ci],thr,minpts,180,rs)
        if g is None: return dict(ok=False,reason=f"{role} yok",fit=None)
        near_pts[role]=g
    # far roller: top-K aday
    far_cands={}
    for ci,role in FAR.items():
        cands=_components(pcf[ci],thr,minpts,180,topk,rs)
        if not cands: return dict(ok=False,reason=f"{role} yok",fit=None)
        far_cands[role]=cands

    best=None  # (sort_key, rec)
    n_eval=0
    for gi,gF in enumerate(far_cands["goalF"]):
        for ti,tF in enumerate(far_cands["touchF"]):
            groups={"goalN":near_pts["goalN"],"goalF":gF,
                    "touchN":near_pts["touchN"],"touchF":tF}
            r=AC.solve_calib(groups,cx,cy,s)
            if r is None: continue
            r.update(ok=True,groups=groups)
            sok,why=AC2.sanity(r,w,h,img=img)
            if not sok: continue
            n_eval+=1
            isc=IC.interior_consistency(r,prob,w,h)
            r["interior"]=isc; r["from_largest"]=(gi==0 and ti==0)
            # sıralama anahtarı: interior var ve >=thr ise birinci sınıf (interior yüksek, residual düşük)
            if isc is not None:
                cls = 0 if isc>=interior_thr else 2  # 0 = doğrulanmış iyi, 2 = düşük interior (şüpheli)
                key=(cls, -isc, r["fit"])
            else:
                cls = 1  # iç-sinyal yok -> doğrulanamaz, orta öncelik, residual'a göre
                key=(cls, 0.0, r["fit"])
            if best is None or key<best[0]:
                best=(key,r)
    if best is None:
        return dict(ok=False,reason="hicbir aday sanity gecmedi",fit=None)
    rec=best[1]; rec["n_cand"]=n_eval; rec["sel_class"]=best[0][0]
    return rec
