#!/usr/bin/env python3
"""PRENSİPLİ calib doğrulayıcı: residual+convexity YETMİYOR (gate-overfit). Asıl test = iç-işaretler.
Doğru calib'in reprojekte merkez-çizgi/ceza-sahası/merkez-yuvarlağı, MODELİN tahmin ettiği iç-kanallara
(center=4, box=5, circle=6) oturmalı. interior_consistency: reprojekte iç-işaret piksellerinin, tahmin-iç-maskeye
yakınlık oranı (chamfer-style). Yüksek=genuine. İç-kanal sinyali yoksa -> unverified (None)."""
import numpy as np, cv2
from calib_train import auto_calib as AC

def interior_consistency(rec, prob, w, h, thr=0.30, dist=14):
    """döner (score 0-1 | None). None = iç-kanal sinyali yetersiz (doğrulanamaz)."""
    k1,k2,H=rec["k1"],rec["k2"],rec["H"]; cx,cy,s=w/2,h/2,w/2; L,Wp=AC.L,18.0
    # tahmin-edilen iç-maske (center|box|circle), tam-res
    pred=np.zeros((h,w),np.uint8)
    for ci in (4,5,6):
        pc=cv2.resize(prob[ci],(w,h)); pred|=(pc>thr).astype(np.uint8)
    if pred.sum()<150: return None    # iç-işaret görünmüyor -> doğrulanamaz
    # tahmin-maskeye mesafe haritası
    dt=cv2.distanceTransform(1-pred,cv2.DIST_L2,3)
    # reprojekte iç-işaretler
    proj=[]
    def add(pts):
        pix=AC.project_metric(np.asarray(pts,float),k1,k2,H,cx,cy,s)
        fin=np.isfinite(pix).all(1)&(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)
        proj.append(pix[fin])
    th=np.linspace(0,2*np.pi,80); add(np.stack([L/2+3*np.cos(th),Wp/2+3*np.sin(th)],1))   # merkez-yuvarlak
    add(np.stack([np.full(40,L/2),np.linspace(0,Wp,40)],1))                                 # merkez-çizgi
    for gx in (0,L):                                                                         # ceza-sahaları
        bx=5 if gx==0 else L-5
        add(np.stack([np.linspace(gx,bx,20),np.full(20,Wp/2-5)],1)); add(np.stack([np.linspace(gx,bx,20),np.full(20,Wp/2+5)],1))
    P=np.vstack([p for p in proj if len(p)]) if any(len(p) for p in proj) else np.empty((0,2))
    if len(P)<20: return None
    d=dt[P[:,1].astype(int),P[:,0].astype(int)]
    return float((d<dist).mean())    # yakınlık oranı
