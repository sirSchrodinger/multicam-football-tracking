#!/usr/bin/env python3
"""DOĞRU DENETLEME (Alperen): sanity-gate GAMEABLE. Gerçek accuracy = reprojekte saha-çizgileri
GERÇEK beyaz-çizgi piksellerine oturuyor mu (görüntü-kanıtı, kandırılamaz).
line_align_score(rec,img) -> median mesafe (px) reprojekte-çizgi -> en yakın beyaz-pixel. DÜŞÜK=doğru.
"""
import numpy as np, cv2
from calib_train import auto_calib as AC

def white_mask(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    th=cv2.morphologyEx(g,cv2.MORPH_TOPHAT,cv2.getStructuringElement(cv2.MORPH_RECT,(15,15)))
    m=(th>28).astype(np.uint8)
    return m

def line_align_score(rec, img, L=None, W=None):
    """döner (median_px, inlier_frac). L,W rec'ten (34x18 veya 40x20)."""
    h,w=img.shape[:2]; k1,k2,H=rec["k1"],rec["k2"],rec["H"]; cx,cy,s=w/2,h/2,w/2
    L=L or rec.get("L",34.0); W=W or rec.get("W",18.0)
    m=white_mask(img)
    if m.sum()<200: return None,0.0
    dt=cv2.distanceTransform(1-m,cv2.DIST_L2,3)
    pts=[]
    def add(a,b,N=50): pts.append(np.linspace(a,b,N))
    add([0,0],[L,0]); add([L,0],[L,W]); add([L,W],[0,W]); add([0,W],[0,0]); add([L/2,0],[L/2,W])  # sınır+merkez
    for gx in (0,L):
        bx=6 if gx==0 else L-6
        add([gx,W/2-6],[bx,W/2-6],20); add([bx,W/2-6],[bx,W/2+6],20); add([bx,W/2+6],[gx,W/2+6],20)
    P=np.vstack(pts)
    pix=AC.project_metric(P,k1,k2,H,cx,cy,s)
    fin=np.isfinite(pix).all(1)&(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)
    pix=pix[fin]
    if len(pix)<20: return None,0.0
    d=dt[pix[:,1].astype(int),pix[:,0].astype(int)]
    return float(np.median(d)), float((d<6).mean())
