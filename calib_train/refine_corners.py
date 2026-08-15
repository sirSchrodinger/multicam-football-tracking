#!/usr/bin/env python3
"""LINE-REFINEMENT: kaba köşeleri (corner-net) GERÇEK beyaz-çizgilere SNAP et.
corner-net rough calib -> reprojekte saha-çizgileri beyaz-çizgiye YAKIN ama birkaç-px kayık.
Köşeleri optimize et (reprojekte çizgi -> en yakın beyaz-pixel mesafesi MINIMIZE) -> accuracy.
refine(corners,img,L,W) -> düzeltilmiş corners. Ortak-lens tavanını kısmen aşar (H'yi çizgiye oturtur)."""
import numpy as np, cv2
from scipy.optimize import minimize
from calib_train import auto_calib as AC
from calib_train.vision_calib import calibrate_from_corners
from calib_train.line_align import white_mask

def _line_pts(L,W):
    seg=[]
    def add(a,b,N=40): seg.append(np.linspace(a,b,N))
    add([0,0],[L,0]);add([L,0],[L,W]);add([L,W],[0,W]);add([0,W],[0,0]);add([L/2,0],[L/2,W])
    for gx in (0,L):
        bx=6 if gx==0 else L-6
        add([gx,W/2-6],[bx,W/2-6],18);add([bx,W/2-6],[bx,W/2+6],18);add([bx,W/2+6],[gx,W/2+6],18)
    return np.vstack(seg)

def refine(corners, img, L=40.0, W=20.0, iters=1):
    h,w=img.shape[:2]; m=white_mask(img)
    if m.sum()<200: return np.asarray(corners,float)
    dt=cv2.distanceTransform(1-m,cv2.DIST_L2,3); dt=np.clip(dt,0,25)
    P=_line_pts(L,W); c0=np.asarray(corners,float).ravel()
    def cost(cv):
        rec=calibrate_from_corners(cv.reshape(4,2),L,W,w,h)
        if not rec.get("ok"): return 1e6
        pix=AC.project_metric(P,rec["k1"],rec["k2"],rec["H"],w/2,h/2,w/2)
        fin=np.isfinite(pix).all(1)
        pin=pix[fin&(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)]
        if len(pin)<len(P)*0.4: return 1e6   # çoğu off-image -> ceza
        d=dt[pin[:,1].astype(int),pin[:,0].astype(int)]
        return d.mean()+25*(1-len(pin)/len(P))   # ort-mesafe + off-image cezası
    res=minimize(cost,c0,method="Powell",options={"maxiter":400,"xtol":1.0,"ftol":0.5})
    return res.x.reshape(4,2) if res.fun<cost(c0) else np.asarray(corners,float)
