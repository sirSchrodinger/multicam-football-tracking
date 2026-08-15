#!/usr/bin/env python3
"""LINE-SEARCH REFINE (accurate calib'in otomatik hali): rough calib (corner-net) -> beklenen saha-çizgilerini
projekte et -> her çizgi boyunca DİK-arama ile GERÇEK beyaz-ridge noktalarını bul -> joint solve_calib (per-saha
lens+H) yeniden-fit. Alperen'in elle-line-fit'inin (7cm) otomatik yaklaşımı. rough yeterince yakınsa accuracy'ye çeker.
"""
import numpy as np, cv2
from calib_train import auto_calib as AC

def _white_ridge(img):
    g=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)
    th=cv2.morphologyEx(g,cv2.MORPH_TOPHAT,cv2.getStructuringElement(cv2.MORPH_RECT,(13,13)))
    return th   # yüksek=beyaz-çizgi olası

def _search_line(ridge, a_pix, b_pix, n=60, half=14):
    """a->b çizgisi boyunca örnekle, her noktada NORMAL yönde ±half ara -> ridge-tepe noktası."""
    h,w=ridge.shape; A=np.array(a_pix,float); B=np.array(b_pix,float); d=B-A
    ln=np.hypot(*d)+1e-9; nrm=np.array([-d[1],d[0]])/ln
    out=[]
    for t in np.linspace(0,1,n):
        c=A+t*d
        best=None; bestv=8.0
        for s in np.linspace(-half,half,2*half+1):
            p=c+s*nrm; x,y=int(p[0]),int(p[1])
            if 0<=x<w and 0<=y<h and ridge[y,x]>bestv: bestv=ridge[y,x]; best=(p[0],p[1])
        if best: out.append(best)
    return np.array(out,float) if len(out)>=8 else None

def refine_line_search(rec, img, L=40.0, W=20.0):
    """döner yeni rec (per-saha lens+H, çizgilere fit) VEYA None (yeterli ridge yok)."""
    h,w=img.shape[:2]; k1,k2,H=rec["k1"],rec["k2"],rec["H"]; cx,cy,s=w/2,h/2,w/2
    ridge=_white_ridge(img)
    # beklenen sınır çizgileri (metrik uçlar) -> projekte -> ridge-ara -> {role:pts}
    ends={"goalN":([0,0],[0,W]),"goalF":([L,0],[L,W]),"touchN":([0,0],[L,0]),"touchF":([0,W],[L,W])}
    groups={}
    for role,(a,b) in ends.items():
        pa=AC.project_metric(np.array([a],float),k1,k2,H,cx,cy,s)[0]
        pb=AC.project_metric(np.array([b],float),k1,k2,H,cx,cy,s)[0]
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()): return None
        pts=_search_line(ridge,pa,pb)
        if pts is None: return None
        groups[role]=pts
    # joint solve (per-saha lens+H) — auto_calib.solve_calib çizgi-noktalarından
    r=AC.solve_calib(groups,cx,cy,s,Wp=W)
    if r is None: return None
    r["L"]=L; r["W"]=W; r["ok"]=True; return r
