#!/usr/bin/env python3
"""VISION joint-lens kalibrasyon: 4 köşe + merkez-yuvarlak (4 kardinal nokta) -> (k1,k2,H) AYNI ANDA fit.
Ortak-lens sabit kalınca head-on sahalarda foreshortening eksik -> iç-işaret kayıyor (vision-WF 7/14 rough).
Lens'i serbest bırakıp yuvarlak-kısıtı ekleyince per-saha distortion çözülür -> iç-işaretler oturur.
corners: [BL,TL,BR,TR] -> kanonik [(0,0),(0,W),(L,0),(L,W)]
circle_pts: [top,bottom,left,right] image -> kanonik yuvarlak kardinal (L/2,W/2±r)/(L/2±r,W/2), r=3
"""
import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.optimize import least_squares
from calib_train import auto_calib as AC
from calib_train.vision_calib import overlay

def calibrate_joint(corners, w, h, L=40.0, W=20.0, circle_pts=None, r=3.0, k1_0=0.167, k2_0=0.240):
    cx,cy,s=w/2,h/2,w/2
    img_pts=[list(c) for c in corners]; world=[[0,0],[0,W],[L,0],[L,W]]
    if circle_pts is not None and len(circle_pts)==4:
        img_pts += [list(c) for c in circle_pts]
        world += [[L/2,W/2+r],[L/2,W/2-r],[L/2-r,W/2],[L/2+r,W/2]]   # top,bottom,left,right
    img_pts=np.array(img_pts,float); world=np.array(world,float)
    # init H ortak-lensle
    un0=AC.und_norm(img_pts,k1_0,k2_0,cx,cy,s); H0,_=cv2.findHomography(un0,world)
    if H0 is None: return dict(ok=False,reason="init H tekil")
    H0=H0/H0[2,2]; p0=[k1_0,k2_0,*H0.ravel()[:8]]
    LAM=8.0   # lens-prior regularizasyonu: ortak-lens'e yakın kal (extreme'e kaçma)
    def resid(p):
        k1,k2=p[0],p[1]; H=np.array([[p[2],p[3],p[4]],[p[5],p[6],p[7]],[p[8],p[9],1.0]])
        un=AC.und_norm(img_pts,k1,k2,cx,cy,s); m=AC.aH(un,H)
        reg=[LAM*(k1-k1_0), LAM*(k2-k2_0)]   # prior: shared lens
        return np.concatenate([(m-world).ravel(), reg])
    sol=least_squares(resid,p0,method="trf",bounds=([0,0]+[-np.inf]*8,[0.5,0.8]+[np.inf]*8),max_nfev=4000)
    k1,k2=sol.x[0],sol.x[1]; H=np.array([[sol.x[2],sol.x[3],sol.x[4]],[sol.x[5],sol.x[6],sol.x[7]],[sol.x[8],sol.x[9],1.0]])
    fit=float(np.sqrt((resid(sol.x)**2).mean()))
    return dict(ok=True,k1=k1,k2=k2,H=H,fit=fit,L=L,W=W,camside="SOL",source="vision_joint")

if __name__=="__main__":
    import json
    idx=int(sys.argv[1]); corners=[[float(a) for a in c.split(",")] for c in sys.argv[2].split()]
    circle=[[float(a) for a in c.split(",")] for c in sys.argv[3].split()] if len(sys.argv)>3 and sys.argv[3]!="-" else None
    L=float(sys.argv[4]) if len(sys.argv)>4 else 40.0; W=float(sys.argv[5]) if len(sys.argv)>5 else 20.0
    man={e['idx']:e for e in json.load(open("calib_train/night/probcache/manifest.json"))}
    e=man[idx]; img=cv2.imread(f"calib_train/cand_big/{e['file']}"); ih,iw=img.shape[:2]
    from calib_train.auto_clean2d import sanity
    rec=calibrate_joint(corners,iw,ih,L,W,circle)
    ok,why=sanity(rec,iw,ih,img=img) if rec.get("ok") else (False,rec.get("reason"))
    cv2.imwrite(f"calib_train/cand/vcal/_jover_{idx}.jpg", overlay(img,rec,L,W))
    print(f"joint k1={rec.get('k1'):.3f} k2={rec.get('k2'):.3f} fit={rec.get('fit'):.3f} sanity={'PASS' if ok else why} -> _jover_{idx}.jpg")
