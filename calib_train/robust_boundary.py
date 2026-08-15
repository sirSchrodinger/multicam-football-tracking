#!/usr/bin/env python3
"""Alperen: EFT yakın-kalede çizgi bir yerden sonra 3B KALE DİREĞİni takip ediyor → direk-pikselleri
outlier, solve'u çekiyor. FIX: her sınır-çizgisini UNDISTORT-uzayında (fisheye kalkınca DÜZ olur) fit et,
çizgiden uzak (direk/gürültü) pikselleri AT, inlier'larla yeniden çöz. 'Kale direğine kadar olan iyi' → onu kullan.
"""
import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calib_train import auto_calib as AC
L,Wp=AC.L,18.0

def _inliers_undist(pts, k1, k2, cx, cy, s, keep=0.75, min_keep=20):
    """undistort -> robust düz-çizgi fit -> perpendicular residual -> inlier (direk/gürültü at)."""
    un=AC.und_norm(np.asarray(pts,float),k1,k2,cx,cy,s)
    vx,vy,x0,y0=cv2.fitLine(un.astype(np.float32),cv2.DIST_HUBER,0,0.01,0.01).ravel()
    a,b=-vy,vx; nrm=np.hypot(a,b)+1e-9; c=-(a*x0+b*y0)
    d=np.abs(a*un[:,0]+b*un[:,1]+c)/nrm
    med=np.median(d); mad=np.median(np.abs(d-med))+1e-6
    thr=max(med+2.5*mad, np.quantile(d,keep))   # MAD-robust + en az %keep tut
    m=d<=thr
    if m.sum()<min_keep: m=d<=np.quantile(d,max(keep,0.85))
    return pts[m], int(len(pts)-m.sum())

def robust_groups(groups, k1, k2, cx, cy, s):
    out={}; dropped={}
    for r,pts in groups.items():
        ip,nd=_inliers_undist(pts,k1,k2,cx,cy,s); out[r]=ip; dropped[r]=nd
    return out, dropped

def solve_robust_boundary(prob, w, h, base, cen=None, cir=None, sanity_fn=None, img=None):
    """base rec'in k1,k2'siyle sınır-piksellerini inlier-filtrele → yeniden çöz (+ center/circle)."""
    from calib_train.center_refine import _solve_center
    from calib_train.auto_calib import groups_from_prob
    groups=base.get('groups') or groups_from_prob(prob,w,h)[0]
    if groups is None: return base, {}
    cx,cy,s=w/2,h/2,w/2
    ig, dropped=robust_groups(groups,base['k1'],base['k2'],cx,cy,s)
    if any(len(v)<15 for v in ig.values()): return base, dropped
    rec2=_solve_center(ig,cen,cir,cx,cy,s,base)   # inlier-sınır + (varsa) center/circle
    if rec2 is None or not rec2.get('converged',True): return base, dropped
    if sanity_fn is not None:
        ok,_=sanity_fn(rec2,w,h,img=img)
        if not ok: return base, dropped
    rec2['src']=base.get('src','base')+'+rob'; rec2['groups']=ig; rec2['dropped']=dropped
    return rec2, dropped

if __name__=="__main__":
    from calib_train.night.four_panel import load, p_warp
    from calib_train import auto_clean2d as AC2
    from calib_train.center_refine import _cc
    from calib_train.measure_marks import measure_marks
    CT=os.path.dirname(os.path.abspath(__file__))
    for idx in ([int(x) for x in sys.argv[1:]] if len(sys.argv)>1 else [52,99,55,37]):
        e,img,prob,w,h,feet=load(idx); base=AC2.calibrate_frame(prob,w,h,img=img,feet=feet)
        cen=_cc(prob,4,w,h); cir=_cc(prob,6,w,h)
        rec2,dropped=solve_robust_boundary(prob,w,h,base,cen,cir,sanity_fn=AC2.sanity,img=img)
        rec2['marks']=measure_marks(rec2,prob,w,h) if rec2.get('ok') else None
        print(f"idx{idx}: base src={base.get('src')} fit={base['fit']:.3f} | robust src={rec2.get('src')} fit={rec2.get('fit',0):.3f} atılan={dropped}")
        wb=p_warp(img,base,w,h); wr=p_warp(img,rec2,w,h)
        H=min(wb.shape[0],wr.shape[0]); montage=np.hstack([cv2.resize(wb,(int(wb.shape[1]*H/wb.shape[0]),H)),np.full((H,6,3),80,np.uint8),cv2.resize(wr,(int(wr.shape[1]*H/wr.shape[0]),H))])
        cv2.imwrite(f"{CT}/cand/_ROBBND_{idx}.jpg",montage); print(f"  -> _ROBBND_{idx}.jpg (sol=mevcut sağ=robust-inlier)")
