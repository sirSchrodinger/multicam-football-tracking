#!/usr/bin/env python3
"""Per-VENUE işaret ölçümü (Alperen: 'ceza sahası + orta yuvarlak sahaya göre uyarla, sabit değil').
seg2 ch6(circle)/ch5(box) piksellerini METRİK uzaya projekte edip her sahanın GERÇEK çember-yarıçapı +
ceza-sahası derinlik/genişliğini ölçer. Draw fonksiyonları bunları kullanır → 2D gerçek-havadan-görünüş.
"""
import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calib_train import auto_calib as AC
L,Wp=AC.L,18.0

def _cc_pts(prob,ci,w,h,thr=0.5,minpts=12):
    pc=cv2.resize(prob[ci],(w,h)); mk=(pc>thr).astype(np.uint8)
    mk=cv2.morphologyEx(mk,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8))
    ncc,lbl,st,_=cv2.connectedComponentsWithStats(mk)
    if ncc<=1: return None
    out=[]
    for li in range(1,ncc):
        if st[li,cv2.CC_STAT_AREA]<minpts: continue
        out.append(np.column_stack(np.where(lbl==li))[:,::-1].astype(float))
    return out or None

def img_to_metric(pts,rec,w,h):
    return AC.aH(AC.und_norm(np.asarray(pts,float),rec['k1'],rec['k2'],w/2,h/2,w/2),rec['H'])

def measure_marks(rec,prob,w,h):
    """döner dict(R, box_n(derinlik,yarigen), box_f(...)) — ölçülemeyen None (şablona düş)."""
    out={'R':None,'box_n':None,'box_f':None}
    # --- orta yuvarlak yarıçapı (ch6) ---
    cir=_cc_pts(prob,6,w,h)
    if cir:
        allp=np.vstack(cir); m=img_to_metric(allp,rec,w,h)
        m=m[np.isfinite(m).all(1)]
        # merkeze yakın olanlar (saha-içi, L/2±8, Wp/2±8) — dış-gürültü ele
        c=(np.abs(m[:,0]-L/2)<8)&(np.abs(m[:,1]-Wp/2)<8)
        if c.sum()>=10:
            d=np.hypot(m[c,0]-L/2,m[c,1]-Wp/2); R=float(np.median(d))
            if 1.2<R<5.5: out['R']=round(R,2)
    # --- ceza sahaları (ch5): yakın(X<L/2) / uzak(X>L/2) ayrı; derinlik + yarı-genişlik ---
    box=_cc_pts(prob,5,w,h)
    if box:
        allp=np.vstack(box); m=img_to_metric(allp,rec,w,h); m=m[np.isfinite(m).all(1)]
        for key,near in (('box_n',True),('box_f',False)):
            sel=(m[:,0]<L/2) if near else (m[:,0]>L/2)
            mm=m[sel]
            if len(mm)<12: continue
            # derinlik = kaleden en uzak box-noktası (yakın: max X; uzak: L-min X)
            depth=float(np.percentile(mm[:,0],90)) if near else float(L-np.percentile(mm[:,0],10))
            hw=float((np.percentile(mm[:,1],90)-np.percentile(mm[:,1],10))/2)  # yarı-genişlik
            if 2.0<depth<12.0 and 3.0<hw<9.0: out[key]=(round(depth,2),round(hw,2))
    return out

if __name__=="__main__":
    import sys, os; sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from calib_train.night.four_panel import load
    from calib_train import auto_clean2d as AC2
    for idx in ([int(x) for x in sys.argv[1:]] if len(sys.argv)>1 else [99,43,34,15,52,88]):
        e,img,prob,w,h,feet=load(idx); rec=AC2.calibrate_frame(prob,w,h,img=img,feet=feet)
        print(f"idx{idx} {e['file'].split('__')[0][:20]:20}: {measure_marks(rec,prob,w,h)}")
