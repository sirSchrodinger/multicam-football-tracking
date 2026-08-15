#!/usr/bin/env python3
"""Alperen sorusu: çizgi-tespiti iyi ama warp/homografi kayık — çünkü solve yalnız 4-sınır kullanıyor,
iyi-tespit-edilen ORTA-ÇEMBER + ORTA-ÇİZGİyi kısıt yapmıyor. Bunları solve'a ekleyip warp düzeliyor mu?
"""
import os, sys, json, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
HERE=os.path.dirname(os.path.abspath(__file__)); CT=os.path.dirname(HERE); ROOT=os.path.dirname(CT)
sys.path.insert(0, ROOT)
from calib_train import auto_calib as AC
from calib_train.pretty_render import render_overlay, venue_name
from scipy.optimize import least_squares
L,Wp=AC.L,18.0
MAN={r['idx']:r for r in json.load(open(os.path.join(HERE,"probcache_v3","manifest.json")))}

def cc_pts(prob,ci,w,h,thr=0.5,minpts=15,maxpts=140,seed=0):
    pc=cv2.resize(prob[ci],(w,h)); mk=(pc>thr).astype(np.uint8)
    mk=cv2.morphologyEx(mk,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    ncc,lbl,st,_=cv2.connectedComponentsWithStats(mk)
    if ncc<=1: return None
    li=1+int(np.argmax(st[1:,cv2.CC_STAT_AREA]))
    pts=np.column_stack(np.where(lbl==li))[:,::-1].astype(float)
    if len(pts)<minpts: return None
    if len(pts)>maxpts: pts=pts[np.random.RandomState(seed).choice(len(pts),maxpts,replace=False)]
    return pts

def solve_center(groups,center_pts,circle_pts,cx,cy,s,base,wc=2.0,wr=3.0):
    CON=[("goalN","X",0.0),("goalF","X",L),("touchN","Y",0.0),("touchF","Y",Wp)]
    h0=[float(base['k1']),float(base['k2']),*base['H'].ravel()[:8].tolist()]
    def resid(pp):
        k1,k2=pp[0],pp[1];H=np.array([[pp[2],pp[3],pp[4]],[pp[5],pp[6],pp[7]],[pp[8],pp[9],1.0]]);out=[]
        for r,kind,tg in CON:
            m=AC.aH(AC.und_norm(groups[r],k1,k2,cx,cy,s),H); out.append((m[:,0]-tg) if kind=="X" else (m[:,1]-tg))
        if center_pts is not None:
            m=AC.aH(AC.und_norm(center_pts,k1,k2,cx,cy,s),H); out.append(wc*(m[:,0]-L/2))          # orta çizgi X=L/2
        if circle_pts is not None:
            m=AC.aH(AC.und_norm(circle_pts,k1,k2,cx,cy,s),H)                                        # çember merkezi=(L/2,Wp/2)
            out.append(np.array([wr*(m[:,0].mean()-L/2), wr*(m[:,1].mean()-Wp/2)]))
        return np.concatenate(out)
    sol=least_squares(resid,h0,method="trf",x_scale="jac",
                      bounds=([0,0]+[-np.inf]*8,[0.45,0.7]+[np.inf]*8),max_nfev=6000)
    k1,k2=sol.x[0],sol.x[1];H=np.array([[sol.x[2],sol.x[3],sol.x[4]],[sol.x[5],sol.x[6],sol.x[7]],[sol.x[8],sol.x[9],1.0]])
    r0=resid([float(base['k1']),float(base['k2']),*base['H'].ravel()[:8]])
    return dict(k1=k1,k2=k2,H=H,fit=float(np.sqrt((resid(sol.x)**2).mean())),camside=base['camside'],src='center'), float(np.sqrt((r0**2).mean()))

def raw_warp(img,rec,w,h,tag):
    S=22;pad=16; Wc=int(L*S+2*pad);Hc=int(Wp*S+2*pad)
    gx,gy=np.meshgrid(np.arange(Wc),np.arange(Hc)); Xm=(gx-pad)/S;Ym=Wp-(gy-pad)/S
    pix=AC.project_metric(np.stack([Xm.ravel(),Ym.ravel()],1),rec['k1'],rec['k2'],rec['H'],w/2,h/2,w/2)
    wp=cv2.remap(img,pix[:,0].reshape(Hc,Wc).astype(np.float32),pix[:,1].reshape(Hc,Wc).astype(np.float32),cv2.INTER_LINEAR,borderValue=(30,30,30))
    def P2(X,Y): return int(pad+X*S),int(pad+(Wp-Y)*S)
    cv2.rectangle(wp,P2(0,0),P2(L,Wp),(0,255,255),2); cv2.line(wp,P2(L/2,0),P2(L/2,Wp),(0,255,255),1); cv2.circle(wp,P2(L/2,Wp/2),int(3*S),(0,255,255),1)
    cv2.rectangle(wp,(0,0),(wp.shape[1],26),(0,0,0),-1); cv2.putText(wp,tag,(6,19),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
    return wp

def circle_support(rec,prob,w,h,Rlist=(2.0,2.5,3.0,3.5)):
    """reprojekte kanonik çemberin ch6(circle) aktivasyonuna oturması (non-gameable keep-best)."""
    from calib_train.multihyp_calib import _sample_prob
    k1,k2,H=rec['k1'],rec['k2'],rec['H']; th=np.linspace(0,2*np.pi,60); best=0.0
    for R in Rlist:
        pix=AC.project_metric(np.c_[L/2+R*np.cos(th),Wp/2+R*np.sin(th)],k1,k2,H,w/2,h/2,w/2)
        m,cov=_sample_prob(prob[6],pix,w,h)
        if m is not None and cov>0.4: best=max(best,m)
    return best

def choose_best(idx, prob, w, h, img, feet):
    """base vs +center: guard'lı keep-best (çember-desteği + sanity)."""
    from calib_train import auto_clean2d as AC2
    from calib_train.auto_calib import groups_from_prob
    base=AC2.calibrate_frame(prob,w,h,img=img,feet=feet)
    if not base.get('ok'): return base,'base(only)'
    groups=base.get('groups') or groups_from_prob(prob,w,h)[0]
    if groups is None: return base,'base(no-groups)'
    cen=cc_pts(prob,4,w,h); cir=cc_pts(prob,6,w,h)
    if cir is None or len(cir)<40: return base,'base(çember-yok)'
    rec2,_=solve_center(groups,cen,cir,w/2,h/2,w/2,base)
    ok2,_=AC2.sanity(rec2,w,h,img=img)
    if not ok2: return base,'base(center-sanity-fail)'
    sb=circle_support(base,prob,w,h); sc=circle_support(rec2,prob,w,h)
    if sc>sb+0.03: rec2['src']='center'; return rec2,f'CENTER (çember-destek {sb:.2f}->{sc:.2f})'
    return base,f'base (çember-destek base {sb:.2f} >= center {sc:.2f})'

def main(idx):
    from calib_train.night.four_panel import load
    from calib_train import auto_clean2d as AC2
    e,img,prob,w,h,feet=load(idx); cx,cy,s=w/2,h/2,w/2
    base=AC2.calibrate_frame(prob,w,h,img=img,feet=feet)
    groups=base['groups'] if 'groups' in base else None
    if groups is None:
        from calib_train.auto_calib import groups_from_prob
        groups,_=groups_from_prob(prob,w,h)
    cen=cc_pts(prob,4,w,h); cir=cc_pts(prob,6,w,h)
    print(f"idx{idx}: base fit={base['fit']:.3f} | center-pts={0 if cen is None else len(cen)} circle-pts={0 if cir is None else len(cir)}")
    rec2,base_resid=solve_center(groups,cen,cir,cx,cy,s,base)
    print(f"  center-constrained fit={rec2['fit']:.3f} (aynı-resid'de base={base_resid:.3f})")
    # yan-yana warp: base vs center
    wb=raw_warp(img,base,w,h,f"BASE warp (fit={base['fit']:.2f})")
    wc=raw_warp(img,rec2,w,h,f"+CENTER warp (fit={rec2['fit']:.2f})")
    cv2.imwrite(f"{CT}/cand/_CENTERFIX_{idx}.jpg",np.hstack([wb,np.full((wb.shape[0],6,3),80,np.uint8),wc]))
    # overlay (center)
    cv2.imwrite(f"{CT}/cand/_OVERLAY_CENTER_{idx}.jpg",render_overlay(img,rec2,venue=venue_name(e['file'])+" +center",idx=idx,judge=True))
    print(f"  -> _CENTERFIX_{idx}.jpg (base|center warp) + _OVERLAY_CENTER_{idx}.jpg")

if __name__=="__main__":
    for idx in ([int(x) for x in sys.argv[1:]] if len(sys.argv)>1 else [99]): main(idx)
