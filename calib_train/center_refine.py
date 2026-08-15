#!/usr/bin/env python3
"""ORTA-ÇEMBER/ÇİZGİ ile kalibrasyon rafinesi (2 Tem gece, Alperen: 'çizgi iyi ama warp kayık').
KÖK: base solve yalnız 4-sınır kullanıyor, iyi-tespit-edilen orta-çember + orta-çizgiyi KISIT yapmıyor
→ orta-saha under-constrained → homografi kayıyor (çemberi 2-14m yanlış koyuyor), warp büyütüyor.
FIX: çember-merkezi=(L/2,Wp/2) + orta-çizgi X=L/2 kısıtlarını solve'a ekle. GUARD'lı keep-best:
reprojekte-çemberin ch6-aktivasyonuna oturması (non-gameable) base'i geçerse VE sanity tutarsa KULLAN.
"""
import numpy as np, cv2
from calib_train import auto_calib as AC
from scipy.optimize import least_squares
L,Wp=AC.L,18.0

def _cc(prob,ci,w,h,thr=0.5,minpts=15,maxpts=140,seed=0):
    pc=cv2.resize(prob[ci],(w,h)); mk=(pc>thr).astype(np.uint8)
    mk=cv2.morphologyEx(mk,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    ncc,lbl,st,_=cv2.connectedComponentsWithStats(mk)
    if ncc<=1: return None
    li=1+int(np.argmax(st[1:,cv2.CC_STAT_AREA]))
    pts=np.column_stack(np.where(lbl==li))[:,::-1].astype(float)
    if len(pts)<minpts: return None
    if len(pts)>maxpts: pts=pts[np.random.RandomState(seed).choice(len(pts),maxpts,replace=False)]
    return pts

def _solve_center(groups,cen,cir,cx,cy,s,base,wc=2.0,wr=3.0):
    CON=[("goalN","X",0.0),("goalF","X",L),("touchN","Y",0.0),("touchF","Y",Wp)]
    h0=[float(base['k1']),float(base['k2']),*np.asarray(base['H']).ravel()[:8].tolist()]
    def resid(pp):
        k1,k2=pp[0],pp[1];H=np.array([[pp[2],pp[3],pp[4]],[pp[5],pp[6],pp[7]],[pp[8],pp[9],1.0]]);out=[]
        for r,kind,tg in CON:
            m=AC.aH(AC.und_norm(groups[r],k1,k2,cx,cy,s),H); out.append((m[:,0]-tg) if kind=="X" else (m[:,1]-tg))
        if cen is not None:
            m=AC.aH(AC.und_norm(cen,k1,k2,cx,cy,s),H); out.append(wc*(m[:,0]-L/2))
        if cir is not None:
            m=AC.aH(AC.und_norm(cir,k1,k2,cx,cy,s),H)
            out.append(np.array([wr*(m[:,0].mean()-L/2), wr*(m[:,1].mean()-Wp/2)]))
        return np.concatenate(out)
    try:
        sol=least_squares(resid,h0,method="trf",x_scale="jac",
                          bounds=([0,0]+[-np.inf]*8,[0.45,0.7]+[np.inf]*8),max_nfev=6000)
    except Exception:
        return None
    k1,k2=sol.x[0],sol.x[1];H=np.array([[sol.x[2],sol.x[3],sol.x[4]],[sol.x[5],sol.x[6],sol.x[7]],[sol.x[8],sol.x[9],1.0]])
    return dict(k1=k1,k2=k2,H=H,fit=float(np.sqrt((resid(sol.x)**2).mean())),
                camside=base['camside'],converged=bool(sol.success))

def circle_support(rec,prob,w,h,Rlist=(2.0,2.5,3.0,3.5)):
    from calib_train.multihyp_calib import _sample_prob
    th=np.linspace(0,2*np.pi,60); best=0.0
    for R in Rlist:
        pix=AC.project_metric(np.c_[L/2+R*np.cos(th),Wp/2+R*np.sin(th)],rec['k1'],rec['k2'],rec['H'],w/2,h/2,w/2)
        m,cov=_sample_prob(prob[6],pix,w,h)
        if m is not None and cov>0.4: best=max(best,m)
    return best

def refine_center(rec, prob, w, h, img=None, sanity_fn=None):
    """rec (base/multihyp/vp) -> çember-kısıtlı keep-best rec. Çember yoksa/yardımcı değilse rec döner."""
    if not rec or not rec.get('ok'): return rec
    cir=_cc(prob,6,w,h)
    if cir is None or len(cir)<40: return rec           # çember güvenli tespit yok
    groups=rec.get('groups')
    if groups is None:
        from calib_train.auto_calib import groups_from_prob
        groups,_=groups_from_prob(prob,w,h)
        if groups is None: return rec
    cen=_cc(prob,4,w,h)
    rec2=_solve_center(groups,cen,cir,w/2,h/2,w/2,rec)
    if rec2 is None or not rec2.get('converged',True): return rec
    if sanity_fn is not None:
        ok2,_=sanity_fn(rec2,w,h,img=img)
        if not ok2: return rec
    sb=circle_support(rec,prob,w,h); sc=circle_support(rec2,prob,w,h)
    if sc>sb+0.03:
        rec2['src']=rec.get('src','base')+'+center'; rec2['ok']=True
        rec2['circle_support']=(round(sb,3),round(sc,3))
        return rec2
    return rec

# ---- RIGID keep-best (skew fix): fisheye'ı popülasyon-prior'una dondur + uzak-çizgi down-weight ----
# KÖK: base solver fisheye'ı [0,0.45] serbest bırakıyor -> gürültülü uzak-piksel k1/k2'yi bozarak
# "açıklıyor" -> undistort eğri -> warp SHEAR. FIX: k1,k2'yi prior±%20'ye kısıtla, uzak çizgileri
# down-weight et. YALNIZ keep-best: çember-yuvarlaklık (shear ölçüsü) İYİLEŞİRSE kabul (idx1 0.99>rigid'i korur).
K1P, K2P = 0.167, 0.240
def solve_rigid(groups, cen, cir, cx, cy, s, base, wfar=0.3, tight=0.20):
    CON=[("goalN","X",0.0,1.0),("goalF","X",L,wfar),("touchN","Y",0.0,1.0),("touchF","Y",Wp,wfar)]
    h0=[K1P,K2P,*np.asarray(base['H']).ravel()[:8].tolist()]   # fisheye PRIOR'dan başla (base'in bozuğundan değil)
    def resid(pp):
        k1,k2=pp[0],pp[1];H=np.array([[pp[2],pp[3],pp[4]],[pp[5],pp[6],pp[7]],[pp[8],pp[9],1.0]]);out=[]
        for r,kind,tg,wt in CON:
            if groups.get(r) is None or len(groups[r])==0: continue
            m=AC.aH(AC.und_norm(groups[r],k1,k2,cx,cy,s),H); out.append(wt*((m[:,0]-tg) if kind=="X" else (m[:,1]-tg)))
        if cen is not None:
            m=AC.aH(AC.und_norm(cen,k1,k2,cx,cy,s),H); out.append(2.0*(m[:,0]-L/2))
        if cir is not None:
            m=AC.aH(AC.und_norm(cir,k1,k2,cx,cy,s),H); out.append(np.array([3.0*(m[:,0].mean()-L/2),3.0*(m[:,1].mean()-Wp/2)]))
        if not out: return np.zeros(1)
        return np.concatenate(out)
    eps=1e-4; lo=[max(0.0,K1P*(1-tight)),max(0.0,K2P*(1-tight))]+[-np.inf]*8; hi=[K1P*(1+tight)+eps,K2P*(1+tight)+eps]+[np.inf]*8
    try: sol=least_squares(resid,h0,method="trf",x_scale="jac",bounds=(lo,hi),max_nfev=8000)
    except Exception: return None
    return dict(k1=float(sol.x[0]),k2=float(sol.x[1]),
                H=np.array([[sol.x[2],sol.x[3],sol.x[4]],[sol.x[5],sol.x[6],sol.x[7]],[sol.x[8],sol.x[9],1.0]]),
                camside=base.get('camside','SOL'),ok=True,converged=bool(sol.success))

def _roundness(rec, cir_pts, cx, cy, s):
    """çember piksel -> metrik -> elips eksen-oranı (1=yuvarlak/oturmuş, <0.7=shear). GT'siz skew ölçüsü."""
    if cir_pts is None or len(cir_pts)<12: return None
    mp=AC.aH(AC.und_norm(cir_pts,rec['k1'],rec['k2'],cx,cy,s),rec['H'])
    mp=mp[np.isfinite(mp).all(1)]
    if len(mp)<12: return None
    c=mp.mean(0); X=mp-c; ev=np.clip(np.linalg.eigvalsh((X.T@X)/len(X)),1e-9,None); R=float(np.sqrt(ev.max()))
    if not (0.4<R<7): return None
    return float(np.sqrt(ev.min()/ev.max()))

def solve_circ(groups, cir, cx, cy, s, base, wcirc=1.0, wcen=2.0):
    """ÇEMBER-OBJEKTİF skew fix: çember-yuvarlaklığını (her piksel merkeze eşit-R) DOĞRUDAN objektife koy.
    Fisheye'ı DONDURMAZ (sadece sane [0.05,0.40] sınırlar) -> doğru fisheye'ı ÇEMBER söyler, ben dayatmam.
    rigid-freeze'in bowtie'sini üretmez; skew'liyi geçerli-kalarak iyileştirir."""
    CON=[("goalN","X",0.0),("goalF","X",L),("touchN","Y",0.0),("touchF","Y",Wp)]
    h0=[float(np.clip(base['k1'],0.06,0.38)),float(np.clip(base['k2'],0.06,0.55)),*np.asarray(base['H']).ravel()[:8].tolist(),2.5]
    def resid(pp):
        k1,k2=pp[0],pp[1];H=np.array([[pp[2],pp[3],pp[4]],[pp[5],pp[6],pp[7]],[pp[8],pp[9],1.0]]);R=pp[10];out=[]
        for r,kind,tg in CON:
            if groups.get(r) is None or len(groups[r])==0: continue
            m=AC.aH(AC.und_norm(groups[r],k1,k2,cx,cy,s),H); out.append((m[:,0]-tg) if kind=="X" else (m[:,1]-tg))
        mc=AC.aH(AC.und_norm(cir,k1,k2,cx,cy,s),H); c=mc.mean(0)
        d=np.hypot(mc[:,0]-c[0],mc[:,1]-c[1])
        out.append(wcirc*(d-abs(R)))                              # çember RADYAL = yuvarlaklık objektifi
        out.append(wcen*np.array([c[0]-L/2,c[1]-Wp/2]))
        if not out: return np.zeros(1)
        return np.concatenate(out)
    lo=[0.05,0.05]+[-np.inf]*8+[1.0]; hi=[0.40,0.55]+[np.inf]*8+[5.0]
    try: sol=least_squares(resid,h0,method="trf",x_scale="jac",bounds=(lo,hi),max_nfev=9000)
    except Exception: return None
    return dict(k1=float(sol.x[0]),k2=float(sol.x[1]),
                H=np.array([[sol.x[2],sol.x[3],sol.x[4]],[sol.x[5],sol.x[6],sol.x[7]],[sol.x[8],sol.x[9],1.0]]),
                camside=base.get('camside','SOL'),ok=True,converged=bool(sol.success),R=float(sol.x[10]))

def refine_rigid(rec, prob, w, h, img=None, sanity_fn=None):
    """rec -> çember-objektif keep-best (skew fix). Yuvarlaklık İYİLEŞİRSE VE sanity tutarsa al, yoksa base.
    NOT: rigid-freeze denendi->bowtie üretti; solve_circ (çemberi objektife koy) geçerli-kalarak iyileştirir."""
    import os as _os
    if _os.environ.get('NO_CIRC'): return rec        # A/B için: circ-refine'i atla (base ölç)
    if not rec or not rec.get('ok'): return rec
    cir=_cc(prob,6,w,h)
    if cir is None or len(cir)<40: return rec           # yuvarlaklık ölçülemez -> dokunma (çembersiz saha)
    groups=rec.get('groups')
    if groups is None:
        from calib_train.auto_calib import groups_from_prob
        groups,_=groups_from_prob(prob,w,h)
        if groups is None: return rec
    cx,cy,s=w/2,h/2,w/2
    rb=_roundness(rec,cir,cx,cy,s)
    if rb is None: return rec
    rig=solve_circ(groups,cir,cx,cy,s,rec)
    if rig is None or not rig.get('converged',True): return rec
    rr=_roundness(rig,cir,cx,cy,s)
    if rr is None or rr <= rb+0.03: return rec           # yuvarlaklık iyileşmezse base tut
    if sanity_fn is not None:
        ok,_=sanity_fn(rig,w,h,img=img)
        if not ok: return rec
    rig['src']=rec.get('src','base')+'+circ'; rig['groups']=groups; rig['marks']=rec.get('marks')
    rig['roundness']=(round(rb,3),round(rr,3))
    return rig
