#!/usr/bin/env python3
"""Tahmin->kalibrasyon ÇEKİRDEĞİ (yeniden kullanılabilir). seg2 rol-çizgi olasılıkları ->
robust çizgi fit -> ÇİZGİ-kısıtlı joint kalibrasyon (ortak-lens+H, METRİK uzayda).
Dönen residual GT-SİZ kalite ölçer -> self-feeding KABUL-KAPISI + watchdog burayı kullanır.
calib_from_pred(prob,w,h) -> dict(ok,fit,k1,k2,H,groups,camside) | reason.
"""
import os, numpy as np, cv2
from scipy.optimize import least_squares
K1S,K2S=0.167,0.240; L=34.0
def und_pix(P,k1,k2,cx,cy,s,z=1.5): u=(P[:,0]-cx)/s;v=(P[:,1]-cy)/s;r2=u*u+v*v;f=1+k1*r2+k2*r2*r2;return np.stack([cx+u*f/z*s,cy+v*f/z*s],1)
def und_norm(P,k1,k2,cx,cy,s): u=(P[:,0]-cx)/s;v=(P[:,1]-cy)/s;r2=u*u+v*v;f=1+k1*r2+k2*r2*r2;return np.stack([u*f,v*f],1)
def aH(uv,H): z=H[2,0]*uv[:,0]+H[2,1]*uv[:,1]+H[2,2];return np.stack([(H[0,0]*uv[:,0]+H[0,1]*uv[:,1]+H[0,2])/z,(H[1,0]*uv[:,0]+H[1,1]*uv[:,1]+H[1,2])/z],1)
def fitL(P):
    vx,vy,x0,y0=cv2.fitLine(np.asarray(P,np.float32),cv2.DIST_HUBER,0,0.01,0.01).ravel()
    a,b=-vy,vx; return np.array([a,b,-(a*x0+b*y0)])
def inter(a,b):D=a[0]*b[1]-b[0]*a[1];return np.array([(a[1]*b[2]-b[1]*a[2])/D,(b[0]*a[2]-a[0]*b[2])/D])

def groups_from_prob(prob,w,h,thr=0.5,minpts=25,maxpts=180,seed=0):
    """prob (>=4,Hs,Ws) -> {role:pts} 4 sınır rolü (en büyük bileşen). Eksikse (None,reason)."""
    rs=np.random.RandomState(seed); groups={}
    for ci,role in enumerate(["goalN","goalF","touchN","touchF"]):
        pc=cv2.resize(prob[ci],(w,h)); mk=(pc>thr).astype(np.uint8)
        mk=cv2.morphologyEx(mk,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
        ncc,lbl,st,_=cv2.connectedComponentsWithStats(mk)
        if ncc>1: mk=(lbl==(1+np.argmax(st[1:,cv2.CC_STAT_AREA]))).astype(np.uint8)
        pts=np.column_stack(np.where(mk>0))[:,::-1].astype(float)
        if len(pts)<minpts: return None,f"{role} bulunamadi({len(pts)})"
        if len(pts)>maxpts: pts=pts[rs.choice(len(pts),maxpts,replace=False)]
        groups[role]=pts
    return groups,None

def solve_calib(groups,cx,cy,s,Wp=18.0):
    CON=[("goalN","X",0.0),("goalF","X",L),("touchN","Y",0.0),("touchF","Y",Wp)]
    METRIC=np.array([[0,0],[0,Wp],[L,0],[L,Wp]],float)
    def corners_in(frame_fn):
        # 4 sınır-çizgisini verilen çerçevede fit et -> köşe kesişimleri (paralel-guard'lı)
        fl={r:fitL(frame_fn(groups[r])) for r in groups}
        cs=[]
        for a,b in ((fl["goalN"],fl["touchN"]),(fl["goalN"],fl["touchF"]),(fl["goalF"],fl["touchN"]),(fl["goalF"],fl["touchF"])):
            D=a[0]*b[1]-b[0]*a[1]
            if not np.isfinite(D) or abs(D)<1e-12: return None   # near-paralel -> dejenere init
            cs.append([(a[1]*b[2]-b[1]*a[2])/D,(b[0]*a[2]-a[0]*b[2])/D])
        cs=np.array(cs); return cs if np.isfinite(cs).all() else None
    def H0_from(c0):
        if c0 is None: return None
        H0,_=cv2.findHomography(c0,METRIC)
        if H0 is None or abs(H0[2,2])<1e-12: return None
        return (H0/H0[2,2]).ravel()[:8]
    # AUDIT-FIX (D1/U1/U4): init resid'le AYNI çerçevede (und_norm) kurulmalı; legacy und_pix-init
    # multistart yedeği olarak kalır (121 A/B'de 21/108 saha farklı basene düşüyor).
    inits=[h for h in (H0_from(corners_in(lambda P: und_norm(P,K1S,K2S,cx,cy,s))),
                       H0_from(corners_in(lambda P: und_pix(P,K1S,K2S,cx,cy,s)))) if h is not None]
    if not inits: return None
    def resid(pp):
        k1,k2=pp[0],pp[1];H=np.array([[pp[2],pp[3],pp[4]],[pp[5],pp[6],pp[7]],[pp[8],pp[9],1.0]]);out=[]
        for r,kind,tg in CON:
            m=aH(und_norm(groups[r],k1,k2,cx,cy,s),H); out.append((m[:,0]-tg) if kind=="X" else (m[:,1]-tg))
        return np.concatenate(out)
    best=None
    for h0 in inits:
        # AUDIT-FIX (D1 kanıtlı): x_scale='jac' — k(~0.2)/H(~5) ölçek-uyumsuzluğu kondisyon bug'ı;
        # 'jac' ile 8/8 GT temiz-girdi EXACT recovery + 96/96 gürültü-trial sıfır blowup.
        sol=least_squares(resid,[K1S,K2S,*h0],method="trf",x_scale="jac",
                          bounds=([0,0]+[-np.inf]*8,[0.45,0.7]+[np.inf]*8),max_nfev=4000)
        f=float(np.sqrt((resid(sol.x)**2).mean()))
        if best is None or f<best[0]: best=(f,sol)
    fit,sol=best
    k1,k2=sol.x[0],sol.x[1];H=np.array([[sol.x[2],sol.x[3],sol.x[4]],[sol.x[5],sol.x[6],sol.x[7]],[sol.x[8],sol.x[9],1.0]])
    # kamera tarafı (rafine k, roller DOĞRU)
    def u2pix(P): un=und_norm(P,k1,k2,cx,cy,s); return np.stack([cx+un[:,0]/1.5*s,cy+un[:,1]/1.5*s],1)
    fl={r:fitL(u2pix(groups[r])) for r in groups}
    c_nN=inter(fl["goalN"],fl["touchN"]);c_nF=inter(fl["goalN"],fl["touchF"]);c_fN=inter(fl["goalF"],fl["touchN"])
    A=c_fN-c_nN;B=c_nF-c_nN;cross=A[0]*B[1]-A[1]*B[0]
    return dict(k1=k1,k2=k2,H=H,fit=fit,camside="SAG" if cross>0 else "SOL",
                converged=bool(sol.success and sol.nfev<4000))   # AUDIT: nfev-tükenmiş fit'e güvenme

def project_metric(P2,k1,k2,H,cx,cy,s):
    """metrik (X,Y) -> görüntü piksel (ileri model: H^-1 -> undistort-norm -> distort -> piksel)."""
    Hi=np.linalg.inv(H); P=np.c_[np.asarray(P2,float),np.ones(len(P2))]
    q=P@Hi.T; un=q[:,0]/q[:,2]; vn=q[:,1]/q[:,2]   # undistort-norm
    ru=np.sqrt(un*un+vn*vn); rd_g=np.linspace(0,2.6,5000); ru_g=rd_g*(1+k1*rd_g*rd_g+k2*rd_g**4)
    rd=np.interp(ru,ru_g,rd_g); sc=np.divide(rd,ru,out=np.ones_like(ru),where=ru>1e-9)
    return np.stack([cx+un*sc*s, cy+vn*sc*s],1)

def clean_label_masks(rec,w,h,Wp=18.0,thick=None):
    """KABUL EDİLEN kalibrasyondan TEMİZ 7-kanal rol-maskesi (metrik çizgileri geri-projeksiyon).
    Teacher'ın kalın/gürültülü maskesi yerine geometrik-mükemmel pseudo-etiket."""
    k1,k2,H=rec["k1"],rec["k2"],rec["H"]; cx,cy,s=w/2,h/2,w/2; th=thick or max(3,w//240)
    m=np.zeros((7,h,w),np.uint8); mg=0.06   # iç-çizgi sadece görüntü-İÇİNDE projekte oluyorsa çiz
    def drawline(ci,a,b,N=80,closed=False,interior=False):
        pts=np.linspace(a,b,N) if not closed else a; pix=project_metric(pts,k1,k2,H,cx,cy,s)
        fin=np.isfinite(pix).all(1)
        ins=fin&(pix[:,0]>-mg*w)&(pix[:,0]<(1+mg)*w)&(pix[:,1]>-mg*h)&(pix[:,1]<(1+mg)*h)
        # iç çizgiler (box/circle/center): çoğu nokta görüntü-içi DEĞİLSE atla (ışınsal kaçışı önle)
        if interior and (ins.sum()<0.7*fin.sum() or ins.sum()<6): return
        keep=ins if interior else (fin&(pix[:,0]>-w)&(pix[:,0]<2*w)&(pix[:,1]>-h)&(pix[:,1]<2*h))
        if keep.sum()>=2: cv2.polylines(m[ci],[pix[keep].astype(np.int32)],closed,255,th,cv2.LINE_AA)
    drawline(0,[0,0],[0,Wp]); drawline(1,[L,0],[L,Wp])           # goalN goalF (sınır = güvenli)
    drawline(2,[0,0],[L,0]); drawline(3,[0,Wp],[L,Wp])           # touchN touchF
    drawline(4,[L/2,0],[L/2,Wp],interior=True)                   # center (iç -> doğrula)
    for gx in (0,L):                                             # box (iç -> doğrula)
        bx=5 if gx==0 else L-5
        for a,b in [([bx,Wp/2-5],[bx,Wp/2+5]),([gx,Wp/2-5],[bx,Wp/2-5]),([gx,Wp/2+5],[bx,Wp/2+5])]:
            drawline(5,a,b,N=30,interior=True)
    thc=np.linspace(0,2*np.pi,60); circ=np.stack([L/2+3*np.cos(thc),Wp/2+3*np.sin(thc)],1)
    drawline(6,circ,None,closed=True,interior=True)              # circle (iç -> doğrula)
    return m

def calib_from_pred(prob,w,h,thr=0.5,minpts=25):
    cx,cy,s=w/2,h/2,w/2
    groups,reason=groups_from_prob(prob,w,h,thr=thr,minpts=minpts)
    if groups is None: return dict(ok=False,reason=reason,fit=None)
    r=solve_calib(groups,cx,cy,s)
    if r is None: return dict(ok=False,reason="H tekil",fit=None)
    r.update(ok=True,groups=groups); return r
