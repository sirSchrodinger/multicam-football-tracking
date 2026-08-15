#!/usr/bin/env python3
"""YÖNTEM: conf_weighted — çizgi fit'inde her pikseli seg-olasılık DEĞERİYLE ağırlıkla.
Baseline largest-component + Huber fitLine yerine:
  - groups_from_prob_w: largest-component AMA her noktanın prob-değerini (ağırlık) sakla.
    Eşik daha düşük (0.35) tutulur -> true-line low-conf kuyruğunu yakala, fence noise prob-ağırlık ile bastırılır.
  - fitL_w: ağırlıklı total-least-squares (SVD/eig) çizgi fit (cv2.fitLine ağırlık desteklemiyor).
  - solve_calib_w: hem ilk-H köşeleri hem least_squares residual'ı sqrt(w) ile ağırlıklandır.
ORİJİNALİ BOZMAZ: auto_calib import edilir, sadece fit mantığı sarmalanır.
"""
import os, numpy as np, cv2
from scipy.optimize import least_squares
import sys
REPO = os.environ.get("HALISAHA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from calib_train import auto_calib as AC
from calib_train.auto_calib import und_pix, und_norm, aH, inter, K1S, K2S, L

def fitL_w(P, w):
    """ağırlıklı TLS çizgi fit -> [a,b,c] (a*x+b*y+c=0, (a,b)=normal). fitL ile aynı konvansiyon."""
    P=np.asarray(P,float); w=np.asarray(w,float)
    ws=w.sum()
    if ws<=1e-9: return AC.fitL(P)
    mean=(P*w[:,None]).sum(0)/ws
    d=P-mean
    C=(d.T*w)@d/ws                      # ağırlıklı kovaryans 2x2
    ev,evec=np.linalg.eigh(C)           # küçük özdeğer -> normal
    n=evec[:,0]; a,b=n[0],n[1]
    return np.array([a,b,-(a*mean[0]+b*mean[1])])

def groups_from_prob_w(prob,w,h,thr=0.35,minpts=25,maxpts=220,seed=0):
    """largest-component + nokta-başı prob ağırlığı döndür. {role:(pts,wt)} | (None,reason)."""
    rs=np.random.RandomState(seed); groups={}
    for ci,role in enumerate(["goalN","goalF","touchN","touchF"]):
        pc=cv2.resize(prob[ci],(w,h)); mk=(pc>thr).astype(np.uint8)
        mk=cv2.morphologyEx(mk,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
        ncc,lbl,st,_=cv2.connectedComponentsWithStats(mk)
        if ncc>1: mk=(lbl==(1+np.argmax(st[1:,cv2.CC_STAT_AREA]))).astype(np.uint8)
        ys,xs=np.where(mk>0)
        if len(xs)<minpts: return None,f"{role} bulunamadi({len(xs)})"
        pts=np.column_stack([xs,ys]).astype(float)
        wt=pc[ys,xs].astype(float)              # AĞIRLIK = seg-olasılık
        if len(pts)>maxpts:
            sel=rs.choice(len(pts),maxpts,replace=False); pts=pts[sel]; wt=wt[sel]
        groups[role]=(pts,wt)
    return groups,None

def solve_calib_w(groups,cx,cy,s,Wp=18.0):
    CON=[("goalN","X",0.0),("goalF","X",L),("touchN","Y",0.0),("touchF","Y",Wp)]
    pts={r:groups[r][0] for r in groups}; wts={r:groups[r][1] for r in groups}
    def cor(k1,k2):
        fl={r:fitL_w(und_pix(pts[r],k1,k2,cx,cy,s),wts[r]) for r in groups}
        return np.array([inter(fl["goalN"],fl["touchN"]),inter(fl["goalN"],fl["touchF"]),
                         inter(fl["goalF"],fl["touchN"]),inter(fl["goalF"],fl["touchF"])])
    c0=cor(K1S,K2S); H0,_=cv2.findHomography(c0,np.array([[0,0],[0,Wp],[L,0],[L,Wp]],float))
    if H0 is None: return None
    H0=H0/H0[2,2]; p0=[K1S,K2S,*H0.ravel()[:8]]
    def resid(pp):
        k1,k2=pp[0],pp[1];H=np.array([[pp[2],pp[3],pp[4]],[pp[5],pp[6],pp[7]],[pp[8],pp[9],1.0]]);out=[]
        for r,kind,tg in CON:
            m=aH(und_norm(pts[r],k1,k2,cx,cy,s),H)
            res=(m[:,0]-tg) if kind=="X" else (m[:,1]-tg)
            out.append(res*np.sqrt(wts[r]))             # AĞIRLIKLI residual
        return np.concatenate(out)
    sol=least_squares(resid,p0,method="trf",
                      bounds=([0,0]+[-np.inf]*8,[0.45,0.7]+[np.inf]*8),max_nfev=4000)
    k1,k2=sol.x[0],sol.x[1];H=np.array([[sol.x[2],sol.x[3],sol.x[4]],[sol.x[5],sol.x[6],sol.x[7]],[sol.x[8],sol.x[9],1.0]])
    # fit metriği: baseline ile karşılaştırılabilir olsun diye AĞIRLIKSIZ RMS raporla
    def resid_uw(pp):
        k1,k2=pp[0],pp[1];H=np.array([[pp[2],pp[3],pp[4]],[pp[5],pp[6],pp[7]],[pp[8],pp[9],1.0]]);out=[]
        for r,kind,tg in CON:
            m=aH(und_norm(pts[r],k1,k2,cx,cy,s),H); out.append((m[:,0]-tg) if kind=="X" else (m[:,1]-tg))
        return np.concatenate(out)
    fit=float(np.sqrt((resid_uw(sol.x)**2).mean()))
    def u2pix(P): un=und_norm(P,k1,k2,cx,cy,s); return np.stack([cx+un[:,0]/1.5*s,cy+un[:,1]/1.5*s],1)
    fl={r:fitL_w(u2pix(pts[r]),wts[r]) for r in groups}
    c_nN=inter(fl["goalN"],fl["touchN"]);c_nF=inter(fl["goalN"],fl["touchF"]);c_fN=inter(fl["goalF"],fl["touchN"])
    A=c_fN-c_nN;B=c_nF-c_nN;cross=A[0]*B[1]-A[1]*B[0]
    # groups'u baseline-uyumlu (sadece pts) yap ki sanity/clean_label_masks sorunsuz çalışsın
    return dict(k1=k1,k2=k2,H=H,fit=fit,camside="SAG" if cross>0 else "SOL",groups=pts)

def calib_from_pred_w(prob,w,h,thr=0.35,minpts=25):
    cx,cy,s=w/2,h/2,w/2
    groups,reason=groups_from_prob_w(prob,w,h,thr=thr,minpts=minpts)
    if groups is None: return dict(ok=False,reason=reason,fit=None)
    r=solve_calib_w(groups,cx,cy,s)
    if r is None: return dict(ok=False,reason="H tekil",fit=None)
    r.update(ok=True); return r
