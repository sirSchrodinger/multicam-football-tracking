#!/usr/bin/env python3
"""Alperen teşhisi: çizgi-tespiti iyi ama warp yamuk çünkü outlier pikseller (pembe'ye karışmış mavi,
kesik/gürültü) least-squares ortalamayı çekiyor. FIX: robust loss (soft_l1/huber) + IRLS-inlier —
uç değerleri otomatik düşük-ağırlıklar. base(linear) vs robust warp karşılaştır."""
import os, sys, json, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
HERE=os.path.dirname(os.path.abspath(__file__)); CT=os.path.dirname(HERE); ROOT=os.path.dirname(CT)
sys.path.insert(0, ROOT)
from calib_train import auto_calib as AC
from scipy.optimize import least_squares
L,Wp=AC.L,18.0

def solve_robust(groups,cx,cy,s,base,fscale=0.6,loss='soft_l1',cen=None,cir=None,wc=2.0,wr=3.0):
    CON=[("goalN","X",0.0),("goalF","X",L),("touchN","Y",0.0),("touchF","Y",Wp)]
    h0=[float(base['k1']),float(base['k2']),*np.asarray(base['H']).ravel()[:8].tolist()]
    def resid(pp):
        k1,k2=pp[0],pp[1];H=np.array([[pp[2],pp[3],pp[4]],[pp[5],pp[6],pp[7]],[pp[8],pp[9],1.0]]);out=[]
        for r,kind,tg in CON:
            m=AC.aH(AC.und_norm(groups[r],k1,k2,cx,cy,s),H); out.append((m[:,0]-tg) if kind=="X" else (m[:,1]-tg))
        if cen is not None:
            m=AC.aH(AC.und_norm(cen,k1,k2,cx,cy,s),H); out.append(wc*(m[:,0]-L/2))
        if cir is not None:
            m=AC.aH(AC.und_norm(cir,k1,k2,cx,cy,s),H); out.append(np.array([wr*(m[:,0].mean()-L/2),wr*(m[:,1].mean()-Wp/2)]))
        return np.concatenate(out)
    try:
        sol=least_squares(resid,h0,method="trf",loss=loss,f_scale=fscale,x_scale="jac",
                          bounds=([0,0]+[-np.inf]*8,[0.45,0.7]+[np.inf]*8),max_nfev=8000)
    except Exception as e:
        print('  robust solve hata',e); return None
    k1,k2=sol.x[0],sol.x[1];H=np.array([[sol.x[2],sol.x[3],sol.x[4]],[sol.x[5],sol.x[6],sol.x[7]],[sol.x[8],sol.x[9],1.0]])
    r=resid(sol.x); med=float(np.median(np.abs(r)))
    return dict(k1=k1,k2=k2,H=H,fit=float(np.sqrt((r**2).mean())),fit_med=med,camside=base['camside'],converged=bool(sol.success))

def warp_panel(img,rec,w,h,tag):
    S=26;pad=16; Wc=int(L*S+2*pad);Hc=int(Wp*S+2*pad)
    gx,gy=np.meshgrid(np.arange(Wc),np.arange(Hc)); Xm=(gx-pad)/S;Ym=Wp-(gy-pad)/S
    pix=AC.project_metric(np.stack([Xm.ravel(),Ym.ravel()],1),rec['k1'],rec['k2'],rec['H'],w/2,h/2,w/2)
    wp=cv2.remap(img,pix[:,0].reshape(Hc,Wc).astype(np.float32),pix[:,1].reshape(Hc,Wc).astype(np.float32),cv2.INTER_LINEAR,borderValue=(28,28,28))
    def P2(X,Y): return int(pad+X*S),int(pad+(Wp-Y)*S)
    w0=(235,238,240)
    cv2.rectangle(wp,P2(0,0),P2(L,Wp),w0,2,cv2.LINE_AA);cv2.line(wp,P2(L/2,0),P2(L/2,Wp),w0,1,cv2.LINE_AA);cv2.circle(wp,P2(L/2,Wp/2),int(3*S),w0,1,cv2.LINE_AA)
    cv2.rectangle(wp,(0,0),(wp.shape[1],26),(0,0,0),-1);cv2.putText(wp,tag,(6,19),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1)
    return wp

def main(idx):
    from calib_train.night.four_panel import load
    from calib_train import auto_clean2d as AC2
    from calib_train.center_refine import _cc
    e,img,prob,w,h,feet=load(idx); cx,cy,s=w/2,h/2,w/2
    base=AC2.calibrate_frame(prob,w,h,img=img,feet=feet)
    groups=base.get('groups')
    if groups is None:
        from calib_train.auto_calib import groups_from_prob; groups,_=groups_from_prob(prob,w,h)
    cen=_cc(prob,4,w,h); cir=_cc(prob,6,w,h)
    rob=solve_robust(groups,cx,cy,s,base,cen=cen,cir=cir)
    print(f"idx{idx}: base fit={base['fit']:.3f} src={base.get('src')} | robust fit={rob['fit']:.3f} med={rob['fit_med']:.3f}")
    wb=warp_panel(img,base,w,h,f"MEVCUT (linear+center) fit={base['fit']:.2f}")
    wr=warp_panel(img,rob,w,h,f"ROBUST (soft_l1) fit={rob['fit']:.2f} med={rob['fit_med']:.2f}")
    cv2.imwrite(f"{CT}/cand/_ROBUST_{idx}.jpg",np.hstack([wb,np.full((wb.shape[0],6,3),80,np.uint8),wr]))
    print(f"  -> _ROBUST_{idx}.jpg")

if __name__=="__main__":
    for idx in ([int(x) for x in sys.argv[1:]] if len(sys.argv)>1 else [99,55,34,37]): main(idx)
