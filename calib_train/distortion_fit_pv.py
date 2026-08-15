#!/usr/bin/env python3
"""SAHA-BAŞINA plumb-line fisheye: her sahanın kendi (k1,k2)'si, merkez=görüntü ortası.
Undistort -> 4 köşe -> homografi -> temiz 2D. Per-venue k'lar kümeleniyorsa otomasyon
için ORTALAMA ortak-model alınır (sonraki adım). Çıktı: cand/_2d_undist_pv.jpg + rapor.
"""
import json, os
import numpy as np, cv2
from scipy.optimize import least_squares

HERE=os.path.dirname(os.path.abspath(__file__)); LF=os.path.join(HERE,"label_frames")
J=json.load(open(os.path.join(HERE,"calib_lines.json"))); ORDER=json.load(open(os.path.join(HERE,"frames.json")))
L,W,S=34.0,18.0,26; LS,WS=int(L*S),int(W*S); NO_CENTER={3,7,9}
STRAIGHT=["goalN","goalF","touchN","touchF","center","boxN"]
def idx_of(k): return ORDER.index(k)+1 if k in ORDER else -1

venues=[]
for k,v in J.items():
    ln=v.get("lines",{})
    if all(ln.get(x) and len(ln[x])>=2 for x in ("goalN","touchN","touchF")):
        img=cv2.imread(os.path.join(LF,k))
        if img is not None: venues.append((k,v,img.shape[1],img.shape[0]))

def undist_norm(pts,k1,k2,cx,cy,s):
    u=(pts[:,0]-cx)/s; vv=(pts[:,1]-cy)/s; r2=u*u+vv*vv; f=1+k1*r2+k2*r2*r2
    return np.stack([u*f,vv*f],1)
def line_resid(P):
    c=P.mean(0); _,_,vt=np.linalg.svd(P-c); nrm=np.array([-vt[0,1],vt[0,0]]); return (P-c)@nrm
def fit_venue(v,w,h,n):
    cx,cy,s=w/2,h/2,w/2; gs=[]
    for lid in STRAIGHT:
        if lid=="center" and n in NO_CENTER: continue
        pts=v["lines"].get(lid)
        if pts and len(pts)>=3: gs.append(np.asarray(pts,float))
    def res(p):
        return np.concatenate([line_resid(undist_norm(g,p[0],p[1],cx,cy,s)) for g in gs])
    base=np.sqrt((res([0,0])**2).mean())
    sol=least_squares(res,[0,0],method="trf",bounds=([-1.2,-1.2],[1.2,1.2]),max_nfev=3000)
    fit=np.sqrt((res(sol.x)**2).mean())
    return sol.x,base,fit,len(gs)

def fitL(P): c=P.mean(0); _,_,vt=np.linalg.svd(P-c); nv=np.array([-vt[0,1],vt[0,0]]); return np.array([nv[0],nv[1],-nv.dot(c)])
def inter(a,b):
    a1,b1,c1=a;a2,b2,c2=b;D=a1*b2-a2*b1
    return None if abs(D)<1e-9 else np.array([(b1*c2-b2*c1)/D,(a2*c1-a1*c2)/D])

thumbs=[]; rep=[]
for k,v,w,h in venues:
    n=idx_of(k); (k1,k2),base,fit,ng=fit_venue(v,w,h,n); cx,cy,s=w/2,h/2,w/2
    rep.append((n,k,k1,k2,base,fit,ng))
    img=cv2.imread(os.path.join(LF,k))
    rd=np.linspace(0,1.8,1500); ru=rd*(1+k1*rd*rd+k2*rd**4)
    mono = np.all(np.diff(ru)>0)
    zoom=1.45; ys,xs=np.mgrid[0:h,0:w].astype(np.float32)
    uo=(xs-cx)/s*zoom; vo=(ys-cy)/s*zoom; ro=np.sqrt(uo*uo+vo*vo)
    rdv=np.interp(ro,ru,rd); sc=np.divide(rdv,ro,out=np.ones_like(ro),where=ro>1e-6)
    und=cv2.remap(img,(cx+uo*sc*s).astype(np.float32),(cy+vo*sc*s).astype(np.float32),cv2.INTER_LINEAR)
    def u2p(pts): un=undist_norm(np.asarray(pts,float),k1,k2,cx,cy,s); return np.stack([cx+un[:,0]/zoom*s,cy+un[:,1]/zoom*s],1)
    def cor(a,b): return inter(fitL(u2p(v["lines"][a])),fitL(u2p(v["lines"][b])))
    th=np.full((WS+30,LS,3),40,np.uint8)
    try:
        src=np.array([cor("goalN","touchN"),cor("goalN","touchF"),cor("goalF","touchN"),cor("goalF","touchF")],float)
        Hm,_=cv2.findHomography(src,np.array([[0,WS],[0,0],[LS,WS],[LS,0]],float))
        top=cv2.warpPerspective(und,Hm,(LS,WS))
        cv2.rectangle(top,(0,0),(LS-1,WS-1),(0,215,255),2);cv2.line(top,(LS//2,0),(LS//2,WS),(0,215,255),1)
        cv2.circle(top,(LS//2,WS//2),int(3*S),(0,215,255),1); th[30:,:]=top
    except Exception: cv2.putText(th,"HATA",(20,WS//2),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
    cv2.putText(th,f"{n}. {k[:15]} k1{k1:+.2f} k2{k2:+.2f}",(4,20),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,230,255),2)
    thumbs.append((n,th))
thumbs.sort(key=lambda t:t[0]); cols=3; rows=(len(thumbs)+cols-1)//cols
sheet=np.full((rows*(WS+38),cols*(LS+8),3),20,np.uint8)
for i,(n,th) in enumerate(thumbs):
    r,c=divmod(i,cols); sheet[r*(WS+38):r*(WS+38)+WS+30,c*(LS+8):c*(LS+8)+LS]=th
cv2.imwrite(os.path.join(HERE,"cand","_2d_undist_pv.jpg"),sheet)
print("=== saha-başına k (residual norm ×1e-3; düşük=düz) ===")
for n,k,k1,k2,b,f,ng in sorted(rep):
    print(f"{n:2d} {k[:18]:18s} k1={k1:+.3f} k2={k2:+.3f}  res {b*1000:5.1f}->{f*1000:5.1f}  ({ng} cizgi)")
ks=np.array([[r[2],r[3]] for r in rep]); print(f"\nk1 medyan={np.median(ks[:,0]):+.3f} std={ks[:,0].std():.3f} | k2 medyan={np.median(ks[:,1]):+.3f} std={ks[:,1].std():.3f}")
print("montaj -> cand/_2d_undist_pv.jpg")
