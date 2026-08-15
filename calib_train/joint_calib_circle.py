#!/usr/bin/env python3
"""Orta yuvarlağı GERÇEK olan sahalarda en-boy oranını(W) + yuvarlak yarıçapını(R) GERİ ÇÖZ.
4 kenar aspect'i belirleyemez -> daire elips çıkar. 'Daire yuvarlak olsun' kısıtı W,R verir.
L=34 sabit birim (mutlak metre bir bilinen ölçü ister); önemli olan W/L aspect + R düzelmesi.
Çıktı: cand/_circle_fix.jpg (yuvarlağı gerçek sahalar, düzeltilmiş daire ile reproject)
"""
import json, os, numpy as np, cv2
from scipy.optimize import least_squares
HERE=os.path.dirname(os.path.abspath(__file__)); LF=os.path.join(HERE,"label_frames")
J=json.load(open(os.path.join(HERE,"calib_lines.json"))); ORDER=json.load(open(os.path.join(HERE,"frames.json")))
C=json.load(open(os.path.join(HERE,"calib_solved.json"))); NO_CENTER={3,7,9}; L=34.0
CIRCLE_REAL=[5,6,8,14]   # kullanıcı: orta yuvarlağı GERÇEK olanlar
def idx_of(k): return ORDER.index(k)+1 if k in ORDER else -1
def und(pts,k1,k2,cx,cy,s): u=(pts[:,0]-cx)/s;vv=(pts[:,1]-cy)/s;r2=u*u+vv*vv;f=1+k1*r2+k2*r2*r2;return np.stack([u*f,vv*f],1)
def aH(uv,H): z=H[2,0]*uv[:,0]+H[2,1]*uv[:,1]+H[2,2];return np.stack([(H[0,0]*uv[:,0]+H[0,1]*uv[:,1]+H[0,2])/z,(H[1,0]*uv[:,0]+H[1,1]*uv[:,1]+H[1,2])/z],1)

def solve_with_circle(name):
    v=J[name]; r0=C[name]; img=cv2.imread(os.path.join(LF,name)); h,w=r0["h"],r0["w"]; cx,cy,s=w/2,h/2,w/2; n=r0["n"]; ln=v["lines"]
    lin=[]   # (pts, axis, target_is_W?  -> ('X',0)/('X',L)/('Y',0)/('Yw') )
    for lid,kind in [("goalN",("X",0.0)),("goalF",("X",L)),("touchN",("Y",0.0)),("touchF",("Yw",None)),("center",("X",L/2))]:
        if lid=="center" and n in NO_CENTER: continue
        p=ln.get(lid)
        if p and len(p)>=2: lin.append((np.asarray(p,float),kind))
    circ=np.asarray(ln["circle"],float)
    p0=[r0["k1"],r0["k2"],*np.array(r0["H"]).ravel()[:8],18.0,3.0]   # +W,R
    def resid(p):
        k1,k2=p[0],p[1]; H=np.array([[p[2],p[3],p[4]],[p[5],p[6],p[7]],[p[8],p[9],1.0]]); Wp,R=p[10],p[11]
        out=[]
        for pts,(kind,tg) in lin:
            m=aH(und(pts,k1,k2,cx,cy,s),H)
            if kind=="X": out.append(m[:,0]-tg)
            elif kind=="Y": out.append(m[:,1]-tg)
            else: out.append(m[:,1]-Wp)        # touchF -> Y=W
        mc=aH(und(circ,k1,k2,cx,cy,s),H)
        d=np.sqrt((mc[:,0]-L/2)**2+(mc[:,1]-Wp/2)**2)-R
        out.append(d*1.0)
        return np.concatenate(out)
    sol=least_squares(resid,p0,method="lm",max_nfev=30000)
    fit=np.sqrt((resid(sol.x)**2).mean())
    return dict(name=name,n=n,w=w,h=h,k1=sol.x[0],k2=sol.x[1],
                H=np.array([[sol.x[2],sol.x[3],sol.x[4]],[sol.x[5],sol.x[6],sol.x[7]],[sol.x[8],sol.x[9],1.0]]).tolist(),
                Wp=sol.x[10],R=sol.x[11],fit=fit)

def reproject(rec):
    name=rec["name"]; img=cv2.imread(os.path.join(LF,name)); h,w=rec["h"],rec["w"]; cx,cy,s=w/2,h/2,w/2
    k1,k2=rec["k1"],rec["k2"]; H=np.array(rec["H"]); Wp,R=rec["Wp"],rec["R"]; Hinv=np.linalg.inv(H)
    rd=np.linspace(0,2.4,2400); rru=rd*(1+k1*rd*rd+k2*rd**4)
    def m2i(M):
        m=np.concatenate([M,np.ones((len(M),1))],1).T; un=Hinv@m; u=un[0]/un[2]; vv=un[1]/un[2]; ru=np.sqrt(u*u+vv*vv)+1e-9
        rr=np.interp(ru,rru,rd); sc=rr/ru; return np.stack([cx+u*sc*s,cy+vv*sc*s],1)
    vis=img.copy(); t=np.linspace(0,1,80)[:,None]; Lh=lambda p,q:p*(1-t)+q*t
    segs=[(Lh(np.array([0,0]),np.array([L,0])),(0,80,255)),(Lh(np.array([0,Wp]),np.array([L,Wp])),(0,80,255)),
          (Lh(np.array([0,0]),np.array([0,Wp])),(60,220,60)),(Lh(np.array([L,0]),np.array([L,Wp])),(60,220,60)),
          (Lh(np.array([L/2,0]),np.array([L/2,Wp])),(0,230,230))]
    thc=np.linspace(0,2*np.pi,140)[:,None]; circ=np.concatenate([L/2+R*np.cos(thc),Wp/2+R*np.sin(thc)],1)
    segs.append((circ,(255,160,0)))
    for M,col in segs:
        P=m2i(M); P=P[(P[:,0]>-60)&(P[:,0]<w+60)&(P[:,1]>-60)&(P[:,1]<h+60)]
        if len(P)>1: cv2.polylines(vis,[P.astype(np.int32)],False,col,3,cv2.LINE_AA)
    sc=900/w; vis=cv2.resize(vis,(int(w*sc),int(h*sc)))
    cv2.putText(vis,f"{rec['n']}. {name[:15]} W/L={Wp/L:.2f} R={R:.1f}(L=34) res{rec['fit']:.2f}",(6,24),cv2.FONT_HERSHEY_SIMPLEX,0.62,(0,230,255),2)
    return vis

recs=[solve_with_circle(k) for k in ORDER if k in C and idx_of(k) in CIRCLE_REAL]
print("=== yuvarlak-kısıtlı çözüm (W/L=aspect, R=yarıçap L=34 biriminde) ===")
for r in sorted(recs,key=lambda r:r['n']):
    print(f"{r['n']:2d} {r['name'][:18]:18s} W={r['Wp']:.1f} (W/L={r['Wp']/L:.2f})  R={r['R']:.2f}  res={r['fit']:.3f}")
ths=[reproject(r) for r in sorted(recs,key=lambda r:r['n'])]
Wt=ths[0].shape[1]; H0=ths[0].shape[0]; cols=2; rows=(len(ths)+1)//2
sheet=np.full((rows*(H0+6),cols*(Wt+6),3),20,np.uint8)
for i,t in enumerate(ths):
    rr,cc=divmod(i,cols); sheet[rr*(H0+6):rr*(H0+6)+t.shape[0],cc*(Wt+6):cc*(Wt+6)+t.shape[1]]=t
cv2.imwrite(os.path.join(HERE,"cand","_circle_fix.jpg"),sheet); print("-> cand/_circle_fix.jpg")
