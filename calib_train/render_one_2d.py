#!/usr/bin/env python3
"""Yuvarlağı gerçek bir sahayı, yuvarlak-kısıtlı (doğru aspect) kalibrasyonla TEMİZ 2D
top-down'a çevir. Hem düz-warp hem üstüne saha modeli (kenar+orta+yuvarlak) çizilir.
Kullanım: python render_one_2d.py [n1 n2 ...]   (varsayılan 5 6)
"""
import sys, json, os, numpy as np, cv2
from scipy.optimize import least_squares
HERE=os.path.dirname(os.path.abspath(__file__)); LF=os.path.join(HERE,"label_frames")
J=json.load(open(os.path.join(HERE,"calib_lines.json"))); ORDER=json.load(open(os.path.join(HERE,"frames.json")))
Csol=json.load(open(os.path.join(HERE,"calib_solved.json"))); NO_CENTER={3,7,9}; L=34.0; S=34
def idx_of(k): return ORDER.index(k)+1
def name_of(n): return ORDER[n-1]
def und(pts,k1,k2,cx,cy,s): u=(pts[:,0]-cx)/s;vv=(pts[:,1]-cy)/s;r2=u*u+vv*vv;f=1+k1*r2+k2*r2*r2;return np.stack([u*f,vv*f],1)
def aH(uv,H): z=H[2,0]*uv[:,0]+H[2,1]*uv[:,1]+H[2,2];return np.stack([(H[0,0]*uv[:,0]+H[0,1]*uv[:,1]+H[0,2])/z,(H[1,0]*uv[:,0]+H[1,1]*uv[:,1]+H[1,2])/z],1)
def fitL(P):c=P.mean(0);_,_,vt=np.linalg.svd(P-c);nv=np.array([-vt[0,1],vt[0,0]]);return np.array([nv[0],nv[1],-nv.dot(c)])
def inter(a,b):D=a[0]*b[1]-b[0]*a[1];return np.array([(a[1]*b[2]-b[1]*a[2])/D,(b[0]*a[2]-a[0]*b[2])/D])

def solve(name):
    v=J[name]; r0=Csol[name]; img=cv2.imread(os.path.join(LF,name)); h,w=img.shape[:2]; cx,cy,s=w/2,h/2,w/2; n=idx_of(name); ln=v["lines"]
    lin=[]
    for lid,kind in [("goalN",("X",0.0)),("goalF",("X",L)),("touchN",("Y",0.0)),("touchF",("Yw",None)),("center",("X",L/2))]:
        if lid=="center" and n in NO_CENTER: continue
        p=ln.get(lid)
        if p and len(p)>=2: lin.append((np.asarray(p,float),kind))
    circ=np.asarray(ln["circle"],float) if ln.get("circle") and len(ln["circle"])>=5 else None
    p0=[r0["k1"],r0["k2"],*np.array(r0["H"]).ravel()[:8],18.0,3.0]
    def resid(p):
        k1,k2=p[0],p[1]; H=np.array([[p[2],p[3],p[4]],[p[5],p[6],p[7]],[p[8],p[9],1.0]]); Wp,R=p[10],p[11]; out=[]
        for pts,(kind,tg) in lin:
            m=aH(und(pts,k1,k2,cx,cy,s),H)
            out.append(m[:,0]-tg if kind=="X" else (m[:,1]-tg if kind=="Y" else m[:,1]-Wp))
        if circ is not None:
            mc=aH(und(circ,k1,k2,cx,cy,s),H); out.append(np.sqrt((mc[:,0]-L/2)**2+(mc[:,1]-Wp/2)**2)-R)
        return np.concatenate(out)
    sol=least_squares(resid,p0,method="lm",max_nfev=30000)
    k1,k2=sol.x[0],sol.x[1]; H=np.array([[sol.x[2],sol.x[3],sol.x[4]],[sol.x[5],sol.x[6],sol.x[7]],[sol.x[8],sol.x[9],1.0]]); Wp,R=sol.x[10],sol.x[11]
    return img,h,w,cx,cy,s,k1,k2,H,Wp,R,n,name

def render(name):
    img,h,w,cx,cy,s,k1,k2,H,Wp,R,n,nm=solve(name)
    M=4.0   # saha dışı pay (m) — oyuncu çıkınca komple kaybolmasın
    LS,WS=int((L+2*M)*S),int((Wp+2*M)*S)
    # undistort image
    rd=np.linspace(0,2.4,2400); rru=rd*(1+k1*rd*rd+k2*rd**4); zoom=1.5
    ys,xs=np.mgrid[0:h,0:w].astype(np.float32); uo=(xs-cx)/s*zoom; vo=(ys-cy)/s*zoom; ro=np.sqrt(uo*uo+vo*vo)
    rdv=np.interp(ro,rru,rd); sc=np.divide(rdv,ro,out=np.ones_like(ro),where=ro>1e-6)
    und_img=cv2.remap(img,(cx+uo*sc*s).astype(np.float32),(cy+vo*sc*s).astype(np.float32),cv2.INTER_LINEAR)
    def u2p(pts): un=und(np.asarray(pts,float),k1,k2,cx,cy,s); return np.stack([cx+un[:,0]/zoom*s,cy+un[:,1]/zoom*s],1)
    ln=J[nm]["lines"]; fl={l:fitL(u2p(ln[l])) for l in ("goalN","goalF","touchN","touchF")}
    src=np.array([inter(fl["goalN"],fl["touchN"]),inter(fl["goalN"],fl["touchF"]),inter(fl["goalF"],fl["touchN"]),inter(fl["goalF"],fl["touchF"])],float)
    # model overlay (Y=0 altta, etrafta M marj -> ekran)
    def MX(X,Y): return (int((M+X)*S), int(WS-(M+Y)*S))
    dst=np.array([MX(0,0),MX(0,Wp),MX(L,0),MX(L,Wp)],float)   # nN,nF,fN,fF
    Hm,_=cv2.findHomography(src,dst); top=cv2.warpPerspective(und_img,Hm,(LS,WS))
    cv2.rectangle(top,MX(0,Wp),MX(L,0),(0,215,255),2)   # saha sınırı (marj içeride)
    cv2.line(top,MX(L/2,0),MX(L/2,Wp),(0,230,230),2)
    cv2.circle(top,MX(L/2,Wp/2),int(R*S),(255,160,0),2)
    cv2.putText(top,f"{n}. {nm[:16]}  L=34 W={Wp:.1f} (W/L={Wp/L:.2f})  yuvarlak R={R:.1f}",(8,26),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    return top

names=[name_of(int(x)) for x in (sys.argv[1:] or ["5","6"])]
tiles=[render(nm) for nm in names]
Wmax=max(t.shape[1] for t in tiles); tot=sum(t.shape[0]+8 for t in tiles)
sheet=np.full((tot,Wmax,3),18,np.uint8); y=0
for t in tiles: sheet[y:y+t.shape[0],0:t.shape[1]]=t; y+=t.shape[0]+8
out=os.path.join(HERE,"cand","_one_2d.jpg"); cv2.imwrite(out,sheet); print("->",out, sheet.shape)
