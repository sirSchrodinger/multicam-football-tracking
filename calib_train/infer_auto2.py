#!/usr/bin/env python3
"""KISITLI otomatik 2D: seg model -> tespit çizgileri -> joint_calib (bozulma+homografi,
kısıt METRİK uzayda) + yuvarlaktan aspect. Elle etiket YOK. v1'deki kenar şişmesini bitirir.
Kullanım: python infer_auto2.py img1 img2 ...  -> cand/_auto2_2d.jpg
"""
import sys, os, numpy as np, cv2, torch, torch.nn as nn
from scipy.optimize import least_squares
HERE=os.path.dirname(os.path.abspath(__file__))
K1S,K2S=0.167,0.240; TW,TH=512,288; L,S,M=34.0,26,4.0
CK=os.path.join(HERE,"seg_ckpt",os.environ.get("SEG_CK","seg_unet_v3.pth"))
def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True),
                                   nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True))
class UNet(nn.Module):
    def __init__(s,nin=3,no=2,b=24):
        super().__init__(); s.d1=cbr(nin,b);s.d2=cbr(b,b*2);s.d3=cbr(b*2,b*4);s.d4=cbr(b*4,b*8);s.p=nn.MaxPool2d(2)
        s.u3=nn.ConvTranspose2d(b*8,b*4,2,2);s.c3=cbr(b*8,b*4);s.u2=nn.ConvTranspose2d(b*4,b*2,2,2);s.c2=cbr(b*4,b*2)
        s.u1=nn.ConvTranspose2d(b*2,b,2,2);s.c1=cbr(b*2,b);s.h=nn.Conv2d(b,no,1)
    def forward(s,x):
        c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2));c4=s.d4(s.p(c3))
        x=s.c3(torch.cat([s.u3(c4),c3],1));x=s.c2(torch.cat([s.u2(x),c2],1));x=s.c1(torch.cat([s.u1(x),c1],1));return s.h(x)
dev="cuda" if torch.cuda.is_available() else "cpu"
net=UNet().to(dev); net.load_state_dict(torch.load(CK,map_location=dev)); net.eval()

def und_pix(P,k1,k2,cx,cy,s,z=1.5): u=(P[:,0]-cx)/s;v=(P[:,1]-cy)/s;r2=u*u+v*v;f=1+k1*r2+k2*r2*r2;return np.stack([cx+u*f/z*s,cy+v*f/z*s],1)
def und_norm(P,k1,k2,cx,cy,s): u=(P[:,0]-cx)/s;v=(P[:,1]-cy)/s;r2=u*u+v*v;f=1+k1*r2+k2*r2*r2;return np.stack([u*f,v*f],1)
def undimg(img,k1,k2,cx,cy,s,z=1.5):
    h,w=img.shape[:2]; rd=np.linspace(0,2.4,2200); rru=rd*(1+k1*rd*rd+k2*rd**4)
    ys,xs=np.mgrid[0:h,0:w].astype(np.float32); uo=(xs-cx)/s*z; vo=(ys-cy)/s*z; ro=np.sqrt(uo*uo+vo*vo)
    rdv=np.interp(ro,rru,rd); sc=np.divide(rdv,ro,out=np.ones_like(ro),where=ro>1e-6)
    return cv2.remap(img,(cx+uo*sc*s).astype(np.float32),(cy+vo*sc*s).astype(np.float32),cv2.INTER_LINEAR)
def aH(uv,H): z=H[2,0]*uv[:,0]+H[2,1]*uv[:,1]+H[2,2];return np.stack([(H[0,0]*uv[:,0]+H[0,1]*uv[:,1]+H[0,2])/z,(H[1,0]*uv[:,0]+H[1,1]*uv[:,1]+H[1,2])/z],1)
def fitL(P):c=P.mean(0);_,_,vt=np.linalg.svd(P-c);nv=np.array([-vt[0,1],vt[0,0]]);return np.array([nv[0],nv[1],-nv.dot(c)])
def inter(a,b):D=a[0]*b[1]-b[0]*a[1];return None if abs(D)<1e-9 else np.array([(a[1]*b[2]-b[1]*a[2])/D,(b[0]*a[2]-a[0]*b[2])/D])
def order4(p):
    p=np.array(p,float);s=p.sum(1);d=p[:,0]-p[:,1];return np.array([p[s.argmin()],p[d.argmax()],p[s.argmax()],p[d.argmin()]])

def run(path):
    img=cv2.imread(path);
    if img is None: return None
    h,w=img.shape[:2]; cx,cy,s=w/2,h/2,w/2
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None].to(dev)
    with torch.no_grad(): p=torch.sigmoid(net(x))[0].cpu().numpy()
    pb=cv2.resize(p[0],(w,h)); pin=cv2.resize(p[1],(w,h))
    vis=img.copy(); vis[pb>0.4]=(0,0,255); vis[pin>0.4]=(0,255,0)
    # ham sınır noktaları -> undistort(shared) ile kaba dörtgen
    Praw=np.column_stack(np.where(pb>0.4))[:,::-1].astype(float)
    fail=lambda: np.full((int((18+2*M)*S),int((L+2*M)*S),3),18,np.uint8)
    if len(Praw)<80: t=fail(); cv2.putText(t,"sinir yok",(8,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2); return montage(path,vis,t)
    Pu=und_pix(Praw,K1S,K2S,cx,cy,s)
    hull=cv2.convexHull(Pu.astype(np.int32)); peri=cv2.arcLength(hull,True); quad=None
    for e in np.linspace(0.01,0.09,18):
        ap=cv2.approxPolyDP(hull,e*peri,True)
        if len(ap)==4: quad=ap.reshape(4,2).astype(float); break
    if quad is None: quad=cv2.boxPoints(cv2.minAreaRect(hull)).astype(float)
    q=order4(quad)
    if np.linalg.norm(q[1]-q[0])<np.linalg.norm(q[3]-q[0]): q=np.array([q[1],q[2],q[3],q[0]])  # uzun kenar yatay
    # kenar çizgileri (undist) -> her ham noktayı en yakın kenara ata
    edges=[(q[3],q[0]),(q[1],q[2]),(q[2],q[3]),(q[0],q[1])]  # X0(goalN),XL(goalF),Y0(touchN),YW(touchF)
    eln=[fitL(np.array(e)) for e in edges]
    dists=np.stack([np.abs(eln[i][0]*Pu[:,0]+eln[i][1]*Pu[:,1]+eln[i][2]) for i in range(4)],1)
    side=dists.argmin(1); near=dists.min(1)<25
    groups={}  # role -> RAW points
    roles=["goalN","goalF","touchN","touchF"]
    rs=np.random.RandomState(0)
    for i,r in enumerate(roles):
        sel=Praw[(side==i)&near]
        if len(sel)>120: sel=sel[rs.choice(len(sel),120,replace=False)]
        if len(sel)>10: groups[r]=sel
    if len(groups)<4: t=fail(); cv2.putText(t,"4 kenar yok",(8,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2); return montage(path,vis,t)
    # yuvarlak: iç maskeden, merkeze yakın, dairemsi blob
    circ=None
    mi=(pin>0.45).astype(np.uint8); mi=cv2.morphologyEx(mi,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
    cnts,_=cv2.findContours(mi,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    best=None
    for c in cnts:
        a=cv2.contourArea(c)
        if a<80: continue
        (xc,yc),rr=cv2.minEnclosingCircle(c); circ_fill=a/(np.pi*rr*rr+1e-6)
        if circ_fill>0.45 and 0.25<xc/w<0.75 and 0.2<yc/h<0.8:
            if best is None or a>best[0]: best=(a,c.reshape(-1,2).astype(float))
    if best is not None:
        cp=best[1]
        if len(cp)>80: cp=cp[np.random.RandomState(1).choice(len(cp),80,replace=False)]
        circ=cp
    # circ (yukarıda) güvenli-daire blob bulunduysa DOLU; seg_v3 iç-tespiti daha iyi ->
    # çemberi aspect kısıtı olarak KULLAN (Wp bound gevşek olduğunda gerçek en-boy verir).
    # circ=None kalırsa (çember yok) Wp bound [17.3,19] standart-varsayıma düşer (güvenli).
    # metrik kısıt: goalN->X0 goalF->XL touchN->Y0 touchF->YW ; (+ yuvarlak yuvarlak)
    CON=[("goalN","X",0.0),("goalF","X",L),("touchN","Y",0.0),("touchF","Yw",None)]
    # init H: kaba köşeler -> rect (Wp=18)
    c0=[inter(eln[0],eln[2]),inter(eln[0],eln[3]),inter(eln[1],eln[2]),inter(eln[1],eln[3])]
    H0,_=cv2.findHomography(np.array(c0,float),np.array([[0,0],[0,18],[L,0],[L,18]],float)); H0=H0/H0[2,2]
    p0=[K1S,K2S,*H0.ravel()[:8],18.0,3.0]
    def resid(pp):
        k1,k2=pp[0],pp[1];H=np.array([[pp[2],pp[3],pp[4]],[pp[5],pp[6],pp[7]],[pp[8],pp[9],1.0]]);Wp,R=pp[10],pp[11];out=[]
        for r,kind,tg in CON:
            m=aH(und_norm(groups[r],k1,k2,cx,cy,s),H)
            out.append(m[:,0]-tg if kind=="X" else (m[:,1]-tg if kind=="Y" else m[:,1]-Wp))
        if circ is not None:
            mc=aH(und_norm(circ,k1,k2,cx,cy,s),H); out.append((np.sqrt((mc[:,0]-L/2)**2+(mc[:,1]-Wp/2)**2)-R))
        return np.concatenate(out)
    # k1 SADECE boundary ile UNDER-CONSTRAINED -> 0'a çöküyordu (fisheye düzeltilmiyor,
    # warp eğri). Gerçekçi balıkgözü aralığına KISITLA (manuel k1 0.08-0.20). Wp da gevşet.
    # Wp bound: çember VARSA gevşet (çember+halfway gerçek en-boyu belirler, non-std
    # genişlik kurtulur); YOKSA standart ~18 varsayımına sıkı-tut (güvenli, çökmez).
    _wlb,_wub=(14.0,24.0) if circ is not None else (17.3,19.0)
    lb=[0.10,0.12]+[-np.inf]*8+[_wlb,0.5]; ub=[0.22,0.40]+[np.inf]*8+[_wub,6.0]
    sol=least_squares(resid,p0,method="trf",bounds=(lb,ub),max_nfev=3000)
    k1,k2=sol.x[0],sol.x[1];H=np.array([[sol.x[2],sol.x[3],sol.x[4]],[sol.x[5],sol.x[6],sol.x[7]],[sol.x[8],sol.x[9],1.0]]);Wp,R=sol.x[10],sol.x[11]
    fit=np.sqrt((resid(sol.x)**2).mean())
    try:
        import json as _json
        _sp=os.path.join(HERE,"seg_auto_solved.json"); _d=_json.load(open(_sp)) if os.path.exists(_sp) else {}
        _d[os.path.basename(path)]={"k1":float(k1),"k2":float(k2),"H":H.tolist(),"Wp":float(Wp),"R":float(R),"fit":float(fit)}
        _json.dump(_d,open(_sp,"w"))
    except Exception as _e: print("save-solved hata:",_e)
    # 2D: undistort + homografi (marjlı)
    uimg=undimg(img,k1,k2,cx,cy,s); LS,WS=int((L+2*M)*S),int((Wp+2*M)*S)
    # H undistorted-normalized->metrik; warp için undistorted-pixel->metrik gerek
    def u2pix(P): un=und_norm(P,k1,k2,cx,cy,s); return np.stack([cx+un[:,0]/1.5*s,cy+un[:,1]/1.5*s],1)
    # köşeleri tekrar (rafine k ile) bul
    fl={r:fitL(u2pix(groups[r])) for r in roles}
    src=np.array([inter(fl["goalN"],fl["touchN"]),inter(fl["goalN"],fl["touchF"]),inter(fl["goalF"],fl["touchN"]),inter(fl["goalF"],fl["touchF"])],float)
    def MX(X,Y): return [ (M+X)*S, WS-(M+Y)*S ]
    dst=np.array([MX(0,0),MX(0,Wp),MX(L,0),MX(L,Wp)],float)
    Hm,_=cv2.findHomography(src,dst); top=cv2.warpPerspective(uimg,Hm,(LS,WS),borderValue=(15,15,15))
    cv2.rectangle(top,tuple(map(int,MX(0,Wp))),tuple(map(int,MX(L,0))),(0,215,255),2)
    cv2.line(top,tuple(map(int,MX(L/2,0))),tuple(map(int,MX(L/2,Wp))),(0,215,255),1)
    cv2.circle(top,tuple(map(int,MX(L/2,Wp/2))),int(R*S),(255,160,0),2)
    cv2.putText(top,f"2D W/L={Wp/L:.2f} res{fit:.2f}m"+(" +yuvarlak" if circ is not None else ""),(8,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,230,255),2)
    return montage(path,vis,top)

def montage(path,vis,top):
    cv2.putText(vis,os.path.basename(path).split('.')[0][:18],(6,24),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,230,255),2)
    Hh=380; vis=cv2.resize(vis,(int(vis.shape[1]*Hh/vis.shape[0]),Hh)); top=cv2.resize(top,(int(top.shape[1]*Hh/top.shape[0]),Hh))
    return np.concatenate([vis,np.full((Hh,6,3),40,np.uint8),top],1)

rows=[run(p) for p in sys.argv[1:]]; rows=[r for r in rows if r is not None]
wmax=max(r.shape[1] for r in rows); sheet=np.full((sum(r.shape[0]+6 for r in rows),wmax,3),20,np.uint8); y=0
for r in rows: sheet[y:y+r.shape[0],0:r.shape[1]]=r; y+=r.shape[0]+6
cv2.imwrite(os.path.join(HERE,"cand","_auto2_2d.jpg"),sheet); print("-> cand/_auto2_2d.jpg",sheet.shape)
