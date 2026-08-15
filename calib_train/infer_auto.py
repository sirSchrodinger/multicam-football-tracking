#!/usr/bin/env python3
"""TAM OTOMATİK hat: ham kare -> seg model -> sınır maskesi -> undistort -> 4 çizgi/köşe
-> ortak-lens + homografi -> 2D. Manuel etiket YOK. Verilen kareler için [orijinal+tespit |
2D] montajı çıkarır. Kullanım: python infer_auto.py img1 img2 ...
"""
import sys, os, glob, numpy as np, cv2, torch, torch.nn as nn
HERE=os.path.dirname(os.path.abspath(__file__))
K1,K2=0.167,0.240; ZOOM=1.5; TW,TH=512,288; L,W,S,M=34.0,18.0,24,4.0
CK=os.path.join(HERE,"seg_ckpt","seg_unet.pth")

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

def undist(img,cx,cy,s,z=ZOOM):
    h,w=img.shape[:2]; rd=np.linspace(0,2.4,2200); rru=rd*(1+K1*rd*rd+K2*rd**4)
    ys,xs=np.mgrid[0:h,0:w].astype(np.float32); uo=(xs-cx)/s*z; vo=(ys-cy)/s*z; ro=np.sqrt(uo*uo+vo*vo)
    rdv=np.interp(ro,rru,rd); sc=np.divide(rdv,ro,out=np.ones_like(ro),where=ro>1e-6)
    return cv2.remap(img,(cx+uo*sc*s).astype(np.float32),(cy+vo*sc*s).astype(np.float32),cv2.INTER_LINEAR)
def order4(p):
    p=np.array(p,float); s=p.sum(1); d=(p[:,0]-p[:,1]); return np.array([p[s.argmin()],p[d.argmax()],p[s.argmax()],p[d.argmin()]])

def run(path):
    img=cv2.imread(path)
    if img is None: return None
    h,w=img.shape[:2]; cx,cy,s=w/2,h/2,w/2
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None].to(dev)
    with torch.no_grad(): p=torch.sigmoid(net(x))[0].cpu().numpy()
    pb=cv2.resize(p[0],(w,h)); pin=cv2.resize(p[1],(w,h))
    # tespit overlay (ham)
    vis=img.copy(); vis[pb>0.4]=(0,0,255); vis[pin>0.4]=(0,255,0)
    # undistort sınır -> dörtgen
    ub=undist((pb*255).astype(np.uint8),cx,cy,s); uimg=undist(img,cx,cy,s)
    m=(ub>100).astype(np.uint8); m=cv2.morphologyEx(m,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    pts=np.column_stack(np.where(m>0))[:,::-1]
    ok=False; top=np.zeros((int((W+2*M)*S),int((L+2*M)*S),3),np.uint8)
    if len(pts)>50:
        hull=cv2.convexHull(pts.astype(np.int32)); peri=cv2.arcLength(hull,True); quad=None
        for e in np.linspace(0.01,0.09,18):
            ap=cv2.approxPolyDP(hull,e*peri,True)
            if len(ap)==4: quad=ap.reshape(4,2).astype(float); break
        if quad is None: quad=cv2.boxPoints(cv2.minAreaRect(hull)).astype(float)
        q=order4(quad)  # TL,TR,BR,BL
        if np.linalg.norm(q[1]-q[0])<np.linalg.norm(q[3]-q[0]): q=np.array([q[1],q[2],q[3],q[0]])  # uzun kenar yatay
        LS,WS=int((L+2*M)*S),int((W+2*M)*S)
        dst=np.array([[M*S,M*S],[(M+L)*S,M*S],[(M+L)*S,(M+W)*S],[M*S,(M+W)*S]],float)
        Hm,_=cv2.findHomography(q,dst)
        if Hm is not None:
            top=cv2.warpPerspective(uimg,Hm,(LS,WS),borderValue=(15,15,15))
            cv2.rectangle(top,(int(M*S),int(M*S)),(int((M+L)*S),int((M+W)*S)),(0,215,255),2)
            cv2.line(top,(int((M+L/2)*S),int(M*S)),(int((M+L/2)*S),int((M+W)*S)),(0,215,255),1)
            for c in q.astype(int): cv2.circle(uimg,tuple(c),8,(0,255,255),-1)
            ok=True
    name=os.path.basename(path).split(".")[0]
    cv2.putText(vis,name[:18],(6,24),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,230,255),2)
    cv2.putText(top,"2D"+("" if ok else " - dortgen yok"),(6,24),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,230,255),2)
    Hh=360; vis=cv2.resize(vis,(int(vis.shape[1]*Hh/vis.shape[0]),Hh)); top=cv2.resize(top,(int(top.shape[1]*Hh/top.shape[0]),Hh))
    return np.concatenate([vis,np.full((Hh,6,3),40,np.uint8),top],1)

paths=sys.argv[1:]
rows=[run(p) for p in paths]; rows=[r for r in rows if r is not None]
wmax=max(r.shape[1] for r in rows)
sheet=np.full((sum(r.shape[0]+6 for r in rows),wmax,3),20,np.uint8); y=0
for r in rows: sheet[y:y+r.shape[0],0:r.shape[1]]=r; y+=r.shape[0]+6
cv2.imwrite(os.path.join(HERE,"cand","_auto_2d.jpg"),sheet); print("-> cand/_auto_2d.jpg",sheet.shape)
