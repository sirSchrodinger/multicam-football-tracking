#!/usr/bin/env python3
"""İLERLEME GÖRSELİ: ham fisheye kare -> OTONOM 2D üstten-görünüm (yan yana).
seg2 rol-çizgi -> joint kalibrasyon (auto_calib) -> metrik->görüntü ters-harita ile top-down.
CPU'ya zorlar (eğitim GPU'suyla çakışmasın). Çıktı: cand/_PROGRESS.jpg
"""
import os, sys, glob, numpy as np, cv2
os.environ["CUDA_VISIBLE_DEVICES"]=""   # CPU zorla
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn
from calib_train import auto_calib as AC
HERE=os.path.dirname(os.path.abspath(__file__)); LF=os.path.join(HERE,"label_frames"); CAND=os.path.join(HERE,"cand")
NC=7; TW=int(os.environ.get("TW","512")); TH=int(os.environ.get("TH","288")); BASE=int(os.environ.get("BASE","24")); L,Wp,S,MG=34.0,18.0,24,3
SEG2=os.environ.get("MODEL",os.path.join(HERE,"seg2_ckpt","seg2_unet.pth")); PRE=os.environ.get("PRE","none")
def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True),nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True))
class UNet(nn.Module):
    def __init__(s,nin=3,no=NC,b=BASE):
        super().__init__(); s.d1=cbr(nin,b);s.d2=cbr(b,b*2);s.d3=cbr(b*2,b*4);s.d4=cbr(b*4,b*8);s.p=nn.MaxPool2d(2)
        s.u3=nn.ConvTranspose2d(b*8,b*4,2,2);s.c3=cbr(b*8,b*4);s.u2=nn.ConvTranspose2d(b*4,b*2,2,2);s.c2=cbr(b*4,b*2)
        s.u1=nn.ConvTranspose2d(b*2,b,2,2);s.c1=cbr(b*2,b);s.h=nn.Conv2d(b,no,1)
    def forward(s,x):
        c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2));c4=s.d4(s.p(c3))
        x=s.c3(torch.cat([s.u3(c4),c3],1));x=s.c2(torch.cat([s.u2(x),c2],1));x=s.c1(torch.cat([s.u1(x),c1],1));return s.h(x)
net=UNet(); net.load_state_dict(torch.load(SEG2,map_location="cpu")); net.eval()
@torch.no_grad()
def predict(img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None]
    return torch.sigmoid(net(x))[0].numpy()

def undimg(img,k1,k2,cx,cy,s,z=1.5):
    h,w=img.shape[:2]; rd=np.linspace(0,2.4,2200); rru=rd*(1+k1*rd*rd+k2*rd**4)
    ys,xs=np.mgrid[0:h,0:w].astype(np.float32); uo=(xs-cx)/s*z; vo=(ys-cy)/s*z; ro=np.sqrt(uo*uo+vo*vo)
    rdv=np.interp(ro,rru,rd); sc=np.divide(rdv,ro,out=np.ones_like(ro),where=ro>1e-6)
    return cv2.remap(img,(cx+uo*sc*s).astype(np.float32),(cy+vo*sc*s).astype(np.float32),cv2.INTER_LINEAR,borderValue=(0,0,0))
def render_top(img,rec):
    """infer_auto_roles renderer: undistort -> köşeleri yeniden-fit -> homografi -> warp (temiz)."""
    k1,k2=rec["k1"],rec["k2"]; groups=rec["groups"]; h,w=img.shape[:2]; cx,cy,s=w/2,h/2,w/2
    LS,WS=int((L+2*MG)*S),int((Wp+2*MG)*S); und=undimg(img,k1,k2,cx,cy,s)
    def u2pix(P): un=AC.und_norm(P,k1,k2,cx,cy,s); return np.stack([cx+un[:,0]/1.5*s,cy+un[:,1]/1.5*s],1)
    fl={r:AC.fitL(u2pix(groups[r])) for r in groups}
    c_nN=AC.inter(fl["goalN"],fl["touchN"]);c_nF=AC.inter(fl["goalN"],fl["touchF"]);c_fN=AC.inter(fl["goalF"],fl["touchN"]);c_fF=AC.inter(fl["goalF"],fl["touchF"])
    mir=rec["camside"]=="SAG"
    def MX(X,Y): return [((L-X if mir else X)+MG)*S, WS-(Y+MG)*S]
    src=np.array([c_nN,c_nF,c_fN,c_fF],float); dst=np.array([MX(0,0),MX(0,Wp),MX(L,0),MX(L,Wp)],float)
    Hm,_=cv2.findHomography(src,dst); top=cv2.warpPerspective(und,Hm,(LS,WS),borderValue=(0,0,0))
    top[top.sum(2)<8]=(28,28,28)
    def P(X,Y): return tuple(map(int,MX(X,Y)))
    cv2.rectangle(top,P(0,0),P(L,Wp),(0,215,255),2); cv2.line(top,P(L/2,0),P(L/2,Wp),(0,215,255),1)
    cv2.circle(top,P(L/2,Wp/2),int(3*S),(0,215,255),1)
    for gx in (0,L):
        bx=5 if gx==0 else L-5; cv2.rectangle(top,P(min(gx,bx),Wp/2-5),P(max(gx,bx),Wp/2+5),(0,215,255),1)
    cam=P(0,0); cv2.circle(top,cam,10,(0,200,255),-1); cv2.putText(top,"KAM",(cam[0]+8,cam[1]-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,255),2)
    return top

# aday havuz: etiketli + harvested, en düşük residual'lı benzersiz sahalardan seç
def norm(name):  # Türkçe-kıvrım + casefold -> ASCII slug (ç/ü/ö/ş/ğ/ı eşlensin)
    t=name.lower().translate(str.maketrans("çüöşğıİ","cuosgii")); return "".join(c for c in t if c.isalnum())[:9]
paths=sorted(glob.glob(os.path.join(LF,"*.jpg")))+sorted(glob.glob(os.path.join(CAND,"*__*__s*.jpg")))
seen=set(); picks=[]
for p in paths:
    img=cv2.imread(p)
    if img is None: continue
    rec=AC.calib_from_pred(predict(img),img.shape[1],img.shape[0])
    if not rec.get("ok") or rec["fit"] is None or rec["fit"]>0.45: continue
    vslug=norm(os.path.basename(p).split("__")[0])
    if vslug in seen: continue
    seen.add(vslug); picks.append((p,img,rec,rec["fit"]))
picks=sorted(picks,key=lambda r:r[3])[:6]
print(f"seçilen {len(picks)} saha (en iyi residual):",[f"{os.path.basename(p)[:10]}={f:.2f}" for p,_,_,f in picks],flush=True)
rows=[]
PW=int((L+2*MG)*S)  # 2D genişlik
for p,img,rec,fit in picks:
    top=render_top(img,rec); WS=top.shape[0]
    raw=cv2.resize(img,(int(img.shape[1]*WS/img.shape[0]),WS))
    cv2.putText(raw,os.path.basename(p).split("__")[0][:16],(8,28),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    cv2.putText(top,f"OTONOM 2D  res={fit:.2f}m  kam={rec['camside']}",(8,WS-12),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,230,255),2)
    rows.append(np.hstack([raw,np.full((WS,6,3),50,np.uint8),top]))
wm=max(r.shape[1] for r in rows); sheet=np.full((sum(r.shape[0]+8 for r in rows)+40,wm,3),18,np.uint8)
cv2.putText(sheet,"HALISAHA OTONOM KALIBRASYON  —  ham fisheye  ->  2D ustten gorunum",(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.8,(80,220,255),2)
y=40
for r in rows: sheet[y:y+r.shape[0],0:r.shape[1]]=r; y+=r.shape[0]+8
out=os.path.join(CAND,"_PROGRESS.jpg"); cv2.imwrite(out,sheet); print("->",out,flush=True)
