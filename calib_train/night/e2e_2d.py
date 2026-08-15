#!/usr/bin/env python3
"""UÇTAN-UCA 2D (tam-otonom): corner_net auto-kalibre -> ft-detektör oyuncular -> 2D radar. Elle-tık YOK.
Kullanım: python e2e_2d.py <IDX>
"""
import os,sys,json,numpy as np,cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from calib_train.train_corner_detector import CornerNet, RES, RH
from calib_train.vision_calib import calibrate_from_corners, overlay
from calib_train import auto_calib as AC
IDX=int(sys.argv[1]); L,W=40.0,20.0
man={e['idx']:e for e in json.load(open("calib_train/night/probcache/manifest.json"))}
e=man[IDX]; img=cv2.imread(f"calib_train/cand_big/{e['file']}"); h,w=img.shape[:2]
# 1) corner_net auto-calib (CPU — küçük model, CUDA transient'e takılma)
net=CornerNet(); net.load_state_dict(torch.load("calib_train/corner_net.pth",map_location="cpu")); net.eval()
with torch.no_grad():
    x=torch.from_numpy(cv2.resize(img,(RES,RH)).transpose(2,0,1).astype(np.float32)/255.)[None]
    c=net(x)[0].numpy().reshape(4,2); corners=np.c_[c[:,0]*w,c[:,1]*h]
rec=calibrate_from_corners(corners,L,W,w,h)
print(f"auto-calib ok={rec.get('ok')} köşeler={corners.astype(int).tolist()}",flush=True)
# 2) ft-detektör
print("RF-DETR...",flush=True)
DEV="cuda" if torch.cuda.is_available() else "cpu"
from rfdetr import RFDETRLargeDeprecated
from PIL import Image
det=RFDETRLargeDeprecated(pretrain_weights="models/weights/checkpoint_ft.pth",device=DEV,num_classes=4)
d=det.predict(Image.fromarray(img[:,:,::-1]),threshold=0.30)
feet=[]; boxes=[]
for (x1,y1,x2,y2),cf in zip(d.xyxy,d.confidence):
    hh=y2-y1; ww=x2-x1
    if hh<20 or ww<=0 or hh/max(ww,1)<1.15: continue
    feet.append([(x1+x2)/2,y2]); boxes.append((int(x1),int(y1),int(x2),int(y2)))
feet=np.array(feet,float) if feet else np.empty((0,2))
# 3) project feet -> metrik (und_norm -> aH)
if len(feet):
    un=AC.und_norm(feet,rec["k1"],rec["k2"],w/2,h/2,w/2); mp=AC.aH(un,rec["H"])
else: mp=np.empty((0,2))
# 4) sol: frame+kutu+calib-çizgi ; sağ: 2D radar
left=overlay(img,rec,L,W)
for (x1,y1,x2,y2) in boxes: cv2.rectangle(left,(x1,y1),(x2,y2),(60,230,60),2)
cv2.putText(left,f"{e['file'].split('__')[0][:16]} — {len(feet)} oyuncu (OTO-KALIB)",(10,34),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)
# radar
S=26; pad=46; Rw=int(L*S+2*pad); Rh=int(W*S+2*pad); radar=np.full((Rh,Rw,3),28,np.uint8)
for i in range(int(L/3)+1):
    x0=int(pad+i*3*S); x1=min(int(pad+(i+1)*3*S),Rw-pad); radar[pad:Rh-pad,x0:x1]=(40,92,40) if i%2 else (46,108,46)
def P(x,y): return int(pad+np.clip(x,0,L)*S),int(pad+(W-np.clip(y,0,W))*S)
cv2.rectangle(radar,P(0,0),P(L,W),(240,240,240),2); cv2.line(radar,P(L/2,0),P(L/2,W),(240,240,240),2)
cv2.circle(radar,P(L/2,W/2),int(3*S),(240,240,240),2)
for gx in (0,L):
    bx=6 if gx==0 else L-6; cv2.rectangle(radar,P(min(gx,bx),W/2-6),P(max(gx,bx),W/2+6),(240,240,240),1)
n=0
for (X,Y) in mp:
    if np.isfinite(X) and np.isfinite(Y) and -2<X<L+2 and -2<Y<W+2:
        cv2.circle(radar,P(X,Y),11,(30,30,235),-1); cv2.circle(radar,P(X,Y),11,(255,255,255),2); n+=1
cv2.putText(radar,f"TAM-OTONOM 2D (corner-net auto-calib, SIFIR tik) — {n} oyuncu",(10,26),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,210,255),1)
Lh=left.shape[0]; radar=cv2.resize(radar,(int(Rw*Lh/Rh),Lh))
cv2.imwrite("calib_train/cand/_E2E_2D.jpg",np.hstack([left,np.full((Lh,6,3),60,np.uint8),radar]))
print(f"oyuncu-2D={n} -> _E2E_2D.jpg")
