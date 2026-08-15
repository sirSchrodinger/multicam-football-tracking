#!/usr/bin/env python3
"""DÜRÜST TEST: sahayı corner-net calib ile top-down WARP et. Köşeler doğruysa beyaz çizgiler DÜZ+DİK.
Değilse çarpık. Radar noktaları hatayı gizler; warp GÖSTERİR. + kanonik grid bindirir (referans).
Kullanım: python warp_topdown.py <IDX>
"""
import os,sys,json,numpy as np,cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from calib_train.train_corner_detector import CornerNet, RES, RH
from calib_train.vision_calib import calibrate_from_corners
from calib_train import auto_calib as AC
IDX=int(sys.argv[1]); L,W=40.0,20.0; S=24; pad=30
man={e['idx']:e for e in json.load(open("calib_train/night/probcache/manifest.json"))}
e=man[IDX]; img=cv2.imread(f"calib_train/cand_big/{e['file']}"); h,w=img.shape[:2]
net=CornerNet(); net.load_state_dict(torch.load("calib_train/corner_net.pth",map_location="cpu")); net.eval()
with torch.no_grad():
    x=torch.from_numpy(cv2.resize(img,(RES,RH)).transpose(2,0,1).astype(np.float32)/255.)[None]
    c=net(x)[0].numpy().reshape(4,2); corners=np.c_[c[:,0]*w,c[:,1]*h]
rec=calibrate_from_corners(corners,L,W,w,h); k1,k2,H=rec["k1"],rec["k2"],rec["H"]; cx,cy,s=w/2,h/2,w/2
# top-down canvas: her (X,Y) metrik -> kaynak görüntü pikseli (project_metric) -> remap
Wc=int(L*S+2*pad); Hc=int(W*S+2*pad)
gx,gy=np.meshgrid(np.arange(Wc),np.arange(Hc))
Xm=(gx-pad)/S; Ym=W-(gy-pad)/S   # canvas -> metrik (origin sol-alt)
P=np.stack([Xm.ravel(),Ym.ravel()],1)
pix=AC.project_metric(P,k1,k2,H,cx,cy,s)   # metrik -> kaynak görüntü pikseli
mapx=pix[:,0].reshape(Hc,Wc).astype(np.float32); mapy=pix[:,1].reshape(Hc,Wc).astype(np.float32)
warp=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR,borderValue=(20,20,20))
# kanonik grid bindir (referans DÜZ çizgiler) — beyaz saha-çizgileri buna oturmalı
def P2(X,Y): return int(pad+X*S),int(pad+(W-Y)*S)
grid=warp.copy()
cv2.rectangle(grid,P2(0,0),P2(L,W),(0,255,255),1); cv2.line(grid,P2(L/2,0),P2(L/2,W),(0,255,255),1)
cv2.circle(grid,P2(L/2,W/2),int(3*S),(0,255,255),1)
for gx2 in (0,L):
    bx=6 if gx2==0 else L-6; cv2.rectangle(grid,P2(min(gx2,bx),W/2-6),P2(max(gx2,bx),W/2+6),(0,255,255),1)
cv2.putText(grid,f"{e['file'].split('__')[0][:16]} TOP-DOWN WARP + kanonik-grid (sari). Beyaz cizgiler sariya OTURMALI+DUZ.",(8,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
sheet=np.hstack([cv2.resize(warp,(Wc,Hc)),np.full((Hc,6,3),60,np.uint8),grid])
cv2.imwrite(f"calib_train/cand/_WARP_{IDX}.jpg",sheet); print(f"köşeler={corners.astype(int).tolist()} -> _WARP_{IDX}.jpg (sol=warp, sağ=+grid)")
