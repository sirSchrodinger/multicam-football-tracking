#!/usr/bin/env python3
"""Bowtie kökü görsel teyit: iyi saha (Demirciler) vs bowtie (Üsküdar/Gaziantep) için
4 sınır-rol kanalı (goalN/goalF/touchN/touchF) heatmap olarak. Far-çizgi (goalF/touchF)
bowtie sahalarda fragmentli/yanlış mı? -> _SEGCHAN_DIAG.jpg
ENV: MODEL=seg2_hr2.pth
"""
import os, sys, glob, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from calib_train.render2d_clean import UNet
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CB=os.path.join(HERE,"cand_big")
TW,TH=1024,576; MODEL=os.environ.get("MODEL",os.path.join(HERE,"seg2_hr2.pth"))
net=UNet(); net.load_state_dict(torch.load(MODEL,map_location="cpu")); net.eval()
ROLES=["goalN","goalF","touchN","touchF"]; COL=[(0,0,255),(0,165,255),(0,255,0),(255,140,0)]
@torch.no_grad()
def prob(img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None]
    return torch.sigmoid(net(x))[0].numpy()
pool=sorted(glob.glob(os.path.join(CB,"*__*__s*.jpg")))
def find(t):
    for p in pool:
        if t.lower().replace("ı","i") in os.path.basename(p).lower().replace("ı","i"): return p
def row(t,tag):
    p=find(t); img=cv2.imread(p); h,w=img.shape[:2]; pr=prob(img)
    disp=cv2.resize(img,(TW,TH)).copy()
    ov=cv2.resize(img,(TW,TH)).astype(np.float32)
    for ci in range(4):
        heat=pr[ci]; m=heat>0.4
        ov[m]=ov[m]*0.25+np.array(COL[ci])*0.75
    ov=ov.astype(np.uint8)
    cv2.putText(disp,f"{tag}: {t[:16]}",(8,28),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
    cv2.putText(ov,"goalN=kirmizi goalF=turuncu touchN=yesil touchF=mavi",(8,28),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    # her far-kanal max-prob (zayıflık ölçer)
    info=f"goalF_max={pr[1].max():.2f} touchF_max={pr[3].max():.2f}"
    cv2.putText(ov,info,(8,TH-12),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)
    return np.hstack([disp,np.full((TH,6,3),60,np.uint8),ov])
rows=[row("Demirciler","İYİ"),row("EFT","İYİ"),row("Gaziantep","BOWTIE"),row("ÜsküdarAnka","BOWTIE")]
W=max(r.shape[1] for r in rows); rows=[cv2.copyMakeBorder(r,0,0,0,W-r.shape[1],cv2.BORDER_CONSTANT,value=(20,20,20)) for r in rows]
sheet=np.vstack([np.vstack([r,np.full((8,W,3),20,np.uint8)]) for r in rows])
out=os.path.join(HERE,"cand","_SEGCHAN_DIAG.jpg"); cv2.imwrite(out,sheet); print("->",out)
