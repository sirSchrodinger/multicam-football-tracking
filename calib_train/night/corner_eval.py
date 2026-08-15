#!/usr/bin/env python3
"""Synth-eğitimli corner_net'i REAL cand_big'de değerlendir: köşe-tahmin -> vision_calib -> sanity-pass oranı.
Genelleme proof'u: synth-eğitimli model real sahaya köşe-tahmin edip kalibre edebiliyor mu?
"""
import os,sys,json,glob,numpy as np,cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from calib_train.train_corner_detector import CornerNet, RES, RH
from calib_train.vision_calib import calibrate_from_corners, overlay
from calib_train.auto_clean2d import sanity
DEV="cuda" if torch.cuda.is_available() else "cpu"
net=CornerNet().to(DEV); net.load_state_dict(torch.load("calib_train/corner_net.pth",map_location=DEV)); net.eval()
@torch.no_grad()
def pred_corners(img):
    h,w=img.shape[:2]
    x=torch.from_numpy(cv2.resize(img,(RES,RH)).transpose(2,0,1).astype(np.float32)/255.)[None].to(DEV)
    c=net(x)[0].cpu().numpy().reshape(4,2)   # [BL,TL,BR,TR] normalized
    return np.c_[c[:,0]*w, c[:,1]*h]
man=json.load(open("calib_train/night/probcache/manifest.json"))
npass=0; rows=[]; sample=[]
for e in man:
    img=cv2.imread(f"calib_train/cand_big/{e['file']}")
    if img is None: continue
    h,w=img.shape[:2]; corners=pred_corners(img)
    rec=calibrate_from_corners(corners,40.0,20.0,w,h)
    ok=False
    if rec.get("ok"):
        ok,why=sanity(rec,w,h,img=img)
    if ok:
        npass+=1
        if len(sample)<10: sample.append((e['idx'],e['file'],img,rec,corners))
    rows.append((e['idx'],e['file'].split('__')[0][:16],ok))
print(f"\nCORNER-NET (synth-eğitimli) REAL kapsama: {npass}/{len(man)} sanity-PASS = %{100*npass/len(man):.1f}")
# sample overlay montaj
if sample:
    panels=[]
    for idx,file,img,rec,corners in sample:
        h,w=img.shape[:2]; o=overlay(img,rec,40,20)
        for p in corners.astype(int): cv2.circle(o,tuple(p),12,(255,255,255),-1)
        cv2.putText(o,file.split('__')[0][:16],(10,32),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,255),2)
        panels.append(cv2.resize(o,(640,int(640*h/w))))
    W=max(p.shape[1] for p in panels); panels=[cv2.copyMakeBorder(p,0,0,0,W-p.shape[1],cv2.BORDER_CONSTANT,value=(20,20,20)) for p in panels]
    cols=2; rows2=(len(panels)+1)//2; cH=max(p.shape[0] for p in panels)
    sheet=np.full((rows2*(cH+6),cols*(W+6),3),12,np.uint8)
    for i,p in enumerate(panels):
        r,c=divmod(i,cols); sheet[r*(cH+6):r*(cH+6)+p.shape[0],c*(W+6):c*(W+6)+p.shape[1]]=p
    cv2.imwrite("calib_train/cand/_CORNERNET_EVAL.jpg",sheet); print("-> _CORNERNET_EVAL.jpg")
