#!/usr/bin/env python3
"""DOĞRU eval: corner-net köşe-tahmin -> vision_calib -> LINE-ALIGN (gerçek accuracy, sanity değil).
line-align<3px & inlier>0.55 = DOĞRU-kalibre saha. Örnek montaj (warp'la görsel-teyit)."""
import os,sys,json,numpy as np,cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from calib_train.train_corner_detector import CornerNet, RES, RH
from calib_train.vision_calib import calibrate_from_corners
from calib_train.line_align import line_align_score
net=CornerNet(); net.load_state_dict(torch.load("calib_train/corner_net.pth",map_location="cpu")); net.eval()
@torch.no_grad()
def pred(img):
    h,w=img.shape[:2]
    x=torch.from_numpy(cv2.resize(img,(RES,RH)).transpose(2,0,1).astype(np.float32)/255.)[None]
    c=net(x)[0].numpy().reshape(4,2); return np.c_[c[:,0]*w,c[:,1]*h]
man=json.load(open("calib_train/night/probcache/manifest.json"))
acc=0; rows=[]
for e in man:
    img=cv2.imread(f"calib_train/cand_big/{e['file']}")
    if img is None: continue
    h,w=img.shape[:2]; rec=calibrate_from_corners(pred(img),40.0,20.0,w,h)
    if not rec.get("ok"): rows.append((e['idx'],None,0)); continue
    px,inl=line_align_score(rec,img,40.0,20.0)
    good=(px is not None and px<3.0 and inl>0.55)
    if good: acc+=1
    rows.append((e['idx'],round(px,1) if px else None,round(inl,2) if px else 0))
n=len(man)
print(f"\nCORNER-NET DOĞRU-kalibre (line-align<3px & inlier>0.55): {acc}/{n} = %{100*acc/n:.1f}")
med=[r[1] for r in rows if r[1] is not None]
print(f"line-align median tüm: {np.median(med):.1f}px" if med else "")
json.dump({"accurate":acc,"n":n,"rows":rows},open("calib_train/cand/_corner_align.json","w"))
