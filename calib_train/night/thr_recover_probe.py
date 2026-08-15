#!/usr/bin/env python3
"""TANILAMA: bowtie/dejenere calib'ler far-çizgi ZAYIFLIĞINDAN mı? Her sahada thr={.5,.4,.3,.25}
calib_from_pred -> (fit, sanity). Düşük-thr daha çok far-çizgi pixel yakalar -> daha iyi fit/sanity?
EVET ise: retrain'siz çok-thr recovery free kazanç. HAYIR ise: temel detection açığı (retrain gerek).
"""
import os, sys, glob, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from calib_train import auto_calib as AC
from calib_train.auto_clean2d import sanity
from calib_train.render2d_clean import UNet
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CB=os.path.join(HERE,"cand_big")
TW,TH=1024,576
net=UNet(); net.load_state_dict(torch.load(os.path.join(HERE,"seg2_hr2.pth"),map_location="cpu")); net.eval()
@torch.no_grad()
def prob(img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None]
    return torch.sigmoid(net(x))[0].numpy()
# grid'den bowtie/fail isimleri (substring)
TARG=["Berkay75","Gaziantep","Laliga","Playdrome","YeşilÇimen","ÜsküdarAnka","58VİP","Demirciler","EFT","5Mevsim"]
pool=sorted(glob.glob(os.path.join(CB,"*__*__s*.jpg")))
def find(t):
    for p in pool:
        if t.lower().replace("ı","i") in os.path.basename(p).lower().replace("ı","i"): return p
    return None
THRS=[0.5,0.4,0.3,0.25]
print(f"{'saha':16s} | "+" | ".join(f"thr{t}" for t in THRS))
for t in TARG:
    p=find(t)
    if not p: print(f"{t:16s} | (bulunamadı)"); continue
    img=cv2.imread(p); h,w=img.shape[:2]
    pr=prob(img); cells=[]
    for thr in THRS:
        rec=AC.calib_from_pred(pr,w,h,thr=thr)
        if not rec.get("ok"): cells.append("  YOK "); continue
        ok,why=sanity(rec,w,h); cells.append(f"{rec['fit']:.2f}{'✓' if ok else '✗'}")
    print(f"{t[:16]:16s} | "+" | ".join(f"{c:>6s}" for c in cells))
