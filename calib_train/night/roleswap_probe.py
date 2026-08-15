#!/usr/bin/env python3
"""HİPOTEZ: bowtie = seg modelinin ROL-KARIŞMASI (near/far çizgiyi ters etiketler).
Her bowtie sahada 4 mantıklı rol-permütasyonunu (identity / goal-swap / touch-swap / both=180°)
solve_calib'e ver -> convex+düşük-residual kurtaran var mı? VARSA: rol-swap düzeltici inşa et.
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
PERMS={"identity":{}, "goalSwap":{"goalN":"goalF","goalF":"goalN"},
       "touchSwap":{"touchN":"touchF","touchF":"touchN"},
       "both180":{"goalN":"goalF","goalF":"goalN","touchN":"touchF","touchF":"touchN"}}
def relabel(groups,perm): return {perm.get(k,k):v for k,v in groups.items()}
TARG=["Berkay75","Gaziantep","Laliga","Playdrome","ÜsküdarAnka","Demirciler","EFT","5Mevsim"]
pool=sorted(glob.glob(os.path.join(CB,"*__*__s*.jpg")))
def find(t):
    for p in pool:
        if t.lower().replace("ı","i") in os.path.basename(p).lower().replace("ı","i"): return p
def run(t):
    p=find(t)
    if not p: return f"{t:14s} (yok)"
    img=cv2.imread(p); h,w=img.shape[:2]; cx,cy,s=w/2,h/2,w/2
    groups,reason=AC.groups_from_prob(prob(img),w,h,thr=0.5)
    if groups is None: return f"{t:14s} çizgi-yok: {reason}"
    cells=[]
    for name,perm in PERMS.items():
        g2=relabel(groups,perm); r=AC.solve_calib(g2,cx,cy,s)
        if r is None: cells.append(f"{name}:H-tekil"); continue
        rec=dict(ok=True,**r); ok,why=sanity(rec,w,h)
        cells.append(f"{name}:{r['fit']:.2f}{'✓' if ok else '✗'}")
    return f"{t[:14]:14s} | "+" | ".join(cells)
for t in TARG: print(run(t),flush=True)
