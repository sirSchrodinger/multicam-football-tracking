#!/usr/bin/env python3
"""HİPOTEZ: bowtie = far-çizgi rolünün LARGEST-component'i YANLIŞ (fence/duvar/file'a kilitleniyor),
gerçek çizgi 2.-3. component. Her far-rol (goalF,touchF) için top-3 component -> tüm combo'ları solve ->
convex+düşük-residual seçen var mı (largest'tan FARKLI)? VARSA: multi-cand recovery inşa et.
"""
import os, sys, glob, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from itertools import product
from calib_train import auto_calib as AC
from calib_train.auto_clean2d import sanity
from calib_train.render2d_clean import UNet
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CB=os.path.join(HERE,"cand_big")
TW,TH=1024,576
net=UNet().cuda() if torch.cuda.is_available() else UNet()
net.load_state_dict(torch.load(os.path.join(HERE,"seg2_hr2.pth"),map_location="cuda" if torch.cuda.is_available() else "cpu")); net.eval()
DEV="cuda" if torch.cuda.is_available() else "cpu"
@torch.no_grad()
def prob(img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None].to(DEV)
    return torch.sigmoid(net(x))[0].cpu().numpy()
def role_cands(pc,w,h,thr=0.5,minpts=25,topk=3,maxpts=180):
    """bir rol kanalı -> top-k component'in nokta-setleri (alan azalan)."""
    pcr=cv2.resize(pc,(w,h)); mk=(pcr>thr).astype(np.uint8)
    mk=cv2.morphologyEx(mk,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    ncc,lbl,st,_=cv2.connectedComponentsWithStats(mk)
    order=np.argsort(-st[1:,cv2.CC_STAT_AREA])+1 if ncc>1 else []
    out=[]
    rs=np.random.RandomState(0)
    for li in order[:topk]:
        pts=np.column_stack(np.where(lbl==li))[:,::-1].astype(float)
        if len(pts)<minpts: continue
        if len(pts)>maxpts: pts=pts[rs.choice(len(pts),maxpts,replace=False)]
        out.append(pts)
    return out
TARG=["Berkay75","Gaziantep","Laliga","Playdrome","ÜsküdarAnka","Cumcum","Selçuklu","YunusArena"]
pool=sorted(glob.glob(os.path.join(CB,"*__*__s*.jpg")))
def find(t):
    for p in pool:
        if t.lower().replace("ı","i") in os.path.basename(p).lower().replace("ı","i"): return p
for t in TARG:
    p=find(t)
    if not p: print(f"{t:13s} (yok)"); continue
    img=cv2.imread(p); h,w=img.shape[:2]; cx,cy,s=w/2,h/2,w/2; pr=prob(img)
    # near roller: largest (genelde güvenilir); far roller: top-3 aday
    gn=role_cands(pr[0],w,h,topk=1); tn=role_cands(pr[2],w,h,topk=1)
    gf=role_cands(pr[1],w,h,topk=3); tf=role_cands(pr[3],w,h,topk=3)
    if not(gn and tn and gf and tf): print(f"{t:13s} aday-eksik (gn{len(gn)} tn{len(tn)} gf{len(gf)} tf{len(tf)})"); continue
    best=None
    for i,(gfi,tfi) in enumerate(product(range(len(gf)),range(len(tf)))):
        groups={"goalN":gn[0],"goalF":gf[gfi],"touchN":tn[0],"touchF":tf[tfi]}
        r=AC.solve_calib(groups,cx,cy,s)
        if r is None: continue
        ok,_=sanity(dict(ok=True,**r),w,h,img=img)
        tagc=f"gf{gfi}tf{tfi}"
        if ok and (best is None or r["fit"]<best[1]): best=(tagc,r["fit"])
    # largest-only sonuç (referans)
    g0={"goalN":gn[0],"goalF":gf[0],"touchN":tn[0],"touchF":tf[0]}; r0=AC.solve_calib(g0,cx,cy,s)
    ok0,_=sanity(dict(ok=True,**r0),w,h,img=img) if r0 else (False,"")
    base=f"{r0['fit']:.2f}{'✓' if ok0 else '✗'}" if r0 else "tekil"
    rec=f"{best[0]} {best[1]:.2f}✓" if best else "YOK"
    print(f"{t[:13]:13s} | largest={base:8s} | multi-cand-best={rec}  (gf-aday={len(gf)} tf-aday={len(tf)})",flush=True)
