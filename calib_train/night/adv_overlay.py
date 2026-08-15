#!/usr/bin/env python3
import os, sys, json, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from calib_train import auto_calib as AC
import method_vanishing_point as VP

REPO = os.environ.get("HALISAHA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE=f"{REPO}/calib_train/night/probcache"
OUT=f"{REPO}/calib_train/night/adv_overlays"
os.makedirs(OUT,exist_ok=True)
man={e["idx"]:e for e in json.load(open(f"{CACHE}/manifest.json"))}

# BGR colors per channel
COL={0:(0,0,255),1:(0,128,255),2:(255,0,0),3:(255,128,0),
     4:(0,255,0),5:(255,0,255),6:(0,255,255)}
NAME={0:"goalN",1:"goalF",2:"touchN",3:"touchF",4:"center",5:"box",6:"circle"}

def overlay(idx, scale=1.0):
    e=man[idx]; file=e["file"]; h=e["h"]; w=e["w"]
    prob=np.load(f"{CACHE}/{idx:03d}.npz")["prob"].astype("float32")
    img=cv2.imread(f"{REPO}/calib_train/cand_big/{file}")
    rec=VP.calib_vp(prob,w,h,img=img)
    if not rec.get("ok"):
        print(idx,"NOT OK",rec.get("reason")); return None
    m=AC.clean_label_masks(rec,w,h)
    ov=img.copy()
    for ci in range(7):
        ys,xs=np.where(m[ci]>0)
        ov[ys,xs]=COL[ci]
    # blend a bit so underlying lines remain visible
    out=cv2.addWeighted(img,0.35,ov,0.65,0)
    if scale!=1.0:
        out=cv2.resize(out,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA)
    p=f"{OUT}/{idx:03d}_{rec.get('src')}_ic{(rec.get('ic') or -1):.2f}.png"
    cv2.imwrite(p,out)
    print(idx,"WROTE",p,"ic=",round(rec.get('ic') or -1,3))
    return p

if __name__=="__main__":
    ids=[int(x) for x in sys.argv[1:]]
    for i in ids: overlay(i, scale=0.5)
