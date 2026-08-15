#!/usr/bin/env python3
"""GERÇEK kapsama + ÇOK-ADAY recovery. 121-havuz: calib_multicand (largest -> fail ise far top-K combo,
residual-gate). multicand-recovered sahaları ayrı işaretle (görsel doğrulama için). Baseline %33.1'i geçer mi?
"""
import os, sys, glob, json, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from calib_train.render2d_clean import UNet
from calib_train.auto_clean2d import sanity
from calib_train.multicand_calib import calib_multicand
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CB=os.path.join(HERE,"cand_big")
TW,TH=1024,576; DEV="cuda" if torch.cuda.is_available() else "cpu"
net=UNet().to(DEV); net.load_state_dict(torch.load(os.path.join(HERE,"seg2_hr2.pth"),map_location=DEV)); net.eval()
@torch.no_grad()
def prob(img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None].to(DEV)
    return torch.sigmoid(net(x))[0].cpu().numpy()
def main():
    pool=sorted(glob.glob(os.path.join(CB,"*__*__s*.jpg"))); pool=[p for p in pool if "_tmp_" not in p]
    npass=0; nmc=0; rows=[]; recovered=[]
    for i,p in enumerate(pool):
        img=cv2.imread(p)
        if img is None: continue
        h,w=img.shape[:2]
        rec,method,ok=calib_multicand(prob(img),w,h,sanity,img=img)
        if ok:
            npass+=1
            if method=="multicand": nmc+=1; recovered.append(os.path.basename(p))
            rows.append((os.path.basename(p),round(rec["fit"],3),method))
        else:
            rows.append((os.path.basename(p),round(rec["fit"],3) if rec.get("fit") else None,"FAIL"))
        if i%40==0: print(f"  {i}/{len(pool)} pass={npass} (mc={nmc})",flush=True)
    n=len(pool); cov=100*npass/max(1,n)
    print(f"\n=== MULTICAND GERÇEK-KAPSAMA {npass}/{n} = %{cov:.1f} (multicand-recovered +{nmc})")
    print("recovered:", recovered)
    json.dump({"n":n,"pass":npass,"multicand_recovered":nmc,"true_coverage_pct":round(cov,1),
               "recovered_files":recovered,"rows":rows},
              open(os.path.join(HERE,"cand","_truecov_multicand.json"),"w"),ensure_ascii=False,indent=1)
if __name__=="__main__": main()
