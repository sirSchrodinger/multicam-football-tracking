#!/usr/bin/env python3
"""KÖŞE-REGRESÖR (uzun-soluklu vision-calib sisteminin çekirdeği).
Hedef: image -> 4 saha-köşesi (BL,TL,BR,TR normalized, occluded/off-frame DAHİL). Regresyon (heatmap değil)
off-frame köşeyi doğal işler. Synth SINIRSIZ köşe-etiket (saha geometrisi bilinir) + (sonra) real vision-label.
Çıktı: corner_net.pth. Eval: real cand_big'de köşe-tahmin -> vision_calib -> sanity-pass oranı.
ENV: STEPS BATCH RES REAL_JSON(opsiyonel vision-label).
"""
import os, sys, time, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from calib_train.make_seg_data2 import synth_sample
RES=int(os.environ.get("RES","256")); RH=int(RES*9/16)
STEPS=int(os.environ.get("STEPS","3000")); BATCH=int(os.environ.get("BATCH","32")); WORKERS=int(os.environ.get("WORKERS","8"))
HERE=os.path.dirname(os.path.abspath(__file__)); DEV="cuda" if torch.cuda.is_available() else "cpu"

def synth_corners(seed):
    img,m,meta=synth_sample(seed,return_meta=True); proj=meta["proj"]; L=meta["L"]; W=meta["W"]
    c=proj(np.array([[0,0],[L,0],[L,W],[0,W]],float))[0]   # BL'(X0Y0) cL0(XL Y0) cLW(XL YW) c0W(X0 YW)
    h,w=img.shape[:2]; cn=np.array([[c[0,0]/w,c[0,1]/h],[c[3,0]/w,c[3,1]/h],[c[1,0]/w,c[1,1]/h],[c[2,0]/w,c[2,1]/h]])
    return img, cn.astype(np.float32).ravel()   # [BL,TL,BR,TR] normalized (TL=c0W, BR=cL0)

PREAL=float(os.environ.get("PREAL","0.5")); REAL_JSON=os.environ.get("REAL_JSON",f"{HERE}/corner_labels.json")
def _load_json(path, subdir):
    if not os.path.exists(path): return []
    d=json.load(open(path)); out=[]
    for k,v in d.items():
        try:
            fn=v.get("file") or v.get("file_label")
            c=np.array(v["corners"],float); c[:,0]/=v["w"]; c[:,1]/=v["h"]
            out.append((f"{HERE}/{subdir}/{fn}", c.astype(np.float32).ravel()))
        except: pass
    return out
def _load_real():
    # HASSAS (Alperen line-calib, label_frames) yüksek-ağırlık + cand_big corner_labels
    acc=_load_json(f"{HERE}/accurate_corners.json","label_frames")
    rough=_load_json(REAL_JSON,"cand_big")
    return acc*3 + rough   # hassasları 3x tekrarla (accuracy ağırlığı)
REAL=_load_real()
def _aug(img):
    return np.clip(img.astype(np.float32)*np.random.uniform(0.6,1.3)+np.random.uniform(-20,20),0,255).astype(np.uint8)
class SynthDS(Dataset):
    def __len__(s): return STEPS*BATCH
    def __getitem__(s,i):
        rng=np.random.RandomState(i*2654435761%(2**31))
        if REAL and rng.rand()<PREAL:
            path,c=REAL[rng.randint(len(REAL))]; img=cv2.imread(path)
            if img is None: img,c=synth_corners(i)
            else: img=_aug(img)
        else:
            img,c=synth_corners(i)
        x=torch.from_numpy(cv2.resize(img,(RES,RH)).transpose(2,0,1).astype(np.float32)/255.)
        return x, torch.from_numpy(c)
def cbr(i,o,st=1): return nn.Sequential(nn.Conv2d(i,o,3,st,1),nn.BatchNorm2d(o),nn.ReLU(True))
class CornerNet(nn.Module):
    def __init__(s,b=32):
        super().__init__()
        s.f=nn.Sequential(cbr(3,b),cbr(b,b,2),cbr(b,b*2),cbr(b*2,b*2,2),cbr(b*2,b*4),cbr(b*4,b*4,2),
                          cbr(b*4,b*8),cbr(b*8,b*8,2),cbr(b*8,b*8,2),nn.AdaptiveAvgPool2d(1))
        s.h=nn.Sequential(nn.Flatten(),nn.Linear(b*8,256),nn.ReLU(True),nn.Linear(256,8))
    def forward(s,x): return s.h(s.f(x))

def main():
    net=CornerNet().to(DEV); opt=torch.optim.Adam(net.parameters(),1e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS)
    dl=DataLoader(SynthDS(),batch_size=BATCH,num_workers=WORKERS,persistent_workers=True,drop_last=True)
    t0=time.time(); step=0
    for x,y in dl:
        if step>=STEPS: break
        x,y=x.to(DEV),y.to(DEV); pred=net(x); loss=F.smooth_l1_loss(pred,y)
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if step%200==0: print(f"step {step}/{STEPS} loss {loss.item():.4f} {(time.time()-t0)/max(1,step+1)*1000:.0f}ms/it",flush=True)
        step+=1
    torch.save(net.state_dict(),f"{HERE}/corner_net.pth"); print(f"BİTTİ {time.time()-t0:.0f}s -> corner_net.pth",flush=True)
if __name__=="__main__": main()
