#!/usr/bin/env python3
"""KEYPOINT+σ eğitimi (GeoCalib deseni): RGB -> NK ısı-haritası + NK güven(σ).
Kayıp = ısı-haritası MSE (görünmez kanal -> sıfıra it, halüsinasyon yok)
      + DSNT alt-piksel koordinat regresyonu (görünür, vis-maskeli)
      + güven BCE(sig, vis) (off-frame keypoint -> düşük güven -> LM down-weight).
Sentetik (3D köşe-kamera) çoğunluk + gerçek saha (3 held-out). Çıktı: kp_ckpt/kp_unet.pth
Doğrulama: held-out gerçek sahalarda görünür keypoint ORTALAMA PİKSEL hatası (düşük=iyi).
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from calib_train.kp_data import real_kp, synth_kp, TW, TH
from calib_train.make_seg_data2 import ORDER, J, real_ok
from calib_train.model_kp import KeypointUNet, dsnt, NK, KP_NAMES
import json
C=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"calib_solved.json")))
BASE=int(os.environ.get("BASE","24")); STEPS=int(os.environ.get("STEPS","3000"))
BATCH=int(os.environ.get("BATCH","8")); NR=int(os.environ.get("NREAL","2"))
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"kp_ckpt"); os.makedirs(OUT,exist_ok=True)
VAL={"AvanosHaliSaha.jpg","KaynarcaAdaHaliSah.jpg","KucukcekmeceIdmanY.jpg"}
reals=[n for n in ORDER if n in C and not C[n].get("bad") and real_ok(n)]
train_reals=[n for n in reals if n not in VAL]; val_reals=[n for n in reals if n in VAL]
print(f"gerçek {len(train_reals)}+{len(val_reals)} val | {NK} keypoint: {KP_NAMES}",flush=True)

def to_t(img,hm,v):
    x=torch.from_numpy(img.transpose(2,0,1).astype(np.float32)/255.)
    y=torch.from_numpy(hm.astype(np.float32)); vv=torch.from_numpy(v.astype(np.float32))
    return x,y,vv
def aug(img):
    return np.clip(img.astype(np.float32)*np.random.uniform(0.72,1.2)+np.random.uniform(-14,14),0,255).astype(np.uint8)
def get_batch(step):
    xs,ys,vs=[],[],[]
    for i in range(BATCH-NR):
        img,hm,v=synth_kp(step*BATCH+i); x,y,vv=to_t(img,hm,v); xs.append(x);ys.append(y);vs.append(vv)
    for _ in range(NR):
        n=train_reals[np.random.randint(len(train_reals))]; r=real_kp(n)
        if r is None: r=synth_kp(step*7+13)
        else: r=(aug(r[0]),r[1],r[2])
        x,y,vv=to_t(*r); xs.append(x);ys.append(y);vs.append(vv)
    return torch.stack(xs),torch.stack(ys),torch.stack(vs)

def gt_xy(y):  # (B,K,H,W) -> (B,K,2) normalize [0,1]
    B,K,H,W=y.shape; flat=y.reshape(B,K,-1); idx=flat.argmax(-1)
    gx=(idx%W).float()/(W-1); gy=(idx//W).float()/(H-1); return torch.stack([gx,gy],-1)

def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev,flush=True)
    net=KeypointUNet(b=BASE).to(dev); opt=torch.optim.Adam(net.parameters(),1e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS); t0=time.time(); best=1e9
    for step in range(STEPS):
        x,y,v=get_batch(step); x,y,v=x.to(dev),y.to(dev),v.to(dev)
        hm,sig=net(x)
        ph=torch.sigmoid(hm); w=1.0+30.0*y
        hm_loss=(w*(ph-y)**2).mean()
        kp,conc=dsnt(hm); g=gt_xy(y)
        vm=v.unsqueeze(-1)
        coord_loss=((vm*(kp-g)**2).sum()/ (v.sum().clamp(min=1)*2))
        conf_loss=F.binary_cross_entropy(sig,v)
        loss=hm_loss+0.15*coord_loss+0.1*conf_loss
        opt.zero_grad();loss.backward();opt.step();sch.step()
        if step%100==0: print(f"step {step}/{STEPS} loss {loss.item():.4f} hm {hm_loss.item():.4f} crd {coord_loss.item():.4f} {(time.time()-t0)/max(1,step+1)*1000:.0f}ms/it",flush=True)
        if step%500==0 and step>0:
            net.eval(); errs=[]
            with torch.no_grad():
                for n in val_reals:
                    r=real_kp(n)
                    if r is None: continue
                    x1,y1,v1=to_t(*r); hm1,sig1=net(x1[None].to(dev)); kp1,_=dsnt(hm1)
                    g1=gt_xy(y1[None].to(dev))[0]; p1=kp1[0].cpu().numpy(); gg=g1.cpu().numpy(); vv=v1.numpy()
                    for i in range(NK):
                        if vv[i]>0:
                            dx=(p1[i,0]-gg[i,0])*TW; dy=(p1[i,1]-gg[i,1])*TH; errs.append((dx*dx+dy*dy)**0.5)
            me=float(np.mean(errs)) if errs else 9e9
            if me<best: best=me; torch.save(net.state_dict(),f"{OUT}/kp_unet.pth")
            print(f"  [val] görünür-keypoint ort piksel hata={me:.1f}px (best {best:.1f}) n={len(errs)}",flush=True); net.train()
    torch.save(net.state_dict(),f"{OUT}/kp_unet_last.pth"); print(f"BİTTİ {time.time()-t0:.0f}s best={best:.1f}px",flush=True)
if __name__=="__main__": main()
