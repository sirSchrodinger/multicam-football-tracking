#!/usr/bin/env python3
"""Yüksek-res seg2 + PARALEL DataLoader (64 vCPU'yu kullan -> GPU starvation YOK).
make_seg_data2 sentetik üretimi tek-thread CPU darboğazıydı; DataLoader num_workers ile
paralelleştirir. TW=768 BASE=32 default. Çıktı: seg2_ckpt/seg2_unet.pth (aynı isim -> guard indirir).
"""
import sys, os, time, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from calib_train.make_seg_data2 import synth_sample, real_sample, real_ok, ORDER, J, NC, CLASSES
TW=int(os.environ.get("TW","768")); TH=int(os.environ.get("TH","432")); BASE=int(os.environ.get("BASE","32"))
STEPS=int(os.environ.get("STEPS","8000")); BATCH=int(os.environ.get("BATCH","12")); WORKERS=int(os.environ.get("WORKERS","24"))
PSYNTH=float(os.environ.get("PSYNTH","0.62"))
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"seg2_ckpt"); os.makedirs(OUT,exist_ok=True)
VAL=set(os.environ.get("VAL","f6_AdaArenaHa.jpg,f83_Meteoroloj.jpg,f13_ArslanHalS.jpg,f96_SamsunTerm.jpg").split(","))  # AUDIT: val SADECE held-out YENİ venue
reals=[n for n in ORDER if n in J and real_ok(n)]; train_reals=[n for n in reals if n not in VAL]; val_reals=[n for n in reals if n in VAL]
print(f"gerçek {len(train_reals)}+{len(val_reals)} val | {NC} sınıf | TW{TW} BASE{BASE} BATCH{BATCH} WORKERS{WORKERS}",flush=True)
def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True),nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True))
class UNet(nn.Module):
    def __init__(s,nin=3,no=NC,b=BASE):
        super().__init__(); s.d1=cbr(nin,b);s.d2=cbr(b,b*2);s.d3=cbr(b*2,b*4);s.d4=cbr(b*4,b*8);s.p=nn.MaxPool2d(2)
        s.u3=nn.ConvTranspose2d(b*8,b*4,2,2);s.c3=cbr(b*8,b*4);s.u2=nn.ConvTranspose2d(b*4,b*2,2,2);s.c2=cbr(b*4,b*2)
        s.u1=nn.ConvTranspose2d(b*2,b,2,2);s.c1=cbr(b*2,b);s.h=nn.Conv2d(b,no,1)
    def forward(s,x):
        c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2));c4=s.d4(s.p(c3))
        x=s.c3(torch.cat([s.u3(c4),c3],1));x=s.c2(torch.cat([s.u2(x),c2],1));x=s.c1(torch.cat([s.u1(x),c1],1));return s.h(x)
def to_t(img,m):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)
    y=torch.from_numpy(np.stack([cv2.resize(m[c],(TW,TH)) for c in range(NC)]).astype(np.float32)/255.)
    return x,y
def aug(img):
    return np.clip(img.astype(np.float32)*np.random.uniform(0.7,1.2)+np.random.uniform(-14,14),0,255).astype(np.uint8)
HERE=os.path.dirname(os.path.abspath(__file__))
PSEUDO=sorted(glob.glob(os.path.join(HERE,"pseudo_data","*_mask.npz"))); PPSEUDO=float(os.environ.get("PPSEUDO","0.30"))
print(f"pseudo-etiket: {len(PSEUDO)} saha",flush=True)
class TrainDS(Dataset):
    def __len__(s): return STEPS*BATCH
    def __getitem__(s,idx):
        rng=np.random.RandomState(idx*2654435761 % (2**31))
        if rng.rand()<PSYNTH: img,m=synth_sample(idx)
        elif PSEUDO and rng.rand()<PPSEUDO:
            pf=PSEUDO[rng.randint(len(PSEUDO))]; m=np.load(pf)['m']; img=aug(cv2.imread(pf.replace('_mask.npz','_img.jpg')))
        else: n=train_reals[rng.randint(len(train_reals))]; img,m=real_sample(n,size=(TW,TH)); img=aug(img)
        return to_t(img,m)
def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev,flush=True)
    _CW=torch.tensor([35.,55.,35.,55.,45.,35.,35.],device=dev).view(1,7,1,1)  # gN gF tN tF ctr box cir
    dl=DataLoader(TrainDS(),batch_size=BATCH,num_workers=WORKERS,pin_memory=True,persistent_workers=True,prefetch_factor=4,drop_last=True)
    net=UNet().to(dev)
    W=os.environ.get("WARM")
    if W and os.path.exists(W):
        try: net.load_state_dict(torch.load(W,map_location=dev)); print(f"WARM-start: {W}",flush=True)
        except Exception as ex: print(f"warm-load fail ({ex}); sıfırdan",flush=True)
    opt=torch.optim.Adam(net.parameters(),1e-3); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS)
    t0=time.time(); step=0; best=0.0
    for x,y in dl:
        if step>=STEPS: break
        net.train(); x,y=x.to(dev,non_blocking=True),y.to(dev,non_blocking=True); pred=net(x)
        # kanal-başı poz-ağırlık: UZAK kale(1)/UZAK taç(3)=55, orta(4)=45 (sağ-üst köşe + orta-çizgi zayıftı), diğer=35
        w=1.0+_CW*y; loss=(w*F.binary_cross_entropy_with_logits(pred,y,reduction="none")).mean()
        opt.zero_grad();loss.backward();opt.step();sch.step()
        if step%100==0: print(f"step {step}/{STEPS} loss {loss.item():.4f} {(time.time()-t0)/max(1,step+1)*1000:.0f}ms/it",flush=True)
        if step%500==0 and step>0:
            torch.save(net.state_dict(),f"{OUT}/seg2_unet.pth"); net.eval(); iou=np.zeros(NC)
            with torch.no_grad():
                for n in val_reals:
                    img,m=real_sample(n,size=(TW,TH)); xv,yv=to_t(img,m); p=torch.sigmoid(net(xv[None].to(dev)))[0].cpu().numpy()
                    for c in range(NC):
                        pb=p[c]>0.4; gb=yv[c].numpy()>0.5; iou[c]+=(pb&gb).sum()/max(1,(pb|gb).sum())
            iou/=len(val_reals); e4=iou[:4].mean()
            if e4>best: best=e4; torch.save(net.state_dict(),f"{OUT}/seg2_best.pth")
            print(f"  [val] rol-IoU 4-kenar-ort={e4:.3f} (best {best:.3f}) "+" ".join(f"{CLASSES[c][:5]}={iou[c]:.2f}" for c in range(4)),flush=True)
        step+=1
    torch.save(net.state_dict(),f"{OUT}/seg2_unet.pth"); print(f"BİTTİ {time.time()-t0:.0f}s best4={best:.3f}",flush=True)
if __name__=="__main__": main()
