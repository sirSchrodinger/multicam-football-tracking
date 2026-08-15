#!/usr/bin/env python3
"""Çizgi-segmentasyon eğitimi: RGB(ham,eğri) -> 2 kanal (sınır / iç çizgiler).
Sentetik (on-the-fly) + 23 gerçek saha (3 held-out doğrulama). Lokal GPU.
Çıktı: seg_ckpt/seg_unet.pth + seg_ckpt/val_*.jpg
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from calib_train.make_seg_data import synth_sample, real_sample, real_ok, ORDER, J

TW,TH=512,288; STEPS=int(os.environ.get("STEPS","3000")); BATCH=int(os.environ.get("BATCH","6"))
NR=int(os.environ.get("NREAL","2"))   # batch içinde gerçek sayısı
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"seg_ckpt"); os.makedirs(OUT,exist_ok=True)
VAL={"AvanosHaliSaha.jpg","KaynarcaAdaHaliSah.jpg","KucukcekmeceIdmanY.jpg"}
reals=[n for n in ORDER if n in J and real_ok(n)]
train_reals=[n for n in reals if n not in VAL]; val_reals=[n for n in reals if n in VAL]
print(f"gerçek: {len(train_reals)} train + {len(val_reals)} val",flush=True)

def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True),
                                   nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True))
class UNet(nn.Module):
    def __init__(s,nin=3,no=2,b=24):
        super().__init__(); s.d1=cbr(nin,b);s.d2=cbr(b,b*2);s.d3=cbr(b*2,b*4);s.d4=cbr(b*4,b*8)
        s.p=nn.MaxPool2d(2)
        s.u3=nn.ConvTranspose2d(b*8,b*4,2,2);s.c3=cbr(b*8,b*4)
        s.u2=nn.ConvTranspose2d(b*4,b*2,2,2);s.c2=cbr(b*4,b*2)
        s.u1=nn.ConvTranspose2d(b*2,b,2,2);s.c1=cbr(b*2,b); s.h=nn.Conv2d(b,no,1)
    def forward(s,x):
        c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2));c4=s.d4(s.p(c3))
        x=s.c3(torch.cat([s.u3(c4),c3],1));x=s.c2(torch.cat([s.u2(x),c2],1));x=s.c1(torch.cat([s.u1(x),c1],1))
        return s.h(x)

def aug(img,m):  # hflip + parlaklık
    if np.random.rand()<0.5: img=img[:,::-1].copy(); m=m[:,:,::-1].copy()
    img=np.clip(img.astype(np.float32)*np.random.uniform(0.75,1.2)+np.random.uniform(-12,12),0,255).astype(np.uint8)
    return img,m
def to_t(img,m):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)
    y=torch.from_numpy(np.stack([cv2.resize(m[c],(TW,TH)) for c in range(2)]).astype(np.float32)/255.)
    return x,y
def get_batch(step):
    xs,ys=[],[]
    for i in range(BATCH-NR):
        img,m=synth_sample(step*BATCH+i); x,y=to_t(img,m); xs.append(x);ys.append(y)
    for _ in range(NR):
        n=train_reals[np.random.randint(len(train_reals))]; img,m=real_sample(n); img,m=aug(img,m)
        x,y=to_t(img,m); xs.append(x);ys.append(y)
    return torch.stack(xs),torch.stack(ys)

def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev,flush=True)
    net=UNet().to(dev); opt=torch.optim.Adam(net.parameters(),1e-3)
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS); t0=time.time()
    for step in range(STEPS):
        x,y=get_batch(step); x,y=x.to(dev),y.to(dev)
        pred=net(x)
        w=1.0+30.0*y                                  # çizgi pikselleri seyrek -> ağırlık
        loss=(w*F.binary_cross_entropy_with_logits(pred,y,reduction="none")).mean()
        opt.zero_grad();loss.backward();opt.step();sch.step()
        if step%100==0:
            print(f"step {step}/{STEPS} loss {loss.item():.4f} {(time.time()-t0)/max(1,step+1)*1000:.0f}ms/it",flush=True)
        if step%500==0 and step>0:
            torch.save(net.state_dict(),f"{OUT}/seg_unet.pth")
            net.eval(); ious=[]
            with torch.no_grad():
                for n in val_reals:
                    img,m=real_sample(n); x,y=to_t(img,m); p=torch.sigmoid(net(x[None].to(dev)))[0].cpu().numpy()
                    pb=(p[0]>0.4); gb=(y[0].numpy()>0.5)
                    iou=(pb&gb).sum()/max(1,(pb|gb).sum()); ious.append(iou)
                    ov=cv2.resize(img,(TW,TH)).copy(); ov[p[0]>0.4]=(0,0,255); ov[p[1]>0.4]=(0,255,0)
                    cv2.imwrite(f"{OUT}/val_{n.split('.')[0]}.jpg",ov)
            print(f"  [val] sınır-IoU {np.mean(ious):.3f}",flush=True); net.train()
    torch.save(net.state_dict(),f"{OUT}/seg_unet.pth"); print(f"BİTTİ {time.time()-t0:.0f}s",flush=True)

if __name__=="__main__": main()
