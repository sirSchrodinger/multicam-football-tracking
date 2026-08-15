#!/usr/bin/env python3
"""ROL-AWARE detektör eğitimi: RGB -> 7 kanal (her çizginin ROLÜ). Tam otonom: çıkarımda
model hangi çizgi yakın-kale/uzak-kale/yakın-taç/uzak-taç diye DOĞRUDAN söyler.
Sentetik (3D köşe-kamera) + 23 gerçek saha (3 held-out). Çıktı: seg2_ckpt/seg2_unet.pth
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from calib_train.make_seg_data2 import synth_sample, real_sample, real_ok, ORDER, J, NC, CLASSES
TW=int(os.environ.get("TW","512")); TH=int(os.environ.get("TH","288")); BASE=int(os.environ.get("BASE","24"))
STEPS=int(os.environ.get("STEPS","4000")); BATCH=int(os.environ.get("BATCH","6")); NR=int(os.environ.get("NREAL","2"))
OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"seg2_ckpt"); os.makedirs(OUT,exist_ok=True)
VAL={"AvanosHaliSaha.jpg","KaynarcaAdaHaliSah.jpg","KucukcekmeceIdmanY.jpg"}
reals=[n for n in ORDER if n in J and real_ok(n)]; train_reals=[n for n in reals if n not in VAL]; val_reals=[n for n in reals if n in VAL]
print(f"gerçek {len(train_reals)}+{len(val_reals)} val | {NC} sınıf: {CLASSES}",flush=True)
def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True),nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True))
class UNet(nn.Module):
    def __init__(s,nin=3,no=NC,b=24):
        super().__init__(); s.d1=cbr(nin,b);s.d2=cbr(b,b*2);s.d3=cbr(b*2,b*4);s.d4=cbr(b*4,b*8);s.p=nn.MaxPool2d(2)
        s.u3=nn.ConvTranspose2d(b*8,b*4,2,2);s.c3=cbr(b*8,b*4);s.u2=nn.ConvTranspose2d(b*4,b*2,2,2);s.c2=cbr(b*4,b*2)
        s.u1=nn.ConvTranspose2d(b*2,b,2,2);s.c1=cbr(b*2,b);s.h=nn.Conv2d(b,no,1)
    def forward(s,x):
        c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2));c4=s.d4(s.p(c3))
        x=s.c3(torch.cat([s.u3(c4),c3],1));x=s.c2(torch.cat([s.u2(x),c2],1));x=s.c1(torch.cat([s.u1(x),c1],1));return s.h(x)
def aug(img,m):
    img=np.clip(img.astype(np.float32)*np.random.uniform(0.75,1.2)+np.random.uniform(-12,12),0,255).astype(np.uint8); return img,m
def to_t(img,m):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)
    y=torch.from_numpy(np.stack([cv2.resize(m[c],(TW,TH)) for c in range(NC)]).astype(np.float32)/255.)
    return x,y
def get_batch(step):
    xs,ys=[],[]
    for i in range(BATCH-NR):
        img,m=synth_sample(step*BATCH+i); x,y=to_t(img,m); xs.append(x);ys.append(y)
    for _ in range(NR):
        n=train_reals[np.random.randint(len(train_reals))]; img,m=real_sample(n); img,m=aug(img,m); x,y=to_t(img,m); xs.append(x);ys.append(y)
    return torch.stack(xs),torch.stack(ys)
def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev,flush=True)
    net=UNet(b=BASE).to(dev); opt=torch.optim.Adam(net.parameters(),1e-3); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS); t0=time.time()
    for step in range(STEPS):
        x,y=get_batch(step); x,y=x.to(dev),y.to(dev); pred=net(x)
        w=1.0+35.0*y; loss=(w*F.binary_cross_entropy_with_logits(pred,y,reduction="none")).mean()
        opt.zero_grad();loss.backward();opt.step();sch.step()
        if step%100==0: print(f"step {step}/{STEPS} loss {loss.item():.4f} {(time.time()-t0)/max(1,step+1)*1000:.0f}ms/it",flush=True)
        if step%500==0 and step>0:
            torch.save(net.state_dict(),f"{OUT}/seg2_unet.pth"); net.eval()
            iou=np.zeros(NC)
            with torch.no_grad():
                for n in val_reals:
                    img,m=real_sample(n); x,y=to_t(img,m); p=torch.sigmoid(net(x[None].to(dev)))[0].cpu().numpy()
                    for c in range(NC):
                        pb=p[c]>0.4; gb=y[c].numpy()>0.5
                        iou[c]+=(pb&gb).sum()/max(1,(pb|gb).sum())
            iou/=len(val_reals)
            print(f"  [val] rol-IoU: "+" ".join(f"{CLASSES[c][:5]}={iou[c]:.2f}" for c in range(NC))+f"  4-kenar-ort={iou[:4].mean():.3f}",flush=True); net.train()
    torch.save(net.state_dict(),f"{OUT}/seg2_unet.pth"); print(f"BİTTİ {time.time()-t0:.0f}s",flush=True)
if __name__=="__main__": main()
