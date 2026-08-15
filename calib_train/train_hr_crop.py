#!/usr/bin/env python3
"""1024-PLAIN weighted-BCE seg2 + pseudo_data (self-feeding) + CROP-AWARE aug.
Neden: self-feeding 512'de plato yaptı (sadece BAŞARILI sahalardan pseudo = confirmation-bias).
Asıl fixable failure = ZAYIF FAR-ÇİZGİ. Çözüm: model'i full-frame + far-strip CROP'larda eğit ->
tiled-inference (eskiden BUST: full-frame model crop'u kaldıramıyordu) artık çalışır = far-recall.
Karışım: synth(PSYNTH) + pseudo(PPSEUDO) + real(kalan); her örnek PCROP olasılıkla far-bias crop.
ENV: TW TH BASE STEPS BATCH WORKERS PSYNTH PPSEUDO PCROP. Çıktı: seg2_ckpt/seg2_best.pth + seg2_unet.pth.
"""
import sys, os, time, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from calib_train.make_seg_data2 import synth_sample, real_sample, real_ok, ORDER, J, NC, CLASSES
HERE=os.path.dirname(os.path.abspath(__file__))
TW=int(os.environ.get("TW","1024")); TH=int(os.environ.get("TH","576")); BASE=int(os.environ.get("BASE","32"))
STEPS=int(os.environ.get("STEPS","8000")); BATCH=int(os.environ.get("BATCH","6")); WORKERS=int(os.environ.get("WORKERS","12"))
PSYNTH=float(os.environ.get("PSYNTH","0.50")); PPSEUDO=float(os.environ.get("PPSEUDO","0.22")); PCROP=float(os.environ.get("PCROP","0.35"))
OUT=os.path.join(HERE,"seg2_ckpt"); os.makedirs(OUT,exist_ok=True)
VAL={"AvanosHaliSaha.jpg","KaynarcaAdaHaliSah.jpg","KucukcekmeceIdmanY.jpg"}
reals=[n for n in ORDER if n in J and real_ok(n)]; train_reals=[n for n in reals if n not in VAL]; val_reals=[n for n in reals if n in VAL]
PSD=os.path.join(HERE,"pseudo_data"); pseudo_imgs=sorted(glob.glob(os.path.join(PSD,"*_img.jpg")))
print(f"gerçek {len(train_reals)}+{len(val_reals)}val | pseudo {len(pseudo_imgs)} | {NC} sınıf | {TW}x{TH} BASE{BASE} BATCH{BATCH} PCROP{PCROP}",flush=True)
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
def load_pseudo(p):
    img=cv2.imread(p); d=np.load(p.replace("_img.jpg","_mask.npz")); m=d["m"]   # (7,Hp,Wp) uint8
    return img,m
def far_crop(img,m,rng):
    """Far-bias rastgele crop: üst/uzak bölgeyi büyüt (model crop'u öğrensin -> tiled inference çalışsın)."""
    h,w=img.shape[:2]
    # crop yüksekliği görüntünün %30-75'i, üst-bias (y0 küçük); genişlik %45-100
    ch=int(h*rng.uniform(0.30,0.75)); cw=int(w*rng.uniform(0.45,1.0))
    y0=int(rng.uniform(0,max(1,h-ch))*rng.uniform(0,0.6))   # üst-bias
    x0=int(rng.uniform(0,max(1,w-cw)))
    y1=min(h,y0+ch); x1=min(w,x0+cw)
    ic=img[y0:y1,x0:x1]; mc=m[:,y0:y1,x0:x1]
    if ic.shape[0]<24 or ic.shape[1]<24: return img,m
    return ic,np.ascontiguousarray(mc)
class TrainDS(Dataset):
    def __len__(s): return STEPS*BATCH
    def __getitem__(s,idx):
        rng=np.random.RandomState(idx*2654435761 % (2**31)); r=rng.rand()
        if r<PSYNTH:
            img,m=synth_sample(idx)
        elif r<PSYNTH+PPSEUDO and pseudo_imgs:
            img,m=load_pseudo(pseudo_imgs[rng.randint(len(pseudo_imgs))]); img=aug(img)
        else:
            n=train_reals[rng.randint(len(train_reals))]; img,m=real_sample(n); img=aug(img)
        if rng.rand()<PCROP:
            img,m=far_crop(img,m,rng)
        return to_t(img,m)
def main():
    dev="cuda" if torch.cuda.is_available() else "cpu"; print("device",dev,flush=True)
    dl=DataLoader(TrainDS(),batch_size=BATCH,num_workers=WORKERS,pin_memory=True,persistent_workers=True,prefetch_factor=4,drop_last=True)
    net=UNet().to(dev); opt=torch.optim.Adam(net.parameters(),1e-3); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS)
    t0=time.time(); step=0; best=0.0
    for x,y in dl:
        if step>=STEPS: break
        net.train(); x,y=x.to(dev,non_blocking=True),y.to(dev,non_blocking=True); pred=net(x)
        w=1.0+35.0*y; loss=(w*F.binary_cross_entropy_with_logits(pred,y,reduction="none")).mean()
        opt.zero_grad();loss.backward();opt.step();sch.step()
        if step%100==0: print(f"step {step}/{STEPS} loss {loss.item():.4f} {(time.time()-t0)/max(1,step+1)*1000:.0f}ms/it",flush=True)
        if step%500==0 and step>0:
            torch.save(net.state_dict(),f"{OUT}/seg2_unet.pth"); net.eval(); iou=np.zeros(NC)
            with torch.no_grad():
                for n in val_reals:
                    img,m=real_sample(n); xv,yv=to_t(img,m); p=torch.sigmoid(net(xv[None].to(dev)))[0].cpu().numpy()
                    for c in range(NC):
                        pb=p[c]>0.4; gb=yv[c].numpy()>0.5; iou[c]+=(pb&gb).sum()/max(1,(pb|gb).sum())
            iou/=len(val_reals); e4=iou[:4].mean()
            if e4>best: best=e4; torch.save(net.state_dict(),f"{OUT}/seg2_best.pth")
            print(f"  [val] rol-IoU 4-kenar-ort={e4:.3f} (best {best:.3f}) "+" ".join(f"{CLASSES[c][:5]}={iou[c]:.2f}" for c in range(4)),flush=True)
        step+=1
    torch.save(net.state_dict(),f"{OUT}/seg2_unet.pth"); print(f"BİTTİ {time.time()-t0:.0f}s best4={best:.3f}",flush=True)
if __name__=="__main__": main()
