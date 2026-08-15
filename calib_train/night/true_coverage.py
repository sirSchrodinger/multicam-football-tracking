#!/usr/bin/env python3
"""GERÇEK kullanılabilir-2D kapsama = SANITY-PASS oranı (residual<0.45 DEĞİL — o yanıltıcı, bowtie sayıyor).
Her cand_big sahası: calib_from_pred + sanity(img, kara-kare reddi dahil) -> pass?
ENV: MODEL TW TH BASE TTA(none|tiled) TAG. Çıktı: stdout + cand/_truecov_<TAG>.json
"""
import os, sys, glob, json, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, torch.nn as nn
from calib_train import auto_calib as AC
from calib_train.auto_clean2d import sanity
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CB=os.path.join(HERE,"cand_big"); NC=7
MODEL=os.environ.get("MODEL",os.path.join(HERE,"seg2_hr2.pth")); BASE=int(os.environ.get("BASE","32"))
TW=int(os.environ.get("TW","1024")); TH=int(os.environ.get("TH","576")); TTA=os.environ.get("TTA","none"); TAG=os.environ.get("TAG","truecov")
DEV="cuda" if torch.cuda.is_available() else "cpu"
def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True),nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True))
class UNet(nn.Module):
    def __init__(s,nin=3,no=NC,b=BASE):
        super().__init__(); s.d1=cbr(nin,b);s.d2=cbr(b,b*2);s.d3=cbr(b*2,b*4);s.d4=cbr(b*4,b*8);s.p=nn.MaxPool2d(2)
        s.u3=nn.ConvTranspose2d(b*8,b*4,2,2);s.c3=cbr(b*8,b*4);s.u2=nn.ConvTranspose2d(b*4,b*2,2,2);s.c2=cbr(b*4,b*2)
        s.u1=nn.ConvTranspose2d(b*2,b,2,2);s.c1=cbr(b*2,b);s.h=nn.Conv2d(b,no,1)
    def forward(s,x):
        c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2));c4=s.d4(s.p(c3))
        x=s.c3(torch.cat([s.u3(c4),c3],1));x=s.c2(torch.cat([s.u2(x),c2],1));x=s.c1(torch.cat([s.u1(x),c1],1));return s.h(x)
net=UNet().to(DEV); net.load_state_dict(torch.load(MODEL,map_location=DEV)); net.eval()
@torch.no_grad()
def _fwd(im):
    x=torch.from_numpy(cv2.resize(im,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None].to(DEV)
    return torch.sigmoid(net(x))[0].cpu().numpy()
@torch.no_grad()
def prob(img):
    h,w=img.shape[:2]; full=_fwd(img)
    if TTA=="none": return full
    acc=cv2.resize(full.transpose(1,2,0),(w,h)).transpose(2,0,1).copy()
    crops=[(0,0,w,int(h*0.6)),(0,0,int(w*0.6),h),(int(w*0.4),0,w,h),(int(w*0.2),0,int(w*0.85),int(h*0.7))]
    for x0,y0,x1,y1 in crops:
        sub=img[y0:y1,x0:x1]
        if sub.shape[0]<40 or sub.shape[1]<40: continue
        pc=_fwd(sub); pc=cv2.resize(pc.transpose(1,2,0),(x1-x0,y1-y0)).transpose(2,0,1)
        acc[:,y0:y1,x0:x1]=np.maximum(acc[:,y0:y1,x0:x1],pc)
    return acc
def main():
    pool=sorted(glob.glob(os.path.join(CB,"*__*__s*.jpg"))); pool=[p for p in pool if "_tmp_" not in p]
    npass=0; rows=[]; fits=[]
    for i,p in enumerate(pool):
        img=cv2.imread(p)
        if img is None: continue
        h,w=img.shape[:2]; rec=AC.calib_from_pred(prob(img),w,h)
        if not rec.get("ok"): rows.append((os.path.basename(p),None,rec.get("reason"))); continue
        ok,why=sanity(rec,w,h,img=img)
        if ok: npass+=1; fits.append(rec["fit"]); rows.append((os.path.basename(p),round(rec["fit"],3),"PASS"))
        else: rows.append((os.path.basename(p),round(rec["fit"],3),why))
        if i%40==0: print(f"  {i}/{len(pool)} pass={npass}",flush=True)
    n=len(pool); cov=100*npass/max(1,n)
    print(f"\n=== {TAG} GERÇEK-KAPSAMA (sanity-PASS) {npass}/{n} = %{cov:.1f} | pass-median-res {np.median(fits):.3f}m" if fits else f"=== {TAG} {npass}/{n}")
    json.dump({"tag":TAG,"model":os.path.basename(MODEL),"tta":TTA,"n":n,"pass":npass,"true_coverage_pct":round(cov,1),
               "pass_median_res":round(float(np.median(fits)),3) if fits else None,"rows":rows},
              open(os.path.join(HERE,"cand",f"_truecov_{TAG}.json"),"w"),ensure_ascii=False,indent=1)
    print(f"-> cand/_truecov_{TAG}.json")
if __name__=="__main__": main()
