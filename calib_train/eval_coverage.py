#!/usr/bin/env python3
"""ÇOK-ÇÖZÜNÜRLÜK KAPSAMA ölçer: bir seg2 ckpt'i bir havuzda gate'le -> kabul-% + median residual.
512(BASE24) vs 768-plain(BASE32) vs 768-improved(BASE32+CLAHE) karşılaştırması için.
ENV: MODEL=ckpt BASE=24/32 TW=512/768 TH PRE=none/clahe POOL=glob  CPU'ya zorlanabilir (DEV=cpu).
Çıktı: stdout özet + cand/_cov_<tag>.json
"""
import sys, os, glob, json, numpy as np, cv2, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calib_train import auto_calib as AC
HERE=os.path.dirname(os.path.abspath(__file__)); NC=7
MODEL=os.environ.get("MODEL",os.path.join(HERE,"seg2_ckpt","seg2_unet.pth"))
BASE=int(os.environ.get("BASE","24")); TW=int(os.environ.get("TW","512")); TH=int(os.environ.get("TH","288"))
PRE=os.environ.get("PRE","none"); TAU=float(os.environ.get("TAU","0.45")); TAG=os.environ.get("TAG","m")
DEV="cpu" if os.environ.get("DEV")=="cpu" or not torch.cuda.is_available() else "cuda"
def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True),nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True))
class UNet(nn.Module):
    def __init__(s,nin=3,no=NC,b=BASE):
        super().__init__(); s.d1=cbr(nin,b);s.d2=cbr(b,b*2);s.d3=cbr(b*2,b*4);s.d4=cbr(b*4,b*8);s.p=nn.MaxPool2d(2)
        s.u3=nn.ConvTranspose2d(b*8,b*4,2,2);s.c3=cbr(b*8,b*4);s.u2=nn.ConvTranspose2d(b*4,b*2,2,2);s.c2=cbr(b*4,b*2)
        s.u1=nn.ConvTranspose2d(b*2,b,2,2);s.c1=cbr(b*2,b);s.h=nn.Conv2d(b,no,1)
    def forward(s,x):
        c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2));c4=s.d4(s.p(c3))
        x=s.c3(torch.cat([s.u3(c4),c3],1));x=s.c2(torch.cat([s.u2(x),c2],1));x=s.c1(torch.cat([s.u1(x),c1],1));return s.h(x)
def clahe(img):
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB); cl=cv2.createCLAHE(2.5,(8,8)); lab[...,0]=cl.apply(lab[...,0]); return cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
net=UNet().to(DEV); net.load_state_dict(torch.load(MODEL,map_location=DEV)); net.eval()
TTA=os.environ.get("TTA","none")   # none | tiled  (flip YOK: yatay flip rol-kanallarını bozar goalN<->goalF)
@torch.no_grad()
def _fwd(im):
    if PRE=="clahe": im=clahe(im)
    x=torch.from_numpy(cv2.resize(im,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None].to(DEV)
    return torch.sigmoid(net(x))[0].cpu().numpy()   # (NC,TH,TW)
@torch.no_grad()
def prob(img):
    h,w=img.shape[:2]; full=_fwd(img)               # tam kare (NC,TH,TW)
    if TTA=="none": return full
    acc=cv2.resize(full.transpose(1,2,0),(w,h)).transpose(2,0,1).copy()   # (NC,h,w) tam-res birikim
    if TTA=="tiled":   # 2x2 + üst-şerit (uzak çizgi) overlapping crop -> yüksek-efektif-çözünürlük
        crops=[(0,0,w,int(h*0.6)),(0,0,int(w*0.6),h),(int(w*0.4),0,w,h),(int(w*0.2),0,int(w*0.85),int(h*0.7))]
        for x0,y0,x1,y1 in crops:
            sub=img[y0:y1,x0:x1];
            if sub.shape[0]<40 or sub.shape[1]<40: continue
            pc=_fwd(sub); pc=cv2.resize(pc.transpose(1,2,0),(x1-x0,y1-y0)).transpose(2,0,1)
            acc[:,y0:y1,x0:x1]=np.maximum(acc[:,y0:y1,x0:x1],pc)
    return acc
def main():
    pool=sorted(glob.glob(os.environ.get("POOL",os.path.join(HERE,"cand_big","*.jpg"))))
    pool=[p for p in pool if "_tmp_" not in p]
    print(f"MODEL={os.path.basename(MODEL)} BASE{BASE} {TW}x{TH} PRE={PRE} havuz={len(pool)} dev={DEV}",flush=True)
    accs=[]; fits=[]; nofar=0; rows=[]
    for i,p in enumerate(pool):
        img=cv2.imread(p)
        if img is None: continue
        r=AC.calib_from_pred(prob(img),img.shape[1],img.shape[0],thr=float(os.environ.get("THR","0.5")),minpts=int(os.environ.get("MINPTS","25")))
        ok=bool(r.get("ok") and r["fit"] is not None and r["fit"]<TAU)
        if ok: accs.append(p); fits.append(r["fit"])
        elif r.get("reason","").startswith(("goalF","goalN","touch")): nofar+=1
        rows.append((os.path.basename(p), r.get("fit"), r.get("reason")))
        if i%50==0: print(f"  {i}/{len(pool)} kabul={len(accs)}",flush=True)
    n=len(pool); cov=100*len(accs)/max(1,n)
    print(f"\n=== {TAG} === KAPSAMA {len(accs)}/{n} = %{cov:.1f} | median-res {np.median(fits):.3f}m | çizgi-bulunamadı {nofar}" if fits else f"=== {TAG} === KAPSAMA 0")
    json.dump({"tag":TAG,"model":os.path.basename(MODEL),"res":f"{TW}x{TH}","pre":PRE,"n":n,"accept":len(accs),
               "coverage_pct":round(cov,1),"median_res":round(float(np.median(fits)),3) if fits else None,
               "rows":[(a,round(b,3) if b else None,c) for a,b,c in rows]},
              open(os.path.join(HERE,"cand",f"_cov_{TAG}.json"),"w"),ensure_ascii=False,indent=1)
    print(f"-> cand/_cov_{TAG}.json",flush=True)
if __name__=="__main__": main()
