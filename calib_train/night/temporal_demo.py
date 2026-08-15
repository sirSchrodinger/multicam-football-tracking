#!/usr/bin/env python3
"""TEMPORAL agregasyon kanıtı (path forward): tek-kare seg gürültülü/eksik far-çizgi verir.
N-kare temporal-MEDYAN (statik çizgi, hareketli oyuncu) far-çizgiyi denoise eder -> daha temiz calib.
Çankaya cam2.mp4 (yerel): tek-kare(ler) vs 30-kare-medyan -> far-çizgi confident-pixel + calib residual.
ENV: VID=raw/cankaya_cam2.mp4 N=30
"""
import os, sys, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from calib_train.render2d_clean import UNet
from calib_train.auto_clean2d import sanity
from calib_train import auto_calib as AC
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VID=os.environ.get("VID","raw/cankaya_cam2.mp4"); N=int(os.environ.get("N","30")); TW,TH=1024,576
DEV="cuda" if torch.cuda.is_available() else "cpu"
net=UNet().to(DEV); net.load_state_dict(torch.load(os.path.join(HERE,"seg2_hr2.pth"),map_location=DEV)); net.eval()
@torch.no_grad()
def prob(img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None].to(DEV)
    return torch.sigmoid(net(x))[0].cpu().numpy()
def farpix(pr,thr=0.5): return int((pr[1]>thr).sum()+(pr[3]>thr).sum())   # goalF+touchF confident pixels
def calib_res(pr,w,h,img):
    rec=AC.calib_from_pred(pr,w,h)
    if not rec.get("ok"): return None,"calib-yok"
    ok,why=sanity(rec,w,h,img=img); return rec["fit"],("PASS" if ok else why)
def main():
    cap=cv2.VideoCapture(VID); nf=int(cap.get(7)); w=int(cap.get(3)); h=int(cap.get(4))
    print(f"{VID}: {nf} kare {w}x{h}, {N} örnek")
    idxs=np.linspace(int(nf*0.1),int(nf*0.9),N).astype(int)
    probs=[]; singles=[]
    for k,fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(fi)); ok,fr=cap.read()
        if not ok: continue
        pr=prob(fr); probs.append(pr)
        if k==0: ref_img=fr.copy()
        singles.append((farpix(pr),)+calib_res(pr,w,h,fr))
    cap.release()
    probs=np.stack(probs); med=np.median(probs,0)   # temporal-medyan 7-kanal
    # tek-kare ort
    sf=np.array([s[0] for s in singles]); sres=[s[1] for s in singles if s[1] is not None]
    spass=sum(1 for s in singles if s[2]=="PASS")
    print(f"\nTEK-KARE: far-pixel ort {sf.mean():.0f} (min {sf.min()}, max {sf.max()}) | "
          f"calib-res ort {np.mean(sres):.2f}m | PASS {spass}/{len(singles)}")
    mf=farpix(med); mres,mwhy=calib_res(med,w,h,ref_img)
    print(f"TEMPORAL-MEDYAN({N}): far-pixel {mf} | calib-res {mres if mres is None else round(mres,2)}m [{mwhy}]")
    print(f"\nfar-pixel kazanç: medyan {mf} vs tek-kare-ort {sf.mean():.0f} = x{mf/max(1,sf.mean()):.2f}")
if __name__=="__main__": main()
