#!/usr/bin/env python3
"""KABUL TESTİ (token-BEDAVA): seg2_hr_far (clutter+uzak-ağırlık) vs seg2_hr2.
1) 20 PARLAK-no-solve kurtuldu mu  2) PARLAK-kontrol regresyon  3) ilk-6 warp sağ-üst skew (görsel).
İkisi de CLAHE'siz. seg2_hr2'ye DOKUNMAZ."""
import os,sys,json,numpy as np,cv2,time
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
HERE=os.path.dirname(os.path.abspath(__file__)); CT=os.path.dirname(HERE); ROOT=os.path.dirname(CT)
sys.path.insert(0,ROOT)
import torch
from calib_train.render2d_clean import UNet
from calib_train import auto_clean2d as AC2
from calib_train.night.four_panel import load, p_warp
MAN={r['idx']:r for r in json.load(open(f"{HERE}/probcache_v3/manifest.json"))}
FAR=os.path.join(CT,"seg2_hr_far.pth"); TW,TH=1024,576
def mk(p): n=UNet(); n.load_state_dict(torch.load(p,map_location="cpu")); n.eval(); return n
@torch.no_grad()
def prob(net,img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None]
    return torch.sigmoid(net(x))[0].numpy()
def solve(net,idx):
    e=MAN[idx]; im=cv2.imread(f"{CT}/cand_big/{e['file']}"); w,h=e['w'],e['h']; im=cv2.resize(im,(w,h))
    rec=AC2.calibrate_frame(prob(net,im),w,h,img=im)
    return rec, im, w, h
def main():
    if not os.path.exists(FAR): print(f"seg2_hr_far.pth YOK — guard indirmesini bekle"); return 1
    n2=mk(f"{CT}/seg2_hr2.pth"); nf=mk(FAR)
    bright_nosolve=[39,38,47,73,65,113,70,96,67,120,9,44,107,82,117,102,41,51,48,53]
    bright_ctrl=[1,3,5,6,7,11,52,90,99,44,64,26,34,55]   # hr2'nin çözdükleri (regresyon kontrolü) - bazıları overlap olabilir
    badgeom=[0,35,59]
    t0=time.time()
    print("=== 20 PARLAK-NO-SOLVE (far kurtardı mı) ===")
    rec_gain=0
    for idx in bright_nosolve:
        r2,_,_,_=solve(n2,idx); rf,_,_,_=solve(nf,idx)
        a=bool(r2.get('ok')); b=bool(rf.get('ok'))
        if b and not a: rec_gain+=1
        fl="  <== FAR KURTARDI" if b and not a else ("  (ikisi de yok)" if not b else "")
        print(f"idx{idx:3d} hr2={a} far={b}{fl}",flush=True)
    print("=== PARLAK-KONTROL (regresyon) ===")
    reg=0
    for idx in bright_ctrl:
        r2,_,_,_=solve(n2,idx); rf,_,_,_=solve(nf,idx)
        a=bool(r2.get('ok')); b=bool(rf.get('ok'))
        if a and not b: reg+=1; print(f"idx{idx:3d} hr2={a} far={b}  <== REGRESYON",flush=True)
    # ilk-6 warp hr2|far YAN-YANA (sağ-üst skew direkt karşılaştırma)
    print("=== ilk-6 warp hr2|far YAN-YANA — sağ-üst skew gözle ===")
    for idx in [1,12,26,34,55,64]:
        r2,im,w,h=solve(n2,idx); rf,_,_,_=solve(nf,idx)
        w2=p_warp(im,r2,w,h) if r2.get('ok') else np.full((300,500,3),40,np.uint8)
        wf=p_warp(im,rf,w,h) if rf.get('ok') else np.full((300,500,3),40,np.uint8)
        H=440; rs=lambda a:cv2.resize(a,(int(a.shape[1]*H/a.shape[0]),H))
        w2,wf=rs(w2),rs(wf)
        for a,t in ((w2,"hr2 (eski)"),(wf,"FAR (yeni)")):
            cv2.rectangle(a,(0,0),(170,30),(18,18,20),-1); cv2.putText(a,t,(8,22),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
        montage=np.hstack([w2,np.full((H,8,3),80,np.uint8),wf])
        cv2.imwrite(f"{CT}/cand/_FARWARP_{idx}.jpg",montage,[cv2.IMWRITE_JPEG_QUALITY,88])
        print(f"  idx{idx} -> _FARWARP_{idx}.jpg (sol=hr2 sağ=far)")
    print("="*56)
    print(f"SONUÇ ({time.time()-t0:.0f}s): parlak-no-solve KURTARILAN {rec_gain}/20 | kontrol REGRESYON {reg}")
    print("KARAR:", "far ÖNER (net kurtardı)" if rec_gain-reg>=3 else ("KARARSIZ — warp'ları gözle" if rec_gain>=reg else "hr2 TUT (far kaybettirdi)"))
if __name__=="__main__": sys.exit(main() or 0)
