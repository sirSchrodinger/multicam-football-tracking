#!/usr/bin/env python3
"""TAM 121 solve-oranı: seg2_hr2 vs seg2_hr_far (clutter+uzak-ağırlık). Token-BEDAVA, agent YOK.
Per-idx kaydeder -> parlak-no-solve kurtarma + regresyon tam dökümü. CLAHE'siz (ikisi de)."""
import os,sys,json,numpy as np,cv2,time
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
HERE=os.path.dirname(os.path.abspath(__file__)); CT=os.path.dirname(HERE); ROOT=os.path.dirname(CT)
sys.path.insert(0,ROOT)
import torch
from calib_train.render2d_clean import UNet
from calib_train import auto_clean2d as AC2
MAN=json.load(open(f"{HERE}/probcache_v3/manifest.json"))
TW,TH=1024,576; FAR=f"{CT}/seg2_hr_far.pth"
def mk(p): n=UNet(); n.load_state_dict(torch.load(p,map_location="cpu")); n.eval(); return n
@torch.no_grad()
def prob(net,img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None]
    return torch.sigmoid(net(x))[0].numpy()
def main():
    if not os.path.exists(FAR): print("seg2_hr_far.pth YOK"); return 1
    n2=mk(f"{CT}/seg2_hr2.pth"); nf=mk(FAR); rows=[]; t0=time.time()
    for i,e in enumerate(MAN):
        idx=e['idx']; im=cv2.imread(f"{CT}/cand_big/{e['file']}")
        if im is None: continue
        w,h=e['w'],e['h']; im=cv2.resize(im,(w,h))
        a=bool(AC2.calibrate_frame(prob(n2,im),w,h,img=im).get('ok'))
        b=bool(AC2.calibrate_frame(prob(nf,im),w,h,img=im).get('ok'))
        rows.append({'idx':idx,'hr2':a,'far':b})
        if i%20==0: print(f"[{i+1}/{len(MAN)}] idx{idx} hr2={a} far={b} ({time.time()-t0:.0f}s)",flush=True)
    o2=sum(r['hr2'] for r in rows); of=sum(r['far'] for r in rows); n=len(rows)
    gain=[r['idx'] for r in rows if r['far'] and not r['hr2']]
    lost=[r['idx'] for r in rows if r['hr2'] and not r['far']]
    json.dump(rows,open(f"{HERE}/cov_far_results.json","w"))
    print("="*56)
    print(f"TAM {n}: hr2 solve={o2} ({100*o2/n:.0f}%) | far solve={of} ({100*of/n:.0f}%)  [{time.time()-t0:.0f}s]")
    print(f"far KURTARDI ({len(gain)}): {gain}")
    print(f"far KAYBETTİRDİ ({len(lost)}): {lost}")
    print(f"NET: {of-o2:+d} saha")
    print("KARAR:", "far ÜRÜN-MODELİ öner" if of-o2>=5 else ("hibrit/karışık — incele" if of>=o2 else "hr2 TUT"))
if __name__=="__main__": sys.exit(main() or 0)
