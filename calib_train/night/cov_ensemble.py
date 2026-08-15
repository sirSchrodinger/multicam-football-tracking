#!/usr/bin/env python3
"""ENSEMBLE (union) kapsama: her sahada far ÖNCE dener (daha iyi geometri), çözemezse hr2 fallback.
-> far'ın +kurtarması AND hr2'nin kapsaması birleşir. Token-BEDAVA. CLAHE'siz. Per-idx kaydeder."""
import os,sys,json,numpy as np,cv2,time
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
HERE=os.path.dirname(os.path.abspath(__file__)); CT=os.path.dirname(HERE); ROOT=os.path.dirname(CT)
sys.path.insert(0,ROOT)
import torch
from calib_train.render2d_clean import UNet
from calib_train import auto_clean2d as AC2
MAN=json.load(open(f"{HERE}/probcache_v3/manifest.json")); TW,TH=1024,576
def mk(p): n=UNet(); n.load_state_dict(torch.load(p,map_location="cpu")); n.eval(); return n
@torch.no_grad()
def prob(net,img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None]
    return torch.sigmoid(net(x))[0].numpy()
def main():
    n2=mk(f"{CT}/seg2_hr2.pth"); nf=mk(f"{CT}/seg2_hr_far.pth"); rows=[]; t0=time.time()
    for i,e in enumerate(MAN):
        idx=e['idx']; im=cv2.imread(f"{CT}/cand_big/{e['file']}")
        if im is None: continue
        w,h=e['w'],e['h']; im=cv2.resize(im,(w,h))
        rf=AC2.calibrate_frame(prob(nf,im),w,h,img=im); a_far=bool(rf.get('ok'))
        r2=AC2.calibrate_frame(prob(n2,im),w,h,img=im); a_hr2=bool(r2.get('ok'))
        ens = a_far or a_hr2
        win = 'far' if a_far else ('hr2' if a_hr2 else 'none')
        rows.append({'idx':idx,'hr2':a_hr2,'far':a_far,'ens':ens,'win':win})
        if i%20==0: print(f"[{i+1}/{len(MAN)}] idx{idx} hr2={a_hr2} far={a_far} ens={ens} ({time.time()-t0:.0f}s)",flush=True)
    n=len(rows); o2=sum(r['hr2'] for r in rows); of=sum(r['far'] for r in rows); oe=sum(r['ens'] for r in rows)
    json.dump(rows,open(f"{HERE}/cov_ensemble_results.json","w"))
    print("="*58)
    print(f"TAM {n}: hr2={o2} ({100*o2/n:.0f}%) | far={of} ({100*of/n:.0f}%) | ENSEMBLE={oe} ({100*oe/n:.0f}%)  [{time.time()-t0:.0f}s]")
    print(f"ensemble vs hr2 NET: {oe-o2:+d} saha | far-only kazanım: {sum(1 for r in rows if r['far'] and not r['hr2'])} | hr2-only (fallback kurtardı): {sum(1 for r in rows if r['hr2'] and not r['far'])}")
if __name__=="__main__": sys.exit(main() or 0)
