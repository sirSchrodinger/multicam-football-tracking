"""hr3 (veya herhangi model) için prob-cache. ENV: MODEL, OUTDIR. Aynı UNet mimarisi (render2d_clean)."""
import os,sys,glob,numpy as np,cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","0"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from calib_train.render2d_clean import UNet
import json
HERE="calib_train"; CB=f"{HERE}/cand_big"
MODEL=os.environ.get("MODEL",f"{HERE}/seg2_hr3.pth"); OUT=os.environ.get("OUTDIR",f"{HERE}/night/probcache_v3")
os.makedirs(OUT,exist_ok=True)
TW,TH=1024,576; DEV="cuda" if torch.cuda.is_available() else "cpu"
net=UNet().to(DEV); net.load_state_dict(torch.load(MODEL,map_location=DEV)); net.eval()
@torch.no_grad()
def prob(img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None].to(DEV)
    return torch.sigmoid(net(x))[0].cpu().numpy().astype(np.float16)
# AYNI sıra: mevcut manifest'i kullan (idx eşleşsin)
man=json.load(open(f"{HERE}/night/probcache/manifest.json")) if os.path.exists(f"{HERE}/night/probcache/manifest.json") else None
import json
if man:
    for e in man:
        im=cv2.imread(f"{CB}/{e['file']}")
        if im is None: continue
        np.savez_compressed(f"{OUT}/{e['idx']:03d}.npz",prob=prob(im),h=e['h'],w=e['w'])
        if e['idx']%30==0: print(f"{e['idx']}/{len(man)}",flush=True)
    json.dump(man,open(f"{OUT}/manifest.json","w"),ensure_ascii=False,indent=1)
    print(f"CACHE v3: {len(man)} frame -> {OUT} (idx mevcut manifest ile eşleşik)")
