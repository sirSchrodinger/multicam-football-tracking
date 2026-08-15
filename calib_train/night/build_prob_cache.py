"""121 cand_big için seg-olasılık CACHE (tek GPU geçişi) -> ajanlar GPU'suz calib prototipler."""
import os,sys,glob,numpy as np,cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","0"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from calib_train.render2d_clean import UNet
HERE="calib_train"; CB=f"{HERE}/cand_big"; OUT=f"{HERE}/night/probcache"; os.makedirs(OUT,exist_ok=True)
TW,TH=1024,576; DEV="cuda" if torch.cuda.is_available() else "cpu"
net=UNet().to(DEV); net.load_state_dict(torch.load(f"{HERE}/seg2_hr2.pth",map_location=DEV)); net.eval()
@torch.no_grad()
def prob(img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None].to(DEV)
    return torch.sigmoid(net(x))[0].cpu().numpy().astype(np.float16)
pool=sorted(glob.glob(f"{CB}/*__*__s*.jpg")); pool=[p for p in pool if "_tmp_" not in p]
manifest=[]
for i,p in enumerate(pool):
    im=cv2.imread(p)
    if im is None: continue
    h,w=im.shape[:2]; np.savez_compressed(f"{OUT}/{i:03d}.npz",prob=prob(im),h=h,w=w)
    manifest.append({"idx":i,"file":os.path.basename(p),"h":h,"w":w})
    if i%30==0: print(f"{i}/{len(pool)}",flush=True)
import json; json.dump(manifest,open(f"{OUT}/manifest.json","w"),ensure_ascii=False,indent=1)
print(f"CACHE: {len(manifest)} frame -> {OUT}")
