#!/usr/bin/env python3
"""SELF-FEEDING pseudo-etiket seti: harvested sahalarda 1024-model -> calib_from_pred -> SANITY-GATE.
Geçenlerden (res<0.4 + sanity-OK) clean_label_masks ile TEMİZ 7-kanal rol-maskesi geri-projekte ->
pseudo_data/<i>_img.jpg + <i>_mask.npz. RunPod eğitimi bunları 26 el-etiket + sentetik'e EKLER.
"""
import os, sys, glob, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from calib_train import auto_calib as AC
from calib_train.render2d_clean import UNet
from calib_train.auto_clean2d import sanity
from calib_train import auto_clean2d as AC2
HERE=os.path.dirname(os.path.abspath(__file__)); CB=os.path.join(HERE,"cand_big"); OUT=os.path.join(HERE,"pseudo_data")
os.makedirs(OUT,exist_ok=True); TW,TH=1024,576; RES_MAX=float(os.environ.get("RES_MAX","0.70"))
net=UNet(); net.load_state_dict(torch.load(os.path.join(HERE,"seg2_hr2.pth"),map_location="cpu")); net.eval()
@torch.no_grad()
def prob(img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None]
    return torch.sigmoid(net(x))[0].numpy()
if __name__=="__main__":
    pool=sorted(glob.glob(os.path.join(CB,"*__*__s*.jpg"))); pool=[p for p in pool if "_tmp_" not in p]
    kept=0; manifest=[]
    for p in pool:
        img=cv2.imread(p)
        if img is None: continue
        h,w=img.shape[:2]; rec=AC2.calibrate_frame(prob(img),w,h,img=img)
        if not rec.get("ok") or rec["fit"] is None or rec["fit"]>RES_MAX: continue
        sok,why=sanity(rec,w,h,img=img)
        if not sok: continue
        m=AC.clean_label_masks(rec,w,h)   # (7,h,w) temiz rol-maskesi
        # 1024'e küçült (yer kazan), maske nearest
        im1=cv2.resize(img,(TW,TH)); m1=np.stack([cv2.resize(m[c],(TW,TH),interpolation=cv2.INTER_NEAREST) for c in range(7)])
        cv2.imwrite(os.path.join(OUT,f"{kept:03d}_img.jpg"),im1,[cv2.IMWRITE_JPEG_QUALITY,90])
        np.savez_compressed(os.path.join(OUT,f"{kept:03d}_mask.npz"),m=m1)
        manifest.append((os.path.basename(p),round(rec["fit"],3))); kept+=1
        print(f"  + {os.path.basename(p)[:24]:24s} res={rec['fit']:.2f} {rec['camside']}",flush=True)
    import json; json.dump(manifest,open(os.path.join(OUT,"manifest.json"),"w"),ensure_ascii=False,indent=1)
    print(f"\nPSEUDO-ETİKET: {kept} saha -> pseudo_data/ (res<{RES_MAX} + sanity-OK)",flush=True)
