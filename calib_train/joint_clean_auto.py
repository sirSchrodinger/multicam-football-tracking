#!/usr/bin/env python3
"""OTONOM temiz top-down: çizgiler 1024-MODEL'den (el-etiket YOK) -> calib_from_pred -> SANITY-GATE
-> joint_clean.clean_topdown. Geçen sahalar render, elenenler 'KAYIK' damgası. Çıktı: cand/_JOINT_AUTO.jpg
"""
import os, sys, glob, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from calib_train import auto_calib as AC
from calib_train.render2d_clean import UNet
from calib_train.joint_clean import clean_topdown, L, Wp, S, MG
from calib_train.auto_clean2d import sanity
HERE=os.path.dirname(os.path.abspath(__file__)); LF=os.path.join(HERE,"label_frames")
import json; ORDER=json.load(open(os.path.join(HERE,"frames.json")))
TW,TH=1024,576
net=UNet(); net.load_state_dict(torch.load(os.path.join(HERE,"seg2_hr2.pth"),map_location="cpu")); net.eval()
@torch.no_grad()
def prob(img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None]
    return torch.sigmoid(net(x))[0].numpy()
def blank(msg):
    o=np.full((int((Wp+2*MG)*S),int((L+2*MG)*S),3),26,np.uint8); cv2.putText(o,msg,(20,o.shape[0]//2),cv2.FONT_HERSHEY_SIMPLEX,0.9,(60,90,235),2); return o
if __name__=="__main__":
    venues=[n for n in ORDER if os.path.exists(os.path.join(LF,n))]
    tiles=[]; ok=0
    for n in venues:
        img=cv2.imread(os.path.join(LF,n)); h,w=img.shape[:2]
        rec=AC.calib_from_pred(prob(img),w,h)
        if not rec.get("ok"): o=blank(f"{n[:14]} CIZGI YOK"); st="cizgi-yok"
        else:
            sok,why=sanity(rec,w,h)
            if not sok: o=blank(f"{n[:14]} KAYIK"); st=f"elendi({why[:18]})"
            else:
                o=clean_topdown(img,rec["k1"],rec["k2"],np.array(rec["H"],float),rec["camside"]); ok+=1
                cv2.putText(o,f"{n.split('.')[0][:14]} OTONOM res={rec['fit']:.2f}m",(8,o.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,230,255),2); st=f"OK {rec['fit']:.2f}"
        tiles.append(o); print(f"  {n[:20]:20s} {st}",flush=True)
    print(f"\nOTONOM render: {ok}/{len(venues)} saha geçti",flush=True)
    cols=3; rows=(len(tiles)+cols-1)//cols; th,tw=tiles[0].shape[:2]
    sheet=np.full((rows*(th+6),cols*(tw+6),3),18,np.uint8)
    for i,t in enumerate(tiles):
        rr,cc=divmod(i,cols); sheet[rr*(th+6):rr*(th+6)+th,cc*(tw+6):cc*(tw+6)+tw]=cv2.resize(t,(tw,th))
    cv2.imwrite(os.path.join(HERE,"cand","_JOINT_AUTO.jpg"),sheet); print("-> cand/_JOINT_AUTO.jpg",flush=True)
