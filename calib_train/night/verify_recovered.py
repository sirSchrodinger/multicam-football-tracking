#!/usr/bin/env python3
"""GÖRSEL DOĞRULAMA: multicand-recovered sahalar GERÇEKTEN doğru mu, yoksa gate-overfit mi?
Her recovered saha: multicand calib -> kalibre saha-çizgilerini frame'e GERİ-PROJEKTE (clean_label_masks) +
2D radar. Çizgiler gerçek beyaz çizgilere oturuyorsa = genuine. -> cand/_VERIFY_RECOVERED.jpg
"""
import os, sys, json, glob, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","0")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from calib_train.render2d_clean import UNet
from calib_train.auto_clean2d import sanity, project_feet, draw_2d
from calib_train.multicand_calib import calib_multicand
from calib_train import auto_calib as AC
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CB=os.path.join(HERE,"cand_big")
TW,TH=1024,576; DEV="cuda" if torch.cuda.is_available() else "cpu"
net=UNet().to(DEV); net.load_state_dict(torch.load(os.path.join(HERE,"seg2_hr2.pth"),map_location=DEV)); net.eval()
@torch.no_grad()
def prob(img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None].to(DEV)
    return torch.sigmoid(net(x))[0].cpu().numpy()
COL=[(0,0,255),(0,165,255),(0,255,0),(255,140,0),(0,255,255),(255,0,200),(255,150,255)]
def overlay_lines(img,rec):
    h,w=img.shape[:2]; m=AC.clean_label_masks(rec,w,h)   # (7,h,w) reprojekte rol-çizgileri
    o=img.copy()
    for c in range(7):
        o[m[c]>80]=COL[c]
    return o
def main():
    J=json.load(open(os.path.join(HERE,"cand","_truecov_multicand.json")))
    recs=J.get("recovered_files",[])
    print(f"{len(recs)} recovered -> doğrula")
    panels=[]
    for name in recs[:16]:
        p=os.path.join(CB,name); img=cv2.imread(p)
        if img is None: continue
        h,w=img.shape[:2]
        rec,method,ok=calib_multicand(prob(img),w,h,sanity,img=img)
        if not ok: continue
        ov=overlay_lines(img,rec)
        cv2.putText(ov,f"{name.split('__')[0][:16]} res={rec['fit']:.2f} [{method}]",(8,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
        twod,n=draw_2d(np.empty((0,2)),rec["camside"],"2D")
        Hp=twod.shape[0]; ovr=cv2.resize(ov,(int(w*Hp/h),Hp))
        panels.append(np.hstack([ovr,np.full((Hp,5,3),60,np.uint8),twod]))
    if not panels: print("panel yok"); return
    W=max(x.shape[1] for x in panels); panels=[cv2.copyMakeBorder(x,0,0,0,W-x.shape[1],cv2.BORDER_CONSTANT,value=(20,20,20)) for x in panels]
    cols=2; rows=(len(panels)+cols-1)//cols; cH=max(x.shape[0] for x in panels)
    sheet=np.full((rows*(cH+8),cols*(W+8),3),12,np.uint8)
    for i,pan in enumerate(panels):
        r,c=divmod(i,cols); y=r*(cH+8); x=c*(W+8); sheet[y:y+pan.shape[0],x:x+pan.shape[1]]=pan
    out=os.path.join(HERE,"cand","_VERIFY_RECOVERED.jpg"); cv2.imwrite(out,sheet); print("->",out)
if __name__=="__main__": main()
