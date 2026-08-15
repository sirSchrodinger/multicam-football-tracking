#!/usr/bin/env python3
"""GENELLEME KANITI: keyfi YENİ saha karesi -> ft-detektör (oyuncu ayakları) + otonom auto-calib
(rol-çizgi seg -> joint_calib -> H + SANITY) -> 2D radar. Çok saha -> tek montaj.
Çankaya-only DEĞİL: her saha bağımsız, elle-tık YOK. Sanity-elenen sahalar dürüstçe işaretlenir.
ENV: SEG=seg2_hr2.pth N=12 OUT=cand/_GENELLEME_GRID.jpg
"""
import os, sys, glob, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from calib_train import auto_calib as AC
from calib_train.auto_clean2d import sanity, project_feet, draw_2d
from calib_train.render2d_clean import UNet
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); ROOT=os.path.dirname(HERE)
CB=os.path.join(HERE,"cand_big"); TW,TH=1024,576
SEG=os.environ.get("SEG",os.path.join(HERE,"seg2_hr2.pth"))
N=int(os.environ.get("N","12")); OUTP=os.environ.get("OUT",os.path.join(HERE,"cand","_GENELLEME_GRID.jpg"))
DEV="cuda" if torch.cuda.is_available() else "cpu"
# --- çizgi-seg modeli (calib) ---
seg=UNet(); seg.load_state_dict(torch.load(SEG,map_location="cpu")); seg.eval()
@torch.no_grad()
def seg_prob(img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None]
    return torch.sigmoid(seg(x))[0].numpy()
# --- ft-detektör (oyuncu) ---
print("RF-DETR yükleniyor...",flush=True)
from rfdetr import RFDETRLargeDeprecated
from PIL import Image
WEIGHTS=os.path.join(ROOT,"models/weights/checkpoint_ft.pth")
det=RFDETRLargeDeprecated(pretrain_weights=WEIGHTS, device=DEV, num_classes=4)
def feet_of(img):
    rgb=img[:,:,::-1]; d=det.predict(Image.fromarray(rgb), threshold=0.30)
    xy=d.xyxy; cf=d.confidence; out=[]
    for (x1,y1,x2,y2),c in zip(xy,cf):
        h=y2-y1; w=x2-x1
        if h<20 or w<=0 or h/max(w,1)<1.15: continue   # top/artefakt filtre
        out.append(((x1+x2)/2.0, y2))
    return np.array(out,float) if out else np.empty((0,2)), d
def boxes_img(img,d,title):
    o=img.copy()
    for (x1,y1,x2,y2),c in zip(d.xyxy,d.confidence):
        h=y2-y1;w=x2-x1
        if h<20 or w<=0 or h/max(w,1)<1.15: continue
        cv2.rectangle(o,(int(x1),int(y1)),(int(x2),int(y2)),(60,230,60),2)
    cv2.putText(o,title,(8,26),cv2.FONT_HERSHEY_SIMPLEX,0.7,(60,230,60),2)
    return o
def panel(p):
    img=cv2.imread(p)
    if img is None: return None
    h,w=img.shape[:2]; name=os.path.basename(p).split("__")[0][:16]
    rec=AC.calib_from_pred(seg_prob(img),w,h)
    feet,d=feet_of(img)
    if not rec.get("ok"):
        L=boxes_img(img,d,f"{name}  [{len(feet)} oyuncu]")
        R=np.full((L.shape[0],int(L.shape[0]*1.4),3),28,np.uint8); cv2.putText(R,"CALIB YOK",(20,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,140,255),2)
        return np.hstack([L,np.full((L.shape[0],6,3),60,np.uint8),R]),False,rec.get("reason","?")
    ok,why=sanity(rec,w,h,img=img)
    mp=project_feet(feet,rec,w,h) if len(feet) else np.empty((0,2))
    tag=f"{name} res={rec['fit']:.2f}m {'OK' if ok else 'KAYIK'}"
    twod,n=draw_2d(mp,rec["camside"],tag)
    Hpx=twod.shape[0]; L=boxes_img(img,d,f"{name} [{len(feet)} oyuncu]"); L=cv2.resize(L,(int(w*Hpx/h),Hpx))
    return np.hstack([L,np.full((Hpx,6,3),60,np.uint8),twod]),ok,why
def main():
    pool=sorted(glob.glob(os.path.join(CB,"*__*__s*.jpg"))); pool=[p for p in pool if "_tmp_" not in p]
    # eval JSON'dan KABUL edilenleri tercih et (anlamlı radar), + birkaç fail (dürüstlük)
    cov=os.path.join(HERE,"cand",os.environ.get("COVJSON","_cov_base121.json"))
    if not os.path.exists(cov): cov=os.path.join(HERE,"cand","_cov_base_hr2_1024.json")
    acc=set(); rej=[]
    if os.path.exists(cov):
        J=json.load(open(cov))
        for a,b,c in J["rows"]:
            full=os.path.join(CB,a)
            if b is not None and b<0.45: acc.add(full)
            else: rej.append(full)
    accept=[p for p in pool if p in acc]; reject=[p for p in pool if p in rej]
    # çeşitlilik: kabul havuzundan eşit-aralık örnek + 2 fail
    sel=[]
    if accept:
        idx=np.linspace(0,len(accept)-1,min(N-2,len(accept))).astype(int); sel=[accept[i] for i in idx]
    sel+=reject[:2]
    print(f"havuz={len(pool)} kabul={len(accept)} -> {len(sel)} panel",flush=True)
    panels=[]; npass=0
    for p in sel:
        r=panel(p)
        if r is None: continue
        pan,ok,why=r; panels.append(pan); npass+=int(ok)
        print(f"  {os.path.basename(p)[:22]:22s} {'PASS' if ok else 'sanity:'+why}",flush=True)
    if not panels: print("panel yok"); return
    Wt=max(x.shape[1] for x in panels); panels=[cv2.copyMakeBorder(x,0,0,0,Wt-x.shape[1],cv2.BORDER_CONSTANT,value=(20,20,20)) for x in panels]
    cols=2; rows=(len(panels)+cols-1)//cols
    cellH=max(x.shape[0] for x in panels)
    sheet=np.full((rows*(cellH+8),cols*(Wt+8),3),12,np.uint8)
    for i,pan in enumerate(panels):
        r,c=divmod(i,cols); y=r*(cellH+8); x=c*(Wt+8); sheet[y:y+pan.shape[0],x:x+pan.shape[1]]=pan
    cv2.imwrite(OUTP,sheet); print(f"\n{npass}/{len(panels)} sanity-PASS -> {OUTP}",flush=True)
if __name__=="__main__": main()
