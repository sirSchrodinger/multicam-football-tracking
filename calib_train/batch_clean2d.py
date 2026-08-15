#!/usr/bin/env python3
"""Çoklu-saha TEMİZ 2D + OTOMATİK KALİTE SKORU (sahayı oturttuk mu?). Boş/yeşili-bol kare tercih.
Kalite = saha-dikdörtgeninin ne kadarı gerçek-dokuyla dolu (good-fraction) + yatay-coverage.
'Oturanları' (yüksek skor) montajla, oturmayanları (kalibrasyon-kayık) say. Çıktı: cand/_BATCH2D.jpg
"""
import os, sys, glob, json, numpy as np, cv2
os.environ["CUDA_VISIBLE_DEVICES"]=""
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn
from calib_train import auto_calib as AC
from calib_train.render2d_clean import UNet, clean2d, L, Wp, S, MG
HERE=os.path.dirname(os.path.abspath(__file__)); LF=os.path.join(HERE,"label_frames"); CB=os.path.join(HERE,"cand_big")
TW,TH=1024,576
net=UNet(); net.load_state_dict(torch.load(os.path.join(HERE,"seg2_hr2.pth"),map_location="cpu")); net.eval()
@torch.no_grad()
def prob(img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None]
    return torch.sigmoid(net(x))[0].numpy()

def clean2d_q(img,rec):
    """clean2d + kalite: saha dikdörtgenindeki good-doku oranı + sütun-coverage."""
    out=clean2d(img,rec); WS,LS=out.shape[:2]
    # saha-içi dikdörtgen (margin hariç) -> gerçek-doku (şematik-çim DEĞİL) oranı
    x0,x1=int(MG*S),int((MG+L)*S); y0,y1=int(MG*S),int((MG+Wp)*S)
    fld=out[y0:y1,x0:x1]
    # şematik çim renkleri (40-46,95-110,38-44) -> gerçek doku bunun dışı
    g=fld.reshape(-1,3).astype(int)
    schem=(np.abs(g[:,0]-43)<10)&(np.abs(g[:,1]-102)<14)&(np.abs(g[:,2]-41)<10)
    realfrac=1-schem.mean()
    # yatay-coverage: kaç sütunda gerçek doku var (diyagonal-şerit düşük olur)
    colreal=(~schem).reshape(y1-y0,x1-x0).any(0); colcov=colreal.mean()
    return out, float(realfrac), float(colcov)

def pick(maxn=24):
    cands=sorted(glob.glob(os.path.join(LF,"*.jpg")))+sorted(glob.glob(os.path.join(CB,"*__*__s*.jpg")),key=lambda p:-int(p.split("__s")[-1][:3]))
    seen=set(); res=[]
    for p in cands:
        img=cv2.imread(p)
        if img is None: continue
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if gray.mean()<70: continue
        hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV); green=((hsv[...,0]>=30)&(hsv[...,0]<=92)&(hsv[...,1]>40)&(hsv[...,2]>40)).mean()
        if green<0.30: continue   # yeşili-bol (emptier/clear) tercih
        rec=AC.calib_from_pred(prob(img),img.shape[1],img.shape[0])
        if not rec.get("ok") or rec["fit"] is None or rec["fit"]>0.45: continue
        key="".join(c for c in os.path.basename(p).split("__")[0].lower() if c.isalnum())[:8]
        if key in seen: continue
        seen.add(key); res.append((p,img,rec))
        if len(res)>=maxn: break
    return res

if __name__=="__main__":
    picks=pick(); print(f"{len(picks)} aday render ediliyor...",flush=True)
    scored=[]
    for p,img,rec in picks:
        out,rf,cc=clean2d_q(img,rec); scored.append((p,img,rec,out,rf,cc))
        print(f"  {os.path.basename(p)[:22]:22s} res={rec['fit']:.2f} doku={rf:.2f} cov={cc:.2f} {'OTURDU' if (rf>0.45 and cc>0.8) else 'kayık'}",flush=True)
    sits=[s for s in scored if s[4]>0.45 and s[5]>0.8]
    print(f"\nOTURAN saha: {len(sits)}/{len(scored)} (doku>0.45 & cov>0.8)",flush=True)
    top=sorted(scored,key=lambda s:-(s[4]*s[5]))[:6]
    rows=[]
    for p,img,rec,out,rf,cc in top:
        WSh=out.shape[0]; raw=cv2.resize(img,(int(img.shape[1]*WSh/img.shape[0]),WSh))
        cv2.putText(raw,os.path.basename(p).split("__")[0][:16],(8,26),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.putText(out,f"res={rec['fit']:.2f}m  doku={rf:.0%}",(8,WSh-12),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,230,255),2)
        rows.append(np.hstack([raw,np.full((WSh,6,3),50,np.uint8),out]))
    wm=max(r.shape[1] for r in rows); sheet=np.full((sum(r.shape[0]+8 for r in rows),wm,3),18,np.uint8); y=0
    for r in rows: sheet[y:y+r.shape[0],0:r.shape[1]]=r; y+=r.shape[0]+8
    cv2.imwrite(os.path.join(HERE,"cand","_BATCH2D.jpg"),sheet); print("-> cand/_BATCH2D.jpg",flush=True)
