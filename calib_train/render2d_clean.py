#!/usr/bin/env python3
"""TEMİZ 2D: saha-dışını + aşırı-gerilen(smeared) bölgeyi BOŞALT (kameranın gerçekten çözdüğü
yeri göster), üstüne KANONİK temiz pitch çizgileri çiz -> her zaman '2D halısaha'ya benzer.
En iyi ışıklı kareden kalibre. Çıktı: cand/_CLEAN2D.jpg (ham | temiz-2D yan yana).
"""
import os, sys, glob, numpy as np, cv2
os.environ["CUDA_VISIBLE_DEVICES"]=""
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn
from calib_train import auto_calib as AC
HERE=os.path.dirname(os.path.abspath(__file__)); LF=os.path.join(HERE,"label_frames"); CAND=os.path.join(HERE,"cand_big")
NC=7; TW,TH,BASE=1024,576,32; L,Wp,S,MG=34.0,18.0,22,4
def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True),nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True))
class UNet(nn.Module):
    def __init__(s,b=BASE):
        super().__init__(); s.d1=cbr(3,b);s.d2=cbr(b,b*2);s.d3=cbr(b*2,b*4);s.d4=cbr(b*4,b*8);s.p=nn.MaxPool2d(2)
        s.u3=nn.ConvTranspose2d(b*8,b*4,2,2);s.c3=cbr(b*8,b*4);s.u2=nn.ConvTranspose2d(b*4,b*2,2,2);s.c2=cbr(b*4,b*2)
        s.u1=nn.ConvTranspose2d(b*2,b,2,2);s.c1=cbr(b*2,b);s.h=nn.Conv2d(b,NC,1)
    def forward(s,x):
        c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2));c4=s.d4(s.p(c3))
        x=s.c3(torch.cat([s.u3(c4),c3],1));x=s.c2(torch.cat([s.u2(x),c2],1));x=s.c1(torch.cat([s.u1(x),c1],1));return s.h(x)
net=UNet(); net.load_state_dict(torch.load(os.path.join(HERE,"seg2_hr2.pth"),map_location="cpu")); net.eval()
@torch.no_grad()
def prob(img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None]
    return torch.sigmoid(net(x))[0].numpy()

def clean2d(img,rec):
    k1,k2,H=rec["k1"],rec["k2"],rec["H"]; h,w=img.shape[:2]; cx,cy,s=w/2,h/2,w/2
    LS,WS=int((L+2*MG)*S),int((Wp+2*MG)*S); Hi=np.linalg.inv(H)
    Xs,Ys=np.meshgrid(np.linspace(-MG,L+MG,LS),np.linspace(-MG,Wp+MG,WS))
    den=Hi[2,0]*Xs+Hi[2,1]*Ys+Hi[2,2]; un=(Hi[0,0]*Xs+Hi[0,1]*Ys+Hi[0,2])/den; vn=(Hi[1,0]*Xs+Hi[1,1]*Ys+Hi[1,2])/den
    ru=np.sqrt(un*un+vn*vn); rg=np.linspace(0,2.6,5000); rug=rg*(1+k1*rg*rg+k2*rg**4)
    rd=np.interp(ru,rug,rg); sc=np.divide(rd,ru,out=np.ones_like(ru),where=ru>1e-9)
    mapx=(cx+un*sc*s).astype(np.float32); mapy=(cy+vn*sc*s).astype(np.float32)
    warp=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR,borderValue=(0,0,0))
    # GERME (smear) tespiti: kaynak-koord output üzerinde yavaş değişiyorsa (küçük gradyan) -> boşalt
    gx=np.abs(np.gradient(mapx,axis=1))+np.abs(np.gradient(mapx,axis=0))
    gy=np.abs(np.gradient(mapy,axis=1))+np.abs(np.gradient(mapy,axis=0))
    stretch=gx+gy
    # kaynak görüntü-dışı + aşırı-germe maskesi
    inb=(mapx>2)&(mapx<w-2)&(mapy>2)&(mapy<h-2)
    good=inb&(stretch>0.18)
    good=cv2.morphologyEx(good.astype(np.uint8),cv2.MORPH_OPEN,np.ones((5,5),np.uint8)).astype(bool)
    # ŞEMATİK zemin: çim yeşili + biçme şeritleri
    base=np.zeros((WS,LS,3),np.uint8)
    for i in range(int(L/3)+1):
        x0=int((MG+i*3)*S); x1=int((MG+(i+1)*3)*S); base[int(MG*S):int((MG+Wp)*S),x0:x1]=(40,95,38) if i%2 else (46,110,44)
    out=base.copy(); out[good]=warp[good]   # çözülen yere gerçek doku, gerisine şematik çim
    # KANONİK temiz çizgiler (her zaman)
    def P(X,Y): return (int((X+MG)*S),int((Wp-Y+MG)*S)) if rec["camside"]=="SAG" else (int((X+MG)*S),int((Y+MG)*S))
    white=(245,245,245)
    cv2.rectangle(out,P(0,0),P(L,Wp),white,2); cv2.line(out,P(L/2,0),P(L/2,Wp),white,2)
    cv2.circle(out,P(L/2,Wp/2),int(3*S),white,2)
    for gx0 in (0,L):
        bx=5 if gx0==0 else L-5
        cv2.rectangle(out,P(min(gx0,bx),Wp/2-5),P(max(gx0,bx),Wp/2+5),white,2)
    cam=P(0,0) if rec["camside"]!="SAG" else P(0,Wp)
    cv2.circle(out,cam,9,(0,200,255),-1); cv2.putText(out,"KAM",(cam[0]+8,cam[1]),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,255),2)
    return out

def pick_venues():
    import json
    picks=[]
    # en iyi ışıklı + iyi-kalibre olanlardan seç (label_frames parlak + cand_big yüksek skor)
    cands=sorted(glob.glob(os.path.join(LF,"*.jpg")))+sorted(glob.glob(os.path.join(CAND,"*__*__s*.jpg")),key=lambda p:-int(p.split("__s")[-1][:3]))
    seen=set()
    for p in cands:
        img=cv2.imread(p)
        if img is None: continue
        if float(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).mean())<70: continue   # ışıklı şart
        rec=AC.calib_from_pred(prob(img),img.shape[1],img.shape[0])
        if not rec.get("ok") or rec["fit"] is None or rec["fit"]>0.4: continue
        key="".join(c for c in os.path.basename(p).split("__")[0].lower() if c.isalnum())[:8]
        if key in seen: continue
        seen.add(key); picks.append((p,img,rec))
        if len(picks)>=4: break
    return picks

if __name__=="__main__":
    picks=pick_venues(); print("seçilen:",[f"{os.path.basename(p)[:12]}={r['fit']:.2f}" for p,_,r in picks],flush=True)
    rows=[]
    for p,img,rec in picks:
        cl=clean2d(img,rec); WSh=cl.shape[0]
        raw=cv2.resize(img,(int(img.shape[1]*WSh/img.shape[0]),WSh))
        cv2.putText(raw,os.path.basename(p).split("__")[0][:16],(8,26),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        cv2.putText(cl,f"res={rec['fit']:.2f}m",(8,WSh-12),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,230,255),2)
        rows.append(np.hstack([raw,np.full((WSh,6,3),50,np.uint8),cl]))
    wm=max(r.shape[1] for r in rows); sheet=np.full((sum(r.shape[0]+8 for r in rows),wm,3),18,np.uint8); y=0
    for r in rows: sheet[y:y+r.shape[0],0:r.shape[1]]=r; y+=r.shape[0]+8
    cv2.imwrite(os.path.join(HERE,"cand","_CLEAN2D.jpg"),sheet); print("-> cand/_CLEAN2D.jpg",flush=True)
