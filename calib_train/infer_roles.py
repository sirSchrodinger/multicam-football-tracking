#!/usr/bin/env python3
"""Rol-aware model çıkarımı: ham kare -> her çizginin ROLÜ renkli. Model "bu yakın kale,
bu uzak kale" diye kendisi söylüyor mu GÖRSEL. Kullanım: python infer_roles.py img...
goalN=kirmizi goalF=turuncu touchN=yesil touchF=mavi center=sari box=mor circle=pembe
"""
import sys, os, numpy as np, cv2, torch, torch.nn as nn
HERE=os.path.dirname(os.path.abspath(__file__)); TW,TH=512,288; NC=7
CL=["goalN","goalF","touchN","touchF","center","box","circle"]
COL=[(0,0,255),(0,150,255),(0,255,0),(255,120,0),(0,255,255),(255,0,200),(255,150,255)]
CK=os.path.join(HERE,"seg2_ckpt","seg2_unet.pth")
def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True),nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True))
class UNet(nn.Module):
    def __init__(s,nin=3,no=NC,b=24):
        super().__init__(); s.d1=cbr(nin,b);s.d2=cbr(b,b*2);s.d3=cbr(b*2,b*4);s.d4=cbr(b*4,b*8);s.p=nn.MaxPool2d(2)
        s.u3=nn.ConvTranspose2d(b*8,b*4,2,2);s.c3=cbr(b*8,b*4);s.u2=nn.ConvTranspose2d(b*4,b*2,2,2);s.c2=cbr(b*4,b*2)
        s.u1=nn.ConvTranspose2d(b*2,b,2,2);s.c1=cbr(b*2,b);s.h=nn.Conv2d(b,no,1)
    def forward(s,x):
        c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2));c4=s.d4(s.p(c3))
        x=s.c3(torch.cat([s.u3(c4),c3],1));x=s.c2(torch.cat([s.u2(x),c2],1));x=s.c1(torch.cat([s.u1(x),c1],1));return s.h(x)
dev="cuda" if torch.cuda.is_available() else "cpu"
net=UNet().to(dev); net.load_state_dict(torch.load(CK,map_location=dev)); net.eval()
def run(path):
    img=cv2.imread(path)
    if img is None: return None
    h,w=img.shape[:2]
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None].to(dev)
    with torch.no_grad(): p=torch.sigmoid(net(x))[0].cpu().numpy()
    vis=img.copy()
    for c in range(NC):
        pc=cv2.resize(p[c],(w,h))
        vis[pc>0.45]=COL[c]
    # lejant
    for i,(n,col) in enumerate(zip(CL,COL)):
        cv2.putText(vis,n,(10,28+i*26),cv2.FONT_HERSHEY_SIMPLEX,0.7,col,2)
    cv2.putText(vis,os.path.basename(path).split('__')[0][:20],(w-360,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    return cv2.resize(vis,(int(w*620/h),620))
rows=[run(p) for p in sys.argv[1:]]; rows=[r for r in rows if r is not None]
wm=max(r.shape[1] for r in rows); sheet=np.full((sum(r.shape[0]+6 for r in rows),wm,3),20,np.uint8); y=0
for r in rows: sheet[y:y+r.shape[0],0:r.shape[1]]=r; y+=r.shape[0]+6
cv2.imwrite(os.path.join(HERE,"cand","_roles.jpg"),sheet); print("-> cand/_roles.jpg")
