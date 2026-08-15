#!/usr/bin/env python3
"""Çözülen saha modelini (kenar+orta+yuvarlak) ORİJİNAL eğri kareye geri yansıt.
Model çizgileri gerçek boyalı çizgilere oturuyorsa kalibrasyon doğru (lekesiz kanıt).
Çıktı: cand/_reproject.jpg
"""
import json, os, numpy as np, cv2
HERE=os.path.dirname(os.path.abspath(__file__)); LF=os.path.join(HERE,"label_frames")
C=json.load(open(os.path.join(HERE,"calib_solved.json"))); ORDER=json.load(open(os.path.join(HERE,"frames.json")))
L,W=34.0,18.0
def metric_to_img(M,k1,k2,H,cx,cy,s):
    Hinv=np.linalg.inv(np.array(H)); m=np.concatenate([M,np.ones((len(M),1))],1).T
    un=Hinv@m; u=un[0]/un[2]; v=un[1]/un[2]; ru=np.sqrt(u*u+v*v)+1e-9
    rd=np.linspace(0,2.2,2000); rru=rd*(1+k1*rd*rd+k2*rd**4)
    rdv=np.interp(ru,rru,rd); sc=rdv/ru
    return np.stack([cx+u*sc*s, cy+v*sc*s],1)
def seg(a,b,nn=60): t=np.linspace(0,1,nn)[:,None]; return a*(1-t)+b*(1-0+0)*0+b*t
def polyline(M,**kw): return M
thumbs=[]
for name in ORDER:
    if name not in C: continue
    r=C[name]; img=cv2.imread(os.path.join(LF,name)); h,w=r["h"],r["w"]; cx,cy,s=w/2,h/2,w/2
    lines=[]
    N=80; t=np.linspace(0,1,N)[:,None]
    def L_(p,q): return p*(1-t)+q*t
    lines+= [L_(np.array([0,0]),np.array([L,0])), L_(np.array([0,W]),np.array([L,W])),
             L_(np.array([0,0]),np.array([0,W])), L_(np.array([L,0]),np.array([L,W])),
             L_(np.array([L/2,0]),np.array([L/2,W]))]
    th=np.linspace(0,2*np.pi,120)[:,None]; circ=np.concatenate([L/2+3*np.cos(th),W/2+3*np.sin(th)],1); lines.append(circ)
    vis=img.copy()
    cols=[(0,80,255),(0,80,255),(60,220,60),(60,220,60),(0,230,230),(255,160,0)]
    for i,M in enumerate(lines):
        P=metric_to_img(M,r["k1"],r["k2"],r["H"],cx,cy,s)
        P=P[(P[:,0]>-50)&(P[:,0]<w+50)&(P[:,1]>-50)&(P[:,1]<h+50)]
        if len(P)>1: cv2.polylines(vis,[P.astype(np.int32)],False,cols[i%len(cols)],3,cv2.LINE_AA)
    sc=900/w; vis=cv2.resize(vis,(int(w*sc),int(h*sc)))
    cv2.putText(vis,f"{r['n']}. {name[:16]} res{r['fit']:.2f}m",(6,24),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,230,255),2)
    thumbs.append((r["n"],vis))
thumbs.sort(key=lambda x:x[0]); H0=thumbs[0][1].shape[0]; Wt=thumbs[0][1].shape[1]
cols=2; rows=(len(thumbs)+cols-1)//cols
sheet=np.full((rows*(H0+6),cols*(Wt+6),3),20,np.uint8)
for i,(n,t) in enumerate(thumbs):
    rr,cc=divmod(i,cols); sheet[rr*(H0+6):rr*(H0+6)+t.shape[0], cc*(Wt+6):cc*(Wt+6)+t.shape[1]]=t
cv2.imwrite(os.path.join(HERE,"cand","_reproject.jpg"),sheet); print("-> cand/_reproject.jpg",sheet.shape)
