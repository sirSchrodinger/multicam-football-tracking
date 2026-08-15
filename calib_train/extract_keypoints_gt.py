#!/usr/bin/env python3
"""Çözülmüş kalibrasyondan (calib_solved.json) her sahanın HAM görüntüdeki keypoint'lerini
çıkar -> detektör eğitim GT'si. Metrik keypoint -> Hinv -> distort -> ham piksel.
Keypoint seti: 4 köşe (corner_nN,nF,fN,fF) + orta-çizgi uçları (center_n,center_f) + yuvarlak merkezi.
center/circle gerçek değilse visible=0. Çıktı: calib_keypoints.json + cand/_kp_gt.jpg
"""
import json, os, numpy as np, cv2
HERE=os.path.dirname(os.path.abspath(__file__)); LF=os.path.join(HERE,"label_frames")
C=json.load(open(os.path.join(HERE,"calib_solved.json"))); ORDER=json.load(open(os.path.join(HERE,"frames.json")))
NO_CENTER={3,7,9}; NO_CIRCLE={1,2,3,4,7,9,10,11,12,13}; L=34.0
def m2i_factory(k1,k2,H,cx,cy,s):
    Hinv=np.linalg.inv(np.array(H)); rd=np.linspace(0,2.5,2500); rru=rd*(1+k1*rd*rd+k2*rd**4)
    def f(M):
        M=np.atleast_2d(M); m=np.concatenate([M,np.ones((len(M),1))],1).T; un=Hinv@m
        u=un[0]/un[2]; vv=un[1]/un[2]; ru=np.sqrt(u*u+vv*vv)+1e-9; rr=np.interp(ru,rru,rd); sc=rr/ru
        return np.stack([cx+u*sc*s,cy+vv*sc*s],1)
    return f
out={}; thumbs=[]
for name in ORDER:
    if name not in C: continue
    r=C[name]; img=cv2.imread(os.path.join(LF,name)); h,w=r["h"],r["w"]; cx,cy,s=w/2,h/2,w/2
    Wp=r.get("Wp",18.0); n=r["n"]; m2i=m2i_factory(r["k1"],r["k2"],r["H"],cx,cy,s)
    KP=[("corner_nN",(0,0),1),("corner_nF",(0,Wp),1),("corner_fN",(L,0),1),("corner_fF",(L,Wp),1),
        ("center_n",(L/2,0),0 if n in NO_CENTER else 1),("center_f",(L/2,Wp),0 if n in NO_CENTER else 1),
        ("circle_c",(L/2,Wp/2),0 if n in NO_CIRCLE else 1)]
    kps={}; vis=img.copy()
    for nm,M,v in KP:
        p=m2i(np.array([M]))[0]; inb=0<=p[0]<w and 0<=p[1]<h
        kps[nm]=dict(x=round(float(p[0]),1),y=round(float(p[1]),1),visible=int(v and inb))
        if kps[nm]["visible"]:
            cv2.circle(vis,(int(p[0]),int(p[1])),9,(0,255,255),-1)
            cv2.putText(vis,nm.replace("corner_","").replace("center_","c").replace("circle_","O"),
                        (int(p[0])+8,int(p[1])-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,180,255),2)
    out[name]=dict(n=n,w=w,h=h,bad=r.get("bad"),aspect_known=r.get("aspect_known",False),kps=kps)
    sc=560/w; vis=cv2.resize(vis,(int(w*sc),int(h*sc)))
    cv2.putText(vis,f"{n}. {name[:14]}",(6,22),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,230,255),2); thumbs.append((n,vis))
json.dump(out,open(os.path.join(HERE,"calib_keypoints.json"),"w"),indent=1,ensure_ascii=False)
nvis=sum(kp["visible"] for v in out.values() for kp in v["kps"].values())
print(f"{len(out)} saha, {nvis} görünür keypoint -> calib_keypoints.json")
for name,v in sorted(out.items(),key=lambda x:x[1]["n"]):
    vv=sum(k["visible"] for k in v["kps"].values()); print(f"  {v['n']:2d} {name[:18]:18s} {vv}/7 kp {'[BOZUK]' if v['bad'] else ''}")
thumbs.sort(key=lambda t:t[0]); cols=4; rows=(len(thumbs)+cols-1)//cols
H0=thumbs[0][1].shape[0]; Wt=thumbs[0][1].shape[1]
sheet=np.full((rows*(H0+6),cols*(Wt+6),3),20,np.uint8)
for i,(n,t) in enumerate(thumbs):
    rr,cc=divmod(i,cols); sheet[rr*(H0+6):rr*(H0+6)+t.shape[0],cc*(Wt+6):cc*(Wt+6)+t.shape[1]]=t
cv2.imwrite(os.path.join(HERE,"cand","_kp_gt.jpg"),sheet); print("-> cand/_kp_gt.jpg")
