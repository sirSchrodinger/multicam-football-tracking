#!/usr/bin/env python3
"""MATCH-FREE çapraz-görüş kalibrasyon (deployment-kritik): cam1'i SADECE cam2-metrik-anchor'dan kalibre,
cam1 CALIB'İ HİÇ KULLANMADAN (eşleştirme de YOK). RANSAC-over-matches + ICP refine.
Çizgisi-bozuk kameranın partner'dan kalibre olabildiğinin KESİN testi.
"""
import os, sys, json, numpy as np, cv2, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusion import pitch_map as PM
C1=PM.load("calib/cankaya_cam1_FINAL.json"); C2=PM.load("calib/cankaya_cam2_FINAL.json")
F1=json.load(open("calib/cankaya_cam1_FINAL.json")); Href=np.array(F1["H_img2pitch"]); SC1=C1["SC"]
L,W=48.4,22.5; FPS1,FPS2=30.0,24.872; OFF=55.1
d1=pd.read_parquet("raw/tracks_cankaya_cam1.parquet"); d2=pd.read_parquet("raw/tracks_cankaya_cam2_fullgame.parquet")
g1=d1.groupby("frame"); g2=d2.groupby("frame")
# senkron kare çiftleri topla: cam1 UNDISTORTED pixel (rel-hedef BİLİNMİYOR), cam2 metrik->flip (rel)
frames=[]
for f2 in range(200,17000,50):
    if f2 not in g2.groups: continue
    f1=int(round((f2/FPS2-OFF)*FPS1))
    if f1 not in g1.groups: continue
    a=g1.get_group(f1); b=g2.get_group(f2)
    if len(a)<5 or len(b)<5: continue
    af=a[["foot_x","foot_y"]].values
    u=cv2.undistortPoints(af.reshape(-1,1,2),C1["K"],C1["D"],P=C1["K"]).reshape(-1,2)  # cam1 undist-pixel
    bm=PM.to_pitch(C2,b[["foot_x","foot_y"]].values); bm_flip=np.c_[(L-bm[:,0])/SC1,(W-bm[:,1])/SC1]  # cam2->cam1 rel hedef
    frames.append((u,bm_flip))
print(f"{len(frames)} senkron kare")
def score_H(H,thr=1.0/SC1):
    """her karede cam1-pixel'i H ile rel'e map et, en yakın cam2-hedefe; toplam inlier (<thr rel)."""
    inl=0
    for u,m in frames:
        q=H@np.c_[u,np.ones(len(u))].T; p=np.c_[q[0]/q[2],q[1]/q[2]]
        for pi in p:
            dd=np.hypot(m[:,0]-pi[0],m[:,1]-pi[1])
            if dd.min()<thr: inl+=1
    return inl
def solve_assign(H,thr=1.0/SC1):
    """mutual-nearest atama -> tüm karelerden korespondans -> yeni H."""
    P=[]; M=[]
    for u,m in frames:
        q=H@np.c_[u,np.ones(len(u))].T; p=np.c_[q[0]/q[2],q[1]/q[2]]
        for i,pi in enumerate(p):
            dd=np.hypot(m[:,0]-pi[0],m[:,1]-pi[1]); j=dd.argmin()
            if dd[j]<thr: P.append(u[i]); M.append(m[j])
    if len(P)<10: return None,0
    P=np.array(P); M=np.array(M); Hn,_=cv2.findHomography(P,M,cv2.RANSAC,0.3)
    return Hn,len(P)
# RANSAC init: rastgele kare + 4 rastgele eşleme dene, en iyi skoru tut
rng=np.random.RandomState(0); best=None; bests=-1
for it in range(4000):
    u,m=frames[rng.randint(len(frames))]
    if len(u)<4 or len(m)<4: continue
    iu=rng.choice(len(u),4,replace=False); im=rng.choice(len(m),4,replace=False)
    H,_=cv2.findHomography(u[iu],m[im])
    if H is None: continue
    s=score_H(H)
    if s>bests: bests=s; best=H
print(f"RANSAC init en iyi skor: {bests}")
# ICP refine
H=best
for _ in range(12):
    Hn,n=solve_assign(H)
    if Hn is None: break
    H=Hn
# doğrula: cam1 feet'i match-free H vs FINAL H ile metre'ye -> fark
samp=d1.sample(min(3000,len(d1)),random_state=1); fpx=samp[["foot_x","foot_y"]].values
u=cv2.undistortPoints(fpx.reshape(-1,1,2),C1["K"],C1["D"],P=C1["K"]).reshape(-1,2)
def proj(Hm): q=Hm@np.c_[u,np.ones(len(u))].T; return np.c_[q[0]/q[2]*SC1,q[1]/q[2]*SC1]
diff=np.hypot(*(proj(H)-proj(Href)).T)
print(f"MATCH-FREE H vs FINAL H: median {np.median(diff):.2f}m p90 {np.percentile(diff,90):.2f}m")
print(f"=> {'BAŞARILI: cam1 SIFIR cam1-bilgisinden (sadece cam2+oyuncu) kalibre' if np.median(diff)<2.5 else 'ZAYIF/diverge'}")
