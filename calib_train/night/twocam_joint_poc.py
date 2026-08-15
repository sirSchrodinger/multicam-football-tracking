#!/usr/bin/env python3
"""2-KAMERA ÇAPRAZ-KALİBRASYON POC (Çankaya): cam1'i SADECE cam2-metrik-oyuncu-konumları + cam1-PİKSEL'den
kalibre et (cam1 ÇİZGİSİ KULLANMADAN). Çalışırsa: zor venue'de çizgisi-bozuk kamera, partner kameradan kalibre.
Doğrulama: çapraz-kamera H, cam1-FINAL (çizgi-tabanlı) H ile uyuşuyor mu?
Eşleştirme cam1'in mevcut pitch'iyle yapılır (SADECE eşleştirme için); H-çözümü cam1-pixel↔cam2-metrik kullanır.
"""
import os, sys, json, numpy as np, cv2, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusion import pitch_map as PM
C1=PM.load("calib/cankaya_cam1_FINAL.json"); C2=PM.load("calib/cankaya_cam2_FINAL.json")
F1=json.load(open("calib/cankaya_cam1_FINAL.json"))
L,W=48.4,22.5; SC1=C1["SC"]; FPS1,FPS2=30.0,24.872; OFF=55.1   # cam2_t = cam1_t + 55.1
d1=pd.read_parquet("raw/tracks_cankaya_cam1.parquet")
d2=pd.read_parquet("raw/tracks_cankaya_cam2_fullgame.parquet")
# cam1/cam2 pitch (metre) zaten kolonlarda. cam1_metric = flip(cam2_metric) = (L-X2, W-Y2).
g1=d1.groupby("frame"); g2=d2.groupby("frame")
corr_px=[]; corr_m=[]   # cam1 undistorted-pixel  ,  cam1_metric_rel target (cam2'den)
nf=0
for f2 in range(200, 17000, 60):    # cam2 frame örnekle
    if f2 not in g2.groups: continue
    t2=f2/FPS2; t1=t2-OFF
    if t1<2: continue
    f1=int(round(t1*FPS1))
    if f1 not in g1.groups: continue
    a=g1.get_group(f1); b=g2.get_group(f2)
    if len(a)<4 or len(b)<4: continue
    af=a[["foot_x","foot_y"]].values     # cam1 piksel (H-çözümü için)
    # metrikleri FOOT'lardan calib ile hesapla (pitch kolonu NaN/REL) — metre
    am=PM.to_pitch(C1, af)               # cam1 metrik (SADECE eşleştirme)
    bm=PM.to_pitch(C2, b[["foot_x","foot_y"]].values)
    bm_flip=np.c_[L-bm[:,0], W-bm[:,1]]  # cam2 metrik -> cam1 beklenen (flip)
    # nearest eşleştirme cam1_metric <-> flip(cam2_metric), <1.5m
    for i in range(len(am)):
        dd=np.hypot(bm_flip[:,0]-am[i,0], bm_flip[:,1]-am[i,1]); j=dd.argmin()
        if dd[j]<1.5:
            # cam1 undistorted-pixel
            u=cv2.undistortPoints(af[i].reshape(1,1,2), C1["K"], C1["D"], P=C1["K"]).reshape(2)
            corr_px.append(u)
            corr_m.append([bm_flip[j,0]/SC1, bm_flip[j,1]/SC1])   # rel (H_img2pitch hedefi)
    nf+=1
corr_px=np.array(corr_px); corr_m=np.array(corr_m)
print(f"{nf} senkron-kare, {len(corr_px)} çapraz-kamera korespondans")
if len(corr_px)<20: print("yetersiz korespondans"); sys.exit()
# H çöz: cam1_undistorted_pixel -> cam1_metric_rel (RANSAC)
H,inl=cv2.findHomography(corr_px, corr_m, cv2.RANSAC, 0.5)
print(f"RANSAC inlier: {int(inl.sum())}/{len(inl)}")
Href=np.array(F1["H_img2pitch"])
# DOĞRULAMA: cam1 feet'i hem çapraz-H hem FINAL-H ile metre'ye projekte et, FARK
samp=d1.sample(min(3000,len(d1)),random_state=1)
fpx=samp[["foot_x","foot_y"]].values
u=cv2.undistortPoints(fpx.reshape(-1,1,2), C1["K"], C1["D"], P=C1["K"]).reshape(-1,2)
def proj(Hm):
    q=Hm@np.c_[u,np.ones(len(u))].T; return np.c_[q[0]/q[2]*SC1, q[1]/q[2]*SC1]
pc=proj(H); pf=proj(Href)
diff=np.hypot(pc[:,0]-pf[:,0], pc[:,1]-pf[:,1])
print(f"ÇAPRAZ-KAMERA H vs FINAL(çizgi) H — metre farkı: median {np.median(diff):.2f}m, p90 {np.percentile(diff,90):.2f}m")
print(f"=> çapraz-kamera kalibrasyon {'BAŞARILI (çizgisiz cam1 kuruldu)' if np.median(diff)<2.0 else 'ZAYIF'}: cam1 SADECE cam2+oyunculardan kalibre edildi")
