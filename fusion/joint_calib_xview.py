#!/usr/bin/env python3
"""ÇAPRAZ-GÖRÜŞ JOINT KALİBRASYON — çekirdek IP (2-kamera, geometri-limitli sahalar için).
Bir kamera KENDİ yakın-yarısında iyi-koşullu; uzak-yarısı belirsiz/fence-kilitli (bowtie kökü).
Partner kamera o uzak-bölgeyi KENDİ yakın-yarısında iyi görür -> çapraz-görüş oyuncu-konumları
uzak-yarıyı PİNLER. calibrate_xview: near-line nokta-korespondansı + cross-view oyuncu-korespondansı
birlikte -> tam H. Lisans: scipy/cv2 (BSD).

Phase-0 NEGATİF-KONTROL (Çankaya): cam1'e SADECE yakın-yarı çizgi ver (uzak dejenere olmalı) +
cam2'nin gördüğü uzak-bölge oyuncuları ekle -> tam cam1 kurtuluyor mu? 3 mod karşılaştır.
"""
import os, sys, json, numpy as np, cv2, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fusion import pitch_map as PM

def calibrate_xview(undist_px, metric_m):
    """undist_px (N,2) undistorted-pixel, metric_m (N,2) hedef metre -> H (undist-pixel->metre), RANSAC."""
    H,inl=cv2.findHomography(undist_px.astype(np.float32), metric_m.astype(np.float32), cv2.RANSAC, 0.5)
    return H, (int(inl.sum()) if inl is not None else 0)

def _proj(H, px):
    q=H@np.c_[px,np.ones(len(px))].T; return np.c_[q[0]/q[2], q[1]/q[2]]

if __name__=="__main__":
    C1=PM.load("calib/cankaya_cam1_FINAL.json"); C2=PM.load("calib/cankaya_cam2_FINAL.json")
    F1=json.load(open("calib/cankaya_cam1_FINAL.json")); SC1=C1["SC"]
    Hp2i=np.array(F1["H_pitch2img"]); Href_i2p=np.array(F1["H_img2pitch"])
    Lm,Wm=48.4,22.5; HALF=Lm/2; FPS1,FPS2=30.0,24.872; OFF=55.1
    # --- cam1 NEAR-HALF çizgi nokta-korespondansları (reproject FINAL, X<HALF) ---
    seg=[]  # metric_m çizgi noktaları (yakın-yarı)
    xs=np.linspace(0,HALF,40)
    seg += [[x,0] for x in xs] + [[x,Wm] for x in xs]           # touchline'lar (yakın yarı)
    seg += [[0,y] for y in np.linspace(0,Wm,30)]                # yakın goal-line X=0
    seg += [[HALF,y] for y in np.linspace(0,Wm,30)]             # orta çizgi X=HALF
    seg=np.array(seg,float)
    rel=seg/SC1
    und_line=_proj(Hp2i, rel)                                    # undistorted-pixel
    und_line += np.random.RandomState(0).normal(0,1.5,und_line.shape)  # gerçekçi tespit gürültüsü
    # --- cross-view oyuncu korespondansları (cam1 undist-pixel <-> cam2-metrik-flip), UZAK bölge ---
    d1=pd.read_parquet("raw/tracks_cankaya_cam1.parquet"); d2=pd.read_parquet("raw/tracks_cankaya_cam2_fullgame.parquet")
    g1=d1.groupby("frame"); g2=d2.groupby("frame")
    XP=[]; XM=[]  # cam1 undist-pixel , metric_m (cam2'den)
    for f2 in range(200,17000,55):
        if f2 not in g2.groups: continue
        f1=int(round((f2/FPS2-OFF)*FPS1))
        if f1 not in g1.groups: continue
        a=g1.get_group(f1); b=g2.get_group(f2)
        if len(a)<5 or len(b)<5: continue
        af=a[["foot_x","foot_y"]].values
        am=PM.to_pitch(C1,af)                                   # cam1 metrik (eşleştirme için)
        bm=PM.to_pitch(C2,b[["foot_x","foot_y"]].values); bmf=np.c_[Lm-bm[:,0],Wm-bm[:,1]]
        u=cv2.undistortPoints(af.reshape(-1,1,2),C1["K"],C1["D"],P=C1["K"]).reshape(-1,2)
        for i in range(len(am)):
            dd=np.hypot(bmf[:,0]-am[i,0],bmf[:,1]-am[i,1]); j=dd.argmin()
            if dd[j]<1.2 and bmf[j,0]>HALF:                     # SADECE UZAK bölge oyuncuları (prize bilgisi)
                XP.append(u[i]); XM.append(bmf[j])
    XP=np.array(XP); XM=np.array(XM)
    print(f"near-line nokta: {len(und_line)} | cross-view UZAK-oyuncu korespondans: {len(XP)}")
    # --- 3 MOD ---
    Hn,_=calibrate_xview(und_line, seg)                          # (a) sadece near-line
    Hx,_=calibrate_xview(XP, XM) if len(XP)>=8 else (None,0)     # (b) sadece cross-view
    Hj,inj=calibrate_xview(np.r_[und_line,XP], np.r_[seg,XM]) if len(XP)>=8 else (None,0)  # (c) near+cross
    # --- DOĞRULAMA: tüm sahada (özellikle UZAK yarı) FINAL'e karşı metre hatası ---
    samp=d1.sample(min(4000,len(d1)),random_state=2); fpx=samp[["foot_x","foot_y"]].values
    u=cv2.undistortPoints(fpx.reshape(-1,1,2),C1["K"],C1["D"],P=C1["K"]).reshape(-1,2)
    ref=PM.to_pitch(C1,fpx)                                      # FINAL metrik (gerçek)
    far=ref[:,0]>HALF
    def err(H,mask=None):
        if H is None: return None
        p=_proj(H,u); d=np.hypot(p[:,0]-ref[:,0],p[:,1]-ref[:,1])
        return np.median(d[mask]) if mask is not None else np.median(d)
    print(f"\nFINAL'e metre hatası (median):")
    print(f"  (a) SADECE near-line     : tüm {err(Hn):.2f}m | UZAK-yarı {err(Hn,far):.2f}m  <- uzak DEJENERE beklenir")
    print(f"  (b) SADECE cross-view    : tüm {err(Hx):.2f}m | UZAK-yarı {err(Hx,far):.2f}m" if Hx is not None else "  (b) cross-view yetersiz")
    print(f"  (c) near + cross-view    : tüm {err(Hj):.2f}m | UZAK-yarı {err(Hj,far):.2f}m  <- KURTARMA")
    if Hj is not None and err(Hn,far) and err(Hj,far):
        print(f"\n=> UZAK-yarı: near-only {err(Hn,far):.2f}m -> near+crossview {err(Hj,far):.2f}m "
              f"({'KURTARDI ✓' if err(Hj,far)<err(Hn,far)*0.6 else 'marjinal'}): çapraz-görüş near-line'ın pinleyemediği uzak-yarıyı pinledi")
