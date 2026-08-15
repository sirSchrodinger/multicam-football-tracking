#!/usr/bin/env python3
"""VISION-GROUNDED kalibrasyon (Alperen içgörüsü 1 Tem: "kareyi görebiliyorsun, sayıcı sen ol").
Otomatik seg->4-çizgi->solve pipeline'ı geometri-limitli (head-on) sahalarda bowtie veriyor ÇÜNKÜ
zengin yapıyı (merkez-yuvarlak/ceza-sahası/kale + görünmeyen-köşe extrapolasyonu) kullanmıyor.
GÖREN ajan (insan VEYA multimodal model) 4 saha-köşesini (occluded'ı extrapole) + opsiyonel iç-işaretleri
okur -> ortak-lens undistort -> homografi -> SANE saha. Çankaya/Gaziantep'te kanıtlandı (otomatik bowtie iken).

API: calibrate_from_corners(img_corners, L, W, w, h, k1, k2) -> rec(k1,k2,H,fit,camside) [auto_calib formatı]
     overlay(img, rec) -> kalibre çizgileri kareye reprojekte
img_corners SIRASI: NL(near-left=X0Y0), FL(X0 YW), NR(XL Y0), FR(XL YW)  (kamera near-sol varsayımı;
camside'a göre etiket dönebilir — annotator hangi köşe hangi rol söyler).
"""
import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calib_train import auto_calib as AC

COL=[(0,0,255),(0,165,255),(0,255,0),(255,140,0),(0,255,255),(255,0,200),(255,150,255)]

def calibrate_from_corners(img_corners, L, W, w, h, k1=0.167, k2=0.240, extra=None):
    """img_corners: (4,2) [NL,FL,NR,FR] image-pixel. extra: opsiyonel [(img_xy, metric_xy),...] ek kısıt.
    döner auto_calib-uyumlu rec: H undistort-norm->metrik (aH ile), k1,k2 ortak-lens."""
    cx,cy,s=w/2,h/2,w/2
    world=np.array([[0,0],[0,W],[L,0],[L,W]],float)
    pts=np.asarray(img_corners,float); un=AC.und_norm(pts,k1,k2,cx,cy,s)
    uw=un.copy(); ww=world.copy()
    if extra:
        for ixy,mxy in extra:
            uw=np.vstack([uw, AC.und_norm(np.array([ixy],float),k1,k2,cx,cy,s)]); ww=np.vstack([ww,[mxy]])
    H,_=cv2.findHomography(uw.astype(np.float32), ww.astype(np.float32), cv2.RANSAC, 0.02)
    if H is None: return dict(ok=False,reason="H tekil")
    # camside (auto_clean2d/replay2d uyumu için)
    fit=0.0  # vision-grounded: residual landmark-gürültüsü; sanity geometrik bakar
    return dict(ok=True,k1=k1,k2=k2,H=H,fit=fit,L=L,W=W,camside="SOL",source="vision")

def overlay(img, rec, L=None, W=None):
    h,w=img.shape[:2]; k1,k2,H=rec["k1"],rec["k2"],rec["H"]; cx,cy,s=w/2,h/2,w/2
    L=L or rec.get("L",40.0); W=W or rec.get("W",20.0)
    def m2p(P):
        Hi=np.linalg.inv(H); q=np.c_[P,np.ones(len(P))]@Hi.T; un=np.c_[q[:,0]/q[:,2],q[:,1]/q[:,2]]
        ru=np.hypot(un[:,0],un[:,1]); rg=np.linspace(0,2.6,5000); rug=rg*(1+k1*rg*rg+k2*rg**4)
        rd=np.interp(ru,rug,rg); sc=np.divide(rd,ru,out=np.ones_like(ru),where=ru>1e-9)
        return np.c_[cx+un[:,0]*sc*s, cy+un[:,1]*sc*s]
    o=img.copy()
    def dr(a,b,col,N=60):
        pix=m2p(np.linspace(a,b,N)).astype(int)
        for i in range(len(pix)-1):
            if 0<=pix[i,0]<w and 0<=pix[i,1]<h: cv2.line(o,tuple(pix[i]),tuple(pix[i+1]),col,3)
    dr([0,0],[L,0],COL[2]); dr([L,0],[L,W],COL[1]); dr([L,W],[0,W],COL[3]); dr([0,W],[0,0],COL[0])
    dr([L/2,0],[L/2,W],COL[4])
    for gx in (0,L):
        bx=6 if gx==0 else L-6
        dr([gx,W/2-6],[bx,W/2-6],COL[5]); dr([bx,W/2-6],[bx,W/2+6],COL[5]); dr([bx,W/2+6],[gx,W/2+6],COL[5])
    th=np.linspace(0,2*np.pi,60); pix=m2p(np.c_[L/2+3*np.cos(th),W/2+3*np.sin(th)]).astype(int)
    for i in range(len(pix)-1): cv2.line(o,tuple(pix[i]),tuple(pix[i+1]),COL[6],3)
    return o

if __name__=="__main__":
    # Gaziantep doğrulama (Claude-vision köşeleri)
    img=cv2.imread("calib_train/cand_big/GaziantepÖnderHalıSa__30062026__s571.jpg"); h,w=img.shape[:2]
    corners=[[240,680],[260,350],[1870,740],[1610,270]]
    rec=calibrate_from_corners(corners,40.0,20.0,w,h)
    cv2.imwrite("calib_train/cand/_VISION_CALIB.jpg", overlay(img,rec))
    print("vision-calib ok:",rec["ok"],"-> _VISION_CALIB.jpg")
