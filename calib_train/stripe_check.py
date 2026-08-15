#!/usr/bin/env python3
"""Şerit-dikliği öz-denetimi: koyu/açık biçme şeritleri dünyada paralel düz çizgiler ->
doğru 2D top-down'da DİKEY olmalı. Top-down'da baskın gradyan yönünü (structure tensor)
ölçüp şerit açısının 90°(dikey)'e ne kadar yakın olduğunu raporlar. Bedava kalibrasyon kanıtı.
"""
import json, os, numpy as np, cv2
HERE=os.path.dirname(os.path.abspath(__file__)); LF=os.path.join(HERE,"label_frames")
J=json.load(open(os.path.join(HERE,"calib_lines.json"))); C=json.load(open(os.path.join(HERE,"calib_solved.json")))
ORDER=json.load(open(os.path.join(HERE,"frames.json"))); L=34.0; S=26
def idx_of(k): return ORDER.index(k)+1
def und_pts(pts,k1,k2,cx,cy,s,z=1.5): u=(pts[:,0]-cx)/s;vv=(pts[:,1]-cy)/s;r2=u*u+vv*vv;f=1+k1*r2+k2*r2*r2;return np.stack([cx+u*f/z*s,cy+vv*f/z*s],1)
def undimg(img,k1,k2,cx,cy,s,z=1.5):
    h,w=img.shape[:2]; rd=np.linspace(0,2.2,2000); rru=rd*(1+k1*rd*rd+k2*rd**4)
    ys,xs=np.mgrid[0:h,0:w].astype(np.float32); uo=(xs-cx)/s*z; vo=(ys-cy)/s*z; ro=np.sqrt(uo*uo+vo*vo)
    rdv=np.interp(ro,rru,rd); sc=np.divide(rdv,ro,out=np.ones_like(ro),where=ro>1e-6)
    return cv2.remap(img,(cx+uo*sc*s).astype(np.float32),(cy+vo*sc*s).astype(np.float32),cv2.INTER_LINEAR)
def fitL(P):c=P.mean(0);_,_,vt=np.linalg.svd(P-c);nv=np.array([-vt[0,1],vt[0,0]]);return np.array([nv[0],nv[1],-nv.dot(c)])
def inter(a,b):D=a[0]*b[1]-b[0]*a[1];return np.array([(a[1]*b[2]-b[1]*a[2])/D,(b[0]*a[2]-a[0]*b[2])/D])

print("=== şerit açısı (90°=dikey ideal) — biçme şeritleri belirgin sahalar ===")
for name in ORDER:
    if name not in C: continue
    r=C[name]; n=r["n"]; Wp=r.get("Wp",18.0); img=cv2.imread(os.path.join(LF,name)); h,w=r["h"],r["w"]; cx,cy,s=w/2,h/2,w/2
    und=undimg(img,r["k1"],r["k2"],cx,cy,s); ln=J[name]["lines"]
    fl={l:fitL(und_pts(np.asarray(ln[l],float),r["k1"],r["k2"],cx,cy,s)) for l in ("goalN","goalF","touchN","touchF")}
    src=np.array([inter(fl["goalN"],fl["touchN"]),inter(fl["goalN"],fl["touchF"]),inter(fl["goalF"],fl["touchN"]),inter(fl["goalF"],fl["touchF"])])
    LS,WS=int(L*S),int(Wp*S); Hm,_=cv2.findHomography(src,np.array([[0,WS],[0,0],[LS,WS],[LS,0]],float))
    top=cv2.warpPerspective(und,Hm,(LS,WS))
    int_=top[int(WS*.18):int(WS*.82), int(LS*.12):int(LS*.88)]
    gray=cv2.GaussianBlur(cv2.cvtColor(int_,cv2.COLOR_BGR2GRAY),(0,0),2.0).astype(np.float32)
    Gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3); Gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    mag=np.sqrt(Gx*Gx+Gy*Gy); m=mag>np.percentile(mag,80)
    Jxx=(Gx[m]**2).sum(); Jyy=(Gy[m]**2).sum(); Jxy=(Gx[m]*Gy[m]).sum()
    grad_ang=np.degrees(0.5*np.arctan2(2*Jxy,Jxx-Jyy))   # baskın gradyan yönü
    stripe_ang=(grad_ang+90)%180                          # şerit çizgi yönü
    dev=min(abs(stripe_ang-90),abs(stripe_ang-90))        # dikeyden sapma
    coher=np.hypot(Jxx-Jyy,2*Jxy)/(Jxx+Jyy+1e-9)          # ne kadar yönlü
    flag="şerit belirgin" if coher>0.25 else "şerit zayıf"
    print(f"  {n:2d} {name[:18]:18s} şerit_açısı={stripe_ang:5.1f}° (dikeyden {abs(stripe_ang-90):4.1f}° sapma)  coher={coher:.2f}  {flag}")
