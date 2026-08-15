#!/usr/bin/env python3
"""2D DOĞRULAMA v2: kamera-çapalı (sol/sağ-alt OTOMATİK) + sütun-DİKLİK + sütun-ARALIK-eşitliği
+ yuvarlak-doğruluğu (gerçek yoksa VARSAYILAN nominal yuvarlak). Uzak kısımları koyultma YOK.
Kullanım: python verify_2d.py n1 n2 ...
"""
import sys, json, os, numpy as np, cv2
from scipy.signal import find_peaks
HERE=os.path.dirname(os.path.abspath(__file__)); LF=os.path.join(HERE,"label_frames")
J=json.load(open(os.path.join(HERE,"calib_lines.json"))); C=json.load(open(os.path.join(HERE,"calib_solved.json")))
ORDER=json.load(open(os.path.join(HERE,"frames.json"))); L,S,M=34.0,24,5.0
def name_of(n): return ORDER[n-1]
def und_pix(P,k1,k2,cx,cy,s,z=1.5): u=(P[:,0]-cx)/s;v=(P[:,1]-cy)/s;r2=u*u+v*v;f=1+k1*r2+k2*r2*r2;return np.stack([cx+u*f/z*s,cy+v*f/z*s],1)
def undimg(img,k1,k2,cx,cy,s,z=1.5):
    h,w=img.shape[:2]; rd=np.linspace(0,2.4,2200); rru=rd*(1+k1*rd*rd+k2*rd**4)
    ys,xs=np.mgrid[0:h,0:w].astype(np.float32); uo=(xs-cx)/s*z; vo=(ys-cy)/s*z; ro=np.sqrt(uo*uo+vo*vo)
    rdv=np.interp(ro,rru,rd); sc=np.divide(rdv,ro,out=np.ones_like(ro),where=ro>1e-6)
    return cv2.remap(img,(cx+uo*sc*s).astype(np.float32),(cy+vo*sc*s).astype(np.float32),cv2.INTER_LINEAR,borderValue=(0,0,0))
def fitL(P):c=P.mean(0);_,_,vt=np.linalg.svd(P-c);nv=np.array([-vt[0,1],vt[0,0]]);return np.array([nv[0],nv[1],-nv.dot(c)])
def inter(a,b):D=a[0]*b[1]-b[0]*a[1];return np.array([(a[1]*b[2]-b[1]*a[2])/D,(b[0]*a[2]-a[0]*b[2])/D])

def render(n):
    name=name_of(n); r=C[name]; img=cv2.imread(os.path.join(LF,name)); h,w=r["h"],r["w"]; cx,cy,s=w/2,h/2,w/2
    k1,k2=r["k1"],r["k2"]; Wp=r.get("Wp",18.0); ln=J[name]["lines"]
    und=undimg(img,k1,k2,cx,cy,s)
    def u2p(p): return und_pix(np.asarray(p,float),k1,k2,cx,cy,s)
    fl={l:fitL(u2p(ln[l])) for l in ("goalN","goalF","touchN","touchF")}
    c_nN=inter(fl["goalN"],fl["touchN"]);c_nF=inter(fl["goalN"],fl["touchF"]);c_fN=inter(fl["goalF"],fl["touchN"]);c_fF=inter(fl["goalF"],fl["touchF"])
    LS,WS=int((L+2*M)*S),int((Wp+2*M)*S)
    def MXl(X,Y): return [(M+X)*S, WS-(M+Y)*S]
    def MXr(X,Y): return [(M+L-X)*S, WS-(M+Y)*S]
    src=np.array([c_nN,c_nF,c_fN,c_fF],float)
    A=c_fN-c_nN; B=c_nF-c_nN; cross=A[0]*B[1]-A[1]*B[0]   # cross>0 => kamera SAĞ (26/26 doğrulandı)
    mirrored=(cross>0); MX=MXr if mirrored else MXl; camside="SAĞ-alt" if mirrored else "SOL-alt"
    Hm,_=cv2.findHomography(src,np.array([MX(0,0),MX(0,Wp),MX(L,0),MX(L,Wp)],float))
    top=cv2.warpPerspective(und,Hm,(LS,WS),borderValue=(0,0,0))
    top[top.sum(2)<10]=(22,22,22)   # görünmeyen=veri yok (koyultma YOK, sadece boş)
    cv2.rectangle(top,tuple(map(int,MX(0,Wp))),tuple(map(int,MX(L,0))),(0,215,255),2)
    cv2.line(top,tuple(map(int,MX(L/2,0))),tuple(map(int,MX(L/2,Wp))),(0,215,255),1)
    cam=MX(0,0); cv2.circle(top,tuple(map(int,cam)),11,(0,200,255),-1); cv2.putText(top,"KAMERA",(int(cam[0])+8,int(cam[1])-6),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,200,255),2)
    # --- sütun DİKLİK + ARALIK eşitliği (iç bölge) ---
    x0,x1,y0,y1=int(LS*.16),int(LS*.84),int(WS*.22),int(WS*.78); inr=top[y0:y1,x0:x1]
    gray=cv2.GaussianBlur(cv2.cvtColor(inr,cv2.COLOR_BGR2GRAY),(0,0),2).astype(np.float32)
    Gx=cv2.Sobel(gray,cv2.CV_32F,1,0);Gy=cv2.Sobel(gray,cv2.CV_32F,0,1);mg=np.sqrt(Gx*Gx+Gy*Gy);mk=mg>np.percentile(mg,82)
    Jxx=(Gx[mk]**2).sum();Jyy=(Gy[mk]**2).sum();Jxy=(Gx[mk]*Gy[mk]).sum()
    stripe=(np.degrees(0.5*np.arctan2(2*Jxy,Jxx-Jyy))+90)%180; sdev=abs(stripe-90)
    prof=cv2.GaussianBlur(gray.mean(0).reshape(1,-1),(0,0),3).ravel(); d=np.abs(np.gradient(prof))
    pk,_=find_peaks(d,distance=10,height=d.max()*0.30)
    cvsp = (np.diff(pk).std()/(np.diff(pk).mean()+1e-6)) if len(pk)>=3 else None
    for px in pk: cv2.line(top,(x0+px,y0),(x0+px,y1),(255,120,255),1)   # tespit edilen şerit sınırları
    # --- yuvarlak: gerçek mi? değilse VARSAYILAN ---
    crot="varsayilan(3m)"; cpts=ln.get("circle"); real=False
    if cpts and len(cpts)>=8:
        un=u2p(cpts); Ph=np.concatenate([un,np.ones((len(un),1))],1).T; q=Hm@Ph; q=(q[:2]/q[2]).T
        try:
            (ex,ey),(MA,ma),ang=cv2.fitEllipse(q.astype(np.float32)); ecc=min(MA,ma)/max(MA,ma); rr=(MA+ma)/4/S
            if ecc>0.70: real=True; crot=f"GERCEK yuv {ecc:.2f} R={rr:.1f}m"; cv2.ellipse(top,(int(ex),int(ey)),(int(MA/2),int(ma/2)),ang,0,360,(255,160,0),2)
        except Exception: pass
    if not real:
        cc=MX(L/2,Wp/2); cv2.circle(top,tuple(map(int,cc)),int(3*S),(150,120,40),1,cv2.LINE_AA)  # ince=varsayılan
    sp=f"{cvsp*100:.0f}%" if cvsp is not None else "desen-yok"
    cv2.putText(top,f"{n}.{name[:12]} kam={camside}  diklik={sdev:.0f}deg  aralik-CV={sp}  yuv:{crot}",
                (8,26),cv2.FONT_HERSHEY_SIMPLEX,0.58,(255,255,255),2)
    return top,(name,camside,sdev,sp,crot)

ns=[int(x) for x in (sys.argv[1:] or [21,12,9])]
tiles=[]; print("=== doğrulama v2 (diklik°, aralik-CV%, yuvarlak) ===")
for n in ns:
    if name_of(n) not in C: continue
    t,info=render(n); tiles.append(t); print(f"  {n} {info[0][:16]:16s} kam={info[1]:7s} diklik={info[2]:.0f}° aralik-CV={info[3]:7s} {info[4]}")
Wm=max(t.shape[1] for t in tiles); sheet=np.full((sum(t.shape[0]+8 for t in tiles),Wm,3),20,np.uint8); y=0
for t in tiles: sheet[y:y+t.shape[0],0:t.shape[1]]=t; y+=t.shape[0]+8
cv2.imwrite(os.path.join(HERE,"cand","_verify_2d.jpg"),sheet); print("-> cand/_verify_2d.jpg")
