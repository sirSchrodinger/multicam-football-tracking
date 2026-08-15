#!/usr/bin/env python3
"""L_geo: GT-siz geometrik tutarlılık skorları (kullanıcı sinyalleri). Doğru calib -> düşük
skor, bozuk calib -> yüksek skor. Self-sup kaybının + watchdog'un + self-train kabul-kapısının
çekirdeği. Bu sürüm SKORLAMA + DOĞRULAMA (numpy); diferansiyellenebilir torch sürümü adım 3'te.

Skorlar (hepsi düşük=iyi):
 template : warp edilmiş beyaz-çizgiler kanonik 34xW şablona ne kadar uzak (chamfer, m) — "2D gerçek sahaya benzer mi"
 stripe   : top-down'da biçme şeritleri dikeyden sapma (derece)
 boxsym   : iki ceza-sahası simetri ihlali |boxN_X+boxF_X-L| (m)
 sign     : kamera-yön belirsizliği (dejenere=yüksek)
Doğrulama: 26 sahada DOĞRU calib vs BOZULMUŞ (H perturbe) -> doğru < bozuk olmalı.
"""
import json, os, numpy as np, cv2
HERE=os.path.dirname(os.path.abspath(__file__)); LF=os.path.join(HERE,"label_frames")
J=json.load(open(os.path.join(HERE,"calib_lines.json"))); C=json.load(open(os.path.join(HERE,"calib_solved.json")))
ORDER=json.load(open(os.path.join(HERE,"frames.json"))); L,S=34.0,22
def und_pix(P,k1,k2,cx,cy,s,z=1.5): u=(P[:,0]-cx)/s;v=(P[:,1]-cy)/s;r2=u*u+v*v;f=1+k1*r2+k2*r2*r2;return np.stack([cx+u*f/z*s,cy+v*f/z*s],1)
def undimg(img,k1,k2,cx,cy,s,z=1.5):
    h,w=img.shape[:2]; rd=np.linspace(0,2.4,2000); rru=rd*(1+k1*rd*rd+k2*rd**4)
    ys,xs=np.mgrid[0:h,0:w].astype(np.float32); uo=(xs-cx)/s*z; vo=(ys-cy)/s*z; ro=np.sqrt(uo*uo+vo*vo)
    rdv=np.interp(ro,rru,rd); sc=np.divide(rdv,ro,out=np.ones_like(ro),where=ro>1e-6)
    return cv2.remap(img,(cx+uo*sc*s).astype(np.float32),(cy+vo*sc*s).astype(np.float32),cv2.INTER_LINEAR,borderValue=(0,0,0))
def fitL(P):c=P.mean(0);_,_,vt=np.linalg.svd(P-c);nv=np.array([-vt[0,1],vt[0,0]]);return np.array([nv[0],nv[1],-nv.dot(c)])
def inter(a,b):D=a[0]*b[1]-b[0]*a[1];return np.array([(a[1]*b[2]-b[1]*a[2])/D,(b[0]*a[2]-a[0]*b[2])/D])

def template_edge(Wp):
    """Kanonik üstten-görünüm halısaha çizgileri (boundary+orta+iki ceza+yuvarlak) -> kenar maskesi."""
    LS,WS=int(L*S),int(Wp*S); t=np.zeros((WS,LS),np.uint8)
    cv2.rectangle(t,(1,1),(LS-2,WS-2),255,1); cv2.line(t,(LS//2,0),(LS//2,WS),255,1)
    cv2.circle(t,(LS//2,WS//2),int(3*S),255,1)
    for gx in (0,L):
        bx=5 if gx==0 else L-5
        cv2.rectangle(t,(int(min(gx,bx)*S),int((Wp/2-5)*S)),(int(max(gx,bx)*S),int((Wp/2+5)*S)),255,1)
    return t

def warp_topdown(name,k1,k2,H_override=None):
    r=C[name]; img=cv2.imread(os.path.join(LF,name)); h,w=r["h"],r["w"]; cx,cy,s=w/2,h/2,w/2; Wp=r.get("Wp",18.0); ln=J[name]["lines"]
    und=undimg(img,k1,k2,cx,cy,s)
    def u2p(p): return und_pix(np.asarray(p,float),k1,k2,cx,cy,s)
    fl={l:fitL(u2p(ln[l])) for l in ("goalN","goalF","touchN","touchF")}
    src=np.array([inter(fl["goalN"],fl["touchN"]),inter(fl["goalN"],fl["touchF"]),inter(fl["goalF"],fl["touchN"]),inter(fl["goalF"],fl["touchF"])],float)
    LS,WS=int(L*S),int(Wp*S); dst=np.array([[0,WS],[0,0],[LS,WS],[LS,0]],float)
    if H_override is not None: dst=(H_override@np.c_[dst,np.ones(4)].T).T; dst=dst[:,:2]/dst[:,2:]
    Hm,_=cv2.findHomography(src,dst); top=cv2.warpPerspective(und,Hm,(LS,WS),borderValue=(0,0,0))
    return top,Wp,Hm

def white_edge(top):
    hsv=cv2.cvtColor(top,cv2.COLOR_BGR2HSV); wht=((hsv[...,1]<90)&(hsv[...,2]>140)).astype(np.uint8)*255
    th=cv2.morphologyEx(cv2.cvtColor(top,cv2.COLOR_BGR2GRAY),cv2.MORPH_TOPHAT,cv2.getStructuringElement(cv2.MORPH_RECT,(15,15)))
    _,tm=cv2.threshold(th,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return cv2.bitwise_and(wht,tm)

def score(name,k1,k2,H_override=None):
    top,Wp,Hm=warp_topdown(name,k1,k2,H_override); LS,WS=int(L*S),int(Wp*S)
    # template chamfer (m): beyaz-kenar -> kanonik şablona uzaklık
    tmpl=template_edge(Wp); dt=cv2.distanceTransform(255-tmpl,cv2.DIST_L2,3)
    we=white_edge(top); d=dt[we>0]
    templ=float(np.median(d))/S if len(d)>30 else 9.9   # metre
    # stripe diklik (derece)
    inr=top[int(WS*.2):int(WS*.8),int(LS*.16):int(LS*.84)]
    g=cv2.GaussianBlur(cv2.cvtColor(inr,cv2.COLOR_BGR2GRAY),(0,0),2).astype(np.float32)
    Gx=cv2.Sobel(g,cv2.CV_32F,1,0);Gy=cv2.Sobel(g,cv2.CV_32F,0,1);mg=np.sqrt(Gx*Gx+Gy*Gy);mk=mg>np.percentile(mg,82)
    Jxx=(Gx[mk]**2).sum();Jyy=(Gy[mk]**2).sum();Jxy=(Gx[mk]*Gy[mk]).sum()
    stp=abs(((np.degrees(0.5*np.arctan2(2*Jxy,Jxx-Jyy))+90)%180)-90)
    # box-simetri (m): calib_solved'daki boxN_X+boxF_X
    r=C[name]; bsym=abs((r.get("boxN_X") or 0)+(r.get("boxF_X") or L)-L) if r.get("boxN_X") is not None else np.nan
    return dict(templ=templ,stripe=stp,boxsym=bsym)

if __name__=="__main__":
    names=[n for n in ORDER if n in C and not C[n].get("bad")]
    print(f"{'saha':18s} {'DOĞRU template/stripe':22s} {'BOZUK template/stripe':22s}")
    rng=np.random.RandomState(0); okc=0; tot=0
    for n in names[:14]:
        r=C[n]
        s_ok=score(n,r["k1"],r["k2"])
        # bozulmuş: top-down'ı hafif projektif perturbe et (yanlış calib taklidi)
        Hp=np.eye(3); Hp[0,1]=rng.uniform(0.12,0.22)*rng.choice([-1,1]); Hp[2,0]=rng.uniform(0.0006,0.0012)*rng.choice([-1,1])
        s_bad=score(n,r["k1"],r["k2"],H_override=Hp)
        better = s_ok["templ"]<s_bad["templ"]
        okc+=better; tot+=1
        print(f"{n[:18]:18s} t={s_ok['templ']:.2f} s={s_ok['stripe']:4.0f} | t={s_bad['templ']:.2f} s={s_bad['stripe']:4.0f}  {'OK' if better else 'X'}")
    print(f"\ntemplate doğru<bozuk: {okc}/{tot} (geometrik skor calib kalitesini AYIRT ediyor mu)")
