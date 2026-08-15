#!/usr/bin/env python3
"""DOĞRU 2D: elle-etiketli kalibrasyon, KAMERA-ÇAPALI yön (kamera köşesi = sol-alt) +
görünmeyen (kameranın göremediği / kadraj-dışı) bölge işaretli. Doğrulama için referans.
Kullanım: python render_correct_2d.py n1 n2 ...  (1-based, varsayılan Avanos+Kaynarca)
"""
import sys, json, os, numpy as np, cv2
HERE=os.path.dirname(os.path.abspath(__file__)); LF=os.path.join(HERE,"label_frames")
J=json.load(open(os.path.join(HERE,"calib_lines.json"))); C=json.load(open(os.path.join(HERE,"calib_solved.json")))
ORDER=json.load(open(os.path.join(HERE,"frames.json"))); L,S,M=34.0,26,5.0
def name_of(n): return ORDER[n-1]
def und_pix(P,k1,k2,cx,cy,s,z=1.5): u=(P[:,0]-cx)/s;v=(P[:,1]-cy)/s;r2=u*u+v*v;f=1+k1*r2+k2*r2*r2;return np.stack([cx+u*f/z*s,cy+v*f/z*s],1)
def undimg(img,k1,k2,cx,cy,s,z=1.5):
    h,w=img.shape[:2]; rd=np.linspace(0,2.4,2200); rru=rd*(1+k1*rd*rd+k2*rd**4)
    ys,xs=np.mgrid[0:h,0:w].astype(np.float32); uo=(xs-cx)/s*z; vo=(ys-cy)/s*z; ro=np.sqrt(uo*uo+vo*vo)
    rdv=np.interp(ro,rru,rd); sc=np.divide(rdv,ro,out=np.ones_like(ro),where=ro>1e-6)
    return cv2.remap(img,(cx+uo*sc*s).astype(np.float32),(cy+vo*sc*s).astype(np.float32),cv2.INTER_LINEAR,borderValue=(0,0,0))
def fitL(P):c=P.mean(0);_,_,vt=np.linalg.svd(P-c);nv=np.array([-vt[0,1],vt[0,0]]);return np.array([nv[0],nv[1],-nv.dot(c)])
def inter(a,b):D=a[0]*b[1]-b[0]*a[1];return np.array([(a[1]*b[2]-b[1]*a[2])/D,(b[0]*a[2]-a[0]*b[2])/D])

def render(name):
    r=C[name]; img=cv2.imread(os.path.join(LF,name)); h,w=r["h"],r["w"]; cx,cy,s=w/2,h/2,w/2
    k1,k2=r["k1"],r["k2"]; Wp=r.get("Wp",18.0); ln=J[name]["lines"]
    und=undimg(img,k1,k2,cx,cy,s)
    def u2p(p): return und_pix(np.asarray(p,float),k1,k2,cx,cy,s)
    fl={l:fitL(u2p(ln[l])) for l in ("goalN","goalF","touchN","touchF")}
    # KAMERA-ÇAPALI: c_nN(yakın kale ∩ yakın taç = kamera köşesi) -> SOL-ALT
    c_nN=inter(fl["goalN"],fl["touchN"]); c_nF=inter(fl["goalN"],fl["touchF"])
    c_fN=inter(fl["goalF"],fl["touchN"]); c_fF=inter(fl["goalF"],fl["touchF"])
    LS,WS=int((L+2*M)*S),int((Wp+2*M)*S)
    def MXl(X,Y): return [(M+X)*S, WS-(M+Y)*S]
    def MXr(X,Y): return [(M+L-X)*S, WS-(M+Y)*S]
    A=c_fN-c_nN; B=c_nF-c_nN; cross=A[0]*B[1]-A[1]*B[0]   # cross>0 => kamera SAĞ-alt
    MX = MXr if cross>0 else MXl
    src=np.array([c_nN,c_nF,c_fN,c_fF],float)
    dst=np.array([MX(0,0),MX(0,Wp),MX(L,0),MX(L,Wp)],float)
    Hm,_=cv2.findHomography(src,dst); top=cv2.warpPerspective(und,Hm,(LS,WS),borderValue=(0,0,0))
    # görünmeyen bölge: warp'ta siyah (kaynak kadraj dışı) -> kırmızımsı tint
    blind=(top.sum(2)<12)
    tint=top.copy(); tint[blind]=(0,0,90)
    top=cv2.addWeighted(top,1.0,tint,0.0,0); top[blind]=(35,35,80)
    cv2.rectangle(top,tuple(map(int,MX(0,Wp))),tuple(map(int,MX(L,0))),(0,215,255),2)
    cv2.line(top,tuple(map(int,MX(L/2,0))),tuple(map(int,MX(L/2,Wp))),(0,215,255),1)
    # kamera işareti (sol-alt, c_nN)
    cam=MX(0,0); cv2.circle(top,tuple(map(int,cam)),12,(0,200,255),-1)
    cv2.putText(top,"KAMERA",(int(cam[0])+10,int(cam[1])-6),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,200,255),2)
    cv2.putText(top,f"{r['n']}. {name[:14]} W/L={Wp/L:.2f}  [kirmizi=gorunmeyen]",(8,28),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,255,255),2)
    return top

ns=[int(x) for x in (sys.argv[1:] or [9,21])]  # Avanos, Kaynarca
tiles=[render(name_of(n)) for n in ns if name_of(n) in C]
Wm=max(t.shape[1] for t in tiles); sheet=np.full((sum(t.shape[0]+8 for t in tiles),Wm,3),20,np.uint8); y=0
for t in tiles: sheet[y:y+t.shape[0],0:t.shape[1]]=t; y+=t.shape[0]+8
cv2.imwrite(os.path.join(HERE,"cand","_correct_2d.jpg"),sheet); print("-> cand/_correct_2d.jpg",sheet.shape)
