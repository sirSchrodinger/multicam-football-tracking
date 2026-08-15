#!/usr/bin/env python3
"""KESTİRME testi: bozulma ORTAK/bilinen (k1=0.167,k2=0.240). Yeni saha için sadece
köşe bulmak yeter mi? -> undistort -> yeşil-maske -> saha dörtgeni -> köşeler. Bunu 14
etiketli sahada KENDİ GT'ne (senin çizgilerinden) kıyaslıyorum. Düşük hata = ucuz otomasyon
çalışır, RunPod gerekmeyebilir. Çıktı: cand/_auto_detect.jpg + hata raporu.
"""
import json, os, numpy as np, cv2
HERE=os.path.dirname(os.path.abspath(__file__)); LF=os.path.join(HERE,"label_frames")
J=json.load(open(os.path.join(HERE,"calib_lines.json"))); ORDER=json.load(open(os.path.join(HERE,"frames.json")))
K1,K2=0.167,0.240; ZOOM=1.5
def idx_of(k): return ORDER.index(k)+1
def und_pts(pts,cx,cy,s):
    u=(pts[:,0]-cx)/s; vv=(pts[:,1]-cy)/s; r2=u*u+vv*vv; f=1+K1*r2+K2*r2*r2
    return np.stack([cx+u*f/ZOOM*s, cy+vv*f/ZOOM*s],1)
def undistort_img(img,cx,cy,s):
    h,w=img.shape[:2]; rd=np.linspace(0,2.2,2000); rru=rd*(1+K1*rd*rd+K2*rd**4)
    ys,xs=np.mgrid[0:h,0:w].astype(np.float32); uo=(xs-cx)/s*ZOOM; vo=(ys-cy)/s*ZOOM; ro=np.sqrt(uo*uo+vo*vo)
    rdv=np.interp(ro,rru,rd); sc=np.divide(rdv,ro,out=np.ones_like(ro),where=ro>1e-6)
    return cv2.remap(img,(cx+uo*sc*s).astype(np.float32),(cy+vo*sc*s).astype(np.float32),cv2.INTER_LINEAR)
def fitL(P):c=P.mean(0);_,_,vt=np.linalg.svd(P-c);nv=np.array([-vt[0,1],vt[0,0]]);return np.array([nv[0],nv[1],-nv.dot(c)])
def inter(a,b):D=a[0]*b[1]-b[0]*a[1];return np.array([(a[1]*b[2]-b[1]*a[2])/D,(b[0]*a[2]-a[0]*b[2])/D])
def order4(pts):  # TL,TR,BR,BL
    pts=np.array(pts,float); s=pts.sum(1); d=np.diff(pts,1).ravel()
    return np.array([pts[s.argmin()],pts[d.argmin()],pts[s.argmax()],pts[d.argmax()]])

def auto_corners(und):
    hsv=cv2.cvtColor(und,cv2.COLOR_BGR2HSV); H,Sx,V=hsv[...,0],hsv[...,1],hsv[...,2]
    g=((H>=30)&(H<=95)&(Sx>30)&(V>30)).astype(np.uint8)*255
    g=cv2.morphologyEx(g,cv2.MORPH_CLOSE,np.ones((31,31),np.uint8))
    g=cv2.morphologyEx(g,cv2.MORPH_OPEN,np.ones((21,21),np.uint8))
    nc,lbl,st,_=cv2.connectedComponentsWithStats(g)
    if nc<2: return None,g
    big=1+np.argmax(st[1:,cv2.CC_STAT_AREA]); g=(lbl==big).astype(np.uint8)*255
    g=cv2.morphologyEx(g,cv2.MORPH_CLOSE,np.ones((51,51),np.uint8))
    cnts,_=cv2.findContours(g,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None,g
    c=max(cnts,key=cv2.contourArea); peri=cv2.arcLength(c,True)
    quad=None
    for e in np.linspace(0.01,0.08,16):
        ap=cv2.approxPolyDP(c,e*peri,True)
        if len(ap)==4: quad=ap.reshape(4,2).astype(float); break
    if quad is None:
        box=cv2.boxPoints(cv2.minAreaRect(c)); quad=box.astype(float)
    return order4(quad),g

thumbs=[]; rep=[]
for name in ORDER:
    if name not in J or not all(J[name]["lines"].get(x) for x in ("goalN","goalF","touchN","touchF")): continue
    img=cv2.imread(os.path.join(LF,name)); h,w=img.shape[:2]; cx,cy,s=w/2,h/2,w/2; n=idx_of(name); ln=J[name]["lines"]
    und=undistort_img(img,cx,cy,s)
    # GT köşeler (senin çizgilerinden, ortak-undistort uzayında)
    fl={l:fitL(und_pts(np.asarray(ln[l],float),cx,cy,s)) for l in ("goalN","goalF","touchN","touchF")}
    gt=order4([inter(fl["goalN"],fl["touchN"]),inter(fl["goalN"],fl["touchF"]),inter(fl["goalF"],fl["touchN"]),inter(fl["goalF"],fl["touchF"])])
    auto,gmask=auto_corners(und)
    vis=und.copy()
    err=None
    if auto is not None:
        # eşle: her GT köşesine en yakın auto köşe
        d=np.linalg.norm(gt[:,None,:]-auto[None,:,:],axis=2); err=float(d.min(1).mean())
        cv2.polylines(vis,[auto.astype(np.int32)],True,(0,0,255),3)
        for p in auto: cv2.circle(vis,tuple(p.astype(int)),10,(0,0,255),-1)
    for p in gt: cv2.circle(vis,tuple(p.astype(int)),9,(0,255,0),2)
    diag=np.linalg.norm(gt[0]-gt[2])
    rep.append((n,name,err,err/diag*100 if err else None))
    sc=560/w; vis=cv2.resize(vis,(int(w*sc),int(h*sc)))
    cv2.putText(vis,f"{n}. {name[:14]} hata={err:.0f}px ({err/diag*100:.0f}%)" if err else f"{n}. {name[:14]} BULUNAMADI",
                (6,24),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,230,255),2)
    thumbs.append((n,vis))
print("=== otomatik köşe-tespiti vs GT (kırmızı=auto, yeşil=GT) ===")
for n,nm,err,pct in sorted(rep):
    print(f"  {n:2d} {nm[:18]:18s} "+(f"hata {err:6.1f}px  ({pct:4.1f}% köşegen)" if err else "BULUNAMADI"))
oks=[p for _,_,e,p in rep if p is not None and p<5]
print(f"\n<%5 köşegen hata (iyi): {len(oks)}/{len(rep)} saha")
thumbs.sort(key=lambda t:t[0]); cols=4; rows=(len(thumbs)+cols-1)//cols
H0,Wt=thumbs[0][1].shape[:2]; sheet=np.full((rows*(H0+6),cols*(Wt+6),3),20,np.uint8)
for i,(n,t) in enumerate(thumbs):
    rr,cc=divmod(i,cols); sheet[rr*(H0+6):rr*(H0+6)+t.shape[0],cc*(Wt+6):cc*(Wt+6)+t.shape[1]]=t
cv2.imwrite(os.path.join(HERE,"cand","_auto_detect.jpg"),sheet); print("-> cand/_auto_detect.jpg")
