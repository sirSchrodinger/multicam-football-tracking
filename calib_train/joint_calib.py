#!/usr/bin/env python3
"""BİRLEŞİK kalibrasyon: bozulma(k1,k2, merkez=görüntü ortası) + homografi(H) AYNI ANDA,
kısıtlar METRİK uzayda: touchN->Y=0, touchF->Y=W, goalN->X=0, goalF->X=L, orta->X=L/2.
Eğri kenarın düz-metrik-çizgiye oturma zorunluluğu lensi tanımlar; H perspektifi alır.
Çıktı: calib_solved.json (saha başına k1,k2,H,residual) + cand/_2d_joint.jpg
"""
import json, os
import numpy as np, cv2
from scipy.optimize import least_squares

HERE=os.path.dirname(os.path.abspath(__file__)); LF=os.path.join(HERE,"label_frames")
J=json.load(open(os.path.join(HERE,"calib_lines.json"))); ORDER=json.load(open(os.path.join(HERE,"frames.json")))
L,W,S=34.0,18.0,26; LS,WS=int(L*S),int(W*S); NO_CENTER={3,7,9}
# (line_id -> (axis, target))  axis 0=X(uzunluk) 1=Y(genişlik)
CONSTR={"goalN":(0,0.0),"goalF":(0,L),"touchN":(1,0.0),"touchF":(1,W),"center":(0,L/2)}
def idx_of(k): return ORDER.index(k)+1 if k in ORDER else -1

def und_norm(pts,k1,k2,cx,cy,s):
    u=(pts[:,0]-cx)/s; vv=(pts[:,1]-cy)/s; r2=u*u+vv*vv; f=1+k1*r2+k2*r2*r2
    return np.stack([u*f,vv*f],1)
def applyH(uv,H):
    z=H[2,0]*uv[:,0]+H[2,1]*uv[:,1]+H[2,2]
    X=(H[0,0]*uv[:,0]+H[0,1]*uv[:,1]+H[0,2])/z
    Y=(H[1,0]*uv[:,0]+H[1,1]*uv[:,1]+H[1,2])/z
    return np.stack([X,Y],1)
def fitL(P): c=P.mean(0);_,_,vt=np.linalg.svd(P-c);nv=np.array([-vt[0,1],vt[0,0]]);return np.array([nv[0],nv[1],-nv.dot(c)])
def inter(a,b):
    a1,b1,c1=a;a2,b2,c2=b;D=a1*b2-a2*b1
    return None if abs(D)<1e-9 else np.array([(b1*c2-b2*c1)/D,(a2*c1-a1*c2)/D])

def solve_venue(name):
    v=J[name]; img=cv2.imread(os.path.join(LF,name)); h,w=img.shape[:2]; n=idx_of(name)
    cx,cy,s=w/2,h/2,w/2; ln=v["lines"]
    center_real = bool(ln.get("center") and len(ln["center"])>=4)   # tahmini orta-çizgi nokta-sayısından
    cons=[]   # (pts_norm-ham, axis, target)
    for lid,(ax,tg) in CONSTR.items():
        if lid=="center" and not center_real: continue
        p=ln.get(lid)
        if p and len(p)>=2: cons.append((np.asarray(p,float),ax,tg))
    # init H: k=0 köşeler -> metrik
    def corners_k(k1,k2):
        fl={l:fitL(und_norm(np.asarray(ln[l],float),k1,k2,cx,cy,s)) for l in ("goalN","goalF","touchN","touchF") if ln.get(l)}
        return (inter(fl["goalN"],fl["touchN"]),inter(fl["goalN"],fl["touchF"]),
                inter(fl["goalF"],fl["touchN"]),inter(fl["goalF"],fl["touchF"]))
    c0=corners_k(0,0); src=np.array(c0,float); dst=np.array([[0,0],[0,W],[L,0],[L,W]],float)
    H0,_=cv2.findHomography(src,dst);
    p0=[0.0,0.0]+list((H0/H0[2,2]).ravel()[:8])
    def resid(p):
        k1,k2=p[0],p[1]; H=np.array([[p[2],p[3],p[4]],[p[5],p[6],p[7]],[p[8],p[9],1.0]])
        out=[]
        for pts,ax,tg in cons:
            m=applyH(und_norm(pts,k1,k2,cx,cy,s),H); out.append(m[:,ax]-tg)
        return np.concatenate(out)
    base=np.sqrt((resid(p0)**2).mean())
    sol=least_squares(resid,p0,method="lm",max_nfev=20000)
    fit=np.sqrt((resid(sol.x)**2).mean())
    k1,k2=sol.x[0],sol.x[1]; H=np.array([[sol.x[2],sol.x[3],sol.x[4]],[sol.x[5],sol.x[6],sol.x[7]],[sol.x[8],sol.x[9],1.0]])
    # kutu derinlikleri (TANI: simetri-kısıtı YOK, sadece nereye düştüğünü ölç)
    def bx(lid):
        p=ln.get(lid)
        if p and len(p)>=2: return float(np.median(applyH(und_norm(np.asarray(p,float),k1,k2,cx,cy,s),H)[:,0]))
        return None
    return dict(name=name,n=n,w=w,h=h,k1=k1,k2=k2,H=H.tolist(),base=base,fit=fit,
                has_center=center_real,boxN_X=bx("boxN"),boxF_X=bx("boxF"))

def render(rec):
    name=rec["name"]; img=cv2.imread(os.path.join(LF,name)); h,w=rec["h"],rec["w"]
    cx,cy,s=w/2,h/2,w/2; k1,k2=rec["k1"],rec["k2"]; ln=J[name]["lines"]
    # ters radyal LUT
    rd=np.linspace(0,1.9,1600); ru=rd*(1+k1*rd*rd+k2*rd**4)
    zoom=1.45; ys,xs=np.mgrid[0:h,0:w].astype(np.float32)
    uo=(xs-cx)/s*zoom; vo=(ys-cy)/s*zoom; ro=np.sqrt(uo*uo+vo*vo)
    rdv=np.interp(ro,ru,rd); sc=np.divide(rdv,ro,out=np.ones_like(ro),where=ro>1e-6)
    und=cv2.remap(img,(cx+uo*sc*s).astype(np.float32),(cy+vo*sc*s).astype(np.float32),cv2.INTER_LINEAR)
    def u2p(pts):
        un=und_norm(np.asarray(pts,float),k1,k2,cx,cy,s); return np.stack([cx+un[:,0]/zoom*s,cy+un[:,1]/zoom*s],1)
    fl={l:fitL(u2p(ln[l])) for l in ("goalN","goalF","touchN","touchF")}
    src=np.array([inter(fl["goalN"],fl["touchN"]),inter(fl["goalN"],fl["touchF"]),
                  inter(fl["goalF"],fl["touchN"]),inter(fl["goalF"],fl["touchF"])],float)
    Hm,_=cv2.findHomography(src,np.array([[0,WS],[0,0],[LS,WS],[LS,0]],float))
    top=cv2.warpPerspective(und,Hm,(LS,WS))
    cv2.rectangle(top,(0,0),(LS-1,WS-1),(0,215,255),2);cv2.line(top,(LS//2,0),(LS//2,WS),(0,215,255),1)
    cv2.circle(top,(LS//2,WS//2),int(3*S),(0,215,255),1)
    th=np.full((WS+30,LS,3),40,np.uint8); th[30:,:]=top
    tag=f"{rec['n']}. {name[:14]} k({k1:+.2f},{k2:+.2f}) res{rec['fit']:.2f}m"
    cv2.putText(th,tag,(4,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,230,255),2)
    if not rec["has_center"]: cv2.putText(th,"orta-cizgi yok",(4,WS+24),cv2.FONT_HERSHEY_SIMPLEX,0.5,(120,180,255),1)
    return th

recs=[solve_venue(k) for k in ORDER if k in J and all(J[k]["lines"].get(x) for x in ("goalN","goalF","touchN","touchF"))]
# mevcut aspect/Wp/bad bilgisini KORU, k/H/fit/box_depth güncelle
try: prev=json.load(open(os.path.join(HERE,"calib_solved.json")))
except Exception: prev={}
out=prev.copy()   # MERGE-YAZIM: bu koşuda olmayan venue kayıtları KORUNUR (26-GT silinme riskine karşı)
for r in recs:
    d=prev.get(r["name"],{}); d.update({kk:r[kk] for kk in ("n","k1","k2","H","fit","has_center","w","h","boxN_X","boxF_X")})
    out[r["name"]]=d
json.dump(out,open(os.path.join(HERE,"calib_solved.json"),"w"),indent=1,ensure_ascii=False)
print("=== birleşik kalibrasyon (residual METRE) ===")
for r in sorted(recs,key=lambda r:r["n"]):
    print(f"{r['n']:2d} {r['name'][:18]:18s} base={r['base']:5.2f}m -> fit={r['fit']:5.3f}m  k=({r['k1']:+.2f},{r['k2']:+.2f}) {'' if r['has_center'] else '[orta yok]'}")
ths=sorted([(r["n"],render(r)) for r in recs]); cols=3; rows=(len(ths)+cols-1)//cols
sheet=np.full((rows*(WS+38),cols*(LS+8),3),20,np.uint8)
for i,(n,t) in enumerate(ths):
    r,c=divmod(i,cols); sheet[r*(WS+38):r*(WS+38)+WS+30,c*(LS+8):c*(LS+8)+LS]=t
cv2.imwrite(os.path.join(HERE,"cand","_2d_joint.jpg"),sheet); print("montaj -> cand/_2d_joint.jpg")
