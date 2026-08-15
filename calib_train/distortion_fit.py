#!/usr/bin/env python3
"""PLUMB-LINE fisheye kalibrasyonu: kullanıcının işaretlediği DÜZ çizgiler (kale/taç/
orta/ceza-ön) undistort sonrası düz olmalı. Tek ORTAK (k1,k2[,merkez]) çözülür (kameralar
aynı lens). Sonra her saha: undistort -> 4 köşe -> homografi -> TEMİZ 2D top-down.

Çıktı: calib_train/distortion.json + cand/_2d_undist.jpg (montaj)
"""
import json, os
import numpy as np, cv2
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
LF = os.path.join(HERE, "label_frames")
J = json.load(open(os.path.join(HERE, "calib_lines.json")))
ORDER = json.load(open(os.path.join(HERE, "frames.json")))
L, W, S = 34.0, 18.0, 26
LS, WS = int(L*S), int(W*S)
NO_CENTER = {3, 7, 9}          # orta-çizgi tahmini -> plumb-line'a SOKMA
STRAIGHT = ["goalN", "goalF", "touchN", "touchF", "center", "boxN"]

def idx_of(k): return ORDER.index(k)+1 if k in ORDER else -1

# --- gerçek etiketli sahalar + boyut ---
venues = []
for k, v in J.items():
    ln = v.get("lines", {})
    if all(ln.get(x) and len(ln[x]) >= 2 for x in ("goalN", "touchN", "touchF")):
        img = cv2.imread(os.path.join(LF, k))
        if img is None: continue
        venues.append((k, v, img.shape[1], img.shape[0]))
print(f"saha: {len(venues)}")

# --- plumb-line grupları (>=3 nokta, düz çizgi, tahmini-değil) ---
groups = []   # (cx, cy, s, pts[N,2])
for k, v, w, h in venues:
    n = idx_of(k); cx, cy, s = w/2, h/2, w/2
    for lid in STRAIGHT:
        if lid == "center" and n in NO_CENTER: continue
        pts = v["lines"].get(lid)
        if pts and len(pts) >= 3:
            groups.append((cx, cy, s, np.asarray(pts, float)))
print(f"plumb-line grubu: {len(groups)} (toplam nokta {sum(len(g[3]) for g in groups)})")

def undist_norm(pts, k1, k2, cx, cy, s):
    u = (pts[:,0]-cx)/s; vv = (pts[:,1]-cy)/s; r2 = u*u+vv*vv
    f = 1 + k1*r2 + k2*r2*r2
    return np.stack([u*f, vv*f], 1)

def line_resid(P):
    c = P.mean(0); _,_,vt = np.linalg.svd(P-c); nrm = np.array([-vt[0,1], vt[0,0]])
    return (P-c) @ nrm     # işaretli dik mesafe

def residuals(p):
    k1, k2 = p[0], p[1]; out = []
    for cx, cy, s, pts in groups:
        out.append(line_resid(undist_norm(pts, k1, k2, cx, cy, s)))
    return np.concatenate(out)

# k=0 (düz homografi) baz residual
base = residuals([0,0]); print(f"baz (bozulmasız) RMS residual = {np.sqrt((base**2).mean())*1000:.2f} (×1e-3 norm)")
sol = least_squares(residuals, [0.0,0.0], method="lm", max_nfev=4000)
k1, k2 = sol.x
fit = residuals(sol.x); print(f"FIT k1={k1:.4f} k2={k2:.4f}  RMS residual = {np.sqrt((fit**2).mean())*1000:.2f} (×1e-3)")
print(f"  iyileşme: {(1-np.sqrt((fit**2).mean())/np.sqrt((base**2).mean()))*100:.0f}%")
json.dump({"model":"brown_radial_norm","k1":k1,"k2":k2,"note":"u=(x-w/2)/(w/2); undist=p*(1+k1r2+k2r4)"},
          open(os.path.join(HERE,"distortion.json"),"w"), indent=1)

# --- ters radyal LUT (undistorted r -> distorted r) ---
rd = np.linspace(0, 1.8, 1200); ru = rd*(1+k1*rd*rd+k2*rd**4)
def undistort_image(img, cx, cy, s, zoom=1.5):
    h, wd = img.shape[:2]; ys, xs = np.mgrid[0:h, 0:wd].astype(np.float32)
    uo = (xs-cx)/s*zoom; vo = (ys-cy)/s*zoom; ro = np.sqrt(uo*uo+vo*vo)
    rdv = np.interp(ro, ru, rd); sc = np.divide(rdv, ro, out=np.ones_like(ro), where=ro>1e-6)
    mapx = (cx + uo*sc*s).astype(np.float32); mapy = (cy + vo*sc*s).astype(np.float32)
    return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR), zoom
def undist_to_pix(pts, k1,k2,cx,cy,s, zoom=1.5):
    un = undist_norm(pts,k1,k2,cx,cy,s)   # normalized undistorted
    return np.stack([cx+un[:,0]/zoom*s, cy+un[:,1]/zoom*s],1)
def fitL(P): c=P.mean(0); _,_,vt=np.linalg.svd(P-c); nv=np.array([-vt[0,1],vt[0,0]]); return np.array([nv[0],nv[1],-nv.dot(c)])
def itr(a,b):
    D=a[0]*b[1]-b[0]*a[1]
    return None if abs(D)<1e-9 else np.array([(a[1]*b[2]-b[1]*a[2])/D*-1+0,0])  # placeholder
def inter(a,b):
    a1,b1,c1=a; a2,b2,c2=b; D=a1*b2-a2*b1
    return None if abs(D)<1e-9 else np.array([(b1*c2-b2*c1)/D,(a2*c1-a1*c2)/D])

thumbs=[]
for k, v, w, h in venues:
    n=idx_of(k); cx,cy,s=w/2,h/2,w/2; img=cv2.imread(os.path.join(LF,k)); ln=v["lines"]
    und,zoom=undistort_image(img,cx,cy,s)
    def cor(a,b): return inter(fitL(undist_to_pix(np.asarray(ln[a],float),k1,k2,cx,cy,s,zoom)),
                               fitL(undist_to_pix(np.asarray(ln[b],float),k1,k2,cx,cy,s,zoom)))
    th=np.full((WS+30,LS,3),40,np.uint8)
    try:
        c_nN=cor("goalN","touchN"); c_nF=cor("goalN","touchF")
        c_fN=cor("goalF","touchN"); c_fF=cor("goalF","touchF")
        src=np.array([c_nN,c_nF,c_fN,c_fF],float)
        dst=np.array([[0,WS],[0,0],[LS,WS],[LS,0]],float)
        Hm,_=cv2.findHomography(src,dst); top=cv2.warpPerspective(und,Hm,(LS,WS))
        cv2.rectangle(top,(0,0),(LS-1,WS-1),(0,215,255),2)
        cv2.line(top,(LS//2,0),(LS//2,WS),(0,215,255),1)
        cv2.circle(top,(LS//2,WS//2),int(3*S),(0,215,255),1); th[30:,:]=top
    except Exception as e:
        cv2.putText(th,f"HATA",(20,WS//2),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
    cv2.putText(th,f"{n}. {k[:16]}",(4,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,230,255),2)
    thumbs.append((n,th))
thumbs.sort(key=lambda t:t[0]); cols=3; rows=(len(thumbs)+cols-1)//cols
sheet=np.full((rows*(WS+38),cols*(LS+8),3),20,np.uint8)
for i,(n,th) in enumerate(thumbs):
    r,c=divmod(i,cols); sheet[r*(WS+38):r*(WS+38)+WS+30, c*(LS+8):c*(LS+8)+LS]=th
out=os.path.join(HERE,"cand","_2d_undist.jpg"); cv2.imwrite(out,sheet); print("montaj ->",out)
