#!/usr/bin/env python3
"""calib_lines.json (kullanıcı etiketleri) -> her saha için gerçek 2D top-down render +
çizgi-fit kalite + eksik/tahmini bayrakları. Montaj çıkarır, gözle bakılır.
"""
import json, os, sys
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
LF = os.path.join(HERE, "label_frames")
J = json.load(open(os.path.join(HERE, "calib_lines.json")))
L, W, S = 34.0, 18.0, 26
LS, WS = int(L*S), int(W*S)
# kullanıcı feedback: orta-yuvarlak / orta-çizgi tahmini olan sahalar (1-based onun sırası)
NO_CIRCLE = {1,2,3,4,7,9,10,11,12,13}
NO_CENTER = {3,7,9}

def fit_line(pts):
    P = np.asarray(pts, float); c = P.mean(0)
    u, s, vt = np.linalg.svd(P - c)
    d = vt[0]; n = np.array([-d[1], d[0]])      # normal
    return np.array([n[0], n[1], -n.dot(c)]), float(s[1]/max(1,len(P))**0.5)  # (a,b,c), residual
def inter(l1, l2):
    a1,b1,c1=l1; a2,b2,c2=l2; D=a1*b2-a2*b1
    if abs(D)<1e-9: return None
    return np.array([(b1*c2-b2*c1)/D, (a2*c1-a1*c2)/D])

# sadece gerçek etiketli sahalar (4 kenar var)
venues=[]
for k,v in J.items():
    ln=v.get("lines",{})
    if all(ln.get(x) and len(ln[x])>=2 for x in ("goalN","touchN","touchF")):
        venues.append((k,v))
print(f"4-kenar etiketli saha: {len(venues)}")

# onun 1-based numarasını çıkar (label_frames sırası)
order=[s for s in json.load(open(os.path.join(HERE,"frames.json")))]
def idx_of(k):
    return order.index(k)+1 if k in order else -1

thumbs=[]; report=[]
for k,v in venues:
    ln=v["lines"]; img=cv2.imread(os.path.join(LF,k))
    if img is None: continue
    n=idx_of(k)
    gN,rN=fit_line(ln["goalN"]); tN,_=fit_line(ln["touchN"]); tF,_=fit_line(ln["touchF"])
    c_nN=inter(gN,tN); c_nF=inter(gN,tF)
    res={"goalN":rN}
    if ln.get("goalF") and len(ln["goalF"])>=2:
        gF,rF=fit_line(ln["goalF"]); c_fN=inter(gF,tN); c_fF=inter(gF,tF); res["goalF"]=rF
    else:
        c_fN=c_fF=None
    ok = all(c is not None for c in (c_nN,c_nF,c_fN,c_fF))
    th=np.full((WS+30, LS, 3), 40, np.uint8)
    flags=[]
    if n in NO_CIRCLE: flags.append("yuvarlak=tahmini")
    if n in NO_CENTER: flags.append("ortacizgi=tahmini")
    cam=v.get("cam");
    if cam and abs(cam[0]-L/2)<L*0.18: flags.append("MERKEZ-KAMERA")
    if ok:
        src=np.array([c_nN,c_nF,c_fN,c_fF],float)
        dst=np.array([[0,WS],[0,0],[LS,WS],[LS,0]],float)  # nN,nF,fN,fF
        Hm,_=cv2.findHomography(src,dst)
        top=cv2.warpPerspective(img, Hm, (LS, WS))
        # saha çerçevesi + orta çizgi + yuvarlak
        cv2.rectangle(top,(0,0),(LS-1,WS-1),(0,215,255),2)
        cv2.line(top,(LS//2,0),(LS//2,WS),(0,215,255),1)
        cv2.circle(top,(LS//2,WS//2),int(3*S),(0,215,255),1)
        th[30:,:]=top
        # köşe-açı sağlık: kenarlar dik mi? (warp sonrası dikdörtgen olmalı, hep öyle) -> onun yerine
        # orijinal köşelerin görüntü içi olup olmadığını yaz
    else:
        cv2.putText(th,"4 kose YOK",(20,WS//2),cv2.FONT_HERSHEY_SIMPLEX,1.0,(0,0,255),2)
    lab=f"{n}. {k[:16]} r={rN:.1f}"
    cv2.putText(th,lab,(4,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,230,255),2)
    if flags:
        cv2.putText(th," ".join(flags),(4,WS+24),cv2.FONT_HERSHEY_SIMPLEX,0.5,(120,180,255),1)
    thumbs.append((n,th)); report.append((n,k,res,flags,cam))

thumbs.sort(key=lambda t:t[0])
# montaj 4 sütun
cols=3; tw=LS; thh=WS+30
rows=(len(thumbs)+cols-1)//cols
sheet=np.full((rows*(thh+8), cols*(tw+8),3),20,np.uint8)
for i,(n,th) in enumerate(thumbs):
    r,c=divmod(i,cols); y=r*(thh+8); x=c*(tw+8); sheet[y:y+thh,x:x+tw]=th
out=os.path.join(HERE,"cand","_2d_validate.jpg")
cv2.imwrite(out,sheet); print("montaj ->",out, sheet.shape)
print("\n=== rapor (fit residual px; düşük=düz/iyi, yüksek=eğri/fisheye veya özensiz) ===")
for n,k,res,flags,cam in sorted(report):
    rr=" ".join(f"{a}={b:.1f}" for a,b in res.items())
    print(f"{n:2d} {k[:20]:20s} {rr:22s} cam[{cam[0]:.0f},{cam[1]:.0f}] {' '.join(flags)}")
