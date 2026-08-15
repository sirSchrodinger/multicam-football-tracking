#!/usr/bin/env python3
"""YAPISAL FIX testi: solve_calib SADECE çizgi-kısıtı kullanıyor (çizgi-boyu DOF serbest -> dejenere/bowtie).
robust = çizgi-kısıtı + KÖŞE nokta-eşleri (4 çizgi-kesişimi -> dikdörtgen köşeleri) -> H pinlenir.
Bowtie sahalar convex+makul-residual'a dönüyor mu? İyi sahalar bozuluyor mu? -> ship kararı.
"""
import os, sys, glob, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from scipy.optimize import least_squares
from calib_train import auto_calib as AC
from calib_train.auto_clean2d import sanity
from calib_train.render2d_clean import UNet
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CB=os.path.join(HERE,"cand_big")
TW,TH=1024,576; L=34.0
net=UNet(); net.load_state_dict(torch.load(os.path.join(HERE,"seg2_hr2.pth"),map_location="cpu")); net.eval()
@torch.no_grad()
def prob(img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None]
    return torch.sigmoid(net(x))[0].numpy()

def solve_robust(groups,cx,cy,s,Wp=18.0,wcorner=3.0):
    """çizgi-kısıtı + köşe nokta-eşi (corner anchors). köşe = init-k fit çizgilerinin kesişimi (distorted px)."""
    CON=[("goalN","X",0.0),("goalF","X",L),("touchN","Y",0.0),("touchF","Y",Wp)]
    K1S,K2S=AC.K1S,AC.K2S
    def cor_px(k1,k2):
        fl={r:AC.fitL(AC.und_pix(groups[r],k1,k2,cx,cy,s)) for r in groups}
        # köşe distorted-pixel: undistort-pixel kesişimini distorted'a geri çevirmek pahalı;
        # bunun yerine köşeyi UNDISTORT-NORM uzayında tut (residual aH ile aynı uzay)
        return fl
    fl0={r:AC.fitL(AC.und_pix(groups[r],K1S,K2S,cx,cy,s)) for r in groups}  # init undistort-px fit
    # köşe undistort-px (init), sonra norm: und_pix kullanıldı (z=1.5) -> norm'a çevir
    def px_to_norm(P): return np.stack([(P[:,0]-cx)/s*1.5,(P[:,1]-cy)/s*1.5],1)
    c_px=np.array([AC.inter(fl0["goalN"],fl0["touchN"]),AC.inter(fl0["goalF"],fl0["touchN"]),
                   AC.inter(fl0["goalF"],fl0["touchF"]),AC.inter(fl0["goalN"],fl0["touchF"])])
    c_norm=px_to_norm(c_px)                        # undistort-norm köşeler (init)
    tgt=np.array([[0,0],[L,0],[L,Wp],[0,Wp]],float)
    H0,_=cv2.findHomography(c_norm,tgt)
    if H0 is None: return None
    H0=H0/H0[2,2]; p0=[K1S,K2S,*H0.ravel()[:8]]
    def resid(pp):
        k1,k2=pp[0],pp[1];H=np.array([[pp[2],pp[3],pp[4]],[pp[5],pp[6],pp[7]],[pp[8],pp[9],1.0]]);out=[]
        for r,kind,tg in CON:
            m=AC.aH(AC.und_norm(groups[r],k1,k2,cx,cy,s),H); out.append((m[:,0]-tg) if kind=="X" else (m[:,1]-tg))
        # KÖŞE nokta-eşi terimi (her iter köşe init-norm sabit; H pinler)
        mc=AC.aH(c_norm,H); out.append(wcorner*(mc-tgt).ravel())
        return np.concatenate(out)
    sol=least_squares(resid,p0,method="trf",bounds=([0,0]+[-np.inf]*8,[0.45,0.7]+[np.inf]*8),max_nfev=4000)
    k1,k2=sol.x[0],sol.x[1];H=np.array([[sol.x[2],sol.x[3],sol.x[4]],[sol.x[5],sol.x[6],sol.x[7]],[sol.x[8],sol.x[9],1.0]])
    # fit = SADECE çizgi-residuali (köşe-terimsiz, karşılaştırılabilir)
    line_res=[]
    for r,kind,tg in CON:
        m=AC.aH(AC.und_norm(groups[r],k1,k2,cx,cy,s),H); line_res.append((m[:,0]-tg) if kind=="X" else (m[:,1]-tg))
    fit=float(np.sqrt((np.concatenate(line_res)**2).mean()))
    def u2pix(P): un=AC.und_norm(P,k1,k2,cx,cy,s); return np.stack([cx+un[:,0]/1.5*s,cy+un[:,1]/1.5*s],1)
    fl={r:AC.fitL(u2pix(groups[r])) for r in groups}
    c_nN=AC.inter(fl["goalN"],fl["touchN"]);c_nF=AC.inter(fl["goalN"],fl["touchF"]);c_fN=AC.inter(fl["goalF"],fl["touchN"])
    A=c_fN-c_nN;B=c_nF-c_nN;cross=A[0]*B[1]-A[1]*B[0]
    return dict(k1=k1,k2=k2,H=H,fit=fit,camside="SAG" if cross>0 else "SOL",ok=True)

TARG=["Berkay75","Gaziantep","Laliga","Playdrome","ÜsküdarAnka","Demirciler","EFT","5Mevsim","24Şubat","Dikmen"]
pool=sorted(glob.glob(os.path.join(CB,"*__*__s*.jpg")))
def find(t):
    for p in pool:
        if t.lower().replace("ı","i") in os.path.basename(p).lower().replace("ı","i"): return p
print(f"{'saha':14s} | {'ORİJİNAL':14s} | {'ROBUST(+köşe)':14s}")
for t in TARG:
    p=find(t)
    if not p: print(f"{t:14s} | (yok)"); continue
    img=cv2.imread(p); h,w=img.shape[:2]; cx,cy,s=w/2,h/2,w/2
    pr=prob(img); groups,reason=AC.groups_from_prob(pr,w,h,thr=0.5)
    if groups is None: print(f"{t[:14]:14s} | çizgi-yok"); continue
    r1=AC.solve_calib(groups,cx,cy,s)
    s1="H-tekil" if r1 is None else (lambda ok,_:f"{r1['fit']:.2f}{'✓' if ok else '✗'}")(*sanity(dict(ok=True,**r1),w,h,img=img))
    r2=solve_robust(groups,cx,cy,s)
    s2="H-tekil" if r2 is None else (lambda ok,_:f"{r2['fit']:.2f}{'✓' if ok else '✗'}")(*sanity(r2,w,h,img=img))
    print(f"{t[:14]:14s} | {s1:14s} | {s2:14s}",flush=True)
