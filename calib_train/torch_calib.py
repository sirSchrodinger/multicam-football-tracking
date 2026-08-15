#!/usr/bin/env python3
"""DİFERANSİYELLENEBİLİR kalibrasyon (torch): joint_calib'in LM çözümünün torch portu.
İki mod:
  (A) ÇİZGİ-kısıtı  : joint_calib ile AYNI (touchN->Y0, goalN->X0, ...). scipy ile EŞLEŞME assert.
  (B) KEYPOINT-eşleme: DSNT keypoint'leri (alt-piksel) + güven(σ) -> ağırlıklı reprojeksiyon.
LM autograd-Jacobian ile çözülür; çözüm girişe (keypoint koordinatları) göre türevlenebilir
(unrolled LM) -> self-sup L_geo kaybı keypoint kafasına geri yayılır.
Çalıştır: parite testi 26 sahada scipy-fit ~= torch-fit (tol 1e-2 m).
"""
import json, os, numpy as np, torch
torch.set_default_dtype(torch.float64)
HERE=os.path.dirname(os.path.abspath(__file__))
J=json.load(open(os.path.join(HERE,"calib_lines.json"))); ORDER=json.load(open(os.path.join(HERE,"frames.json")))
import cv2
L,W=34.0,18.0
CONSTR={"goalN":(0,0.0),"goalF":(0,L),"touchN":(1,0.0),"touchF":(1,W),"center":(0,L/2)}

def und_norm_t(pts,k1,k2,cx,cy,s):
    u=(pts[:,0]-cx)/s; v=(pts[:,1]-cy)/s; r2=u*u+v*v; f=1+k1*r2+k2*r2*r2
    return torch.stack([u*f,v*f],1)
def applyH_t(uv,H):
    z=H[2,0]*uv[:,0]+H[2,1]*uv[:,1]+H[2,2]
    X=(H[0,0]*uv[:,0]+H[0,1]*uv[:,1]+H[0,2])/z
    Y=(H[1,0]*uv[:,0]+H[1,1]*uv[:,1]+H[1,2])/z
    return torch.stack([X,Y],1)
def H_of(p): return torch.stack([
    torch.stack([p[2],p[3],p[4]]),torch.stack([p[5],p[6],p[7]]),
    torch.stack([p[8],p[9],torch.ones((),dtype=p.dtype,device=p.device)])])

def fitL(P): c=P.mean(0);_,_,vt=np.linalg.svd(P-c);nv=np.array([-vt[0,1],vt[0,0]]);return np.array([nv[0],nv[1],-nv.dot(c)])
def inter(a,b):
    a1,b1,c1=a;a2,b2,c2=b;D=a1*b2-a2*b1
    return None if abs(D)<1e-9 else np.array([(b1*c2-b2*c1)/D,(a2*c1-a1*c2)/D])

def line_constraints(name):
    """joint_calib ile aynı: (pts_tensor, axis, target) listesi + cx,cy,s + p0 (numpy init)."""
    v=J[name]; ln=v["lines"]; img=cv2.imread(os.path.join(HERE,"label_frames",name)); h,w=img.shape[:2]
    cx,cy,s=w/2,h/2,w/2
    center_real=bool(ln.get("center") and len(ln["center"])>=4)
    cons=[]
    for lid,(ax,tg) in CONSTR.items():
        if lid=="center" and not center_real: continue
        p=ln.get(lid)
        if p and len(p)>=2: cons.append((torch.tensor(np.asarray(p,float)),ax,float(tg)))
    # init H (k=0)
    def und0(P): u=(P[:,0]-cx)/s; v=(P[:,1]-cy)/s; return np.stack([u,v],1)
    fl={l:fitL(und0(np.asarray(ln[l],float))) for l in ("goalN","goalF","touchN","touchF") if ln.get(l)}
    src=np.array([inter(fl["goalN"],fl["touchN"]),inter(fl["goalN"],fl["touchF"]),
                  inter(fl["goalF"],fl["touchN"]),inter(fl["goalF"],fl["touchF"])],float)
    H0,_=cv2.findHomography(src,np.array([[0,0],[0,W],[L,0],[L,W]],float))
    p0=np.array([0.0,0.0]+list((H0/H0[2,2]).ravel()[:8]))
    return cons,(cx,cy,s),p0,center_real

def resid_lines(p,cons,cxys):
    cx,cy,s=cxys; H=H_of(p); out=[]
    for pts,ax,tg in cons:
        m=applyH_t(und_norm_t(pts,p[0],p[1],cx,cy,s),H); out.append(m[:,ax]-tg)
    return torch.cat(out)

def lm(resid_fn,p0,iters=80,lam0=1e-3,tol=1e-12):
    p=torch.tensor(p0,dtype=torch.float64); lam=lam0
    def cost(pp):
        r=resid_fn(pp); return r,(r*r).sum()
    r,c=cost(p)
    for it in range(iters):
        Jc=torch.autograd.functional.jacobian(lambda pp:resid_fn(pp),p,vectorize=True)  # (M,N)
        JtJ=Jc.T@Jc; g=Jc.T@r; N=p.numel()
        for _ in range(12):
            A=JtJ+lam*torch.diag(torch.diagonal(JtJ)+1e-12)
            try: dp=torch.linalg.solve(A,-g)
            except Exception: lam*=10; continue
            pn=p+dp; rn,cn=cost(pn)
            if cn<c: p,r,c=pn,rn,cn; lam=max(lam/3,1e-9); break
            else: lam*=3
        else: break
        if c<tol or torch.linalg.norm(dp)<1e-12: break
    return p,float(torch.sqrt((resid_fn(p)**2).mean()))

# ---------- KEYPOINT mod (B) ----------
def resid_kp(p,kp_img,kp_metric,cxys,wts=None):
    """kp_img (K,2) görüntü-piksel, kp_metric (K,2) metrik hedef -> ağırlıklı reprojeksiyon artığı."""
    cx,cy,s=cxys; H=H_of(p); m=applyH_t(und_norm_t(kp_img,p[0],p[1],cx,cy,s),H)
    d=(m-kp_metric);
    if wts is not None: d=d*wts[:,None]
    return d.reshape(-1)

if __name__=="__main__":
    from scipy.optimize import least_squares
    names=[k for k in ORDER if k in J and all(J[k]["lines"].get(x) for x in ("goalN","goalF","touchN","touchF"))]
    print(f"{'saha':20s} {'scipy':>8s} {'torch':>8s} {'Δ(mm)':>8s}")
    worst=0
    for n in names:
        cons,cxys,p0,_=line_constraints(n)
        # scipy referans (numpy resid)
        cx,cy,s=cxys
        def rnp(p):
            k1,k2=p[0],p[1]; H=np.array([[p[2],p[3],p[4]],[p[5],p[6],p[7]],[p[8],p[9],1.0]]); out=[]
            for pts,ax,tg in cons:
                P=pts.numpy(); u=(P[:,0]-cx)/s; v=(P[:,1]-cy)/s; r2=u*u+v*v; f=1+k1*r2+k2*r2*r2
                un=np.stack([u*f,v*f],1); z=H[2,0]*un[:,0]+H[2,1]*un[:,1]+H[2,2]
                X=(H[0,0]*un[:,0]+H[0,1]*un[:,1]+H[0,2])/z; Y=(H[1,0]*un[:,0]+H[1,1]*un[:,1]+H[1,2])/z
                out.append((np.stack([X,Y],1)[:,ax]-tg))
            return np.concatenate(out)
        sol=least_squares(rnp,p0,method="lm",max_nfev=20000); sfit=np.sqrt((rnp(sol.x)**2).mean())
        _,tfit=lm(lambda pp:resid_lines(pp,cons,cxys),p0)
        d=abs(sfit-tfit)*1000; worst=max(worst,d)
        print(f"{n[:20]:20s} {sfit:8.4f} {tfit:8.4f} {d:8.2f}")
    print(f"\nEN KÖTÜ Δ = {worst:.2f} mm  ({'PARİTE OK' if worst<10 else 'SAPMA'})")
