"""İnsan-tıklanan landmark NOKTALARINDAN joint fisheye+homografi (ham kare).
world (landmarks.py) <-> img pixel eşleşmesi -> least_squares [k1,k2,H8].
Temiz insan-noktası (özellikle uzak-köşe) -> skew YOK. auto_calib makinesi."""
import os,sys,numpy as np
sys.path.insert(0,'.')
from scipy.optimize import least_squares
from calib_train import auto_calib as AC

def solve_points(img_pts, world_pts, w, h, k1_0=0.167, k2_0=0.240):
    """img_pts,world_pts: Nx2. w,h: kare boyu. Döndürür rec(k1,k2,H,fit,ok)."""
    img_pts=np.asarray(img_pts,float); world_pts=np.asarray(world_pts,float)
    cx,cy,s=w/2,h/2,w/2
    # H0: undistort(prior) noktalarından hızlı homografi
    import cv2
    un=AC.und_norm(img_pts,k1_0,k2_0,cx,cy,s)
    H0,_=cv2.findHomography(world_pts, un, 0)
    if H0 is None: H0=np.eye(3)
    h0=[k1_0,k2_0,*(H0.ravel()[:8]/H0[2,2]).tolist()]
    def resid(pp):
        k1,k2=pp[0],pp[1]; H=np.array([[pp[2],pp[3],pp[4]],[pp[5],pp[6],pp[7]],[pp[8],pp[9],1.0]])
        proj=AC.project_metric(world_pts,k1,k2,H,cx,cy,s)   # world->img pixel
        return (proj-img_pts).ravel()
    lo=[0.02,0.02]+[-np.inf]*8; hi=[0.45,0.7]+[np.inf]*8
    try: sol=least_squares(resid,h0,method="trf",x_scale="jac",bounds=(lo,hi),max_nfev=9000)
    except Exception as e: return {'ok':False,'reason':str(e)}
    H=np.array([[sol.x[2],sol.x[3],sol.x[4]],[sol.x[5],sol.x[6],sol.x[7]],[sol.x[8],sol.x[9],1.0]])
    fit=float(np.sqrt((resid(sol.x)**2).mean()))
    return {'ok':True,'k1':float(sol.x[0]),'k2':float(sol.x[1]),'H':H,'fit':fit,'camside':'SOL'}

if __name__=="__main__":
    # ROUND-TRIP self-test: bilinen calib -> world köşe-projekte -> gürültü ekle -> geri-çöz -> recover?
    w,h=1920,1080; cx,cy,s=w/2,h/2,w/2
    L,W=34.0,18.0
    world=np.array([[0,0],[L,0],[L,W],[0,W],[L/2,W/2],[0,W/2-1.5],[0,W/2+1.5],[L,W/2-1.5],[L,W/2+1.5]],float)
    k1t,k2t=0.19,0.22
    Ht=np.array([[8.0,-2.0,700.0],[1.0,6.0,300.0],[0.001,0.004,1.0]])
    img=AC.project_metric(world,k1t,k2t,Ht,cx,cy,s)
    rng=np.random.RandomState(0); img_noisy=img+rng.normal(0,2.0,img.shape)   # 2px tıklama gürültüsü
    rec=solve_points(img_noisy,world,w,h)
    print("self-test ok:",rec.get('ok'),"fit(px):",round(rec.get('fit',9e9),2))
    # recover edilen calib ile world tekrar projekte -> orijinal img'e ne kadar yakın
    if rec['ok']:
        rep=AC.project_metric(world,rec['k1'],rec['k2'],rec['H'],cx,cy,s)
        err=np.linalg.norm(rep-img,axis=1)
        print(f"recover reprojeksiyon hata: med={np.median(err):.2f}px max={err.max():.2f}px  (2px gürültüde <4px beklenir)")
        print(f"k1: gerçek {k1t} -> recover {rec['k1']:.3f} | k2: {k2t} -> {rec['k2']:.3f}")
