"""İndirilen clicks JSON (base + box noktaları) -> joint [k1,k2,H,bd,bw] -> calib + warp.
Box boyu (bd,bw) SERBEST aranır (amatör saha değişken). Kullanım: python calib/solve_venue_clicks.py <clicks.json> <frame.jpg>"""
import os,sys,json,numpy as np,cv2
sys.path.insert(0,'.'); os.environ['CUDA_VISIBLE_DEVICES']=''
from scipy.optimize import least_squares
from calib.landmarks import base_landmarks, BOX_IDS
from calib_train import auto_calib as AC
L,W=34.0,18.0
BASEW={lid:np.array(xy,float) for (lid,_,xy,_,_) in base_landmarks(L,W)}
def box_world(lid,bd,bw):
    cy=W/2; hw=bw/2
    m={'box_near_front_y0':(bd,cy-hw),'box_near_front_yW':(bd,cy+hw),'box_near_goal_y0':(0,cy-hw),'box_near_goal_yW':(0,cy+hw),
       'box_far_front_y0':(L-bd,cy-hw),'box_far_front_yW':(L-bd,cy+hw),'box_far_goal_y0':(L,cy-hw),'box_far_goal_yW':(L,cy+hw)}
    return np.array(m[lid],float)
def main(clicks_path, frame_path):
    pts=json.loads(open(os.path.expanduser(clicks_path)).read()).get('points',{})
    img=cv2.imread(frame_path); h,w=img.shape[:2]; cx,cy,s=w/2,h/2,w/2
    fip,fwp,bip,bk=[],[],[],[]
    for lid,xy in pts.items():
        if lid in BASEW: fip.append(xy); fwp.append(BASEW[lid])
        elif lid in BOX_IDS: bip.append(xy); bk.append(lid)
    fip=np.array(fip,float) if fip else np.empty((0,2)); fwp=np.array(fwp,float) if fwp else np.empty((0,2))
    bip=np.array(bip,float) if bip else np.empty((0,2))
    N=len(fip)+len(bip)
    print(f"kullanılan: {len(fip)} sabit + {len(bip)} box = {N} nokta")
    if N<5: print("YETERSİZ (<5)"); return
    # H0: prior-undistort + sabit noktalardan (yoksa box-goal)
    ip0=np.vstack([fip,bip]) if len(bip) else fip
    wp0=np.vstack([fwp,[box_world(k,5,9) for k in bk]]) if len(bip) else fwp
    un=AC.und_norm(ip0,0.167,0.240,cx,cy,s); H0,_=cv2.findHomography(wp0,un,0)
    if H0 is None: H0=np.eye(3)
    x0=[0.167,0.240,*(H0.ravel()[:8]/H0[2,2]).tolist(),5.0,9.0]
    def resid(pp):
        k1,k2=pp[0],pp[1];H=np.array([[pp[2],pp[3],pp[4]],[pp[5],pp[6],pp[7]],[pp[8],pp[9],1.0]]);bd,bw=pp[10],pp[11]
        wp=fwp.copy() if len(fwp) else np.empty((0,2))
        if len(bk): bw_=np.array([box_world(k,bd,bw) for k in bk]); wp=np.vstack([wp,bw_]) if len(wp) else bw_
        ip=np.vstack([fip,bip]) if len(bip) and len(fip) else (bip if len(bip) else fip)
        proj=AC.project_metric(wp,k1,k2,H,cx,cy,s)
        return (proj-ip).ravel()
    lo=[0.02,0.02]+[-np.inf]*8+[3.0,6.0]; hi=[0.45,0.7]+[np.inf]*8+[8.0,14.0]
    sol=least_squares(resid,x0,method="trf",x_scale="jac",bounds=(lo,hi),max_nfev=12000)
    k1,k2=sol.x[0],sol.x[1];H=np.array([[sol.x[2],sol.x[3],sol.x[4]],[sol.x[5],sol.x[6],sol.x[7]],[sol.x[8],sol.x[9],1.0]]);bd,bw=sol.x[10],sol.x[11]
    err=np.abs(resid(sol.x)).reshape(-1,2); perr=np.linalg.norm(err,axis=1)
    print(f"calib: k1={k1:.3f} k2={k2:.3f} box(bd={bd:.1f},bw={bw:.1f}) | reprojeksiyon med={np.median(perr):.1f}px max={perr.max():.1f}px")
    from calib_train.night.four_panel import p_warp
    rec={'k1':k1,'k2':k2,'H':H,'camside':'SOL','ok':True,'marks':None}
    warp=p_warp(cv2.resize(img,(w,h)),rec,w,h)
    cv2.imwrite('calib/venue34_warp_manual.jpg',warp,[cv2.IMWRITE_JPEG_QUALITY,92])
    print("-> calib/venue34_warp_manual.jpg")
if __name__=="__main__": main(sys.argv[1],sys.argv[2])
