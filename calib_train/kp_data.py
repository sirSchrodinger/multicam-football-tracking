#!/usr/bin/env python3
"""Keypoint eğitim verisi: RGB + NK ısı-haritası + görünürlük(vis). Rol = hangi keypoint.
GERÇEK: kullanıcı etiketlerinden köşe/orta/yuvarlak (raw görüntüde, görünürse). SENTETİK:
3D köşe-kamera ile rol-keypoint projeksiyonu (make_seg_data2.synth_sample meta). Off-frame
keypoint vis=0 -> σ/LM down-weight eder.
"""
import json, os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calib_train.make_seg_data2 import synth_sample, ORDER, J, IW as SW, IH as SH
from calib_train.model_kp import NK, KP_NAMES
HERE=os.path.dirname(os.path.abspath(__file__)); LF=os.path.join(HERE,"label_frames")
C=json.load(open(os.path.join(HERE,"calib_solved.json")))
TW,TH=512,288; SIGMA=3.0
def fitL(P):P=np.asarray(P,float);c=P.mean(0);_,_,vt=np.linalg.svd(P-c);nv=np.array([-vt[0,1],vt[0,0]]);return np.array([nv[0],nv[1],-nv.dot(c)])
def inter(a,b):
    D=a[0]*b[1]-b[0]*a[1]
    return None if abs(D)<1e-9 else np.array([(a[1]*b[2]-b[1]*a[2])/D,(b[0]*a[2]-a[0]*b[2])/D])
def heatmaps(kps,vis,w,h):
    """kps: list of (x,y) görüntü-piksel (w,h ölçeğinde); vis: NK bool -> (NK,TH,TW)+vis."""
    hm=np.zeros((NK,TH,TW),np.float32); yy,xx=np.mgrid[0:TH,0:TW]
    sx,sy=TW/w,TH/h
    for i in range(NK):
        if vis[i] and kps[i] is not None:
            x,y=kps[i][0]*sx,kps[i][1]*sy
            if 0<=x<TW and 0<=y<TH: hm[i]=np.exp(-((xx-x)**2+(yy-y)**2)/(2*SIGMA**2))
            else: vis[i]=0
        else: vis[i]=0
    return hm,np.array(vis,np.float32)

def real_kp(name):
    img=cv2.imread(os.path.join(LF,name))
    if img is None: return None
    h,w=img.shape[:2]; ln=J[name]["lines"]
    def L_(k): return fitL(ln[k]) if ln.get(k) and len(ln[k])>=2 else None
    gN,gF,tN,tF=L_("goalN"),L_("goalF"),L_("touchN"),L_("touchF"); cen=L_("center")
    cen_real=ln.get("center") and len(ln["center"])>=4
    kps=[None]*NK; vis=[0]*NK
    if gN is not None and tN is not None: kps[0]=inter(gN,tN); vis[0]=1
    if gN is not None and tF is not None: kps[1]=inter(gN,tF); vis[1]=1
    if gF is not None and tN is not None: kps[2]=inter(gF,tN); vis[2]=1
    if gF is not None and tF is not None: kps[3]=inter(gF,tF); vis[3]=1
    if cen is not None and cen_real:
        if tN is not None: kps[4]=inter(cen,tN); vis[4]=1
        if tF is not None: kps[5]=inter(cen,tF); vis[5]=1
    if ln.get("circle") and len(ln["circle"])>=6: kps[6]=np.array(ln["circle"],float).mean(0); vis[6]=1
    hm,v=heatmaps(kps,vis,w,h); return cv2.resize(img,(TW,TH)),hm,v

def synth_kp(seed):
    img,_,meta=synth_sample(seed,return_meta=True); proj=meta["proj"]; L,W,ntY,ftY=meta["L"],meta["W"],meta["ntY"],meta["ftY"]
    M=np.array([[0,ntY],[0,ftY],[L,ntY],[L,ftY],[L/2,ntY],[L/2,ftY],[L/2,W/2]],float)
    pp,z=proj(M); kps=[None]*NK; vis=[0]*NK
    for i in range(NK):
        if z[i]>0 and 0<=pp[i,0]<SW and 0<=pp[i,1]<SH: kps[i]=pp[i]; vis[i]=1
    hm,v=heatmaps(kps,vis,SW,SH); return cv2.resize(img,(TW,TH)),hm,v

if __name__=="__main__":
    reals=[n for n in ORDER if n in C and not C[n].get("bad")]
    print(f"gerçek saha: {len(reals)}")
    r=real_kp(reals[0]); print("real:",r[0].shape,r[1].shape,"vis",r[2].astype(int),sum(r[2]),"görünür")
    s=synth_kp(0); print("synth:",s[0].shape,s[1].shape,"vis",s[2].astype(int),sum(s[2]),"görünür")
    # önizleme: gerçek + sentetik keypoint overlay
    tiles=[]
    COL=[(0,0,255),(0,150,255),(0,255,0),(255,120,0),(0,255,255),(255,0,200),(255,150,255)]
    for getter,ids in [(real_kp,reals[:4]),(synth_kp,[0,1,2,3])]:
        for x in ids:
            img,hm,v=getter(x); o=img.copy()
            for i in range(NK):
                if v[i]:
                    yx=np.unravel_index(hm[i].argmax(),hm[i].shape); cv2.circle(o,(yx[1],yx[0]),6,COL[i],-1)
            tiles.append(o)
    sheet=np.full(((TH+4)*2,(TW+4)*4,3),20,np.uint8)
    for j,t in enumerate(tiles[:8]):
        rr,cc=divmod(j,4); sheet[rr*(TH+4):rr*(TH+4)+TH,cc*(TW+4):cc*(TW+4)+TW]=t
    cv2.imwrite(os.path.join(HERE,"cand","_kpdata.jpg"),sheet); print("-> cand/_kpdata.jpg")
