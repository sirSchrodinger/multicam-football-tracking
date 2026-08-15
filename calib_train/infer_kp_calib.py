#!/usr/bin/env python3
"""TAM OTONOM çıkarım: ham kare -> KeypointUNet (NK ısı-haritası + σ) -> DSNT alt-piksel
keypoint -> torch LM (keypoint-eşleme, σ-ağırlıklı) -> k1,k2,H -> üstten-görünüm 2D.
Metrik hedefler KANONİK (34x18); homografi gerçek ölçeği soğurur. σ düşük keypoint
LM'de down-weight edilir (off-frame/güvensiz). Çıktı: cand/_kp_calib.jpg (saha montajı).
Doğrulama: residual(m) + L_geo(template/stripe) seg2 rol-tabanlıya karşı.
"""
import sys, os, json, numpy as np, cv2, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_default_dtype(torch.float32)
from calib_train.model_kp import KeypointUNet, dsnt, dsnt_win, NK, KP_NAMES, kp_metric
from calib_train.make_seg_data2 import ORDER, J
from calib_train import torch_calib as tc
HERE=os.path.dirname(os.path.abspath(__file__)); LF=os.path.join(HERE,"label_frames")
C=json.load(open(os.path.join(HERE,"calib_solved.json")))
TW,TH=512,288; L,W=34.0,18.0; S=26; LS,WS=int(L*S),int(W*S)
DEV="cuda" if torch.cuda.is_available() else "cpu"
KPM=kp_metric(L,W)   # (7,2) kanonik: köşeler+orta uçları+yuvarlak merkez
# corner keypoint sırası model_kp: c_nN(0,0) c_nF(0,W) c_fN(L,0) c_fF(L,W) center_n center_f circle_c
CORNERS=[0,1,2,3]

def load_net(ckpt):
    torch.set_default_dtype(torch.float32)   # torch_calib import'u float64 yapmıştı; net float32
    net=KeypointUNet().to(DEV); net.load_state_dict(torch.load(ckpt,map_location=DEV)); net.float().eval(); return net

def predict_kp(net,img):
    h,w=img.shape[:2]; x=cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.
    with torch.no_grad():
        hm,sig=net(torch.from_numpy(x)[None].to(DEV)); kp,peak=dsnt_win(hm); _,conc=dsnt(hm)
    kp=kp[0].cpu().numpy(); sig=sig[0].cpu().numpy(); conc=conc[0].cpu().numpy(); peak=peak[0].cpu().numpy()
    pix=np.stack([kp[:,0]*w,kp[:,1]*h],1)   # orijinal piksel (pencereli decode)
    return pix,sig,conc,peak

def calibrate(pix,conf,cx,cy,s,thr=0.40):
    """σ-ağırlıklı keypoint LM. c_nN(en yakın köşe) genelde occluded -> köşe ŞARTI YOK;
    yüksek-güvenli TÜM noktalar (>=4, dejenere değil) kullanılır. döner: p(10), fit(m) ya da None."""
    use=np.array([i for i in range(NK) if conf[i]>thr])
    if len(use)<4: return None,None,use
    P=KPM[use]   # metrik konumlar dejenere mi (en az 2 farklı X ve 2 farklı Y)
    if len(np.unique(np.round(P[:,0],3)))<2 or len(np.unique(np.round(P[:,1],3)))<2: return None,None,use
    kp_img=torch.tensor(pix[use],dtype=torch.float64)
    kp_met=torch.tensor(P,dtype=torch.float64); wts=torch.tensor(conf[use],dtype=torch.float64)
    # init: TÜM görünür noktalarla k=0 homografi (>=4 -> least-squares)
    src=pix[use].astype(np.float64); un=np.stack([(src[:,0]-cx)/s,(src[:,1]-cy)/s],1)
    H0,_=cv2.findHomography(un,P.astype(np.float64))
    if H0 is None: return None,None,use
    p0=np.array([0.0,0.0]+list((H0/H0[2,2]).ravel()[:8]))
    p,fit=tc.lm(lambda pp:tc.resid_kp(pp,kp_img,kp_met,(cx,cy,s),wts),p0)
    return p.numpy(),fit,use

def render2d(img,p,cx,cy,s):
    """Tam ileri-model ters-haritası: metrik çıkış pikseli -> H^-1 -> undistort-norm ->
    radyal-ters -> distorted-norm -> kaynak piksel -> remap. (warpPerspective hilesi YOK)."""
    k1,k2=p[0],p[1]; H=np.array([[p[2],p[3],p[4]],[p[5],p[6],p[7]],[p[8],p[9],1.0]]); Hi=np.linalg.inv(H)
    Xs,Ys=np.meshgrid(np.linspace(0,L,LS),np.linspace(0,W,WS))   # metrik ızgara
    den=Hi[2,0]*Xs+Hi[2,1]*Ys+Hi[2,2]
    un=(Hi[0,0]*Xs+Hi[0,1]*Ys+Hi[0,2])/den; vn=(Hi[1,0]*Xs+Hi[1,1]*Ys+Hi[1,2])/den   # undistort-norm
    ru=np.sqrt(un*un+vn*vn)   # undistort yarıçap -> distort yarıçap (LUT)
    rd_g=np.linspace(0,2.6,5000); ru_g=rd_g*(1+k1*rd_g*rd_g+k2*rd_g**4)
    rd=np.interp(ru,ru_g,rd_g); sc=np.divide(rd,ru,out=np.ones_like(ru),where=ru>1e-9)
    mapx=(cx+un*sc*s).astype(np.float32); mapy=(cy+vn*sc*s).astype(np.float32)
    top=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR,borderValue=(0,0,0))
    cv2.rectangle(top,(0,0),(LS-1,WS-1),(0,215,255),2);cv2.line(top,(LS//2,0),(LS//2,WS),(0,215,255),1)
    cv2.circle(top,(LS//2,WS//2),int(3*S),(0,215,255),1)
    return top

def run(ckpt):
    net=load_net(ckpt); names=[n for n in ORDER if n in C and not C[n].get("bad")]
    tiles=[]; rows=[]
    for n in names:
        img=cv2.imread(os.path.join(LF,n));
        if img is None: continue
        h,w=img.shape[:2]; cx,cy,s=w/2,h/2,w/2
        pix,sig,conc,peak=predict_kp(net,img); conf=np.sqrt(np.clip(sig*peak,0,1))
        p,fit,use=calibrate(pix,conf,cx,cy,s)
        nkp=len(use)
        if p is None:
            top=np.full((WS,LS,3),30,np.uint8); cv2.putText(top,f"KALIB YOK ({nkp} nokta)",(8,WS//2),cv2.FONT_HERSHEY_SIMPLEX,0.6,(80,120,255),2); fitv=None
        else:
            top=render2d(img,p,cx,cy,s); fitv=fit
        rows.append((n,fitv,nkp))
        th=np.full((WS+26,LS,3),40,np.uint8); th[26:]=top
        tag=f"{n[:16]} res={'%.2f'%fit if p is not None else 'X'}m kp={nkp}"
        cv2.putText(th,tag,(4,18),cv2.FONT_HERSHEY_SIMPLEX,0.45,(0,230,255),1); tiles.append(th)
    cols=3; rr=(len(tiles)+cols-1)//cols; sheet=np.full((rr*(WS+30),cols*(LS+8),3),20,np.uint8)
    for i,t in enumerate(tiles):
        r,c=divmod(i,cols); sheet[r*(WS+30):r*(WS+30)+WS+26,c*(LS+8):c*(LS+8)+LS]=t
    cv2.imwrite(os.path.join(HERE,"cand","_kp_calib.jpg"),sheet)
    ok=[r for r in rows if r[1] is not None]
    print(f"kalibre {len(ok)}/{len(rows)} saha | residual med={np.median([r[1] for r in ok]):.2f}m "
          f"<0.6m: {sum(1 for r in ok if r[1]<0.6)}/{len(ok)}")
    for n,f,nk in sorted(rows,key=lambda r:(r[1] is None,r[1] or 9)):
        print(f"  {n[:22]:22s} res={'%.3f'%f if f is not None else '   X  '}m  nokta={nk}")
    print("-> cand/_kp_calib.jpg")

if __name__=="__main__":
    ck=sys.argv[1] if len(sys.argv)>1 else os.path.join(HERE,"kp_ckpt","kp_unet.pth")
    if not os.path.exists(ck): ck=os.path.join(HERE,"kp_ckpt","kp_unet_last.pth")
    print("ckpt",ck); run(ck)
