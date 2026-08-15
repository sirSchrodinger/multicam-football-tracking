#!/usr/bin/env python3
"""OTONOM temiz 2D radar: yeni saha karesi -> 1024-model rol-çizgi -> joint kalibrasyon ->
SANITY GATE (quad konveks+alan + PCA-çöküş, araştırma TOP-5) -> oyuncu ayakları metriğe
projekte -> çizilmiş şematik saha + oyuncu noktaları. Foto-warp YOK (smear yok).
Kullanım: python auto_clean2d.py <frame.jpg|video.mp4> [tracks.parquet] [frame_idx]
"""
import os, sys, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch, torch.nn as nn
from calib_train import auto_calib as AC
from calib_train.render2d_clean import UNet
HERE=os.path.dirname(os.path.abspath(__file__)); TW,TH=1024,576; L,Wp,S,pad=34.0,18.0,26,46
net=UNet(); net.load_state_dict(torch.load(os.path.join(HERE,"seg2_hr2.pth"),map_location="cpu")); net.eval()
@torch.no_grad()
def prob(img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None]
    return torch.sigmoid(net(x))[0].numpy()

def calibrate_frame(prob_arr, w, h, img=None, feet=None):
    """ANA seg2-line kalibrasyon girişi (v2.3, 2 Tem audit-uygulaması). KEEP-BEST zinciri:
    1) BASELINE (largest-CC + yamalı-solver) HARD-check'leri geçerse AYNEN KORU
       (hard = fit/konverjans + sanity + validity_v2 + çim-çıpası + feet-QA;
       soft support-tabanı baseline'a UYGULANMAZ — churn-kanıtı: Kıbrıs kaybı).
    2) baseline hard-fail ise MULTİ-HİPOTEZ repick (top-K kombo, v2.2-seçim: çim+validity+feet+sup-taban).
    3) o da yoksa VP-recovery (eski davranış).
    Dönen rec['src'] = baseline|multihyp|vp. feet: [[x,y,conf],...] opsiyonel (feetcache)."""
    from calib_train import auto_calib as AC
    from calib_train.multihyp_calib import validity_v2, grass_inside, feet_score, solve_multihyp, select_v2, groups_topk
    from calib_train.center_refine import refine_center
    from itertools import product
    def _rc(rec):
        rec=refine_center(rec,prob_arr,w,h,img=img,sanity_fn=sanity)      # 1) center keep-best
        if rec and rec.get('ok'):
            try:
                from calib_train.center_refine import _cc, circle_support
                from calib_train.robust_boundary import solve_robust_boundary
                cen=_cc(prob_arr,4,w,h); cir=_cc(prob_arr,6,w,h)
                r2,_=solve_robust_boundary(prob_arr,w,h,rec,cen,cir,sanity_fn=sanity,img=img)  # 2) direk/outlier at
                if r2 and r2.get('ok'):                                    # keep-best: çember-desteği düşmesin
                    if cir is None or circle_support(r2,prob_arr,w,h)>=circle_support(rec,prob_arr,w,h)-0.02:
                        rec=r2
            except Exception: pass
            # 2.5) SKEW-FIX (çember-objektif) GERİ ALINDI: roundness-metriği Goodhart-game'lendi,
            #      Alperen'in gözü base'i daha iyi buldu (warp'ın geri kalanı bozuluyordu). refine_rigid çağrısı KALDIRILDI.
            try:
                from calib_train.measure_marks import measure_marks
                rec['marks']=measure_marks(rec,prob_arr,w,h)              # 3) per-venue çember/kutu ölçüsü
            except Exception: rec['marks']=None
        return rec
    base=AC.calib_from_pred(prob_arr,w,h)
    if base.get('ok') and base['fit']<=0.8 and base.get('converged',True):
        ok,_=sanity(base,w,h,img=img)
        if ok:
            vok,_=validity_v2(base,w,h)
            if vok:
                g=grass_inside(base,img,w,h) if img is not None else None
                if g is None or g>=0.45:
                    fi,spread=feet_score(base,feet,w,h)
                    if fi is None or (fi>=0.6 and spread):
                        base['src']='baseline'; return _rc(base)
    # multihyp repick (v2.2 tam-seçim)
    cands,_=groups_topk(prob_arr,w,h,K=3)
    if cands is not None:
        from calib_train.multihyp_calib import ROLES
        best=None
        for combo in product(*[range(len(cands[r])) for r in ROLES]):
            groups={r:cands[r][ci] for r,ci in zip(ROLES,combo)}
            rec=AC.solve_calib(groups,w/2,h/2,w/2)
            if rec is None or rec['fit']>0.8 or not rec.get('converged',True): continue
            ok,_=sanity(rec,w,h,img=img)
            if not ok: continue
            acc,score,_=select_v2(rec,prob_arr,w,h,img=img,feet=feet)
            if acc and (best is None or score>best[0]): best=(score,combo,rec)
        if best is not None:
            _,combo,rec=best; rec.update(ok=True,src='multihyp',mh_combo=list(combo)); return _rc(rec)
    # VP fallback (lazy import, dairesel-import kırma)
    from calib_train.vp_calib import calib_vp_strict
    r=calib_vp_strict(prob_arr,w,h,img=img)
    if r is not None and r.get('ok'): r['src']=r.get('vp_source','vp')
    return _rc(r)

def sanity(rec,w,h,img=None,min_bright=42.0):
    """diyagonal-çöküş / dejenere H reddet. img verilirse KARA-KARE de reddet (model
    karanlıkta çizgi halüsine edip false-PASS veriyordu: 58VİP/Avanos). döner (ok, reason)."""
    # AUDIT-FIX (U1/D1): fit/konverjans kontrolü — idx30 fit=3.19m + nfev-tükenmiş PASS geçiyordu.
    if rec.get("fit") is not None and rec["fit"]>0.8: return False,f"fit yüksek ({rec['fit']:.2f}m>0.8)"
    if rec.get("converged") is False: return False,"solver konverjans yok (nfev tükendi)"
    if img is not None:
        v=float(img.reshape(-1,img.shape[-1]).mean()) if img.ndim==3 else float(img.mean())
        if v<min_bright: return False,f"kare çok karanlık ({v:.0f}<{min_bright:.0f}) -> halüsinasyon riski"
    k1,k2,H=rec["k1"],rec["k2"],rec["H"]; cx,cy,s=w/2,h/2,w/2
    cm=np.array([[0,0],[L,0],[L,Wp],[0,Wp]],float); pix=AC.project_metric(cm,k1,k2,H,cx,cy,s)
    if not np.isfinite(pix).all(): return False,"H sonsuz"
    # konvekslik + yön (çapraz çarpımlar aynı işaret)
    d=np.diff(np.vstack([pix,pix[0]]),axis=0); cr=d[:,0]*np.roll(d[:,1],-1)-d[:,1]*np.roll(d[:,0],-1)
    if not (np.all(cr>0) or np.all(cr<0)): return False,"quad konveks değil (bowtie/çöküş)"
    area=0.5*abs(np.dot(pix[:,0],np.roll(pix[:,1],-1))-np.dot(pix[:,1],np.roll(pix[:,0],-1)))
    if area < 0.04*w*h: return False,f"alan çok küçük ({area/(w*h):.1%})"
    # PCA çöküş: metrik ızgara -> görüntü, σ2/σ1
    gx,gy=np.meshgrid(np.linspace(0,L,8),np.linspace(0,Wp,8)); G=np.c_[gx.ravel(),gy.ravel()]
    gp=AC.project_metric(G,k1,k2,H,cx,cy,s); gp=gp[np.isfinite(gp).all(1)]
    if len(gp)<10: return False,"ızgara projeksiyon bozuk"
    sv=np.linalg.svd(gp-gp.mean(0),compute_uv=False)
    if sv[1]/max(sv[0],1e-9) < 0.12: return False,f"PCA çöküş (σ2/σ1={sv[1]/sv[0]:.02f})"
    return True,"ok"

def project_feet(feet,rec,w,h):
    k1,k2,H=rec["k1"],rec["k2"],rec["H"]; cx,cy,s=w/2,h/2,w/2
    un=AC.und_norm(np.asarray(feet,float),k1,k2,cx,cy,s); return AC.aH(un,H)

def draw_2d(metric_pts,camside,title,marks=None):
    Wpx=int(L*S+2*pad); Hpx=int(Wp*S+2*pad)
    im=np.full((Hpx,Wpx,3),28,np.uint8)
    for i in range(int(L/3)+1):
        x0=int(pad+i*3*S); x1=min(int(pad+(i+1)*3*S),Wpx-pad); im[pad:Hpx-pad,x0:x1]=(40,92,40) if i%2 else (46,108,46)
    def P(x,y):
        if camside=="SAG": x=L-x; y=Wp-y
        return int(pad+x*S),int(pad+(Wp-y)*S)
    wht=(240,240,240); mk=marks or {}
    cv2.rectangle(im,P(0,0),P(L,Wp),wht,2); cv2.line(im,P(L/2,0),P(L/2,Wp),wht,2)
    R=mk.get('R') or 3.0; cv2.circle(im,P(L/2,Wp/2),int(R*S),wht,2)                 # per-venue çember
    bn=mk.get('box_n') or (5.0,5.0); bf=mk.get('box_f') or (5.0,5.0)               # per-venue ceza sahası
    for gx,(dep,hw) in ((0.0,bn),(L,bf)):
        bx=dep if gx==0 else L-dep
        cv2.rectangle(im,P(min(gx,bx),Wp/2-hw),P(max(gx,bx),Wp/2+hw),wht,1)
        cv2.line(im,P(gx,Wp/2-1.5),P(gx,Wp/2+1.5),(0,0,235),4)
    n=0
    for (x,y) in metric_pts:
        if not(np.isfinite(x) and np.isfinite(y)) or not(-1<x<L+1 and -1<y<Wp+1): continue
        px,py=P(np.clip(x,0,L),np.clip(y,0,Wp)); cv2.circle(im,(px,py),10,(30,30,235),-1); cv2.circle(im,(px,py),10,(255,255,255),2); n+=1
    cam=P(0,0); cv2.circle(im,cam,9,(0,200,255),-1); cv2.putText(im,"KAM",(cam[0]+8,cam[1]),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,255),1)
    cv2.putText(im,title,(8,21),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,210,255),1)
    return im,n

def grab(path,fidx):
    if path.lower().endswith((".jpg",".png")): return cv2.imread(path)
    cap=cv2.VideoCapture(path); cap.set(cv2.CAP_PROP_POS_FRAMES,fidx); ok,im=cap.read(); cap.release()
    return im if ok else None

if __name__=="__main__":
    src=sys.argv[1]; tracks=sys.argv[2] if len(sys.argv)>2 else None; F=int(sys.argv[3]) if len(sys.argv)>3 else 24
    img=grab(src,F); assert img is not None,"kare okunamadı"; h,w=img.shape[:2]
    rec=calibrate_frame(prob(img),w,h,img=img)   # VP-entegre: baseline koru, bowtie'de VP-recovery dene
    if not rec.get("ok"): print("KALİB YOK:",rec.get("reason")); sys.exit()
    ok,why=sanity(rec,w,h,img=img); src_tag=rec.get("vp_source","?")
    print(f"calib res={rec['fit']:.2f}m camside={rec['camside']} src={src_tag} | SANITY: {'GEÇTİ' if ok else 'ELENDİ -> '+why}")
    feet=[]
    if tracks and os.path.exists(tracks):
        import pandas as pd; d=pd.read_parquet(tracks); g=d[d.frame==F]
        feet=g[["foot_x","foot_y"]].values
    mp=project_feet(feet,rec,w,h) if len(feet) else np.empty((0,2))
    tag=f"OTONOM 2D  res={rec['fit']:.2f}m  {src_tag}  {'SANITY-OK' if ok else 'KAYIK-ELENDI'}"
    twod,n=draw_2d(mp,rec["camside"],tag)
    Hpx=twod.shape[0]; raw=cv2.resize(img,(int(w*Hpx/h),Hpx))
    sheet=np.hstack([raw,np.full((Hpx,6,3),60,np.uint8),twod])
    out=os.path.join(HERE,"cand","_AUTO2D_"+os.path.basename(src).split(".")[0][:10]+".jpg")
    cv2.imwrite(out,sheet); print(f"oyuncu={n} -> {out}")
