#!/usr/bin/env python3
"""TEMPORAL otonom kalibrasyon (DEPLOYMENT yolu — KANITLANDI 30 Haz: Çankaya tek-kare 1.38m → temporal 0.23m).
Statik saha-çizgileri çok karede SABİT, oyuncu/gürültü HAREKETLİ → seg-olasılıklarının temporal-MEDYAN'ı
far-çizgiyi denoise eder → güvenilir calib. Tek-kareden ÇOK daha sağlam (deploy'da video zaten var).

API:  temporal_calibrate(video_path, seg_fn, N=30) -> (rec, ok, why)
      seg_fn(bgr_frame) -> (7,Hs,Ws) sigmoid olasılık (model'e bağımsız kalsın diye dışarıdan verilir).
CLI:  python -m calib_train.temporal_calib <video> [N]   (seg2_hr2.pth yükler + overlay yazar)
"""
import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calib_train import auto_calib as AC
from calib_train.auto_clean2d import sanity

def temporal_median_prob(video_path, seg_fn, N=30, lo=0.1, hi=0.9):
    """video'dan N kare örnekle -> seg -> temporal-medyan 7-kanal olasılık + referans kare."""
    cap=cv2.VideoCapture(video_path); nf=int(cap.get(7))
    if nf<=1: cap.release(); raise RuntimeError(f"video okunamadı/boş: {video_path}")
    idxs=np.linspace(int(nf*lo),int(nf*hi),N).astype(int); probs=[]; ref=None
    for k,fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(fi)); ok,fr=cap.read()
        if not ok: continue
        probs.append(seg_fn(fr))
        if k==len(idxs)//2: ref=fr.copy()
    cap.release()
    if len(probs)<3: raise RuntimeError(f"yeterli kare yok ({len(probs)})")
    return np.median(np.stack(probs),0), ref

def temporal_calibrate(video_path, seg_fn, N=30, use_vp=True):
    """döner (rec, ok, why). İKİ KAZANCI birleştirir: temporal-median (gürültü-denoise) +
    VP-recovery (fence-lock/bowtie kurtarma, calib_vp_strict; baseline-PASS korunur). use_vp=False -> sadece baseline."""
    med,ref=temporal_median_prob(video_path,seg_fn,N)
    h,w=ref.shape[:2]
    if use_vp:
        from calib_train.vp_calib import calib_vp_strict
        rec=calib_vp_strict(med,w,h,img=ref)
    else:
        rec=AC.calib_from_pred(med,w,h)
    if not rec.get("ok"): return rec,False,rec.get("reason","calib-yok")
    ok,why=sanity(rec,w,h,img=ref); rec["_ref"]=ref
    return rec,ok,why

if __name__=="__main__":
    import torch
    from calib_train.render2d_clean import UNet
    vid=sys.argv[1]; N=int(sys.argv[2]) if len(sys.argv)>2 else 30
    HERE=os.path.dirname(os.path.abspath(__file__)); DEV="cuda" if torch.cuda.is_available() else "cpu"
    net=UNet().to(DEV); net.load_state_dict(torch.load(os.path.join(HERE,"seg2_hr2.pth"),map_location=DEV)); net.eval()
    TW,TH=1024,576
    @torch.no_grad()
    def seg_fn(bgr):
        x=torch.from_numpy(cv2.resize(bgr,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None].to(DEV)
        return torch.sigmoid(net(x))[0].cpu().numpy()
    rec,ok,why=temporal_calibrate(vid,seg_fn,N)
    print(f"TEMPORAL calib({N}kare): res={rec.get('fit')} camside={rec.get('camside')} | sanity: {'PASS' if ok else why}")
    if rec.get("ok"):
        ref=rec["_ref"]; h,w=ref.shape[:2]; m=AC.clean_label_masks(rec,w,h)
        COL=[(0,0,255),(0,165,255),(0,255,0),(255,140,0),(0,255,255),(255,0,200),(255,150,255)]
        o=ref.copy()
        for c in range(7): o[m[c]>80]=COL[c]
        out=os.path.join(HERE,"cand","_TEMPORAL_CALIB.jpg"); cv2.imwrite(out,o); print("->",out)
