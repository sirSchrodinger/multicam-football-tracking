#!/usr/bin/env python3
"""KARAR: karanlık sahalarda inference-CLAHE hr2 segmentasyonunu/çözümünü iyileştiriyor mu?
İyileştiriyorsa BEDAVA kazanç (cache'i CLAHE ile yeniden üret). Token/RunPod yok.
Her karanlık idx: prob(raw) vs prob(CLAHE) → boundary-kanal aktivasyon kütlesi + calibrate_frame ok?"""
import os,sys,json,numpy as np,cv2,time
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
HERE=os.path.dirname(os.path.abspath(__file__)); CT=os.path.dirname(HERE); ROOT=os.path.dirname(CT)
sys.path.insert(0,ROOT)
from calib_train import auto_clean2d as AC2
MAN=json.load(open(f"{HERE}/probcache_v3/manifest.json"))
def clahe(img):
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB); cl=cv2.createCLAHE(2.5,(8,8))
    lab[...,0]=cl.apply(lab[...,0]); return cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
def bright(img): return float(cv2.cvtColor(cv2.resize(img,(320,180)),cv2.COLOR_BGR2GRAY).mean())
# en karanlık 14 (b<60)
cand=[]
for e in MAN:
    im=cv2.imread(f"{CT}/cand_big/{e['file']}")
    if im is None: continue
    cand.append((e,bright(im)))
dark=[e for e,b in sorted(cand,key=lambda x:x[1]) if b<60][:14]
print(f"karanlık test seti: {len(dark)} saha\n"+"="*70)
def bmass(p): return float(p[:4].mean())  # 4 sınır kanalı ort aktivasyon
res=[]
t0=time.time()
for e in dark:
    idx=e['idx']; im=cv2.imread(f"{CT}/cand_big/{e['file']}"); w,h=e['w'],e['h']; im=cv2.resize(im,(w,h))
    imc=clahe(im)
    p_raw=AC2.prob(im); p_cl=AC2.prob(imc)
    r_raw=AC2.calibrate_frame(p_raw,w,h,img=im)         # sanity orijinal img ile
    r_cl =AC2.calibrate_frame(p_cl ,w,h,img=im)
    ok_r=bool(r_raw.get('ok')); ok_c=bool(r_cl.get('ok'))
    res.append((idx,ok_r,ok_c,bmass(p_raw),bmass(p_cl)))
    flag="  <== CLAHE KAZANDIRDI" if (ok_c and not ok_r) else ("  (raw'da vardı)" if ok_r and not ok_c else "")
    print(f"idx{idx:3d} {e['file'].split('__')[0][:20]:20s} ok raw={ok_r} clahe={ok_c} | akt raw={bmass(p_raw):.4f} clahe={bmass(p_cl):.4f}{flag}",flush=True)
nr=sum(r[1] for r in res); nc=sum(r[2] for r in res)
gain=sum(1 for r in res if r[2] and not r[1]); lost=sum(1 for r in res if r[1] and not r[2])
akt_up=sum(1 for r in res if r[4]>r[3]*1.15)
print("="*70)
print(f"SONUÇ ({time.time()-t0:.0f}s): solve raw={nr}/14  clahe={nc}/14 | CLAHE kazandırdı {gain}, kaybettirdi {lost} | aktivasyon-artan {akt_up}/14")
print("KARAR:", "CLAHE BEDAVA KAZANÇ → cache'i CLAHE ile yeniden üret" if nc>nr+1 or akt_up>=9 else "CLAHE tek başına yetmiyor → dark-aug RunPod eğitimi gerek")
