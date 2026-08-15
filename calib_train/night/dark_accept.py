#!/usr/bin/env python3
"""KABUL TESTİ (token-BEDAVA): seg2_hr_dark vs seg2_hr2, solve-oranı.
14 KARANLIK (dark-aug kazandırdı mı) + 24 PARLAK-kontrol (regresyon var mı).
dark2 CLAHE'SİZ eğitildi (train_hr_dark2=ağırlıklı-BCE) → inference iki modelde de CLAHE'siz.
KARAR: net-pozitifse hr_dark'ı öner."""
import os,sys,json,numpy as np,cv2,time
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
HERE=os.path.dirname(os.path.abspath(__file__)); CT=os.path.dirname(HERE); ROOT=os.path.dirname(CT)
sys.path.insert(0,ROOT)
import torch
from calib_train.render2d_clean import UNet
from calib_train import auto_clean2d as AC2
MAN=json.load(open(f"{HERE}/probcache_v3/manifest.json"))
TW,TH=1024,576
DARK=os.path.join(CT,"seg2_hr_dark.pth")
def clahe(img):
    lab=cv2.cvtColor(img,cv2.COLOR_BGR2LAB); cl=cv2.createCLAHE(2.5,(8,8))
    lab[...,0]=cl.apply(lab[...,0]); return cv2.cvtColor(lab,cv2.COLOR_LAB2BGR)
def mk(p): n=UNet(); n.load_state_dict(torch.load(p,map_location="cpu")); n.eval(); return n
@torch.no_grad()
def prob(net,img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None]
    return torch.sigmoid(net(x))[0].numpy()
def bright(img): return float(cv2.cvtColor(cv2.resize(img,(320,180)),cv2.COLOR_BGR2GRAY).mean())
def main():
    if not os.path.exists(DARK):
        print(f"seg2_hr_dark.pth YOK ({DARK}) — guard indirmesini bekle"); return 1
    net2=mk(f"{CT}/seg2_hr2.pth"); netd=mk(DARK)
    rows=[]
    for e in MAN:
        im=cv2.imread(f"{CT}/cand_big/{e['file']}")
        if im is None: continue
        rows.append((e,bright(im)))
    rows.sort(key=lambda x:x[1])
    dark=[e for e,b in rows if b<60][:14]
    bright_ctrl=[e for e,b in rows if b>=100][-24:]   # en parlak 24 = regresyon kontrolü
    def run(net,e,use_clahe):
        im=cv2.imread(f"{CT}/cand_big/{e['file']}"); w,h=e['w'],e['h']; im=cv2.resize(im,(w,h))
        p=prob(net,clahe(im) if use_clahe else im)
        return bool(AC2.calibrate_frame(p,w,h,img=im).get('ok'))
    t0=time.time()
    print("=== KARANLIK 14 (ikisi de CLAHE'siz) ===")
    d2=dd=0
    for e in dark:
        a=run(net2,e,False); b=run(netd,e,False); d2+=a; dd+=b
        fl="  <== DARK KAZANDI" if b and not a else ("  (kayıp)" if a and not b else "")
        print(f"idx{e['idx']:3d} {e['file'].split('__')[0][:20]:20s} hr2={a} dark={b}{fl}",flush=True)
    print(f"KARANLIK: hr2={d2}/14  dark={dd}/14")
    print("=== PARLAK-kontrol 24 (regresyon) ===")
    b2=bd=0
    for e in bright_ctrl:
        a=run(net2,e,False); b=run(netd,e,False); b2+=a; bd+=b
        fl="  <== REGRESYON" if a and not b else ""
        if a!=b: print(f"idx{e['idx']:3d} {e['file'].split('__')[0][:20]:20s} hr2={a} dark={b}{fl}",flush=True)
    print(f"PARLAK: hr2={b2}/24  dark={bd}/24")
    gain=dd-d2; reg=b2-bd
    print("="*60)
    print(f"SONUÇ ({time.time()-t0:.0f}s): karanlık net {gain:+d} | parlak regresyon {reg:+d} (pozitif=kötü)")
    verdict="hr_dark ÖNER (net kazanç, regresyonu bastırıyor)" if gain-reg>=2 else ("KARARSIZ — el-inceleme" if gain-reg>=0 else "hr2 TUT (dark net kaybettirdi)")
    print("KARAR:",verdict)
if __name__=="__main__": sys.exit(main() or 0)
