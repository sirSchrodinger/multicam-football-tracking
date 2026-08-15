#!/usr/bin/env python3
"""Kalibre saha-çizgilerini kareye GERİ-PROJEKTE et, overlay PNG yaz (görsel doğrulama)."""
import os, sys, json, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
REPO = os.environ.get("HALISAHA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from calib_train import auto_calib as AC
from calib_train.night import method_conf_weighted as M
OUT="/tmp/claude-1000/-home-schrodiger/e775b40a-a57e-4b9f-a260-3633299dc6e8/scratchpad"
CACHE=os.path.join(REPO,"calib_train/night/probcache")
man={e["idx"]:e for e in json.load(open(os.path.join(CACHE,"manifest.json")))}
# kanal renkleri (BGR): sınırlar belirgin, iç-işaretler ayırt edici
COL={0:(0,255,0),1:(0,200,255),2:(255,255,0),3:(255,0,0),4:(203,0,255),5:(255,0,170),6:(0,255,255)}
NAME={0:"goalN",1:"goalF",2:"touchN",3:"touchF",4:"center(pembe)",5:"box(mor)",6:"circle(sari)"}

def overlay(idx,tag):
    e=man[idx]; fn,h,w=e["file"],e["h"],e["w"]
    prob=np.load(f"{CACHE}/{idx:03d}.npz")["prob"].astype("float32")
    img=cv2.imread(f"{REPO}/calib_train/cand_big/{fn}")
    rec=M.calib_from_pred_w(prob,w,h)
    if not rec.get("ok"): print(f"idx{idx} NO CALIB"); return
    mk=AC.clean_label_masks(rec,w,h)
    ov=img.copy()
    for ci in range(7):
        m=mk[ci]>0
        ov[m]=COL[ci]
    out=img.copy(); cv2.addWeighted(ov,0.85,out,0.15,0,out)
    cv2.putText(out,f"{tag} idx{idx} fit={rec['fit']:.2f} {rec['camside']}",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(255,255,255),2,cv2.LINE_AA)
    # küçült (Read kolay)
    sc=1000/w; out=cv2.resize(out,(int(w*sc),int(h*sc)))
    p=f"{OUT}/wf_conf_weighted_{tag}_{idx}.png"; cv2.imwrite(p,out)
    print(f"idx{idx} -> {p}")

if __name__=="__main__":
    for a in sys.argv[1:]:
        tag,idx=a.split(":"); overlay(int(idx),tag)
