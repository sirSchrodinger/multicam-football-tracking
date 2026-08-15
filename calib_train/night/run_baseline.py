#!/usr/bin/env python3
"""Baseline: largest-component calib_from_pred + sanity. Tabloyu yaz."""
import os, sys, json, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
REPO = os.environ.get("HALISAHA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from calib_train import auto_calib as AC
from calib_train import auto_clean2d as AC2
from calib_train import interior_check as IC

CACHE=os.path.join(REPO,"calib_train/night/probcache")
man=json.load(open(os.path.join(CACHE,"manifest.json")))

rows=[]
for e in man:
    idx,fn,h,w=e["idx"],e["file"],e["h"],e["w"]
    prob=np.load(f"{CACHE}/{idx:03d}.npz")["prob"].astype("float32")
    img=cv2.imread(f"{REPO}/calib_train/cand_big/{fn}")
    rec=AC.calib_from_pred(prob,w,h)
    ok=False; why="no-calib"
    ic=None
    if rec.get("ok"):
        sok,why=AC2.sanity(rec,w,h,img=img)
        ok=sok
        if sok:
            ic=IC.interior_consistency(rec,prob,w,h)
    rows.append(dict(idx=idx,file=fn,ok=bool(ok),why=why,fit=rec.get("fit"),ic=ic))

P=sum(r["ok"] for r in rows)
print(f"BASELINE PASS {P}/{len(rows)}")
json.dump(rows,open(f"{CACHE}/../baseline_rows.json","w"),indent=0)
for r in rows:
    if r["ok"]:
        print(f"  PASS idx={r['idx']:3d} fit={r['fit']:.2f} ic={r['ic']} {r['file'][:34]}")
