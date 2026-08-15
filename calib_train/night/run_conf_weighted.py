#!/usr/bin/env python3
import os, sys, json, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
REPO = os.environ.get("HALISAHA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from calib_train import auto_calib as AC
from calib_train import auto_clean2d as AC2
from calib_train import interior_check as IC
from calib_train.night import method_conf_weighted as M

CACHE=os.path.join(REPO,"calib_train/night/probcache")
man=json.load(open(os.path.join(CACHE,"manifest.json")))
base={r["idx"]:r for r in json.load(open(os.path.join(REPO,"calib_train/night/baseline_rows.json")))}

rows=[]
for e in man:
    idx,fn,h,w=e["idx"],e["file"],e["h"],e["w"]
    prob=np.load(f"{CACHE}/{idx:03d}.npz")["prob"].astype("float32")
    img=cv2.imread(f"{REPO}/calib_train/cand_big/{fn}")
    try:
        rec=M.calib_from_pred_w(prob,w,h)
    except Exception as ex:
        rec=dict(ok=False,reason=f"exc:{ex}",fit=None)
    ok=False; why="no-calib"; ic=None
    if rec.get("ok"):
        sok,why=AC2.sanity(rec,w,h,img=img); ok=sok
        if sok: ic=IC.interior_consistency(rec,prob,w,h)
    rows.append(dict(idx=idx,file=fn,ok=bool(ok),why=why,fit=rec.get("fit"),ic=ic,
                     base_ok=bool(base[idx]["ok"]),base_ic=base[idx]["ic"]))

P=sum(r["ok"] for r in rows); BP=sum(r["base_ok"] for r in rows)
newrec=[r for r in rows if r["ok"] and not r["base_ok"]]
regress=[r for r in rows if r["base_ok"] and not r["ok"]]
print(f"METHOD PASS {P}/{len(rows)}  (baseline {BP})")
print(f"new_recovered={len(newrec)}  regressions={len(regress)}")
print("--- NEW RECOVERED (method PASS, baseline FAIL) ---")
for r in sorted(newrec,key=lambda x:-(x['ic'] or 0)):
    print(f"  idx={r['idx']:3d} fit={r['fit']:.2f} ic={r['ic']:.3f} {r['file'][:38]}")
print("--- REGRESSIONS (baseline PASS, method FAIL) ---")
for r in regress:
    print(f"  idx={r['idx']:3d} base_ic={r['base_ic']} why={r['why'][:40]} {r['file'][:30]}")
json.dump(rows,open(f"{CACHE}/../conf_weighted_rows.json","w"),indent=0)
