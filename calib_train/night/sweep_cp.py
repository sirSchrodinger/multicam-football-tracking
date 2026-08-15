#!/usr/bin/env python3
import os, sys, json, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from calib_train import auto_calib as AC
from calib_train import auto_clean2d as AC2
from calib_train.night import method_convex_penalty as MP

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(REPO, "calib_train", "night", "probcache")
man = json.load(open(os.path.join(CACHE, "manifest.json")))

# precompute baseline
base_pass = set()
data = []
for e in man:
    idx, file, h, w = e["idx"], e["file"], e["h"], e["w"]
    prob = np.load(os.path.join(CACHE, f"{idx:03d}.npz"))["prob"].astype("float32")
    img = cv2.imread(os.path.join(REPO, "calib_train", "cand_big", file))
    rb = AC.calib_from_pred(prob, w, h)
    if bool(rb.get("ok")) and AC2.sanity(rb, w, h, img=img)[0]:
        base_pass.add(idx)
    data.append((idx, file, h, w, prob, img))

for wgt in (0.5, 1.0, 2.0, 6.0):
    mp = set()
    for idx, file, h, w, prob, img in data:
        rm = MP.calib_from_pred_cp(prob, w, h, weight=wgt)
        if bool(rm.get("ok")) and AC2.sanity(rm, w, h, img=img)[0]:
            mp.add(idx)
    nr = sorted(mp - base_pass)
    rg = sorted(base_pass - mp)
    print(f"weight={wgt}: pass={len(mp)} new_rec={len(nr)}{nr} regress={len(rg)}{rg}")
print(f"baseline={len(base_pass)}")
