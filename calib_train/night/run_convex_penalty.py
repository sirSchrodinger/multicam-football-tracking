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

base_pass, meth_pass = set(), set()
rows = []
for e in man:
    idx, file, h, w = e["idx"], e["file"], e["h"], e["w"]
    prob = np.load(os.path.join(CACHE, f"{idx:03d}.npz"))["prob"].astype("float32")
    img = cv2.imread(os.path.join(REPO, "calib_train", "cand_big", file))
    # baseline
    rb = AC.calib_from_pred(prob, w, h)
    bok = bool(rb.get("ok")) and AC2.sanity(rb, w, h, img=img)[0]
    # method
    rm = MP.calib_from_pred_cp(prob, w, h)
    mok = bool(rm.get("ok")) and AC2.sanity(rm, w, h, img=img)[0]
    if bok: base_pass.add(idx)
    if mok: meth_pass.add(idx)
    rows.append((idx, file, bok, mok,
                 None if not rb.get("ok") else round(rb["fit"], 2),
                 None if not rm.get("ok") else round(rm["fit"], 2)))

new_rec = sorted(meth_pass - base_pass)
regress = sorted(base_pass - meth_pass)
print(f"BASELINE pass: {len(base_pass)}/121")
print(f"METHOD   pass: {len(meth_pass)}/121")
print(f"NEW recovered (meth pass, base fail): {len(new_rec)} -> {new_rec}")
print(f"REGRESSIONS (base pass, meth fail): {len(regress)} -> {regress}")
print("--- new recovered detail ---")
for idx in new_rec:
    r = [x for x in rows if x[0] == idx][0]
    print(f"  idx={idx:3d} {r[1][:34]:34s} base_fit={r[4]} meth_fit={r[5]}")
print("--- regression detail ---")
for idx in regress:
    r = [x for x in rows if x[0] == idx][0]
    print(f"  idx={idx:3d} {r[1][:34]:34s} base_fit={r[4]} meth_fit={r[5]}")

json.dump({"base_pass": sorted(base_pass), "meth_pass": sorted(meth_pass),
           "new_rec": new_rec, "regress": regress},
          open("/tmp/claude-1000/-home-schrodiger/e775b40a-a57e-4b9f-a260-3633299dc6e8/scratchpad/cp_result.json", "w"))
