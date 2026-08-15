#!/usr/bin/env python3
import os, sys, json, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from calib_train import auto_calib as AC
from calib_train import auto_clean2d as AC2
from calib_train import interior_check as IC
from calib_train.night import method_convex_penalty as MP

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(REPO, "calib_train", "night", "probcache")
SCR = "/tmp/claude-1000/-home-schrodiger/e775b40a-a57e-4b9f-a260-3633299dc6e8/scratchpad"
man = json.load(open(os.path.join(CACHE, "manifest.json")))
byidx = {e["idx"]: e for e in man}

# channel colors (BGR): boundaries green; center=pink, box=purple, circle=cyan
COL = {0: (0,255,0),1:(0,255,0),2:(0,255,0),3:(0,200,255),
       4:(203,0,255),5:(255,0,150),6:(255,255,0)}

def overlay(idx, rec, tag, weight=None):
    e = byidx[idx]; file,h,w = e["file"],e["h"],e["w"]
    img = cv2.imread(os.path.join(REPO,"calib_train","cand_big",file)).copy()
    m = AC.clean_label_masks(rec, w, h)
    for ci in range(7):
        mask = m[ci] > 0
        img[mask] = (0.35*np.array(img[mask]) + 0.65*np.array(COL[ci])).astype(np.uint8)
    cv2.putText(img, tag, (12,40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,0), 5)
    cv2.putText(img, tag, (12,40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,255), 2)
    out = os.path.join(SCR, f"wf_convex_penalty_{idx}.png")
    # downscale for viewing
    sc = 1100.0/w
    img = cv2.resize(img,(int(w*sc),int(h*sc)))
    cv2.imwrite(out, img); return out

WEIGHT = 2.0
targets = {"new_rec":[12,14,112], "regress":[65,111]}
for kind, idxs in targets.items():
    for idx in idxs:
        e = byidx[idx]; file,h,w = e["file"],e["h"],e["w"]
        prob = np.load(os.path.join(CACHE,f"{idx:03d}.npz"))["prob"].astype("float32")
        img = cv2.imread(os.path.join(REPO,"calib_train","cand_big",file))
        rm = MP.calib_from_pred_cp(prob,w,h,weight=WEIGHT)
        rb = AC.calib_from_pred(prob,w,h)
        ic_m = IC.interior_consistency(rm,prob,w,h) if rm.get("ok") else None
        ic_b = IC.interior_consistency(rb,prob,w,h) if rb.get("ok") else None
        sm = AC2.sanity(rm,w,h,img=img)
        sb = AC2.sanity(rb,w,h,img=img)
        print(f"[{kind}] idx={idx} {file[:30]} | METH ok={rm.get('ok')} fit={rm.get('fit')} sanity={sm} IC={ic_m} | BASE ok={rb.get('ok')} fit={rb.get('fit')} sanity={sb} IC={ic_b}")
        if rm.get("ok"):
            p = overlay(idx, rm, f"idx{idx} METH w={WEIGHT} fit={rm['fit']:.2f} IC={ic_m}")
            print("   ->",p)
