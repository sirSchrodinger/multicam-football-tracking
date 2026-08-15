#!/usr/bin/env python3
"""VP-RECOVERY production calib (Workflow kazananı, adversaryal-doğrulandı 1 Tem).
calib_vp_strict: largest-component default sanity-PASS ise KORU (regresyon-yok); aksi halde VP-geometri
prefiltre + çok-aday far-line + sanity + interior_consistency ile recovery DENE, ama sadece IC>=IC_MIN
(default 0.45 — genuine recovery'ler burada kümeleniyor; düşük-IC kuyruk gate-overfit=SAHTE) kabul et.
Headline-şişme yok; sadece görsel-savunulabilir kurtarmalar.
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calib_train import auto_calib as AC
from calib_train import auto_clean2d as AC2
from calib_train.night.method_vanishing_point import calib_vp
# WARP-DOĞRULANDI (1 Tem, AM): montaj _VP_RECOVERED_AM.jpg her recovery'yi top-down warp+kanonik-grid ile
# GÖZLE denetledim. IC 0.45-0.49 bandı gate-overfit=SAHTE (idx26/58 ic0.475 bowed/collapse, idx60 ic0.46
# merkez-yuvarlak doğrulanamaz); GERÇEK kurtarmalar IC>=0.50'de kümeleniyor (idx6 ic0.52, idx118 ic0.55 —
# tutarlı saha, düz çizgi, yuvarlak merkez-çember; sadece ~%10 yatay ofset = sistemik ölçek bandı ±%13).
# -> IC_MIN 0.45'ten 0.50'ye çekildi (SAHTE'leri ELE, headline şişirme değil KÜÇÜLT).
IC_MIN=float(os.environ.get("VP_IC_MIN","0.50"))
# default-source (baseline sanity-fail ama VP-plausible en-büyük config): bowed-sahte idx12 ic0.20 buradaydı.
DEF_IC_MIN=float(os.environ.get("VP_DEF_IC_MIN","0.35"))

def calib_vp_strict(prob, w, h, img=None, ic_min=IC_MIN, def_ic_min=DEF_IC_MIN):
    """DOĞRU ENTEGRASYON: baseline (largest-component) sanity-PASS ise AYNEN koru (REGRESYON YOK).
    SADECE baseline-fail'de VP-recovery dene; sadece IC>=ic_min (genuine, gate-overfit değil) kabul.
    default-source recovery de IC>=def_ic_min zorunlu (düşük-IC bowed = warp-sahte)."""
    base=AC.calib_from_pred(prob,w,h)
    if base.get("ok"):
        ok,_=AC2.sanity(base,w,h,img=img)
        if ok:
            base["vp_source"]="baseline"; return base
    # baseline FAIL -> VP recovery katmanı
    rec=calib_vp(prob,w,h,img=img)
    if not rec.get("ok"): return rec
    if rec.get("vp_source")=="vp":
        ic=rec.get("ic")
        if ic is None or ic<ic_min:
            return dict(ok=False, reason=f"vp-recovery dusuk-IC {ic} (<{ic_min}) -> reddedildi")
    elif rec.get("vp_source")=="default":
        # calib_vp default'u kabul etti ama baseline sanity-fail'di (VP-plausible default) -> IC-tabanı iste
        ic=rec.get("ic")
        if ic is None or ic<def_ic_min:
            return dict(ok=False, reason=f"vp-default dusuk-IC {ic} (<{def_ic_min}) -> reddedildi")
    return rec

if __name__=="__main__":
    import json, cv2
    cv2.setNumThreads(2)
    REPO = os.environ.get("HALISAHA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); CACHE=f"{REPO}/calib_train/night/probcache"
    man=json.load(open(f"{CACHE}/manifest.json"))
    npass=0; nrec=0; recovered=[]; rows={}
    for e in man:
        idx,file,h,w=e["idx"],e["file"],e["h"],e["w"]
        prob=np.load(f"{CACHE}/{idx:03d}.npz")["prob"].astype("float32")
        img=cv2.imread(f"{REPO}/calib_train/cand_big/{file}")
        rec=calib_vp_strict(prob,w,h,img=img); ok=bool(rec.get("ok"))
        if ok:
            npass+=1
            if rec.get("vp_source")=="vp": nrec+=1; recovered.append((idx,file,round(rec.get("ic") or 0,3),round(rec.get("fit") or 0,3)))
        rows[idx]=dict(file=file,ok=ok,src=rec.get("vp_source"),ic=rec.get("ic"),fit=rec.get("fit"))
    json.dump({"pass":npass,"vp_recovered":nrec,"recovered":recovered,"rows":rows},
              open(f"{REPO}/calib_train/cand/_vp_strict.json","w"),ensure_ascii=False,indent=1)
    print(f"VP-STRICT (IC>={IC_MIN}): {npass}/{len(man)} PASS | vp-recovery {nrec} | recovered idx: {[r[0] for r in recovered]}")
