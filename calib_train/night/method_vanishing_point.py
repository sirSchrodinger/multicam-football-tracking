#!/usr/bin/env python3
"""VANISHING-POINT geometrik kısıtlı kalibrasyon (RECOVERY KATMANI).

Fikir: touchN+touchF metrik uzayda paralel (Y=sabit) -> görüntüde TEK vanishing point'e (VP_X)
yakınsar; goalN+goalF (X=sabit) -> VP_Y'ye. Saha kenar-rolleri için seg2 EN BÜYÜK bileşeni
sık sık UZAK çizgiyi (goalF/touchF) çit/duvara kilitliyor -> bowtie/çöküş.

Bu modül: her uzak-rol için BİRDEN ÇOK aday bileşen çıkarır, her konfigürasyon için VP geometrisini
test eder (konveks quad + her iki VP saha-DIŞINDA + ufuk-çizgisi saha-üstünde). VP-makul konfigleri
solve_calib'e verir, sanity'den geçenler arasından interior_consistency (iç-işaret tutarlılığı, GERÇEK
sinyal) EN YÜKSEK olanı seçer. Default (tüm-en-büyük) zaten geçiyorsa onu korur (regresyon önleme).

ORİJİNALİ BOZMAZ: auto_calib.solve_calib / sanity / interior_consistency aynen kullanılır.
"""
import os, sys, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from calib_train import auto_calib as AC
from calib_train import auto_clean2d as AC2
from calib_train import interior_check as IC

ROLES=["goalN","goalF","touchN","touchF"]

def components(pc, w, h, thr=0.5, minpts=25, maxpts=180, topk=3, seed=0):
    """rol-kanal olasılığı -> en büyük topk bileşenin nokta kümesi listesi (büyükten küçüğe)."""
    rs=np.random.RandomState(seed)
    pc=cv2.resize(pc,(w,h)); mk=(pc>thr).astype(np.uint8)
    mk=cv2.morphologyEx(mk,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
    ncc,lbl,st,_=cv2.connectedComponentsWithStats(mk)
    if ncc<=1: return []
    order=np.argsort(st[1:,cv2.CC_STAT_AREA])[::-1]+1
    out=[]
    for li in order[:topk]:
        pts=np.column_stack(np.where(lbl==li))[:,::-1].astype(float)
        if len(pts)<minpts: continue
        if len(pts)>maxpts: pts=pts[rs.choice(len(pts),maxpts,replace=False)]
        out.append(pts)
    return out

def _quad(fl):
    """fit-çizgi sözlüğü -> 4 köşe sırayla [c00, cL0, cLW, c0W] (saat yönü kapalı poligon)."""
    c00=AC.inter(fl["goalN"],fl["touchN"]); cL0=AC.inter(fl["goalF"],fl["touchN"])
    cLW=AC.inter(fl["goalF"],fl["touchF"]); c0W=AC.inter(fl["goalN"],fl["touchF"])
    return np.array([c00,cL0,cLW,c0W])

def _convex(q):
    d=np.diff(np.vstack([q,q[0]]),axis=0)
    cr=d[:,0]*np.roll(d[:,1],-1)-d[:,1]*np.roll(d[:,0],-1)
    return np.all(cr>0) or np.all(cr<0)

def vp_plausible(fl, w, h):
    """VP geometrik makullük: konveks + her iki VP quad-DIŞI + ufuk saha-üstü (tek taraf).
    Dejenere/head-on (VP sonsuz) durumunu paralel=dış kabul ederek tolere eder."""
    q=_quad(fl)
    if not np.isfinite(q).all(): return False
    diag=np.hypot(w,h)
    if np.abs(q).max() > 12*diag: return False     # köşe çok uzakta -> çökmüş
    if not _convex(q): return False
    poly=q.reshape(-1,1,2).astype(np.float32)
    def Dcross(a,b): return a[0]*b[1]-b[0]*a[1]
    VPx=None; VPy=None
    Dx=Dcross(fl["touchN"],fl["touchF"]); Dy=Dcross(fl["goalN"],fl["goalF"])
    inf_x=abs(Dx)<1e-6; inf_y=abs(Dy)<1e-6
    if not inf_x: VPx=AC.inter(fl["touchN"],fl["touchF"])
    if not inf_y: VPy=AC.inter(fl["goalN"],fl["goalF"])
    # VP'ler quad içindeyse REDDET (çizgiler saha içinde kesişiyor -> uzak çizgi yanlış)
    for VP in (VPx,VPy):
        if VP is not None and np.isfinite(VP).all():
            if cv2.pointPolygonTest(poly,(float(VP[0]),float(VP[1])),False) > 0: return False
    # ufuk çizgisi: iki VP'den geç; tüm köşeler tek tarafta olmalı (saha ufkun altında)
    if VPx is not None and VPy is not None and np.isfinite(VPx).all() and np.isfinite(VPy).all():
        if np.hypot(*(VPx-VPy)) > 1e-3:
            hl=AC.fitL([VPx,VPy])
            sg=hl[0]*q[:,0]+hl[1]*q[:,1]+hl[2]
            if not (np.all(sg>1e-6) or np.all(sg<-1e-6)): return False
    return True

def _fitlines(groups):
    return {r:AC.fitL(groups[r]) for r in ROLES}

def calib_vp(prob, w, h, img=None, topk_far=3, topk_near=2, max_combo=40, min_bright=42.0):
    """VP-kısıtlı recovery. döner rec(dict) + rec['ok'] + rec['vp_source'] etiketi.
    Strateji: default(en-büyük) sanity geçer & interior makulse koru; aksi halde aday bileşenleri
    VP-prefiltre + solve + sanity + interior ile tara, en iyi interior_consistency'yi seç."""
    cx,cy,s=w/2,h/2,w/2
    # karanlık-kare reddi (sanity ile aynı, erken çıkış)
    if img is not None:
        v=float(img.reshape(-1,img.shape[-1]).mean()) if img.ndim==3 else float(img.mean())
        if v<min_bright: return dict(ok=False,reason=f"kare cok karanlik ({v:.0f})")
    cand={}
    for ci,role in enumerate(ROLES):
        tk=topk_far if role in ("goalF","touchF") else topk_near
        cand[role]=components(prob[ci],w,h,topk=tk)
        if not cand[role]:
            return dict(ok=False,reason=f"{role} bulunamadi")
    # --- default config (tum en-buyuk) ---
    def eval_cfg(sel):
        groups={r:cand[r][sel[i]] for i,r in enumerate(ROLES)}
        fl=_fitlines(groups)
        if not vp_plausible(fl,w,h): return None
        rec=AC.solve_calib(groups,cx,cy,s)
        if rec is None: return None
        ok,why=AC2.sanity(rec,w,h,img=img)
        if not ok: return None
        ic=IC.interior_consistency(rec,prob,w,h)
        rec.update(ok=True,groups=groups,ic=ic)
        return rec
    # default first; guclu baseline-pass'i KORU (hiz + regresyon onleme)
    default=eval_cfg((0,0,0,0))
    if default is not None and (default.get("ic") or 0.0) >= 0.15:
        default["vp_source"]="default"; return default
    # enumerate combos, far-lines varied most
    import itertools
    ng=[len(cand[r]) for r in ROLES]  # order goalN,goalF,touchN,touchF
    combos=list(itertools.product(range(ng[0]),range(ng[1]),range(ng[2]),range(ng[3])))
    # prioritize: keep near at 0 first, vary far; cap
    combos.sort(key=lambda c:(c[0]+c[2], c[1]+c[3]))
    combos=combos[:max_combo]
    results=[]
    if default is not None: results.append(("default",default))
    for sel in combos:
        if sel==(0,0,0,0): continue
        rec=eval_cfg(sel)
        if rec is not None: results.append(("vp",rec))
    if not results:
        return dict(ok=False,reason="hicbir VP-makul config sanity gecmedi")
    # secim: default zaten varsa ve interior makulse onu koru; aksi halde en yuksek interior
    def keyf(item):
        tag,rec=item; ic=rec.get("ic"); ic=-1.0 if ic is None else ic
        return (ic, -rec["fit"])
    results.sort(key=keyf, reverse=True)
    best_tag,best=results[0]
    # regresyon korumasi: default sanity-PASS ise ve en-iyi onunla ayni degilse,
    # ancak interior belirgin daha iyiyse degistir (>0.05 mutlak), yoksa default'ta kal
    if default is not None:
        dic=default.get("ic"); dic=-1.0 if dic is None else dic
        bic=best.get("ic"); bic=-1.0 if bic is None else bic
        if best is not default and bic < dic+0.05:
            best,best_tag=default,"default"
    best["vp_source"]=best_tag
    return best

if __name__=="__main__":
    import json
    REPO = os.environ.get("HALISAHA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    CACHE=f"{REPO}/calib_train/night/probcache"
    man=json.load(open(f"{CACHE}/manifest.json"))
    cv2.setNumThreads(2)
    npass=0; out={}
    for e in man:
        idx,file,h,w=e["idx"],e["file"],e["h"],e["w"]
        prob=np.load(f"{CACHE}/{idx:03d}.npz")["prob"].astype("float32")
        img=cv2.imread(f"{REPO}/calib_train/cand_big/{file}")
        rec=calib_vp(prob,w,h,img=img)
        ok=bool(rec.get("ok"))
        out[idx]=dict(file=file,ok=ok,fit=rec.get("fit"),ic=rec.get("ic"),src=rec.get("vp_source"),reason=rec.get("reason"))
        if ok: npass+=1
        print(idx,ok,rec.get("vp_source"),None if rec.get("ic") is None else round(rec.get("ic"),3),flush=True)
    json.dump(out,open(f"{os.path.dirname(__file__)}/vp_result.json","w"),indent=0)
    print("VP PASS:",npass,"/",len(man),flush=True)
