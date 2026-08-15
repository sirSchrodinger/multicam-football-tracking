#!/usr/bin/env python3
"""TEŞHİS: goalF (uzak-kale) <-> center (orta-çizgi) karışması (Alperen hipotezi #1).
Mekanik: uzak-kale karanlık/görünmezse solver goalF'i CENTER çizgisine atar -> reprojekte
X=L (far-goal) gerçek orta-çizgiye oturur -> saha ~2x çöker = YAMUK.

SİNYAL (non-gameable, held-out center kanalı ch4):
  c_at_far    = ch4(center-prob) reprojekte far-goal (X=L) çizgisi boyunca
  c_at_center = ch4(center-prob) reprojekte center   (X=L/2) çizgisi boyunca
  confusion   = c_at_far - c_at_center   (YÜKSEK => far-goal aslında orta-çizgide = karışma)

interior_support SADECE c_at_center'a bakar (pozitif destek); c_at_far'ı GÖRMEZ -> kör nokta.
Çıktı: night/_center_confusion.json (per-idx) + konsol tablo + hedef idx overlay'leri.
"""
import os, sys, json, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))
from calib_train import auto_calib as AC
from calib_train.multihyp_calib import groups_topk, ROLES, _sample_prob
L,Wp=AC.L,18.0
PC=os.path.join(HERE,"probcache_v3")
MAN={r['idx']:r for r in json.load(open(os.path.join(PC,"manifest.json")))}
CEN=json.load(open(os.path.join(ROOT,"cand","_census_hr3.json")))
CANDDIR=os.path.join(ROOT,"cand_big")

def load_prob(idx):
    d=np.load(os.path.join(PC,f"{idx:03d}.npz")); return d['prob'].astype(np.float32), int(d['w']), int(d['h'])

def solve_for_combo(prob,w,h,combo):
    """census combo -> groups (topK, alan-sıralı, combo=CC index) -> solve_calib rec."""
    cands,why=groups_topk(prob,w,h,K=3)
    if cands is None: return None
    try:
        groups={r:cands[r][ci] for r,ci in zip(ROLES,combo)}
    except (IndexError,KeyError):
        # combo topK'yı aşıyorsa largest-CC'ye düş
        groups={r:cands[r][0] for r in ROLES}
    return AC.solve_calib(groups,w/2,h/2,w/2), groups, cands

def sample_center_chan(rec,prob,w,h,X):
    """ch4 (center-prob) reprojekte X-sabit metrik çizgi (Y:0..Wp) boyunca ortalama."""
    k1,k2,H=rec["k1"],rec["k2"],rec["H"]; cx,cy,s=w/2,h/2,w/2
    line=np.linspace([X,0],[X,Wp],60)
    pix=AC.project_metric(line,k1,k2,H,cx,cy,s)
    m,cov=_sample_prob(prob[4],pix,w,h)   # ch4 = center
    return (m if m is not None else 0.0), cov, pix

def diag_idx(idx):
    ce=CEN.get(str(idx))
    if not ce or ce.get('src') is None: return None
    prob,w,h=load_prob(idx)
    combo=ce.get('combo',[0,0,0,0]) or [0,0,0,0]
    out=solve_for_combo(prob,w,h,combo)
    if out is None or out[0] is None: return None
    rec,groups,cands=out
    c_far,cov_far,_   =sample_center_chan(rec,prob,w,h,L)      # far-goal X=L
    c_cen,cov_cen,_   =sample_center_chan(rec,prob,w,h,L/2)    # center X=L/2
    # goalF kanalı (ch1) reprojekte far-goal boyunca (doğruysa yüksek olmalı)
    k1,k2,H=rec["k1"],rec["k2"],rec["H"]
    gf_far,_=_sample_prob(prob[1],AC.project_metric(np.linspace([L,0],[L,Wp],60),k1,k2,H,w/2,h/2,w/2),w,h)
    return dict(idx=idx,src=ce['src'],fit=round(float(rec['fit']),3),score=ce.get('score'),
                combo=list(combo),camside=rec['camside'],
                c_at_far=round(c_far,3),cov_far=round(cov_far,2),
                c_at_center=round(c_cen,3),cov_center=round(cov_cen,2),
                gf_at_far=round(gf_far if gf_far else 0.0,3),
                confusion=round(c_far-c_cen,3),
                k1=round(float(rec['k1']),3),k2=round(float(rec['k2']),3))

def main():
    targets=[int(x) for x in sys.argv[1:]] if len(sys.argv)>1 else list(range(121))
    rows=[]
    for idx in targets:
        try:
            r=diag_idx(idx)
            if r: rows.append(r)
        except Exception as e:
            print(f"idx{idx} HATA: {e}")
    rows.sort(key=lambda r:-r['confusion'])
    print(f"\n{'idx':>4} {'src':>9} {'fit':>5} {'score':>6} {'c@far':>6} {'c@cen':>6} {'CONF':>6} {'gf@far':>6} {'combo':>12}")
    print("-"*80)
    for r in rows:
        flag=" <== KARISMA" if r['confusion']>0.15 else ""
        print(f"{r['idx']:>4} {str(r['src']):>9} {r['fit']:>5} {str(r['score']):>6} "
              f"{r['c_at_far']:>6} {r['c_at_center']:>6} {r['confusion']:>6} {r['gf_at_far']:>6} {str(r['combo']):>12}{flag}")
    json.dump(rows,open(os.path.join(HERE,"_center_confusion.json"),"w"),ensure_ascii=False,indent=1)
    n_conf=sum(1 for r in rows if r['confusion']>0.15)
    print(f"\nToplam çözülmüş: {len(rows)} | c_at_far>c_at_center+0.15 (karışma-şüphe): {n_conf}")
    print(f"-> {os.path.join(HERE,'_center_confusion.json')}")

if __name__=="__main__": main()
