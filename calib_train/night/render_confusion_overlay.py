#!/usr/bin/env python3
"""GÖRSEL TEŞHİS: yamuk sahalarda seg2'nin her rolü NEREYE ateşlediği + solver'ın kurduğu saha.
Her idx için 2-satır montaj:
  ÜST : ham kare + REPROJEKTE solved saha (goalN=kırmızı goalF=turuncu-kalın center=sarı
        touchN=yeşil touchF=mavi box=mor circle=pembe) -> solver sahayı NEREYE koydu
  ALT : ham kare + seg2 KANAL maskeleri (aynı renk kodu, >0.4) -> model her rolü NEREYE ateşledi
Karşılaştır: reprojekte far-goal (turuncu çizgi, üst) gerçek goalF-maskesine (turuncu, alt) oturuyor mu?
center-maskesi (sarı) ayrı bir çizgi mi? solver far-goal'ü center-maskesine mi koydu?
Kullanım: python render_confusion_overlay.py 10 13 35 59 61 90 5 1
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
# BGR renkler: goalN goalF touchN touchF center box circle
COL=[(40,40,235),(40,150,255),(40,220,40),(235,150,40),(40,230,235),(230,40,200),(200,120,230)]
NAME=["goalN","goalF","touchN","touchF","center","box","circle"]

def load_prob(idx):
    d=np.load(os.path.join(PC,f"{idx:03d}.npz")); return d['prob'].astype(np.float32), int(d['w']), int(d['h'])

def solve_for_combo(prob,w,h,combo):
    cands,why=groups_topk(prob,w,h,K=3)
    if cands is None: return None,None
    try: groups={r:cands[r][ci] for r,ci in zip(ROLES,combo)}
    except (IndexError,KeyError): groups={r:cands[r][0] for r in ROLES}
    return AC.solve_calib(groups,w/2,h/2,w/2), groups

def draw_field(img,rec):
    k1,k2,H=rec["k1"],rec["k2"],rec["H"]; cx,cy,s=img.shape[1]/2,img.shape[0]/2,img.shape[1]/2
    o=img.copy()
    def ln(a,b,ci,th,N=80,closed=False):
        pts=np.linspace(a,b,N) if not closed else a
        pix=AC.project_metric(pts,k1,k2,H,cx,cy,s); fin=np.isfinite(pix).all(1)
        if fin.sum()>=2: cv2.polylines(o,[pix[fin].astype(np.int32)],closed,COL[ci],th,cv2.LINE_AA)
    ln([0,0],[0,Wp],0,4); ln([L,0],[L,Wp],1,5)              # goalN goalF(kalın)
    ln([0,0],[L,0],2,3); ln([0,Wp],[L,Wp],3,3)              # touchN touchF
    ln([L/2,0],[L/2,Wp],4,4)                                 # center(sarı)
    for gx in (0.0,L):
        bx=5 if gx==0 else L-5
        ln([gx,Wp/2-5],[bx,Wp/2-5],5,2); ln([bx,Wp/2-5],[bx,Wp/2+5],5,2); ln([bx,Wp/2+5],[gx,Wp/2+5],5,2)
    th=np.linspace(0,2*np.pi,60); circ=np.stack([L/2+3*np.cos(th),Wp/2+3*np.sin(th)],1); ln(circ,None,6,2,closed=True)
    return cv2.addWeighted(o,0.75,img,0.25,0)

def draw_channels(img,prob):
    h,w=img.shape[:2]; o=img.copy()
    for ci in range(7):
        m=cv2.resize(prob[ci],(w,h))>0.4
        o[m]=COL[ci]
    return cv2.addWeighted(o,0.6,img,0.4,0)

def one(idx):
    ce=CEN.get(str(idx))
    if not ce: return None
    img=cv2.imread(os.path.join(CANDDIR,MAN[idx]['file']))
    if img is None: return None
    prob,w,h=load_prob(idx); img=cv2.resize(img,(w,h))
    combo=ce.get('combo',[0,0,0,0]) or [0,0,0,0]
    top=img.copy(); tag=f"idx{idx} src={ce.get('src')}"
    if ce.get('src') is not None:
        rec,groups=solve_for_combo(prob,w,h,combo)
        if rec is not None:
            top=draw_field(img,rec)
            gf,_=_sample_prob(prob[1],AC.project_metric(np.linspace([L,0],[L,Wp],60),rec['k1'],rec['k2'],rec['H'],w/2,h/2,w/2),w,h)
            cf,_=_sample_prob(prob[4],AC.project_metric(np.linspace([L,0],[L,Wp],60),rec['k1'],rec['k2'],rec['H'],w/2,h/2,w/2),w,h)
            tag=f"idx{idx} {ce.get('src')} fit={rec['fit']:.2f} gf@far={gf or 0:.2f} c@far={cf or 0:.2f} combo={combo}"
    bot=draw_channels(img,prob)
    for im,t in ((top,tag),(bot,"KANALLAR: "+" ".join(f"{NAME[i]}" for i in range(7)))):
        cv2.rectangle(im,(0,0),(im.shape[1],30),(0,0,0),-1)
        cv2.putText(im,t,(6,21),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1,cv2.LINE_AA)
    sheet=np.vstack([top,np.full((4,w,3),255,np.uint8),bot])
    # küçült (montaj için)
    sc=760/w; sheet=cv2.resize(sheet,(int(w*sc),int(sheet.shape[0]*sc)))
    return sheet

def main():
    idxs=[int(x) for x in sys.argv[1:]] if len(sys.argv)>1 else [10,13,35,59,61,90,5,1]
    for idx in idxs:
        s=one(idx)
        if s is None: print(f"idx{idx} atlandı"); continue
        out=os.path.join(ROOT,"cand",f"_CONF_{idx:03d}.jpg"); cv2.imwrite(out,s); print(f"-> {out}")

if __name__=="__main__": main()
