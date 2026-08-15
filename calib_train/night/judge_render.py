#!/usr/bin/env python3
"""LEVER A: fix'ler (center+robust+marks) AÇIK haldeyken tüm 121 sahayı judge-input olarak render et.
Her görsel: SOL = kamera karesi + REPROJEKTE kalibrasyon sınırı (kenar-renkli) ; SAĞ = 2D radar (fix'li marks).
boundary_judge.js bunları okuyup USABLE/MARGINAL/WRONG verir → fix-sonrası gerçek ürün-bar sayısı.
Çıktı: cand/judge2/{idx}.jpg + scratchpad/autorun/judge_args.json ([[idx,venue,path,'auto'],...]).
"""
import os, sys, json, re, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
HERE=os.path.dirname(os.path.abspath(__file__)); CT=os.path.dirname(HERE); ROOT=os.path.dirname(CT)
sys.path.insert(0, ROOT)
from calib_train import auto_calib as AC
from calib_train import auto_clean2d as AC2
from calib_train.pretty_render import _text, INK, MUTE
from calib_train.night.four_panel import load
L,Wp=34.0,18.0
OUT=os.path.join(CT,"cand","judge2"); os.makedirs(OUT,exist_ok=True)
MAN={r['idx']:r for r in json.load(open(os.path.join(HERE,"probcache_v3","manifest.json")))}
# BGR, four_panel rol-renkleriyle aynı (rubric: yakınkale=camgöbeği, uzakkale=mercan, yakıntaç=yeşil, uzaktaç=mavi)
EDGES=[("goalN",(0.0,None),(232,200,80)),("goalF",(L,None),(90,110,245)),
       ("touchN",(None,0.0),(150,220,120)),("touchF",(None,Wp),(230,170,70))]

def _proj(rec,Xs,Ys,w,h):
    P=np.stack([Xs,Ys],1)
    px=AC.project_metric(P,rec['k1'],rec['k2'],rec['H'],w/2,h/2,w/2)
    return px[np.isfinite(px).all(1)]

def sol_overlay(img,rec,w,h):
    o=img.copy(); N=140
    for name,(xc,yc),col in EDGES:
        if xc is not None: Xs=np.full(N,xc); Ys=np.linspace(0,Wp,N)
        else: Xs=np.linspace(0,L,N); Ys=np.full(N,yc)
        px=_proj(rec,Xs,Ys,w,h)
        if len(px)>=2: cv2.polylines(o,[px.astype(np.int32)],False,col,3,cv2.LINE_AA)
    # orta çizgi (soluk) — kutu/çember çizmiyoruz, judge onları yok sayıyor
    px=_proj(rec,np.full(N,L/2),np.linspace(0,Wp,N),w,h)
    if len(px)>=2: cv2.polylines(o,[px.astype(np.int32)],False,(70,200,240),1,cv2.LINE_AA)
    cv2.rectangle(o,(0,0),(o.shape[1],40),(14,15,17),-1)
    _text(o,(14,12),"reprojekte kalibrasyon sınırı (fix'li)",22,INK)
    lg=[("yakın kale",(232,200,80)),("UZAK kale",(90,110,245)),("yakın taç",(150,220,120)),("uzak taç",(230,170,70))]
    x=14
    for nm,c in lg:
        cv2.circle(o,(x+7,h-20),6,c,-1); _text(o,(x+18,h-28),nm,13,INK,bold=False); x+=int(w/4.3)
    return o

def render(idx):
    e,img,prob,w,h,feet=load(idx)
    rec=AC2.calibrate_frame(prob,w,h,img=img,feet=feet if feet else None)
    if rec and rec.get('ok'):
        sol=sol_overlay(img,rec,w,h)
        mp=AC2.project_feet(np.array([[f[0],f[1]] for f in feet],float),rec,w,h) if feet else np.empty((0,2))
        mp=mp[np.isfinite(mp).all(1)] if len(mp) else mp
        rad,n=AC2.draw_2d(mp,rec.get("camside","SOL"),"2D radar",marks=rec.get("marks"))
        src=rec.get('src','auto')
    else:
        sol=np.full((h,w,3),(28,31,35),np.uint8); _text(sol,(14,h//2),"kalibrasyon YOK",26,(90,90,235))
        rad=np.full((h,w//2,3),(20,22,25),np.uint8); _text(rad,(14,rad.shape[0]//2),"—",26,MUTE); n=0; src="NONE"
    H=560
    r=lambda a:cv2.resize(a,(int(a.shape[1]*H/a.shape[0]),H))
    sol_r,rad_r=r(sol),r(rad)
    montage=np.hstack([sol_r,np.full((H,8,3),(20,22,25),np.uint8),rad_r])
    vn=re.sub(r'(?<=[a-zçğışöü])(?=[A-ZÇĞİŞÖÜ])',' ',e['file'].split('__')[0])
    p=os.path.join(OUT,f"{idx:03d}.jpg"); cv2.imwrite(p,montage,[cv2.IMWRITE_JPEG_QUALITY,88])
    return idx,vn,p,src,(rec.get('ok') if rec else False)

def main():
    idxs=sorted(MAN)
    args=[]
    for i,idx in enumerate(idxs):
        try:
            idx,vn,p,src,ok=render(idx)
            args.append([idx,vn,p,"auto"])
            if i%15==0: print(f"[{i+1}/{len(idxs)}] idx{idx} {vn[:22]} src={src} ok={ok}",flush=True)
        except Exception as ex:
            print(f"idx{idx} FAIL {ex}",flush=True)
    ap=os.path.join(ROOT,"scratchpad","autorun","judge_args.json")
    json.dump(args,open(ap,"w"),ensure_ascii=False)
    print(f"\nRENDER BİTTİ: {len(args)}/{len(idxs)} görsel -> {OUT}\nargs -> {ap}",flush=True)

if __name__=="__main__": main()
