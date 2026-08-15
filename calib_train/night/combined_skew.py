"""Birleşik warp-doğruluk skoru — GT'siz, bilinen geometriyle:
  roundness : çember piksel -> metrik -> elips eksen-oranı (1=yuvarlak, shear'ı yakalar)
  center_off: çember merkezi -> metrik, |. - (L/2,W/2)| (kalibrasyon konum-hatası, Alperen fikri)
  circle_ok : çember güvenilir tespit edildi mi (yoksa şerit gerekir - universal fallback TODO)
Çıktı: combined_skew_results.json + özet dağılım."""
import os; os.environ['CUDA_VISIBLE_DEVICES']=''
import json,cv2,numpy as np,sys,time
sys.path.insert(0,'.')
from calib_train import auto_clean2d as AC2
CT="calib_train"; HERE=f"{CT}/night"; L,W=34.0,18.0
MAN=[r for r in json.load(open(f"{HERE}/probcache_v3/manifest.json"))]
def measure(idx):
    d=np.load(f"{HERE}/probcache_v3/{idx:03d}.npz"); prob=d['prob'].astype(np.float32); w,h=int(d['w']),int(d['h'])
    rec=AC2.calibrate_frame(prob,w,h)
    if not rec.get('ok'): return {'idx':idx,'solve':False}
    m=(cv2.resize(prob[6],(w,h))>0.4).astype(np.uint8)
    r={'idx':idx,'solve':True,'circle_ok':False,'roundness':None,'center_off':None}
    nn,lab,st,cent=cv2.connectedComponentsWithStats(m,8)
    if nn>=2:
        big=1+int(np.argmax(st[1:,cv2.CC_STAT_AREA])); ys,xs=np.where(lab==big)
        if len(xs)>=12:
            pts=np.stack([xs,ys],1).astype(float)
            mp=AC2.project_feet(np.column_stack([pts,np.ones(len(pts))]),rec,w,h)
            mp=mp[np.isfinite(mp).all(1)]
            if len(mp)>=12:
                c=mp.mean(0); X=mp-c; cov=(X.T@X)/len(X); ev=np.clip(np.linalg.eigvalsh(cov),1e-9,None)
                R=float(np.sqrt(ev.max())); ratio=float(np.sqrt(ev.min()/ev.max()))
                off=float(np.hypot(c[0]-L/2,c[1]-W/2))
                if 0.4<R<7:
                    r.update(circle_ok=True,roundness=round(ratio,3),center_off=round(off,2),R=round(R,2))
    return r
def main():
    t=time.time(); rows=[]
    for i,e in enumerate(MAN):
        rows.append(measure(e['idx']))
        if i%15==0: print(f"[{i+1}/{len(MAN)}] idx{e['idx']} ({time.time()-t:.0f}s)",flush=True)
    json.dump(rows,open(f"{HERE}/combined_skew_results.json","w"))
    solved=[r for r in rows if r['solve']]; circ=[r for r in solved if r['circle_ok']]
    good=[r for r in circ if r['roundness']>=0.7 and r['center_off']<=3]
    skew=[r for r in circ if r['roundness']<0.7 or r['center_off']>3]
    nocirc=[r for r in solved if not r['circle_ok']]
    print("="*60)
    print(f"çözen {len(solved)} | çember-ölçülebilir {len(circ)} | çembersiz/güvenilmez {len(nocirc)} (şerit gerekir)")
    print(f"çember-ölçülenlerden: OTURUYOR {len(good)} | SKEW {len(skew)}")
    if skew:
        skew.sort(key=lambda r:r['roundness'])
        print("en yamuk 12:", [(r['idx'],r['roundness'],r['center_off']) for r in skew[:12]])
    print(f"DÜRÜST: warp-doğruluğu çember-ölçülebilen {len(circ)} sahada bilinir; {len(nocirc)} çembersiz saha ölçüm-dışı (şerit-sinyali sıradaki)")
if __name__=="__main__": main()
