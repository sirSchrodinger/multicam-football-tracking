"""Auto-calib'i çalıştır -> tüm landmark'ları world->image projekte et -> ön-seed JSON.
clicker bunu okuyup noktaları ÖNCEDEN yerleştirir (insan sadece yanlışları sürükler). Ölçeklenir UX."""
import os,sys,json,numpy as np,cv2
sys.path.insert(0,'.'); os.environ['CUDA_VISIBLE_DEVICES']=''
from calib_train import auto_clean2d as AC2
from calib_train import auto_calib as AC
from calib.landmarks import landmarks
idx=int(sys.argv[1]); L,W=34.0,18.0
MAN={r['idx']:r for r in json.load(open('calib_train/night/probcache_v3/manifest.json'))}
e=MAN[idx]; frame=f"calib_train/cand_big/{e['file']}"
d=np.load(f"calib_train/night/probcache_v3/{idx:03d}.npz");prob=d['prob'].astype(np.float32);w,h=int(d['w']),int(d['h'])
img=cv2.imread(frame); H_,W_=img.shape[:2]
rec=AC2.calibrate_frame(cv2.resize(prob if prob.ndim==3 else prob,(w,h)) if False else prob,w,h)
pts={}
if rec.get('ok'):
    LM=landmarks(L,W)  # base+box
    world=np.array([xy for (_,_,xy,_,_) in LM],float)
    # w,h prob-res; frame gerçek res -> ölçekle
    px=AC.project_metric(world,rec['k1'],rec['k2'],rec['H'],w/2,h/2,w/2)
    sx,sy=W_/w, H_/h
    for (lid,_,_,_,off),p in zip(LM,px):
        if np.all(np.isfinite(p)) and 0<=p[0]<w and 0<=p[1]<h:
            pts[lid]=[round(float(p[0]*sx),1),round(float(p[1]*sy),1)]
    print(f"idx{idx}: auto ön-tahmin {len(pts)} landmark (calib src={rec.get('src')})")
else:
    print(f"idx{idx}: auto çözemedi -> boş seed (sıfırdan tıkla)")
out=f"calib/preseed_{idx}.json"; json.dump({'points':pts},open(out,'w'))
# ham kareyi kaydet + clicker üret
cv2.imwrite(f"calib/venue{idx}_raw.jpg",img)
print(f"-> {out}, calib/venue{idx}_raw.jpg")
