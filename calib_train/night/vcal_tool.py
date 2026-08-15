"""Vision-calib ajan tool'u. grid <idx> -> gridli saha oku. cal <idx> 'x,y x,y x,y x,y' [L W] -> overlay."""
import os,sys,json,numpy as np,cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calib_train.vision_calib import calibrate_from_corners, overlay
from calib_train.auto_clean2d import sanity
CACHE="calib_train/night/probcache"; man={e['idx']:e for e in json.load(open(f"{CACHE}/manifest.json"))}
OUT="calib_train/cand/vcal"; os.makedirs(OUT,exist_ok=True)
def gridrender(idx):
    e=man[idx]; img=cv2.imread(f"calib_train/cand_big/{e['file']}"); h,w=img.shape[:2]; o=img.copy()
    for x in range(0,w,100):
        cv2.line(o,(x,0),(x,h),(0,255,255),1); cv2.putText(o,str(x),(x+2,18),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
    for y in range(0,h,100):
        cv2.line(o,(0,y),(w,y),(0,255,255),1); cv2.putText(o,str(y),(2,y+16),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,255),1)
    p=f"{OUT}/_grid_{idx}.jpg"; cv2.imwrite(p,o); print(f"{e['file']} {w}x{h} -> {p}")
def cal(idx,corners,L,W):
    e=man[idx]; img=cv2.imread(f"calib_train/cand_big/{e['file']}"); h,w=img.shape[:2]
    rec=calibrate_from_corners(corners,L,W,w,h)
    if not rec.get("ok"): print("CALIB FAIL:",rec.get("reason")); return
    ok,why=sanity(rec,w,h,img=img)
    p=f"{OUT}/_over_{idx}.jpg"; cv2.imwrite(p,overlay(img,rec,L,W))
    # KÖŞELERİ PER-DOSYA KAYDET (concurrency-safe; merge_labels ile birleştirilir)
    PD="calib_train/corner_labels_parts"; os.makedirs(PD,exist_ok=True)
    json.dump({"file":e["file"],"corners":corners,"L":L,"W":W,"w":w,"h":h,"sanity":bool(ok)},
              open(f"{PD}/{idx}.json","w"),ensure_ascii=False)
    print(f"sanity: {'PASS' if ok else 'FAIL('+why+')'} -> {p} | KAYDEDİLDİ parts/{idx}.json")
if __name__=="__main__":
    if sys.argv[1]=="grid": gridrender(int(sys.argv[2]))
    else:
        idx=int(sys.argv[2]); cs=[[float(a) for a in c.split(",")] for c in sys.argv[3].split()]
        L=float(sys.argv[4]) if len(sys.argv)>4 else 40.0; W=float(sys.argv[5]) if len(sys.argv)>5 else 20.0
        cal(idx,cs,L,W)
