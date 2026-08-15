#!/usr/bin/env python3
"""Vision-calib RECOVERY tek-komut: idx + agent-okunan 4 köşe (yüzde x,y) -> ortak-lens kalibrasyon
(L=34,W=18 pipeline-uyumlu) -> ŞIK overlay + 2D radar. Otomatik seg2'nin yamuk verdiği sahayı
agent-vision ile kurtarır. corners: [NL,FL,NR,FR] yüzde (0-100), NL=near-sol...

CLI: python vision_recover.py IDX xNL yNL xFL yFL xNR yNR xFR yFR  (yüzde)
API: recover(idx, corners_pct) -> (rec, out_path)
"""
import os, sys, json, numpy as np, cv2
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from calib_train import auto_calib as AC
from calib_train import vision_calib as VC
from calib_train.pretty_render import render_overlay, venue_name
L,Wp=34.0,18.0
MAN={r['idx']:r for r in json.load(open(os.path.join(HERE,"night","probcache_v3","manifest.json")))}

def recover(idx, corners_pct, outdir=None):
    outdir=outdir or os.path.join(HERE,"cand","recover"); os.makedirs(outdir,exist_ok=True)
    img=cv2.imread(os.path.join(HERE,"cand_big",MAN[idx]['file']))
    if img is None: return None,None
    h,w=img.shape[:2]
    corners_px=[[c[0]/100.0*w, c[1]/100.0*h] for c in corners_pct]
    rec=VC.calibrate_from_corners(corners_px, L, Wp, w, h)
    if not rec.get("ok"): return rec,None
    rec['fit']=rec.get('fit',0.0)
    # camside: near-sol köşe (NL) ve near-sağ (NR) x'ine göre; vision_calib SOL varsayıyor
    can=render_overlay(img,rec,venue=venue_name(MAN[idx]['file'])+" (vision-recover)",idx=idx,judge=True)
    out=os.path.join(outdir,f"{idx:03d}.jpg"); cv2.imwrite(out,can,[cv2.IMWRITE_JPEG_QUALITY,92])
    return rec,out

if __name__=="__main__":
    idx=int(sys.argv[1]); vals=[float(x) for x in sys.argv[2:10]]
    corners=[[vals[0],vals[1]],[vals[2],vals[3]],[vals[4],vals[5]],[vals[6],vals[7]]]
    rec,out=recover(idx,corners)
    print(f"idx{idx} ok={rec.get('ok')} -> {out}")
