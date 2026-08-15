#!/usr/bin/env python3
"""Agent-vision kalibrasyon için ETİKETLİ IZGARA overlay. Ham kareye %0-100 ızgara (her %5'te
ince, %10'da kalın+etiket) çizer → agent saha KÖŞELERİNİ (x%,y%) olarak okur → piksele çevrilir →
vision_calib.calibrate_from_corners. Pixel-tık yerine yüzde-okuma daha güvenilir (±%2).
Kullanım: python vision_grid.py 13 35 59  (idx'ler) → cand/vgrid/NNN.jpg + pct2px(idx,xpct,ypct)
"""
import os, sys, json, numpy as np, cv2
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from PIL import Image, ImageDraw, ImageFont
MAN={r['idx']:r for r in json.load(open(os.path.join(HERE,"night","probcache_v3","manifest.json")))}
_F="/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"
if not os.path.exists(_F): _F="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def draw_grid(img, step=10):
    h,w=img.shape[:2]; o=img.copy()
    # ince %5 ızgara
    for p in range(0,101,5):
        x=int(w*p/100); y=int(h*p/100)
        cv2.line(o,(x,0),(x,h),(90,90,90),1,cv2.LINE_AA)
        cv2.line(o,(0,y),(w,y),(90,90,90),1,cv2.LINE_AA)
    # kalın %10 ızgara + numaralar
    pim=Image.fromarray(cv2.cvtColor(o,cv2.COLOR_BGR2RGB)); d=ImageDraw.Draw(pim); fnt=ImageFont.truetype(_F,22)
    for p in range(0,101,step):
        x=int(w*p/100); y=int(h*p/100)
        d.line([(x,0),(x,h)],fill=(255,235,60),width=1)
        d.line([(0,y),(w,y)],fill=(255,235,60),width=1)
        if p>0 and p<100:
            d.text((x+2,2),f"{p}",font=fnt,fill=(255,235,60)); d.text((x+2,h-26),f"{p}",font=fnt,fill=(255,235,60))
            d.text((2,y+1),f"{p}",font=fnt,fill=(120,220,255)); d.text((w-30,y+1),f"{p}",font=fnt,fill=(120,220,255))
    d.text((6,h//2-14),"x% üst/alt sarı · y% sol/sağ mavi",font=ImageFont.truetype(_F,18),fill=(255,255,255))
    return cv2.cvtColor(np.array(pim),cv2.COLOR_RGB2BGR)

def render_idx(idx, outdir=None):
    outdir=outdir or os.path.join(HERE,"cand","vgrid"); os.makedirs(outdir,exist_ok=True)
    img=cv2.imread(os.path.join(HERE,"cand_big",MAN[idx]['file']))
    if img is None: return None
    g=draw_grid(img)
    out=os.path.join(outdir,f"{idx:03d}.jpg"); cv2.imwrite(out,g,[cv2.IMWRITE_JPEG_QUALITY,90])
    return out, img.shape[1], img.shape[0]

def pct2px(xpct, ypct, w, h):
    return [w*xpct/100.0, h*ypct/100.0]

if __name__=="__main__":
    idxs=[int(x) for x in sys.argv[1:]] if len(sys.argv)>1 else [13,35,59,61]
    for idx in idxs:
        r=render_idx(idx)
        if r: print(f"idx{idx} -> {r[0]} ({r[1]}x{r[2]})")
