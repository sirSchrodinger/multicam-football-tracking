#!/usr/bin/env python3
"""4-PANEL boru-hattı görseli (Alperen isteği, gözle-anla):
  1) HAM + oyuncu AYAK-ALTI kırmızı tespit   2) ÇİZGİ tespit (seg2 rol-kanalları)
  3) WARP (kuş-bakışı düzleştirme)            4) 2D şematik saha + oyuncu noktaları
Şık tema, Inter etiket. Kullanım: python four_panel.py 15 90 35 ...
"""
import os, sys, json, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
HERE=os.path.dirname(os.path.abspath(__file__)); CT=os.path.dirname(HERE); ROOT=os.path.dirname(CT)
sys.path.insert(0, ROOT)
from calib_train import auto_calib as AC
from calib_train import auto_clean2d as AC2
from calib_train.pretty_render import _text, INK, MUTE, ROLE_COL as PR
L,Wp=34.0,18.0
MAN={r['idx']:r for r in json.load(open(os.path.join(HERE,"probcache_v3","manifest.json")))}
CACHE=os.path.join(HERE,"probcache_v3"); FEET=os.path.join(HERE,"feetcache")
# seg2 rol renkleri (BGR) — çizgi paneli
RC=[(232,200,80),(90,110,245),(150,220,120),(230,170,70),(70,200,240),(210,120,220),(160,150,245)]
RN=["yakın kale","UZAK kale","yakın taç","uzak taç","orta çizgi","ceza sah.","yuvarlak"]

def load(idx):
    e=MAN[idx]; img=cv2.imread(os.path.join(CT,"cand_big",e['file']))
    d=np.load(os.path.join(CACHE,f"{idx:03d}.npz")); prob=d['prob'].astype(np.float32); w,h=int(d['w']),int(d['h'])
    img=cv2.resize(img,(w,h))
    fp=os.path.join(FEET,f"{idx:03d}.json"); feet=json.load(open(fp))['feet'] if os.path.exists(fp) else []
    return e,img,prob,w,h,feet

def p_ham(img,feet):
    o=img.copy(); cv2.rectangle(o,(0,0),(o.shape[1],40),(14,15,17),-1)
    for f in feet:
        x,y=int(f[0]),int(f[1]); cv2.circle(o,(x,y),7,(40,40,235),-1); cv2.circle(o,(x,y),7,(255,255,255),1)
    _text(o,(14,12),f"1 · HAM + oyuncu ayak-tespiti ({len(feet)})",22,INK)
    return o

def p_lines(img,prob,w,h):
    o=img.copy()
    for ci in range(7):
        m=cv2.resize(prob[ci],(w,h))>0.4
        o[m]=(0.35*o[m]+0.65*np.array(RC[ci])).astype(np.uint8)
    cv2.rectangle(o,(0,0),(o.shape[1],40),(14,15,17),-1)
    _text(o,(14,12),"2 · ÇİZGİ tespiti (modelin gördüğü)",22,INK)
    x=14
    for i in range(7):
        cv2.circle(o,(x+7,h-22),6,RC[i],-1); _text(o,(x+18,h-30),RN[i],13,INK,bold=False); x+=int(w/7.2)
    return o

def p_warp(img,rec,w,h):
    # TAM WARP (fade/transparanlık YOK) — gerçek remap; uzak-yarı fiziksel olarak düşük-çözünürlük.
    S=18;pad=14; Wc=int(L*S+2*pad);Hc=int(Wp*S+2*pad)
    gx,gy=np.meshgrid(np.arange(Wc),np.arange(Hc)); Xm=(gx-pad)/S;Ym=Wp-(gy-pad)/S
    pix=AC.project_metric(np.stack([Xm.ravel(),Ym.ravel()],1),rec['k1'],rec['k2'],rec['H'],w/2,h/2,w/2)
    wp=cv2.remap(img,pix[:,0].reshape(Hc,Wc).astype(np.float32),pix[:,1].reshape(Hc,Wc).astype(np.float32),
                 cv2.INTER_LINEAR,borderValue=(26,28,31))
    def P2(X,Y): return int(pad+X*S),int(pad+(Wp-Y)*S)
    wht=(232,236,239); mk=rec.get('marks') or {}
    cv2.rectangle(wp,P2(0,0),P2(L,Wp),wht,2,cv2.LINE_AA); cv2.line(wp,P2(L/2,0),P2(L/2,Wp),wht,1,cv2.LINE_AA)
    R=mk.get('R') or 3.0; cv2.circle(wp,P2(L/2,Wp/2),int(R*S),wht,1,cv2.LINE_AA)
    bn=mk.get('box_n') or (5.0,5.0); bf=mk.get('box_f') or (5.0,5.0)
    for g0,(dep,hw) in ((0.0,bn),(L,bf)):
        bx=dep if g0==0 else L-dep
        cv2.rectangle(wp,P2(min(g0,bx),Wp/2-hw),P2(max(g0,bx),Wp/2+hw),wht,1,cv2.LINE_AA)
    _text(wp,(14,10),"3 · WARP (kuş-bakışı, tam)",20,INK)
    _text(wp,(14,Hc-24),"uzak yarı doğal düşük-çözünürlük (kamera pikseli az) · oyuncular 3B → çizgilenir",11,MUTE,bold=False)
    return wp

def four(idx):
    e,img,prob,w,h,feet=load(idx)
    rec=AC2.calibrate_frame(prob,w,h,img=img,feet=feet if feet else None)
    p1=p_ham(img,feet); p2=p_lines(img,prob,w,h)
    if rec and rec.get('ok'):
        p3=p_warp(img,rec,w,h)
        mp=AC2.project_feet(np.array([[f[0],f[1]] for f in feet],float),rec,w,h) if feet else np.empty((0,2))
        mp=mp[np.isfinite(mp).all(1)] if len(mp) else mp
        p4,n=AC2.draw_2d(mp,rec.get("camside","SOL"),"4   2D radar",marks=rec.get("marks"))
        _text(p4,(8,p4.shape[0]-22),f"{n} oyuncu saha-icine projekte",13,MUTE,bold=False)
    else:
        p3=np.full((h,w,3),(28,31,35),np.uint8); _text(p3,(14,h//2),"3 · WARP: kalibrasyon yok",22,(90,90,235))
        p4=p3.copy()
    # 2x2 grid, ortak kutu, koyu zemin
    Ht,Wt=560,960; BG=(20,22,25)
    def box(a,t=None):
        s=min(Wt/a.shape[1],Ht/a.shape[0]); r=cv2.resize(a,(int(a.shape[1]*s),int(a.shape[0]*s)))
        c=np.full((Ht,Wt,3),BG,np.uint8); y0=(Ht-r.shape[0])//2; x0=(Wt-r.shape[1])//2
        c[y0:y0+r.shape[0],x0:x0+r.shape[1]]=r; return c
    g=8
    top=np.hstack([box(p1),np.full((Ht,g,3),BG,np.uint8),box(p2)])
    bot=np.hstack([box(p3),np.full((Ht,g,3),BG,np.uint8),box(p4)])
    TH=46; body=np.vstack([top,np.full((g,top.shape[1],3),BG,np.uint8),bot])
    canvas=np.full((TH+body.shape[0],body.shape[1],3),BG,np.uint8); canvas[TH:]=body
    import re; vn=re.sub(r'(?<=[a-zçğışöü])(?=[A-ZÇĞİŞÖÜ])',' ',e['file'].split('__')[0])
    _text(canvas,(16,10),f"{vn} — boru hattı: ham → çizgi → warp → 2D",24,INK)
    src=rec.get('src','?') if rec and rec.get('ok') else 'YOK'
    _text(canvas,(body.shape[1]-12,12),f"otomatik · kalib={src}",14,MUTE,bold=False,anchor="ra")
    return canvas

if __name__=="__main__":
    idxs=[int(x) for x in sys.argv[1:]] if len(sys.argv)>1 else [15,90,17,35]
    for idx in idxs:
        out=os.path.join(CT,"cand",f"_4PANEL_{idx:03d}.jpg"); cv2.imwrite(out,four(idx),[cv2.IMWRITE_JPEG_QUALITY,90])
        print(f"idx{idx} -> {out}")
