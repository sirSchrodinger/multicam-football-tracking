#!/usr/bin/env python3
"""ÜRÜN demo: Çankaya gerçek-maç konsolide oyuncu-konumundan ŞIK maç-dashboard.
Sol: 2D saha + oyuncular (renk=oyuncu, iz, GK amber, predicted=soluk).
Sağ: per-oyuncu istatistik (koşu mesafesi, tepe hız) — asıl satılan çıktı.
Girdi: stats_out/active_game/player_state_continuous.parquet (player_id,frame,t_sec,x,y,status,conf).
"""
import os, sys, numpy as np, cv2, pandas as pd
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))
from calib_train.pretty_render import _text, BG, PANEL, INK, MUTE
from PIL import Image, ImageDraw, ImageFont
L,Wp=34.0,18.0
_F="/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"
if not os.path.exists(_F): _F="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
REPO=os.path.dirname(ROOT)  # halisaha-stats (stats_out burada)
STATE=os.path.join(REPO,"stats_out","active_game","player_state_continuous.parquet")
# oyuncu renkleri (13) — ayırt-edici, hoş
np.random.seed(7)
PCOL=[(94,169,247),(120,200,120),(240,150,90),(200,130,230),(90,210,230),(140,160,250),
      (230,180,70),(120,220,180),(240,120,140),(170,200,110),(110,180,240),(220,160,200),(150,230,140)]

def per_player_stats(df, fps=25.0):
    """observed noktalardan savgol-mesafe + tepe-hız (interp/pred hariç)."""
    from scipy.signal import savgol_filter
    out={}
    for pid,g in df.groupby('player_id'):
        g=g[g.status=='observed'].sort_values('frame')
        if len(g)<20: out[pid]=dict(dist=0.0,vmax=0.0,pts=0); continue
        x=g.x.values; y=g.y.values; t=g.t_sec.values
        w=min(15,len(x)//2*2+1);
        if w>=5: x=savgol_filter(x,w,2); y=savgol_filter(y,w,2)
        dt=np.diff(t); dd=np.hypot(np.diff(x),np.diff(y))
        v=np.divide(dd,dt,out=np.zeros_like(dd),where=dt>0)*3.6  # km/h
        ok=(dt>0)&(dt<1.0)&(v<40)  # fiziksel-dışı sıçrama hariç
        out[pid]=dict(dist=float(dd[ok].sum()), vmax=float(np.percentile(v[ok],98) if ok.sum() else 0), pts=int(len(g)))
    return out

def pick_frame(df):
    """en çok observed + yayılmış oyuncu olan kareyi seç."""
    obs=df[df.status=='observed']
    c=obs.groupby('frame').agg(n=('player_id','nunique'), sx=('x','std'), sy=('y','std'))
    c['score']=c.n + (c.sx.fillna(0)+c.sy.fillna(0))/10
    return int(c.score.idxmax())

def draw_pitch_players(df, frame, gk_pids, W=1000):
    S=(W-80)/L; H=int(Wp*S+80); pad=40
    im=np.full((H,W,3),(30,46,32),np.uint8)
    for i in range(int(L/2)+1):
        x0=int(pad+i*2*S); x1=min(int(pad+(i+1)*2*S),W-pad); im[pad:H-pad,x0:x1]=(40,66,44) if i%2 else (46,76,50)
    def P(x,y): return int(pad+np.clip(x,0,L)*S), int(pad+(Wp-np.clip(y,0,Wp))*S)
    wht=(220,228,222)
    cv2.rectangle(im,P(0,0),P(L,Wp),wht,2); cv2.line(im,P(L/2,0),P(L/2,Wp),wht,1)
    cv2.circle(im,P(L/2,Wp/2),int(3*S),wht,1)
    for gx in (0.0,L):
        bx=5 if gx==0 else L-5
        cv2.rectangle(im,P(min(gx,bx),Wp/2-5),P(max(gx,bx),Wp/2+5),wht,1)
    # izler (son ~1.5s observed)
    tr=df[(df.frame>frame-38)&(df.frame<=frame)]
    for pid,g in tr.groupby('player_id'):
        col=PCOL[int(pid)%len(PCOL)]; g=g.sort_values('frame')
        pts=[P(r.x,r.y) for r in g.itertuples() if r.status=='observed']
        for a,b in zip(pts,pts[1:]): cv2.line(im,a,b,col,2,cv2.LINE_AA)
    # oyuncular
    now=df[df.frame==frame]
    n=0
    for r in now.itertuples():
        col=PCOL[int(r.player_id)%len(PCOL)]; px,py=P(r.x,r.y)
        solid=(r.status=='observed')
        rad=11 if int(r.player_id) in gk_pids else 9
        if solid: cv2.circle(im,(px,py),rad,col,-1); cv2.circle(im,(px,py),rad,(245,245,245),2)
        else: cv2.circle(im,(px,py),rad,col,1)  # predicted/interp = içi boş
        if int(r.player_id) in gk_pids: _text(im,(px-6,py-8),"K",13,(20,20,20))
        n+=1
    return im,n

def main():
    df=pd.read_parquet(STATE)
    fps=df.frame.nunique()/(df.t_sec.max()-df.t_sec.min())
    stats=per_player_stats(df,fps)
    # GK = en çok kale-ucunda kalan 2 pid (x<3 veya x>L-3 baskın)
    gk=[]
    for pid,g in df.groupby('player_id'):
        near=(g.x<5).mean(); far=(g.x>L-5).mean()
        if max(near,far)>0.42: gk.append(int(pid))
    frame=pick_frame(df)
    pitch,n=draw_pitch_players(df,frame,set(gk),W=1000)
    PH=pitch.shape[0]
    # sağ panel: istatistik tablosu
    RW=520; panel=np.full((PH,RW,3),PANEL,np.uint8)
    _text(panel,(24,20),"Oyuncu istatistikleri",22,INK)
    _text(panel,(24,50),"koşu mesafesi · tepe hız (bu segment)",13,MUTE,bold=False)
    y=90; rows=sorted(stats.items(),key=lambda kv:-kv[1]['dist'])
    _text(panel,(24,y),"oyuncu",13,MUTE,bold=False); _text(panel,(300,y),"mesafe",13,MUTE,bold=False,anchor="ra"); _text(panel,(470,y),"tepe hız",13,MUTE,bold=False,anchor="ra"); y+=26
    for pid,st in rows:
        col=PCOL[int(pid)%len(PCOL)]; cv2.circle(panel,(30,y+8),7,col,-1)
        lab=f"oyuncu {pid}"+("  (kaleci)" if int(pid) in gk else "")
        _text(panel,(46,y),lab,15,INK,bold=False)
        _text(panel,(300,y+1),f"{st['dist']:.0f} m",15,INK,anchor="ra")
        _text(panel,(470,y+1),f"{st['vmax']:.1f} km/s",14,MUTE,bold=False,anchor="ra")
        y+=30
    y+=8
    _text(panel,(24,y),"* mesafe: yalnız gözlenen kareler, savgol-düzeltilmiş, fiziksel-dışı sıçrama hariç",11,MUTE,bold=False); y+=18
    _text(panel,(24,y),"* ölçek ±%13 (tek-görüntü); on-site tek ölçümle ±%2'ye iner",11,MUTE,bold=False)
    # birleştir
    TH=52; body=np.hstack([pitch,np.full((PH,10,3),BG,np.uint8),panel])
    BW=body.shape[1]; canvas=np.full((TH+PH,BW+24,3),BG,np.uint8); canvas[TH:,12:12+BW]=body
    _text(canvas,(16,14),"Çankaya Halı Saha — canlı 2D",24,INK)
    _text(canvas,(BW+8,16),f"{n} oyuncu · t={df[df.frame==frame].t_sec.iloc[0]:.0f}s · gerçek maç",14,MUTE,bold=False,anchor="ra")
    out=os.path.join(ROOT,"cand","_PRODUCT_DEMO.jpg"); cv2.imwrite(out,canvas,[cv2.IMWRITE_JPEG_QUALITY,92])
    print(f"-> {out}  ({n} oyuncu, frame {frame}, GK={gk})")

if __name__=="__main__": main()
