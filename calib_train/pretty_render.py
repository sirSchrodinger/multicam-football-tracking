#!/usr/bin/env python3
"""ŞIK render (Alperen: çirkin-sarı YOK, tema+font). Ham kare + reprojekte saha (glow'lu ince
çizgi, Inter tipografi) | 2D radar. Hem agent-görsel-yargı hem galeri-deliverable için.
Yeniden kullanılabilir: from calib_train.pretty_render import render_overlay
"""
import os, sys, json, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from calib_train import auto_calib as AC
from PIL import Image, ImageDraw, ImageFont
L,Wp=AC.L,18.0

# --- tema ---
BG=(20,22,25)            # charcoal (BGR)
PANEL=(28,31,35)
INK=(238,240,243)        # near-white
MUTE=(150,156,164)
# rol renkleri (BGR) — sınırlar ayırt-edici, far-goal vurgulu
ROLE_COL={'goalN':(232,200,80),'goalF':(90,110,245),'touchN':(150,220,120),
          'touchF':(230,170,70),'center':(70,200,240),'box':(210,120,220),'circle':(160,150,245)}
ROLE_TR={'goalN':'yakın kale','goalF':'UZAK kale','touchN':'yakın taç','touchF':'uzak taç',
         'center':'orta çizgi','box':'ceza sahası','circle':'orta yuvarlak'}
_FONT="/usr/share/fonts/opentype/inter/Inter-SemiBold.otf"
_FONTR="/usr/share/fonts/opentype/inter/Inter-Regular.otf"
if not os.path.exists(_FONT): _FONT=_FONTR="/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def _font(sz,bold=True):
    try: return ImageFont.truetype(_FONT if bold else _FONTR, sz)
    except Exception: return ImageFont.load_default()

def _text(img,xy,s,sz=22,col=INK,bold=True,anchor="la"):
    """BGR numpy'ye PIL ile Inter yazı."""
    pim=Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)); d=ImageDraw.Draw(pim)
    d.text(xy,s,font=_font(sz,bold),fill=(col[2],col[1],col[0]),anchor=anchor)
    img[:]=cv2.cvtColor(np.array(pim),cv2.COLOR_RGB2BGR)

def _glowline(img,pix,closed,col,th):
    fin=np.isfinite(pix).all(1)
    if fin.sum()<2: return
    P=[pix[fin].astype(np.int32)]
    cv2.polylines(img,P,closed,(12,14,16),th+4,cv2.LINE_AA)   # koyu glow
    cv2.polylines(img,P,closed,col,th,cv2.LINE_AA)

def draw_field(img,rec,alpha=0.9):
    k1,k2,H=rec['k1'],rec['k2'],rec['H']; h,w=img.shape[:2]; cx,cy,s=w/2,h/2,w/2
    o=img.copy()
    def ln(a,b,role,th,N=90,closed=False):
        pts=np.linspace(a,b,N) if not closed else a
        _glowline(o,AC.project_metric(pts,k1,k2,H,cx,cy,s),closed,ROLE_COL[role],th)
    ln([0,0],[0,Wp],'goalN',3); ln([L,0],[L,Wp],'goalF',4)
    ln([0,0],[L,0],'touchN',3); ln([0,Wp],[L,Wp],'touchF',3)
    ln([L/2,0],[L/2,Wp],'center',2)
    for gx in (0.0,L):
        bx=5 if gx==0 else L-5
        ln([gx,Wp/2-5],[bx,Wp/2-5],'box',2); ln([bx,Wp/2-5],[bx,Wp/2+5],'box',2); ln([bx,Wp/2+5],[gx,Wp/2+5],'box',2)
    th=np.linspace(0,2*np.pi,64); circ=np.stack([L/2+3*np.cos(th),Wp/2+3*np.sin(th)],1)
    ln(circ,None,'circle',2,closed=True)
    return cv2.addWeighted(o,alpha,img,1-alpha,0)

def draw_radar(rec,feet=None,W=560):
    S=(W-60)/L; H=int(Wp*S+60); pad=30
    im=np.full((H,W,3),PANEL,np.uint8)
    # çizgili çim
    for i in range(int(L/2)+1):
        x0=int(pad+i*2*S); x1=min(int(pad+(i+1)*2*S),W-pad)
        im[pad:H-pad,x0:x1]=(46,74,48) if i%2 else (52,84,54)
    cs=rec.get('camside','SAG')
    def P(x,y):
        if cs=="SAG": x=L-x; y=Wp-y
        return int(pad+x*S),int(pad+(Wp-y)*S)
    wht=(226,230,233)
    cv2.rectangle(im,P(0,0),P(L,Wp),wht,2)
    cv2.line(im,P(L/2,0),P(L/2,Wp),wht,1)
    cv2.circle(im,P(L/2,Wp/2),int(3*S),wht,1)
    for gx in (0.0,L):
        bx=5 if gx==0 else L-5
        cv2.rectangle(im,P(min(gx,bx),Wp/2-5),P(max(gx,bx),Wp/2+5),wht,1)
        cv2.line(im,P(gx,Wp/2-1.5),P(gx,Wp/2+1.5),ROLE_COL['goalF'] if gx==L else ROLE_COL['goalN'],3)
    n=0
    if feet is not None and len(feet):
        for (x,y) in feet:
            if not(np.isfinite(x) and np.isfinite(y)) or not(-1<x<L+1 and -1<y<Wp+1): continue
            px,py=P(np.clip(x,0,L),np.clip(y,0,Wp))
            cv2.circle(im,(px,py),7,(60,60,235),-1); cv2.circle(im,(px,py),7,(240,240,240),1); n+=1
    cam=P(0,0); cv2.circle(im,cam,7,(70,200,245),-1)
    lx=min(cam[0]+9,W-64); ly=max(cam[1]-8,14)
    _text(im,(lx,ly),"kamera",13,MUTE)
    return im,n

def _legend(img,x,y):
    for role in ['goalN','goalF','touchN','touchF','center','box','circle']:
        c=ROLE_COL[role]; cv2.circle(img,(x+7,y+8),6,c,-1)
        _text(img,(x+20,y),ROLE_TR[role],14,INK,bold=False); x+=140
        if x>img.shape[1]-140: x=x; # tek satır sığmazsa taşar; panel geniş

def render_overlay(img,rec,venue="",idx=None,metrics="",feet=None,judge=False):
    """LEFT ham+overlay | RIGHT radar. judge=True ise verdict-text YOK (agent yargılasın)."""
    h,w=img.shape[:2]; over=draw_field(img,rec)
    LW=980; sc=LW/w; left=cv2.resize(over,(LW,int(h*sc)))
    LH=left.shape[0]
    radar,nrad=draw_radar(rec,feet=feet,W=int(LW*0.52))
    # radar'ı doğal-oranında sağ-kolonda dikey ortala (alt-yarı boş kalmasın)
    rcol=np.full((LH,radar.shape[1],3),BG,np.uint8)
    y0=max(0,(LH-radar.shape[0])//2); rcol[y0:y0+min(radar.shape[0],LH)]=radar[:LH-y0]
    body=np.hstack([left,np.full((LH,10,3),BG,np.uint8),rcol])
    BW=body.shape[1]; TH=52; FH=44
    canvas=np.full((TH+LH+FH,BW+24,3),BG,np.uint8)
    canvas[TH:TH+LH,12:12+BW]=body
    _text(canvas,(16,14),venue or "saha",24,INK)
    tag=f"otonom kalibrasyon" if idx is None else f"otonom kalibrasyon · idx {idx}"
    _text(canvas,(BW+8,16),tag,15,MUTE,bold=False,anchor="ra")
    _legend(canvas,16,TH+LH+13)
    if metrics and not judge:
        _text(canvas,(BW+8,TH+LH+15),metrics,14,MUTE,bold=False,anchor="ra")
    return canvas

# --- standalone: census combo'dan tek idx render ---
def _load_ctx():
    PC=os.path.join(HERE,"night","probcache_v3")
    MAN={r['idx']:r for r in json.load(open(os.path.join(PC,"manifest.json")))}
    CEN=json.load(open(os.path.join(HERE,"cand","_census_hr3.json")))
    return PC,MAN,CEN

def solve_idx(idx,PC,MAN,CEN):
    from calib_train.multihyp_calib import groups_topk, ROLES
    ce=CEN.get(str(idx))
    if not ce or ce.get('src') is None: return None,None,None,None
    d=np.load(os.path.join(PC,f"{idx:03d}.npz")); prob=d['prob'].astype(np.float32); w,h=int(d['w']),int(d['h'])
    combo=ce.get('combo',[0,0,0,0]) or [0,0,0,0]
    cands,_=groups_topk(prob,w,h,K=3)
    if cands is None: return None,None,None,None
    try: groups={r:cands[r][ci] for r,ci in zip(ROLES,combo)}
    except (IndexError,KeyError): groups={r:cands[r][0] for r in ROLES}
    rec=AC.solve_calib(groups,w/2,h/2,w/2)
    return rec,prob,(w,h),ce

def venue_name(fname):
    base=fname.split("__")[0]
    import re; return re.sub(r'(?<=[a-zçğışöü])(?=[A-ZÇĞİŞÖÜ])',' ',base)

if __name__=="__main__":
    PC,MAN,CEN=_load_ctx()
    outdir=os.path.join(HERE,"cand","judge"); os.makedirs(outdir,exist_ok=True)
    idxs=[int(x) for x in sys.argv[1:]] if len(sys.argv)>1 else [10,13,35,59,61,90,5,1]
    for idx in idxs:
        rec,prob,wh,ce=solve_idx(idx,PC,MAN,CEN)
        if rec is None: print(f"idx{idx} calib yok"); continue
        w,h=wh; img=cv2.imread(os.path.join(HERE,"cand_big",MAN[idx]['file'])); img=cv2.resize(img,(w,h))
        vn=venue_name(MAN[idx]['file'])
        can=render_overlay(img,rec,venue=vn,idx=idx,judge=True)
        out=os.path.join(outdir,f"{idx:03d}.jpg"); cv2.imwrite(out,can,[cv2.IMWRITE_JPEG_QUALITY,92])
        print(f"idx{idx} {vn} -> {out}")
