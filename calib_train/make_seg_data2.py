#!/usr/bin/env python3
"""ROL-AWARE eğitim verisi (tam otonom): maske 7 KANAL = her çizginin ROLÜ ayrı sınıf:
  0 goalN(yakın kale) 1 goalF(uzak kale) 2 touchN(yakın taç) 3 touchF(uzak taç)
  4 center 5 box 6 circle
Model çıkarımda hangi çizgi hangisi diye DOĞRUDAN söyler -> geometrik tahmin YOK.
- GERÇEK: 26 saha, roller kullanıcı etiketlerinden.
- SENTETİK: 3D KÖŞE-KAMERA (kamera X=0 kalesi arkasında, bir yanda, yüksek) -> gerçekçi yakın/uzak.
Önizleme: cand/_segdata2.jpg
"""
import json, os, numpy as np, cv2
HERE=os.path.dirname(os.path.abspath(__file__)); LF=os.path.join(HERE,"label_frames")
J=json.load(open(os.path.join(HERE,"calib_lines.json"))); ORDER=json.load(open(os.path.join(HERE,"frames.json")))
IW,IH=640,360
CLASSES=["goalN","goalF","touchN","touchF","center","box","circle"]; NC=len(CLASSES)
COL=[(0,0,255),(0,150,255),(0,255,0),(255,120,0),(0,255,255),(255,0,200),(255,150,255)]  # görselleştirme
K1,K2=0.167,0.240
_rdg=np.linspace(0,2.6,3000)
def distort(uv_n,k1=K1,k2=K2):
    ru=np.sqrt((uv_n**2).sum(1))+1e-9; rrd=_rdg*(1+k1*_rdg*_rdg+k2*_rdg**4)
    rd=np.interp(ru,rrd,_rdg); return uv_n*(rd/ru)[:,None]

# ---------- GERÇEK ----------
def real_ok(name):
    ln=J.get(name,{}).get("lines",{})
    return all(ln.get(x) and len(ln[x])>=2 for x in ("goalN","touchN","touchF"))
def real_sample(name,size=None):  # AUDIT-fix: 640-darboğaz — eğitim hedef-res'unda rasterize
    img=cv2.imread(os.path.join(LF,name));
    if img is None: return None
    h,w=img.shape[:2]; ln=J[name]["lines"]; th=max(3,w//240)
    m=np.zeros((NC,h,w),np.uint8)
    def draw(ci,ids,closed=False):
        for lid in ids:
            p=ln.get(lid)
            if p and len(p)>=2: cv2.polylines(m[ci],[np.array(p,np.int32)],closed,255,th,cv2.LINE_AA)
    draw(0,["goalN"]); draw(1,["goalF"]); draw(2,["touchN"]); draw(3,["touchF"])
    if ln.get("center") and len(ln["center"])>=4: draw(4,["center"])
    draw(5,["boxN","boxF"])
    if ln.get("circle") and len(ln["circle"])>=6: draw(6,["circle"],closed=True)
    W2,H2=size or (IW,IH)
    img=cv2.resize(img,(W2,H2)); m=np.stack([cv2.resize(m[c],(W2,H2)) for c in range(NC)])
    return img,m

# ---------- SENTETİK (3D köşe-kamera) ----------
def look_R(cam,tgt):
    f=tgt-cam; f=f/np.linalg.norm(f); wup=np.array([0,0,1.])
    r=np.cross(f,wup); r=r/(np.linalg.norm(r)+1e-9); u=np.cross(r,f)
    return np.stack([r,-u,f])
def _bg_clutter(img,rng):
    """Gerçek halısaha çevresi: kubbe-kiriş (kavisli), file-ızgara, reklam bandı, dikey destek.
    Maskeye YAZILMAZ -> negatif örnek: model bunları pitch-çizgisinden ayırmayı öğrenir."""
    H,W_=img.shape[:2]
    # 1) KUBBE KİRİŞLERİ — üst bölgede parlak kavisli yaylar (pitch-çizgisine en çok benzeyen confuser)
    for _ in range(rng.randint(3,9)):
        apex=rng.randint(-30,int(H*0.32)); amp=rng.randint(14,70); c=int(rng.randint(105,205)); thk=rng.randint(1,3)
        xs=np.linspace(0,W_,60); ys=apex+amp*np.sin(np.linspace(0.15,np.pi-0.15,60))+rng.randint(-5,5)
        cv2.polylines(img,[np.stack([xs,ys],1).astype(np.int32)],False,(c,c,c),thk,cv2.LINE_AA)
    # 2) DİKEY KİRİŞ DESTEKLERİ
    for _ in range(rng.randint(4,13)):
        x=rng.randint(0,W_); c=int(rng.randint(85,175))
        cv2.line(img,(x,0),(x+rng.randint(-14,14),rng.randint(int(H*0.12),int(H*0.5))),(c,c,c),rng.randint(1,2),cv2.LINE_AA)
    # 3) FİLE IZGARASI — kenarlarda/üstte çapraz ağ
    if rng.rand()<0.7:
        step=rng.randint(9,20); ytop=rng.randint(int(H*0.2),int(H*0.6))
        for x in range(0,W_,step):
            c=int(rng.randint(55,120)); cv2.line(img,(x,0),(x,ytop),(c,c,c),1,cv2.LINE_AA)
        if rng.rand()<0.5:
            for y in range(0,ytop,step):
                c=int(rng.randint(55,120)); cv2.line(img,(0,y),(W_,y),(c,c,c),1,cv2.LINE_AA)
    # 4) REKLAM / DUVAR BANTLARI — üst bölgede yatay dikdörtgenler
    for _ in range(rng.randint(0,3)):
        y=rng.randint(int(H*0.04),int(H*0.3)); hh=rng.randint(6,22); col=rng.randint(0,210,3)
        cv2.rectangle(img,(rng.randint(0,W_//2),y),(rng.randint(W_//2,W_),y+hh),tuple(int(v) for v in col),-1)

def synth_sample(seed, return_meta=False):
    rng=np.random.RandomState(seed); L=rng.uniform(26,42); W=rng.uniform(16,24)
    side=rng.randint(2)   # 0: kamera Y~0 tarafında, 1: Y~W
    cyc=rng.uniform(-2,W*0.35) if side==0 else rng.uniform(W*0.65,W+2)
    cam=np.array([rng.uniform(-3,0.5), cyc, rng.uniform(3.2,6.0)])
    tgt=np.array([L*rng.uniform(0.55,0.8), W*rng.uniform(0.4,0.6), 0.]); R=look_R(cam,tgt)
    f=rng.uniform(230,340); k1=K1+rng.uniform(-0.05,0.05); k2=K2+rng.uniform(-0.06,0.06)
    def proj(P2):  # pitch(x,y,0) -> görüntü pikseli (distorted)
        P3=np.column_stack([P2,np.zeros(len(P2))]); Pc=(P3-cam)@R.T; z=np.clip(Pc[:,2],1e-3,None)
        xn=Pc[:,0]/z; yn=Pc[:,1]/z; d=distort(np.stack([xn,yn],1),k1,k2)
        return np.stack([f*d[:,0]+IW/2, f*d[:,1]+IH/2],1), Pc[:,2]
    # roller: goalN=X0, goalF=XL, touchN=(Y0 if side==0 else YW), touchF=öteki
    ntY=0.0 if side==0 else W; ftY=W-ntY
    base=np.array([rng.randint(45,90),rng.randint(95,150),rng.randint(45,90)],np.uint8)
    img=np.full((IH,IW,3),[max(20,base[0]-30),max(30,base[1]-45),max(20,base[2]-30)],np.uint8)
    def polyfill(P2,col):
        pp,z=proj(P2);
        if (z>0).all(): cv2.fillConvexPoly(img,pp.astype(np.int32),col)
    polyfill(np.array([[0,0],[L,0],[L,W],[0,W]],float),base.tolist())
    nb=max(6,int(L/3))
    for i in range(nb):
        x0,x1=i*L/nb,(i+1)*L/nb; col=np.clip(base.astype(int)+(14 if i%2 else -14),0,255)
        polyfill(np.array([[x0,0],[x1,0],[x1,W],[x0,W]],float),col.tolist())
    m=np.zeros((NC,IH,IW),np.uint8); thk=rng.randint(2,4)
    def line(ci,a,b,N=50,closed=False):
        t=np.linspace(0,1,N)[:,None]; P=a*(1-t)+b*t; pp,z=proj(P)
        if (z>0).any(): cv2.polylines(m[ci],[pp[z>0].astype(np.int32)],closed,255,thk,cv2.LINE_AA)
    line(0,np.array([0,0.]),np.array([0,W]))        # goalN X=0
    line(1,np.array([L,0.]),np.array([L,W]))        # goalF X=L
    line(2,np.array([0,ntY]),np.array([L,ntY]))     # touchN
    line(3,np.array([0,ftY]),np.array([L,ftY]))     # touchF
    line(4,np.array([L/2,0.]),np.array([L/2,W]))    # center
    for gx in (0,L):                                 # box (her iki uç)
        bx=5 if gx==0 else L-5
        for a,b in [([bx,W/2-5],[bx,W/2+5]),([gx,W/2-5],[bx,W/2-5]),([gx,W/2+5],[bx,W/2+5])]:
            line(5,np.array(a,float),np.array(b,float),N=20)
    thc=np.linspace(0,2*np.pi,60); circ=np.stack([L/2+2.5*np.cos(thc),W/2+2.5*np.sin(thc)],1)
    pp,z=proj(circ);
    if (z>0).all(): cv2.polylines(m[6],[pp.astype(np.int32)],True,255,thk,cv2.LINE_AA)
    # beyaz çizgileri görüntüye bas (tüm roller)
    anyline=(m.max(0)>0); img[anyline]=(235,235,235)
    # oyuncular
    for _ in range(rng.randint(0,12)):
        P=np.array([[rng.uniform(2,L-2),rng.uniform(1,W-1)]]); pp,z=proj(P)
        if z[0]<=0: continue
        fx,fy=pp[0]; hh=rng.randint(16,38); ww=max(4,hh//3)
        cv2.rectangle(img,(int(fx-ww),int(fy-hh)),(int(fx+ww),int(fy)),tuple(int(c) for c in rng.randint(0,255,3)),-1)
    # dış kenar koyu/kahve
    poly,_=proj(np.array([[0,0],[L,0],[L,W],[0,W]],float)); out=np.ones((IH,IW),np.uint8)*255
    cv2.fillConvexPoly(out,poly.astype(np.int32),0)
    brown=np.array([rng.randint(28,60),rng.randint(34,70),rng.randint(42,85)],np.uint8)
    img[out>0]=(img[out>0]*0.4+brown*0.6).astype(np.uint8)
    # GÖLGELER (Kalekapısı/ATAK gibi sahalar için)
    for _ in range(rng.randint(0,4)):
        sh=np.zeros((IH,IW),np.uint8)
        cv2.line(sh,(rng.randint(-50,IW),rng.randint(-50,IH)),(rng.randint(0,IW),rng.randint(0,IH+50)),255,rng.randint(18,70))
        img[sh>0]=(img[sh>0]*rng.uniform(0.4,0.72)).astype(np.uint8)
    # ÇEVRE-CLUTTER (gerçek halısaha domain gap): kubbe-kirişi + file-ızgarası + reklam
    # -> maskeye GİRMEZ; model "yapısal çizgi (kavisli/ızgara) pitch-çizgisi DEĞİL" öğrenir.
    #    Loş sahalarda pitch-çizgisi soluk kalınca model bunları çizgi sanıp blob üretiyordu.
    _bg_clutter(img,rng)
    # AYDINLATMA (karanlık varyant dahil) + gürültü
    img=np.clip(img*rng.uniform(0.45,1.18)+rng.normal(0,8,img.shape),0,255).astype(np.uint8)
    if return_meta: return img,m,{"proj":proj,"L":L,"W":W,"ntY":ntY,"ftY":ftY}
    return img,m

def overlay(img,m):
    o=img.copy()
    for c in range(NC): o[m[c]>80]=COL[c]
    return o

if __name__=="__main__":
    reals=[n for n in ORDER if n in J and real_ok(n)]; print(f"gerçek: {len(reals)}")
    tiles=[overlay(*real_sample(n)) for n in reals[:4] if real_sample(n)]
    tiles+=[overlay(*synth_sample(sd)) for sd in range(8)]
    cols=4; rows=(len(tiles)+cols-1)//cols
    sheet=np.full((rows*(IH+24),cols*(IW+6),3),20,np.uint8)
    lbl=["GERCEK"]*4+["SENTETIK"]*8
    for i,t in enumerate(tiles):
        r,c=divmod(i,cols); y=r*(IH+24); x=c*(IW+6); sheet[y+20:y+20+IH,x:x+IW]=t
        cv2.putText(sheet,lbl[i],(x+4,y+15),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    # rol-renk lejantı
    leg=" ".join(f"{CLASSES[c]}" for c in range(NC))
    cv2.imwrite(os.path.join(HERE,"cand","_segdata2.jpg"),sheet)
    print("rol-renk: goalN=kirmizi goalF=turuncu touchN=yesil touchF=mavi center=sari box=mor circle=pembe")
    print("-> cand/_segdata2.jpg")
