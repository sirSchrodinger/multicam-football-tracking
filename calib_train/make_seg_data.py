#!/usr/bin/env python3
"""Çizgi-segmentasyon eğitim verisi: 2 kanal (sınır-çizgileri / iç-çizgiler).
- GERÇEK: 25 etiketli saha (ham görüntü + kullanıcı çizgilerinden maske).
- SENTETİK: rastgele saha/kamera/şerit/oyuncu + gerçek-bozulma; sınırsız.
Model: RGB(ham, eğri) -> maske. Çıkarımda maske -> undistort -> 4 sınır çizgisi -> homografi.
Önizleme: cand/_segdata.jpg
"""
import json, os, numpy as np, cv2
HERE=os.path.dirname(os.path.abspath(__file__)); LF=os.path.join(HERE,"label_frames")
J=json.load(open(os.path.join(HERE,"calib_lines.json"))); ORDER=json.load(open(os.path.join(HERE,"frames.json")))
IW,IH=640,360
BOUND=["goalN","goalF","touchN","touchF"]; INNER=["center","boxN","boxF","circle"]
K1,K2=0.167,0.240

# ---------- GERÇEK ----------
def real_sample(name):
    img=cv2.imread(os.path.join(LF,name));
    if img is None: return None
    h,w=img.shape[:2]; ln=J[name]["lines"]
    mb=np.zeros((h,w),np.uint8); mi=np.zeros((h,w),np.uint8)
    def draw(dst,ids,th):
        for lid in ids:
            p=ln.get(lid)
            if p and len(p)>=2:
                P=np.array(p,np.int32)
                if lid=="circle" and len(p)>=5: cv2.polylines(dst,[P],True,255,th,cv2.LINE_AA)
                else: cv2.polylines(dst,[P],False,255,th,cv2.LINE_AA)
    draw(mb,BOUND,max(3,w//240)); draw(mi,INNER,max(3,w//260))
    img=cv2.resize(img,(IW,IH)); mb=cv2.resize(mb,(IW,IH)); mi=cv2.resize(mi,(IW,IH))
    return img,np.stack([mb,mi],0)

def real_ok(name):
    ln=J.get(name,{}).get("lines",{})
    return all(ln.get(x) and len(ln[x])>=2 for x in BOUND)

# ---------- SENTETİK ----------
_rd=np.linspace(0,2.6,3000)
def distort(uv_n):  # ideal(undist) normalized -> distorted normalized (barrel), merkez=0
    ru=np.sqrt((uv_n**2).sum(1))+1e-9
    rrd=_rd*(1+K1*_rd*_rd+K2*_rd**4)         # rd->ru tablosu
    rd=np.interp(ru,rrd,_rd); sc=rd/ru
    return uv_n*sc[:,None]
def synth_sample(seed):
    rng=np.random.RandomState(seed)
    L=rng.uniform(26,42); W=rng.uniform(16,24)
    # rastgele kamera: saha dikdörtgenini görüntüde yamuğa eşle (yakın kenar geniş-alt)
    cxp,cyp=IW/2,IH/2; s=IW/2
    # köşe ekran konumları (yamuk) + jitter; kamera tarafı rastgele
    nearY=rng.uniform(0.78,0.96); farY=rng.uniform(0.06,0.24)
    nbX0=rng.uniform(0.02,0.18); nbX1=rng.uniform(0.82,0.98)
    fbX0=rng.uniform(0.30,0.46); fbX1=rng.uniform(0.54,0.70)
    sk=rng.uniform(-0.12,0.12)
    img_corners=np.array([[nbX0+sk,nearY],[nbX1+sk,nearY],[fbX1,farY],[fbX0,farY]])*[IW,IH]  # nN,nF? -> (0,0),(L,0)? sırası
    # pitch köşeleri: (0,0)(L,0)(L,W)(0,W)
    pc=np.array([[0,0],[L,0],[L,W],[0,W]],float)
    H,_=cv2.findHomography(pc,img_corners)
    def proj(P):  # pitch -> ideal ekran -> distort
        Ph=np.concatenate([P,np.ones((len(P),1))],1).T; q=H@Ph; q=(q[:2]/q[2]).T
        n=(q-[cxp,cyp])/s; d=distort(n); return d*s+[cxp,cyp]
    base=np.array([rng.randint(40,90),rng.randint(90,150),rng.randint(40,90)],np.uint8)  # yeşil
    img=np.zeros((IH,IW,3),np.uint8); img[:]= [max(20,base[0]-25),max(30,base[1]-40),max(20,base[2]-25)]
    # saha poligonu (distorted) -> maske içi
    poly=proj(np.array([[0,0],[L,0],[L,W],[0,W]],float)).astype(np.int32)
    cv2.fillConvexPoly(img,poly,base.tolist())
    # şeritler (pitch X bantları)
    nb=int(L/3)
    for i in range(nb):
        x0,x1=i*L/nb,(i+1)*L/nb; band=proj(np.array([[x0,0],[x1,0],[x1,W],[x0,W]],float)).astype(np.int32)
        shade= 14 if i%2 else -14
        col=np.clip(base.astype(int)+shade,0,255).astype(np.uint8); cv2.fillConvexPoly(img,band,col.tolist())
    mb=np.zeros((IH,IW),np.uint8); mi=np.zeros((IH,IW),np.uint8)
    def line(dst,a,b,th,N=40):
        P=proj(np.linspace(a,b,N)).astype(np.int32); cv2.polylines(dst,[P],False,255,th,cv2.LINE_AA)
    thb=rng.randint(2,4)
    for a,b in [([0,0],[L,0]),([0,W],[L,W]),([0,0],[0,W]),([L,0],[L,W])]: line(mb,np.array(a,float),np.array(b,float),thb)
    line(mi,np.array([L/2,0.]),np.array([L/2,W]),thb)         # orta
    for gx in (0,L):                                          # ceza sahaları
        bx=5 if gx==0 else L-5
        for a,b in [([bx,W/2-5],[bx,W/2+5]),([gx,W/2-5],[bx,W/2-5]),([gx,W/2+5],[bx,W/2+5])]:
            line(mi,np.array(a,float),np.array(b,float),thb)
    th=np.linspace(0,2*np.pi,60); circ=np.stack([L/2+3*np.cos(th),W/2+3*np.sin(th)],1)
    Pc=proj(circ).astype(np.int32); cv2.polylines(mi,[Pc],True,255,thb,cv2.LINE_AA)
    # beyaz çizgileri görüntüye de bas (model RGB'den öğrensin)
    img[mb>0]=(235,235,235); img[mi>0]=(225,225,225)
    # oyuncular (rastgele renkli bloblar, ayak sahada)
    for _ in range(rng.randint(0,12)):
        px,py=rng.uniform(2,L-2),rng.uniform(1,W-1); foot=proj(np.array([[px,py]]))[0]
        hh=rng.randint(18,40); ww=max(4,hh//3); col=tuple(int(c) for c in rng.randint(0,255,3))
        cv2.rectangle(img,(int(foot[0]-ww),int(foot[1]-hh)),(int(foot[0]+ww),int(foot[1])),col,-1)
    # dış kenar (kahve/duvar) - saha dışını koyu/kahve yap
    outside=np.ones((IH,IW),np.uint8)*255; cv2.fillConvexPoly(outside,poly,0)
    brown=np.array([rng.randint(30,60),rng.randint(35,70),rng.randint(45,85)],np.uint8)
    img[outside>0]=(img[outside>0]*0.4+brown*0.6).astype(np.uint8)
    # aydınlatma + gürültü
    img=np.clip(img*rng.uniform(0.7,1.15)+rng.normal(0,7,img.shape),0,255).astype(np.uint8)
    return img,np.stack([mb,mi],0)

if __name__=="__main__":
    reals=[n for n in ORDER if n in J and real_ok(n)]
    print(f"gerçek saha: {len(reals)}")
    tiles=[]
    for n in reals[:4]:
        r=real_sample(n)
        if r: img,m=r; ov=img.copy(); ov[m[0]>80]=(0,0,255); ov[m[1]>80]=(0,255,0); tiles.append(ov)
    for sd in range(8):
        img,m=synth_sample(sd); ov=img.copy(); ov[m[0]>80]=(0,0,255); ov[m[1]>80]=(0,255,0); tiles.append(ov)
    cols=4; rows=(len(tiles)+cols-1)//cols
    sheet=np.full((rows*(IH+6),cols*(IW+6),3),20,np.uint8)
    for i,t in enumerate(tiles):
        rr,cc=divmod(i,cols); sheet[rr*(IH+6):rr*(IH+6)+IH,cc*(IW+6):cc*(IW+6)+IW]=t
    cv2.imwrite(os.path.join(HERE,"cand","_segdata.jpg"),sheet)
    print("önizleme -> cand/_segdata.jpg (üst sıra GERÇEK, alt SENTETİK; kırmızı=sınır yeşil=iç)")
