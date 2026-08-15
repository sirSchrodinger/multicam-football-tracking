#!/usr/bin/env python3
"""MULTİ-HİPOTEZ KALİBRASYON (audit adım-4, en yüksek kaldıraç).
KÖK (D2-kanıtlı): doğru çizgi çoğu kez BULUNUYOR ama largest-CC yanlış blob'u (çit/koşu-bandı) seçiyor
+ 4-çizgi=sıfır-artıklık → solver yanlışı residual'dan GÖREMEZ (D1: 63 düşük-residual-yanlış vaka).
ÇÖZÜM: rol-başına top-K bileşen → ≤K^4 kombo exact-solve → SEÇİM fit'te KULLANILMAYAN iç-kanal
desteğiyle (center/box/circle geri-projeksiyon = held-out kanıt; residual/gate GAMEABLE, bu değil).
API: solve_multihyp(prob,w,h,img=None,K=3) -> (rec|None, debug)  [rec auto_calib formatı + mh_support]"""
import numpy as np, cv2
from calib_train import auto_calib as AC
from calib_train.auto_clean2d import sanity

ROLES=("goalN","goalF","touchN","touchF")

def groups_topk(prob,w,h,thr=0.5,minpts=25,maxpts=180,K=3,seed=0):
    """rol-başına en-büyük K bileşenin nokta-kümeleri: {role:[pts,...]} (alan-sıralı)."""
    rs=np.random.RandomState(seed); out={}
    for ci,role in enumerate(ROLES):
        pc=cv2.resize(prob[ci],(w,h)); mk=(pc>thr).astype(np.uint8)
        mk=cv2.morphologyEx(mk,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
        ncc,lbl,st,_=cv2.connectedComponentsWithStats(mk)
        cands=[]
        if ncc>1:
            order=np.argsort(st[1:,cv2.CC_STAT_AREA])[::-1]+1
            for li in order[:K]:
                pts=np.column_stack(np.where(lbl==li))[:,::-1].astype(float)
                if len(pts)<minpts: continue
                if len(pts)>maxpts: pts=pts[rs.choice(len(pts),maxpts,replace=False)]
                cands.append(pts)
        if not cands: return None,f"{role} aday yok"
        out[role]=cands
    return out,None

def _sample_prob(ch,pix,w,h):
    """kanal-prob'unu (model-res) projekte-çizgi pikselleri boyunca örnekle -> ortalama."""
    hs,ws=ch.shape
    fin=np.isfinite(pix).all(1)&(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)
    if fin.sum()<8: return None,0.0
    xs=(pix[fin,0]*ws/w).astype(int).clip(0,ws-1); ys=(pix[fin,1]*hs/h).astype(int).clip(0,hs-1)
    return float(ch[ys,xs].mean()), float(fin.mean())

def interior_support(rec,prob,w,h,Wp=18.0):
    """FİT'TE KULLANILMAYAN iç-kanallardan destek skoru (held-out kanıt, non-gameable seçim).
    center(4): orta-çizgi; box(5): iki ceza-kutusu (derinlik 5 ve 6m dene); circle(6): R-taraması.
    Döner (skor, detay). Skor = görünür-iç-kanıtın prob-ortalamalarının ağırlıklı ortalaması."""
    k1,k2,H=rec["k1"],rec["k2"],rec["H"]; cx,cy,s=w/2,h/2,w/2; L=AC.L
    P=lambda M: AC.project_metric(np.asarray(M,float),k1,k2,H,cx,cy,s)
    parts=[]; det={}
    # orta-çizgi
    m,cov=_sample_prob(prob[4],P(np.linspace([L/2,0],[L/2,Wp],60)),w,h)
    if m is not None and cov>0.3: parts.append((m,1.0)); det['center']=round(m,3)
    # ceza-kutuları (derinlik belirsiz: 5m ve 6m dene, iyisini al; iki kale)
    box_best=[]
    for depth in (5.0,6.0):
        for gx in (0.0,L):
            bx=depth if gx==0 else L-depth
            seg=np.vstack([np.linspace([gx,Wp/2-5],[bx,Wp/2-5],25),
                           np.linspace([bx,Wp/2-5],[bx,Wp/2+5],25),
                           np.linspace([bx,Wp/2+5],[gx,Wp/2+5],25)])
            m,cov=_sample_prob(prob[5],P(seg),w,h)
            if m is not None and cov>0.3: box_best.append(m)
    if box_best: mb=float(np.max(box_best)); parts.append((mb,1.0)); det['box']=round(mb,3)
    # çember (R standart değil: 1.1-3.7m -> tarama, en iyi R)
    th=np.linspace(0,2*np.pi,48); circ_best=[]
    for R in (1.5,2.0,2.5,3.0,3.5):
        m,cov=_sample_prob(prob[6],P(np.c_[L/2+R*np.cos(th),Wp/2+R*np.sin(th)]),w,h)
        if m is not None and cov>0.5: circ_best.append(m)
    if circ_best: mc=float(np.max(circ_best)); parts.append((mc,0.7)); det['circle']=round(mc,3)
    if not parts: return 0.0,det   # hiç iç-kanıt görünmüyor -> ayırt edici değil
    sc=float(sum(m*wt for m,wt in parts)/sum(wt for _,wt in parts))
    det['n_evidence']=len(parts)
    return sc,det

def solve_multihyp(prob,w,h,img=None,K=3,fit_max=0.8,thr=0.5):
    """tüm rol-kombolarını çöz, sanity+fit ile buda, iç-kanal-desteğiyle SEÇ.
    Döner (best_rec|None, debug). best_rec: auto_calib formatı + mh_support/mh_combo/mh_ncand."""
    cands,why=groups_topk(prob,w,h,thr=thr,K=K)
    if cands is None: return None,{'reason':why}
    from itertools import product
    cx,cy,s=w/2,h/2,w/2
    combos=list(product(*[range(len(cands[r])) for r in ROLES]))
    scored=[]; tried=0
    for combo in combos:
        groups={r:cands[r][ci] for r,ci in zip(ROLES,combo)}
        rec=AC.solve_calib(groups,cx,cy,s)
        tried+=1
        if rec is None or rec['fit']>fit_max or not rec.get('converged',True): continue
        ok,_=sanity(rec,w,h,img=img)          # konveks/alan/PCA/karanlık + fit-gate
        if not ok: continue
        sup,det=interior_support(rec,prob,w,h)
        scored.append((sup,combo,rec,det))
    if not scored: return None,{'reason':'hicbir kombo sanity+fit gecmedi','tried':tried}
    scored.sort(key=lambda x:-x[0])
    sup,combo,rec,det=scored[0]
    rec.update(ok=True,mh_support=round(sup,4),mh_combo=list(combo),mh_ncand=len(scored),mh_detail=det)
    return rec,{'tried':tried,'passed':len(scored),
                'top3':[(round(s,3),list(c)) for s,c,_,_ in scored[:3]]}

# ---------------- SEÇİCİ-V2 (2 Tem gece): sıkı-geometri + yayılım-katı destek + feet ----------------
# V1-desteğin çuvalladığı yer (görsel-altın-set kanıtı): dejenere H uzun metrik-çizgiyi kısa-yay/çökük
# bölgeye düşürüp yaygın prob-kütlesinden skor topluyor (idx107 pembe-girdap s=0.23=max!).
# V2: (a) HARD-VALIDITY — projekte orta-çizgi düz+uzun+görüntü-içi, çember-elips makul-ölçekli,
# quad kenar-uzunlukları makul; (b) destek COV>=0.5 katı; (c) feet varsa inside+SPREAD zorunlu.
def _polyline_ok(pix,w,h,min_len_frac=0.12,min_in=0.5,min_straight=0.6):
    fin=np.isfinite(pix).all(1); p=pix[fin]
    if len(p)<8: return False
    inm=(p[:,0]>=0)&(p[:,0]<w)&(p[:,1]>=0)&(p[:,1]<h)
    if inm.mean()<min_in: return False
    q=p[inm]; seg=np.linalg.norm(np.diff(q,axis=0),axis=1).sum()
    diag=np.hypot(w,h)
    if seg<min_len_frac*diag: return False           # çöküp kısalmış çizgi
    ee=np.linalg.norm(q[-1]-q[0])
    return ee>=min_straight*seg                      # katlanmış/spiral çizgi reddi

def validity_v2(rec,w,h,Wp=18.0):
    k1,k2,H=rec["k1"],rec["k2"],rec["H"]; cx,cy,s=w/2,h/2,w/2; L=AC.L; diag=np.hypot(w,h)
    P=lambda M: AC.project_metric(np.asarray(M,float),k1,k2,H,cx,cy,s)
    # orta-çizgi düz+uzun+içeride
    if not _polyline_ok(P(np.linspace([L/2,0],[L/2,Wp],60)),w,h): return False,"orta-çizgi dejenere"
    # çember-elips makul ölçek (aspect-kontrolü KALDIRILDI: fisheye köşe-kamerada meşru elips
    # çok eksantrik olabiliyor — altın-set 7 iyi sahayı yanlış reddettirdi)
    th=np.linspace(0,2*np.pi,60); cp=P(np.c_[L/2+2.0*np.cos(th),Wp/2+2.0*np.sin(th)])
    fin=np.isfinite(cp).all(1)
    if fin.sum()>=36:
        q=cp[fin]; c=q.mean(0); r=np.linalg.norm(q-c,axis=1)
        if not (0.008*diag<=r.mean()<=0.28*diag): return False,f"çember ölçek anormal ({r.mean():.0f}px)"
    # quad kenarları makul (çökme reddi)
    corners=P([[0,0],[L,0],[L,Wp],[0,Wp]])
    if not np.isfinite(corners).all(): return False,"köşe inf"
    for i in range(4):
        if np.linalg.norm(corners[i]-corners[(i+1)%4])<0.06*diag: return False,"quad kenarı çökük"
    return True,"ok"

def feet_score(rec,feet,w,h,Wp=18.0):
    """feet [[x,y,conf],...] -> (inside_frac|None, spread_ok). <5 feet -> (None,True)."""
    if feet is None or len(feet)<5: return None,True
    F=np.asarray([[f[0],f[1]] for f in feet],float)
    m=AC.aH(AC.und_norm(F,rec['k1'],rec['k2'],w/2,h/2,w/2),rec['H'])
    fin=np.isfinite(m).all(1); m=m[fin]
    if len(m)<5: return 0.0,False
    ins=((m[:,0]>=-1)&(m[:,0]<=AC.L+1)&(m[:,1]>=-1)&(m[:,1]<=Wp+1))
    mi=m[ins]
    spread_ok=True
    if len(mi)>=6:
        ext=(mi[:,0].max()-mi[:,0].min(),mi[:,1].max()-mi[:,1].min())
        spread_ok=ext[0]>=6.0 and ext[1]>=3.0    # oyuncular gerçek sahada toplu-iğne olmaz
    return float(ins.mean()),spread_ok

def grass_inside(rec,img,w,h,Wp=18.0,margin=1.5,n=24):
    """GÖRÜNÜM-ÇIPASI (gameable değil: gerçek pikseller): projekte saha-içi ızgara noktalarında
    görüntü YEŞİL/çim mi? Mavi-bina/pembe-girdap/tribün warp'larını anında düşürür.
    Döner green_frac (0-1) veya None (çoğu nokta görüntü-dışı)."""
    k1,k2,H=rec["k1"],rec["k2"],rec["H"]; cx,cy,s=w/2,h/2,w/2; L=AC.L
    gx,gy=np.meshgrid(np.linspace(margin,L-margin,n),np.linspace(margin,Wp-margin,max(6,n//2)))
    pix=AC.project_metric(np.c_[gx.ravel(),gy.ravel()],k1,k2,H,cx,cy,s)
    fin=np.isfinite(pix).all(1)&(pix[:,0]>=0)&(pix[:,0]<w)&(pix[:,1]>=0)&(pix[:,1]<h)
    if fin.mean()<0.4: return None
    p=pix[fin].astype(int)
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    px=hsv[p[:,1],p[:,0]]
    hgood=(px[:,0]>=30)&(px[:,0]<=95)                       # yeşil ton
    grassish=hgood&((px[:,1]>25)|(px[:,2]<110))             # doygun-yeşil VEYA koyu-çim
    return float(grassish.mean())

def far_grounding(rec,prob,w,h,Wp=18.0):
    """REJECTION sinyali (2 Tem akşam bulgusu): reprojekte KALE çizgileri kendi kanal-aktivasyonuna
    oturuyor mu. Yamuk sahalarda far-kenar duvara/orta-sahaya kilitlenir -> gf@far ~0 (audit-day:
    good 0.6-0.9, yamuk 0.02-0.25). Non-gameable (interior_support'un BOUNDARY versiyonu, selector
    bunu görmüyordu). Döner dict(gf,gn,best). best=max(gf,gn) (bir kale grounded ise saha tanımlı)."""
    k1,k2,H=rec['k1'],rec['k2'],rec['H']; cx,cy,s=w/2,h/2,w/2; Lm=AC.L
    P=lambda a,b: AC.project_metric(np.linspace(a,b,60),k1,k2,H,cx,cy,s)
    gf,_=_sample_prob(prob[1],P([Lm,0],[Lm,Wp]),w,h)
    gn,_=_sample_prob(prob[0],P([0,0],[0,Wp]),w,h)
    gf=gf or 0.0; gn=gn or 0.0
    return dict(gf=round(gf,3),gn=round(gn,3),best=round(max(gf,gn),3))

def select_v2(rec,prob,w,h,img=None,feet=None,grass_min=0.45):
    """tek adayın v2-değerlendirmesi -> (accept:bool, score:float, why:str)."""
    ok,why=validity_v2(rec,w,h)
    if not ok: return False,0.0,why
    if img is not None:
        g=grass_inside(rec,img,w,h)
        if g is not None and g<grass_min: return False,g,f"saha-içi çim değil (green={g:.2f})"
    fi,spread=feet_score(rec,feet,w,h)
    if fi is not None and (fi<0.6 or not spread): return False,fi,f"feet reddi (in={fi:.2f},spread={spread})"
    sup,det=interior_support(rec,prob,w,h)
    # ALTIN-SET kalibre no-feet tabanı: boş-karede tek çıpa iç-kanal; genuine min sup=0.055,
    # bad'lerin çoğu <0.05 (sup>=0.05: 14/14 genuine kalır, 15/18 bad düşer).
    if fi is None and sup<0.05: return False,sup,f"boş-kare + iç-destek zayıf (sup={sup:.3f})"
    score=sup+(0.5*fi if fi is not None else 0.0)
    return True,float(score),"ok"

# ---------------- YÖNELİM ÇÖZÜCÜ (2 Tem, Alperen rotasyon-hatası bulgusu) ----------------
# KÖK: seg2 kale-çizgisini (kısa kenar) taç-çizgisiyle (uzun kenar) karıştırınca solver sahayı
# 90° dönük/squished oturtuyor (idx 7,8,9,22,77,83). Savunma: normal VE 90°-swap rol-atamasını
# ayrı çöz, çim+iç-destek+feet ile GERÇEK yönelimi seç (gameable değil).
def _solve_oriented(cands,combo,swap,w,h):
    """combo -> groups; swap=True ise goal<->touch rol-atamasını takas (90° döndür)."""
    g={r:cands[r][ci] for r,ci in zip(ROLES,combo)}
    if swap:
        # goalN/goalF (X-kısıt) <-> touchN/touchF (Y-kısıt) yer değiştir
        g={'goalN':g['touchN'],'goalF':g['touchF'],'touchN':g['goalN'],'touchF':g['goalF']}
    return AC.solve_calib(g,w/2,h/2,w/2)

def solve_oriented_best(prob,w,h,img=None,feet=None,K=3,fit_max=0.8):
    """Her iki yönelim × top-K kombo -> select_v2 skoru en yüksek KABUL. Rotasyon-savunmalı.
    Döner (rec|None, debug). rec['orient']='normal'|'swap', rec['src']='oriented'."""
    from itertools import product
    cands,why=groups_topk(prob,w,h,K=K)
    if cands is None: return None,{'reason':why}
    best=None; tried=0
    for swap in (False,True):
        for combo in product(*[range(len(cands[r])) for r in ROLES]):
            rec=_solve_oriented(cands,combo,swap,w,h); tried+=1
            if rec is None or rec['fit']>fit_max or not rec.get('converged',True): continue
            ok,_=sanity(rec,w,h,img=img)
            if not ok: continue
            acc,score,_=select_v2(rec,prob,w,h,img=img,feet=feet)
            if acc and (best is None or score>best[0]):
                rec['orient']='swap' if swap else 'normal'; best=(score,rec)
    if best is None: return None,{'reason':'hicbir yönelim+kombo gecmedi','tried':tried}
    r=best[1]; r['src']='oriented'; return r,{'tried':tried,'score':round(best[0],4),'orient':r['orient']}

# ---------------- STRIPE-ROTASYON BELT (Alperen içgörüsü, ihtiyatlı) ----------------
def _stripe_axis_dev(rec,img,w,h,L=34.0,Wp=18.0,S=10):
    """warp'ta şerit eksen-sapması (0=eksene hizalı) + tutarlılık."""
    Wc=int(L*S);Hc=int(Wp*S);gx,gy=np.meshgrid(np.arange(Wc),np.arange(Hc))
    Xm=gx/S;Ym=Wp-gy/S;P=np.stack([Xm.ravel(),Ym.ravel()],1)
    pix=AC.project_metric(P,rec['k1'],rec['k2'],rec['H'],w/2,h/2,w/2)
    wp=cv2.remap(img,pix[:,0].reshape(Hc,Wc).astype(np.float32),pix[:,1].reshape(Hc,Wc).astype(np.float32),cv2.INTER_LINEAR,borderValue=(0,0,0))
    g=cv2.cvtColor(wp,cv2.COLOR_BGR2GRAY).astype(np.float32); g=cv2.GaussianBlur(g,(0,0),1.5); mask=g>8
    if mask.sum()<400: return 90.0,0.0
    gx2=cv2.Sobel(g,cv2.CV_32F,1,0,ksize=3); gy2=cv2.Sobel(g,cv2.CV_32F,0,1,ksize=3)
    Jxx=(gx2*gx2)[mask].sum();Jyy=(gy2*gy2)[mask].sum();Jxy=(gx2*gy2)[mask].sum()
    ang=np.degrees(0.5*np.arctan2(2*Jxy,Jxx-Jyy)+np.pi/2)%180
    coh=float(np.sqrt((Jxx-Jyy)**2+4*Jxy**2)/(Jxx+Jyy+1e-6))
    return float(min(min(ang,180-ang),abs(ang-90))),coh

def rotation_belt(rec,prob,w,h,img,feet=None):
    """rec ROTASYON-hatalı mı? güçlü-şerit(coh>0.6) + eksen-sapması>25° ise swap-yönelim dene;
    swap hem şerit-daha-hizalı hem select_v2 kabul-edilir ise SWAP'ı döndür. Değilse rec aynen.
    İHTİYATLI: sinyal zayıfsa DOKUNMA (yanlış-swap riski > kazanç)."""
    if img is None: return rec
    dev,coh=_stripe_axis_dev(rec,img,w,h)
    if coh<0.6 or dev<25: return rec           # şerit zayıf ya da zaten hizalı -> dokunma
    from itertools import product
    cands,_=groups_topk(prob,w,h,K=1)
    if cands is None: return rec
    g={r:cands[r][0] for r in ROLES}
    gs={'goalN':g['touchN'],'goalF':g['touchF'],'touchN':g['goalN'],'touchF':g['goalF']}
    rs=AC.solve_calib(gs,w/2,h/2,w/2)
    if rs is None: return rec
    dev2,coh2=_stripe_axis_dev(rs,img,w,h)
    acc,_,_=select_v2(rs,prob,w,h,img=img,feet=feet)
    if acc and dev2<dev-10:                     # swap belirgin daha-hizalı + geçerli
        rs['src']=rec.get('src','?')+'+rotbelt'; rs['rot_fixed']=True; return rs
    return rec
