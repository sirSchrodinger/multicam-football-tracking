#!/usr/bin/env python3
"""OBJEKTİF geometrik kalibrasyon metriği (Alperen kriteri: çizgiler gerçek geometriye PARALEL mi).
Fikir: halısahadaki biçim-şeritleri + beyaz çizgiler gerçek dünyada saha EKSENLERİNE paraleldir.
Doğru kalibrasyonda, saha-içi her noktada kalibrasyonun implied yerel-eksen yönü (dX,dY projeksiyonu)
görüntüdeki baskın doku-yönüyle (structure tensor) ÇAKIŞIR. Yamuk'ta çakışmaz.
Non-gameable, AJANSIZ. Döner mean_align_deg (düşük=iyi) + coverage.

Kullanım: python geom_consistency.py            -> tüm boundary-etiketli idx'lerde metriği ölç + AUC
          python geom_consistency.py 13 6 0     -> tek tek (auto)
"""
import os, sys, json, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))
from calib_train import auto_calib as AC
from calib_train.multihyp_calib import groups_topk, ROLES
L,Wp=AC.L,18.0
PC=os.path.join(HERE,"probcache_v3")
MAN={r['idx']:r for r in json.load(open(os.path.join(PC,"manifest.json")))}
CEN=json.load(open(os.path.join(ROOT,"cand","_census_hr3.json")))

def struct_orient(gray, xs, ys, win=11):
    """her (x,y) etrafında structure-tensor baskın-yön (radyan, undirected 0..pi) + coherence 0..1."""
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=3); gy=cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=3)
    Jxx=cv2.boxFilter(gx*gx,-1,(win,win)); Jyy=cv2.boxFilter(gy*gy,-1,(win,win)); Jxy=cv2.boxFilter(gx*gy,-1,(win,win))
    h,w=gray.shape; out_ang=[]; out_coh=[]
    for x,y in zip(xs,ys):
        xi,yi=int(round(x)),int(round(y))
        if not(0<=xi<w and 0<=yi<h): out_ang.append(np.nan); out_coh.append(0); continue
        a,b,c=Jxx[yi,xi],Jyy[yi,xi],Jxy[yi,xi]
        # baskın gradyan yönü; DOKU-çizgisi ona DİK. tr, det
        tr=a+b; d=np.sqrt(max((a-b)**2+4*c*c,0)); l1=(tr+d)/2; l2=(tr-d)/2
        coh=float((l1-l2)/(l1+l2+1e-6))
        grad_ang=0.5*np.arctan2(2*c,a-b)   # baskın gradyan yönü
        line_ang=grad_ang+np.pi/2           # çizgi/doku yönü (gradyana dik)
        out_ang.append(line_ang%np.pi); out_coh.append(coh)
    return np.array(out_ang), np.array(out_coh)

def _ang_line(dv):
    return (np.arctan2(dv[1],dv[0]))%np.pi

def _dang(a,b):
    d=abs(a-b)%np.pi; return min(d,np.pi-d)   # undirected açı farkı 0..pi/2

def calib_geom_score(img, rec, nx=10, ny=6, d=0.6, coh_min=0.25):
    """saha-içi ızgarada doku-yönü vs kalibrasyon-ekseni min-açı-farkı. Döner (mean_deg, cov, n)."""
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)
    k1,k2,H=rec['k1'],rec['k2'],rec['H']; cx,cy,s=img.shape[1]/2,img.shape[0]/2,img.shape[1]/2
    P=lambda M: AC.project_metric(np.asarray(M,float),k1,k2,H,cx,cy,s)
    gxv=np.linspace(2,L-2,nx); gyv=np.linspace(2,Wp-2,ny)
    pts=[]; axisA=[]; axisB=[]
    for X in gxv:
        for Y in gyv:
            p0=P([[X,Y]])[0]; px=P([[X+d,Y]])[0]; py=P([[X,Y+d]])[0]
            if not(np.isfinite(p0).all() and np.isfinite(px).all() and np.isfinite(py).all()): continue
            pts.append(p0); axisA.append(_ang_line(px-p0)); axisB.append(_ang_line(py-p0))
    if len(pts)<8: return None
    pts=np.array(pts); axisA=np.array(axisA); axisB=np.array(axisB)
    ang,coh=struct_orient(gray,pts[:,0],pts[:,1])
    good=np.isfinite(ang)&(coh>=coh_min)
    if good.sum()<6: return dict(mean_deg=None,cov=float(good.mean()),n=int(good.sum()))
    # her noktada doku-yönü, iki eksenden HANGİSİNE yakınsa o (şeritler bir eksene paralel)
    dmin=np.array([min(_dang(ang[i],axisA[i]),_dang(ang[i],axisB[i])) for i in range(len(ang)) if good[i]])
    return dict(mean_deg=float(np.degrees(dmin.mean())), med_deg=float(np.degrees(np.median(dmin))),
                cov=float(good.mean()), n=int(good.sum()))

def render_geom(idx, out=None):
    """saha-içi ızgarayı yerel doku-vs-eksen uyumuna göre renkli çiz (yeşil=paralel, kırmızı=skew)."""
    rec,wh=solve_auto(idx)
    if rec is None: return None
    img=cv2.resize(img_of(idx),wh); gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY).astype(np.float32)
    k1,k2,H=rec['k1'],rec['k2'],rec['H']; cx,cy,s=img.shape[1]/2,img.shape[0]/2,img.shape[1]/2
    P=lambda M: AC.project_metric(np.asarray(M,float),k1,k2,H,cx,cy,s)
    o=img.copy(); nx,ny,d=14,8,0.6
    devs=[]
    for X in np.linspace(2,L-2,nx):
        for Y in np.linspace(2,Wp-2,ny):
            p0=P([[X,Y]])[0]; px=P([[X+d,Y]])[0]; py=P([[X,Y+d]])[0]
            if not(np.isfinite(p0).all() and np.isfinite(px).all()): continue
            ang,coh=struct_orient(gray,[p0[0]],[p0[1]])
            if not np.isfinite(ang[0]) or coh[0]<0.25: continue
            dv=min(_dang(ang[0],_ang_line(px-p0)),_dang(ang[0],_ang_line(py-p0)))
            deg=np.degrees(dv); devs.append(deg)
            t=min(deg/25.0,1.0); col=(int(60*(1-t)),int(200*(1-t)),int(60+195*t))  # yeşil->kırmızı
            aa=_ang_line(px-p0) if _dang(ang[0],_ang_line(px-p0))<_dang(ang[0],_ang_line(py-p0)) else _ang_line(py-p0)
            e=8; cv2.line(o,(int(p0[0]-e*np.cos(aa)),int(p0[1]-e*np.sin(aa))),(int(p0[0]+e*np.cos(aa)),int(p0[1]+e*np.sin(aa))),(20,20,20),3)
            cv2.line(o,(int(p0[0]-e*np.cos(ang[0])),int(p0[1]-e*np.sin(ang[0]))),(int(p0[0]+e*np.cos(ang[0])),int(p0[1]+e*np.sin(ang[0]))),col,2)
            cv2.circle(o,(int(p0[0]),int(p0[1])),3,col,-1)
    m=float(np.mean(devs)) if devs else -1
    cv2.rectangle(o,(0,0),(o.shape[1],34),(0,0,0),-1)
    cv2.putText(o,f"idx{idx} {MAN[idx]['file'].split('__')[0][:22]}  geom-skew={m:.1f}deg (dusuk=paralel=iyi)",(6,23),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),1,cv2.LINE_AA)
    out=out or os.path.join(ROOT,"cand",f"_GEOM_{idx:03d}.jpg"); cv2.imwrite(out,o); return out,m

def solve_auto(idx):
    ce=CEN.get(str(idx))
    if not ce or ce.get('src') is None: return None,None
    d=np.load(os.path.join(PC,f"{idx:03d}.npz")); prob=d['prob'].astype(np.float32); w,h=int(d['w']),int(d['h'])
    combo=ce.get('combo',[0,0,0,0]) or [0,0,0,0]
    cands,_=groups_topk(prob,w,h,K=3)
    if cands is None: return None,None
    try: g={r:cands[r][ci] for r,ci in zip(ROLES,combo)}
    except (IndexError,KeyError): g={r:cands[r][0] for r in ROLES}
    return AC.solve_calib(g,w/2,h/2,w/2),(w,h)

def img_of(idx):
    return cv2.imread(os.path.join(ROOT,"cand_big",MAN[idx]['file']))

def main():
    if len(sys.argv)>1:
        for idx in [int(x) for x in sys.argv[1:]]:
            rec,wh=solve_auto(idx)
            if rec is None: print(f"idx{idx}: calib yok"); continue
            img=cv2.resize(img_of(idx),wh)
            print(f"idx{idx}: {calib_geom_score(img,rec)}")
        return
    # DOĞRULAMA: boundary-etiketli (auto) idx'lerde metrik vs USABLE/WRONG
    BND=json.load(open(os.path.join(ROOT,"cand","_boundary_results.json")))
    lab={r['idx']:r['verdict'] for r in BND['per_idx'] if r['source']=='auto'}
    rows=[]
    for idx,verd in lab.items():
        rec,wh=solve_auto(idx)
        if rec is None: continue
        img=img_of(idx)
        if img is None: continue
        sc=calib_geom_score(cv2.resize(img,wh),rec)
        if sc and sc.get('mean_deg') is not None:
            rows.append((idx,verd,sc['mean_deg'],sc['med_deg'],sc['cov']))
    rows.sort(key=lambda r:r[2])
    print(f"{'idx':>4} {'verdict':>9} {'mean°':>6} {'med°':>6} {'cov':>5}")
    for idx,v,m,md,cov in rows: print(f"{idx:>4} {v:>9} {m:>6.1f} {md:>6.1f} {cov:>5.2f}")
    # ayrım: USABLE vs (MARGINAL/WRONG)
    U=[m for _,v,m,_,_ in rows if v=='USABLE']; NU=[m for _,v,m,_,_ in rows if v!='USABLE']
    import statistics as st
    if U and NU:
        print(f"\nUSABLE mean°: n={len(U)} med={st.median(U):.1f} | NOT-USABLE: n={len(NU)} med={st.median(NU):.1f}")
        allv=sorted(set(U+NU)); best=None
        for t in allv:
            tp=sum(1 for x in U if x<=t); tn=sum(1 for x in NU if x>t)
            j=tp/len(U)+tn/len(NU)-1
            if best is None or j>best[0]: best=(j,t,tp/len(U),tn/len(NU))
        print(f"en iyi eşik={best[1]:.1f}° Youden J={best[0]:.2f} (USABLE-keep={best[2]:.2f} NOTUSABLE-reject={best[3]:.2f})")

if __name__=="__main__": main()
