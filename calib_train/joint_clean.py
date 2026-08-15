#!/usr/bin/env python3
"""2D JOINT ama KÖŞELER TEMİZ: el-etiketli joint_calib (calib_solved, doğru) ile metrik->görüntü
ters-warp; saha-dışı + aşırı-gerilen(smear) bölgeyi yeşil-şematik çime çevir (siyah üçgen/smear YOK);
üstüne kanonik beyaz çizgiler. 26 sahayı montajla. Çıktı: cand/_JOINT_CLEAN.jpg
"""
import os, sys, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calib_train import auto_calib as AC
HERE=os.path.dirname(os.path.abspath(__file__)); LF=os.path.join(HERE,"label_frames")
C=json.load(open(os.path.join(HERE,"calib_solved.json"))); J=json.load(open(os.path.join(HERE,"calib_lines.json")))
ORDER=json.load(open(os.path.join(HERE,"frames.json")))
L,Wp,S,MG=34.0,18.0,24,3
def camside_of(img,k1,k2,H,ln):
    h,w=img.shape[:2]; cx,cy,s=w/2,h/2,w/2
    def fl(name): return AC.fitL(AC.und_norm(np.asarray(ln[name],float),k1,k2,cx,cy,s))
    g=["goalN","goalF","touchN","touchF"]
    if not all(ln.get(x) for x in g): return "SOL"
    L_={x:fl(x) for x in g}
    c_nN=AC.inter(L_["goalN"],L_["touchN"]);c_nF=AC.inter(L_["goalN"],L_["touchF"]);c_fN=AC.inter(L_["goalF"],L_["touchN"])
    cr=(c_fN[0]-c_nN[0])*(c_nF[1]-c_nN[1])-(c_fN[1]-c_nN[1])*(c_nF[0]-c_nN[0])
    return "SAG" if cr>0 else "SOL"

def clean_topdown(img,k1,k2,H,camside,Wpf=None,Rf=None,bdn=None,bdf=None,bhwn=None,bhwf=None):
    # AUDIT: per-field en-boy (Wpf) + çember-yarıçap (Rf) — verilmezse default (Alperen: her saha farklı)
    Wp=Wpf if Wpf else globals()['Wp']; R=Rf if Rf else 3.0
    BDN=bdn or 5.0; BDF=bdf or 5.0; BHWN=bhwn or 5.5; BHWF=bhwf or 5.5
    h,w=img.shape[:2]; cx,cy,s=w/2,h/2,w/2
    LS,WS=int((L+2*MG)*S),int((Wp+2*MG)*S); Hi=np.linalg.inv(H)
    Xs,Ys=np.meshgrid(np.linspace(-MG,L+MG,LS),np.linspace(-MG,Wp+MG,WS))
    den=Hi[2,0]*Xs+Hi[2,1]*Ys+Hi[2,2]; un=(Hi[0,0]*Xs+Hi[0,1]*Ys+Hi[0,2])/den; vn=(Hi[1,0]*Xs+Hi[1,1]*Ys+Hi[1,2])/den
    ru=np.sqrt(un*un+vn*vn); rg=np.linspace(0,2.6,5000); rug=rg*(1+k1*rg*rg+k2*rg**4)
    rd=np.interp(ru,rug,rg); sc=np.divide(rd,ru,out=np.ones_like(ru),where=ru>1e-9)
    mapx=(cx+un*sc*s).astype(np.float32); mapy=(cy+vn*sc*s).astype(np.float32)
    warp=cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR,borderValue=(0,0,0))
    # germe maskesi (kaynak-koord yavaş değişen yer = smear) + görüntü-dışı
    st=np.abs(np.gradient(mapx,axis=1))+np.abs(np.gradient(mapx,axis=0))+np.abs(np.gradient(mapy,axis=1))+np.abs(np.gradient(mapy,axis=0))
    inb=(mapx>2)&(mapx<w-2)&(mapy>2)&(mapy<h-2); good=inb&(st>0.20)
    good=cv2.morphologyEx(good.astype(np.uint8),cv2.MORPH_OPEN,np.ones((5,5),np.uint8)).astype(bool)
    # şematik çim zemin (biçme şeritleri)
    out=np.zeros((WS,LS,3),np.uint8)
    for i in range(int(L/3)+1):
        x0=int((MG+i*3)*S); x1=int((MG+(i+1)*3)*S); out[int(MG*S):int((MG+Wp)*S),x0:x1]=(40,95,38) if i%2 else (46,110,44)
    out[good]=warp[good]
    # kanonik beyaz çizgiler (her zaman temiz)
    def P(X,Y): return (int((X+MG)*S),int((Y+MG)*S))
    wht=(245,245,245)
    cv2.rectangle(out,P(0,0),P(L,Wp),wht,2); cv2.line(out,P(L/2,0),P(L/2,Wp),wht,2)
    cv2.circle(out,P(L/2,Wp/2),int(R*S),wht,2)
    for gx in (0,L):
        BD,BHW=(BDN,BHWN) if gx==0 else (BDF,BHWF)
        bx=BD if gx==0 else L-BD; cv2.rectangle(out,P(min(gx,bx),Wp/2-BHW),P(max(gx,bx),Wp/2+BHW),wht,1)
        cv2.line(out,P(gx,Wp/2-1.5),P(gx,Wp/2+1.5),(0,0,235),4)
    cam=P(0,0) if camside=="SOL" else P(0,Wp)
    cv2.circle(out,cam,9,(0,200,255),-1); cv2.putText(out,"KAM",(cam[0]+8,cam[1]),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,200,255),2)
    return out

if __name__=="__main__":
    venues=[n for n in ORDER if n in C and not C[n].get("bad")]
    tiles=[]
    for n in venues:
        img=cv2.imread(os.path.join(LF,n)); r=C[n]
        if img is None or "H" not in r: continue
        H=np.array(r["H"],float); k1,k2=r["k1"],r["k2"]; cs=camside_of(img,k1,k2,H,J[n]["lines"])
        out=clean_topdown(img,k1,k2,H,cs,Wpf=r.get("Wp"),Rf=r.get("R_use"),bdn=r.get("box_depth_n"),bdf=r.get("box_depth_f"),bhwn=r.get("box_hw_n"),bhwf=r.get("box_hw_f"))
        cv2.putText(out,f"{n.split('.')[0][:16]}  res={r.get('fit',0):.2f}m",(8,out.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,230,255),2)
        tiles.append(out); print(f"  {n[:20]:20s} res={r.get('fit',0):.2f} {cs}",flush=True)
    th=max(t.shape[0] for t in tiles); tw=max(t.shape[1] for t in tiles)   # per-field Wp -> farklı boyut, uniform-pad
    tiles=[cv2.copyMakeBorder(t,0,th-t.shape[0],0,tw-t.shape[1],cv2.BORDER_CONSTANT,value=(18,18,18)) for t in tiles]
    cols=3; rows=(len(tiles)+cols-1)//cols
    sheet=np.full((rows*(th+6),cols*(tw+6),3),18,np.uint8)
    for i,t in enumerate(tiles):
        rr,cc=divmod(i,cols); sheet[rr*(th+6):rr*(th+6)+th,cc*(tw+6):cc*(tw+6)+tw]=t
    cv2.imwrite(os.path.join(HERE,"cand","_JOINT_CLEAN.jpg"),sheet); print(f"{len(tiles)} saha -> cand/_JOINT_CLEAN.jpg",flush=True)
