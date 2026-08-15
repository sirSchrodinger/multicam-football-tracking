#!/usr/bin/env python3
"""ROL-AWARE otonom 2D: seg2 model her çizginin ROLÜNÜ verir (goalN/goalF/touchN/touchF) ->
geometrik tahmin YOK. Her rolün noktalarına çizgi -> köşe -> joint kalibrasyon (ortak-lens
+homografi) -> kamera-tarafı (cross-kuralı, roller doğru olduğu için doğru) -> 2D (correct_2d stili).
res>0.6m ise BAŞARISIZ damgalar (çöp gösterme). Kullanım: python infer_auto_roles.py img...
"""
import sys, os, numpy as np, cv2, torch, torch.nn as nn
from scipy.optimize import least_squares
HERE=os.path.dirname(os.path.abspath(__file__)); K1S,K2S=0.167,0.240; TW,TH=512,288; NC=7; L,S,M=34.0,26,5.0
CK=os.path.join(HERE,"seg2_ckpt","seg2_unet.pth")
def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True),nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True))
class UNet(nn.Module):
    def __init__(s,nin=3,no=NC,b=24):
        super().__init__(); s.d1=cbr(nin,b);s.d2=cbr(b,b*2);s.d3=cbr(b*2,b*4);s.d4=cbr(b*4,b*8);s.p=nn.MaxPool2d(2)
        s.u3=nn.ConvTranspose2d(b*8,b*4,2,2);s.c3=cbr(b*8,b*4);s.u2=nn.ConvTranspose2d(b*4,b*2,2,2);s.c2=cbr(b*4,b*2)
        s.u1=nn.ConvTranspose2d(b*2,b,2,2);s.c1=cbr(b*2,b);s.h=nn.Conv2d(b,no,1)
    def forward(s,x):
        c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2));c4=s.d4(s.p(c3))
        x=s.c3(torch.cat([s.u3(c4),c3],1));x=s.c2(torch.cat([s.u2(x),c2],1));x=s.c1(torch.cat([s.u1(x),c1],1));return s.h(x)
dev="cuda" if torch.cuda.is_available() else "cpu"
net=UNet().to(dev); net.load_state_dict(torch.load(CK,map_location=dev)); net.eval()
def und_pix(P,k1,k2,cx,cy,s,z=1.5): u=(P[:,0]-cx)/s;v=(P[:,1]-cy)/s;r2=u*u+v*v;f=1+k1*r2+k2*r2*r2;return np.stack([cx+u*f/z*s,cy+v*f/z*s],1)
def und_norm(P,k1,k2,cx,cy,s): u=(P[:,0]-cx)/s;v=(P[:,1]-cy)/s;r2=u*u+v*v;f=1+k1*r2+k2*r2*r2;return np.stack([u*f,v*f],1)
def undimg(img,k1,k2,cx,cy,s,z=1.5):
    h,w=img.shape[:2]; rd=np.linspace(0,2.4,2200); rru=rd*(1+k1*rd*rd+k2*rd**4)
    ys,xs=np.mgrid[0:h,0:w].astype(np.float32); uo=(xs-cx)/s*z; vo=(ys-cy)/s*z; ro=np.sqrt(uo*uo+vo*vo)
    rdv=np.interp(ro,rru,rd); sc=np.divide(rdv,ro,out=np.ones_like(ro),where=ro>1e-6)
    return cv2.remap(img,(cx+uo*sc*s).astype(np.float32),(cy+vo*sc*s).astype(np.float32),cv2.INTER_LINEAR)
def aH(uv,H): z=H[2,0]*uv[:,0]+H[2,1]*uv[:,1]+H[2,2];return np.stack([(H[0,0]*uv[:,0]+H[0,1]*uv[:,1]+H[0,2])/z,(H[1,0]*uv[:,0]+H[1,1]*uv[:,1]+H[1,2])/z],1)
def fitL(P):  # gürültüye-dayanıklı (Huber) çizgi fit -> (a,b,c) a x+b y+c=0
    vx,vy,x0,y0=cv2.fitLine(np.asarray(P,np.float32),cv2.DIST_HUBER,0,0.01,0.01).ravel()
    a,b=-vy,vx; return np.array([a,b,-(a*x0+b*y0)])
def inter(a,b):D=a[0]*b[1]-b[0]*a[1];return np.array([(a[1]*b[2]-b[1]*a[2])/D,(b[0]*a[2]-a[0]*b[2])/D])

def run(path):
    img=cv2.imread(path)
    if img is None: return None
    h,w=img.shape[:2]; cx,cy,s=w/2,h/2,w/2
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None].to(dev)
    with torch.no_grad(): p=torch.sigmoid(net(x))[0].cpu().numpy()
    Wp=18.0; LS,WS=int((L+2*M)*S),int((Wp+2*M)*S)
    def fail(msg):
        print(f"  {os.path.basename(path)[:24]:24s} FAIL: {msg}",flush=True)
        t=np.full((WS,LS,3),20,np.uint8); cv2.putText(t,msg,(20,WS//2),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2)
        cv2.putText(t,os.path.basename(path).split('__')[0][:20],(8,WS-12),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        return cv2.resize(t,(int(LS*620/WS),620))
    # her rolün ham noktaları (model DOĞRUDAN veriyor)
    rs=np.random.RandomState(0); groups={}
    for ci,role in enumerate(["goalN","goalF","touchN","touchF"]):
        pc=cv2.resize(p[ci],(w,h)); mk=(pc>0.5).astype(np.uint8)
        mk=cv2.morphologyEx(mk,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8))
        ncc,lbl,st,_=cv2.connectedComponentsWithStats(mk)
        if ncc>1: mk=(lbl==(1+np.argmax(st[1:,cv2.CC_STAT_AREA]))).astype(np.uint8)   # en büyük bileşen (aykırı blob at)
        pts=np.column_stack(np.where(mk>0))[:,::-1].astype(float)
        if len(pts)<25: return fail(f"{role} bulunamadi")
        if len(pts)>180: pts=pts[rs.choice(len(pts),180,replace=False)]
        groups[role]=pts
    CON=[("goalN","X",0.0),("goalF","X",L),("touchN","Y",0.0),("touchF","Y",Wp)]
    # init H: k=shared köşeler
    def cor(k1,k2):
        fl={r:fitL(und_pix(groups[r],k1,k2,cx,cy,s)) for r in groups}
        return np.array([inter(fl["goalN"],fl["touchN"]),inter(fl["goalN"],fl["touchF"]),inter(fl["goalF"],fl["touchN"]),inter(fl["goalF"],fl["touchF"])])
    c0=cor(K1S,K2S); H0,_=cv2.findHomography(c0,np.array([[0,0],[0,Wp],[L,0],[L,Wp]],float)); H0=H0/H0[2,2]
    p0=[K1S,K2S,*H0.ravel()[:8]]
    def resid(pp):
        k1,k2=pp[0],pp[1];H=np.array([[pp[2],pp[3],pp[4]],[pp[5],pp[6],pp[7]],[pp[8],pp[9],1.0]]);out=[]
        for r,kind,tg in CON:
            m=aH(und_norm(groups[r],k1,k2,cx,cy,s),H); out.append((m[:,0]-tg) if kind=="X" else (m[:,1]-tg))
        return np.concatenate(out)
    sol=least_squares(resid,p0,method="trf",bounds=([0,0]+[-np.inf]*8,[0.45,0.7]+[np.inf]*8),max_nfev=4000)
    k1,k2=sol.x[0],sol.x[1];H=np.array([[sol.x[2],sol.x[3],sol.x[4]],[sol.x[5],sol.x[6],sol.x[7]],[sol.x[8],sol.x[9],1.0]])
    fit=np.sqrt((resid(sol.x)**2).mean())
    if fit>0.6: return fail(f"calib zayif (res={fit:.2f}m)")
    # köşeler (rafine k) + kamera tarafı (cross, roller DOĞRU)
    def u2pix(P): un=und_norm(P,k1,k2,cx,cy,s); return np.stack([cx+un[:,0]/1.5*s,cy+un[:,1]/1.5*s],1)
    fl={r:fitL(u2pix(groups[r])) for r in groups}
    c_nN=inter(fl["goalN"],fl["touchN"]);c_nF=inter(fl["goalN"],fl["touchF"]);c_fN=inter(fl["goalF"],fl["touchN"]);c_fF=inter(fl["goalF"],fl["touchF"])
    A=c_fN-c_nN;B=c_nF-c_nN;cross=A[0]*B[1]-A[1]*B[0]; mirrored=(cross>0); camside="SAG-alt" if mirrored else "SOL-alt"
    uimg=undimg(img,k1,k2,cx,cy,s)
    def MXl(X,Y): return [(M+X)*S, WS-(M+Y)*S]
    def MXr(X,Y): return [(M+L-X)*S, WS-(M+Y)*S]
    MX=MXr if mirrored else MXl
    src=np.array([c_nN,c_nF,c_fN,c_fF],float); dst=np.array([MX(0,0),MX(0,Wp),MX(L,0),MX(L,Wp)],float)
    Hm,_=cv2.findHomography(src,dst); top=cv2.warpPerspective(uimg,Hm,(LS,WS),borderValue=(0,0,0))
    top[top.sum(2)<10]=(22,22,22)
    cv2.rectangle(top,tuple(map(int,MX(0,Wp))),tuple(map(int,MX(L,0))),(0,215,255),2)
    cv2.line(top,tuple(map(int,MX(L/2,0))),tuple(map(int,MX(L/2,Wp))),(0,215,255),1)
    cv2.circle(top,tuple(map(int,MX(L/2,Wp/2))),int(3*S),(150,120,40),1,cv2.LINE_AA)
    cam=MX(0,0); cv2.circle(top,tuple(map(int,cam)),11,(0,200,255),-1); cv2.putText(top,"KAMERA",(int(cam[0])+8,int(cam[1])-6),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,200,255),2)
    cv2.putText(top,f"2D kam={camside} res{fit:.2f}m",(8,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,230,255),2)
    cv2.putText(top,os.path.basename(path).split('__')[0][:20],(8,WS-12),cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
    print(f"  {os.path.basename(path)[:24]:24s} OK   res={fit:.3f}m kam={camside}",flush=True)
    return cv2.resize(top,(int(LS*620/WS),620))

rows=[run(p) for p in sys.argv[1:]]; rows=[r for r in rows if r is not None]
print("="*48,flush=True)
wm=max(r.shape[1] for r in rows); sheet=np.full((sum(r.shape[0]+6 for r in rows),wm,3),20,np.uint8); y=0
for r in rows: sheet[y:y+r.shape[0],0:r.shape[1]]=r; y+=r.shape[0]+6
cv2.imwrite(os.path.join(HERE,"cand","_auto_roles_2d.jpg"),sheet); print("-> cand/_auto_roles_2d.jpg")
