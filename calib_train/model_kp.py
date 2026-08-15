#!/usr/bin/env python3
"""Rol-aware KEYPOINT modeli: RGB -> K ısı-haritası + K belirsizlik(σ). DSNT soft-argmax ile
alt-piksel decode. Kalın-maske yerine seyrek keypoint+güven -> kalibrasyon hassasiyeti.
K=7 keypoint (rol = hangi heatmap): c_nN c_nF c_fN c_fF (4 köşe) + center_n center_f + circle_c.
Metrik konumları sabit: köşeler (0,0)(0,W)(L,0)(L,W); orta (L/2,0)(L/2,W); yuvarlak (L/2,W/2).
σ = öğrenilen güven (off-frame/görünmez keypoint düşük σ -> LM down-weight eder).
"""
import torch, torch.nn as nn, torch.nn.functional as F
KP_NAMES=["c_nN","c_nF","c_fN","c_fF","center_n","center_f","circle_c"]
NK=len(KP_NAMES)
def kp_metric(L,W):
    import numpy as np
    return np.array([[0,0],[0,W],[L,0],[L,W],[L/2,0],[L/2,W],[L/2,W/2]],float)

def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True),
                                   nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True))
class KeypointUNet(nn.Module):
    def __init__(s,nin=3,nk=NK,b=24):
        super().__init__(); s.d1=cbr(nin,b);s.d2=cbr(b,b*2);s.d3=cbr(b*2,b*4);s.d4=cbr(b*4,b*8);s.p=nn.MaxPool2d(2)
        s.u3=nn.ConvTranspose2d(b*8,b*4,2,2);s.c3=cbr(b*8,b*4);s.u2=nn.ConvTranspose2d(b*4,b*2,2,2);s.c2=cbr(b*4,b*2)
        s.u1=nn.ConvTranspose2d(b*2,b,2,2);s.c1=cbr(b*2,b);s.head=nn.Conv2d(b,nk,1)
        s.sig=nn.Sequential(nn.AdaptiveAvgPool2d(1),nn.Flatten(),nn.Linear(b*8,nk))  # σ kafası (bottleneck'ten)
    def forward(s,x):
        c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2));c4=s.d4(s.p(c3))
        sig=torch.sigmoid(s.sig(c4))                    # (B,NK) güven
        u=s.c3(torch.cat([s.u3(c4),c3],1));u=s.c2(torch.cat([s.u2(u),c2],1));u=s.c1(torch.cat([s.u1(u),c1],1))
        hm=s.head(u)                                    # (B,NK,H,W) logit
        return hm, sig

def dsnt(hm,temp=1.0):
    """ısı-haritası -> alt-piksel (x,y) [0,1] (diferansiyellenebilir) + konsantrasyon-güveni."""
    B,K,H,W=hm.shape
    p=F.softmax(hm.reshape(B,K,-1)*temp,dim=-1).reshape(B,K,H,W)
    xs=torch.linspace(0,1,W,device=hm.device); ys=torch.linspace(0,1,H,device=hm.device)
    px=p.sum(2); py=p.sum(3)                            # marjinaller
    ex=(px*xs).sum(-1); ey=(py*ys).sum(-1)             # (B,K) beklenen konum
    # konsantrasyon (düşük varyans=güvenli) -> ekstra güven sinyali
    vx=(px*(xs-ex.unsqueeze(-1))**2).sum(-1); vy=(py*(ys-ey.unsqueeze(-1))**2).sum(-1)
    conc=torch.exp(-(vx+vy)*8.0)                        # (B,K) [0,1]
    return torch.stack([ex,ey],-1), conc               # (B,K,2), (B,K)

def dsnt_win(hm,r=7):
    """Pencereli alt-piksel: argmax tepesi etrafında r-yarıçaplı softmax (dağınık-kuyruk
    merkez-yanlılığını yok eder). Çıkarım için (diferansiyellenemez argmax adımı)."""
    B,K,H,W=hm.shape; out=torch.zeros(B,K,2,device=hm.device); peak=torch.zeros(B,K,device=hm.device)
    flat=hm.reshape(B,K,-1); idx=flat.argmax(-1); py0=(idx//W); px0=(idx%W)
    pk=torch.sigmoid(flat.max(-1).values)              # tepe güveni
    for b in range(B):
        for k in range(K):
            y0=int(py0[b,k]); x0=int(px0[b,k])
            y1,y2=max(0,y0-r),min(H,y0+r+1); x1,x2=max(0,x0-r),min(W,x0+r+1)
            patch=hm[b,k,y1:y2,x1:x2]; w=F.softmax(patch.reshape(-1),dim=0).reshape(patch.shape)
            yy=torch.arange(y1,y2,device=hm.device).float(); xx=torch.arange(x1,x2,device=hm.device).float()
            ey=(w.sum(1)*yy).sum(); ex=(w.sum(0)*xx).sum()
            out[b,k,0]=ex/(W-1); out[b,k,1]=ey/(H-1); peak[b,k]=pk[b,k]
    return out,peak

if __name__=="__main__":
    net=KeypointUNet(); x=torch.randn(2,3,288,512)
    hm,sig=net(x); kp,conc=dsnt(hm)
    print("hm",tuple(hm.shape),"sig",tuple(sig.shape),"kp",tuple(kp.shape),"conc",tuple(conc.shape))
    print("KP_NAMES",KP_NAMES,"params",sum(p.numel() for p in net.parameters())//1000,"K")
