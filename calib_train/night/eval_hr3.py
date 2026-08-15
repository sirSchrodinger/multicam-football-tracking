#!/usr/bin/env python3
"""seg2_hr3 (RunPod fine-tune, pseudo-ground-line) vs seg2_hr2 (mevcut) DEĞERLENDİR.
Alperen sorusu: direk-takibi azaldı mı. + kaba kapsama (solve+sanity oranı)."""
import os, sys, json, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","")
HERE=os.path.dirname(os.path.abspath(__file__)); CT=os.path.dirname(HERE); ROOT=os.path.dirname(CT)
sys.path.insert(0, ROOT)
import torch
from calib_train.render2d_clean import UNet
from calib_train import auto_clean2d as AC2, auto_calib as AC
from calib_train.night.four_panel import p_warp, load as fload
from calib_train.pretty_render import _text, INK
MAN={r['idx']:r for r in json.load(open(os.path.join(HERE,"probcache_v3","manifest.json")))}
TW,TH=1024,576
def mk(path):
    n=UNet(); n.load_state_dict(torch.load(path,map_location="cpu")); n.eval(); return n
@torch.no_grad()
def prob(net,img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None]
    return torch.sigmoid(net(x))[0].numpy()

def goaln_overlay(img,pr,w,h,tag):
    o=img.copy(); m=cv2.resize(pr[0],(w,h))>0.4    # goalN kanalı
    o[m]=(0.3*o[m]+0.7*np.array([40,40,235])).astype(np.uint8)
    cv2.rectangle(o,(0,0),(w,34),(0,0,0),-1); _text(o,(10,8),tag,20,INK); return o

def main(idxs):
    n2=mk(f"{CT}/seg2_hr2.pth"); n3=mk(f"{CT}/seg2_hr3.pth")
    for idx in idxs:
        e=MAN[idx]; img=cv2.imread(f"{CT}/cand_big/{e['file']}"); w,h=int(np.load(f'{HERE}/probcache_v3/{idx:03d}.npz')['w']),int(np.load(f'{HERE}/probcache_v3/{idx:03d}.npz')['h'])
        img=cv2.resize(img,(w,h))
        _,_,_,_,_,feet=fload(idx)
        p2=prob(n2,img); p3=prob(n3,img)
        r2=AC2.calibrate_frame(p2,w,h,img=img,feet=feet); r3=AC2.calibrate_frame(p3,w,h,img=img,feet=feet)
        # goalN overlay + warp, hr2 üst / hr3 alt
        g2=goaln_overlay(img,p2,w,h,f"hr2 goalN  (kalib {'OK' if r2.get('ok') else 'YOK'})")
        g3=goaln_overlay(img,p3,w,h,f"hr3 goalN  (kalib {'OK' if r3.get('ok') else 'YOK'})")
        w2=p_warp(img,r2,w,h) if r2.get('ok') else np.full((300,500,3),30,np.uint8)
        w3=p_warp(img,r3,w,h) if r3.get('ok') else np.full((300,500,3),30,np.uint8)
        H=360
        row=lambda a,b:np.hstack([cv2.resize(a,(int(a.shape[1]*H/a.shape[0]),H)),np.full((H,6,3),80,np.uint8),cv2.resize(b,(int(b.shape[1]*H/b.shape[0]),H))])
        top=row(g2,g3); bot=row(w2,w3); W=max(top.shape[1],bot.shape[1])
        pad=lambda a:cv2.copyMakeBorder(a,0,0,0,W-a.shape[1],cv2.BORDER_CONSTANT,value=(20,20,20))
        out=np.vstack([pad(top),np.full((6,W,3),80,np.uint8),pad(bot)])
        cv2.imwrite(f"{CT}/cand/_HR3EVAL_{idx}.jpg",out)
        print(f"idx{idx}: hr2 src={r2.get('src')} fit={r2.get('fit',0):.2f} | hr3 src={r3.get('src')} fit={r3.get('fit',0):.2f} -> _HR3EVAL_{idx}.jpg")

if __name__=="__main__":
    main([int(x) for x in sys.argv[1:]] if len(sys.argv)>1 else [52,99])
