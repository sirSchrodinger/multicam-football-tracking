#!/usr/bin/env python3
"""TEMPORAL YIELD — CANLI listeden (slug-eşleştirme YOK, eskime YOK). 29.06 filtre listesini gez ->
videoSrc'li ilk N venue -> her birinde 12 ayrı-zaman kare -> tek-kare vs temporal calib. Dürüst yield.
Incremental: cand/_temporal_live.json.
"""
import os, sys, json, re, subprocess, requests, time, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","0"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from calib_train.render2d_clean import UNet
from calib_train.auto_clean2d import sanity
from calib_train import auto_calib as AC
H={'User-Agent':'Mozilla/5.0','Accept':'*/*','X-Requested-With':'XMLHttpRequest','Referer':'https://sosyalhalisaha.com/'}
DATE=os.environ.get("DATE","29.06.2026"); NV=int(os.environ.get("NV","12"))
def sget(u,xhr=True):
    h=dict(H)
    if not xhr: h.pop("X-Requested-With",None)
    return requests.get(u,headers=h,timeout=15).text
def enumerate_live(date,limit=40):
    u=f"https://sosyalhalisaha.com/xhr/filtre/___{date}__"; pg=1; out=[]
    while u and pg<=25 and len(out)<limit:
        try: d=json.loads(sget(u))
        except: break
        if d.get("status")!="success": break
        for m in d.get("data",[]):
            nm=(m.get("place") or {}).get("name","?"); mu=m.get("url")
            if mu: out.append((nm,mu))
        u=d.get("nextPage"); pg+=1; time.sleep(0.08)
    return out
def videosrc(murl):
    for _ in range(2):
        try:
            html=sget(murl,xhr=False); mm=re.search(r"videoSrc\s*=\s*(\[.*?\]);",html,re.S)
            if mm:
                uu=[a["url"] for a in json.loads(mm.group(1)) if a.get("url")]
                if uu: return uu
        except: pass
        time.sleep(0.3)
RC="-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5".split()
def grab(url,t,o):
    try: subprocess.run(["ffmpeg","-y",*RC,"-ss",str(t),"-i",url,"-frames:v","1","-q:v","3",o],capture_output=True,timeout=45)
    except: pass
    return os.path.exists(o) and os.path.getsize(o)>5000
TW,TH=1024,576; DEV="cuda" if torch.cuda.is_available() else "cpu"
net=UNet().to(DEV); net.load_state_dict(torch.load("calib_train/seg2_hr2.pth",map_location=DEV)); net.eval()
@torch.no_grad()
def prob(img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None].to(DEV)
    return torch.sigmoid(net(x))[0].cpu().numpy()
def test(name,vurl):
    td=f"/tmp/tl_{abs(hash(name))%99999}"; os.makedirs(td,exist_ok=True)
    probs=[]; ref=None; sres=[]; spass=0; ns=0
    for i,t in enumerate(range(120,600,40)):  # 12 kare
        o=f"{td}/f{i}.jpg"
        if not grab(vurl,t,o): continue
        im=cv2.imread(o)
        if im is None: continue
        ns+=1; pr=prob(im); probs.append(pr)
        if ref is None: ref=im.copy()
        r=AC.calib_from_pred(pr,im.shape[1],im.shape[0])
        if r.get("ok"):
            ok,_=sanity(r,im.shape[1],im.shape[0],img=im)
            if r["fit"]: sres.append(r["fit"])
            if ok: spass+=1
    if ns<5: return {"venue":name[:24],"status":f"kare-az({ns})"}
    med=np.median(np.stack(probs),0); h,w=ref.shape[:2]
    rm=AC.calib_from_pred(med,w,h); okm,whym=(sanity(rm,w,h,img=ref) if rm.get("ok") else (False,"calib-yok"))
    return {"venue":name[:24],"status":"ok","n":ns,"single_pass":spass,
            "single_res":round(float(np.mean(sres)),3) if sres else None,
            "temporal_pass":bool(okm),"temporal_res":round(float(rm["fit"]),3) if rm.get("ok") else None,"why":whym}
def main():
    venues=enumerate_live(DATE); print(f"{DATE}: {len(venues)} maç listelendi",flush=True)
    results=[]; OUT="calib_train/cand/_temporal_live.json"; done=0
    for nm,mu in venues:
        if done>=NV: break
        vv=videosrc(mu)
        if not vv: continue
        try: r=test(nm,vv[0])
        except Exception as e: r={"venue":nm[:24],"status":f"err:{str(e)[:30]}"}
        if r.get("status")=="ok": done+=1
        results.append(r); print(json.dumps(r,ensure_ascii=False),flush=True)
        json.dump(results,open(OUT,"w"),ensure_ascii=False,indent=1)
    ok=[r for r in results if r.get("status")=="ok"]
    if ok:
        tp=sum(1 for r in ok if r["temporal_pass"]); sp=sum(r["single_pass"] for r in ok); sn=sum(r["n"] for r in ok)
        print(f"\n=== TEMPORAL YIELD: {tp}/{len(ok)} saha PASS = %{100*tp/len(ok):.0f} | tek-kare kare-PASS %{100*sp/max(1,sn):.0f} ===")
    else: print("\n=== erişilebilir saha YOK ===")
if __name__=="__main__": main()
