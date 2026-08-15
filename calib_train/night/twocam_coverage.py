#!/usr/bin/env python3
"""2-KAMERA KALİBRASYON KAPSAMASI: hipotez = tek kameradan bowtie (head-on) saha, ZIT kameradan kalibre olur.
Canlı listeden 2-kameralı venue'ler -> her iki kamerayı temporal-kalibre -> union-kapsama (≥1 kamera PASS)
tek-kameradan (cam0) yüksek mi? Bu, zor-saha kapsama sorununu 2-kameranın çözüp çözmediğini ölçer.
"""
import os, sys, json, re, subprocess, requests, time, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","0"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from calib_train.render2d_clean import UNet
from calib_train.auto_clean2d import sanity
from calib_train import auto_calib as AC
H={'User-Agent':'Mozilla/5.0','Accept':'*/*','X-Requested-With':'XMLHttpRequest','Referer':'https://sosyalhalisaha.com/'}
DATE=os.environ.get("DATE","29.06.2026"); NV=int(os.environ.get("NV","10"))
def sget(u,xhr=True):
    h=dict(H)
    if not xhr: h.pop("X-Requested-With",None)
    return requests.get(u,headers=h,timeout=15).text
def enum_live(date,limit=60):
    u=f"https://sosyalhalisaha.com/xhr/filtre/___{date}__"; pg=1; out=[]
    while u and pg<=25 and len(out)<limit:
        try: d=json.loads(sget(u))
        except: break
        if d.get("status")!="success": break
        for m in d.get("data",[]):
            if m.get("url"): out.append(((m.get("place") or {}).get("name","?"),m["url"]))
        u=d.get("nextPage"); pg+=1; time.sleep(0.08)
    return out
def vids(murl):
    for _ in range(2):
        try:
            html=sget(murl,xhr=False); mm=re.search(r"videoSrc\s*=\s*(\[.*?\]);",html,re.S)
            if mm: return [a["url"] for a in json.loads(mm.group(1)) if a.get("url")]
        except: pass
        time.sleep(0.3)
    return []
RC="-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5".split()
def grab(url,t,o):
    try: subprocess.run(["ffmpeg","-y",*RC,"-ss",str(t),"-i",url,"-frames:v","1","-q:v","3",o],capture_output=True,timeout=40)
    except: pass
    return os.path.exists(o) and os.path.getsize(o)>5000
TW,TH=1024,576; DEV="cuda" if torch.cuda.is_available() else "cpu"
net=UNet().to(DEV); net.load_state_dict(torch.load("calib_train/seg2_hr2.pth",map_location=DEV)); net.eval()
@torch.no_grad()
def prob(img):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None].to(DEV)
    return torch.sigmoid(net(x))[0].cpu().numpy()
def temporal_one(vurl,tag):
    td=f"/tmp/tc_{tag}"; os.makedirs(td,exist_ok=True); probs=[]; ref=None
    for i,t in enumerate(range(120,560,44)):  # 10 kare
        o=f"{td}/f{i}.jpg"
        if not grab(vurl,t,o): continue
        im=cv2.imread(o)
        if im is None: continue
        probs.append(prob(im))
        if ref is None: ref=im.copy()
    if len(probs)<4: return None
    med=np.median(np.stack(probs),0); h,w=ref.shape[:2]
    rec=AC.calib_from_pred(med,w,h)
    if not rec.get("ok"): return (False,None)
    ok,_=sanity(rec,w,h,img=ref); return (ok,round(float(rec["fit"]),3))
def main():
    venues=enum_live(DATE); print(f"{DATE}: {len(venues)} maç",flush=True)
    results=[]; done=0; OUT="calib_train/cand/_twocam_cov.json"
    for nm,mu in venues:
        if done>=NV: break
        vv=vids(mu)
        if len(vv)<2: continue   # SADECE 2-kameralı
        r0=temporal_one(vv[0],f"{abs(hash(nm))%9999}_0")
        r1=temporal_one(vv[1],f"{abs(hash(nm))%9999}_1")
        if r0 is None and r1 is None: continue
        c0=bool(r0 and r0[0]); c1=bool(r1 and r1[0]); union=c0 or c1
        rec={"venue":nm[:24],"cam0_pass":c0,"cam0_res":r0[1] if r0 else None,
             "cam1_pass":c1,"cam1_res":r1[1] if r1 else None,"union_pass":union}
        results.append(rec); done+=1; print(json.dumps(rec,ensure_ascii=False),flush=True)
        json.dump(results,open(OUT,"w"),ensure_ascii=False,indent=1)
    if results:
        c0=sum(1 for r in results if r["cam0_pass"]); un=sum(1 for r in results if r["union_pass"])
        n=len(results)
        print(f"\n=== 2-KAMERA: {n} venue | tek-kam(cam0) PASS {c0}/{n}=%{100*c0/n:.0f} | "
              f"2-kam UNION PASS {un}/{n}=%{100*un/n:.0f} | KAZANÇ +{un-c0} saha ===")
    else: print("\n=== 2-kameralı erişilebilir venue yok ===")
if __name__=="__main__": main()
