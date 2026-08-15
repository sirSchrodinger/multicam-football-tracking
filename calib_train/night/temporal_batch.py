#!/usr/bin/env python3
"""ÇOK-SAHA TEMPORAL YIELD (dürüst genelleme sayısı): N yeni saha -> her birinde tek-kare vs temporal calib.
"Temporal yeni sahaların %kaçında PASS + ne kadar iyileştiriyor" = deployment-ilgili metrik.
Video erişimi ARALIKLI -> 0-kare venue atlanır. Incremental yazar: cand/_temporal_batch.json + log.
"""
import os, sys, json, re, subprocess, requests, numpy as np, cv2
os.environ.setdefault("CUDA_VISIBLE_DEVICES","0"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from calib_train.render2d_clean import UNet
from calib_train.auto_clean2d import sanity
from calib_train import auto_calib as AC
H={'User-Agent':'Mozilla/5.0','Accept':'*/*','X-Requested-With':'XMLHttpRequest','Referer':'https://sosyalhalisaha.com/'}
def sget(u,xhr=True):
    h=dict(H)
    if not xhr: h.pop("X-Requested-With",None)
    return requests.get(u,headers=h,timeout=20).text
def url_on_date(sub,date):
    import time as _t
    norm=lambda s:(s or "").lower().replace("ı","i").replace("İ","i").replace(" ","")   # BOŞLUK-DUYARSIZ (slug vs spaced-ad)
    sub=norm(sub); u=f"https://sosyalhalisaha.com/xhr/filtre/___{date}__"; pg=1
    cand_urls=[]
    while u and pg<=20:                          # 150+ maç = 15+ sayfa -> pg limiti yükseltildi
        try: d=json.loads(sget(u))
        except: break
        if d.get("status")!="success": break
        for m in d.get("data",[]):
            v=norm((m.get("place") or {}).get("name",""))
            if sub[:14] in v and m.get("url"): cand_urls.append(m["url"])   # ilk 14 char (slug kırpılmış olabilir)
        u=d.get("nextPage"); pg+=1; _t.sleep(0.08)                     # politeness
    for mu in cand_urls:                          # videoSrc extraction (retry'lı, eşleşenler arası)
        for _ in range(2):
            try:
                html=sget(mu,xhr=False); mm=re.search(r"videoSrc\s*=\s*(\[.*?\]);",html,re.S)
                if mm:
                    uu=[a["url"] for a in json.loads(mm.group(1)) if a.get("url")]
                    if uu: return uu
            except: pass
            _t.sleep(0.3)
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
def test_venue(sub,date):
    vv=url_on_date(sub,date)
    if not vv: return {"venue":sub,"status":"url-yok"}
    td=f"/tmp/tb_{sub[:8]}"; os.makedirs(td,exist_ok=True)
    probs=[]; ref=None; sres=[]; spass=0; ns=0
    for i,t in enumerate(range(120,640,32)):  # 17 kare
        o=f"{td}/f{i}.jpg"
        if not grab(vv[0],t,o): continue
        im=cv2.imread(o)
        if im is None: continue
        ns+=1; pr=prob(im); probs.append(pr)
        if ref is None: ref=im.copy()
        r=AC.calib_from_pred(pr,im.shape[1],im.shape[0])
        if r.get("ok"):
            ok,_=sanity(r,im.shape[1],im.shape[0],img=im)
            if r["fit"]: sres.append(r["fit"])
            if ok: spass+=1
    if ns<6: return {"venue":sub,"status":f"kare-az({ns})"}
    med=np.median(np.stack(probs),0); h,w=ref.shape[:2]
    rm=AC.calib_from_pred(med,w,h); okm,whym=(sanity(rm,w,h,img=ref) if rm.get("ok") else (False,"calib-yok"))
    return {"venue":sub,"date":date,"status":"ok","n_frames":ns,
            "single_pass":spass,"single_res_avg":round(float(np.mean(sres)),3) if sres else None,
            "temporal_pass":bool(okm),"temporal_res":round(float(rm["fit"]),3) if rm.get("ok") else None,"temporal_why":whym if not okm else "PASS"}
def main():
    vlist=json.load(open("/tmp/venue_list.json"))
    results=[]; OUT="calib_train/cand/_temporal_batch.json"
    for sub,date,score in vlist:
        try: r=test_venue(sub,date); r["score"]=score
        except Exception as e: r={"venue":sub,"status":f"err:{str(e)[:40]}"}
        results.append(r); print(json.dumps(r,ensure_ascii=False),flush=True)
        json.dump(results,open(OUT,"w"),ensure_ascii=False,indent=1)  # incremental
    ok=[r for r in results if r.get("status")=="ok"]
    tp=sum(1 for r in ok if r["temporal_pass"]); sp=sum(r["single_pass"] for r in ok); sn=sum(r["n_frames"] for r in ok)
    print(f"\n=== ÖZET: {len(ok)} erişilebilir saha | TEMPORAL-PASS {tp}/{len(ok)} = %{100*tp/max(1,len(ok)):.0f} | "
          f"tek-kare-PASS oranı %{100*sp/max(1,sn):.0f} ===")
if __name__=="__main__": main()
