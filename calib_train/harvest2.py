#!/usr/bin/env python3
"""ÖLÇEKLİ harvest: çok-tarih + sayfalama -> benzersiz venue -> paralel ffmpeg frame (t=600) ->
skor -> cand_big/. "1000 saha" havuzu (keyfi-saha otomatik-kalibrasyon kapsamını ölçmek için).
ENV: MAX_VENUES(500) DAYS(120) WORKERS(16). Çıktı: cand_big/<slug>__<date>__s<score>.jpg + candidates_big.json
"""
import os, re, json, time, subprocess, datetime as dt
import requests, cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"cand_big"); os.makedirs(OUT,exist_ok=True)
H={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36',
   'Accept':'application/json, text/plain, */*','X-Requested-With':'XMLHttpRequest','Referer':'https://sosyalhalisaha.com/'}
MAX_VENUES=int(os.environ.get("MAX_VENUES","500")); DAYS=int(os.environ.get("DAYS","120")); WORKERS=int(os.environ.get("WORKERS","16")); SS=600
t0=time.time()
def log(m): print(f"[{(time.time()-t0)/60:.1f}dk] {m}",flush=True)
def slug(v): return "".join(c for c in (v or "x") if c.isalnum())[:22] or "x"
def sget(url,xhr=True,timeout=20):
    h=dict(H);
    if not xhr: h.pop("X-Requested-With",None)
    return requests.get(url,headers=h,timeout=timeout).text

def enumerate_venues():
    base=dt.date(2026,6,30); seen={};
    for dd in range(DAYS):
        date=(base-dt.timedelta(days=dd)).strftime("%d.%m.%Y")
        url=f"https://sosyalhalisaha.com/xhr/filtre/___{date}__"; page=1
        while url and page<=6:
            try: d=json.loads(sget(url,xhr=True))
            except Exception: break
            if d.get("status")!="success": break
            for m in d.get("data",[]):
                v=(m.get("place") or {}).get("name","?"); sl=slug(v)
                if sl not in seen and m.get("url"): seen[sl]=(v,date,m["url"])
            url=d.get("nextPage"); page+=1
            time.sleep(0.05)
        if dd%10==0: log(f"{date}: benzersiz venue={len(seen)}")
        if len(seen)>=MAX_VENUES: break
    return list(seen.values())[:MAX_VENUES]

def pull(url,out):
    rc=["-reconnect","1","-reconnect_streamed","1","-reconnect_delay_max","5"]
    for cmd in (["ffmpeg","-y",*rc,"-ss",str(SS),"-i",url,"-frames:v","1","-q:v","3",out],
                ["ffmpeg","-y",*rc,"-i",url,"-ss",str(SS),"-frames:v","1","-q:v","3",out]):
        try: subprocess.run(cmd,capture_output=True,timeout=90)
        except Exception: pass
        if os.path.exists(out) and os.path.getsize(out)>5000: return True
    return False
def score(path):
    im=cv2.imread(path)
    if im is None: return None
    hsv=cv2.cvtColor(im,cv2.COLOR_BGR2HSV); Hh,Ss,Vv=hsv[...,0],hsv[...,1],hsv[...,2]
    bright=float(Vv.mean()); sat=float(Ss.mean()); green=float(((Hh>=30)&(Hh<=92)&(Ss>40)&(Vv>40)).mean())
    blown=float((Vv>245).mean()); dark=float((Vv<40).mean())
    th=cv2.morphologyEx(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),cv2.MORPH_TOPHAT,cv2.getStructuringElement(cv2.MORPH_RECT,(17,17)))
    line=float((th>40).mean())
    comp=(0.30*(1-min(1,abs(bright-140)/120))+0.18*min(1,sat/55)+0.25*min(1,green/0.40)+0.17*min(1,line/0.06)+0.10)*max(0,1-min(1,blown*5+dark*1.5))
    return dict(score=round(comp,3),bright=round(bright,1),green=round(green,3),dark=round(dark,3))
def work(args):
    v,date,murl=args; sl=slug(v)
    try: html=sget(murl,xhr=False)
    except Exception: return None
    mm=re.search(r"videoSrc\s*=\s*(\[.*?\]);",html,re.S)
    if not mm: return None
    try: arr=json.loads(mm.group(1)); url=arr[0]["url"]
    except Exception: return None
    tmp=os.path.join(OUT,f"_tmp_{sl}_{date.replace('.','')}.jpg")
    if not pull(url,tmp): return None
    sc=score(tmp)
    if sc is None:
        try: os.remove(tmp)
        except: pass
        return None
    final=os.path.join(OUT,f"{sl}__{date.replace('.','')}__s{int(sc['score']*1000):03d}.jpg"); os.replace(tmp,final)
    return dict(venue=v,date=date,file=os.path.basename(final),**sc)

def main():
    log("venue enumerate...")
    cands=enumerate_venues(); log(f"{len(cands)} benzersiz venue -> frame çek (paralel {WORKERS})")
    recs=[]
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i,f in enumerate(as_completed([ex.submit(work,c) for c in cands])):
            r=f.result()
            if r: recs.append(r)
            if i%25==0: log(f"  ilerleme {i}/{len(cands)} başarılı={len(recs)}")
    ranked=sorted(recs,key=lambda r:-r["score"])
    json.dump({"n":len(recs),"venues":ranked},open(os.path.join(HERE,"candidates_big.json"),"w"),ensure_ascii=False,indent=1)
    log(f"BİTTİ — {len(recs)} kare çekildi / {len(cands)} venue. iyi(>0.3)={sum(1 for r in recs if r['score']>0.3)}")
if __name__=="__main__": main()
