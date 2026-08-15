#!/usr/bin/env python3
"""Yeni-saha taze klip: filtre XHR API'den venue-adı eşleşen maç URL'si bul -> ffmpeg ile kısa klip çek.
Kullanım: python get_clip.py <venue_substr> <out.mp4> [ss=600] [dur=30]
"""
import os, sys, json, re, subprocess, datetime as dt, requests
H={'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/126 Safari/537.36',
   'Accept':'application/json, text/plain, */*','X-Requested-With':'XMLHttpRequest','Referer':'https://sosyalhalisaha.com/'}
def sget(url,xhr=True):
    h=dict(H)
    if not xhr: h.pop("X-Requested-With",None)
    return requests.get(url,headers=h,timeout=25).text
def video_urls(murl):
    """maç-detay sayfasından videoSrc array'i -> [cam url, ...]"""
    try: html=sget(murl,xhr=False)
    except Exception: return []
    mm=re.search(r"videoSrc\s*=\s*(\[.*?\]);",html,re.S)
    if not mm: return []
    try: return [a["url"] for a in json.loads(mm.group(1)) if a.get("url")]
    except Exception: return []
def find_url(sub):
    sub=sub.lower().replace("ı","i")
    base=dt.date(2026,6,30)
    for dd in range(15):   # 15-gün retention
        date=(base-dt.timedelta(days=dd)).strftime("%d.%m.%Y")
        url=f"https://sosyalhalisaha.com/xhr/filtre/___{date}__"; page=1
        while url and page<=6:
            try: d=json.loads(sget(url))
            except Exception: break
            if d.get("status")!="success": break
            for m in d.get("data",[]):
                v=((m.get("place") or {}).get("name","") or "").lower().replace("ı","i")
                if sub in v and m.get("url"):
                    vids=video_urls(m["url"])
                    if vids: return vids[0], (m.get("place") or {}).get("name",""), date
            url=d.get("nextPage"); page+=1
    return None,None,None
if __name__=="__main__":
    sub=sys.argv[1]; out=sys.argv[2]; ss=sys.argv[3] if len(sys.argv)>3 else "600"; dur=sys.argv[4] if len(sys.argv)>4 else "30"
    u,name,date=find_url(sub)
    if not u: print("URL YOK:",sub); sys.exit(1)
    print(f"BULUNDU: {name} {date}\n{u[:90]}")
    rc=["-reconnect","1","-reconnect_streamed","1","-reconnect_delay_max","5"]
    cmd=["ffmpeg","-y",*rc,"-ss",ss,"-i",u,"-t",dur,"-c","copy",out]
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=180)
    if not os.path.exists(out) or os.path.getsize(out)<10000:
        # copy başarısızsa re-encode
        cmd=["ffmpeg","-y",*rc,"-ss",ss,"-i",u,"-t",dur,"-an",out]
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=240)
    sz=os.path.getsize(out) if os.path.exists(out) else 0
    print(f"klip: {out} {sz/1e6:.1f}MB"); print(r.stderr[-300:] if sz<10000 else "OK")
