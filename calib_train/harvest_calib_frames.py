#!/usr/bin/env python3
"""Calib için İYİ-IŞIKLI aday kare topla + gözle aynı kriterlere göre skorla.
Farklı tarihlerden tesis maçlarını enumerate eder, her birinden t=600s'te 1 kare
çeker, parlaklık/renk/yeşil-çim/çizgi-netliğine göre skorlar. Sporland-tipi IR-gri
(sat~0) ve Atakum-tipi güneş-patlaması (blown yüksek) DÜŞÜK skor alır -> elenir.

Çıktı: calib_train/cand/<venue>__<date>__s<score>.jpg  + cand/candidates.json
SONRA: en iyi kareleri GÖZLE seçip label_frames'e koy + frames.json + rebuild.
"""
import sys, os, json, subprocess, time, re
sys.path.insert(0, "."); sys.path.insert(0, "eval")
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2, numpy as np
from overnight_runner import _sget

HERE = os.path.dirname(os.path.abspath(__file__))
CAND = os.path.join(HERE, "cand"); os.makedirs(CAND, exist_ok=True)
DATES = [f"{d:02d}.06.2026" for d in range(29, 0, -1)]   # 29.06 .. 01.06 (geniş, çeşitli saha)
SS = 600           # maç içine atla (orta-oyun, ışık yerleşik)
MAX_PULLS = int(os.environ.get("MAX_PULLS", "130"))
t0 = time.time()
def log(m): print(f"[{(time.time()-t0)/60:.1f}dk] {m}", flush=True)
def slug(v): return "".join(c for c in v if c.isalnum())[:20] or "x"

def enumerate_candidates():
    cands = []  # (venue, date, url)
    seen_pairs = set()
    for date in DATES:
        try:
            d = json.loads(_sget(f"https://sosyalhalisaha.com/xhr/filtre/___{date}__", xhr=True))
        except Exception as e:
            log(f"{date}: liste hata {e}"); continue
        if d.get("status") != "success": continue
        for m in d.get("data", []):
            venue = (m.get("place") or {}).get("name", "?")
            try:
                html = _sget(m["url"])
                mm = re.search(r"videoSrc\s*=\s*(\[.*?\]);", html, re.S)
                if not mm: continue
                arr = json.loads(mm.group(1))
                if not arr: continue
                url = arr[0]["url"]
            except Exception:
                continue
            kp = (slug(venue), date)
            if kp in seen_pairs: continue
            seen_pairs.add(kp)
            cands.append((venue, date, url))
            time.sleep(0.15)
        log(f"{date}: toplam aday={len(cands)}")
        if len(cands) >= MAX_PULLS: break
    return cands[:MAX_PULLS]

def pull_frame(url, out):
    rc = ["-reconnect","1","-reconnect_streamed","1","-reconnect_delay_max","5"]
    for cmd in (["ffmpeg","-y",*rc,"-ss",str(SS),"-i",url,"-frames:v","1","-q:v","3",out],
                ["ffmpeg","-y",*rc,"-i",url,"-ss",str(SS),"-frames:v","1","-q:v","3",out]):
        try:
            subprocess.run(cmd, capture_output=True, timeout=90)
        except Exception:
            pass
        if os.path.exists(out) and os.path.getsize(out) > 5000:
            return True
    return False

def score_frame(path):
    im = cv2.imread(path)
    if im is None: return None
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV); H,S,V = hsv[...,0],hsv[...,1],hsv[...,2]
    bright = float(V.mean()); sat = float(S.mean())
    green = float(((H>=30)&(H<=92)&(S>40)&(V>40)).mean())
    blown = float((V>245).mean()); dark = float((V<40).mean())
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT,(17,17)))
    line = float((th>40).mean())
    b = 1.0 - min(1.0, abs(bright-140)/120)       # parlaklık 140 civarı ideal
    s = min(1.0, sat/55)                           # renkli mi (IR-gri -> 0)
    g = min(1.0, green/0.40)                        # yeşil-çim görünür mü
    l = min(1.0, line/0.06)                         # çizgi-net mi
    pen = max(0.0, 1.0 - min(1.0, blown*5 + dark*1.5))  # güneş-patlama / karanlık cezası
    comp = (0.30*b + 0.18*s + 0.25*g + 0.17*l + 0.10) * pen
    return dict(score=round(comp,3), bright=round(bright,1), sat=round(sat,1),
                green=round(green,3), line=round(line,3), blown=round(blown,4), dark=round(dark,3))

def work(args):
    venue, date, url = args
    tmp = os.path.join(CAND, f"_tmp_{slug(venue)}_{date.replace('.','')}.jpg")
    if not pull_frame(url, tmp): return None
    sc = score_frame(tmp)
    if sc is None:
        try: os.remove(tmp)
        except: pass
        return None
    final = os.path.join(CAND, f"{slug(venue)}__{date.replace('.','')}__s{int(sc['score']*1000):03d}.jpg")
    os.replace(tmp, final)
    rec = dict(venue=venue, date=date, file=os.path.basename(final), **sc)
    log(f"  {slug(venue):20s} {date} score={sc['score']:.3f} br={sc['bright']:.0f} sat={sc['sat']:.0f} grn={sc['green']:.2f} blown={sc['blown']:.3f}")
    return rec

def main():
    log("aday enumerate ediliyor...")
    cands = enumerate_candidates()
    log(f"{len(cands)} aday çekilecek (paralel)")
    recs = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for f in as_completed([ex.submit(work, c) for c in cands]):
            r = f.result()
            if r: recs.append(r)
    # tesis başına en iyi
    best = {}
    for r in recs:
        k = slug(r["venue"])
        if k not in best or r["score"] > best[k]["score"]:
            best[k] = r
    ranked = sorted(best.values(), key=lambda r: -r["score"])
    json.dump({"all": recs, "best_per_venue": ranked}, open(os.path.join(CAND,"candidates.json"),"w"),
              ensure_ascii=False, indent=1)
    log(f"BİTTİ — {len(recs)} kare, {len(ranked)} benzersiz tesis")
    log("--- en iyi 24 (tesis başına) ---")
    for r in ranked[:24]:
        log(f"  {r['score']:.3f}  {r['venue'][:30]:30s} br={r['bright']:.0f} sat={r['sat']:.0f} grn={r['green']:.2f} line={r['line']:.3f} blown={r['blown']:.3f}")

if __name__ == "__main__":
    main()
