#!/usr/bin/env python3
"""vision-corners workflow çıktısından (agent-okunan 4 köşe) TÜM sahalar için vision-calib
recovery overlay üretir. Girdi: cand/_vision_corners.json (per_idx). Çıktı: cand/recover/NNN.jpg
+ cand/_recover_judgelist.json (judge workflow'a beslenecek).
"""
import os, sys, json
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
sys.path.insert(0, os.path.dirname(ROOT))
from calib_train.vision_recover import recover
SRC=os.path.join(ROOT,"cand","_vision_corners.json")

def main():
    data=json.load(open(SRC))
    rows=data.get("per_idx",data) if isinstance(data,dict) else data
    done=[]; skip=[]
    for r in rows:
        idx=r['idx']
        if not r.get('calibratable',False): skip.append((idx,"not_calibratable")); continue
        try:
            corners=[r['near_left'],r['far_left'],r['near_right'],r['far_right']]
            rec,out=recover(idx,corners)
            if out: done.append(dict(idx=idx,venue=r.get('venue',''),conf=r.get('confidence'),path=out))
            else: skip.append((idx,f"recover-fail:{rec.get('reason') if rec else 'None'}"))
        except Exception as e:
            skip.append((idx,f"err:{e}"))
    json.dump(done,open(os.path.join(ROOT,"cand","_recover_judgelist.json"),"w"),ensure_ascii=False)
    print(f"recovery overlay üretildi: {len(done)} | atlanan: {len(skip)}")
    for i,why in skip[:20]: print(f"  skip idx{i}: {why}")
    print(f"-> cand/recover/*.jpg + cand/_recover_judgelist.json")

if __name__=="__main__": main()
