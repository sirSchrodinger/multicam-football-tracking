#!/usr/bin/env python3
"""FINAL MERGE: her saha için {auto-calib, vision-recovery} kalibrasyonlarından agent-verdict'i
DAHA İYİ olanı seç → per-saha en-iyi otonom kalibrasyon + dürüst final coverage.
Girdi: cand/_judge_results.json (auto) + cand/_judge_recover_results.json (recovery, opsiyonel).
Çıktı: cand/_final_best.json + konsol özet."""
import os, sys, json
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
RANK={"GOOD":4,"DECENT":3,"ROUGH":2,"BAD":1,"UNSOLVABLE":0,None:-1}
USABLE={"GOOD","DECENT"}

def load(p):
    if not os.path.exists(p): return {}
    d=json.load(open(p)); rows=d.get("per_idx",d) if isinstance(d,dict) else d
    return {r['idx']:r for r in rows}

def main():
    auto=load(os.path.join(ROOT,"cand","_judge_results.json"))
    rec=load(os.path.join(ROOT,"cand","_judge_recover_results.json"))
    idxs=sorted(set(auto)|set(rec))
    out=[]
    for i in idxs:
        a=auto.get(i,{}); r=rec.get(i,{})
        av=a.get('final',a.get('judge')); rv=r.get('final',r.get('judge'))
        if RANK.get(rv,-1)>RANK.get(av,-1):
            best,src,verd,reason=rv,'vision-recover',rv,r.get('reason','')
        else:
            best,src,verd,reason=av,'auto-seg2',av,a.get('reason','')
        out.append(dict(idx=i,venue=(a.get('venue') or r.get('venue') or ''),
                        final=verd,source=src,auto=av,recover=rv,reason=reason))
    json.dump(out,open(os.path.join(ROOT,"cand","_final_best.json"),"w"),ensure_ascii=False,indent=1)
    from collections import Counter
    c=Counter(o['final'] for o in out); srcu=Counter(o['source'] for o in out if o['final'] in USABLE)
    usable=[o for o in out if o['final'] in USABLE]
    print(f"toplam saha (herhangi calib denendi): {len(out)}")
    print("final verdict:", dict(c))
    print(f"KULLANILABİLİR (GOOD+DECENT): {len(usable)}/{len(out)}  kaynak: {dict(srcu)}")
    rec_gain=[o for o in usable if o['source']=='vision-recover']
    print(f"vision-recovery'nin kurtardığı (auto kötüyken): {len(rec_gain)} -> {[o['idx'] for o in rec_gain]}")
    print("\nKULLANILABİLİR sahalar:")
    for o in sorted(usable,key=lambda x:(-RANK[x['final']],x['idx'])):
        print(f"  idx{o['idx']:>3} {o['final']:>6} [{o['source']:>14}] {o['venue']}")
    print(f"\n-> cand/_final_best.json")

if __name__=="__main__": main()
