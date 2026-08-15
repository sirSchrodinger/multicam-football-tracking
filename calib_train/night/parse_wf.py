#!/usr/bin/env python3
"""Genel workflow-transcript parser: agent-*.jsonl (prompt->idx) + journal.jsonl (agentId->result)
join eder. judge/verify/corners hepsine yarar. Workflow'un formal return'ünü beklemeden per-idx
sonucu diskten kurtarır.
Kullanım: python parse_wf.py <wf_transcript_dir> <mode:judge|corners> <out.json>
"""
import os, sys, json, re, glob

def extract_idx(text):
    m=re.search(r'/(?:judge|recover|vgrid)/(\d{3})\.jpg', text) or re.search(r'idx\s+(\d+)', text)
    return int(m.group(1)) if m else None

def agent_prompt(path):
    """ilk user mesajının metni (prompt)."""
    try:
        for l in open(path):
            j=json.loads(l); m=j.get('message',{})
            if isinstance(m,dict) and m.get('role')=='user':
                c=m.get('content')
                if isinstance(c,str): return c
                if isinstance(c,list):
                    for x in c:
                        if isinstance(x,dict) and x.get('type')=='text': return x.get('text','')
                        if isinstance(x,str): return x
    except Exception: pass
    return ""

def main():
    D=sys.argv[1]; mode=sys.argv[2]; out=sys.argv[3]
    # agentId -> (idx, prompt)
    amap={}
    for f in glob.glob(os.path.join(D,"agent-*.jsonl")):
        aid=os.path.basename(f)[len("agent-"):-len(".jsonl")]
        p=agent_prompt(f); idx=extract_idx(p)
        is_verify=("ŞÜPHECİ" in p or "İKİNCİ GÖZ" in p or "çürüt" in p.lower())
        amap[aid]=dict(idx=idx,verify=is_verify)
    # agentId -> result (journal)
    res={}
    for l in open(os.path.join(D,"journal.jsonl")):
        try:
            j=json.loads(l)
            if j.get('type')=='result' and 'result' in j and j.get('agentId'):
                res[j['agentId']]=j['result']
        except Exception: pass
    # join
    per={}
    for aid,meta in amap.items():
        idx=meta['idx']; r=res.get(aid)
        if idx is None or r is None: continue
        d=per.setdefault(idx,{})
        if mode=='corners':
            d['corners']=r
        else:
            if meta['verify']: d['verify']=r
            else: d['judge']=r
    rows=[]
    for idx,d in sorted(per.items()):
        if mode=='corners':
            c=d.get('corners',{}); rows.append(dict(idx=idx,**c))
        else:
            jv=(d.get('judge') or {}).get('verdict'); ver=d.get('verify')
            final=jv
            if ver is not None and jv in ('GOOD','DECENT'):
                final=jv if ver.get('agrees') else ver.get('verdict',jv)
            rows.append(dict(idx=idx,judge=jv,final=final,
                             worst=(d.get('judge') or {}).get('worst_element'),
                             reason=(d.get('judge') or {}).get('reason',''),
                             verify_reason=(ver or {}).get('reason') if ver else None))
    json.dump({'per_idx':rows},open(out,'w'),ensure_ascii=False,indent=1)
    from collections import Counter
    if mode=='corners':
        c=Counter(r.get('calibratable') for r in rows); print(f"corners: {len(rows)} | calibratable {dict(c)}")
    else:
        c=Counter(r['final'] for r in rows); print(f"judge: {len(rows)} | final {dict(c)}")
        usable=[r for r in rows if r['final'] in ('GOOD','DECENT')]
        print(f"kullanılabilir (GOOD+DECENT): {len(usable)} -> {[r['idx'] for r in usable]}")
    print(f"-> {out}")

if __name__=="__main__": main()
