#!/usr/bin/env python3
"""F4 deliverable: judge sonucundan (agent-verdict) ŞIK galeri üretir.
1) YEREL tam galeri (cand/gallery.html, file:// overlay'ler, verdict-gruplu) — Alperen göz gezdirir.
2) Artifact-rapor için sıkıştırılmış hero JPEG'ler (cand/hero/, ~900px q80) + hero_manifest.json.
Girdi: cand/_judge_results.json (workflow çıktısı, {per_idx:[{idx,venue,final,judge,worst,reason,...}]}).
"""
import os, sys, json, base64, cv2, numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
JUDGE=os.path.join(ROOT,"cand","_judge_results.json")
JDIR=os.path.join(ROOT,"cand","judge")
ORD=["GOOD","DECENT","ROUGH","BAD","UNSOLVABLE"]
COL={"GOOD":"#37d67a","DECENT":"#6fcf97","ROUGH":"#f2c94c","BAD":"#eb5757","UNSOLVABLE":"#828a94"}

def load():
    d=json.load(open(JUDGE)); return d.get("per_idx",d) if isinstance(d,dict) else d

def build_local(rows):
    by={v:[] for v in ORD}
    for r in rows: by.get(r.get("final","?"),by.setdefault(r.get("final","?"),[])).append(r)
    parts=["""<!doctype html><meta charset=utf-8><title>Halısaha otonom kalibrasyon — galeri</title>
<style>body{background:#141619;color:#eef;font-family:Inter,system-ui,sans-serif;margin:0;padding:24px}
h1{font-weight:600;font-size:22px}h2{margin:28px 0 10px;font-size:16px;font-weight:600}
.g{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:14px}
.c{background:#1c1f23;border-radius:10px;overflow:hidden;border:1px solid #262a2f}
.c img{width:100%;display:block}.m{padding:8px 12px;font-size:12px;color:#aab}
.b{display:inline-block;padding:2px 8px;border-radius:6px;font-weight:600;color:#111;font-size:11px}
.chip{color:#141619}</style><h1>Halısaha — otonom tek-kare kalibrasyon (agent-değerlendirmeli)</h1>"""]
    tot=len(rows)
    counts=" · ".join(f"<b style='color:{COL[v]}'>{v} {len(by.get(v,[]))}</b>" for v in ORD)
    parts.append(f"<p style='color:#aab'>{tot} saha · {counts}</p>")
    for v in ORD:
        items=by.get(v,[])
        if not items: continue
        parts.append(f"<h2 style='color:{COL[v]}'>{v} — {len(items)}</h2><div class=g>")
        for r in sorted(items,key=lambda x:x['idx']):
            img=os.path.join(JDIR,f"{r['idx']:03d}.jpg")
            reason=(r.get('reason') or '').replace('<','&lt;')
            parts.append(f"<div class=c><img src='file://{img}'>"
                         f"<div class=m><span class=b style='background:{COL[v]}'>{v}</span> "
                         f"idx {r['idx']} · {r.get('venue','')}<br>{reason}</div></div>")
        parts.append("</div>")
    out=os.path.join(ROOT,"cand","gallery.html"); open(out,"w").write("".join(parts))
    return out,{v:len(by.get(v,[])) for v in ORD},tot

def build_heroes(rows, per=2):
    """her verdict'ten örnek hero'lar, sıkıştırılmış (Artifact data-URI için)."""
    hdir=os.path.join(ROOT,"cand","hero"); os.makedirs(hdir,exist_ok=True)
    picked=[]; seen={v:0 for v in ORD}
    for v in ORD:
        for r in sorted([x for x in rows if x.get('final')==v],key=lambda x:x['idx']):
            if seen[v]>=per: break
            src=os.path.join(JDIR,f"{r['idx']:03d}.jpg"); im=cv2.imread(src)
            if im is None: continue
            sc=900/im.shape[1]; im=cv2.resize(im,(900,int(im.shape[0]*sc)))
            ok,buf=cv2.imencode(".jpg",im,[cv2.IMWRITE_JPEG_QUALITY,80])
            picked.append(dict(idx=r['idx'],venue=r.get('venue',''),verdict=v,reason=r.get('reason',''),
                               datauri="data:image/jpeg;base64,"+base64.b64encode(buf).decode()))
            seen[v]+=1
    json.dump(picked,open(os.path.join(hdir,"hero_manifest.json"),"w"),ensure_ascii=False)
    return picked

if __name__=="__main__":
    rows=load()
    out,counts,tot=build_local(rows)
    heroes=build_heroes(rows)
    print(f"galeri -> {out}  ({tot} saha, {counts})")
    print(f"hero -> {len(heroes)} örnek (cand/hero/hero_manifest.json)")
