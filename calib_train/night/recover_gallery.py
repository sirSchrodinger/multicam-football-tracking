#!/usr/bin/env python3
"""auto | recovery yan-yana karşılaştırma galerisi (25 saha). Strict verdict'ler etiketli.
Alperen göz gezdirir. cand/recover_gallery.html (file:// overlay'ler)."""
import os, json
HERE=os.path.dirname(os.path.abspath(__file__)); ROOT=os.path.dirname(HERE)
A={r['idx']:r for r in json.load(open(f"{ROOT}/cand/_judge_results.json"))['per_idx']}
R={r['idx']:r for r in json.load(open(f"{ROOT}/cand/_judge_recover_results.json"))['per_idx']}
COL={"GOOD":"#37d67a","DECENT":"#6fcf97","ROUGH":"#f2c94c","BAD":"#eb5757","UNSOLVABLE":"#828a94",None:"#555"}
def badge(v): return f"<span style='background:{COL.get(v,'#555')};color:#111;padding:1px 7px;border-radius:5px;font-size:11px;font-weight:700'>{v}</span>"
rows=[]
for i in sorted(R):
    av=A.get(i,{}).get('final'); rv=R[i].get('final'); ven=R[i].get('venue','')
    ap=f"{ROOT}/cand/judge/{i:03d}.jpg"; rp=f"{ROOT}/cand/recover/{i:03d}.jpg"
    better=" · <b style='color:#6fcf97'>recovery daha iyi</b>" if {'GOOD':4,'DECENT':3,'ROUGH':2,'BAD':1,None:0}.get(rv,0)>{'GOOD':4,'DECENT':3,'ROUGH':2,'BAD':1,None:0}.get(av,0) else ""
    rows.append(f"""<div class=row><div class=h>idx {i} · {ven}{better}</div>
<div class=pair>
<figure><img src='file://{ap}'><figcaption>otomatik {badge(av)}</figcaption></figure>
<figure><img src='file://{rp}'><figcaption>ajan-köşeli {badge(rv)}</figcaption></figure>
</div></div>""")
html=f"""<!doctype html><meta charset=utf-8><title>auto vs recovery — halısaha</title>
<style>body{{background:#14171a;color:#e9ede9;font-family:Inter,system-ui,sans-serif;margin:0;padding:24px;max-width:1100px;margin:auto}}
h1{{font-weight:600}}.row{{margin:22px 0;border-top:1px solid #262a2f;padding-top:12px}}
.h{{font-size:14px;color:#aab;margin-bottom:8px}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
img{{width:100%;border-radius:8px;border:1px solid #262a2f;display:block}}figcaption{{font-size:12px;color:#8b9490;margin-top:6px}}</style>
<h1>Otomatik kalibrasyon vs ajan-köşeli recovery</h1>
<p style='color:#8b9490'>25 saha, sol=otomatik seg2, sağ=ajan 4-köşe okur. Etiketler katı "tüm-çizgi" bar (agent-judge+verify).</p>
{''.join(rows)}"""
open(f"{ROOT}/cand/recover_gallery.html","w").write(html)
print(f"-> {ROOT}/cand/recover_gallery.html ({len(R)} saha)")
if __name__=="__main__": pass
