#!/usr/bin/env python3
"""calib/make_paint_html.py — PAINT tarzı serbest çizgi+nokta annotasyon aracı.

Alperen gördüğü beyaz ÇİZGİleri trace eder (2+ tık = boydan boya), net gördüğü
NOKTAları basar. Görünmeyen köşeleri çizmesine gerek YOK — solver çizgileri
kesiştirip kurtarır. Çıktı: ~/Downloads/paint_cankaya_cam2.json (watch_paint.sh).
"""
import base64, json as _json, sys
from pathlib import Path
sys.path.insert(0, ".")
import cv2
from calib.paint_labels import LINES, POINTS

IMG = "calib/undist_clean.png"
OUT = "calib/paint.html"

LINE_DEF = [[i, lab, col, "line"] for (i, lab, col, _k, _v) in LINES]
PT_DEF   = [[i, lab, col, "point"] for (i, lab, col) in POINTS]

img = cv2.imread(IMG)
if img is None:
    raise SystemExit(f"okunamadi: {IMG}")
h, w = img.shape[:2]
_, buf = cv2.imencode(".png", img)
b64 = base64.b64encode(buf).decode()

HTML = r"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<title>halısaha paint — çizgi + nokta</title>
<style>
 html,body{margin:0;height:100%;background:#1c1c1e;color:#eee;font:13px/1.45 -apple-system,system-ui,sans-serif;overflow:hidden}
 #wrap{display:flex;height:100%}
 #side{width:340px;flex:none;background:#2c2c2e;padding:11px;box-sizing:border-box;overflow:auto}
 h1{font-size:15px;margin:0 0 4px} h2{font-size:12px;margin:12px 0 5px;color:#9b9b9f;text-transform:uppercase;letter-spacing:.04em}
 .seg{display:flex;gap:6px;margin:6px 0}
 .seg button{flex:1}
 button{background:#3a3a3c;color:#eee;border:0;border-radius:7px;padding:7px 9px;cursor:pointer;font-size:12px}
 button.on{background:#0a84ff;color:#fff} button.act{background:#0a84ff;color:#fff}
 button.go{background:#0a84ff;color:#fff} button.warn{background:#ff9f0a;color:#000}
 .lab{display:flex;align-items:center;gap:8px;padding:6px 8px;border-radius:7px;margin:3px 0;cursor:pointer;border:1px solid #3a3a3c;background:#333}
 .lab.act{outline:2px solid #fff} .lab.done{background:#22351f}
 .sw{width:14px;height:14px;border-radius:4px;flex:none;border:1px solid #fff}
 .lab .t{flex:1;font-size:12px} .lab .m{font-size:12px;color:#9f9;width:20px;text-align:right}
 .hint{font-size:11.5px;color:#c7c7cc;margin:7px 0}
 #cv{flex:1;display:block;background:#000;cursor:crosshair}
 #loupe{position:fixed;right:12px;top:12px;width:190px;height:190px;border:2px solid #fff;border-radius:10px;background:#000;pointer-events:none;display:none;z-index:9}
 #coord{position:fixed;left:352px;bottom:10px;background:#000a;padding:4px 9px;border-radius:6px;font-variant-numeric:tabular-nums}
 #ta{width:100%;height:70px;background:#1c1c1e;color:#9f9;border:1px solid #3a3a3c;border-radius:6px;font:10px monospace;box-sizing:border-box;margin-top:6px}
</style></head><body>
<div id="wrap">
 <div id="side">
  <h1>çizgi + nokta</h1>
  <div class="hint">Beyaz çizgileri <b>trace</b> et (çizgi boyunca 2+ tık), net gördüğün işaretleri <b>nokta</b> bas. Görünmeyen köşeleri ÇİZME — çizgileri kesiştirip ben bulurum.<br>tekerlek=zoom · sağ/orta-sürükle=pan · <b>u</b>=geri al · <b>Esc</b>=çizgiyi bitir</div>
  <div class="seg">
   <button id="tline" class="on" onclick="setTool('line')">✏️ ÇİZGİ</button>
   <button id="tpoint" onclick="setTool('point')">⦿ NOKTA</button>
  </div>
  <h2>Çizgiler (trace)</h2><div id="lines"></div>
  <h2>Noktalar</h2><div id="points"></div>
  <div class="seg" style="margin-top:10px">
   <button onclick="undo()">↶ geri al (u)</button>
   <button onclick="delActive()" class="warn">aktifi sil</button>
  </div>
  <div class="seg">
   <button onclick="dl()" class="go">⤓ JSON indir</button>
   <button onclick="fit()">sığdır</button>
   <button onclick="clearAll()">temizle</button>
  </div>
  <div class="hint">En az: YAKIN kale çizgisi + YAKIN yan çizgi + UZAK yan çizgi + santra. Ne kadar çok çizgi/nokta, o kadar iyi.</div>
  <textarea id="ta" readonly></textarea>
 </div>
 <canvas id="cv"></canvas>
</div>
<canvas id="loupe" width="190" height="190"></canvas>
<div id="coord">—</div>
<script>
const IMGW=__W__, IMGH=__H__;
const LINE_DEF=__LINES__, PT_DEF=__POINTS__;   // [id,label,color,type]
const DEF={}; LINE_DEF.concat(PT_DEF).forEach(d=>DEF[d[0]]=d);
const lines={};   // id -> [[x,y],...]
const points={};  // id -> [x,y]
let tool='line', active=LINE_DEF[0][0];

const img=new Image(); img.src="data:image/png;base64,__B64__";
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const lp=document.getElementById('loupe'), lx=lp.getContext('2d');
let view={s:1,ox:0,oy:0}, mouse={x:0,y:0}, drag=null;

function resize(){cv.width=cv.clientWidth;cv.height=cv.clientHeight;draw();}
function fit(){const s=Math.min(cv.width/IMGW,cv.height/IMGH);view.s=s;view.ox=(cv.width-IMGW*s)/2;view.oy=(cv.height-IMGH*s)/2;draw();}
function toImg(px,py){return [(px-view.ox)/view.s,(py-view.oy)/view.s];}
function toScr(ix,iy){return [ix*view.s+view.ox,iy*view.s+view.oy];}

function setTool(t){tool=t;document.getElementById('tline').className=t=='line'?'on':'';
 document.getElementById('tpoint').className=t=='point'?'on':'';
 // aktif etiketi geçerli araca çevir
 const pool=t=='line'?LINE_DEF:PT_DEF; if(!pool.find(d=>d[0]==active))active=pool[0][0];
 buildList();draw();}

function draw(){
 ctx.fillStyle='#000';ctx.fillRect(0,0,cv.width,cv.height);
 ctx.save();ctx.translate(view.ox,view.oy);ctx.scale(view.s,view.s);
 if(img.complete)ctx.drawImage(img,0,0);ctx.restore();
 // çizgiler
 for(const id in lines){const P=lines[id];if(!P.length)continue;const col=DEF[id][2],isA=(id==active);
   ctx.strokeStyle=col;ctx.lineWidth=(isA?3:2);ctx.beginPath();
   P.forEach((q,i)=>{const[sx,sy]=toScr(q[0],q[1]);i?ctx.lineTo(sx,sy):ctx.moveTo(sx,sy);});ctx.stroke();
   P.forEach(q=>{const[sx,sy]=toScr(q[0],q[1]);ctx.beginPath();ctx.arc(sx,sy,isA?4:3,0,7);ctx.fillStyle=col;ctx.fill();
     ctx.lineWidth=1;ctx.strokeStyle='#fff';ctx.stroke();});}
 // noktalar
 for(const id in points){const[sx,sy]=toScr(points[id][0],points[id][1]);const col=DEF[id][2],isA=(id==active);
   ctx.beginPath();ctx.arc(sx,sy,isA?7:5,0,7);ctx.fillStyle=col;ctx.fill();
   ctx.lineWidth=isA?2.5:1.5;ctx.strokeStyle='#fff';ctx.stroke();}
 sync();
}
function loupe(px,py){const[ix,iy]=toImg(px,py),z=10,r=95/z;
 lx.imageSmoothingEnabled=false;lx.fillStyle='#000';lx.fillRect(0,0,190,190);
 if(img.complete)lx.drawImage(img,ix-r,iy-r,2*r,2*r,0,0,190,190);
 lx.strokeStyle='#0f0';lx.lineWidth=1;lx.beginPath();lx.moveTo(95,0);lx.lineTo(95,190);lx.moveTo(0,95);lx.lineTo(190,95);lx.stroke();
 lx.fillStyle=DEF[active][2];lx.fillRect(91,91,8,8);}

function row(d){const done=(d[3]=='line')?(lines[d[0]]&&lines[d[0]].length):(d[0] in points);
 const cnt=(d[3]=='line')?((lines[d[0]]||[]).length):((d[0] in points)?1:0);
 const e=document.createElement('div');e.className='lab'+(d[0]==active?' act':'')+(done?' done':'');
 e.innerHTML='<span class="sw" style="background:'+d[2]+'"></span><span class="t">'+d[1]+'</span><span class="m">'+(cnt?(d[3]=='line'?cnt:'✓'):'')+'</span>';
 e.onclick=()=>{active=d[0];tool=d[3];setTool(tool);};return e;}
function buildList(){const lc=document.getElementById('lines'),pc=document.getElementById('points');
 lc.innerHTML='';pc.innerHTML='';LINE_DEF.forEach(d=>lc.appendChild(row(d)));PT_DEF.forEach(d=>pc.appendChild(row(d)));}
function jsonObj(){const ln={};for(const id in lines)if(lines[id].length>=2)ln[id]=lines[id];
 return {image:"calib/undist_clean.png",space:"undistorted",lines:ln,points};}
function sync(){document.getElementById('ta').value=JSON.stringify(jsonObj());buildList();}

function addAt(ix,iy){const v=[Math.round(ix*10)/10,Math.round(iy*10)/10];
 if(tool=='line'){(lines[active]=lines[active]||[]).push(v);} else {points[active]=v;}draw();}
function undo(){if(tool=='line'){const P=lines[active];if(P&&P.length)P.pop();} else {delete points[active];}draw();}
function delActive(){if(tool=='line')delete lines[active];else delete points[active];draw();}
function clearAll(){for(const k in lines)delete lines[k];for(const k in points)delete points[k];draw();}
function dl(){const b=new Blob([JSON.stringify(jsonObj(),null,1)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='paint_cankaya_cam2.json';a.click();}

cv.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect();mouse.x=e.clientX-r.left;mouse.y=e.clientY-r.top;
 if(drag){view.ox+=mouse.x-drag.x;view.oy+=mouse.y-drag.y;drag.x=mouse.x;drag.y=mouse.y;}
 const[ix,iy]=toImg(mouse.x,mouse.y);
 document.getElementById('coord').textContent=DEF[active][1].slice(0,22)+'  ('+ix.toFixed(1)+', '+iy.toFixed(1)+')';
 lp.style.display='block';loupe(mouse.x,mouse.y);draw();});
cv.addEventListener('mouseleave',()=>{lp.style.display='none';});
cv.addEventListener('mousedown',e=>{if(e.button==1||e.button==2){const r=cv.getBoundingClientRect();drag={x:e.clientX-r.left,y:e.clientY-r.top};e.preventDefault();}});
window.addEventListener('mouseup',()=>{drag=null;});
cv.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('click',e=>{if(e.button!=0)return;const r=cv.getBoundingClientRect();
 const[ix,iy]=toImg(e.clientX-r.left,e.clientY-r.top);
 if(ix<0||iy<0||ix>IMGW||iy>IMGH)return;addAt(ix,iy);});
cv.addEventListener('wheel',e=>{e.preventDefault();const r=cv.getBoundingClientRect();const mx=e.clientX-r.left,my=e.clientY-r.top;
 const f=e.deltaY<0?1.15:1/1.15;const[ix,iy]=toImg(mx,my);view.s*=f;view.ox=mx-ix*view.s;view.oy=my-iy*view.s;draw();},{passive:false});
window.addEventListener('keydown',e=>{if(e.key=='u'){undo();}else if(e.key=='Escape'){/*çizgi zaten serbest*/}else return;e.preventDefault();});
window.addEventListener('resize',resize);
img.onload=()=>{resize();fit();};
setTool('line');buildList();resize();
</script></body></html>"""

HTML = (HTML.replace("__W__", str(w)).replace("__H__", str(h))
        .replace("__LINES__", _json.dumps(LINE_DEF, ensure_ascii=False))
        .replace("__POINTS__", _json.dumps(PT_DEF, ensure_ascii=False))
        .replace("__B64__", b64))
Path(OUT).write_text(HTML, encoding="utf-8")
print(f"yazıldı: {OUT}  ({len(HTML)//1024} KB, foto {w}x{h}, {len(LINE_DEF)} çizgi + {len(PT_DEF)} nokta etiketi)")
