#!/usr/bin/env python3
"""calib/make_guided_html.py — REHBERLİ tek-tek nokta tıklayıcı (2D + foto, yan yana).

Alperen'in istediği akış: her landmark için SOL'daki 2D şemada o nokta YANIP SÖNER,
SAĞ'daki gerçek (undistorted) fotoğrafta yerine basılır, sonra OTOMATİK bir sonrakine
geçer. "önce 2D, sonra saha" — her nokta için. Görünmeyeni ATLA.

Kadraj-dışı köşeler (yakın köşeler 10,11) kesildiği için saha 6-gen görünür;
onlar otomatik atlanır → ~13 görünür nokta.

Çıktı: ~/Downloads/clicks_cankaya_cam2.json  (watch_clicks.sh yakalar → solver).
"""
import argparse, base64, json as _json, sys
from pathlib import Path
sys.path.insert(0, ".")
import cv2
from calib.landmarks import landmarks, BOX_BD_NOM, BOX_BW_NOM

# Genel: herhangi kamera/saha icin clicker (B2B genelleme). Default = cam2.
_ap = argparse.ArgumentParser()
_ap.add_argument("--cam", default="cankaya_cam2")
_ap.add_argument("--img", default="calib/undist_clean.png")
_ap.add_argument("--out", default="calib/click2.html")
_ap.add_argument("--prev", default="calib/clicks_cankaya_cam2.json")
_ap.add_argument("--L", type=float, default=33.0)
_ap.add_argument("--W", type=float, default=18.0)
_ap.add_argument("--space", default="undistorted")  # distorted | undistorted
_a, _ = _ap.parse_known_args()
IMG, OUT, PREV, L, W = _a.img, _a.out, _a.prev, _a.L, _a.W
DLNAME = f"clicks_{_a.cam}.json"
GROUP_COL = {"post": "#af52de", "center": "#0a84ff", "midline": "#5856d6",
             "corner": "#ff9f0a", "box": "#34c759"}

LM = landmarks(L, W)
# [id, num.label, color, offFrame, worldX, worldY, group]
POINTS = [[lid, f"{i}. {label}", GROUP_COL[grp], bool(off), float(x), float(y), grp]
          for i, (lid, label, (x, y), grp, off) in enumerate(LM, 1)]

INIT = {}
_pp = Path(PREV)
if _pp.exists():
    try:
        INIT = _json.loads(_pp.read_text()).get("points", {})
    except Exception:
        INIT = {}

img = cv2.imread(IMG)
if img is None:
    raise SystemExit(f"okunamadi: {IMG}")
h, w = img.shape[:2]
_, buf = cv2.imencode(".png", img)
b64 = base64.b64encode(buf).decode()

HTML = r"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<title>halısaha rehberli tıklayıcı</title>
<style>
 html,body{margin:0;height:100%;background:#1c1c1e;color:#eee;font:13px/1.45 -apple-system,system-ui,sans-serif;overflow:hidden}
 #top{height:96px;box-sizing:border-box;padding:8px 14px;background:#2c2c2e;border-bottom:1px solid #3a3a3c;display:flex;flex-direction:column;gap:5px}
 #top .row{display:flex;align-items:center;gap:12px}
 #step{font-size:13px;color:#9b9b9f;font-variant-numeric:tabular-nums}
 #name{font-size:19px;font-weight:700;display:flex;align-items:center;gap:9px}
 #chip{width:16px;height:16px;border-radius:50%;border:2px solid #fff;flex:none}
 #hint{font-size:12.5px;color:#c7c7cc}
 #bar{display:flex;gap:3px;margin-left:auto}
 #bar .b{width:13px;height:13px;border-radius:3px;background:#48484a;font-size:9px;text-align:center;line-height:13px;color:#000}
 button{background:#0a84ff;color:#fff;border:0;border-radius:7px;padding:7px 12px;cursor:pointer;font-size:12.5px}
 button.sec{background:#3a3a3c;color:#eee}
 button.warn{background:#ff9f0a;color:#000}
 #main{display:flex;height:calc(100% - 96px)}
 #left{width:46%;max-width:760px;flex:none;background:#0b1f14;border-right:2px solid #3a3a3c;position:relative}
 #cv2d{display:block;width:100%;height:100%}
 #right{flex:1;position:relative;background:#000}
 #cv{display:block;width:100%;height:100%;cursor:crosshair}
 #loupe{position:absolute;right:10px;top:10px;width:190px;height:190px;border:2px solid #fff;border-radius:10px;background:#000;pointer-events:none;display:none;z-index:9}
 #coord{position:absolute;left:10px;bottom:10px;background:#000a;padding:4px 9px;border-radius:6px;font-variant-numeric:tabular-nums;font-size:12px}
 #ph{position:absolute;left:10px;top:10px;background:#000a;padding:5px 10px;border-radius:8px;font-size:12px;max-width:60%}
 #done{position:absolute;left:0;right:0;bottom:10px;text-align:center;font-size:12px;color:#9f9}
</style></head><body>
<div id="top">
 <div class="row">
  <span id="step"></span>
  <span id="name"><span id="chip"></span><span id="nm"></span></span>
  <span id="bar"></span>
 </div>
 <div id="hint"></div>
 <div class="row">
  <button onclick="place_skip()" class="warn">⊘ görünmüyor / atla (s)</button>
  <button onclick="prev()" class="sec">‹ geri (←)</button>
  <button onclick="next()" class="sec">ileri › (→ / Enter)</button>
  <button onclick="dl()">⤓ JSON indir</button>
  <button onclick="fit()" class="sec">sığdır</button>
  <span id="done"></span>
 </div>
</div>
<div id="main">
 <div id="left"><canvas id="cv2d"></canvas></div>
 <div id="right">
  <canvas id="cv"></canvas>
  <canvas id="loupe" width="190" height="190"></canvas>
  <div id="ph">SOL'daki 2D'de <b>yanıp sönen</b> noktayı bul → burada gerçek yerine <b>sol-tık</b>.</div>
  <div id="coord">—</div>
 </div>
</div>
<script>
const IMGW=__W__, IMGH=__H__, L=__L__, WW=__WW__;
const POINTS=__POINTS__;            // [id,label,color,off,wx,wy,grp]
const N=POINTS.length;
const pos=Object.assign({},__INIT__);
const skipped={};
let cur=(function(){for(let i=0;i<N;i++)if(!(POINTS[i][0] in pos))return i;return 0;})();

const img=new Image(); img.src="data:image/png;base64,__B64__";
const c2=document.getElementById('cv2d'), x2=c2.getContext('2d');
const cv=document.getElementById('cv'),  ctx=cv.getContext('2d');
const lp=document.getElementById('loupe'), lx=lp.getContext('2d');
let view={s:1,ox:0,oy:0}, mouse={x:0,y:0}, drag=null, pulse=0;

/* ---------- layout ---------- */
function layout(){
 const L1=document.getElementById('left'), R=document.getElementById('right');
 c2.width=L1.clientWidth; c2.height=L1.clientHeight;
 cv.width=R.clientWidth;  cv.height=R.clientHeight;
 fit();
}
function fit(){const s=Math.min(cv.width/IMGW,cv.height/IMGH);view.s=s;
 view.ox=(cv.width-IMGW*s)/2;view.oy=(cv.height-IMGH*s)/2;drawPhoto();}

/* ---------- 2D pitch ---------- */
function tf2d(){const cw=c2.width,ch=c2.height,pad=46;
 const s=Math.min((cw-2*pad)/L,(ch-2*pad)/WW);
 const ox=(cw-L*s)/2, oy=(ch-WW*s)/2;
 return {s,ox,oy};}
function w2c(x,y){const t=tf2d();return [t.ox+x*t.s, t.oy+(WW-y)*t.s];}  // y=0 alt

function draw2d(){
 const cw=c2.width,ch=c2.height; x2.clearRect(0,0,cw,ch);
 x2.fillStyle='#0b6b2e'; const a=w2c(0,0),b=w2c(L,WW);
 x2.fillRect(a[0],b[1],b[0]-a[0],a[1]-b[1]);
 x2.strokeStyle='#fff'; x2.lineWidth=2; x2.lineJoin='round';
 // çevre
 poly([[0,0],[L,0],[L,WW],[0,WW],[0,0]]);
 // orta çizgi + çember
 seg(L/2,0,L/2,WW);
 circle(L/2,WW/2,3.0);
 dot2(L/2,WW/2,2,'#fff');
 // kaleler (3m)
 x2.strokeStyle='#ffd60a'; seg(0,WW/2-1.5,0,WW/2+1.5); seg(L,WW/2-1.5,L,WW/2+1.5);
 // iki ceza sahası da (yeşil, nominal) — tam standart saha
 x2.strokeStyle='#34c759';
 poly([[0,WW/2-__BW__/2],[__BD__,WW/2-__BW__/2],[__BD__,WW/2+__BW__/2],[0,WW/2+__BW__/2]]);
 poly([[L,WW/2-__BW__/2],[L-__BD__,WW/2-__BW__/2],[L-__BD__,WW/2+__BW__/2],[L,WW/2+__BW__/2]]);
 // kesik köşeler işareti (kadraj dışı near corners)
 x2.fillStyle='rgba(255,60,60,.16)';
 x2.fillRect(...rect(0,0,5,3.5)); x2.fillRect(...rect(0,WW-3.5,5,3.5));
 // kamera etiketi
 x2.fillStyle='#0a84ff'; x2.font='bold 13px sans-serif'; x2.textAlign='center';
 const cam=w2c(L/2,0); x2.fillText('▼ KAMERA bu kenarda (yakın)', cam[0], cam[1]+24);
 const far=w2c(L/2,WW); x2.fillText('uzak kenar', far[0], far[1]-12);
 // noktalar
 POINTS.forEach((p,i)=>{
   const [sx,sy]=w2c(p[4],p[5]); const isCur=i==cur;
   const placed=(p[0] in pos), sk=skipped[p[0]];
   let r=isCur?11:6;
   if(isCur){ // pulsing halka
     const pr=14+5*Math.sin(pulse/14);
     x2.beginPath();x2.arc(sx,sy,pr,0,7);x2.strokeStyle='#fff';x2.lineWidth=3;x2.stroke();
     x2.beginPath();x2.arc(sx,sy,pr+6,0,7);x2.strokeStyle=p[2];x2.lineWidth=2;x2.stroke();
   }
   x2.globalAlpha=isCur?1:(placed?0.95:(sk?0.3:0.6));
   x2.beginPath();x2.arc(sx,sy,r,0,7);x2.fillStyle=p[2];x2.fill();
   x2.lineWidth=isCur?3:1.5;x2.strokeStyle='#fff';x2.stroke();
   x2.fillStyle='#fff';x2.font='bold '+(isCur?13:10)+'px sans-serif';x2.textAlign='center';x2.textBaseline='middle';
   x2.fillText(String(i+1),sx,sy);
   if(placed&&!isCur){x2.fillStyle='#0f0';x2.fillText('✓',sx+r+6,sy-r);}
   if(sk){x2.fillStyle='#f66';x2.fillText('✕',sx+r+6,sy-r);}
   x2.globalAlpha=1;
 });
 x2.textBaseline='alphabetic';
}
function poly(P){x2.beginPath();P.forEach((q,i)=>{const c=w2c(q[0],q[1]);i?x2.lineTo(c[0],c[1]):x2.moveTo(c[0],c[1]);});x2.stroke();}
function seg(x0,y0,x1,y1){const a=w2c(x0,y0),b=w2c(x1,y1);x2.beginPath();x2.moveTo(a[0],a[1]);x2.lineTo(b[0],b[1]);x2.stroke();}
function circle(x,y,rm){const t=tf2d();const c=w2c(x,y);x2.beginPath();x2.arc(c[0],c[1],rm*t.s,0,7);x2.stroke();}
function dot2(x,y,r,col){const c=w2c(x,y);x2.beginPath();x2.arc(c[0],c[1],r,0,7);x2.fillStyle=col;x2.fill();}
function rect(x,y,wm,hm){const a=w2c(x,y+hm),t=tf2d();return [a[0],a[1],wm*t.s,hm*t.s];}

/* ---------- photo ---------- */
function toImg(px,py){return [(px-view.ox)/view.s,(py-view.oy)/view.s];}
function toScr(ix,iy){return [ix*view.s+view.ox,iy*view.s+view.oy];}
function drawPhoto(){
 ctx.fillStyle='#000';ctx.fillRect(0,0,cv.width,cv.height);
 ctx.save();ctx.translate(view.ox,view.oy);ctx.scale(view.s,view.s);
 if(img.complete)ctx.drawImage(img,0,0);ctx.restore();
 POINTS.forEach((p,i)=>{if(!(p[0] in pos))return;const[sx,sy]=toScr(pos[p[0]][0],pos[p[0]][1]);
   const isCur=i==cur;
   ctx.beginPath();ctx.arc(sx,sy,isCur?7:4.5,0,7);ctx.fillStyle=p[2];ctx.globalAlpha=isCur?1:.55;ctx.fill();
   ctx.globalAlpha=1;ctx.lineWidth=isCur?2.5:1.2;ctx.strokeStyle='#fff';ctx.stroke();
   ctx.fillStyle='#fff';ctx.font='bold 11px sans-serif';ctx.fillText(String(i+1),sx+6,sy-6);
 });
}
function loupe(px,py){const[ix,iy]=toImg(px,py),z=10,r=95/z;
 lx.imageSmoothingEnabled=false;lx.fillStyle='#000';lx.fillRect(0,0,190,190);
 if(img.complete)lx.drawImage(img,ix-r,iy-r,2*r,2*r,0,0,190,190);
 lx.strokeStyle='#0f0';lx.lineWidth=1;lx.beginPath();
 lx.moveTo(95,0);lx.lineTo(95,190);lx.moveTo(0,95);lx.lineTo(190,95);lx.stroke();
 lx.fillStyle=POINTS[cur][2];lx.fillRect(91,91,8,8);}

/* ---------- state ---------- */
function header(){
 const p=POINTS[cur];
 const visIdx=POINTS.slice(0,cur+1).filter(q=>!q[3]).length;
 const visTot=POINTS.filter(q=>!q[3]).length;
 document.getElementById('step').textContent='adım '+visIdx+' / '+visTot+'   (nokta #'+(cur+1)+'/'+N+')';
 document.getElementById('chip').style.background=p[2];
 document.getElementById('nm').textContent=p[1]+(p[3]?'   — KADRAJ DIŞI (kesik köşe, atla)':'');
 document.getElementById('hint').innerHTML=p[3]
   ? 'Bu köşe genelde KESİK (kadraj dışı, saha burada 6-gen). Görüyorsan bas, göremiyorsan <b>atla (s)</b>.'
   : 'SOL 2D şemada <b>'+(cur+1)+' numara yanıp sönüyor</b>. Aynı yeri sağ fotoğrafta bul, <b>sol-tık</b>la bas. Görmüyorsan <b>atla</b>. <b>Kale direklerini (1·2·3·4) bas → gerçek metre buradan.</b>';
 const nb=Object.keys(pos).length, ns=Object.keys(skipped).length;
 document.getElementById('done').textContent=nb+' basıldı · '+ns+' atlandı';
 const bar=document.getElementById('bar');bar.innerHTML='';
 POINTS.forEach((q,i)=>{const e=document.createElement('span');e.className='b';
   e.style.background=(q[0] in pos)?'#34c759':(skipped[q[0]]?'#ff453a':(i==cur?'#0a84ff':'#48484a'));
   e.textContent=(q[0] in pos)?'✓':(skipped[q[0]]?'✕':String(i+1));
   e.title=q[1];e.onclick=()=>{cur=i;refresh();};bar.appendChild(e);});
}
function refresh(){header();draw2d();drawPhoto();}
function nextOpen(){for(let k=1;k<=N;k++){const j=(cur+k)%N;const q=POINTS[j];
   if(!(q[0] in pos)&&!skipped[q[0]]){cur=j;return;} } cur=(cur+1)%N; }
function place(ix,iy){pos[POINTS[cur][0]]=[Math.round(ix*10)/10,Math.round(iy*10)/10];
   delete skipped[POINTS[cur][0]];nextOpen();refresh();}
function place_skip(){skipped[POINTS[cur][0]]=true;delete pos[POINTS[cur][0]];nextOpen();refresh();}
function next(){cur=(cur+1)%N;refresh();}
function prev(){cur=(cur-1+N)%N;refresh();}
function jsonObj(){const points={};POINTS.forEach(p=>{if(p[0] in pos)points[p[0]]=pos[p[0]];});
   return {image:"__IMGPATH__",space:"__SPACE__",mode:"points",points};}
function dl(){const b=new Blob([JSON.stringify(jsonObj(),null,1)],{type:'application/json'});
   const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='__DLNAME__';a.click();}

/* ---------- photo events ---------- */
cv.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect();
 mouse.x=e.clientX-r.left;mouse.y=e.clientY-r.top;
 if(drag){view.ox+=mouse.x-drag.x;view.oy+=mouse.y-drag.y;drag.x=mouse.x;drag.y=mouse.y;}
 const[ix,iy]=toImg(mouse.x,mouse.y);
 document.getElementById('coord').textContent='#'+(cur+1)+'  ('+ix.toFixed(1)+', '+iy.toFixed(1)+')';
 lp.style.display='block';loupe(mouse.x,mouse.y);drawPhoto();});
cv.addEventListener('mouseleave',()=>{lp.style.display='none';});
cv.addEventListener('mousedown',e=>{if(e.button==1||e.button==2){drag={x:mouse.x,y:mouse.y};e.preventDefault();}});
window.addEventListener('mouseup',()=>{drag=null;});
cv.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('click',e=>{if(e.button!=0)return;const r=cv.getBoundingClientRect();
 const[ix,iy]=toImg(e.clientX-r.left,e.clientY-r.top);
 if(ix<0||iy<0||ix>IMGW||iy>IMGH)return;place(ix,iy);});
cv.addEventListener('wheel',e=>{e.preventDefault();const f=e.deltaY<0?1.15:1/1.15;
 const[ix,iy]=toImg(mouse.x,mouse.y);view.s*=f;view.ox=mouse.x-ix*view.s;view.oy=mouse.y-iy*view.s;drawPhoto();},{passive:false});
window.addEventListener('keydown',e=>{
 if(e.key=='s'){place_skip();} else if(e.key=='ArrowRight'||e.key=='Enter'){next();}
 else if(e.key=='ArrowLeft'){prev();} else if(e.key=='u'){delete pos[POINTS[cur][0]];delete skipped[POINTS[cur][0]];refresh();}
 else return; e.preventDefault();});
window.addEventListener('resize',layout);
// jump by clicking a 2D dot
c2.addEventListener('click',e=>{const r=c2.getBoundingClientRect();const mx=e.clientX-r.left,my=e.clientY-r.top;
 let best=-1,bd=22;POINTS.forEach((p,i)=>{const c=w2c(p[4],p[5]);const d=Math.hypot(c[0]-mx,c[1]-my);if(d<bd){bd=d;best=i;}});
 if(best>=0){cur=best;refresh();}});

function tick(){pulse++;draw2d();requestAnimationFrame(tick);}
img.onload=()=>{layout();refresh();};
layout();refresh();tick();
</script></body></html>"""

HTML = (HTML.replace("__W__", str(w)).replace("__H__", str(h))
        .replace("__L__", str(L)).replace("__WW__", str(W))
        .replace("__BD__", str(BOX_BD_NOM)).replace("__BW__", str(BOX_BW_NOM))
        .replace("__POINTS__", _json.dumps(POINTS, ensure_ascii=False))
        .replace("__INIT__", _json.dumps(INIT))
        .replace("__IMGPATH__", IMG).replace("__SPACE__", _a.space).replace("__DLNAME__", DLNAME)
        .replace("__B64__", b64))
Path(OUT).write_text(HTML, encoding="utf-8")
print(f"yazıldı: {OUT}  ({len(HTML)//1024} KB, foto {w}x{h}, {len(POINTS)} landmark, {len(INIT)} ön-yüklü)")
