#!/usr/bin/env python3
"""calib/make_clicker_html.py — tek-dosya tarayıcı NOKTA-tıklayıcı (point mode).

11 numaralı landmark (landmarks.py / field_2d.png ile aynı). Numarayı seç,
gerçek (undistorted) görüntüde yerine bas; görmediğini ATLA. Loupe + zoom/pan.
'JSON indir' -> ~/Downloads/clicks_cankaya_cam2.json  (solver onu okur).
"""
import base64
from pathlib import Path
import json as _json
import sys
sys.path.insert(0, ".")
import cv2
from calib.landmarks import landmarks

IMG = sys.argv[1] if len(sys.argv) > 1 else "calib/undist_clean.png"
OUT = sys.argv[2] if len(sys.argv) > 2 else "calib/click.html"
PREV = sys.argv[3] if len(sys.argv) > 3 else "calib/clicks_cankaya_cam2.json"
VENUE = Path(IMG).stem
SPACE = "raw" if len(sys.argv) > 1 else "undistorted"   # ham-kare modu (yeni saha) vs Cankaya undistorted
GROUP_COL = {"post": "#af52de", "center": "#0a84ff", "midline": "#5856d6",
             "corner": "#ff9f0a", "box": "#34c759"}

LM = landmarks(33.0, 18.0)
POINTS = [[lid, f"{i}. {label}", GROUP_COL[grp], bool(off)]
          for i, (lid, label, _, grp, off) in enumerate(LM, 1)]

# önceki tıklamaları (varsa) ön-yükle: Alperen sıfırdan başlamasın, yalnız kutu köşelerini eklesin
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
ok, buf = cv2.imencode(".png", img)
b64 = base64.b64encode(buf).decode()

HTML = """<!doctype html><html lang="tr"><head><meta charset="utf-8">
<title>halısaha nokta tıklayıcı</title>
<style>
 html,body{margin:0;height:100%;background:#1c1c1e;color:#eee;font:13px/1.4 -apple-system,system-ui,sans-serif;overflow:hidden}
 #wrap{display:flex;height:100%}
 #side{width:330px;flex:none;padding:12px;box-sizing:border-box;background:#2c2c2e;overflow:auto}
 #cv{flex:1;display:block;cursor:crosshair;background:#000}
 h1{font-size:15px;margin:0 0 6px}
 .pt{padding:6px 8px;border-radius:7px;margin:3px 0;cursor:pointer;border:1px solid #3a3a3c;background:#3a3a3c;display:flex;align-items:center;gap:7px}
 .pt.act{outline:2px solid #fff}
 .pt.skip{opacity:.45;text-decoration:line-through}
 .dot{width:12px;height:12px;border-radius:50%;flex:none;border:1px solid #fff}
 .pt .lbl{flex:1;font-size:12px}
 .pt .mk{font-size:14px;width:16px;text-align:center}
 button{background:#0a84ff;color:#fff;border:0;border-radius:7px;padding:7px 10px;margin:3px 3px 3px 0;cursor:pointer;font-size:12px}
 button.sec{background:#3a3a3c}
 #loupe{position:fixed;right:12px;top:12px;width:180px;height:180px;border:2px solid #fff;border-radius:10px;background:#000;pointer-events:none;display:none;z-index:9}
 #coord{position:fixed;left:342px;bottom:10px;background:#000a;padding:4px 8px;border-radius:6px;font-variant-numeric:tabular-nums}
 #ta{width:100%;height:80px;background:#1c1c1e;color:#9f9;border:1px solid #3a3a3c;border-radius:6px;font:11px monospace;box-sizing:border-box;margin-top:6px}
 .hint{opacity:.75;font-size:11px;margin:7px 0}
</style></head><body>
<div id="wrap">
 <div id="side">
  <h1>nokta tıklayıcı</h1>
  <div class="hint">2D şemadaki numarayı seç → görüntüde yerine <b>sol-tık</b>. Görmüyorsan <b>atla</b>.<br>
   tekerlek=zoom · sağ/orta-tuş sürükle=pan · <b>u</b>=geri · <b>n/b</b>=gez · <b>s</b>=atla</div>
  <div id="list"></div>
  <div style="margin-top:8px">
   <button onclick="skip()">⊘ atla (görünmüyor)</button>
   <button onclick="undo()" class="sec">↶ geri (u)</button>
  </div>
  <div style="margin-top:6px">
   <button onclick="dl()">⤓ JSON indir</button>
   <button onclick="cp()" class="sec">kopyala</button>
   <button onclick="fit()" class="sec">sığdır</button>
  </div>
  <div class="hint">En az 4 nokta (kale/köşe/orta/orta-çizgi karışık) yeter; ne kadar çok ve yayılmış o kadar iyi.</div>
  <textarea id="ta" readonly></textarea>
 </div>
 <canvas id="cv"></canvas>
</div>
<canvas id="loupe" width="180" height="180"></canvas>
<div id="coord">—</div>
<script>
const W=__W__, H=__H__;
const POINTS=__POINTS__;                 // [id,label,color,offFrame]
const pos=Object.assign({},__INIT__);    // id -> [x,y] (önceki tıklamalar ön-yüklü)
const skipped={};                        // id -> true
let cur=(function(){for(let i=0;i<POINTS.length;i++)if(!(POINTS[i][0] in pos))return i;return 0;})();
const img=new Image(); img.src="data:image/png;base64,__B64__";
const cv=document.getElementById('cv'), ctx=cv.getContext('2d');
const lp=document.getElementById('loupe'), lx=lp.getContext('2d');
let view={s:1,ox:0,oy:0}; let mouse={x:0,y:0};

function resize(){cv.width=cv.clientWidth;cv.height=cv.clientHeight;draw();}
function fit(){const s=Math.min(cv.width/W,cv.height/H);view.s=s;view.ox=(cv.width-W*s)/2;view.oy=(cv.height-H*s)/2;draw();}
function toImg(px,py){return [(px-view.ox)/view.s,(py-view.oy)/view.s];}
function toScr(ix,iy){return [ix*view.s+view.ox,iy*view.s+view.oy];}

function draw(){
 ctx.fillStyle='#000';ctx.fillRect(0,0,cv.width,cv.height);
 ctx.save();ctx.translate(view.ox,view.oy);ctx.scale(view.s,view.s);
 if(img.complete)ctx.drawImage(img,0,0);
 ctx.strokeStyle='rgba(0,255,255,.16)';ctx.lineWidth=0.5/view.s;
 for(let x=0;x<=W;x+=100){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,H);ctx.stroke();}
 for(let y=0;y<=H;y+=100){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
 ctx.restore();
 POINTS.forEach((p,i)=>{const id=p[0];if(!(id in pos))return;const[sx,sy]=toScr(pos[id][0],pos[id][1]);
   const act=i==cur;
   ctx.beginPath();ctx.arc(sx,sy,act?7:5,0,7);ctx.fillStyle=p[2];ctx.fill();
   ctx.lineWidth=act?2.5:1.5;ctx.strokeStyle='#fff';ctx.stroke();
   ctx.fillStyle='#fff';ctx.font='bold 12px sans-serif';ctx.fillText(String(i+1),sx+7,sy-7);
 });
 sync();
}
function loupe(px,py){const[ix,iy]=toImg(px,py);const z=9,r=90/z;
 lx.imageSmoothingEnabled=false;lx.fillStyle='#000';lx.fillRect(0,0,180,180);
 if(img.complete)lx.drawImage(img,ix-r,iy-r,2*r,2*r,0,0,180,180);
 lx.strokeStyle='#0f0';lx.lineWidth=1;lx.beginPath();
 lx.moveTo(90,0);lx.lineTo(90,180);lx.moveTo(0,90);lx.lineTo(180,90);lx.stroke();
 const c=POINTS[cur][2];lx.fillStyle=c;lx.fillRect(86,86,8,8);}
function list(){const d=document.getElementById('list');d.innerHTML='';
 POINTS.forEach((p,i)=>{const e=document.createElement('div');
  const done=(p[0] in pos), sk=skipped[p[0]];
  e.className='pt'+(i==cur?' act':'')+(sk?' skip':'');
  const mk=sk?'⊘':(done?'✓':'·');
  e.innerHTML='<span class="dot" style="background:'+p[2]+'"></span>'
    +'<span class="lbl">'+p[1]+(p[3]?' <i style="opacity:.7">(kadraj dışı?)</i>':'')+'</span>'
    +'<span class="mk">'+mk+'</span>';
  e.onclick=()=>{cur=i;draw();};d.appendChild(e);});}
function jsonObj(){const points={};POINTS.forEach(p=>{if(p[0] in pos)
   points[p[0]]=[Math.round(pos[p[0]][0]*10)/10,Math.round(pos[p[0]][1]*10)/10];});
 return {image:"calib/undist_clean.png",space:"undistorted",mode:"points",points};}
function sync(){document.getElementById('ta').value=JSON.stringify(jsonObj(),null,1);list();}
function nextOpen(){for(let k=1;k<=POINTS.length;k++){const j=(cur+k)%POINTS.length;
   if(!(POINTS[j][0] in pos)&&!skipped[POINTS[j][0]]){cur=j;return;}}}
function place(ix,iy){pos[POINTS[cur][0]]=[ix,iy];delete skipped[POINTS[cur][0]];nextOpen();draw();}
function skip(){skipped[POINTS[cur][0]]=true;delete pos[POINTS[cur][0]];nextOpen();draw();}
function undo(){delete pos[POINTS[cur][0]];delete skipped[POINTS[cur][0]];draw();}
function dl(){const b=new Blob([JSON.stringify(jsonObj(),null,1)],{type:'application/json'});
 const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='clicks_cankaya_cam2.json';a.click();}
function cp(){navigator.clipboard.writeText(JSON.stringify(jsonObj(),null,1));}

cv.addEventListener('mousemove',e=>{const r=cv.getBoundingClientRect();
 mouse.x=e.clientX-r.left;mouse.y=e.clientY-r.top;
 if(drag){view.ox+=mouse.x-drag.x;view.oy+=mouse.y-drag.y;drag.x=mouse.x;drag.y=mouse.y;}
 const[ix,iy]=toImg(mouse.x,mouse.y);
 document.getElementById('coord').textContent=(cur+1)+'. ('+ix.toFixed(1)+', '+iy.toFixed(1)+')';
 lp.style.display='block';loupe(mouse.x,mouse.y);draw();});
cv.addEventListener('mouseleave',()=>{lp.style.display='none';});
let drag=null;
cv.addEventListener('mousedown',e=>{if(e.button==1||e.button==2){drag={x:mouse.x,y:mouse.y};e.preventDefault();}});
window.addEventListener('mouseup',()=>{drag=null;});
cv.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('click',e=>{if(e.button!=0)return;const[ix,iy]=toImg(mouse.x,mouse.y);
 if(ix<0||iy<0||ix>W||iy>H)return;place(ix,iy);});
cv.addEventListener('wheel',e=>{e.preventDefault();const f=e.deltaY<0?1.15:1/1.15;
 const[ix,iy]=toImg(mouse.x,mouse.y);view.s*=f;view.ox=mouse.x-ix*view.s;view.oy=mouse.y-iy*view.s;draw();},{passive:false});
window.addEventListener('keydown',e=>{if(e.key=='u')undo();else if(e.key=='s')skip();
 else if(e.key=='n'){cur=(cur+1)%POINTS.length;draw();}
 else if(e.key=='b'){cur=(cur-1+POINTS.length)%POINTS.length;draw();}else return;e.preventDefault();});
window.addEventListener('resize',resize);
img.onload=()=>{resize();fit();};
resize();
</script></body></html>"""

HTML = (HTML.replace("__W__", str(w)).replace("__H__", str(h))
        .replace("__POINTS__", _json.dumps(POINTS, ensure_ascii=False))
        .replace("__INIT__", _json.dumps(INIT))
        .replace("__B64__", b64))
# venue-uyumlu metadata (download adı + space + image yolu)
HTML = (HTML.replace("clicks_cankaya_cam2.json", f"clicks_{VENUE}.json")
        .replace('image:"calib/undist_clean.png"', f'image:"{IMG}"')
        .replace('space:"undistorted"', f'space:"{SPACE}"'))
Path(OUT).write_text(HTML, encoding="utf-8")
print(f"yazıldı: {OUT}  ({len(HTML)//1024} KB, görsel {w}x{h}, {len(POINTS)} landmark)")
