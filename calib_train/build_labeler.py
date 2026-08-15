#!/usr/bin/env python3
"""Tek-dosya (self-contained) calib labeler üretir: kareleri base64 data-URI olarak
gömer -> file:// açılınca canvas TAINT olmaz -> canlı 2D warp localhost'suz çalışır.
Bağımlılık YOK (ham JPEG byte). Çıktı: calib_train/labeler_embed.html

TASARIM:
 1) Kamera nerede? -> şemaya TIKLA. Kamera sol/sağ arka, köşe, kale arkası fark etmez;
    sen yerleştirirsin. sol/sağ İSİM YOK -> her şey YAKIN/UZAK (kamera referanslı).
 2) Listeden çizgi seç -> şemada o çizgi YANAR (kamera konumuna göre doğru kenar).
    O çizgiyi sahada bul, üstünde istediğin kadar nokta tıkla (eğri ise 10-15 olur).
 3) Köşeleri kesişimden ben bulurum. Canlı 2D kuş-bakışı altta.
"""
import os, base64, json

HERE = os.path.dirname(os.path.abspath(__file__))
FR_DIR = os.path.join(HERE, "label_frames")
# build_labeler frame listesini frames.json'dan okur (yeniden-harvest bunu güncelleyebilir)
fj = os.path.join(HERE, "frames.json")
if os.path.exists(fj):
    FRAMES = json.load(open(fj))
else:
    FRAMES = ["Atakum.jpg","Atlantik.jpg","Avanos.jpg","Ayazma.jpg","Aydinoglu.jpg",
              "Bahcelievler.jpg","Cengiz.jpg","Kibris.jpg","R2_KbrsDorukHal.jpg",
              "R_AvanosHalSaha.jpg","R_BahelievlerBES.jpg","Rize.jpg","Sporland.jpg","Yesil.jpg"]

img_data, tot, used = {}, 0, []
for fn in FRAMES:
    p = os.path.join(FR_DIR, fn)
    if not os.path.exists(p):
        print("ATLANDI(yok):", fn); continue
    with open(p, "rb") as f:
        raw = f.read()
    img_data[fn] = "data:image/jpeg;base64," + base64.b64encode(raw).decode()
    tot += len(raw); used.append(fn)
    print(f"{fn:26s} {len(raw)//1024} KB")
FRAMES = used
print(f"--- {len(FRAMES)} kare, ham ~{tot//1024//1024} MB ---")

HTML = r"""<!DOCTYPE html><html lang="tr"><head><meta charset="utf-8">
<title>Halısaha Calib — kamera-yerleşimli, canlı 2D</title>
<style>
 *{box-sizing:border-box}
 body{margin:0;font-family:system-ui,Arial;background:#1b1d22;color:#e8e8e8;overflow:hidden}
 #top{position:fixed;top:0;left:0;right:0;height:44px;background:#0f1115;display:flex;
   align-items:center;gap:8px;padding:0 10px;z-index:10;border-bottom:1px solid #333;font-size:13px}
 #main{position:absolute;top:44px;bottom:0;left:0;right:352px}
 #cvwrap{position:absolute;inset:0;overflow:hidden;background:#000}
 #cv{position:absolute;left:0;top:0;cursor:crosshair}
 #zoombar{position:absolute;left:12px;right:12px;bottom:10px;height:42px;background:rgba(15,17,21,.85);
   border:1px solid #333;border-radius:9px;display:flex;align-items:center;gap:10px;padding:0 14px;z-index:6}
 #zoombar input[type=range]{flex:1;height:22px;cursor:pointer}
 #zoombar span{font-size:12px;color:#9ab;min-width:96px}
 #zoombar button{padding:4px 8px;font-size:12px}
 #side{position:fixed;top:44px;bottom:0;right:0;width:352px;background:#15171c;overflow-y:auto;padding:8px;border-left:1px solid #333}
 .ln{display:flex;align-items:center;gap:7px;padding:7px;margin:3px 0;border-radius:5px;background:#22252c;cursor:pointer;font-size:13px}
 .ln.active{background:#2d6cdf}.ln.done{background:#1f6b3a}
 .ln .sw{width:12px;height:12px;border-radius:3px;flex:0 0 auto}
 .ln small{opacity:.7;margin-left:auto;font-size:11px}
 button{background:#2d6cdf;color:#fff;border:0;padding:6px 10px;border-radius:5px;cursor:pointer;font-size:13px}
 button.alt{background:#444}button.warn{background:#a33}button.cam{background:#c98a1a}
 h4{margin:11px 0 4px;font-size:12px;color:#9ab;text-transform:uppercase;letter-spacing:.5px}
 #schema,#prev{width:100%;background:#08130c;border:1px solid #333;border-radius:6px;margin-top:4px;display:block;cursor:pointer}
 #loupe{position:fixed;width:160px;height:160px;border:2px solid #2d6cdf;border-radius:50%;pointer-events:none;display:none;z-index:20;overflow:hidden;background:#000}
 #loupe canvas{position:absolute}
 .cross{position:absolute;background:#2d6cdf;opacity:.8}
 #hint{font-size:12px;color:#9ab;line-height:1.45}
 #pstat,#cstat{font-size:12px;color:#9ab;margin-top:3px}
</style></head><body>
<div id="top">
 <b>Calib</b>
 <button class="alt" onclick="prev()">◀</button><span id="status"></span><button class="alt" onclick="next()">▶</button>
 <button class="alt" onclick="resetView()">Tam saha</button>
 <button class="warn" onclick="undo()">Geri al (z)</button>
 <button class="warn" onclick="clrLine()">Çizgiyi sil</button>
 <button onclick="save()">💾 JSON indir</button>
</div>
<div id="main"><div id="cvwrap"><canvas id="cv"></canvas></div>
 <div id="zoombar"><span style="min-width:auto">🔍 yakınlık</span>
  <input type="range" id="zoom" min="0.4" max="14" step="0.01" value="1">
  <span id="zlbl">1.0×</span>
  <button class="alt" onclick="zReset()">sığdır</button></div></div>
<div id="side">
 <h4>1) Kamera nerede? — şemaya tıkla</h4>
 <canvas id="schema" width="336" height="232"></canvas>
 <div id="cstat"></div>
 <button class="cam" style="margin-top:5px;width:100%" onclick="camPlace()">📷 Kamerayı (yeniden) yerleştir</button>
 <div id="hint" style="margin-top:8px">Kamerayı koyduktan sonra listeden çizgi seç → şemada <b>yanar</b>.
  O çizgiyi sahada bul, üstünde <b>istediğin kadar nokta</b> tıkla (eğri taç/çember 10-15 olur).
  Görünmeyen köşeyi tıklama — kesişimden ben bulurum. Tekerlek=zoom, Shift+sürükle=pan.</div>
 <h4>2) Çizgiler (yanan = sıradaki)</h4><div id="list"></div>
 <h4>3) Canlı 2D kuş-bakışı</h4>
 <canvas id="prev" width="336" height="190"></canvas>
 <div id="pstat"></div>
</div>
<div id="loupe"><canvas id="lcv" width="160" height="160"></canvas>
 <div class="cross" style="left:79px;top:0;bottom:0;width:1px"></div><div class="cross" style="left:0;right:0;top:79px;height:1px"></div></div>
<script>
const IMG_DATA = __IMG_DATA__;
const FRAMES = __FRAMES__;
// SOL/SAĞ YOK -> yakın/uzak, kamera referanslı (kamerayı sen yerleştiriyorsun)
const LINES=[
 {id:"goalN",t:"YAKIN kale çizgisi (kameraya yakın dip)"},
 {id:"goalF",t:"UZAK kale çizgisi (karşı dip)"},
 {id:"touchN",t:"YAKIN taç (kameraya yakın uzun kenar)"},
 {id:"touchF",t:"UZAK taç (karşı/duvar tarafı uzun kenar)"},
 {id:"center",t:"Orta çizgi"},
 {id:"boxN",t:"YAKIN ceza sahası — ön çizgi"},
 {id:"boxF",t:"UZAK ceza sahası — ön çizgi (simetri)"},
 {id:"circle",t:"Orta yuvarlak (5-6 nokta)"}];
const COLS={goalN:"#ff5050",goalF:"#ff9a40",touchN:"#50ff80",touchF:"#50c0ff",center:"#ffe040",boxN:"#c080ff",boxF:"#8a7bff",circle:"#ff80c0"};
const cv=document.getElementById("cv"),ctx=cv.getContext("2d");
const lcv=document.getElementById("lcv"),lctx=lcv.getContext("2d"),loupe=document.getElementById("loupe");
const pvcv=document.getElementById("prev"),pctx=pvcv.getContext("2d");
const scv=document.getElementById("schema"),sctx=scv.getContext("2d");
let fi=0,img=new Image(),view={x:0,y:0,s:1},data={},active=0,camMode=false,s0=1;
const PL=34,PW=18;
function key(){return FRAMES[fi]}
function cur(){return data[key()]}
function load(){img=new Image();img.onload=()=>{resetView();setTimeout(refreshPreview,30);};img.src=IMG_DATA[key()];
 if(!cur())data[key()]={lines:{},cam:null,absent:{}};if(!cur().absent)cur().absent={};active=0;camMode=!cur().cam;renderList();drawSchema();status();}
function resetView(){const w=document.getElementById("cvwrap");cv.width=w.clientWidth;cv.height=w.clientHeight;
 view.s=Math.min(cv.width/img.width,cv.height/img.height);s0=view.s;
 view.x=(cv.width-img.width*view.s)/2;view.y=(cv.height-img.height*view.s)/2;syncZoom();draw();}
function setZoom(m,ax,ay){m=Math.max(0.4,Math.min(14,m));const cx=ax==null?cv.width/2:ax,cy=ay==null?cv.height/2:ay;
 const ix=(cx-view.x)/view.s,iy=(cy-view.y)/view.s;view.s=s0*m;view.x=cx-ix*view.s;view.y=cy-iy*view.s;syncZoom();draw();}
function syncZoom(){const z=document.getElementById("zoom");if(z)z.value=(view.s/s0);
 const l=document.getElementById("zlbl");if(l)l.textContent=(view.s/s0).toFixed(1)+"×";}
function zReset(){resetView();}
function draw(){ctx.fillStyle="#000";ctx.fillRect(0,0,cv.width,cv.height);
 ctx.save();ctx.translate(view.x,view.y);ctx.scale(view.s,view.s);ctx.drawImage(img,0,0);ctx.restore();
 const d=cur();
 for(const id in d.lines){const pts=d.lines[id];ctx.strokeStyle=COLS[id]||"#fff";ctx.fillStyle=ctx.strokeStyle;ctx.lineWidth=2.5;
   ctx.beginPath();pts.forEach((p,i)=>{const X=p[0]*view.s+view.x,Y=p[1]*view.s+view.y;i?ctx.lineTo(X,Y):ctx.moveTo(X,Y)});ctx.stroke();
   pts.forEach(p=>{ctx.beginPath();ctx.arc(p[0]*view.s+view.x,p[1]*view.s+view.y,3.5,0,7);ctx.fill();});}
 const C=corners();if(C)for(const k in C){const p=C[k];ctx.fillStyle="#ffff00";
   ctx.beginPath();ctx.arc(p[0]*view.s+view.x,p[1]*view.s+view.y,6,0,7);ctx.fill();}
 }
function refreshPreview(){updatePreview(corners());}
// ---- kamera + şema ----
function geom(){const M=40;return {mx:M,my:M,uw:scv.width-2*M,uh:scv.height-2*M};}
function P2C(X,Y){const g=geom();return [g.mx+(X/PL)*g.uw, g.my+(1-Y/PW)*g.uh];}   // Y=0 altta
function C2P(px,py){const g=geom();return [(px-g.mx)/g.uw*PL, (1-(py-g.my)/g.uh)*PW];}
function nf(){const c=cur()&&cur().cam;const cx=c?c[0]:0,cy=c?c[1]:0;
 return {ngX:cx<PL/2?0:PL, fgX:cx<PL/2?PL:0, ntY:cy<PW/2?0:PW, ftY:cy<PW/2?PW:0};}
function drawSchema(){const W=scv.width,H=scv.height;sctx.fillStyle="#08130c";sctx.fillRect(0,0,W,H);
 const a=P2C(0,0),b=P2C(PL,PW);sctx.fillStyle="#0c2a16";
 sctx.fillRect(Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.abs(b[0]-a[0]),Math.abs(b[1]-a[1]));
 const F=nf(),aid=LINES[active].id,c=cur()&&cur().cam;
 const seg=(x0,y0,x1,y1,col,on)=>{const p=P2C(x0,y0),q=P2C(x1,y1);sctx.strokeStyle=on?col:"#3a5a44";
   sctx.lineWidth=on?4:1.5;if(on){sctx.shadowColor=col;sctx.shadowBlur=11;}else sctx.shadowBlur=0;
   sctx.beginPath();sctx.moveTo(p[0],p[1]);sctx.lineTo(q[0],q[1]);sctx.stroke();sctx.shadowBlur=0;};
 seg(F.ngX,0,F.ngX,PW,COLS.goalN,aid=="goalN");
 seg(F.fgX,0,F.fgX,PW,COLS.goalF,aid=="goalF");
 seg(0,F.ntY,PL,F.ntY,COLS.touchN,aid=="touchN");
 seg(0,F.ftY,PL,F.ftY,COLS.touchF,aid=="touchF");
 seg(PL/2,0,PL/2,PW,COLS.center,aid=="center");
 const bx=F.ngX==0?5:PL-5;
 seg(bx,PW/2-5,bx,PW/2+5,COLS.boxN,aid=="boxN");
 seg(F.ngX,PW/2-5,bx,PW/2-5,COLS.boxN,aid=="boxN");seg(F.ngX,PW/2+5,bx,PW/2+5,COLS.boxN,aid=="boxN");
 const bxf=F.fgX==0?5:PL-5;   // uzak ceza (simetri)
 seg(bxf,PW/2-5,bxf,PW/2+5,COLS.boxF,aid=="boxF");
 seg(F.fgX,PW/2-5,bxf,PW/2-5,COLS.boxF,aid=="boxF");seg(F.fgX,PW/2+5,bxf,PW/2+5,COLS.boxF,aid=="boxF");
 const cc=P2C(PL/2,PW/2),ce=P2C(PL/2,PW/2+3),rr=Math.hypot(ce[0]-cc[0],ce[1]-cc[1]);
 sctx.strokeStyle=aid=="circle"?COLS.circle:"#3a5a44";sctx.lineWidth=aid=="circle"?3:1.5;
 if(aid=="circle"){sctx.shadowColor=COLS.circle;sctx.shadowBlur=11;}sctx.beginPath();sctx.arc(cc[0],cc[1],rr,0,7);sctx.stroke();sctx.shadowBlur=0;
 if(c){const cp=P2C(c[0],c[1]),ctr=P2C(PL/2,PW/2);
   sctx.strokeStyle="rgba(255,210,74,.45)";sctx.setLineDash([4,4]);sctx.lineWidth=1.5;
   sctx.beginPath();sctx.moveTo(cp[0],cp[1]);sctx.lineTo(ctr[0],ctr[1]);sctx.stroke();sctx.setLineDash([]);
   sctx.fillStyle="#ffd24a";sctx.beginPath();sctx.arc(cp[0],cp[1],7,0,7);sctx.fill();
   sctx.fillStyle="#000";sctx.font="bold 11px Arial";sctx.textAlign="center";sctx.fillText("📷",cp[0],cp[1]+4);
   sctx.fillStyle="#ffd24a";sctx.font="bold 10px Arial";sctx.fillText("KAMERA",cp[0],cp[1]-12);sctx.textAlign="left";}
 if(camMode||!c){sctx.fillStyle="rgba(0,0,0,.58)";sctx.fillRect(0,0,W,H);
   sctx.fillStyle="#ffd24a";sctx.font="bold 14px Arial";sctx.textAlign="center";
   sctx.fillText("📷 Kamera nerede? Şemaya tıkla",W/2,H/2-6);
   sctx.font="11px Arial";sctx.fillStyle="#cde";sctx.fillText("sahaya hangi köşe/kenardan baktığını işaretle",W/2,H/2+14);sctx.textAlign="left";}
 const cs=document.getElementById("cstat");
 cs.textContent=c?("kamera yerleşti — yakın kale "+(F.ngX==0?"sol":"sağ")+" dipte, yakın taç "+(F.ntY==0?"alt":"üst")+" kenarda (şemada)"):"önce kamerayı yerleştir";}
function camPlace(){camMode=true;drawSchema();}
scv.addEventListener("click",e=>{const r=scv.getBoundingClientRect();
 const px=(e.clientX-r.left)*scv.width/r.width,py=(e.clientY-r.top)*scv.height/r.height;
 if(camMode||!cur().cam){const p=C2P(px,py);cur().cam=[p[0],p[1]];camMode=false;persist();renderList();drawSchema();refreshPreview();}});
// ---- geometri ----
function fitLine(pts){let n=pts.length,sx=0,sy=0;pts.forEach(p=>{sx+=p[0];sy+=p[1]});const mx=sx/n,my=sy/n;
 let sxx=0,sxy=0,syy=0;pts.forEach(p=>{const dx=p[0]-mx,dy=p[1]-my;sxx+=dx*dx;sxy+=dx*dy;syy+=dy*dy});
 const th=0.5*Math.atan2(2*sxy,sxx-syy);const a=Math.sin(th),b=-Math.cos(th);return [a,b,-(a*mx+b*my)];}
function inter(l1,l2){const [a1,b1,c1]=l1,[a2,b2,c2]=l2;const d=a1*b2-a2*b1;if(Math.abs(d)<1e-9)return null;
 return [(b1*c2-b2*c1)/d,(a2*c1-a1*c2)/d];}
function corners(){const d=cur().lines;
 if(!(d.goalN&&d.goalN.length>=2&&d.touchN&&d.touchN.length>=2&&d.touchF&&d.touchF.length>=2))return null;
 const gN=fitLine(d.goalN),tN=fitLine(d.touchN),tF=fitLine(d.touchF);
 const C={c_nN:inter(gN,tN),c_nF:inter(gN,tF)};
 if(d.goalF&&d.goalF.length>=2){const gF=fitLine(d.goalF);C.c_fN=inter(gF,tN);C.c_fF=inter(gF,tF);}
 for(const k in C)if(!C[k])return null;return C;}
function homography(src,dst){let A=[],b=[];for(let i=0;i<4;i++){const[x,y]=src[i],[u,v]=dst[i];
  A.push([x,y,1,0,0,0,-u*x,-u*y]);b.push(u);A.push([0,0,0,x,y,1,-v*x,-v*y]);b.push(v);}
 const h=solve(A,b);if(!h)return null;return [h[0],h[1],h[2],h[3],h[4],h[5],h[6],h[7],1];}
function solve(A,b){const n=8;for(let i=0;i<n;i++)A[i]=A[i].concat([b[i]]);
 for(let c=0;c<n;c++){let p=c;for(let r=c+1;r<n;r++)if(Math.abs(A[r][c])>Math.abs(A[p][c]))p=r;
  [A[c],A[p]]=[A[p],A[c]];if(Math.abs(A[c][c])<1e-9)return null;
  for(let r=0;r<n;r++)if(r!=c){const f=A[r][c]/A[c][c];for(let k=c;k<=n;k++)A[r][k]-=f*A[c][k];}}
 return A.map((row,i)=>row[n]/row[i]);}
function invert3(h){const[a,b,c,d,e,f,g,i,j]=h;const A=e*j-f*i,B=c*i-b*j,Cc=b*f-c*e,D=f*g-d*j,E=a*j-c*g,F=c*d-a*f,G=d*i-e*g,Hh=b*g-a*i,I=a*e-b*d;
 const det=a*A+b*D+c*G;if(Math.abs(det)<1e-9)return null;return [A/det,B/det,Cc/det,D/det,E/det,F/det,G/det,Hh/det,I/det];}
function updatePreview(C){const W=pvcv.width,H=pvcv.height;pctx.fillStyle="#08130c";pctx.fillRect(0,0,W,H);
 const sc=Math.min((W-30)/PL,(H-26)/PW),ox=(W-PL*sc)/2,oy=(H-PW*sc)/2;
 const SX=(X,Y)=>[ox+X*sc, oy+(PW-Y)*sc];
 const cam=cur().cam, fx=!!(cam&&cam[0]>PL/2), fy=!!(cam&&cam[1]>PW/2);   // kamera tarafına göre yönlendir (ayna düzelt)
 const MX=(X,Y)=>SX(fx?PL-X:X, fy?PW-Y:Y);
 if(!C||!C.c_fN){drawPitchFrame(ox,oy,sc);
   document.getElementById("pstat").textContent="4 köşe için: yakın+uzak kale + yakın+uzak taç işaretle";return;}
 const src=[C.c_nN,C.c_nF,C.c_fN,C.c_fF],dst=[MX(0,0),MX(0,PW),MX(PL,0),MX(PL,PW)];
 const H2=homography(src,dst);if(!H2){drawPitchFrame(ox,oy,sc);document.getElementById("pstat").textContent="homografi çözülemedi";return;}
 const inv=invert3(H2);let warped=false;
 if(inv){try{const tmp=document.createElement("canvas");tmp.width=img.width;tmp.height=img.height;
   const tc=tmp.getContext("2d");tc.drawImage(img,0,0);const sd=tc.getImageData(0,0,img.width,img.height).data;
   const out=pctx.createImageData(W,H);
   for(let y=0;y<H;y++)for(let x=0;x<W;x++){const w=inv[6]*x+inv[7]*y+inv[8];
     const ix=(inv[0]*x+inv[1]*y+inv[2])/w,iy=(inv[3]*x+inv[4]*y+inv[5])/w;
     if(ix>=0&&ix<img.width&&iy>=0&&iy<img.height){const si=((iy|0)*img.width+(ix|0))*4,di=(y*W+x)*4;
       out.data[di]=sd[si];out.data[di+1]=sd[si+1];out.data[di+2]=sd[si+2];out.data[di+3]=255;}}
   pctx.putImageData(out,0,0);warped=true;}catch(err){warped=false;}}
 drawPitchFrame(ox,oy,sc);
 document.getElementById("pstat").textContent=warped?"✓ canlı 2D kuş-bakışı":"✓ köşeler tamam";}
function drawPitchFrame(ox,oy,sc){pctx.strokeStyle="#ffe040";pctx.lineWidth=1.6;pctx.strokeRect(ox,oy,PL*sc,PW*sc);
 pctx.beginPath();pctx.moveTo(ox+PL*sc/2,oy);pctx.lineTo(ox+PL*sc/2,oy+PW*sc);pctx.stroke();
 pctx.beginPath();pctx.arc(ox+PL*sc/2,oy+PW*sc/2,3*sc,0,7);pctx.stroke();}
function renderList(){const el=document.getElementById("list");el.innerHTML="";
 LINES.forEach((L,i)=>{const has=cur().lines[L.id];const ab=(cur().absent||{})[L.id];const d=document.createElement("div");
  d.className="ln"+(i==active?" active":"")+(has?" done":"");
  if(ab)d.style.opacity="0.45";
  const abTxt=ab=="yok"?"YOK":(ab=="gorunmuyor"?"GÖRMÜYORUM":"—");
  const abCol=ab=="yok"?"#e05555":(ab=="gorunmuyor"?"#e0a030":"#666");
  d.innerHTML="<span class='sw' style='background:"+COLS[L.id]+"'></span><span"+(ab?" style='text-decoration:line-through'":"")+">"+L.t+"</span><small>"+(has?has.length+" nk":"")+"</small>"+
   "<button class='abtn' style='margin-left:auto;font-size:10px;padding:1px 5px;border:1px solid "+abCol+";color:"+abCol+";background:transparent;border-radius:3px;cursor:pointer'>"+abTxt+"</button>";
  d.querySelector(".abtn").onclick=(e)=>{e.stopPropagation();const a=cur().absent;
   a[L.id]=a[L.id]=="yok"?"gorunmuyor":(a[L.id]=="gorunmuyor"?undefined:"yok");
   if(!a[L.id])delete a[L.id];persist();renderList();};
  d.onclick=()=>{active=i;renderList();drawSchema();};el.appendChild(d);});}
function status(){document.getElementById("status").textContent=(fi+1)+"/"+FRAMES.length+" "+key();}
function toImg(e){const r=cv.getBoundingClientRect();return [(e.clientX-r.left-view.x)/view.s,(e.clientY-r.top-view.y)/view.s];}
cv.addEventListener("click",e=>{if(panned)return;if(!cur().cam){alert("Önce sağ üstteki şemada kamerayı yerleştir.");return;}
 const[x,y]=toImg(e);const id=LINES[active].id;
 if(!cur().lines[id])cur().lines[id]=[];cur().lines[id].push([x,y]);renderList();draw();refreshPreview();persist();});
cv.addEventListener("wheel",e=>{e.preventDefault();const r=cv.getBoundingClientRect();
 const mx=e.clientX-r.left,my=e.clientY-r.top;setZoom((view.s/s0)*(e.deltaY<0?1.12:1/1.12),mx,my);},{passive:false});
document.getElementById("zoom").addEventListener("input",e=>setZoom(parseFloat(e.target.value)));
let panning=false,panned=false,pst=null;
cv.addEventListener("mousedown",e=>{if(e.button==1||e.shiftKey||e.button==2){panning=true;panned=false;pst={x:e.clientX-view.x,y:e.clientY-view.y};}});
window.addEventListener("mouseup",()=>{panning=false;setTimeout(()=>panned=false,30);pst=null;});
window.addEventListener("contextmenu",e=>{if(e.target==cv)e.preventDefault();});
window.addEventListener("mousemove",e=>{if(pst){view.x=e.clientX-pst.x;view.y=e.clientY-pst.y;panned=true;draw();}
 const r=cv.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom){loupe.style.display="none";return;}
 loupe.style.display="block";loupe.style.left=(e.clientX+18)+"px";loupe.style.top=(e.clientY-178)+"px";
 const[ix,iy]=toImg(e);lctx.fillStyle="#000";lctx.fillRect(0,0,160,160);lctx.drawImage(img,ix-16,iy-16,32,32,0,0,160,160);
 lctx.strokeStyle=COLS[LINES[active].id];lctx.lineWidth=1;lctx.beginPath();lctx.moveTo(80,0);lctx.lineTo(80,160);lctx.moveTo(0,80);lctx.lineTo(160,80);lctx.stroke();});
function undo(){const id=LINES[active].id;const a=cur().lines[id];if(a&&a.length){a.pop();if(!a.length)delete cur().lines[id];}renderList();draw();refreshPreview();persist();}
function clrLine(){delete cur().lines[LINES[active].id];renderList();draw();refreshPreview();persist();}
function next(){if(fi<FRAMES.length-1){fi++;load();}}function prev(){if(fi>0){fi--;load();}}
function persist(){localStorage.setItem("calib_lines_v4",JSON.stringify(data));}
function save(){persist();const blob=new Blob([JSON.stringify(data,null,1)],{type:"application/json"});
 const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download="calib_lines_new.json";a.click();}
window.addEventListener("keydown",e=>{if(e.target.tagName=="INPUT")return;if(e.key=="ArrowRight")next();if(e.key=="ArrowLeft")prev();if(e.key=="z")undo();});
window.addEventListener("resize",resetView);
try{const s=localStorage.getItem("calib_lines_v4");if(s)data=JSON.parse(s);}catch(_){}
load();
</script></body></html>"""

HTML = HTML.replace("__IMG_DATA__", json.dumps(img_data)).replace("__FRAMES__", json.dumps(FRAMES))
out = os.path.join(HERE, "labeler_embed.html")
with open(out, "w") as f:
    f.write(HTML)
print("YAZILDI ->", out, f"({os.path.getsize(out)//1024//1024} MB)")
