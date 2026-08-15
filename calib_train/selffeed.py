#!/usr/bin/env python3
"""SELF-FEEDING detektör (mean-teacher + GT-siz geometrik kabul-kapısı).
Teacher rol-çizgi tahmin -> ÇİZGİ-kısıtlı joint kalibrasyon -> residual<τ ise KABUL ->
kalibrasyondan TEMİZ rol-maskesi geri-projeksiyon (geometrik-mükemmel pseudo-etiket) ->
student eğit -> teacher = EMA(student). Kabul-kapısı residual'a bakar (güvene DEĞİL) ->
confirmation-bias'a karşı. Curriculum: kolay(parlak/düşük-res)->zor. Watchdog: holdout
pass-rate + residual; plato/U-dip'te dur.

Modlar:
  gate  : havuzda kabul-kapısını çalıştır + temiz-etiket overlay (cand/_selffeed_gate.jpg). DRY-RUN.
  train : self-feeding döngüsü. Çıktı: selffeed_ckpt/student.pth + teacher.pth + log.
Kullanım: python selffeed.py gate [pool_glob]   |   python selffeed.py train
"""
import sys, os, glob, json, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from calib_train import auto_calib as AC
from calib_train.make_seg_data2 import synth_sample, real_sample, real_ok, ORDER, J, NC, CLASSES
HERE=os.path.dirname(os.path.abspath(__file__)); LF=os.path.join(HERE,"label_frames")
TW,TH=int(os.environ.get("TW","512")),int(os.environ.get("TH","288")); BASE=24
DEV="cuda" if torch.cuda.is_available() else "cpu"
TAU=float(os.environ.get("TAU","0.45"))      # kabul residual eşiği (m)
SEG2=os.path.join(HERE,"seg2_ckpt","seg2_unet.pth")
OUT=os.path.join(HERE,"selffeed_ckpt"); os.makedirs(OUT,exist_ok=True)
def cbr(i,o): return nn.Sequential(nn.Conv2d(i,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True),nn.Conv2d(o,o,3,padding=1),nn.BatchNorm2d(o),nn.ReLU(True))
class UNet(nn.Module):
    def __init__(s,nin=3,no=NC,b=BASE):
        super().__init__(); s.d1=cbr(nin,b);s.d2=cbr(b,b*2);s.d3=cbr(b*2,b*4);s.d4=cbr(b*4,b*8);s.p=nn.MaxPool2d(2)
        s.u3=nn.ConvTranspose2d(b*8,b*4,2,2);s.c3=cbr(b*8,b*4);s.u2=nn.ConvTranspose2d(b*4,b*2,2,2);s.c2=cbr(b*4,b*2)
        s.u1=nn.ConvTranspose2d(b*2,b,2,2);s.c1=cbr(b*2,b);s.h=nn.Conv2d(b,no,1)
    def forward(s,x):
        c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2));c4=s.d4(s.p(c3))
        x=s.c3(torch.cat([s.u3(c4),c3],1));x=s.c2(torch.cat([s.u2(x),c2],1));x=s.c1(torch.cat([s.u1(x),c1],1));return s.h(x)

def load(ckpt):
    n=UNet().to(DEV); n.load_state_dict(torch.load(ckpt,map_location=DEV)); n.eval(); return n
@torch.no_grad()
def predict_prob(net,img):
    h,w=img.shape[:2]; x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)[None].to(DEV)
    p=torch.sigmoid(net(x))[0].cpu().numpy(); return p,w,h

def gate_one(net,path,tau=TAU):
    img=cv2.imread(path)
    if img is None: return None
    prob,w,h=predict_prob(net,img); rec=AC.calib_from_pred(prob,w,h)
    rec["path"]=path; rec["accept"]=bool(rec.get("ok") and rec["fit"] is not None and rec["fit"]<tau)
    rec["img"]=img; rec["w"]=w; rec["h"]=h; return rec

# ---------------- GATE (dry-run) ----------------
def run_gate(pool):
    net=load(SEG2); recs=[]
    for p in pool:
        r=gate_one(net,p)
        if r is None: continue
        recs.append(r); st="KABUL" if r["accept"] else " red "
        print(f"  [{st}] {os.path.basename(p)[:24]:24s} res={('%.3f'%r['fit']) if r['fit'] is not None else r.get('reason','?'):>8s}",flush=True)
    acc=[r for r in recs if r["accept"]]
    print(f"\nKABUL {len(acc)}/{len(recs)} (τ={TAU}m)  med-res={np.median([r['fit'] for r in acc]):.3f}m" if acc else "KABUL 0")
    # overlay: ham görüntü + TEMİZ geri-projekte rol-maskesi (pseudo-etiket doğru mu?)
    COL=[(0,0,255),(0,150,255),(0,255,0),(255,120,0),(0,255,255),(255,0,200),(255,150,255)]
    tiles=[]
    for r in acc[:9]:
        m=AC.clean_label_masks(r,r["w"],r["h"]); o=r["img"].copy()
        for c in range(7): o[m[c]>80]=COL[c]
        o=cv2.resize(o,(TW,TH)); cv2.putText(o,f"res{r['fit']:.2f} {r['camside']}",(6,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,230,255),2)
        tiles.append(o)
    if tiles:
        cols=3; rows=(len(tiles)+cols-1)//cols; sheet=np.full((rows*(TH+6),cols*(TW+6),3),20,np.uint8)
        for i,t in enumerate(tiles):
            rr,cc=divmod(i,cols); sheet[rr*(TH+6):rr*(TH+6)+TH,cc*(TW+6):cc*(TW+6)+TW]=t
        cv2.imwrite(os.path.join(HERE,"cand","_selffeed_gate.jpg"),sheet); print("-> cand/_selffeed_gate.jpg")
    return recs

# ---------------- TRAIN (self-feeding) ----------------
def to_t(img,m):
    x=torch.from_numpy(cv2.resize(img,(TW,TH)).transpose(2,0,1).astype(np.float32)/255.)
    y=torch.from_numpy(np.stack([cv2.resize(m[c],(TW,TH)) for c in range(NC)]).astype(np.float32)/255.)
    return x,y
def aug(img):
    return np.clip(img.astype(np.float32)*np.random.uniform(0.7,1.2)+np.random.uniform(-14,14),0,255).astype(np.uint8)

def run_train(pool):
    """KEEP-BEST self-feeding: teacher = pool-üstünde EN İYİ student anlık-görüntüsü (baseline=seg2).
    Teacher ASLA bozulmaz (yalnız student baseline'ı KESİN geçince swap). Pseudo-etiket SADECE
    çok-temiz kalibrasyondan (τ_pseudo). Watchdog: PATIENCE refresh boyunca en-iyi artmazsa dur."""
    STEPS=int(os.environ.get("STEPS","3000")); BATCH=int(os.environ.get("BATCH","6"))
    REFRESH=int(os.environ.get("REFRESH","300")); PATIENCE=int(os.environ.get("PATIENCE","4"))
    TAU_EVAL=float(os.environ.get("TAU_EVAL","0.45")); TAU_PSEUDO=float(os.environ.get("TAU_PSEUDO","0.28"))
    reals=[n for n in ORDER if n in J and real_ok(n)]
    VAL={"AvanosHaliSaha.jpg","KaynarcaAdaHaliSah.jpg","KucukcekmeceIdmanY.jpg"}
    train_reals=[n for n in reals if n not in VAL]
    student=load(SEG2); teacher=load(SEG2)
    for p in teacher.parameters(): p.requires_grad_(False)
    opt=torch.optim.Adam(student.parameters(),5e-4); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,STEPS)
    t0=time.time(); hist=[]
    def eval_count(net,tau):
        net.eval(); c=0; fits=[]
        for p in pool:
            r=gate_one(net,p,tau)
            if r and r["accept"]: c+=1; fits.append(r["fit"])
        return c,(float(np.median(fits)) if fits else 9.9)
    def gen_pseudo(net,tau):
        net.eval(); acc=[]
        for p in pool:
            r=gate_one(net,p,tau)
            if r and r["accept"]:
                acc.append((r["img"],AC.clean_label_masks(r,r["w"],r["h"]),r["fit"]))
        return acc
    base_cnt,base_med=eval_count(teacher,TAU_EVAL); best=base_cnt
    pseudo=gen_pseudo(teacher,TAU_PSEUDO)
    print(f"BASELINE kabul {base_cnt}/{len(pool)} med {base_med:.3f}m | pseudo(τ={TAU_PSEUDO}) {len(pseudo)}",flush=True)
    torch.save(teacher.state_dict(),f"{OUT}/teacher_best.pth"); since=0
    def get_batch(step):
        xs=[];ys=[]
        for i in range(BATCH):
            u=np.random.rand()
            if u<0.40: img,m=synth_sample(step*BATCH+i)
            elif u<0.70 and pseudo: img,m,_=pseudo[np.random.randint(len(pseudo))]; img=aug(img)
            else:
                n=train_reals[np.random.randint(len(train_reals))]; img,m=real_sample(n); img=aug(img)
            x,y=to_t(img,m); xs.append(x);ys.append(y)
        return torch.stack(xs),torch.stack(ys)
    for step in range(STEPS):
        student.train(); x,y=get_batch(step); x,y=x.to(DEV),y.to(DEV)
        pred=student(x); w=1.0+35.0*y; loss=(w*F.binary_cross_entropy_with_logits(pred,y,reduction="none")).mean()
        opt.zero_grad();loss.backward();opt.step();sch.step()
        if step%100==0: print(f"step {step}/{STEPS} loss {loss.item():.4f} best {best} pseudo {len(pseudo)} {(time.time()-t0)/max(1,step+1)*1000:.0f}ms/it",flush=True)
        if step%REFRESH==0 and step>0:
            sc,smed=eval_count(student,TAU_EVAL); hist.append((step,sc,smed,best))
            if sc>best:   # KESİN iyileşme -> teacher'ı student ile değiştir, pseudo'yu yenile
                best=sc; teacher.load_state_dict(student.state_dict()); teacher.eval()
                for p in teacher.parameters(): p.requires_grad_(False)
                pseudo=gen_pseudo(teacher,TAU_PSEUDO); since=0
                torch.save(teacher.state_dict(),f"{OUT}/teacher_best.pth")
                print(f"  [refresh] student {sc} med {smed:.3f}m -> YENİ EN İYİ {best} (pseudo {len(pseudo)})",flush=True)
            else:
                since+=1; print(f"  [refresh] student {sc} med {smed:.3f}m (en iyi {best}, {since}/{PATIENCE} durgun)",flush=True)
                if since>=PATIENCE: print("  [watchdog] en-iyi artmıyor -> dur",flush=True); break
    json.dump(hist,open(f"{OUT}/selffeed_hist.json","w"))
    print(f"BİTTİ {time.time()-t0:.0f}s  BASELINE {base_cnt} -> EN İYİ {best} (+{best-base_cnt})  teacher_best.pth",flush=True)

if __name__=="__main__":
    mode=sys.argv[1] if len(sys.argv)>1 else "gate"
    g=sys.argv[2] if len(sys.argv)>2 else os.path.join(LF,"*.jpg")
    pool=sorted(glob.glob(g))
    print(f"mod={mode} havuz={len(pool)} dev={DEV} τ={TAU}",flush=True)
    if mode=="gate": run_gate(pool)
    else: run_train(pool)
