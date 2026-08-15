"""UÇTAN-UCA zero-setup maç analizi (YENİ saha): seg-auto calib + basit-tracking +
top-down + heatmap + per-oyuncu mesafe. Tek komut. Elle-etiket YOK.
Basit tracker: kare-kare metrik-konum greedy-NN eşleme (max-atlama kapılı) → sürekli track.
Kullanım: python match_analysis_zs.py <video> <fr0> <n_sec> <out_prefix> [label]"""
import sys, os, json, subprocess, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
from top_down_view import und_norm, topdown  # noqa
from scipy.optimize import linear_sum_assignment

video = sys.argv[1]; FR0 = int(sys.argv[2]); NSEC = float(sys.argv[3]); PFX = sys.argv[4]
label = sys.argv[5] if len(sys.argv) > 5 else os.path.basename(video)
L, S, M = 34.0, 20, 3.0

cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS) or 25
cap.set(cv2.CAP_PROP_POS_FRAMES, FR0); ok, f0 = cap.read()
tmp_img = "/tmp/ma_frame.jpg"; cv2.imwrite(tmp_img, f0)
_solved = os.path.join(HERE, "seg_auto_solved.json")
if os.path.exists(_solved):
    os.remove(_solved)
subprocess.run([sys.executable, "infer_auto2.py", tmp_img], cwd=HERE,
               env=dict(os.environ, SEG_CK="seg_unet_v3.pth"),
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
rec = json.load(open(_solved))[os.path.basename(tmp_img)]
k1, k2 = float(rec["k1"]), float(rec["k2"]); H = np.array(rec["H"], float); Wp = float(rec["Wp"])
h, w = f0.shape[:2]; cx, cy, s = w/2, h/2, w/2
bg = topdown(f0, rec, None); WS, LS = bg.shape[0], bg.shape[1]
def MX(X, Y): return (int((M+X)*S), int(WS-(M+Y)*S))
def feet_to_top(feet):
    m = und_norm(np.asarray(feet, float), k1, k2, cx, cy, s)
    z = H[2,0]*m[:,0]+H[2,1]*m[:,1]+H[2,2]
    return np.stack([(H[0,0]*m[:,0]+H[0,1]*m[:,1]+H[0,2])/z, (H[1,0]*m[:,0]+H[1,1]*m[:,1]+H[1,2])/z], 1)
print(f"{label}: zero-setup calib k1={k1:.2f} Wp={Wp:.1f}", flush=True)

from viz.detect_frame import FrameDetector
det = FrameDetector()

# --- basit greedy-NN tracker (metrik uzayda) ---
OFPS = 5; step = max(1, int(fps/OFPS)); dt = step/fps
frames = list(range(FR0, FR0+int(NSEC*fps), step))
tracks = {}   # tid -> {"pos":(X,Y), "hist":[(t,X,Y)], "last":t, "col":..}
next_tid = 0; MAXJUMP = 3.0   # kare-arası max metrik atlama (m) — vcap*dt ~ 6*0.2
PAL = [(230,159,0),(86,180,233),(0,158,115),(240,228,66),(0,114,178),(213,94,0),(204,121,167),
       (255,109,182),(109,182,255),(182,219,109),(219,109,182),(109,219,182),(255,180,90),(90,180,255)]
vw = None; tmp = "/tmp/ma_raw.mp4"; heat = np.zeros((WS, LS), np.float32); archive = []
for fi, fr in enumerate(frames):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, frame = cap.read()
    if not ok:
        continue
    r = det.detect(frame); t = fi*dt
    dets = []
    if len(r["feet"]):
        mp = feet_to_top(r["feet"])
        dets = [(X, Y) for X, Y in mp if -M < X < L+M and -M < Y < Wp+M]
    # greedy-NN eşleme (Hungarian, MAXJUMP kapılı)
    tids = list(tracks.keys())
    if tids and dets:
        C = np.array([[np.hypot(tracks[ti]["pos"][0]-dx, tracks[ti]["pos"][1]-dy) for dx, dy in dets] for ti in tids])
        ri, ci = linear_sum_assignment(C); assigned = set()
        for a, b in zip(ri, ci):
            if C[a, b] < MAXJUMP:
                ti = tids[a]; tracks[ti]["pos"] = dets[b]; tracks[ti]["hist"].append((t, *dets[b])); tracks[ti]["last"] = t
                assigned.add(b)
        for b, (dx, dy) in enumerate(dets):
            if b not in assigned:
                tracks[next_tid] = {"pos": (dx, dy), "hist": [(t, dx, dy)], "last": t, "col": PAL[next_tid % len(PAL)]}; next_tid += 1
    elif dets:
        for dx, dy in dets:
            tracks[next_tid] = {"pos": (dx, dy), "hist": [(t, dx, dy)], "last": t, "col": PAL[next_tid % len(PAL)]}; next_tid += 1
    # eski track'leri arşivle (>1.5s görünmedi) — mesafe için geçmişi koru
    for ti in [t_ for t_ in tracks if t-tracks[t_]["last"] > 1.5]:
        archive.append(tracks.pop(ti))
    # çiz + heatmap
    canvas = bg.copy()
    for ti, tr in tracks.items():
        p = MX(*tr["pos"]); cv2.circle(canvas, p, 9, tr["col"], -1); cv2.circle(canvas, p, 9, (255,255,255), 1)
        for j in range(max(1, len(tr["hist"])-8), len(tr["hist"])):
            a = MX(tr["hist"][j-1][1], tr["hist"][j-1][2]); b = MX(tr["hist"][j][1], tr["hist"][j][2]); cv2.line(canvas, a, b, tr["col"], 2)
        if 0 <= p[1] < WS and 0 <= p[0] < LS:
            cv2.circle(heat, p, 15, 1, -1)
    cv2.putText(canvas, f"{label} zero-setup maç analizi | {len(dets)} oyuncu | t={t:.0f}s", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    o = cv2.resize(frame, (int(w*WS/h), WS))
    out = np.hstack([o, np.full((WS, 5, 3), 40, np.uint8), canvas])
    if vw is None:
        vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), OFPS, (out.shape[1], out.shape[0]))
    vw.write(out)
    if fi % 10 == 0:
        print(f"  {fi+1}/{len(frames)} · {len(tracks)} aktif track", flush=True)
cap.release(); vw.release()

# per-track mesafe (arşiv + hayatta kalanlar), kapsam-filtreli
all_tr = archive + list(tracks.values())
dists = []
for tr in all_tr:
    hist = tr["hist"]
    if len(hist) < 3:
        continue
    hh = np.array(hist)  # (n,3): t,X,Y
    span = hh[-1, 0]-hh[0, 0]
    step = np.hypot(np.diff(hh[:, 1]), np.diff(hh[:, 2])); dtv = np.diff(hh[:, 0])
    spd = step/np.maximum(dtv, 1e-3); ok = spd < 8.0
    dists.append({"dist": float(step[ok].sum()), "span": float(span), "cov": span/(NSEC)})
dists.sort(key=lambda d: -d["dist"])
long = [d for d in dists if d["cov"] > 0.5]
print(f"\n=== {label} zero-setup per-track MESAFE ({NSEC:.0f}s) ===", flush=True)
for d in dists[:14]:
    print(f"  dist={d['dist']:.0f}m span={d['span']:.0f}s cov={d['cov']:.2f}", flush=True)
if long:
    dm = np.array([d["dist"] for d in long])
    print(f"uzun-track ({len(long)}) med={np.median(dm):.0f}m (yeni-saha, zero-setup, basit-tracker)", flush=True)
subprocess.run(["ffmpeg","-y","-i",tmp,"-c:v","libx264","-pix_fmt","yuv420p","-loglevel","error",f"{PFX}_video.mp4"], check=True)
heat = cv2.GaussianBlur(heat, (0,0), 13); heat /= heat.max()+1e-9
hc = cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_JET)
hm_out = cv2.addWeighted(bg, 0.5, hc, 0.6, 0)
cv2.putText(hm_out, f"{label} zero-setup HEATMAP", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
cv2.imwrite(f"{PFX}_heatmap.jpg", hm_out)
print(f"BİTTİ -> {PFX}_video.mp4 + {PFX}_heatmap.jpg ({len(frames)} kare)", flush=True)
