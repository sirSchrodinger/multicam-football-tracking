"""Zero-setup TAKIM-TAKTİK top-down: jersey-renk 2-takım ayrımı + takım-renkli konumlar
+ takım territory (ısı). Sahayı yukarıdan, iki takım ayrı renk. Elle-etiket YOK.
Kullanım: python team_tactical.py <video> <fr0> <n_sec> <out.jpg> [label]"""
import sys, os, json, subprocess, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
from top_down_view import und_norm, topdown  # noqa
from sklearn.cluster import KMeans

video = sys.argv[1]; FR0 = int(sys.argv[2]); NSEC = float(sys.argv[3]); OUT = sys.argv[4]
label = sys.argv[5] if len(sys.argv) > 5 else os.path.basename(video)
L, S, M = 34.0, 20, 3.0

cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS) or 25
cap.set(cv2.CAP_PROP_POS_FRAMES, FR0); ok, f0 = cap.read()
tmp_img = "/tmp/tt_frame.jpg"; cv2.imwrite(tmp_img, f0)
if os.path.exists(os.path.join(HERE, "seg_auto_solved.json")):
    os.remove(os.path.join(HERE, "seg_auto_solved.json"))
subprocess.run([sys.executable, "infer_auto2.py", tmp_img], cwd=HERE,
               env=dict(os.environ, SEG_CK="seg_unet_v3.pth"), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
rec = json.load(open(os.path.join(HERE, "seg_auto_solved.json")))[os.path.basename(tmp_img)]
k1, k2 = float(rec["k1"]), float(rec["k2"]); H = np.array(rec["H"], float); Wp = float(rec["Wp"])
h, w = f0.shape[:2]; cx, cy, s = w/2, h/2, w/2
bg = topdown(f0, rec, None); WS, LS = bg.shape[0], bg.shape[1]
def MX(X, Y): return (int((M+X)*S), int(WS-(M+Y)*S))
def feet_to_top(feet):
    m = und_norm(np.asarray(feet, float), k1, k2, cx, cy, s); z = H[2,0]*m[:,0]+H[2,1]*m[:,1]+H[2,2]
    return np.stack([(H[0,0]*m[:,0]+H[0,1]*m[:,1]+H[0,2])/z, (H[1,0]*m[:,0]+H[1,1]*m[:,1]+H[1,2])/z], 1)

from viz.detect_frame import FrameDetector
det = FrameDetector()

# jersey renk + pozisyon topla
cols, poss = [], []
frames = list(range(FR0, FR0+int(NSEC*fps), max(1, int(fps/4))))
for fr in frames:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, frame = cap.read()
    if not ok:
        continue
    r = det.detect(frame)
    if not len(r["boxes"]):
        continue
    tp = feet_to_top(r["feet"])
    for b, (X, Y) in zip(r["boxes"], tp):
        if not (-M < X < L+M and -M < Y < Wp+M):
            continue
        x0, y0, x1, y1 = [int(v) for v in b]
        yb = y0 + int((y1-y0)*0.15); ye = y0 + int((y1-y0)*0.55)  # üst-gövde (forma)
        crop = frame[max(0, yb):ye, max(0, x0):x1]
        if crop.size < 30:
            continue
        cols.append(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).mean(0)); poss.append((X, Y))
cols = np.array(cols); poss = np.array(poss)
print(f"{label}: {len(cols)} oyuncu-örnek, calib Wp={Wp:.0f}", flush=True)
# 2-takım KMeans (HSV hue+sat ağırlıklı)
feat = cols.copy(); feat[:, 0] *= 2.0  # hue ağırlık
km = KMeans(2, n_init=5, random_state=0).fit(feat); lab = km.labels_
# takım renkleri: küme median BGR (görsel için)
tcol = []
for t in [0, 1]:
    hsvm = np.median(cols[lab == t], 0).astype(np.uint8).reshape(1, 1, 3)
    tcol.append([int(v) for v in cv2.cvtColor(hsvm, cv2.COLOR_HSV2BGR)[0, 0]])
print(f"takım renkleri (BGR): {tcol}  denge: {np.bincount(lab)}", flush=True)

# çizim: territory ısı (takım başına) + noktalar
out = bg.copy()
for t in [0, 1]:
    heat = np.zeros((WS, LS), np.float32)
    for (X, Y) in poss[lab == t]:
        p = MX(X, Y)
        if 0 <= p[1] < WS and 0 <= p[0] < LS:
            cv2.circle(heat, p, 16, 1, -1)
    heat = cv2.GaussianBlur(heat, (0, 0), 14); heat /= heat.max()+1e-9
    tint = np.zeros_like(out); tint[:] = tcol[t]
    out = np.where((heat[..., None] > 0.15), cv2.addWeighted(out, 0.6, tint, 0.4, 0), out).astype(np.uint8)
for (X, Y), t in zip(poss, lab):
    p = MX(X, Y); cv2.circle(out, p, 8, tcol[t], -1); cv2.circle(out, p, 8, (255, 255, 255), 1)
cv2.putText(out, f"{label} zero-setup TAKIM-TAKTIK (2 takim jersey-renk, {np.bincount(lab).tolist()})",
            (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
cv2.imwrite(OUT, out); print(f"yazıldı {OUT}", flush=True)
