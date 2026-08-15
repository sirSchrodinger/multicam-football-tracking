"""Zero-setup yukarıdan-2D HEATMAP: seg-auto calib + pencere boyunca oyuncu-konumu
biriktir -> sahada 'aksiyon nerede' (kuş-bakışı). Sıfır kurulum, elle-etiket yok.
Kullanım: python top_down_heatmap.py <video> <fr0> <n_sec> <out.jpg> [label]"""
import sys, os, json, subprocess, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
from top_down_view import und_norm, topdown  # noqa: E402

video = sys.argv[1]; FR0 = int(sys.argv[2]); NSEC = float(sys.argv[3]); OUT = sys.argv[4]
label = sys.argv[5] if len(sys.argv) > 5 else os.path.basename(video)
L, S, M = 34.0, 20, 3.0

cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS) or 25
cap.set(cv2.CAP_PROP_POS_FRAMES, FR0); ok, f0 = cap.read()
tmp_img = "/tmp/segcalib_hm.jpg"; cv2.imwrite(tmp_img, f0)
_solved = os.path.join(HERE, "seg_auto_solved.json")
if os.path.exists(_solved):
    os.remove(_solved)
env = dict(os.environ, SEG_CK="seg_unet_v3.pth")
subprocess.run([sys.executable, "infer_auto2.py", tmp_img], cwd=HERE, env=env,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
rec = json.load(open(_solved))[os.path.basename(tmp_img)]
k1, k2 = float(rec["k1"]), float(rec["k2"]); H = np.array(rec["H"], float); Wp = float(rec["Wp"])
h, w = f0.shape[:2]; cx, cy, s = w/2, h/2, w/2
bg = topdown(f0, rec, None); WS, LS = bg.shape[0], bg.shape[1]
def MX(X, Y): return (int((M+X)*S), int(WS-(M+Y)*S))
print(f"{label}: calib k1={k1:.2f} Wp={Wp:.1f}", flush=True)

from viz.detect_frame import FrameDetector
det = FrameDetector()
def feet_to_top(feet):
    m = und_norm(np.asarray(feet, float), k1, k2, cx, cy, s)
    z = H[2, 0]*m[:, 0]+H[2, 1]*m[:, 1]+H[2, 2]
    return (H[0, 0]*m[:, 0]+H[0, 1]*m[:, 1]+H[0, 2])/z, (H[1, 0]*m[:, 0]+H[1, 1]*m[:, 1]+H[1, 2])/z

heat = np.zeros((WS, LS), np.float32); n_pos = 0
frames = list(range(FR0, FR0+int(NSEC*fps), max(1, int(fps/6))))
for i, fr in enumerate(frames):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, frame = cap.read()
    if not ok:
        continue
    r = det.detect(frame)
    if len(r["feet"]):
        mx, my = feet_to_top(r["feet"])
        for X, Y in zip(mx, my):
            if -M < X < L+M and -M < Y < Wp+M:
                px, py = MX(X, Y)
                if 0 <= py < WS and 0 <= px < LS:
                    cv2.circle(heat, (px, py), 16, 1, -1); n_pos += 1
    if i % 10 == 0:
        print(f"  {i+1}/{len(frames)} · {n_pos} konum", flush=True)
cap.release()
heat = cv2.GaussianBlur(heat, (0, 0), 13); heat /= heat.max()+1e-9
hc = cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_JET)
out = cv2.addWeighted(bg, 0.5, hc, 0.6, 0)
cv2.putText(out, f"{label} — YUKARIDAN HEATMAP (zero-setup, {len(frames)} kare, {n_pos} konum)",
            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
cv2.imwrite(OUT, out); print(f"yazıldı {OUT} ({n_pos} konum)", flush=True)
