"""Yukarıdan-2D VİDEO: seg-auto calib (bir kez) + warp arka-plan (bir kez) + kare-kare
oyuncu noktaları. Sahayı yukarıdan gör, oyuncular hareket ederken. Statik kamera.
Kullanım: python top_down_video.py <video> <fr0> <n_sec> <fps> <out.mp4>"""
import sys, os, json, subprocess, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
from top_down_view import undimg, und_norm, topdown  # noqa: E402

video = sys.argv[1]; FR0 = int(sys.argv[2]); NSEC = float(sys.argv[3]); OFPS = int(sys.argv[4])
OUT = sys.argv[5]
L, S, M = 34.0, 20, 3.0

# 1) seg-auto calib (bir temiz kare) — infer_auto2 subprocess
cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS) or 25
cap.set(cv2.CAP_PROP_POS_FRAMES, FR0); ok, f0 = cap.read()
tmp_img = "/tmp/segcalib_frame.jpg"; cv2.imwrite(tmp_img, f0)
_solved = os.path.join(HERE, "seg_auto_solved.json")
if os.path.exists(_solved):
    os.remove(_solved)
env = dict(os.environ, SEG_CK="seg_unet_v3.pth")
subprocess.run([sys.executable, "infer_auto2.py", tmp_img], cwd=HERE, env=env,
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
rec = json.load(open(_solved))[os.path.basename(tmp_img)]
k1, k2 = float(rec["k1"]), float(rec["k2"]); H = np.array(rec["H"], float); Wp = float(rec["Wp"])
h, w = f0.shape[:2]; cx, cy, s = w/2, h/2, w/2
print(f"calib: k1={k1:.2f} Wp={Wp:.1f}", flush=True)

# 2) warp arka-plan (bir kez) — top_down_view.topdown ile ama oyuncu-suz
bg = topdown(f0, rec, None)
LS, WS = bg.shape[1], bg.shape[0]
def MX(X, Y): return (int((M+X)*S), int(WS-(M+Y)*S))

# 3) detektör
from viz.detect_frame import FrameDetector
det = FrameDetector(); print("detektör hazır", flush=True)

def feet_to_top(feet):
    m = und_norm(np.asarray(feet, float), k1, k2, cx, cy, s)
    z = H[2, 0]*m[:, 0]+H[2, 1]*m[:, 1]+H[2, 2]
    return (H[0, 0]*m[:, 0]+H[0, 1]*m[:, 1]+H[0, 2])/z, (H[1, 0]*m[:, 0]+H[1, 1]*m[:, 1]+H[1, 2])/z

step = max(1, int(round(fps/OFPS)))
frames = list(range(FR0, FR0+int(NSEC*fps), step))
vw = None; tmp = "/tmp/topvid_raw.mp4"
for k, fr in enumerate(frames):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fr); ok, frame = cap.read()
    if not ok:
        continue
    r = det.detect(frame); canvas = bg.copy()
    if len(r["feet"]):
        mx, my = feet_to_top(r["feet"])
        for X, Y in zip(mx, my):
            if -M < X < L+M and -M < Y < Wp+M:
                px, py = MX(X, Y); cv2.circle(canvas, (px, py), 9, (60, 60, 235), -1)
                cv2.circle(canvas, (px, py), 9, (255, 255, 255), 2)
    # yan yana: orijinal + top-down
    Hh = 420; o = cv2.resize(frame, (int(w*Hh/h), Hh)); t = cv2.resize(canvas, (int(LS*Hh/WS), Hh))
    cv2.putText(o, f"{len(r['feet'])} oyuncu", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2)
    cv2.putText(t, "YUKARIDAN 2D (zero-setup)", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2)
    out = np.hstack([o, np.full((Hh, 5, 3), 40, np.uint8), t])
    if vw is None:
        vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), OFPS, (out.shape[1], out.shape[0]))
    vw.write(out)
    if k % 5 == 0:
        print(f"  {k+1}/{len(frames)}", flush=True)
cap.release(); vw.release()
subprocess.run(["ffmpeg", "-y", "-i", tmp, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-loglevel", "error", OUT], check=True)
print(f"BİTTİ -> {OUT} ({len(frames)} kare @{OFPS}fps)", flush=True)
