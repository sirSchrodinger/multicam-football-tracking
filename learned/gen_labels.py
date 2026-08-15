"""Kalibre bir sahadan learned-model eğitim etiketleri üret (tıklama YOK — calib'den).
cankaya: doğrulanmış calib → 11 halısaha-keypoint + çizgi seti reprojekte → eğitim
çifti (frame.jpg + frame.json). Doğrulama: keypoint'ler gerçek saha-özelliklerinde mi?
Kullanım: python -m learned.gen_labels
"""
import sys
import os
import json
import cv2
import numpy as np
sys.path.insert(0, ".")
from pitch.homography import PitchHomography
from learned.kp_template import keypoints_m, KP_NAMES, line_segments_m

OUT = "learned/data/cankaya"
os.makedirs(OUT, exist_ok=True)
homo = PitchHomography.load("calib/cankaya_cam2_FINAL.json")
L, W = 36.5, 17.0        # cankaya template koordinatı (calib bu uzayda)
CIRCLE_R = 3.0

kp_m = keypoints_m(L, W, CIRCLE_R)
kp_px = homo.pitch_to_pixel(kp_m, distorted=True)   # ham-piksel
segs_m = line_segments_m(L, W, CIRCLE_R)

VIDEO = "raw/cankaya_cam2.mp4"
cap = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
N = 40
idxs = np.linspace(int(total * 0.1), int(total * 0.9), N).astype(int)

# statik kamera → keypoint/çizgi TÜM karelerde aynı; kare başına yaz (farklı oyuncu/ışık)
ann = {"keypoints": {KP_NAMES[i]: [float(kp_px[i, 0]), float(kp_px[i, 1])]
                     for i in range(len(kp_px))},
       "lines": {name: [[float(homo.pitch_to_pixel(np.array([p]), distorted=True)[0, 0]),
                         float(homo.pitch_to_pixel(np.array([p]), distorted=True)[0, 1])]
                        for p in pts]
                 for name, pts in segs_m.items()},
       "img_wh": [W0, H0], "venue": "cankaya", "L": L, "W": W}

saved = 0
for fi in idxs:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, fr = cap.read()
    if not ok:
        continue
    cv2.imwrite(f"{OUT}/f{fi:06d}.jpg", fr)
    json.dump({**ann, "frame": int(fi)}, open(f"{OUT}/f{fi:06d}.json", "w"))
    saved += 1
cap.release()
print(f"{saved} eğitim çifti -> {OUT}/", flush=True)

# --- DOĞRULAMA: keypoint'ler gerçek saha-özelliklerinde mi? ---
cap = cv2.VideoCapture(VIDEO); cap.set(cv2.CAP_PROP_POS_FRAMES, int(idxs[len(idxs)//2]))
ok, fr = cap.read(); cap.release()
vis = fr.copy()
for name, pts in ann["lines"].items():
    pts = np.array(pts, int)
    for i in range(len(pts) - (0 if name == "Circle central" else 1)):
        cv2.line(vis, tuple(pts[i]), tuple(pts[(i+1) % len(pts)]), (0, 200, 255), 2)
for i, (x, y) in enumerate(kp_px):
    cv2.circle(vis, (int(x), int(y)), 10, (255, 0, 255), -1)
    cv2.circle(vis, (int(x), int(y)), 10, (255, 255, 255), 2)
    cv2.putText(vis, KP_NAMES[i][:6], (int(x)+10, int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
cv2.putText(vis, "EGITIM ETIKETI dogrulama: 11 keypoint (magenta) gercek ozellik uzerinde mi?",
            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.imwrite("learned/data/cankaya_label_verify.jpg", vis)
print("doğrulama -> learned/data/cankaya_label_verify.jpg", flush=True)
