#!/usr/bin/env python3
"""Otomatik calib ÇIKARIM: gerçek kare -> çizgi-maske -> model -> keypoint -> homografi
-> flatten + 2D. MANUEL TIK YOK. Eğitilen keypoint detektörünü gerçek sahalarda test eder.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2, torch
from calib_train.synth_gen import KP_NAMES, pitch_geometry
from calib_train.model import CalibUNet

H, Wd = 288, 512
CKPT = sys.argv[2] if len(sys.argv) > 2 else "calib_train/ckpt_local/calib_unet.pth"


def turf_line_mask(img):
    """Gerçek kareden eğitim-domeniyle uyumlu çizgi-maskesi (turf-maskeli beyaz çizgi)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV); Hc, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    green = ((Hc >= 30) & (Hc <= 95) & (S > 35) & (V > 35)).astype(np.uint8) * 255
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    nc, lbl, st, _ = cv2.connectedComponentsWithStats(green)
    if nc > 1:
        big = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]); green = (lbl == big).astype(np.uint8) * 255
    green = cv2.dilate(green, np.ones((9, 9), np.uint8))
    white = ((S < 85) & (V > 150)).astype(np.uint8) * 255
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25)))
    _, tm = cv2.threshold(th, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_and(cv2.bitwise_and(white, tm), green)
    return cv2.dilate(mask, np.ones((2, 2), np.uint8))


def detect_keypoints(net, mask_small, dev, thr=0.3):
    x = torch.from_numpy((mask_small.astype(np.float32) / 255.0)[None, None]).to(dev)
    with torch.no_grad():
        hm = net(x)[0].cpu().numpy()
    kps = {}
    for i, name in enumerate(KP_NAMES):
        h = hm[i]; mx = h.max()
        if mx >= thr:
            v, u = np.unravel_index(h.argmax(), h.shape)
            kps[name] = (float(u), float(v), float(mx))
    return kps


def main():
    img_path = sys.argv[1]
    img = cv2.imread(img_path) if not img_path.endswith(".mp4") else None
    if img is None and img_path.endswith(".mp4"):
        cap = cv2.VideoCapture(img_path); cap.set(cv2.CAP_PROP_POS_FRAMES, 800); _, img = cap.read(); cap.release()
    H0, W0 = img.shape[:2]
    mask = turf_line_mask(img)
    mask_s = cv2.resize(mask, (Wd, H))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    net = CalibUNet(len(KP_NAMES)).to(dev)
    net.load_state_dict(torch.load(CKPT, map_location=dev)); net.eval()
    kps = detect_keypoints(net, mask_s, dev)
    # küçük-uzay keypoint -> tam-kare ölçek
    sx, sy = W0 / Wd, H0 / H
    L, W = 34.0, 18.0
    tmpl, _, _, _ = pitch_geometry(L, W)
    src, dst = [], []
    for name, (u, v, c) in kps.items():
        src.append([u * sx, v * sy]); dst.append(list(tmpl[name]))
    out = dict(n_kp=len(kps), kps={k: v for k, v in kps.items()})
    vis = img.copy()
    for name, (u, v, c) in kps.items():
        cv2.circle(vis, (int(u * sx), int(v * sy)), 9, (0, 255, 255), -1)
        cv2.putText(vis, f"{name}", (int(u * sx) + 8, int(v * sy)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1)
    base = os.path.basename(img_path).split(".")[0]
    cv2.imwrite(f"scratchpad/autorun/autocal_{base}_kp.jpg", vis)
    top_ok = False
    if len(src) >= 4:
        Hm, inl = cv2.findHomography(np.array(src, float), np.array(dst, float), cv2.RANSAC, 5.0)
        if Hm is not None:
            S = 26
            T = np.array([[S, 0, 0], [0, -S, W * S], [0, 0, 1]], float)
            top = cv2.warpPerspective(img, T @ Hm, (int(L * S), int(W * S)))
            for seg in [[(0, 0), (L, 0)], [(L, 0), (L, W)], [(L, W), (0, W)], [(0, W), (0, 0)], [(L / 2, 0), (L / 2, W)]]:
                a = (int(seg[0][0] * S), int((W - seg[0][1]) * S)); b = (int(seg[1][0] * S), int((W - seg[1][1]) * S))
                cv2.line(top, a, b, (0, 215, 255), 2)
            cv2.imwrite(f"scratchpad/autorun/autocal_{base}_top.jpg", top)
            top_ok = True
            out["inliers"] = int(inl.sum()) if inl is not None else 0
    print(json.dumps({"img": base, "n_kp": len(kps), "kps": list(kps.keys()),
                      "top_down": top_ok, "inliers": out.get("inliers")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
