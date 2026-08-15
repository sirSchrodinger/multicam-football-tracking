#!/usr/bin/env python3
"""Kalibre edilmiş bir kameradan BEDAVA saha-çizgi eğitim etiketi üret.

Fikir: homografi biliniyorsa saha şablonunun çizgilerini görüntüye geri-projekte
et -> piksel-doğru çizgi maskesi (elle çizim YOK). Her saha bir kez kalibre
edilince, tüm kareleri otomatik etiketlenir. Birden çok saha biriktikçe bu
çiftler (kare, maske) bir saha-çizgi/keypoint dedektörünü EĞİTİR -> yeni sahalar
elle-tıklamasız kalibre olur. Bu, statik tek kameranın TEK geometrisini verir;
GENELLEME için farklı saha/açılar gerekir (her biri bir kez kalibre).

Kullanım:
  venv/bin/python calib/export_line_labels.py CALIB.json VIDEO.mp4 OUTDIR [N]
"""
import sys, json
from pathlib import Path
import numpy as np, cv2


def project_lines(calib, shape, thick=4):
    """calib (FINAL.json) -> (H,W) uint8 çizgi maskesi, undist uzayında."""
    Hp2i = np.array(calib["H_pitch2img"], float)
    Lr, Wr = calib["template"]["dims_m"]
    SC = (calib.get("scale_anchor") or {}).get("scale_factor", 1.0)
    front = np.sign((Hp2i @ np.array([Lr / 2, Wr / 2, 1.0]))[2])
    m = np.zeros(shape[:2], np.uint8)

    def Pf(X, Y):
        q = Hp2i @ np.array([X, Y, 1.0]); return q[0] / q[2], q[1] / q[2], q[2]

    def seg(a, b, t):
        (x0, y0, w0), (x1, y1, w1) = Pf(*a), Pf(*b)
        if np.sign(w0) == front and np.sign(w1) == front:
            cv2.line(m, (int(x0), int(y0)), (int(x1), int(y1)), 255, t)

    # çevre + orta çizgi
    for a, b in [((0, 0), (Lr, 0)), ((Lr, 0), (Lr, Wr)), ((Lr, Wr), (0, Wr)),
                 ((0, Wr), (0, 0)), ((Lr / 2, 0), (Lr / 2, Wr))]:
        seg(a, b, thick)
    # santra çemberi (gerçek 3m -> REL)
    r = 3.0 / SC if SC else 3.0
    th = np.linspace(0, 2 * np.pi, 90)
    cc = [(Lr / 2 + r * np.cos(t), Wr / 2 + r * np.sin(t)) for t in th]
    for i in range(len(cc) - 1):
        seg(cc[i], cc[i + 1], max(2, thick - 1))
    # ceza sahaları (tıklanan kutu ~bd5 bw8 REL)
    for xb, xf in [(0, 5), (Lr, Lr - 5)]:
        for a, b in [((xb, Wr / 2 - 4), (xf, Wr / 2 - 4)), ((xf, Wr / 2 - 4), (xf, Wr / 2 + 4)),
                     ((xf, Wr / 2 + 4), (xb, Wr / 2 + 4))]:
            seg(a, b, max(2, thick - 1))
    return m


def main():
    if len(sys.argv) < 4:
        print(__doc__); return
    calib = json.load(open(sys.argv[1]))
    video, outdir = sys.argv[2], Path(sys.argv[3])
    N = int(sys.argv[4]) if len(sys.argv) > 4 else 12
    K = np.array(calib["K"], float); D = np.array(calib["dist"], float)
    (outdir / "img").mkdir(parents=True, exist_ok=True)
    (outdir / "mask").mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video); total = int(cap.get(7))
    idx = np.linspace(total * 0.05, total * 0.95, N).astype(int)
    mask = None
    for k, fi in enumerate(idx):
        cap.set(1, int(fi)); ok, fr = cap.read()
        if not ok:
            continue
        und = cv2.undistort(fr, K, D)
        if mask is None:                         # statik kamera -> maske sabit
            mask = project_lines(calib, und.shape)
        cv2.imwrite(str(outdir / "img" / f"{k:04d}.png"), und)
        cv2.imwrite(str(outdir / "mask" / f"{k:04d}.png"), mask)
    cap.release()
    cov = 100 * mask.mean() / 255 if mask is not None else 0
    print(f"yazıldı: {outdir}  ({len(idx)} çift, maske kapsama %{cov:.2f})")
    print("NOT: statik tek kamera = tek geometri. Genelleme için çok-saha gerekir.")


if __name__ == "__main__":
    main()
