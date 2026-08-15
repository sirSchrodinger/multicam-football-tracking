#!/usr/bin/env python3
"""Saha kalibrasyonu: tanidik nokta yazismalari -> metrik homografi + QA + top-down.

Akis:
  1. frame'i k1 ile undistort et (cizgiler duzlesir; bkz. best_k1).
  2. UNDISTORTED uzayda tiklanan nokta(lar) + her birinin saha-metre karsiligi.
  3. noktalar HAM-piksele geri-distort edilir (projectPoints), boylece
     PitchHomography (HAM-piksel bekler) ile birebir uyumlu kalir.
  4. calibrate_manual (>=4 nokta, RANSAC) -> H, reprojection QA.
  5. qa_overlay (metrik grid HAM frame'e geri-projekte; cizgiler UYMALI) + top-down.

ONEMLI: 4 saha-kosesi DAYATMAZ. Bir kose kare disinda olabilir. Bunun yerine
gorunen herhangi tanidik isaretler kullanilir (kale direkleri 3 m capali, gorunen
koseler, orta yuvarlak). pick_points.py rehberli tiklamayi yapar.

Saha konvansiyonu (dikdortgen, origin yakin-kale ALT-touchline kosesi):
    yakin-kale end-line x=0, uzak-kale end-line x=L   (uzunluk)
    alt touchline y=0, ust touchline y=W              (genislik)
    kaleler W/2'de ortali, agiz 3 m.

CLI (4-kose yolu, eski uyumluluk):
  venv/bin/python calibrate_field.py FRAME.png --k1npy best_k1.npy --cam cankaya_cam2 \
      --L 40 --W 25 --corners "FLx,FLy FRx,FRy NRx,NRy NLx,NLy"   # undistorted px
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch.homography import PitchHomography
from pitch.template import PitchTemplate


def redistort(pts_und, K, dist):
    """UNDISTORTED piksel -> HAM (distorted) piksel (Brown-Conrady ileri model)."""
    P = np.asarray(pts_und, np.float64).reshape(-1, 2)
    Kinv = np.linalg.inv(K)
    norm = (Kinv @ np.c_[P, np.ones(len(P))].T).T[:, :2]
    obj = np.c_[norm, np.ones(len(P))].astype(np.float64)
    raw, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, dist.reshape(-1))
    return raw.reshape(-1, 2)


def run_calibration(frame, K, dist, img_und_pts, world_pts, L, W, cam,
                    outdir="calib", tag="", labels=None, ppm=18, ransac_px=8.0):
    """UNDISTORTED tiklamalar + saha-metre -> H, QA overlay, top-down, JSON kaydet.

    img_und_pts: (N>=4,2) undistorted piksel; world_pts: (N,2) metre. Doner dict.
    """
    img_und = np.asarray(img_und_pts, np.float64).reshape(-1, 2)
    world = np.asarray(world_pts, np.float64).reshape(-1, 2)
    if len(img_und) < 4:
        raise ValueError(f"homografi icin >=4 nokta gerek, alinan {len(img_und)}")

    corners_raw = redistort(img_und, K, dist)
    rt = cv2.undistortPoints(corners_raw.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    rt_err = float(np.abs(rt - img_und).max())

    homo = PitchHomography(cam, PitchTemplate.seven_a_side(L=L, W=W))
    homo.set_distortion(K, dist)
    qa = homo.calibrate_manual(corners_raw, world, ransac_thresh_px=ransac_px)
    rep = homo.reprojection_error()

    outdir = Path(outdir); outdir.mkdir(exist_ok=True)
    sfx = f"_{tag}" if tag else ""
    cal_path = outdir / f"{cam}{sfx}.json"
    homo.save(str(cal_path))

    # --- QA overlay: metrik grid + perimeter HAM frame'e geri-projekte ---
    # frame HAM -> pitch_to_pixel(distorted=True) ile HAM piksele projekte et (yoksa
    # undistorted-piksel ham-kareye cizilir = distorsiyon kaymasi, sahte kotu QA).
    ov = frame.copy()
    for xm in np.arange(0, L + .01, 5):
        seg = homo.pitch_to_pixel(np.array([[xm, 0], [xm, W]], float), distorted=True).astype(int)
        cv2.line(ov, tuple(seg[0]), tuple(seg[1]), (0, 200, 255), 1)
    for ym in np.arange(0, W + .01, 5):
        seg = homo.pitch_to_pixel(np.array([[0, ym], [L, ym]], float), distorted=True).astype(int)
        cv2.line(ov, tuple(seg[0]), tuple(seg[1]), (0, 200, 255), 1)
    per = homo.pitch_to_pixel(np.array([[0, 0], [L, 0], [L, W], [0, W]], float),
                              distorted=True).astype(int)
    cv2.polylines(ov, [per.reshape(-1, 1, 2)], True, (0, 0, 255), 3)
    # kullanicinin tikladigi noktalar (HAM uzayda) + reprojeksiyon hatasi cizgisi
    world_reproj = homo.pitch_to_pixel(world, distorted=True)
    for i, (xr, yr) in enumerate(corners_raw.astype(int)):
        cv2.circle(ov, (xr, yr), 7, (255, 0, 255), -1)
        wp = world_reproj[i].astype(int)
        cv2.line(ov, (xr, yr), tuple(wp), (0, 255, 0), 1)   # yesil = artik hata
        if labels:
            cv2.putText(ov, labels[i], (xr + 8, yr), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 255), 2)
    cv2.putText(ov, f"QA: turuncu 5m grid saha cizgilerine UYMALI | reproj={rep:.2f}m "
                f"| {len(img_und)} nokta{(' '+tag) if tag else ''}",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    qa_path = outdir / f"qa_overlay_{cam}{sfx}.png"
    cv2.imwrite(str(qa_path), ov)

    # --- top-down: undistorted frame'i kus-bakisi sahaya warp et ---
    Wt, Ht = int(L * ppm), int(W * ppm)
    und = cv2.undistort(frame, K, dist)
    # undistorted->topdown homografisini world korolari uzerinden kur
    dpix = world.copy().astype(np.float32)
    dpix[:, 0] = dpix[:, 0] * ppm
    dpix[:, 1] = (W - dpix[:, 1]) * ppm
    Hwarp, _ = cv2.findHomography(img_und.astype(np.float32).reshape(-1, 1, 2),
                                  dpix.reshape(-1, 1, 2), cv2.RANSAC, 8.0)
    top = cv2.warpPerspective(und, Hwarp, (Wt, Ht))
    for ym in range(0, int(W) + 1, 5):
        cv2.line(top, (0, int((W - ym) * ppm)), (Wt, int((W - ym) * ppm)), (255, 255, 255), 1)
    for xm in range(0, int(L) + 1, 5):
        cv2.line(top, (xm * ppm, 0), (xm * ppm, Ht), (255, 255, 255), 1)
    cv2.putText(top, f"TOP-DOWN {L:.0f}x{W:.0f}m{(' '+tag) if tag else ''}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    td_path = outdir / f"topdown_{cam}{sfx}.png"
    cv2.imwrite(str(td_path), top)

    return {"homo": homo, "reproj_m": rep, "qa": qa, "rt_err_px": rt_err,
            "cal_path": str(cal_path), "qa_path": str(qa_path), "td_path": str(td_path)}


def parse_corners(s):
    out = [float(t) for t in s.replace(",", " ").split()]
    if len(out) != 8:
        raise ValueError("4 kose = 8 sayi bekleniyor (FL FR NR NL)")
    return np.array(out, float).reshape(4, 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frame")
    ap.add_argument("--k1npy", default="best_k1.npy")
    ap.add_argument("--cam", default="cankaya_cam2")
    ap.add_argument("--L", type=float, default=40.0)
    ap.add_argument("--W", type=float, default=25.0)
    ap.add_argument("--corners", required=True,
                    help='"FLx,FLy FRx,FRy NRx,NRy NLx,NLy" UNDISTORTED piksel')
    ap.add_argument("--outdir", default="calib")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])

    k1, fx, cx, cy = np.load(a.k1npy)
    K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], float)
    dist = np.array([k1, 0, 0, 0, 0], float)
    frame = cv2.imread(a.frame)
    if frame is None:
        sys.exit(f"frame okunamadi: {a.frame}")
    cu = parse_corners(a.corners)
    L, W = float(a.L), float(a.W)
    world = np.array([[0, W], [L, W], [L, 0], [0, 0]], float)  # FL FR NR NL
    r = run_calibration(frame, K, dist, cu, world, L, W, a.cam, a.outdir, a.tag,
                        labels=["FL", "FR", "NR", "NL"])
    print(f"round-trip {r['rt_err_px']:.3f}px | reproj {r['reproj_m']:.3f}m")
    print(f"-> {r['cal_path']}\n-> {r['qa_path']}\n-> {r['td_path']}")


if __name__ == "__main__":
    main()
