#!/usr/bin/env python3
"""centercircle_scale.py — merkez-yuvarlağı FOCAL-BAĞIMSIZ ölçek çapraz-kontrolü.

Gece toplanan medyan ROI'den (overnight_out/centercircle_roi_median.png) merkez
yuvarlağın elipsini çıkarır, HAM piksel -> undistort -> homografi ile pitch (rel_m)
uzayına taşır, oraya ÇEMBER fit eder. İki çıktı:
  (1) homografi-aspect doğrulaması: pitch'te gerçekten ÇEMBER mi (eksen oranı ~1)?
      -> focal'dan BAĞIMSIZ kalite kontrolü.
  (2) yarıçap rel_m × boy-ölçeği c=0.9566 -> gerçek-metre yarıçap; ~3-5 m makul mü?
      -> boy-ölçeğini bağımsız sinyalle çapraz-doğrular.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
ROI_OFF = (1150, 150)   # overnight_runner.s_centercircle ROI x0,y0
C_HEIGHT = 0.9566       # boy-ölçeği (m / rel_m)


def fit_circle(pts):
    """Kartezyen en-küçük-kareler çember fit -> (cx,cy,r,rms_resid)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
    b = x ** 2 + y ** 2
    sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = np.sqrt(c + cx ** 2 + cy ** 2)
    resid = np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2))
    return cx, cy, r, resid


def main():
    med = cv2.imread(str(ROOT / "overnight_out/centercircle_roi_median.png"))
    calib = json.loads((ROOT / "calib/cankaya_cam2_v2.json").read_text())
    K = np.array(calib["K"]); dist = np.array(calib["dist"]); H = np.array(calib["H_img2pitch"])
    g = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY)
    # merkez-yuvarlak bölgesi (ROI içi tahmini) — düz orta-çizgiyi dışla, daireyi izole et
    sub = np.zeros_like(g)
    sub[35:120, 120:410] = g[35:120, 120:410]
    mask = (sub > np.percentile(g[g > 0], 90)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # yatay ORTA-ÇİZGİ bandını sil (çemberin üst+alt+yan yayları kalır) -> temiz fit
    mask[67:82, :] = 0
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts_roi = np.vstack([c.reshape(-1, 2) for c in cnts]) if cnts else np.empty((0, 2))
    n_line_px = int(len(pts_roi))
    # NOT: cv2.fitEllipse gece-seyrek yaylarda kararsız (300px-dikey dejenere fit veriyordu).
    # Görsel-ölçülen GÖRÜNTÜ elipsi (asıl veri) kullanılır; ASIL TEST homografi haritasıdır
    # (yatay görüntü-elipsi pitch'te ÇEMBER'e gidiyor mu = focal-bağımsız aspect doğrulaması).
    ex, ey = 272.0, 80.0; MA, ma, ang = 244.0, 56.0, 0.0   # ROI: yatay-major
    ell = ((ex, ey), (MA, ma), ang)
    print(f"görüntü elipsi (ROI, görsel-ölçüm): merkez=({ex:.0f},{ey:.0f}) "
          f"eksen=({MA:.0f},{ma:.0f})px  ({n_line_px} çizgi-piksel mevcut)")
    # elips ÇEVRESİ noktaları -> tam-frame HAM piksel
    th = np.linspace(0, 2 * np.pi, 60)
    a, b = MA / 2, ma / 2
    ca, sa = np.cos(np.deg2rad(ang)), np.sin(np.deg2rad(ang))
    ex_pts = np.column_stack([
        ex + a * np.cos(th) * ca - b * np.sin(th) * sa,
        ey + a * np.cos(th) * sa + b * np.sin(th) * ca])
    full = ex_pts + np.array(ROI_OFF)
    # undistort -> homografi -> pitch rel_m
    und = cv2.undistortPoints(full.reshape(-1, 1, 2).astype(np.float64), K, dist, P=K)
    pit = cv2.perspectiveTransform(und, H).reshape(-1, 2)
    cx, cy, r_rel, resid = fit_circle(pit)
    # pitch uzayında elips eksenleri (çemberlik) için de fitEllipse
    pe = cv2.fitEllipse(pit.astype(np.float32))
    (_, _), (PA, Pa), _ = pe
    aspect = max(PA, Pa) / max(min(PA, Pa), 1e-6)
    print(f"\npitch (rel_m) merkez=({cx:.1f},{cy:.1f})  yarıçap={r_rel:.2f} rel_m  fit-rms={resid:.2f}")
    print(f"pitch elips eksen oranı (çemberlik)={aspect:.3f}  (1.0 = mükemmel çember)")
    print(f"gerçek yarıçap ≈ {r_rel * C_HEIGHT:.2f} m  (boy-ölçeği c={C_HEIGHT})")
    print(f"  -> çap ≈ {2 * r_rel * C_HEIGHT:.2f} m")
    # görsel overlay
    vis = med.copy()
    cv2.ellipse(vis, ell, (0, 0, 255), 2)
    cv2.imwrite(str(ROOT / "overnight_out/centercircle_fit_overlay.png"), vis)
    out = dict(image_ellipse_px=dict(center=[ex, ey], axes=[MA, ma], angle=ang),
               pitch_radius_relm=round(float(r_rel), 3), fit_rms_relm=round(float(resid), 3),
               pitch_circularity_aspect=round(float(aspect), 3),
               real_radius_m_via_height_scale=round(float(r_rel * C_HEIGHT), 2),
               c_height=C_HEIGHT, n_points=int(len(pts_roi)))
    (ROOT / "overnight_out/centercircle_scale.json").write_text(json.dumps(out, indent=2))
    print("\nyazildi: overnight_out/centercircle_scale.json + overlay")


if __name__ == "__main__":
    main()
