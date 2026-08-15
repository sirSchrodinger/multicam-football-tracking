#!/usr/bin/env python3
"""calib/solve_paint.py — PAINT (çizgi+nokta) annotasyondan homografi.

Girdi : calib/paint_cankaya_cam2.json  (yoksa ~/Downloads'tan en yenisi)
Çıktı : calib/cankaya_cam2_lines.json + calib/overlay_paint.png

Yöntem:
  1. Her trace edilen çizgiyi görüntü-uzayında TLS (PCA) ile doğruya oturt.
  2. X-çizgisi × Y-çizgisi kesişimleri = dünya köşe/orta noktaları — GÖRÜNMEYEN
     köşe (kesik near köşeler) de bu kesişimden kurtulur (kadrajda olması gerekmez).
  3. + doğrudan tıklanan noktalar (santra, direkler, ceza-ön köşeleri).
  4. (L,W,bd,bw) reproj-min araması (kale 3m + ceza sahası çıpa).
  5. Nihai homografi + overlay + DÜRÜST img→saha (metre) hata raporu.
"""
import sys, json, glob, os
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, ".")
from pitch.template import PitchTemplate
from pitch.homography import PitchHomography
from calib.paint_labels import LINES, POINTS, intersections_world, point_world
from calib.recalib import load_intrinsics

PAINT = "calib/paint_cankaya_cam2.json"
IMG = "calib/undist_clean.png"
OUT_JSON = "calib/cankaya_cam2_lines.json"
OUT_OVL = "calib/overlay_paint.png"
LINE_KIND = {i: (k, v) for (i, _, _, k, v) in LINES}
LINE_COL = {i: c for (i, _, c, _, _) in LINES}
PT_COL = {i: c for (i, _, c) in POINTS}


def _find_paint():
    p = Path(PAINT)
    cands = sorted(glob.glob(str(Path.home() / "Downloads" / "paint_cankaya_cam2*.json")),
                   key=os.path.getmtime, reverse=True)
    if cands:
        newest = cands[0]
        if (not p.exists()) or os.path.getmtime(newest) > os.path.getmtime(p):
            p.write_text(Path(newest).read_text())
            print(f"[paint ~/Downloads'tan -> {PAINT}: {os.path.basename(newest)}]")
    if not p.exists():
        sys.exit(f"paint yok: {PAINT} veya ~/Downloads/paint_cankaya_cam2*.json")
    return p


def fit_line(pts):
    """TLS doğru: (a,b,c), a^2+b^2=1, a*u+b*v+c=0."""
    P = np.asarray(pts, float)
    m = P.mean(0)
    u, s, vt = np.linalg.svd(P - m)
    n = vt[-1]                       # en küçük tekil yön = normal
    n = n / np.linalg.norm(n)
    c = -float(n @ m)
    return float(n[0]), float(n[1]), c


def intersect(L1, L2):
    a1, b1, c1 = L1; a2, b2, c2 = L2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None
    x = (-c1 * b2 + c2 * b1) / det
    y = (-a1 * c2 + a2 * c1) / det
    return np.array([x, y])


def fit_quick(img_pts, world_pts):
    if len(img_pts) < 4:
        return None, np.inf
    Hh, _ = cv2.findHomography(world_pts.astype(np.float64), img_pts.astype(np.float64), 0)
    if Hh is None:
        return None, np.inf
    proj = cv2.perspectiveTransform(world_pts.reshape(-1, 1, 2), Hh).reshape(-1, 2)
    return Hh, float(np.median(np.linalg.norm(proj - img_pts, axis=1)))


def solve(save=True, L_lo=28.0, L_hi=40.0, W_lo=14.0, W_hi=20.0, asp_lo=1.6, asp_hi=2.2,
          bd_lo=3.5, bd_hi=7.5, bw_lo=6.0, bw_hi=13.0):
    d = json.loads(_find_paint().read_text())
    raw_lines = {k: np.array(v, float) for k, v in d.get("lines", {}).items() if len(v) >= 2}
    raw_pts = {k: np.array(v, float) for k, v in d.get("points", {}).items()}
    K, dist = load_intrinsics()
    fitted = {i: fit_line(p) for i, p in raw_lines.items()}
    print("trace edilen çizgiler:", list(raw_lines), "| noktalar:", list(raw_pts))

    # --- sabit (params-bağımsız) görüntü-noktaları: kesişimler + doğrudan noktalar ---
    inter_img = {}   # (xi,yi) -> img xy
    for (xi, yi) in intersections_world(1.0, 1.0):      # geçerli çiftler (val'leri sonra ölçekle)
        if xi in fitted and yi in fitted:
            q = intersect(fitted[xi], fitted[yi])
            if q is not None:
                inter_img[(xi, yi)] = q
    img_named = list(raw_pts.items())                   # [(label, xy)]
    if len(inter_img) + len(img_named) < 4:
        sys.exit(f"yetersiz kısıt: {len(inter_img)} kesişim + {len(img_named)} nokta (<4). "
                 f"daha çok çizgi/nokta gerek.")
    print(f"kesişim noktası: {len(inter_img)}  (görünmeyen köşeler dahil)  + doğrudan: {len(img_named)}")

    has_box = any(k.startswith("box_") for k in raw_pts)

    def assemble(L, W, bd, bw):
        imgs, wlds = [], []
        for (xi, yi), q in inter_img.items():
            X = LINE_KIND[xi][1] * L
            Y = LINE_KIND[yi][1] * W
            imgs.append(q); wlds.append([X, Y])
        for lab, xy in img_named:
            wp = point_world(lab, L, W, bd, bw)
            if wp is not None:
                imgs.append(xy); wlds.append(list(wp))
        return np.array(imgs, float), np.array(wlds, float)

    bd_grid = np.arange(bd_lo, bd_hi + .01, 0.5) if has_box else [5.0]
    bw_grid = np.arange(bw_lo, bw_hi + .01, 0.5) if has_box else [9.0]
    best = (np.inf, 33.0, 18.0, 5.0, 9.0)
    for L in np.arange(L_lo, L_hi + .01, 0.5):
        for W in np.arange(W_lo, W_hi + .01, 0.5):
            if not (asp_lo <= L / W <= asp_hi):
                continue
            for bd in bd_grid:
                if bd >= L / 2:
                    continue
                for bw in bw_grid:
                    if bw >= W - 0.5:
                        continue
                    ip, wp = assemble(L, W, bd, bw)
                    _, e = fit_quick(ip, wp)
                    if e < best[0]:
                        best = (e, float(L), float(W), float(bd), float(bw))
    _, L, W, bd, bw = best
    box_note = f" kutu bd={bd:.1f} bw={bw:.1f}" if has_box else ""
    print(f"\nseçilen  L={L:.1f}  W={W:.1f}  aspect={L/W:.2f}{box_note}  | quick reproj={best[0]:.2f}px")

    img_pts, world_pts = assemble(L, W, bd, bw)
    tmpl = PitchTemplate.five_a_side(L=L, W=W, center_circle_r_m=3.0, has_penalty_box=False)
    homo = PitchHomography("cankaya_cam2", tmpl)
    homo.set_distortion(K, dist)
    homo.calibrate_manual(img_pts, world_pts, ransac_thresh_px=6.0, already_undistorted=True)

    # DÜRÜST img->saha hata (metre): tıklanan görüntü noktalarını sahaya taşı, dünya ile kıyasla
    pitch = homo.pixel_to_pitch(img_pts, already_undistorted=True)
    merr = np.linalg.norm(pitch - world_pts, axis=1)
    proj = homo.pitch_to_pixel(world_pts)
    perr = np.linalg.norm(proj - img_pts, axis=1)
    print(f"img->saha hata: med={np.median(merr):.2f}m p95={np.percentile(merr,95):.2f}m  "
          f"(saha ~{L:.0f}x{W:.0f}m → ~%{100*np.median(merr)/W:.1f} en)")
    print(f"reproj (px):    med={np.median(perr):.2f} p95={np.percentile(perr,95):.2f}")

    if save:
        homo.status = "manual"; homo.calib_method = "manual_paint"
        homo.box_bd, homo.box_bw = float(bd), float(bw)
        homo.save(OUT_JSON)
        ov = homo.qa_overlay(cv2.imread(IMG))
        cc = homo._dims_m(); th = np.linspace(0, 2*np.pi, 60)
        circ = np.c_[cc[0]/2 + 3.0*np.cos(th), cc[1]/2 + 3.0*np.sin(th)]
        cp = homo.pitch_to_pixel(circ)
        for k in range(len(cp)-1):
            cv2.line(ov, tuple(cp[k].astype(int)), tuple(cp[k+1].astype(int)), (255, 0, 255), 2)
        cyw = cc[1]/2.0
        box = np.array([[0, cyw-bw/2], [bd, cyw-bw/2], [bd, cyw+bw/2], [0, cyw+bw/2], [0, cyw-bw/2]])
        bp = homo.pitch_to_pixel(box)
        for k in range(len(bp)-1):
            cv2.line(ov, tuple(bp[k].astype(int)), tuple(bp[k+1].astype(int)), (255, 200, 0), 2)
        # trace edilen çizgileri (Alperen'in çizdiği) ince beyaz noktalı göster
        for i, P in raw_lines.items():
            for q in P:
                cv2.circle(ov, tuple(np.round(q).astype(int)), 4, (255, 255, 255), -1)
        cv2.imwrite(OUT_OVL, ov)
        print(f"\nyazıldı: {OUT_JSON} + {OUT_OVL}")
    return homo


if __name__ == "__main__":
    solve()
