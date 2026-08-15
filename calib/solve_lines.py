#!/usr/bin/env python3
"""calib/solve_lines.py — tıklanan çizgilerden homografi kur + overlay + QA.

Girdi : calib/clicks_cankaya_cam2.json (click_landmarks.py çıktısı)
Çıktı : calib/cankaya_cam2_lines.json  (yeni kalibrasyon)
        calib/overlay_lines.png         (undistorted frame üstüne template)
Konsol: metre-domeni çizgi artığı + köşe koşullanması + kale-genişliği ölçek-kontrolü.

Köşeler 4 çevre çizgisinin (endline_x0/xL, touchline_y0/yW) kesişiminden kurtarılır
(occluded yakın-sol köşe LEGAL). halfway = QA. Kale direkleri = ölçek doğrulaması:
homografi yakın+uzak kale ağzını metreye taşır; gerçek 3m'e oranla W yeniden ölçeklenir.
"""
import sys, json
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, ".")
from pitch.template import PitchTemplate
from pitch import line_calib
from calib.recalib import load_intrinsics

CLICKS = "calib/clicks_cankaya_cam2.json"
IMG = "calib/undist_clean.png"


def _find_clicks():
    """calib/ yoksa ~/Downloads'tan al (tarayıcı oraya indirir) ve calib/'e kopyala."""
    p = Path(CLICKS)
    if p.exists():
        return p
    dl = Path.home() / "Downloads" / "clicks_cankaya_cam2.json"
    if dl.exists():
        p.write_text(dl.read_text())
        print(f"[clicks ~/Downloads'tan alındı -> {CLICKS}]")
        return p
    sys.exit(f"clicks bulunamadı: {CLICKS} veya {dl}")
OUT_JSON = "calib/cankaya_cam2_lines.json"
OUT_OVL = "calib/overlay_lines.png"
GOAL_W = 3.0  # kale ağzı (m) — ölçek çıpası


def solve(L=33.0, W=18.0, has_box=True, save=True):
    d = json.loads(_find_clicks().read_text())
    lines = {k: np.asarray(v, float) for k, v in d["lines"].items()}
    points = {k: np.asarray(v, float) for k, v in d.get("points", {}).items()}
    K, dist = load_intrinsics()
    tmpl = PitchTemplate.five_a_side(L=L, W=W, center_circle_r_m=3.0, has_penalty_box=has_box)

    have = set(lines)
    need = {"endline_x0", "endline_xL", "touchline_y0", "touchline_yW"}
    print("çizgiler:", {k: len(v) for k, v in lines.items()})
    if not need <= have:
        sys.exit(f"4 çevre çizgisi şart; eksik: {need - have}")

    homo, corners, qa = line_calib.build_homography_from_lines(
        "cankaya_cam2", tmpl, lines, K=K, dist=dist, ransac_px=4.0)

    # ---- QA ----
    print(f"\nL={L}  W={W}  aspect={L/W:.2f}")
    print(f"metre-artığı  median={qa['median_m']:.3f}m  p95={qa['p95_m']:.3f}m")
    for nm, q in qa["per_line"].items():
        print(f"   {nm:14s} med={q['median_m']:.3f}  p95={q['p95_m']:.3f}  n={q['n']}")
    print("köşe koşullanma (sin=kesişim açısı, ill=zayıf):")
    for cid, c in corners.items():
        p = c["img_und"]
        ps = f"({p[0]:.0f},{p[1]:.0f})" if p is not None else "None"
        print(f"   {cid}: {ps:14s} sin={c['sin_angle']:.2f} extrap={c['extrap']:.2f} ill={c['ill']}")
    print(f"n_well={homo._qa.get('n_well_conditioned')}  reproj_med={homo.reprojection_error():.2f}px")

    # ---- kale-genişliği ölçek kontrolü ----
    def gw(a, b):
        if a in points and b in points:
            P = homo.pixel_to_pitch(np.array([points[a], points[b]]), already_undistorted=True)
            return float(np.linalg.norm(P[0] - P[1])), P
        return None, None
    for lab, a, b in [("yakın", "goal_near_L", "goal_near_R"), ("uzak", "goal_far_L", "goal_far_R")]:
        w, P = gw(a, b)
        if w:
            print(f"kale-ağzı({lab}) ölçülen={w:.2f}m  (gerçek {GOAL_W}m)  "
                  f"-> W_öneri={W*GOAL_W/w:.1f}m   yerler={np.round(P,1).tolist()}")

    if save:
        homo.status = "manual"
        homo.calib_method = "manual_line"
        homo.save(OUT_JSON)
        ov = homo.qa_overlay(cv2.imread(IMG))
        # zengin overlay: orta yuvarlak + tıklanan çizgi noktaları
        cc = homo._dims_m(); th = np.linspace(0, 2*np.pi, 60)
        circ = np.c_[cc[0]/2 + 3.0*np.cos(th), cc[1]/2 + 3.0*np.sin(th)]
        cp = homo.pitch_to_pixel(circ)
        for i in range(len(cp)-1):
            cv2.line(ov, tuple(cp[i].astype(int)), tuple(cp[i+1].astype(int)), (255, 0, 255), 2)
        for nm, pts in lines.items():
            for (x, y) in pts:
                cv2.circle(ov, (int(x), int(y)), 4, (255, 255, 0), -1)
        cv2.imwrite(OUT_OVL, ov)
        print(f"\nyazıldı: {OUT_JSON} + {OUT_OVL}")
    return homo, corners, qa


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=float, default=33.0)
    ap.add_argument("--W", type=float, default=18.0)
    ap.add_argument("--no-box", action="store_true")
    ap.add_argument("--dry", action="store_true", help="kaydetme, sadece QA")
    a = ap.parse_args()
    solve(L=a.L, W=a.W, has_box=not a.no_box, save=not a.dry)
