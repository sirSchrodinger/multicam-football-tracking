#!/usr/bin/env python3
"""calib/solve_points.py — tıklanan NOKTA-işaretlerden homografi (point mode).

Girdi : calib/clicks_cankaya_cam2.json (mode=points) — yoksa ~/Downloads'tan al.
Çıktı : calib/cankaya_cam2_lines.json + calib/overlay_lines.png

Ölçek: kale ağzı = 3 m model'e gömülü (direk = merkez∓1.5). Saha boyu (L,W)
reprojeksiyon-minimum aramasıyla GERÇEK metreye oturur. YAKIN ceza sahası 4
köşesi tıklanmışsa, kutu boyutu (bd=derinlik, bw=en) de SERBEST aranır — amatör
sahada ölçü değişir; kutu dikdörtgen+simetrik olduğundan boyutu bilinmese bile
yakın yarıyı sabitler (yakın direkler genelde örtülü/yere-değmiyor → güvenilmez).
"""
import sys, json
from pathlib import Path
import numpy as np, cv2
sys.path.insert(0, ".")
from pitch.template import PitchTemplate
from pitch.homography import PitchHomography
from calib.landmarks import landmarks, BOX_IDS, BOX_BD_NOM, BOX_BW_NOM, POST_PAIRS
from calib.recalib import load_intrinsics

CLICKS = "calib/clicks_cankaya_cam2.json"
IMG = "calib/undist_clean.png"
OUT_JSON = "calib/cankaya_cam2_lines.json"
OUT_OVL = "calib/overlay_lines.png"


def _find_clicks():
    """En yeni indirmeyi al — tarayıcı '(1)' ekine dayanıklı (orphan-koruma)."""
    import glob, os
    p = Path(CLICKS)
    cands = sorted(glob.glob(str(Path.home() / "Downloads" / "clicks_cankaya_cam2*.json")),
                   key=os.path.getmtime, reverse=True)
    if cands:
        newest = cands[0]
        if (not p.exists()) or os.path.getmtime(newest) > os.path.getmtime(p):
            p.write_text(Path(newest).read_text())
            print(f"[clicks ~/Downloads'tan -> {CLICKS}: {os.path.basename(newest)}]")
    if not p.exists():
        sys.exit(f"clicks yok: {CLICKS} veya ~/Downloads/clicks_cankaya_cam2*.json")
    return p


def world_for(ids, L, W, bd=BOX_BD_NOM, bw=BOX_BW_NOM):
    d = {lid: np.array(xy, float) for lid, _, xy, _, _ in landmarks(L, W, bd, bw)}
    return {i: d[i] for i in ids if i in d}


def fit_quick(img_pts, world_pts):
    """world->undistorted px, en-küçük-kareler homografi + medyan reproj (px)."""
    if len(img_pts) < 4:
        return None, np.inf
    Hh, _ = cv2.findHomography(world_pts.astype(np.float64),
                              img_pts.astype(np.float64), 0)
    if Hh is None:
        return None, np.inf
    proj = cv2.perspectiveTransform(world_pts.reshape(-1, 1, 2).astype(np.float64), Hh).reshape(-1, 2)
    err = np.linalg.norm(proj - img_pts, axis=1)
    return Hh, float(np.median(err))


def solve(save=True, seed_L=33.0, fix_L=None, fix_W=None, drop=(),
          W_lo=14.0, W_hi=20.0, L_lo=28.0, L_hi=40.0, asp_lo=1.6, asp_hi=2.2,
          bd_lo=3.5, bd_hi=7.5, bw_lo=6.0, bw_hi=13.0):
    d = json.loads(_find_clicks().read_text())
    pts = {k: np.array(v, float) for k, v in d.get("points", {}).items()}
    for x in drop:
        pts.pop(x, None)
    if drop:
        print("DÜŞÜRÜLEN (güvenilmez):", list(drop))
    ids = list(pts)
    if len(ids) < 4:
        sys.exit(f"en az 4 nokta gerek, {len(ids)} var: {ids}")
    img_pts = np.array([pts[i] for i in ids])
    has_box = any(i in BOX_IDS for i in ids)
    K, dist = load_intrinsics()
    print("tıklanan:", ids, "| kutu köşesi:", [i for i in ids if i in BOX_IDS] or "yok")

    if fix_L and fix_W:
        L, W, bd, bw = float(fix_L), float(fix_W), BOX_BD_NOM, BOX_BW_NOM
        wp = world_for(ids, L, W, bd, bw); _, e = fit_quick(img_pts, np.array([wp[i] for i in ids]))
        scale_note = f"L,W SABİTLENDİ (override) reproj={e:.2f}px"
    else:
        # (L,W[,bd,bw]) arama — GERÇEKÇİ halısaha kutusuna sınırlı (aspect 1.6-2.2)
        bd_grid = np.arange(bd_lo, bd_hi + .01, 0.5) if has_box else [BOX_BD_NOM]
        bw_grid = np.arange(bw_lo, bw_hi + .01, 0.5) if has_box else [BOX_BW_NOM]
        best = (np.inf, seed_L, 18.0, BOX_BD_NOM, BOX_BW_NOM)
        for L in np.arange(L_lo, L_hi + .01, 0.5):
            for W in np.arange(W_lo, W_hi + .01, 0.5):
                if not (asp_lo <= L / W <= asp_hi):
                    continue
                for bd in bd_grid:
                    if bd >= L / 2:        # kutu sahanın yarısını geçemez
                        continue
                    for bw in bw_grid:
                        if bw >= W - 0.5:  # kutu eni sahadan dar
                            continue
                        wp = world_for(ids, L, W, bd, bw)
                        _, ee = fit_quick(img_pts, np.array([wp[i] for i in ids]))
                        if ee < best[0]:
                            best = (ee, float(L), float(W), float(bd), float(bw))
        _, L, W, bd, bw = best
        box_note = f" kutu bd={bd:.1f} bw={bw:.1f}" if has_box else ""
        scale_note = f"reproj-min, aspect∈[{asp_lo},{asp_hi}]{box_note} reproj={best[0]:.2f}px"

    # --- ÖLÇEK KİMLİĞİ: gerçek metre yalnız 3m kale-ağzıyla (direk çifti) sabitlenir ---
    pair = next((g for g, (a, b) in POST_PAIRS.items() if a in ids and b in ids), None)
    def _err_at(Lx, Wx):
        wp = world_for(ids, Lx, Wx, bd, bw)
        _, e = fit_quick(img_pts, np.array([wp[i] for i in ids])); return e
    base_e = _err_at(L, W)
    flat = (_err_at(min(L * 1.1, L_hi), W) - base_e < 1.0) and (_err_at(L, min(W * 1.1, W_hi)) - base_e < 1.0)
    if pair is None or flat:
        scale_conf = "ÖLÇEK GÜVENİ: DÜŞÜK — gerçek metre SABİTLENEMEDİ (kale direk-çifti yok / hedef düz). L,W ±~%10 TAHMİN."
    else:
        scale_conf = f"ÖLÇEK GÜVENİ: İYİ — '{pair}' kale 3m ağzı çıpa."

    print(f"\nseçilen  L={L:.1f}  W={W:.1f}  aspect={L/W:.2f}  | {scale_note}")
    print(scale_conf)

    # ---- nihai homografi (RANSAC + QA) ----
    tmpl = PitchTemplate.five_a_side(L=L, W=W, center_circle_r_m=3.0, has_penalty_box=False)
    wp = world_for(ids, L, W, bd, bw)
    world_pts = np.array([wp[i] for i in ids])
    homo = PitchHomography("cankaya_cam2", tmpl)
    homo.set_distortion(K, dist)
    homo.calibrate_manual(img_pts, world_pts, ransac_thresh_px=6.0, already_undistorted=True)

    # ---- per-nokta reproj ----
    proj = homo.pitch_to_pixel(world_pts)
    err = np.linalg.norm(proj - img_pts, axis=1)
    print(f"reproj  med={np.median(err):.2f}px  p95={np.percentile(err,95):.2f}px")
    for i, e in sorted(zip(ids, err), key=lambda t: -t[1]):
        flag = "  <-- BÜYÜK (yanlış etiket?)" if e > 25 else ""
        print(f"   {i:18s} {e:6.1f}px{flag}")

    if save:
        homo.status = "manual"; homo.calib_method = "manual_point"
        homo.box_bd, homo.box_bw = float(bd), float(bw)
        homo.save(OUT_JSON)
        ov = homo.qa_overlay(cv2.imread(IMG))
        cc = homo._dims_m(); th = np.linspace(0, 2*np.pi, 60)
        circ = np.c_[cc[0]/2 + 3.0*np.cos(th), cc[1]/2 + 3.0*np.sin(th)]
        cp = homo.pitch_to_pixel(circ)
        for k in range(len(cp)-1):
            cv2.line(ov, tuple(cp[k].astype(int)), tuple(cp[k+1].astype(int)), (255, 0, 255), 2)
        # yakın ceza sahası (mavi) — kutu çıpasını göster
        if True:
            cyw = cc[1] / 2.0
            box = np.array([[0, cyw - bw/2], [bd, cyw - bw/2], [bd, cyw + bw/2], [0, cyw + bw/2], [0, cyw - bw/2]])
            bp = homo.pitch_to_pixel(box)
            for k in range(len(bp)-1):
                cv2.line(ov, tuple(bp[k].astype(int)), tuple(bp[k+1].astype(int)), (255, 200, 0), 2)
        cv2.imwrite(OUT_OVL, ov)
        print(f"\nyazıldı: {OUT_JSON} + {OUT_OVL}")
    return homo


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--L", type=float, default=None)
    ap.add_argument("--W", type=float, default=None)
    ap.add_argument("--drop", default="", help="virgülle ayrık id listesi")
    a = ap.parse_args()
    solve(save=not a.dry, fix_L=a.L, fix_W=a.W,
          drop=tuple(x for x in a.drop.split(",") if x))
