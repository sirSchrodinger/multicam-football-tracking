#!/usr/bin/env python3
"""pick_lines.py — Gorunmeyen-kose CIZGI kalibratoru (GUI), pick_points.py'nin
line-mode kardesi.

Amator sabit-kamera halisahada kose TIKLANAMAZ: yakin-alt kose + korner kadraj
DISI, uzak-uc piksel-fakiri (Alperen icgoru #2/#3). Cozum: koseyi tiklamak
yerine her ADLI sinir cizgisi boyunca >=2 nokta tiklat; pitch.line_calib
TLS-dogru fit eder, kesistirir, KADRAJ-DISI koseleri sentezler, H kurar.

Bu dosya line_calib'i DEGISTIRMEZ (tum kadraj-disi kose sentezi + durustluk
kapisi + metre-QA orada, dogrulanmis). pick_lines.py SADECE (a) GUI ile bir
{ad:(N,2)} sozlugu toplar ve (b) ince/durust bir orkestrator'dur.

KULLANIM (Lenovo'da, ekran acikken):
  venv/bin/python pick_lines.py calib/cankaya_cam2_frame_raw.png \
      --k1npy calib/cankaya_cam2_distortion.npy --cam cankaya_cam2 --L 34 --W 18

KONTROLLER:
  Sol-tik = aktif cizgiye nokta ekle        n/Tab/Space = sonraki cizgi
  [ ] = cizgiler arasi atla    s = bu cizgiyi atla (halfway)   u = geri al
  r = sifirla    g = 2-kale-direk nokta modu (ayna-kirici; olcek YALNIZ --lock-scale ile)
  q/Enter = kalibre et    ESC = iptal
  Her cizgiye 3. (genis-arali) nokta koy: 2-nokta fit rms=0 -> kadraj-disi
  ekstrapolasyon gurultuye duyarli.

DURUSTLUK (line_calib'ten miras, degistirilmez):
  - Tek dürüst kabul sinyali METRE-domeni line_point_residual_qa'dir.
  - reprojection_error() 4-kose tam-fitte ~0'dir; KAPI OLARAK KULLANILMAZ.
  - <4 sonlu kose -> RuntimeError (kose icin iki parent cizgi sart; halfway tek
    basina kurtarmaz). Reddedilirse kanonik json YAZILMAZ (audit P1).
  - covered_frac yalniz metre-QA + kosullanma ile birlikte raporlanir; bu
    sahada 1.0 olctuk -> mutlak sinir DEGIL (asiri-iyimser).
  - Olcek kilidi VARSAYILAN KAPALI (scale_anchor=None -> relative_m, icgoru #5);
    point_lms verilse bile --lock-scale olmadan metre IDDIASI yok.

Lisans: yalniz OpenCV (BSD) + numpy. GPU yok.
"""
from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pitch.template import PitchTemplate
from pitch.line_calib import (
    fit_line_tls, corners_from_lines, coverage_confidence_map,
    calibrate_from_lines,
)

# --- sinir cizgileri (kose parent'lari) + halfway (kose URETMEZ, yalniz QA) ---
_BOUNDARY_KEYS = ("touchline_yW", "touchline_y0", "endline_x0", "endline_xL")

# kale-direk nokta-isaretleri: ad -> world (X, Y)-ureteci(L,W). Y = W/2 -/+ 1.5.
_GOALPOST_WORLD = OrderedDict([
    ("near_low",  lambda L, W: (0.0, W / 2.0 - 1.5)),
    ("near_high", lambda L, W: (0.0, W / 2.0 + 1.5)),
    ("far_low",   lambda L, W: (L,   W / 2.0 - 1.5)),
    ("far_high",  lambda L, W: (L,   W / 2.0 + 1.5)),
])


# =============================================================== PURE HELPERS ==
def load_intrinsics(k1npy):
    """[k1, fx, cx, cy] npy (pick_points.py konvansiyonu) -> (K(3,3), dist(5,)).

    fy=fx (kare piksel varsayimi); dist = [k1, 0, 0, 0, 0] Brown-Conrady.
    """
    k1, fx, cx, cy = (float(v) for v in np.load(k1npy))
    K = np.array([[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.array([k1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return K, dist


def line_menu(L, W):
    """Sirali [(world_key, insan-etiketi)]. 4 sinir = kose parent'lari; halfway
    SON ve opsiyonel (kose uretmez, yalniz metre-QA/kosullanma)."""
    return [
        ("touchline_yW", "UST tac cizgisi (uzak kenar, y=W)"),
        ("touchline_y0", "ALT tac cizgisi (yakin kenar, y=0) - korner kadraj-disi olabilir"),
        ("endline_x0",   "YAKIN kale cizgisi (x=0)"),
        ("endline_xL",   "UZAK kale cizgisi (x=L) - piksel-fakiri, dusuk-guven"),
        ("halfway",      "Orta saha cizgisi (opsiyonel; kose URETMEZ, yalniz QA)"),
    ]


def assemble_line_clicks(picks):
    """[(name,(x,y)), ...] -> {name: (N,2) ndarray}; YALNIZ N>=2 olan cizgiler.

    Tek-nokta cizgileri (fit edilemez) dusulur."""
    groups = OrderedDict()
    for name, xy in picks:
        groups.setdefault(name, []).append((float(xy[0]), float(xy[1])))
    return {k: np.asarray(v, dtype=np.float64)
            for k, v in groups.items() if len(v) >= 2}


def goalpost_point_lms(gp, L, W):
    """{ad: img_px} -> [(img_px(2,), world_xy(2,))] ayna-kirici nokta-isaretler.

    Adlar: near_low/near_high/far_low/far_high -> world (0 ya da L, W/2 -/+ 1.5).
    gp'de bulunmayan adlar atlanir (kismi kale gorunumu)."""
    Lf, Wf = float(L), float(W)
    out = []
    for name, wfn in _GOALPOST_WORLD.items():
        if name not in gp or gp[name] is None:
            continue
        ip = np.asarray(gp[name], dtype=np.float64).reshape(2)
        out.append((ip, np.asarray(wfn(Lf, Wf), dtype=np.float64)))
    return out


def _coverage_png(cov):
    """cov["conf"] (GORECELI guven, ASLA mutlak hata sinırı) -> viridis BGR PNG.

    flipud -> y yukari (saha sol-alt origin). [0,1] -> uint8 -> COLORMAP_VIRIDIS.
    """
    conf = np.asarray(cov["conf"], dtype=np.float64)
    conf = np.flipud(np.clip(conf, 0.0, 1.0))
    u8 = (conf * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(u8, cv2.COLORMAP_VIRIDIS)
    # gorunurluk icin buyut (en az ~480px genislik), nearest
    h, w = color.shape[:2]
    sc = max(1, int(np.ceil(480.0 / max(1, w))))
    if sc > 1:
        color = cv2.resize(color, (w * sc, h * sc),
                           interpolation=cv2.INTER_NEAREST)
    return color


# ===================================================== HONEST ORCHESTRATOR ====
def run_line_calibration(frame, K, dist, line_clicks, L, W, cam,
                         outdir="calib", tag="", point_lms=None,
                         linemap=None, lock_scale=False, ransac_px=4.0):
    """Ince/durust orkestrator: PURE compute+gate (calibrate_from_lines), sonra
    side-car yaz.

    calibrate_from_lines'i frame_und=None ve out_json/out_overlay/out_coverage=
    None ile cagirir (sessiz-yanlis kayit yok; kayit/cizim BURADA durustce yapilir).
    - lock_scale False ise scale_anchor=None (icgoru #5; point_lms olsa bile metre
      iddiasi yok).
    - saved ise: homo.save(json); frame varsa qa_overlay (HAM frame ver; qa_overlay
      iceride undistort eder -- zaten-undistorted frame verirsen CIFT undistort olur,
      kacinilacak tek bug) + _coverage_png yaz.
    - saved degil + frame varsa: qa_overlay_<cam><sfx>_REJECTED.png (tani; kanonik
      json YAZILMAZ, audit P1).
    - res["paths"] eklenir.

    NOT: <4 sonlu kose -> calibrate_from_lines RuntimeError firlatir; burada
    YAKALANMAZ (cagiran/main() yakalar).
    """
    L, W = float(L), float(W)
    template = PitchTemplate.seven_a_side(L=L, W=W)
    res = calibrate_from_lines(
        cam, template, line_clicks,
        frame_und=None, K=K, dist=dist, point_lms=point_lms,
        static_line_map=linemap,
        out_json=None, out_overlay=None, out_coverage=None,
        ransac_px=float(ransac_px),
    )
    homo = res["homo"]
    corners = res["corners"]

    # ICGORU #5: metre kilidi varsayilan KAPALI -> relative_m. point_lms olcek
    # tohumladiysa (build icinde) burada geri al; yalniz acikca --lock-scale ile kal.
    if not lock_scale:
        homo.scale_anchor = None

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    sfx = f"_{tag}" if tag else ""
    cal_path = str(outdir / f"{cam}{sfx}.json")
    overlay_path = str(outdir / f"qa_overlay_{cam}{sfx}.png")
    coverage_path = str(outdir / f"coverage_{cam}{sfx}.png")
    rejected_path = str(outdir / f"qa_overlay_{cam}{sfx}_REJECTED.png")

    paths = {}
    if res["saved"]:
        homo.save(cal_path)
        paths["cal"] = cal_path
        if frame is not None:
            # HAM frame -> qa_overlay iceride undistort eder (cift-undistort yok)
            cv2.imwrite(overlay_path, homo.qa_overlay(frame))
            paths["overlay"] = overlay_path
            marked = [c["world"] for c in corners.values()
                      if c["img_und"] is not None
                      and np.all(np.isfinite(c["img_und"]))]
            cov = coverage_confidence_map(
                homo, frame.shape[:2],
                marked_pts_m=(np.asarray(marked, float) if len(marked) >= 3
                              else None))
            cv2.imwrite(coverage_path, _coverage_png(cov))
            paths["coverage"] = coverage_path
            res["coverage"] = cov
    else:
        # REDDEDILDI: kanonik json YOK, yalniz tani overlay'i
        if frame is not None and homo.H_img2pitch is not None:
            cv2.imwrite(rejected_path, homo.qa_overlay(frame))
            paths["rejected_overlay"] = rejected_path

    res["paths"] = paths
    return res


# =================================================================== GUI =======
def _fit_endpoints(fit, length):
    """Fit edilmis dogruyu (c0 +/- length*t) iki uca uzat (undistorted px)."""
    c0 = np.asarray(fit["c0"], float)
    t = np.asarray(fit["t"], float)
    return c0 - length * t, c0 + length * t


def _missing_boundaries(line_clicks, point_lms):
    """Kalibre ONCESI >=4-sonlu-kose kapisini GUI'de on-kontrol et.

    Doner: (eksik_sinir_set, tahmini_sonlu_kose_sayisi). Eksik bos VE sonlu>=4
    ise hazir."""
    have = {k for k in _BOUNDARY_KEYS
            if k in line_clicks and len(line_clicks[k]) >= 2}
    missing = set(_BOUNDARY_KEYS) - have
    # sonlu kose tahmini: parent ciftleri tamam olanlari say
    fits = {k: fit_line_tls(v) for k, v in line_clicks.items()
            if k in _BOUNDARY_KEYS and len(v) >= 2}
    n_corner = 0
    if fits:
        cs = corners_from_lines(fits, 1.0, 1.0)  # boyut onemsiz; sadece sonluluk
        n_corner = sum(1 for c in cs.values()
                       if c["img_und"] is not None
                       and np.all(np.isfinite(c["img_und"])))
    n_finite = n_corner + (len(point_lms) if point_lms else 0)
    return missing, n_finite


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frame", nargs="?",
                    default="calib/cankaya_cam2_frame_raw.png")
    ap.add_argument("--k1npy", default="calib/cankaya_cam2_distortion.npy")
    ap.add_argument("--cam", default="cankaya_cam2")
    ap.add_argument("--L", type=float, default=34.0,
                    help="saha boyu (X) m -- PROVISIONAL, override edilebilir")
    ap.add_argument("--W", type=float, default=18.0,
                    help="saha eni (Y) m -- PROVISIONAL")
    ap.add_argument("--outdir", default="calib")
    ap.add_argument("--tag", default="lines")
    ap.add_argument("--linemap", default=None,
                    help="opsiyonel ridge olasilik haritasi (.npy)")
    ap.add_argument("--lock-scale", action="store_true",
                    help="kale-direkleriyle metre kilidi (VARSAYILAN KAPALI)")
    ap.add_argument("--pad-frac", type=float, default=0.18,
                    help="kadraj-disi koseler icin sanal-tuval bosluk orani")
    ap.add_argument("--no-halfway", action="store_true",
                    help="orta saha cizgisini menude gizle")
    ap.add_argument("--max-w", type=int, default=1700)
    ap.add_argument("--max-h", type=int, default=950)
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])

    K, dist = load_intrinsics(a.k1npy)
    raw = cv2.imread(a.frame)
    if raw is None:
        sys.exit(f"frame okunamadi: {a.frame}")
    und = cv2.undistort(raw, K, dist)
    Hf, Wf = und.shape[:2]

    linemap = None
    if a.linemap:
        linemap = np.load(a.linemap)

    # --- sanal tuval: kadraj-disi koseler + uzatilmis cizgiler gutter'da gozuksun
    pad_x = int(round(Wf * a.pad_frac))
    pad_y = int(round(Hf * a.pad_frac))
    cH, cW = Hf + 2 * pad_y, Wf + 2 * pad_x
    canvas0 = np.full((cH, cW, 3), 28, np.uint8)
    # hatched / dim margin (tiklanamaz bolge)
    for yy in range(0, cH, 16):
        cv2.line(canvas0, (0, yy), (cW, yy), (44, 44, 44), 1)
    canvas0[pad_y:pad_y + Hf, pad_x:pad_x + Wf] = und

    scale = min(a.max_w / cW, a.max_h / cH, 1.0)

    menu = line_menu(a.L, a.W)
    if a.no_halfway:
        menu = [m for m in menu if m[0] != "halfway"]

    print("\n=== CIZGI menusu (her cizgiye >=2, tercihen 3 genis-arali nokta) ===")
    for i, (k, lab) in enumerate(menu):
        print(f"  [{i+1}] {k}: {lab}")
    print("Kontrol: sol-tik ekle | n/Tab/Space sonraki | [ ] atla-gez | s skip | "
          "u geri | r sifirla | g kale-direk | q kalibre | ESC iptal\n")

    # lazy: pick_points yalniz GUI'de import edilir (headless-safe)
    try:
        from pick_points import draw_loupe
    except Exception:
        draw_loupe = None

    st = {"cursor": (0, 0), "click": None}

    def on_mouse(event, x, y, flags, param):
        st["cursor"] = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            st["click"] = (x, y)

    win = "cizgi kalibratoru - tikla"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    picks = []                      # [(name,(x_und,y_und)), ...]
    goalposts = OrderedDict()       # name -> (x_und,y_und)
    gp_names = list(_GOALPOST_WORLD)
    gp_idx = 0
    active = 0                      # menu indeksi
    gmode = False

    def canvas_to_und(cx, cy):
        return (cx / scale - pad_x, cy / scale - pad_y)

    def und_to_disp(ux, uy):
        return (int(round((ux + pad_x) * scale)), int(round((uy + pad_y) * scale)))

    while True:
        canvas = canvas0.copy()
        line_clicks = assemble_line_clicks(picks)

        # --- canli fit + kose kosullanma onizleme
        fits_all = {}
        for name, pts in line_clicks.items():
            try:
                fits_all[name] = fit_line_tls(pts)
            except Exception:
                pass
        bfits = {k: v for k, v in fits_all.items() if k in _BOUNDARY_KEYS}
        cs = corners_from_lines(bfits, a.L, a.W) if bfits else {}
        diag = max(cW, cH)

        # cizgi uzantilari (dashed-vari: tam INTER ile uzatip ciz)
        for name, fit in fits_all.items():
            p0, p1 = _fit_endpoints(fit, diag)
            cv2.line(canvas, und_to_disp(*p0), und_to_disp(*p1),
                     (90, 90, 90), 1)

        # tiklanan noktalar
        for name, (ux, uy) in picks:
            cv2.circle(canvas, und_to_disp(ux, uy), 5, (255, 0, 255), -1)
        for nm, (ux, uy) in goalposts.items():
            d = und_to_disp(ux, uy)
            cv2.circle(canvas, d, 6, (0, 165, 255), -1)
            cv2.putText(canvas, nm, (d[0] + 6, d[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)

        # kose renk-kodu: yesil=iyi&kadrajda, cyan=kadraj-disi&iyi, amber=ill, kirmizi=yok
        for cid, c in cs.items():
            p = c["img_und"]
            if p is None or not np.all(np.isfinite(p)):
                continue
            d = und_to_disp(p[0], p[1])
            on_frame = (0 <= p[0] < Wf and 0 <= p[1] < Hf)
            if c["ill"]:
                col = (0, 200, 255)         # amber
            elif on_frame:
                col = (0, 255, 0)           # yesil
            else:
                col = (255, 255, 0)         # cyan (kadraj-disi ama iyi)
            in_canvas = (0 <= d[0] < int(cW * scale) and 0 <= d[1] < int(cH * scale))
            if in_canvas:
                cv2.drawMarker(canvas, d, col, cv2.MARKER_TILTED_CROSS, 18, 2)
                cv2.putText(canvas, f"{cid} s={c['sin_angle']:.2f}",
                            (d[0] + 8, d[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
            else:
                # kenar-ok fallback (margin'in disinda)
                ex = int(np.clip(d[0], 6, int(cW * scale) - 6))
                ey = int(np.clip(d[1], 6, int(cH * scale) - 6))
                cv2.arrowedLine(canvas, (ex, ey), d if in_canvas else (ex, ey),
                                col, 1)
                cv2.putText(canvas, f"{cid}->", (ex, ey),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        # ust bar
        cv2.rectangle(canvas, (0, 0), (cW, 70), (0, 0, 0), -1)
        if gmode:
            nm = gp_names[gp_idx] if gp_idx < len(gp_names) else "(bitti)"
            cv2.putText(canvas, f"[KALE-DIREK] tikla: {nm}", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            cv2.putText(canvas,
                        "ayna-kirici; olcek YALNIZ --lock-scale ile  |  g: cik",
                        (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        else:
            k, lab = menu[active]
            ncl = len(line_clicks.get(k, []))
            cv2.putText(canvas, f"[{active+1}/{len(menu)}] {lab}  (n={ncl})",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            miss, nfin = _missing_boundaries(line_clicks,
                                             goalpost_point_lms(goalposts, a.L, a.W))
            if miss or nfin < 4:
                msg = ("NOT-READY: eksik sinir -> "
                       + (", ".join(sorted(miss)) if miss
                          else f"sonlu kose {nfin}/4"))
                cv2.putText(canvas, msg, (12, 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 80, 255), 2)
            else:
                cv2.putText(canvas, "HAZIR: q=kalibre et", (12, 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        disp = cv2.resize(canvas, (int(cW * scale), int(cH * scale)))

        # rubber-band: aktif cizginin son noktasindan imlece on-fit cizgisi
        cur = st["cursor"]
        if not gmode:
            k = menu[active][0]
            kpts = line_clicks.get(k)
            if kpts is not None and len(kpts) >= 1:
                last = und_to_disp(kpts[-1][0], kpts[-1][1])
                cv2.line(disp, last, cur, (0, 255, 255), 1)

        # buyutec: imlec gercek undistorted frame uzerindeyse
        ux, uy = canvas_to_und(*cur)
        if draw_loupe is not None and 0 <= ux < Wf and 0 <= uy < Hf:
            # pick_points.draw_loupe(disp, frame_und, scale, cursor) konvansiyonu:
            # cursor/scale = und koord. Bizde offset var -> gecici cursor uret.
            fake_cursor = (int((ux) * scale), int((uy) * scale))
            try:
                draw_loupe(disp, und, scale, fake_cursor)
            except Exception:
                pass

        cv2.imshow(win, disp)

        # tiklama isle
        if st["click"] is not None:
            dx, dy = st["click"]; st["click"] = None
            ux, uy = canvas_to_und(dx, dy)
            if gmode:
                if gp_idx < len(gp_names):
                    goalposts[gp_names[gp_idx]] = (ux, uy)
                    gp_idx += 1
            else:
                picks.append((menu[active][0], (ux, uy)))
            continue

        kc = cv2.waitKey(15) & 0xFF
        if kc in (ord('q'), 13):
            break
        elif kc == 27:
            print("iptal (kayit yok)."); cv2.destroyAllWindows(); return
        elif kc in (ord('n'), 9, ord(' ')):
            active = (active + 1) % len(menu)
        elif kc == ord(']'):
            active = (active + 1) % len(menu)
        elif kc == ord('['):
            active = (active - 1) % len(menu)
        elif kc == ord('s'):
            print(f"  atlandi: {menu[active][0]}")
            active = (active + 1) % len(menu)
        elif kc == ord('u'):
            if gmode and goalposts:
                rm = list(goalposts)[-1]; goalposts.pop(rm)
                gp_idx = max(0, gp_idx - 1)
                print(f"  geri (kale): {rm}")
            elif picks:
                rm = picks.pop()
                print(f"  geri: {rm[0]}")
        elif kc == ord('r'):
            picks.clear(); goalposts.clear(); gp_idx = 0
            print("  sifirlandi")
        elif kc == ord('g'):
            gmode = not gmode
            print(f"  kale-direk modu: {'ACIK' if gmode else 'kapali'}")
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()

    line_clicks = assemble_line_clicks(picks)
    plm = goalpost_point_lms(goalposts, a.L, a.W) or None

    # kalibre ONCESI durust on-kontrol (crash yerine isimlendir)
    miss, nfin = _missing_boundaries(line_clicks, plm)
    if miss or nfin < 4:
        print("NOT-READY -> kalibrasyon yapilmadi.")
        if miss:
            print("  eksik sinir cizgisi:", ", ".join(sorted(miss)))
        print(f"  sonlu kose: {nfin}/4 (kose icin iki parent cizgi sart; "
              "halfway tek basina kurtarmaz).")
        return

    try:
        res = run_line_calibration(
            raw, K, dist, line_clicks, a.L, a.W, a.cam,
            outdir=a.outdir, tag=a.tag, point_lms=plm,
            linemap=linemap, lock_scale=a.lock_scale, ransac_px=4.0)
    except RuntimeError as e:
        print(f"REDDEDILDI (durustluk kapisi): {e}")
        print("  -> uzak kale/end-line geri-kurtarilamiyor; o bolgeyi "
              "no-coverage isaretle.")
        return

    homo = res["homo"]
    qa = res["qa_line"]
    cc = homo._qa.get("corner_cond", {}) if homo._qa else {}
    ill = [cid for cid, c in cc.items() if c.get("ill")]
    nwell = homo._qa.get("n_well_conditioned", "?") if homo._qa else "?"
    print(f"\nsaved={res['saved']}  scale_anchor={homo.scale_anchor}")
    print(f"metre-QA: median={qa['median_m']:.3f} m  p95={qa['p95_m']:.3f} m")
    print(f"iyi-kosullu kose: {nwell}/4   ill: {ill or '-'}")
    print(f"reproj(degenerate ~0, KAPI DEGIL)={homo.reprojection_error():.4f} px")
    for kk, vv in res.get("paths", {}).items():
        print(f"  {kk}: {vv}")
    if not res["saved"]:
        print("KAYIT YOK: gate/inversiyon basarisiz -> kanonik json yazilmadi.")


if __name__ == "__main__":
    main()
