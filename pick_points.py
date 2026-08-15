#!/usr/bin/env python3
"""Rehberli saha-nokta tiklayici (GUI). 4 kose DAYATMAZ — gorunen tanidik
isaretleri tek tek sorar; gormedigin atlarsin. >=4 nokta -> metrik homografi.

En guvenilir isaretler: KALE DIREKLERI (agiz 3 m kesin), gorunen saha koseleri,
orta yuvarlak. Lens once undistort edilir (cizgiler duz; tiklamak kolay).

KULLANIM (Lenovo'da, ekran acikken):
  venv/bin/python pick_points.py calib/cankaya_cam2_frame_raw.png \
      --k1npy calib/cankaya_cam2_distortion.npy --cam cankaya_cam2 --L 40 --W 25

KONTROLLER:
  Sol tik = aktif isareti koy (sonrakine gec)
  s = bu isareti ATLA      u = son tiki geri al      r = hepsini sifirla
  q veya Enter = BITIR ve kalibre et       ESC = kaydetmeden cik
  Imlec yaninda 4x buyutec (kucuk/uzak direkleri hassas tiklamak icin).

Cikti: calib/<cam>.json + qa_overlay_<cam>.png + topdown_<cam>.png
QA overlay'de turuncu 5 m grid saha cizgilerine OTURMALI. Oturmuyorsa tekrar tikla.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrate_field as cf


def landmarks(L, W):
    """(aciklama, (worldX, worldY)) — guvenilirden zora. Kaleler W/2'de ortali, 3 m."""
    cyW = W / 2.0
    return [
        ("Yakin kale - ALT direk (zemine degdigi yer)",   (0.0, cyW - 1.5)),
        ("Yakin kale - UST direk (zemine degdigi yer)",   (0.0, cyW + 1.5)),
        ("Uzak kale  - ALT direk (zemine degdigi yer)",   (L,   cyW - 1.5)),
        ("Uzak kale  - UST direk (zemine degdigi yer)",   (L,   cyW + 1.5)),
        ("Yakin-ALT saha kosesi (alt cizgi x yakin kale cizgisi)", (0.0, 0.0)),
        ("Yakin-UST saha kosesi (ust cizgi x yakin kale cizgisi)", (0.0, W)),
        ("Uzak-ALT saha kosesi",                          (L,   0.0)),
        ("Uzak-UST saha kosesi",                          (L,   W)),
        ("Orta yuvarlak MERKEZI",                         (L/2, cyW)),
        ("Ceza yayi / penalti noktasi varsa ATLA (s)",    None),  # placeholder, atla
    ]


state = {"cursor": (0, 0), "click": None}


def on_mouse(event, x, y, flags, param):
    state["cursor"] = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN:
        state["click"] = (x, y)


def draw_loupe(disp, frame_und, scale, cursor, zoom=4, r=36):
    """Imlec altindaki bolgeyi tam-cozunurlukten buyutup sag-uste yapistir."""
    dh, dw = disp.shape[:2]
    cxd, cyd = cursor
    fx, fy = int(cxd / scale), int(cyd / scale)          # tam-coz koordinat
    H, Wf = frame_und.shape[:2]
    x0, y0 = max(0, fx - r), max(0, fy - r)
    x1, y1 = min(Wf, fx + r), min(H, fy + r)
    patch = frame_und[y0:y1, x0:x1]
    if patch.size == 0:
        return
    side = (2 * r) * zoom
    lp = cv2.resize(patch, (side, side), interpolation=cv2.INTER_NEAREST)
    cv2.line(lp, (side // 2, 0), (side // 2, side), (0, 0, 255), 1)
    cv2.line(lp, (0, side // 2), (side, side // 2), (0, 0, 255), 1)
    px, py = dw - side - 10, 10
    disp[py:py + side, px:px + side] = lp
    cv2.rectangle(disp, (px, py), (px + side, py + side), (0, 255, 255), 2)
    cv2.putText(disp, f"({fx},{fy})", (px, py + side + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frame")
    ap.add_argument("--k1npy", default="calib/cankaya_cam2_distortion.npy")
    ap.add_argument("--cam", default="cankaya_cam2")
    ap.add_argument("--L", type=float, default=40.0)
    ap.add_argument("--W", type=float, default=25.0)
    ap.add_argument("--outdir", default="calib")
    ap.add_argument("--max-w", type=int, default=1600)
    ap.add_argument("--max-h", type=int, default=900)
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])

    k1, fx, cx, cy = np.load(a.k1npy)
    K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1]], float)
    dist = np.array([k1, 0, 0, 0, 0], float)
    frame = cv2.imread(a.frame)
    if frame is None:
        sys.exit(f"frame okunamadi: {a.frame}")
    und = cv2.undistort(frame, K, dist)
    H, Wf = und.shape[:2]
    scale = min(a.max_w / Wf, a.max_h / H, 1.0)
    base = cv2.resize(und, (int(Wf * scale), int(H * scale)))

    LM = landmarks(a.L, a.W)
    print("\n=== Tiklanacak isaretler (gormedigini 's' ile atla) ===")
    for i, (name, w) in enumerate(LM):
        print(f"  [{i+1}] {name}" + (f"  -> saha({w[0]:.1f},{w[1]:.1f})m" if w else "  (ATLA)"))
    print("Kontrol: sol-tik koy | s atla | u geri | r sifirla | q bitir | ESC iptal\n")

    win = "saha noktalari - tikla"
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win, on_mouse)

    picks = []   # (img_und_xy, world_xy, label)
    idx = 0
    while True:
        # bos olmayan (atlanabilir) bir sonraki isareti bul
        while idx < len(LM) and LM[idx][1] is None:
            idx += 1
        disp = base.copy()
        # konmus noktalar
        for (ix, iy), _w, lab in picks:
            p = (int(ix * scale), int(iy * scale))
            cv2.circle(disp, p, 6, (255, 0, 255), -1)
            cv2.putText(disp, lab, (p[0] + 6, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
        # ust talimat barı
        cv2.rectangle(disp, (0, 0), (disp.shape[1], 64), (0, 0, 0), -1)
        if idx < len(LM):
            name, w = LM[idx]
            cv2.putText(disp, f"[{len(picks)+1}] TIKLA: {name}", (12, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(disp, f"saha({w[0]:.1f},{w[1]:.1f})m  |  s:atla  u:geri  q:bitir({len(picks)})",
                        (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        else:
            cv2.putText(disp, f"BITTI ({len(picks)} nokta). q=kalibre et, u=geri",
                        (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        draw_loupe(disp, und, scale, state["cursor"])
        cv2.imshow(win, disp)

        # tiklama islensin
        if state["click"] is not None and idx < len(LM):
            dx, dy = state["click"]; state["click"] = None
            ix, iy = dx / scale, dy / scale
            name, w = LM[idx]
            lab = name.split(" - ")[0].split(" (")[0][:14]
            picks.append(((ix, iy), w, lab))
            idx += 1
            continue

        k = cv2.waitKey(15) & 0xFF
        if k in (ord('q'), 13):           # bitir + kalibre
            break
        elif k == 27:                     # ESC iptal
            print("iptal edildi (kayit yok)."); cv2.destroyAllWindows(); return
        elif k == ord('s') and idx < len(LM):
            print(f"  atlandi: {LM[idx][0]}"); idx += 1
        elif k == ord('u') and picks:
            (_, _, lab) = picks.pop()
            # geri alirken idx'i en yakin dolu isarete getir
            idx = max(0, idx - 1)
            while idx > 0 and LM[idx][1] is None:
                idx -= 1
            print(f"  geri alindi: {lab}")
        elif k == ord('r'):
            picks.clear(); idx = 0; print("  sifirlandi")
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()

    if len(picks) < 4:
        sys.exit(f"yetersiz nokta ({len(picks)}); homografi icin >=4 gerek.")
    img_und = np.array([p[0] for p in picks], float)
    world = np.array([p[1] for p in picks], float)
    labels = [p[2] for p in picks]
    print(f"\n{len(picks)} nokta -> kalibrasyon...")
    r = cf.run_calibration(frame, K, dist, img_und, world, a.L, a.W, a.cam,
                           a.outdir, tag="", labels=labels)
    print(f"round-trip {r['rt_err_px']:.3f}px | reprojection {r['reproj_m']:.3f} m")
    print(f"per-zone QA: {r['qa'].get('per_zone')}")
    print(f"-> {r['cal_path']}\n-> {r['qa_path']}\n-> {r['td_path']}")
    print("\nQA overlay'i ac: turuncu 5m grid saha cizgilerine oturmali. "
          "Oturmadiysa pick_points'i tekrar calistir.")


if __name__ == "__main__":
    main()
