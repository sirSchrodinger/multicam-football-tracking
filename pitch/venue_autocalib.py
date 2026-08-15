#!/usr/bin/env python3
"""pitch/venue_autocalib.py — per-tesis OTOMATİK kamera kalibrasyonu (orkestratör).

Amaç: her tesisin SABİT kamerasını otomatik kalibre et (homografi: undistort-px ->
saha-metre, doğru ASPECT), böylece metrik istatistik HER tesiste çalışsın.
Kilit kolaylık: oto-kalibrasyon sadece HOMOGRAFİYİ doğru kursa yeter — MUTLAK
ÖLÇEK kalibrasyonda None kalır, sonradan pitch.height_scale (~1.75m evrensel
cetvel) ile doldurulur. Yani problem "tam metrik kalibrasyon"dan "doğru-perspektif
homografi + dürüst QA-gate"e iner.

Pipeline (mevcut blokları birleştirir + eksik parçaları ekler):
  1. build_static_line_map   (auto_calib) — temporal medyan, oyuncu-inpaint
  2. per-tesis fisheye k1     (homography.estimate_distortion_from_lines) — eksik-parça #1
  3. undistort-ÖNCE-chamfer   (yük-taşıyan düzeltme) — line map undistort
  4. coarse chamfer kayıt     (auto_calib.register_template_chamfer, multi-start simetri)
  5. proje-template + ridge-snap + refit  (line_calib mantığı) -> DÜZGÜN QA karşılıkları
  6. 2-kat gate               (auto_calib.self_verify_gate + topdown_stats.accept_calib_qa)
  7. kabul -> calib JSON (scale None); red -> None (dürüst refuse, manuel'e düş)

DÜRÜSTLÜK: sessizce-yanlış H YAYMAZ. Gate reddederse None döner.
Lisans: OpenCV(BSD) + numpy + scipy(auto_calib içinde). GPU yok.
"""
from __future__ import annotations

import numpy as np
import cv2

from pitch.homography import PitchHomography
from pitch.template import PitchTemplate
from pitch import auto_calib
from pitch.line_calib import fit_line_tls


# --------------------------------------------------------------------------- #
# Fisheye: çizgi-ridge'lerini bowed-polyline olarak çıkar (k1 tahmini için)
# --------------------------------------------------------------------------- #
def extract_line_polylines(L_static, thr: float = 0.35, min_len: int = 80,
                           max_lines: int = 8):
    """Statik çizgi haritasından uzun-ince ridge'leri (gerçekte-düz saha çizgileri)
    nokta-dizisi (bowed polyline) olarak çıkar. estimate_distortion_from_lines
    bunları düzleştirip k1 arar."""
    binm = (np.asarray(L_static) > thr).astype(np.uint8) * 255
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    cnts, _ = cv2.findContours(binm, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in sorted(cnts, key=cv2.contourArea, reverse=True):
        pts = c.reshape(-1, 2).astype(np.float64)
        if len(pts) < min_len:
            continue
        # ince-uzun mu? (bbox aspect) — dolu lekeleri (oyuncu artığı) ele
        x, y, w, h = cv2.boundingRect(c)
        if max(w, h) < min_len or min(w, h) > 0.35 * max(w, h):
            continue
        # uzun eksene göre sırala (polyline düzeni)
        f = fit_line_tls(pts[:: max(1, len(pts) // 60)])  # seyrek örnek, hız
        s = (pts - f["c0"]) @ f["t"]
        polys.append(pts[np.argsort(s)])
        if len(polys) >= max_lines:
            break
    return polys


def field_mask_and_corners(frames_bgr, K=None, dist=None):
    """Yeşil-saha maskesi (çit/arka-planı siler) + saha trapez-köşeleri (chamfer init).
    İLERLEME (29 Haz): çit-kilitlenmesini çözer + sahaya-hizalı başlangıç verir.
    DÜRÜST: tek başına homografi-LOCK sağlamıyor (gece-oblik sahnede line-IoU düşük);
    sağlam lock için analitik VP-rektifikasyon / etiketli-çizgi (line_calib) gerek.
    Döner: (field_mask uint8, img_corners (4,2) bl,br,tr,tl | None)."""
    med = np.median(np.stack(frames_bgr), 0).astype(np.uint8)
    if K is not None:
        med = cv2.undistort(med, np.asarray(K), np.asarray(dist))
    hsv = cv2.cvtColor(med, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 40, 30), (90, 255, 255))
    green = cv2.morphologyEx(green, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, np.ones((35, 35), np.uint8))
    nn, lab, stats, _ = cv2.connectedComponentsWithStats(green)
    if nn <= 1:
        return green, None
    big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    field = (lab == big).astype(np.uint8)
    cnt, _ = cv2.findContours(field, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnt, key=cv2.contourArea)
    peri = cv2.arcLength(c, True); quad = None
    for eps in np.linspace(0.01, 0.10, 30):
        ap = cv2.approxPolyDP(c, eps * peri, True)
        if len(ap) == 4:
            quad = ap.reshape(4, 2).astype(np.float32); break
    if quad is None:
        quad = cv2.boxPoints(cv2.minAreaRect(c))
    s = quad.sum(1); d = np.diff(quad, 1).ravel()
    corners = np.array([quad[np.argmax(d)], quad[np.argmax(s)],
                        quad[np.argmin(d)], quad[np.argmin(s)]], np.float32)  # bl,br,tr,tl
    return field, corners


def estimate_fisheye(L_static):
    """Çizgi ridge'lerinden tek-parametreli k1 kestir. Doner (K,dist) veya (None,None)."""
    polys = extract_line_polylines(L_static)
    if len(polys) < 2:
        return None, None
    H, W = L_static.shape[:2]
    try:
        K, dist = PitchHomography.estimate_distortion_from_lines((H, W), polys)
        return K, dist
    except Exception:  # noqa: BLE001
        return None, None


# --------------------------------------------------------------------------- #
# Refine: coarse H ile template çizgilerini projekte et, ridge'e snap, refit
# --------------------------------------------------------------------------- #
def _poly_area(flat):
    c = np.asarray(flat, float).reshape(4, 2)
    x, y = c[:, 0], c[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def robust_chamfer(L_und, template, multi_start: bool = True):
    """register_template_chamfer + ÇÖKME-ÖNLEME: 4-köşe quad'ı tek piksele
    çökerten dejenere minimumu ALAN-cezasıyla engeller (DT-chamfer'ın bilinen
    failure-mode'u). Döner: (H_img2pitch | None, cost)."""
    from scipy.optimize import minimize
    DT = auto_calib._dt_from_mask(L_und)
    line_pts_m = np.asarray(template.line_points_m(step_m=0.5), float)
    P = np.hstack([line_pts_m, np.ones((len(line_pts_m), 1))])
    h, w = L_und.shape[:2]
    ys, xs = np.where(L_und > 0.35)
    if xs.size < 50:
        return None, np.inf

    def _proj_spread(flat):
        """Projekte template'in in-frame (sx,sy,inb_frac); dejenere-çökme tespiti."""
        try:
            H_p2i = auto_calib._H_from_corner_img_pts(template, flat.reshape(4, 2))
        except cv2.error:
            return 0.0, 0.0, 0.0
        pr = (H_p2i @ P.T).T
        wv = pr[:, 2]; valid = np.abs(wv) > 1e-9
        if valid.sum() < len(P) * 0.5:
            return 0.0, 0.0, 0.0
        uv = pr[valid, :2] / wv[valid, None]
        inb = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        if inb.sum() < 50:
            return 0.0, 0.0, 0.0
        return float(np.ptp(uv[inb, 0])), float(np.ptp(uv[inb, 1])), float(inb.mean())

    def cost(flat):
        base = auto_calib._chamfer_cost(flat, template, DT, line_pts_m)
        sx, sy, _ = _proj_spread(flat)
        # ÇÖKME CEZASI: projekte template yeterince yayılmazsa ağır ceza
        spen = 0.0
        if sx < 0.30 * w:
            spen += (0.30 * w - sx) / w * 40.0
        if sy < 0.16 * h:
            spen += (0.16 * h - sy) / h * 40.0
        return base + spen

    hyps = auto_calib._init_corner_hypotheses(L_und, template)
    if not multi_start:
        hyps = hyps[:1]
    best, bestc = None, np.inf
    for hyp in hyps:
        res = minimize(cost, hyp.reshape(-1), method="Nelder-Mead",
                       options=dict(maxiter=2500, xatol=0.5, fatol=1e-3))
        sx, sy, _ = _proj_spread(res.x)
        if res.fun < bestc and sx >= 0.30 * w and sy >= 0.16 * h:
            bestc, best = float(res.fun), res.x.reshape(4, 2)
    if best is None:
        return None, np.inf
    H_p2i = auto_calib._H_from_corner_img_pts(template, best)
    return np.linalg.inv(H_p2i), bestc


def _snap_and_filter(proj_pts, L_und, radius: int = 6, thr: float = 0.3):
    """Projekte noktaları en yakın ridge pikseline çek; ridge YOKSA DÜŞÜR.
    Döner: (snapped (M,2), keep_mask (N,) bool)."""
    lm = np.asarray(L_und, dtype=np.float64)
    h, w = lm.shape[:2]
    snapped = []
    keep = np.zeros(len(proj_pts), bool)
    for i, (x, y) in enumerate(proj_pts):
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
        y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        win = lm[y0:y1, x0:x1]
        if win.max() < thr:
            continue
        dy, dx = np.unravel_index(int(np.argmax(win)), win.shape)
        snapped.append([x0 + dx, y0 + dy]); keep[i] = True
    return np.asarray(snapped, float).reshape(-1, 2), keep


def refine_via_snap(coarse_H, L_und, template, camera_id, K=None, dist=None,
                    step_m: float = 0.5, snap_radius: int = 6,
                    min_corr: int = 12):
    """Coarse H'den DÜZGÜN-QA homografi: template çizgi noktalarını projekte et,
    gözlenen ridge'e snap'le, calibrate_manual ile yeniden-fit (reproject QA dolar).

    Döner: PitchHomography veya None (yetersiz karşılık -> dürüst refuse)."""
    world = template.line_points_m(step_m=step_m)               # (M,2) metre
    H_p2i = np.linalg.inv(np.asarray(coarse_H, float))
    proj = cv2.perspectiveTransform(world.reshape(-1, 1, 2), H_p2i).reshape(-1, 2)
    h, w = L_und.shape[:2]
    inb = (proj[:, 0] >= 0) & (proj[:, 0] < w) & (proj[:, 1] >= 0) & (proj[:, 1] < h)
    if inb.sum() < min_corr:
        return None
    # DEJENERE-ÇÖKME GUARD: chamfer bazen tüm template'i tek piksele çökertir
    # (DT-maliyeti ~0 ama anlamsız). Projeksiyon yeterince yayılmamışsa REDDET.
    pin = proj[inb]
    sx = pin[:, 0].max() - pin[:, 0].min(); sy = pin[:, 1].max() - pin[:, 1].min()
    if sx < 0.15 * w or sy < 0.10 * h:
        return None   # çökmüş H -> dürüst refuse (VP-init gerekli)
    snapped, keep = _snap_and_filter(proj[inb], L_und, radius=snap_radius)
    world_in = world[inb][keep]
    if len(snapped) < min_corr:
        return None
    # snapped noktalar da yeterince yayılmalı (kollinear -> dejenere homografi)
    if (np.ptp(snapped[:, 0]) < 0.15 * w) or (np.ptp(snapped[:, 1]) < 0.08 * h):
        return None
    homo = PitchHomography(camera_id, template)
    if K is not None and dist is not None:
        homo.set_distortion(np.asarray(K), np.asarray(dist))
    try:
        homo.calibrate_manual(snapped, world_in, ransac_thresh_px=4.0,
                              already_undistorted=True)
    except (RuntimeError, ValueError):
        return None   # dejenere noktalar -> dürüst refuse
    homo.calib_method = "auto_snap"
    return homo


# --------------------------------------------------------------------------- #
# Gate: 2 katman (line-IoU/spread + metre-QA), dürüst refuse
# --------------------------------------------------------------------------- #
def _gate(homo, L_und, template):
    reasons = []
    gate = auto_calib.self_verify_gate(homo, L_und, template)
    if not gate.get("accept"):
        reasons += [f"self_verify: {r}" for r in gate.get("reasons", [])]
    qa_ok = True
    try:
        from stats.topdown_stats import accept_calib_qa
        qa_ok, qa_reasons = accept_calib_qa(getattr(homo, "_qa", None))
        if not qa_ok:
            reasons += [f"qa: {r}" for r in qa_reasons]
    except Exception as e:  # noqa: BLE001
        reasons.append(f"accept_calib_qa atlandı: {e}")
    accept = bool(gate.get("accept")) and bool(qa_ok)
    return accept, dict(gate=gate, reasons=reasons,
                        median_px=(homo._qa or {}).get("median_px"))


# --------------------------------------------------------------------------- #
# Ana giriş
# --------------------------------------------------------------------------- #
def autocalibrate_venue(video_path: str, camera_id: str,
                        template: PitchTemplate | None = None,
                        distortion=None, player_boxes_per_frame=None,
                        n_frames: int = 400, L_static=None,
                        out_json: str | None = None) -> tuple:
    """Tam oto-kalibrasyon. Döner: (PitchHomography|None, report dict).

    template verilmezse seven_a_side default. distortion=(K,dist) verilirse o
    kullanılır; yoksa çizgilerden k1 kestirilir (best-effort). L_static verilirse
    (önceden hesaplı) video tekrar okunmaz (hız)."""
    if template is None:
        template = PitchTemplate.seven_a_side()
    rep = {"camera_id": camera_id, "accept": False, "stage": None}

    # 1. statik çizgi haritası
    if L_static is None:
        L_static = auto_calib.build_static_line_map(
            video_path, n_frames=n_frames,
            player_boxes_per_frame=player_boxes_per_frame)
    rep["line_px_frac"] = float((L_static > 0.35).mean())

    # 2. fisheye k1
    if distortion is not None:
        K, dist = distortion
        rep["fisheye"] = "verildi"
    else:
        K, dist = estimate_fisheye(L_static)
        rep["fisheye"] = "kestirildi" if K is not None else "yok"

    # 3. undistort-ÖNCE
    L_und = cv2.undistort(L_static, np.asarray(K), np.asarray(dist)) if K is not None else L_static

    # 4. coarse chamfer (çökme-önlemeli, multi-start simetri)
    try:
        coarse_H, resid = robust_chamfer(L_und, template, multi_start=True)
    except Exception as e:  # noqa: BLE001
        rep["stage"] = f"chamfer hata: {e}"; return None, rep
    if coarse_H is None:
        rep["stage"] = "chamfer çöktü/yetersiz çizgi (refuse)"; return None, rep
    rep["chamfer_resid_px"] = float(resid)

    # 5. refine + düzgün QA
    homo = refine_via_snap(coarse_H, L_und, template, camera_id, K=K, dist=dist)
    if homo is None:
        rep["stage"] = "yetersiz ridge-karşılığı (refuse)"; return None, rep
    rep["median_px"] = float((homo._qa or {}).get("median_px", float("nan")))
    rep["n_landmarks"] = int((homo._qa or {}).get("n_landmarks", 0))

    # 6. gate
    accept, ginfo = _gate(homo, L_und, template)
    rep["accept"] = accept; rep["gate_reasons"] = ginfo["reasons"]

    # 7. kaydet / refuse (ölçek None — sonradan height_scale doldurur)
    if accept:
        homo.status = "auto_accepted"; homo.source_clip = str(video_path)
        if out_json:
            homo.save(out_json)
        rep["stage"] = "kabul"
        return homo, rep
    rep["stage"] = "gate reddetti (refuse)"
    return None, rep


if __name__ == "__main__":
    import sys, json
    vid = sys.argv[1] if len(sys.argv) > 1 else "raw/cankaya_cam2_clip2400.mp4"
    cam = sys.argv[2] if len(sys.argv) > 2 else "auto_test"
    homo, rep = autocalibrate_venue(vid, cam, template=PitchTemplate.seven_a_side(L=34, W=18))
    print(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
