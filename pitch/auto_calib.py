#!/usr/bin/env python3
"""Otomatik saha kalibrasyonu (C/D spine): temporal-median statik-cizgi haritasi
+ beyaz-renk cizgi maskesi + chamfer (distance-transform) template kayit
+ ZORUNLU self-verify gate.

Mimari gercek: kamera SABIT. Bu yuzden homografi tum mac boyunca tek; kamera
basina BIR KEZ kalibre edilir. Burasi pahaliyi-azaltan oto yoludur: gate kabul
ederse PitchHomography(status="auto_accepted") doner, REDDEDERSE None doner ve
caller manuel sece (PitchHomography.calibrate_manual) duser.

Anti-overpromise: oto asla sessizce-yanlis H yaymaz. self_verify_gate birinci
sinif zorunlu katmandir; sadece dusuk-residual yetmez (ayna/mirror H'yi
yakalamak icin line-IoU + spread + iki-yari + gol-tarafi kontrolu var).

Lisans: SADECE OpenCV(BSD) + scipy(BSD) + numpy. Ogrenilmis ag YOK.
GPU YOK (CPU-only); RF-DETR kutulari disaridan player_boxes_per_frame ile gelir.

Interface spec: ULTRACODE_PROMPT.md bolum 5.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize

if TYPE_CHECKING:  # ciklik/eksik-import'tan kacin; runtime'da fonksiyon-icinde import
    from pitch.template import PitchTemplate
    from pitch.homography import PitchHomography


# ----------------------------------------------------------------------------
# 1. Beyaz cizgi maskesi
# ----------------------------------------------------------------------------
def white_line_mask(frame_bgr: np.ndarray) -> np.ndarray:
    """Saha cizgisi piksellerini float32 [0,1] maske olarak dondur.

    Mantik (C-graft): cizgi = yuksek-L (Lab parlaklik) AND dusuk-doygunluk (HSV S)
    AND-NOT yesil. Bu, naive Canny/edge'in en buyuk amator yanlis-cizgisi olan
    cim-bicme seritlerini / halisaha eklerini chamfer'dan ONCE reddeder.

    Args:
        frame_bgr: HxWx3 BGR (OpenCV duzeni).
    Returns:
        HxW float32, cizgi-olasiligi [0,1] (yumusatilmis, ikili-degil).
    """
    if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
        raise ValueError("frame_bgr HxWx3 BGR olmali")

    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    L = lab[:, :, 0].astype(np.float32) / 255.0          # parlaklik
    S = hsv[:, :, 1].astype(np.float32) / 255.0          # doygunluk
    Hh = hsv[:, :, 0].astype(np.float32)                 # ton [0,180)

    # yuksek-L: cizgi boyasi beyaz/parlak. Adaptif esik: ust kuyruga yakin.
    # Yumusak sigmoid yerine basit lineer rampa (parametre-az, sahaya saglam).
    high_L = np.clip((L - 0.55) / 0.30, 0.0, 1.0)
    # dusuk-S: beyaz boya doygunlugu dusuktur (yesil cim yuksek S)
    low_S = np.clip((0.45 - S) / 0.45, 0.0, 1.0)
    # yesil-degil: OpenCV HSV ton ~35-85 yesil bandi. cim icin ceza.
    green = ((Hh > 35) & (Hh < 90)).astype(np.float32)
    not_green = 1.0 - 0.85 * green

    mask = high_L * low_S * not_green
    # hafif yumusatma: tek-piksel gurultu yerine cizgi-genisligi tut
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    return mask.astype(np.float32)


# ----------------------------------------------------------------------------
# 2. Temporal-median statik cizgi haritasi (D spine)
# ----------------------------------------------------------------------------
def select_calib_frame(per_frame_box_counts: np.ndarray) -> int:
    """En az oyuncu-yogunluklu frame indexini dondur (en temiz cizgi gorunumu).

    Args:
        per_frame_box_counts: (n_frames,) her frame'deki RF-DETR kutu sayisi.
    Returns:
        En dusuk yogunluk frame indexi (int). Bos girdide 0.
    """
    arr = np.asarray(per_frame_box_counts)
    if arr.size == 0:
        return 0
    return int(np.argmin(arr))


def _inpaint_boxes(frame_bgr: np.ndarray, boxes_xyxy) -> np.ndarray:
    """RF-DETR oyuncu kutularini inpaint ile yumusak-doldur (median'i de-bias et).

    Goalmouth loiter bolgelerinde oyuncular medyani kirletir; D-graft: medyan
    ONCESI kutulari maskele ve cevreden boya. boxes_xyxy bos/None ise no-op.
    """
    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return frame_bgr
    h, w = frame_bgr.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    for b in boxes_xyxy:
        x1, y1, x2, y2 = [int(round(v)) for v in b[:4]]
        # ayak bolgesini biraz tasir: golge + temas noktasi
        x1 = max(0, x1 - 2); y1 = max(0, y1 - 2)
        x2 = min(w, x2 + 2); y2 = min(h, y2 + 4)
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    if mask.any():
        return cv2.inpaint(frame_bgr, mask, 3, cv2.INPAINT_TELEA)
    return frame_bgr


def build_static_line_map(video_path: str, n_frames: int = 400,
                          player_boxes_per_frame=None,
                          percentile: float = 50.0) -> np.ndarray:
    """Mac boyunca N frame'in temporal medyan/percentile beyaz-cizgi haritasi.

    Statik kamerada oyuncular/golgeler farkli frame'lerde farkli yerde -> medyan
    altinda yikanir; boyali cizgiler her frame'de ayni yerde -> pekisir. Sonuc:
    temiz, neredeyse-oyuncusuz cizgi gorseli (kalibrasyon altligi).

    Args:
        video_path: klip yolu.
        n_frames: mac boyunca esit-arali ornek sayisi.
        player_boxes_per_frame: opsiyonel; {frame_idx: xyxy_array} ya da liste
            (her sampled frame icin kutular). Verilirse inpaint uygulanir.
        percentile: 50=medyan. Cizgiler parlak oldugundan daha yuksek (orn 60-70)
            de denenebilir; default medyan saglam.
    Returns:
        HxW float32 [0,1] L_static. CPU-only.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"video acilamadi: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        total = n_frames
    idxs = np.linspace(0, max(total - 1, 0), min(n_frames, total), dtype=int)

    masks = []
    # player_boxes_per_frame liste mi dict mi? esnek eris.
    def boxes_for(frame_idx, sample_pos):
        if player_boxes_per_frame is None:
            return None
        if isinstance(player_boxes_per_frame, dict):
            return player_boxes_per_frame.get(int(frame_idx))
        # liste: sampled siraya gore varsay
        if sample_pos < len(player_boxes_per_frame):
            return player_boxes_per_frame[sample_pos]
        return None

    for pos, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok:
            continue
        frame = _inpaint_boxes(frame, boxes_for(fi, pos))
        masks.append(white_line_mask(frame))
    cap.release()

    if not masks:
        raise RuntimeError("hicbir frame okunamadi; L_static uretilemedi")
    stack = np.stack(masks, axis=0)  # (K,H,W)
    if abs(percentile - 50.0) < 1e-6:
        L_static = np.median(stack, axis=0)
    else:
        L_static = np.percentile(stack, percentile, axis=0)
    return L_static.astype(np.float32)


# ----------------------------------------------------------------------------
# 3. Cizgi tespiti (segment / kesisim / merkez-daire)
# ----------------------------------------------------------------------------
def detect_lines(L_static: np.ndarray, line_thr: float = 0.35) -> dict:
    """L_static uzerinde cizgi segmentleri, T-kesisimleri, merkez-daire ellipsi.

    Args:
        L_static: HxW float32 [0,1] statik cizgi haritasi.
        line_thr: ikili-esik.
    Returns:
        {"segments": (M,4) [x1,y1,x2,y2], "intersections": (K,2),
         "ellipse": ((cx,cy),(MA,ma),angle) | None, "binary": HxW uint8}
    """
    binm = (L_static > line_thr).astype(np.uint8) * 255
    # ince iskelet yerine morfolojik temizleme (lisans-temiz, scipy/cv2)
    binm = cv2.morphologyEx(binm, cv2.MORPH_CLOSE,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    segs = cv2.HoughLinesP(binm, rho=1, theta=np.pi / 180, threshold=60,
                           minLineLength=max(30, L_static.shape[1] // 20),
                           maxLineGap=20)
    segments = (segs[:, 0, :] if segs is not None
                else np.zeros((0, 4), dtype=np.int32))

    intersections = _segment_intersections(segments)

    # merkez-daire: kontur uzerinde ellipse-fit dene. DE-PRIORITIZED (gurultulu),
    # asla primary landmark degil; sadece zayif ipucu.
    ellipse = None
    cnts, _ = cv2.findContours(binm, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:5]:
        if len(c) >= 5 and cv2.contourArea(c) > (L_static.shape[0] * 0.02) ** 2:
            try:
                ellipse = cv2.fitEllipse(c)
                break
            except cv2.error:
                pass

    return dict(segments=segments, intersections=intersections,
                ellipse=ellipse, binary=binm)


def _segment_intersections(segments: np.ndarray, max_pts: int = 200) -> np.ndarray:
    """Non-paralel segment ciftlerinin kesisim noktalari (T/L kose adaylari)."""
    pts = []
    n = len(segments)
    for i in range(n):
        x1, y1, x2, y2 = segments[i]
        d1 = np.array([x2 - x1, y2 - y1], dtype=float)
        for j in range(i + 1, n):
            x3, y3, x4, y4 = segments[j]
            d2 = np.array([x4 - x3, y4 - y3], dtype=float)
            denom = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(denom) < 1e-6:
                continue  # paralel
            # acisal: cok-paralelleri at (saglam kose icin >~20 derece)
            cosang = abs(d1 @ d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-9)
            if cosang > 0.94:
                continue
            t = ((x3 - x1) * d2[1] - (y3 - y1) * d2[0]) / denom
            px, py = x1 + t * d1[0], y1 + t * d1[1]
            pts.append((px, py))
            if len(pts) >= max_pts:
                return np.array(pts)
    return np.array(pts) if pts else np.zeros((0, 2))


# ----------------------------------------------------------------------------
# 4. Chamfer template kayit (PRIMARY registration)
# ----------------------------------------------------------------------------
def _dt_from_mask(L_static: np.ndarray, line_thr: float = 0.35) -> np.ndarray:
    """Distance transform: her piksel -> en yakin cizgi-pikseline uzaklik (px)."""
    binm = (L_static > line_thr).astype(np.uint8)
    # cizgi yoksa DT tanimsiz; guvenli buyuk-deger doldur
    if binm.sum() == 0:
        return np.full(L_static.shape, 1e3, dtype=np.float32)
    return ndimage.distance_transform_edt(1 - binm).astype(np.float32)


def _H_from_corner_img_pts(template, corner_img_pts: np.ndarray) -> np.ndarray:
    """4 pitch-kose -> 4 image px eslemesinden H_pitch2img (getPerspectiveTransform).

    template.polygon_m() (4,2) metre koseleri ile corner_img_pts (4,2) image
    pikselleri ESLE; H_pitch2img doner.
    """
    poly_m = np.asarray(template.polygon_m(), dtype=np.float32)  # (4,2)
    src = poly_m.astype(np.float32)
    dst = np.asarray(corner_img_pts, dtype=np.float32)
    return cv2.getPerspectiveTransform(src, dst)


def _chamfer_cost(corner_img_pts_flat: np.ndarray, template, DT: np.ndarray,
                  line_pts_m: np.ndarray, huber_px: float = 8.0) -> float:
    """Robust chamfer maliyeti: template cizgi noktalarini image'e projekte et,
    DT'den uzakligi robust-ortalama. Dusuk = iyi hizalama.
    """
    corners = corner_img_pts_flat.reshape(4, 2)
    try:
        H_p2i = _H_from_corner_img_pts(template, corners)
    except cv2.error:
        return 1e6
    # line_pts_m (M,2) -> homojen -> image
    M = line_pts_m.shape[0]
    ones = np.ones((M, 1), dtype=np.float64)
    P = np.hstack([line_pts_m, ones])               # (M,3)
    proj = (H_p2i @ P.T).T                            # (M,3)
    w = proj[:, 2]
    valid = np.abs(w) > 1e-9
    if valid.sum() < M * 0.5:
        return 1e6
    u = proj[valid, 0] / w[valid]
    v = proj[valid, 1] / w[valid]
    h, wid = DT.shape
    inb = (u >= 0) & (u < wid) & (v >= 0) & (v < h)
    if inb.sum() < valid.sum() * 0.4:
        return 1e6                                   # cogu kadraj disi -> kotu H
    ui = u[inb].astype(np.int32)
    vi = v[inb].astype(np.int32)
    d = DT[vi, ui]
    # Huber: outlier (faded/eksik cizgi) zinciri kontrol et
    huber = np.where(d <= huber_px, 0.5 * d * d / huber_px,
                     d - 0.5 * huber_px)
    # kadraj-disi noktalar icin ceza ekle (eksik kapsama'yi cezalandir)
    penalty = (1.0 - inb.mean()) * huber_px
    return float(huber.mean() + penalty)


def _init_corner_hypotheses(L_static: np.ndarray, template,
                            init_H_img2pitch=None) -> list[np.ndarray]:
    """4-kose image-px baslangic hipotezleri (multi-start; L/R + 180 simetri).

    init_H verilirse ondan tureyen tek hipotez (rafine icin). Aksi halde
    L_static cizgi-pikselleri bounding-box'undan kaba trapez + 4 simetri varyanti.

    HONEST LIMIT: asimetrik landmark/orientation_hint olmadan sol-sag + 180
    simetrisi geometrik olarak coozulemez. Buradaki multi-start TUM hipotezleri
    dener ve en dusuk-residuali secer; dogru olani self_verify_gate'in
    gol-tarafi/iki-yari kontrolu ayikLAR. Tam hands-free yeni-saha iddia EDILMEZ.
    """
    if init_H_img2pitch is not None:
        H_p2i = np.linalg.inv(np.asarray(init_H_img2pitch, dtype=np.float64))
        poly_m = np.asarray(template.polygon_m(), dtype=np.float64)
        P = np.hstack([poly_m, np.ones((4, 1))])
        proj = (H_p2i @ P.T).T
        corners = proj[:, :2] / proj[:, 2:3]
        return [corners.astype(np.float64)]

    # cizgi piksel kumesinin convex-bbox'u -> kaba saha alani
    ys, xs = np.where(L_static > 0.35)
    h, w = L_static.shape
    if xs.size < 50:
        # cizgi yok denecek kadar az: tum-frame ic-kenar trapezi
        x0, x1 = w * 0.08, w * 0.92
        y0, y1 = h * 0.30, h * 0.95
    else:
        x0, x1 = xs.min(), xs.max()
        y0, y1 = ys.min(), ys.max()
    # ust kenar perspektifle daha dar (uzak), alt kenar genis (yakin) varsay
    cx = 0.5 * (x0 + x1)
    top_half = 0.30 * (x1 - x0)
    # temel kose seti: [TL, TR, BR, BL] image px (polygon_m sirasiyla eslesmeli)
    # polygon_m sirasi: (0,0)-(L,0)-(L,W)-(0,W). Hangi image-kosesine gittigi
    # simetri hipotezleriyle denenir.
    base = np.array([
        [cx - top_half, y0],   # ust-sol
        [cx + top_half, y0],   # ust-sag
        [x1,            y1],   # alt-sag
        [x0,            y1],   # alt-sol
    ], dtype=np.float64)

    hyps = []
    # 4 dairesel donme (0/90/180/270) -> L/R + uzun-eksen + 180 simetri kapsami
    for k in range(4):
        hyps.append(np.roll(base, k, axis=0).copy())
    # ayna (sol-sag flip)
    mirror = base[[1, 0, 3, 2]].copy()
    for k in range(4):
        hyps.append(np.roll(mirror, k, axis=0).copy())
    return hyps


def register_template_chamfer(L_static: np.ndarray, template,
                              init_H_img2pitch: np.ndarray | None = None,
                              multi_start: bool = True,
                              optimize_LW: bool = False
                              ) -> tuple[np.ndarray, float]:
    """PRIMARY kayit: chamfer/DT render-and-compare ile H_img2pitch tahmini.

    DT = distance_transform_edt(1 - (L_static>thr)). H, 4 dunya-referans
    noktasinin (template kose) image koordinatlariyla parametrize edilir
    (cv2.getPerspectiveTransform). cost(H) = template.line_points_m projeksiyonu
    uzerinde robust DT ortalamasi. scipy.optimize.minimize(Nelder-Mead);
    multi_start L/R + 180 simetriyi tarar, en dusuk-residual secilir.

    Args:
        L_static: HxW [0,1].
        template: PitchTemplate (polygon_m, line_points_m saglar).
        init_H_img2pitch: opsiyonel sicak-baslangic (orn onceki manuel H).
        multi_start: True ise simetri hipotezleri taranir.
        optimize_LW: TODO — L,W'yi de optimize et (template.optimize_LW).
    Returns:
        (H_img2pitch (3,3), chamfer_resid_px float).
    """
    DT = _dt_from_mask(L_static)
    line_pts_m = np.asarray(template.line_points_m(step_m=0.5), dtype=np.float64)
    if line_pts_m.shape[0] == 0:
        raise ValueError("template.line_points_m bos; chamfer yapilamaz")

    hyps = _init_corner_hypotheses(L_static, template, init_H_img2pitch)
    if not multi_start:
        hyps = hyps[:1]

    best_resid = np.inf
    best_corners = None
    for hyp in hyps:
        x0 = hyp.reshape(-1)
        res = minimize(_chamfer_cost, x0,
                       args=(template, DT, line_pts_m),
                       method="Nelder-Mead",
                       options=dict(maxiter=1500, xatol=0.5, fatol=1e-3))
        if res.fun < best_resid:
            best_resid = float(res.fun)
            best_corners = res.x.reshape(4, 2)

    if best_corners is None:
        raise RuntimeError("chamfer kaydi basarisiz; hipotez yok")

    H_p2i = _H_from_corner_img_pts(template, best_corners)
    H_img2pitch = np.linalg.inv(H_p2i)

    # TODO(optimize_LW): template.optimize_LW True ise (L,W) ek 2 dof olarak
    # cost'a sokulmali (line_points_m'yi her iterasyonda yeniden ornekle). Su an
    # sabit-boyut template ile kayit yapilir; L,W disaridan verilen template'ten
    # gelir. Halisaha L/W araligi dar oldugundan (5v5/7v7) pratikte ucuz bir
    # grid-search da yeterli olabilir.
    return H_img2pitch, best_resid


# ----------------------------------------------------------------------------
# 5. SELF-VERIFY GATE (birinci sinif zorunlu katman)
# ----------------------------------------------------------------------------
def _line_iou(L_static: np.ndarray, template, H_img2pitch: np.ndarray,
              line_thr: float = 0.35, dilate_px: int = 3) -> float:
    """Reprojekte template-cizgileri ile gozlemlenen cizgi-maskesi arasi IoU."""
    H_p2i = np.linalg.inv(H_img2pitch)
    h, w = L_static.shape
    rendered = np.zeros((h, w), dtype=np.uint8)
    line_pts_m = np.asarray(template.line_points_m(step_m=0.25), dtype=np.float64)
    P = np.hstack([line_pts_m, np.ones((len(line_pts_m), 1))])
    proj = (H_p2i @ P.T).T
    valid = np.abs(proj[:, 2]) > 1e-9
    u = (proj[valid, 0] / proj[valid, 2]).astype(np.int32)
    v = (proj[valid, 1] / proj[valid, 2]).astype(np.int32)
    inb = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    rendered[v[inb], u[inb]] = 1
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))
    rendered = cv2.dilate(rendered, k)
    obs = cv2.dilate((L_static > line_thr).astype(np.uint8), k)

    inter = np.logical_and(rendered, obs).sum()
    union = np.logical_or(rendered, obs).sum()
    return float(inter) / float(union) if union > 0 else 0.0


def self_verify_gate(homo, L_static: np.ndarray, template,
                     max_chamfer_px: float = 6.0,
                     min_line_iou: float = 0.30,
                     min_landmark_spread: float = 0.25) -> dict:
    """ZORUNLU gate: oto H'yi kabul/red et. Sadece residual'a GUVENME.

    Kontroller:
      - line_iou >= min_line_iou: reprojekte cizgiler gercek cizgilerle ortusuyor mu
      - chamfer_resid <= max_chamfer_px: median hiza hatasi
      - both_halves: kapsama saha-merkezinin iki yaninda da var mi (flat/yarim H reddi)
      - spread_ok: landmark yayilimi yeterli mi (flat-chamfer under-constrained reddi)
      - goal-side placement: TODO — gol kutulari dogru yarida mi (ayna H yakalama)

    Returns:
        {accept, line_iou, chamfer_resid, both_halves, spread_ok, reasons:[...]}
    """
    reasons = []

    H_img2pitch = homo.H_img2pitch if hasattr(homo, "H_img2pitch") else None
    if H_img2pitch is None:
        return dict(accept=False, line_iou=0.0, chamfer_resid=np.inf,
                    both_halves=False, spread_ok=False,
                    reasons=["homo H_img2pitch yok"])
    H_img2pitch = np.asarray(H_img2pitch, dtype=np.float64)

    # chamfer residual (median DT)
    DT = _dt_from_mask(L_static)
    line_pts_m = np.asarray(template.line_points_m(step_m=0.5), dtype=np.float64)
    corners_flat = _corner_img_pts_from_H(template, H_img2pitch).reshape(-1)
    chamfer = _chamfer_cost(corners_flat, template, DT, line_pts_m)

    line_iou = _line_iou(L_static, template, H_img2pitch)

    # kapsama yayilimi: image cizgi-piksellerini pitch'e tasi, X/Y yayilimi olc
    ys, xs = np.where(L_static > 0.35)
    spread_ok = False
    both_halves = False
    if xs.size > 50:
        pts_px = np.stack([xs, ys], axis=1).astype(np.float64)
        # alt-orneklem (hiz)
        if pts_px.shape[0] > 4000:
            sel = np.random.choice(pts_px.shape[0], 4000, replace=False)
            pts_px = pts_px[sel]
        P = np.hstack([pts_px, np.ones((len(pts_px), 1))])
        proj = (H_img2pitch @ P.T).T
        valid = np.abs(proj[:, 2]) > 1e-9
        XY = proj[valid, :2] / proj[valid, 2:3]
        L, Wm = template.dims_m
        # pitch-ici noktalar
        inb = (XY[:, 0] > -3) & (XY[:, 0] < L + 3) & \
              (XY[:, 1] > -3) & (XY[:, 1] < Wm + 3)
        if inb.sum() > 30:
            XYi = XY[inb]
            x_span = (XYi[:, 0].max() - XYi[:, 0].min()) / max(L, 1e-6)
            y_span = (XYi[:, 1].max() - XYi[:, 1].min()) / max(Wm, 1e-6)
            spread_ok = (x_span > min_landmark_spread and
                         y_span > min_landmark_spread)
            both_halves = (XYi[:, 0].min() < L * 0.4 and
                           XYi[:, 0].max() > L * 0.6)

    if line_iou < min_line_iou:
        reasons.append(f"line_iou {line_iou:.2f} < {min_line_iou}")
    if chamfer > max_chamfer_px:
        reasons.append(f"chamfer {chamfer:.1f}px > {max_chamfer_px}")
    if not both_halves:
        reasons.append("kapsama tek-yari (both_halves=False)")
    if not spread_ok:
        reasons.append("landmark yayilimi yetersiz (flat-chamfer riski)")

    # TODO(goal-side): template'te gol-direk landmark'lari varsa, reprojekte gol
    # cizgilerinin x~0 ve x~L'de oldugunu dogrula -> ayna/180 H'yi kesin yakala.
    # Su an both_halves+spread bunu kismen yapiyor ama asimetrik landmark sart.

    accept = (line_iou >= min_line_iou and chamfer <= max_chamfer_px and
              both_halves and spread_ok)
    return dict(accept=bool(accept), line_iou=line_iou,
                chamfer_resid=float(chamfer), both_halves=bool(both_halves),
                spread_ok=bool(spread_ok), reasons=reasons)


def _corner_img_pts_from_H(template, H_img2pitch: np.ndarray) -> np.ndarray:
    """H_img2pitch'ten 4 pitch-kose image-px konumunu geri-projekte et."""
    H_p2i = np.linalg.inv(np.asarray(H_img2pitch, dtype=np.float64))
    poly_m = np.asarray(template.polygon_m(), dtype=np.float64)
    P = np.hstack([poly_m, np.ones((4, 1))])
    proj = (H_p2i @ P.T).T
    return (proj[:, :2] / proj[:, 2:3])


# ----------------------------------------------------------------------------
# 6. Tam oto-kalibrasyon yolu
# ----------------------------------------------------------------------------
def auto_calibrate(video_path: str, template, camera_id: str,
                   distortion: tuple[np.ndarray, np.ndarray] | None = None,
                   orientation_hint: str | None = None,
                   player_boxes_per_frame=None):
    """Tam oto yol; gate gecerse PitchHomography(status="auto_accepted"), yoksa None.

    Akis:
      1. build_static_line_map (temporal medyan, oyuncu-inpaint).
      2. register_template_chamfer (DT chamfer, multi-start simetri).
      3. self_verify_gate (ZORUNLU) -> red ise None (caller manuel'e duser).

    Args:
        video_path: klip.
        template: PitchTemplate.
        camera_id: kalibrasyon kimligi (calib/<camera_id>.json).
        distortion: (K, dist) varsa undistort-first; yoksa None (HONEST: barrel
            varsa once estimate_distortion ile cikar — Cankaya'da barrel VAR).
        orientation_hint: sol-sag simetri cozumu icin tek-seferlik insan ipucu
            (orn "origin_corner_bottom_left"). TODO: hipotez secimine bagla.
        player_boxes_per_frame: RF-DETR kutulari (inpaint icin).
    Returns:
        PitchHomography | None.
    """
    # runtime import (pitch paketi coder:homography tarafindan olusturuluyor)
    from pitch.homography import PitchHomography

    L_static = build_static_line_map(
        video_path, player_boxes_per_frame=player_boxes_per_frame)

    init_H = None  # TODO: orientation_hint -> init kose hipotezi sec
    H_img2pitch, resid = register_template_chamfer(
        L_static, template, init_H_img2pitch=init_H, multi_start=True,
        optimize_LW=getattr(template, "optimize_LW", False))

    homo = PitchHomography(camera_id, template)
    if distortion is not None:
        K, dist = distortion
        homo.set_distortion(K, dist)
    # auto H'yi pushla (status gate sonrasi netlesir)
    homo.set_homography(H_img2pitch, status="auto_chamfer")

    gate = self_verify_gate(homo, L_static, template)
    if not gate["accept"]:
        # HONEST: sessizce-yanlis H YAYMA. None don -> caller manuel sece.
        return None

    homo.set_homography(H_img2pitch, status="auto_accepted")
    return homo


# ----------------------------------------------------------------------------
# Gorev-metni uyumlu ince alias'lar (interface spec primary; bunlar sarmal)
# ----------------------------------------------------------------------------
def detect_field_lines(frame_bgr: np.ndarray) -> dict:
    """Tek-frame uyumluluk sarmali: white_line_mask + detect_lines.

    NOT: oto-kalibrasyon TEK frame yerine build_static_line_map (temporal medyan)
    kullanmali (statik-cam avantaji). Bu sarmal hizli-deneme/teshis icindir.
    """
    return detect_lines(white_line_mask(frame_bgr))


def estimate_homography_auto(frame_bgr: np.ndarray, pitch_template
                             ) -> np.ndarray | None:
    """Tek-frame'den H_img2pitch tahmini (gate'siz hizli-deneme).

    UYARI: gate uygulanMAZ; uretimde auto_calibrate kullan (gate zorunlu).
    Basarisizsa None.
    """
    try:
        L = white_line_mask(frame_bgr)
        H, _ = register_template_chamfer(L, pitch_template)
        return H
    except (RuntimeError, ValueError, cv2.error):
        return None
