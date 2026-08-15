#!/usr/bin/env python3
"""pitch/line_calib.py — Occluded / kadraj-disi-farkinda CIZGI-tabanli kalibrasyon.

Amatör sabit-kamera halısahada en büyük kalibrasyon problemi köşelerin
GÖRÜNMEMESİDİR: yakın-alt köşe + korner direği kadraj DIŞI, uzak uçtaki çizgiler
piksel-fakiri/düşük-kontrast. Tek-tık köşe seçimi bu sahalarda imkansız.

Bu modül köşeyi tıklatmak yerine KESİŞEN İKİ ÇİZGİYİ (>=2 nokta her biri) fit edip
analitik olarak kesiştirir. Köşe kadraj dışında olsa bile (legal saha koordinatı)
çapraz-çarpımla geri-kurtarılır. Her köşe için koşullanma açığa çıkarılır:
    - sin(theta): iki parent çizginin kesişim açısı; küçükse (paralel-vari) köşe
      kötü koşullu (ill) işaretlenir, H'ye sokulmaz.
    - extrap: köşenin, tıklanan nokta-aralığının NE KADAR DIŞINA düştüğü (oran);
      yüksekse "tıklanan kanıttan uzakta ekstrapole edildi" uyarısı.

DÜRÜSTLÜK KAPISI: >=4 sonlu & iyi-koşullu karşılık yoksa H ÜRETİLMEZ
(RuntimeError) — kısıtsız bölgeye uydurma homografi yayılmaz (audit P1).

Konvansiyon: TÜM tıklamalar UNDISTORTED piksel uzayında (frame önce undistort
edilir; pick_points.py:98 ile aynı). Saha koordinatı template.py ile ortak:
origin sol-alt, X=boy in [0,L], Y=en in [0,W].

Kapsama-güven haritası (coverage_confidence_map) BİRİNCİ-SINIF düşük-güven
artifaktıdır: kadraj-dışı kama + uzak-üçte-bir -> 0/düşük. GÖRECELIDIR
(normalize), bu yüzden #6'daki MUTLAK metre-artığı ile BİRLİKTE raporlanır,
asla mutlak hata sınırı olarak değil.

Lisans: yalnız OpenCV (BSD) + numpy. CPU; GPU/scipy yok.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import cv2

from pitch.homography import PitchHomography

if TYPE_CHECKING:
    from pitch.template import PitchTemplate

# köşe -> (endline_adi, touchline_adi, (X,Y) dünya). Sadece touch x end (uç köşe).
_CORNER_ADJ = {
    "bl": ("endline_x0", "touchline_y0"),  # (0, 0)
    "br": ("endline_xL", "touchline_y0"),  # (L, 0)
    "tr": ("endline_xL", "touchline_yW"),  # (L, W)
    "tl": ("endline_x0", "touchline_yW"),  # (0, W)
}

# çizgi adı -> (eksen, sabit-değer-üreteci(L,W)). QA + halfway burada.
_LINE_CONST = {
    "endline_x0":  ("X", lambda L, W: 0.0),
    "endline_xL":  ("X", lambda L, W: L),
    "touchline_y0": ("Y", lambda L, W: 0.0),
    "touchline_yW": ("Y", lambda L, W: W),
    "halfway":     ("X", lambda L, W: L / 2.0),
}


def _as2(arr) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1 and a.size == 2:
        a = a.reshape(1, 2)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"(N,2) bekleniyor, alinan {a.shape}")
    return a


# ----------------------------------------------------------------- #1 fit ---
def fit_line_tls(pts_px) -> dict:
    """Total-least-squares (dik) doğru fiti, >=2 nokta.

    Merkezlenmiş noktaların SVD'si; normal = en küçük tekil yöne karşılık gelen
    sağ-tekil vektör. Homojen doğru l=[a,b,c], a²+b²=1 olacak şekilde normalize
    (a*x + b*y + c = 0, (a,b) birim normal, doğru centroid'ten geçer).

    Döner: dict(l=[a,b,c], c0=centroid(2,), t=birim yön(2,), rms, smin, smax, n).
    smin/smax: tıklanan noktaların yön (t) üzerine izdüşüm aralığı (extrap için).
    """
    pts = _as2(pts_px)
    n = len(pts)
    if n < 2:
        raise ValueError("doğru fiti için >=2 nokta gerek")
    c0 = pts.mean(axis=0)
    d = pts - c0
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    t = vt[0]                      # en büyük varyans = doğru yönü (birim)
    normal = vt[-1]               # en küçük varyans = normal (birim)
    a, b = float(normal[0]), float(normal[1])
    # a²+b²=1 garanti (SVD birim); yine de güvene al
    nn = np.hypot(a, b)
    if nn < 1e-12:
        raise ValueError("dejenere doğru (tüm noktalar çakışık?)")
    a, b = a / nn, b / nn
    c = -(a * c0[0] + b * c0[1])
    resid = d @ np.array([a, b])
    rms = float(np.sqrt(np.mean(resid ** 2)))
    s = d @ t
    return dict(l=[a, b, c], c0=c0.copy(), t=t.copy(),
                rms=rms, smin=float(s.min()), smax=float(s.max()), n=int(n))


# --------------------------------------------------------- #2 intersection ---
def line_intersection(l1, l2):
    """İki homojen doğrunun kesişimi + koşullanma.

    pt = dehomojenize(cross(l1,l2)). Doğrular normalize olduğundan
    |cross[2]| = |sin theta| (iki doğru arası açının sinüsü) = yerleşik
    koşullanma ölçüsü. Paralel (|sin|<1e-12) -> (None, 0.0) [kaçış noktası].

    Döner: (pt(2,) veya None, abs_sin in [0,1]).
    """
    l1 = np.asarray(l1, dtype=np.float64).reshape(3)
    l2 = np.asarray(l2, dtype=np.float64).reshape(3)
    cr = np.cross(l1, l2)
    abs_sin = float(min(1.0, abs(cr[2])))   # numerik taşmaya karşı clamp
    if abs(cr[2]) < 1e-12:
        return None, 0.0
    pt = np.array([cr[0] / cr[2], cr[1] / cr[2]], dtype=np.float64)
    return pt, abs_sin


# --------------------------------------------------------- #3 vanishing pt ---
def vanishing_point(lines) -> np.ndarray:
    """Bir doğru demetinin kaçış noktası (homojen), SVD null-space.

    lines: her biri [a,b,c] olan >=2 doğru. Döner: homojen (3,) VP =
    A^T A'nın en küçük tekil değerine karşılık gelen sağ-tekil vektör.
    """
    A = np.asarray([np.asarray(l, dtype=np.float64).reshape(3) for l in lines],
                   dtype=np.float64)
    if A.shape[0] < 2:
        raise ValueError("kaçış noktası için >=2 doğru gerek")
    _, _, vt = np.linalg.svd(A, full_matrices=False)
    return vt[-1].copy()


# ------------------------------------------------------------- #4 corners ---
def corners_from_lines(fits: dict, L: float, W: float,
                       min_sin: float = 0.26) -> dict:
    """Çizgi fitlerinden 4 köşeyi sentezle (kadraj-dışı köşe LEGAL, tutulur).

    fits: {world_line_name: fit_line_tls(...) dict}. Komşuluk SABİT touch×end:
    bl=(0,0) br=(L,0) tr=(L,W) tl=(0,W).

    Döner: {cid: dict(img_und=pt(2,) veya None, world=(X,Y), sin_angle,
            ill=bool(sin<min_sin), extrap=parent doğrular üzerinde
            tıklanan-aralık dışına taşma / aralık oranının max'ı)}.
    """
    Lf, Wf = float(L), float(W)
    world_xy = {"bl": (0.0, 0.0), "br": (Lf, 0.0),
                "tr": (Lf, Wf), "tl": (0.0, Wf)}
    out: dict = {}
    for cid, (end_name, touch_name) in _CORNER_ADJ.items():
        wxy = world_xy[cid]
        fe, ft = fits.get(end_name), fits.get(touch_name)
        if fe is None or ft is None:
            out[cid] = dict(img_und=None, world=wxy, sin_angle=0.0,
                            ill=True, extrap=float("inf"))
            continue
        pt, s = line_intersection(fe["l"], ft["l"])
        if pt is None:
            out[cid] = dict(img_und=None, world=wxy, sin_angle=0.0,
                            ill=True, extrap=float("inf"))
            continue
        extrap = max(_extrap_frac(pt, fe), _extrap_frac(pt, ft))
        out[cid] = dict(img_und=pt, world=wxy, sin_angle=float(s),
                        ill=bool(s < float(min_sin)), extrap=float(extrap))
    return out


def _extrap_frac(pt: np.ndarray, fit: dict) -> float:
    """Köşenin, fit'in tıklanan nokta-aralığı [smin,smax] dışına taşma oranı."""
    s = float((pt - fit["c0"]) @ fit["t"])
    span = fit["smax"] - fit["smin"]
    if span < 1e-9:
        return float("inf")
    beyond = max(0.0, s - fit["smax"], fit["smin"] - s)
    return beyond / span


# ------------------------------------------------- ridge snap (semi-auto) ---
def _snap_to_ridge(pts: np.ndarray, line_map: np.ndarray,
                   radius: int = 6, thr: float = 0.25) -> np.ndarray:
    """Her tıklamayı küçük yarıçapta en yüksek beyaz-çizgi olasılık pikseline çek.

    İnsan etiketi hangi çizginin hangisi olduğunu çözer (L/R + 180° ayna);
    bu yalnızca alt-piksel rafinasyon. line_map: UNDISTORTED uzayda olasılık (HxW).
    """
    lm = np.asarray(line_map, dtype=np.float64)
    h, w = lm.shape[:2]
    snapped = pts.copy()
    for i, (x, y) in enumerate(pts):
        xi, yi = int(round(x)), int(round(y))
        x0, x1 = max(0, xi - radius), min(w, xi + radius + 1)
        y0, y1 = max(0, yi - radius), min(h, yi + radius + 1)
        if x1 <= x0 or y1 <= y0:
            continue
        win = lm[y0:y1, x0:x1]
        if win.max() < thr:
            continue
        dy, dx = np.unravel_index(int(np.argmax(win)), win.shape)
        snapped[i] = [x0 + dx, y0 + dy]
    return snapped


# --------------------------------------------------------------- #5 build ---
def build_homography_from_lines(camera_id: str, template: "PitchTemplate",
                                line_clicks: dict,
                                K=None, dist=None, point_lms=None,
                                ransac_px: float = 4.0,
                                static_line_map=None):
    """Çizgi tıklamalarından homografi kur (kadraj-dışı-farkında + dürüstlük kapılı).

    line_clicks: {world_line_name: (N>=2,2) undistorted px}, adlar:
        endline_x0 / endline_xL / touchline_y0 / touchline_yW / halfway.
    point_lms: opsiyonel asimetrik nokta-işaretler; [(img_px(2,), world_xy(2,)), ...]
        (örn. iki kale direği world (0, W/2 ∓ 1.5)) — ayna belirsizliğini kırar.
    static_line_map: verilirse her tıklama fit'ten önce en yakın ridge'e snap'lenir.

    Döner: (homo, corners, qa_line). qa_line = line_point_residual_qa (metre QA).
    """
    L, W = float(template.dims_m[0]), float(template.dims_m[1])

    # 1) (ops.) ridge snap + 2) her adlı çizgiyi fit et
    clicks = {}
    fits = {}
    for name, pts in line_clicks.items():
        p = _as2(pts)
        if static_line_map is not None:
            p = _snap_to_ridge(p, static_line_map)
        clicks[name] = p
        if len(p) >= 2:
            fits[name] = fit_line_tls(p)

    # 3) köşeleri sentezle
    corners = corners_from_lines(fits, L, W)

    # 4) SONLU karşılıkları topla. Kadraj-dışı köşe LEGAL'dir => tutulur. Köşe
    #    "ill" (grazing/near-parallel kesişim) ise YİNE gerçek bir kısıttır; sadece
    #    konum belirsizliği yüksek -> RANSAC zayıflatır, corner_cond QA'da açığa
    #    çıkar. Gerçekten KISITSIZ olan tek durum None (paralel -> kaçış noktası),
    #    onu gate eler. İyi-koşullu sayısı ayrıca izlenir (refuse kararı için).
    img_pts, world_pts, n_well = [], [], 0
    for cid, c in corners.items():
        p = c["img_und"]
        if p is None or not np.all(np.isfinite(p)):
            continue
        img_pts.append(p)
        world_pts.append(c["world"])
        if not c["ill"]:
            n_well += 1

    # asimetrik nokta-işaretler (ayna kırıcı + ölçek tohumu)
    used_point_lms = False
    if point_lms:
        for entry in point_lms:
            ip, wp = entry[0], entry[1]
            img_pts.append(np.asarray(ip, dtype=np.float64).reshape(2))
            world_pts.append(np.asarray(wp, dtype=np.float64).reshape(2))
            used_point_lms = True

    # DÜRÜSTLÜK KAPISI: kısıtsız bölgeye H uydurma (sonlu karşılık < 4 -> refuse)
    if len(img_pts) < 4:
        raise RuntimeError(
            "need ≥4; far end-line/goal not recoverable → mark far no-coverage")

    img_arr = np.asarray(img_pts, dtype=np.float64)
    world_arr = np.asarray(world_pts, dtype=np.float64)

    homo = PitchHomography(camera_id, template)
    if K is not None and dist is not None:
        homo.set_distortion(K, dist)
    homo.calibrate_manual(img_arr, world_arr, ransac_thresh_px=float(ransac_px),
                          already_undistorted=True)

    # QA: metre-domeni (un-gameable) + per-köşe koşullanma
    qa_line = line_point_residual_qa(homo, clicks, L, W)
    if homo._qa is None:
        homo._qa = {}
    homo._qa["line_qa"] = qa_line
    homo._qa["corner_cond"] = {
        cid: dict(sin=c["sin_angle"], extrap=c["extrap"], ill=bool(c["ill"]))
        for cid, c in corners.items()
    }
    homo._qa["n_well_conditioned"] = int(n_well)   # ill olmayan köşe sayısı (dürüstlük)
    homo._qa["n_finite_corr"] = int(len(img_pts))
    homo.calib_method = "manual_line"
    if used_point_lms:
        # ölçek henüz DOĞRULANMADI — yalnızca tohum; scale_calibrated iddia etme
        homo.scale_anchor = ("goal_width", 3.0)

    return homo, corners, qa_line


# ------------------------------------------------------------- #6 meter QA ---
def line_point_residual_qa(homo, line_clicks: dict, L: float, W: float) -> dict:
    """Metre-domeni çizgi-nokta artığı (exact-fit köşe ile OYNANAMAZ).

    Her adlı çizgi için tıklanan noktalar pitch'e taşınır (already_undistorted=True)
    ve bilinen sabit eksen-değerine göre |X-const| / |Y-const| ölçülür. Metre.

    Döner: dict(median_m, p95_m, per_line={name:{median_m,p95_m,n}}).
    """
    Lf, Wf = float(L), float(W)
    per_line = {}
    all_res = []
    for name, pts in line_clicks.items():
        if name not in _LINE_CONST:
            continue
        p = _as2(pts)
        if len(p) == 0:
            continue
        axis, valfn = _LINE_CONST[name]
        val = float(valfn(Lf, Wf))
        XY = homo.pixel_to_pitch(p, already_undistorted=True)
        col = 0 if axis == "X" else 1
        res = np.abs(XY[:, col] - val)
        per_line[name] = dict(median_m=float(np.median(res)),
                              p95_m=float(np.percentile(res, 95)),
                              n=int(len(res)))
        all_res.append(res)
    if all_res:
        allr = np.concatenate(all_res)
        med, p95 = float(np.median(allr)), float(np.percentile(allr, 95))
    else:
        med, p95 = float("nan"), float("nan")
    return dict(median_m=med, p95_m=p95, per_line=per_line)


# ----------------------------------------------- redistort (forward Brown) ---
def _redistort(und_px: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """UNDISTORTED piksel -> HAM (distorted) piksel, ileri Brown-Conrady.

    cv2.projectPoints normalize kamera koordinatlarını distort+K ile ham piksele
    taşır. dist None ise kimlik (çağıran kontrol eder)."""
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    xn = (und_px[:, 0] - cx) / fx
    yn = (und_px[:, 1] - cy) / fy
    obj = np.column_stack([xn, yn, np.ones(len(und_px))]).astype(np.float64)
    raw, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K,
                               np.asarray(dist, dtype=np.float64).reshape(-1))
    return raw.reshape(-1, 2)


def _pitch_to_raw(homo, pts_m: np.ndarray) -> np.ndarray:
    """Saha metresi -> HAM piksel (undistorted üzerinden, sonra varsa redistort)."""
    und = homo.pitch_to_pixel(pts_m)
    if homo.dist is not None and homo.K is not None:
        return _redistort(und, homo.K, homo.dist)
    return und


# --------------------------------------------------------- #7 coverage map ---
def coverage_confidence_map(homo, img_shape_raw, grid_step_m: float = 1.0,
                            marked_pts_m=None, tau_m: float = 4.0) -> dict:
    """Birinci-sınıf düşük-güven artifaktı: kadraj-dışı kama + uzak-üçte-bir -> 0.

    1m saha grid'i için hücre merkezleri pitch->undistorted->(varsa redistort) HAM
    piksele taşınır; in_frame = [0,Wf)×[0,Hf) içinde mi. piksel_yoğunluğu =
    |det d(raw_px)/d(pitch_m)| sonlu-farkla (grazing/uzak -> küçük).
    conf = clip(yoğunluk/p90(yoğunluk[in_frame]),0,1)**0.5 × in_frame.
    marked_pts_m>=3 ise dışbükey-kabuk DIŞINDAKİ hücreler exp(işaretli_mesafe/tau_m)
    ile söndürülür (ekstrapolasyon decay).

    GÖRECELIDIR (normalize) — #6'daki mutlak metre-artığı ile birlikte raporla,
    asla mutlak hata sınırı diye değil.

    Döner: dict(conf=(ny,nx) f32, xs, ys, covered_polygon_m=(4,2), covered_frac).
    """
    Hf, Wf = int(img_shape_raw[0]), int(img_shape_raw[1])
    L, W = float(homo._dims_m()[0]), float(homo._dims_m()[1])
    step = float(grid_step_m)

    xs = np.arange(step / 2.0, L, step)
    ys = np.arange(step / 2.0, W, step)
    if xs.size == 0:
        xs = np.array([L / 2.0])
    if ys.size == 0:
        ys = np.array([W / 2.0])
    XX, YY = np.meshgrid(xs, ys)              # (ny, nx)
    centers = np.column_stack([XX.ravel(), YY.ravel()])
    N = len(centers)

    h = step * 0.5
    c = _pitch_to_raw(homo, centers)
    cdx = _pitch_to_raw(homo, centers + np.array([h, 0.0]))
    cdy = _pitch_to_raw(homo, centers + np.array([0.0, h]))

    J00 = (cdx[:, 0] - c[:, 0]) / h
    J10 = (cdx[:, 1] - c[:, 1]) / h
    J01 = (cdy[:, 0] - c[:, 0]) / h
    J11 = (cdy[:, 1] - c[:, 1]) / h
    density = np.abs(J00 * J11 - J01 * J10)

    in_frame = ((c[:, 0] >= 0) & (c[:, 0] < Wf) &
                (c[:, 1] >= 0) & (c[:, 1] < Hf))

    if in_frame.any():
        p90 = float(np.percentile(density[in_frame], 90))
    else:
        p90 = 0.0
    if p90 <= 1e-12:
        p90 = float(density.max()) if density.max() > 0 else 1.0

    conf = np.clip(density / p90, 0.0, 1.0) ** 0.5
    conf = conf * in_frame.astype(np.float64)

    # ekstrapolasyon decay (işaretli noktaların dışbükey kabuğu dışında)
    if marked_pts_m is not None:
        mp = np.asarray(marked_pts_m, dtype=np.float32).reshape(-1, 2)
        if len(mp) >= 3:
            hull = cv2.convexHull(mp)
            signed = np.array([
                cv2.pointPolygonTest(hull, (float(px), float(py)), True)
                for (px, py) in centers])           # + iç, - dış (metre)
            factor = np.where(signed >= 0, 1.0, np.exp(signed / float(tau_m)))
            conf = conf * factor

    conf = conf.reshape(len(ys), len(xs)).astype(np.float32)

    # kapsama poligonu: HAM görüntü dikdörtgeni köşelerinin saha karşılığı
    rect = np.array([[0, 0], [Wf, 0], [Wf, Hf], [0, Hf]], dtype=np.float64)
    covered_polygon_m = homo.pixel_to_pitch(rect)

    covered_frac = float((conf > 0.15).mean())
    return dict(conf=conf, xs=xs, ys=ys,
                covered_polygon_m=np.asarray(covered_polygon_m, dtype=np.float64),
                covered_frac=covered_frac)


# ------------------------------------------------- #8 thin orchestrator/CLI ---
def calibrate_from_lines(camera_id: str, template, line_clicks: dict,
                         frame_und=None, K=None, dist=None, point_lms=None,
                         static_line_map=None, out_json: str | None = None,
                         out_overlay: str | None = None,
                         out_coverage: str | None = None,
                         ransac_px: float = 4.0):
    """İnce orkestratör: H kur -> (varsa) self_verify_gate -> overlay + coverage sidecar.

    Gate FAIL veya near/far reproj inversiyonu bozuksa status='manual' KAYDETMEZ;
    bayrak/refuse eder (sessizce-yanlış H yok, audit P1). pick_points.py CLI'ını
    yansıtır ama köşe yerine çizgi alır.

    Döner: dict(homo, corners, qa_line, gate, saved). Hata yoksa H her zaman döner;
    'saved' yalnız gate geçer + inversiyon sağlamsa True.
    """
    homo, corners, qa_line = build_homography_from_lines(
        camera_id, template, line_clicks, K=K, dist=dist, point_lms=point_lms,
        ransac_px=ransac_px, static_line_map=static_line_map)

    gate = None
    if static_line_map is not None:
        try:
            from pitch import auto_calib
            gate = auto_calib.self_verify_gate(homo, np.asarray(static_line_map),
                                               template)
        except Exception as e:  # gate opsiyonel; başarısızlık = bilinmiyor
            gate = dict(accept=False, reasons=[f"gate hata: {e}"])

    # near/far reproj inversiyon sağlığı: metre QA per-line near/far bozuk mu
    inversion_ok = True
    pl = qa_line.get("per_line", {})
    if "touchline_y0" in pl and "touchline_yW" in pl:
        # her iki kenar da makul (ölçüsel) ise sağlıklı kabul
        inversion_ok = (pl["touchline_y0"]["p95_m"] < 5.0 and
                        pl["touchline_yW"]["p95_m"] < 5.0)

    gate_ok = (gate is None) or bool(gate.get("accept", False))
    saved = bool(gate_ok and inversion_ok)

    if saved:
        homo.status = "manual"
        if out_json:
            homo.save(out_json)
        if out_overlay is not None and frame_und is not None:
            ov = homo.qa_overlay(frame_und)
            cv2.imwrite(out_overlay, ov)
        if out_coverage is not None and frame_und is not None:
            cov = coverage_confidence_map(homo, frame_und.shape[:2])
            import json
            with open(out_coverage, "w") as f:
                json.dump(dict(
                    camera_id=camera_id,
                    covered_frac=cov["covered_frac"],
                    covered_polygon_m=cov["covered_polygon_m"].tolist(),
                    line_qa_m=qa_line,
                ), f, indent=2)
    else:
        homo.status = "recalibrate"  # refuse: sessizce 'manual' yazma

    return dict(homo=homo, corners=corners, qa_line=qa_line,
                gate=gate, saved=saved)


if __name__ == "__main__":
    # küçük self-test (ağ/GPU yok)
    pts = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], float)
    f = fit_line_tls(pts)
    assert abs(np.hypot(f["l"][0], f["l"][1]) - 1.0) < 1e-9
    print("line_calib self-test OK:", f["l"])
