#!/usr/bin/env python3
"""Top-down, guven-agirlikli, durustluk-onceli per-player + per-zone metrik katmani.

Bu modul per-frame track parquet'ini (export_tracks semasi: tid, frame, t_sec,
foot_x, foot_y, box_h, box_w, conf, bottom_cropped, pitch_x, pitch_y, in_pitch)
TESPITI YENIDEN KOSTURMADAN ust-acidan bir metrik raporuna cevirir. Diskte
pitch_x NaN oldugu icin ayak noktalari UCAR projekte edilir (tek geciste,
vektorize), TEK projeksiyon kaynagi olarak pitch.homography.PitchHomography
kullanilir.

Tasarim kararlari (kilitli):
  1. v2 calib kanonik-kalite; bozuk (default-isimli, 65px) calib accept_calib_qa
     ile FAIL-CLOSED cenberlenir -> ondan asla metrik uretilmez.
  2. Tek projeksiyon: yalnizca PitchHomography. _CalibProjector YENIDEN YAZILMAZ.
  3. Kinematik BIR KEZ hesaplanir; ayni hiz dizisi sprint sayimina beslenir
     (mevcut cift-_track_kinematics bug'ini onler).
  4. coverage_mask pitch_to_pixel (undistorted) ile; sinir binleri dusuk-guven.

DURUSTLUK: olcek-capasi (scale_anchor) null oldugu surece birim 'relative_m'dir,
asla 'm'/'km/h' degil; ivme ASLA raporlanmaz; sprint sayisi yalnizca gercek 'm'
modunda; kapsama disi/dusuk-kapsama hucreleri NaN ('unknown'), 0 degil.

Lisans: numpy/scipy/pandas/cv2 (BSD/MIT). GPU yok. matplotlib yok (render ayri).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# stats_report ayni repo kokunde top-level modul; pytest/CLI farkli cwd'lerde
# import edilebilsin diye repo kokunu yola ekle (idempotent).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Reuse honest constants from stats_report WITHOUT calling its per-tid groupby
# paths. Sabitler degisirse tek kaynak korunur.
try:
    from stats_report import (  # type: ignore
        MAX_SPEED_MPS, MAX_GAP_S, SMOOTH_WIN_S, MIN_TRACK_FRAMES,
    )
except Exception:  # pragma: no cover - savunmaci fallback
    MAX_SPEED_MPS = 11.0
    MAX_GAP_S = 1.0
    SMOOTH_WIN_S = 0.5
    MIN_TRACK_FRAMES = 12

# accept_calib_qa esikleri
QA_MEDIAN_PX_MAX = 20.0
QA_MIN_LANDMARKS = 6
QA_NEAR_FAR_RATIO = 3.0  # near > 3*far => geometrik olarak ters H

# guven modeli sabitleri (Design 1)
CONF_REF = 0.7        # bunun ustu tam guven
BOX_H_REF = 90.0      # box yuksekligi referansi (px)
BOX_H_FLOOR = 0.2     # cok kucuk kutu min agirligi
SIGMA_REF_M = 0.5     # pos_sigma yarim-deger noktasi (m)
FOOT_SIGMA_PX = 3.0   # ayak-noktasi taban belirsizligi (px)

FAR_CONF_FRAC = 0.4   # tracklet'in >%40'i uzak-ucteyse hiz guveni 'low'


# ===========================================================================
# 1) Calib kapisi: bozuk calib metrik uretemez (fail-closed)
# ===========================================================================
def accept_calib_qa(qa: dict | None) -> tuple[bool, list[str]]:
    """Calib QA sozlugunu kabul/reddet. Fail-closed.

    Reddeder (ve sebep listesi doner) eger: qa yok; median_px > 20.0;
    n_landmarks < 6; near > 3*far (geometrik olarak ters H).

    Dogrulandi: v2 (median 9.87) gecer; kanonik default cankaya_cam2.json
    (median 65) reddedilir.
    """
    reasons: list[str] = []
    if not qa or not isinstance(qa, dict):
        return False, ["qa missing or not a dict (fail-closed)"]

    med = qa.get("median_px", None)
    if med is None or not np.isfinite(med):
        reasons.append("median_px missing/non-finite (fail-closed)")
    elif float(med) > QA_MEDIAN_PX_MAX:
        reasons.append(
            f"median_px {float(med):.1f} > {QA_MEDIAN_PX_MAX} (reprojection too high)")

    nlm = qa.get("n_landmarks", 0) or 0
    if int(nlm) < QA_MIN_LANDMARKS:
        reasons.append(
            f"n_landmarks {int(nlm)} < {QA_MIN_LANDMARKS} (too few correspondences)")

    pz = qa.get("per_zone", {}) or {}
    near, far = pz.get("near"), pz.get("far")
    if (near is not None and far is not None
            and np.isfinite(near) and np.isfinite(far) and far > 0
            and float(near) > QA_NEAR_FAR_RATIO * float(far)):
        reasons.append(
            f"near {float(near):.1f} > {QA_NEAR_FAR_RATIO}x far {float(far):.1f} "
            "(geometrically backwards H)")

    return (len(reasons) == 0), reasons


# ===========================================================================
# Prep dataclass: tek-gecis hazirlik ciktisi (her sey tid->t_sec sirali)
# ===========================================================================
@dataclass
class Prep:
    # sirali ham/turetilmis diziler (uzunluk = n_rows)
    tid: np.ndarray
    t_sec: np.ndarray
    foot_x: np.ndarray
    foot_y: np.ndarray
    box_h: np.ndarray
    conf: np.ndarray
    px: np.ndarray            # pitch X (relative_m) veya pixel X (fallback)
    py: np.ndarray            # pitch Y / pixel Y
    in_pitch: np.ndarray      # bool
    in_frame: np.ndarray      # bool
    zone: np.ndarray          # int8: 0 near / 1 mid / 2 far
    pos_sigma_m: np.ndarray   # konum belirsizligi (m, fallback'te px)
    w_conf: np.ndarray        # [0,1] guven agirligi
    # per-track indeks yapilari
    uniq: np.ndarray
    starts: np.ndarray
    counts: np.ndarray
    codes: np.ndarray         # uniq icindeki indeks (searchsorted)
    # meta
    fps: float = 25.0
    unit: str = "relative_m"  # relative_m | px | m
    scale_quality: str = "relative"   # relative | approx | exact
    unit_label: str = "relative_m"    # 'relative_m' | 'm (approx +-N%)' | 'm' | 'px'
    band_pct: float | None = None
    space: str = "pitch"      # pitch | pixel
    frame_wh: tuple = (1920, 1080)
    pitch_dims: tuple = (34.0, 18.0)
    per_zone_px: dict = field(default_factory=dict)


# ===========================================================================
# 2) prepare: tek geciste projeksiyon + zone + guven kolonlari
# ===========================================================================
def _local_jacobian_m_per_px(homo, foot_xy: np.ndarray) -> np.ndarray:
    """Yerel Jacobian buyuklugu (m/px), pixel_to_pitch'in 1px pertürbasyonuyla.

    _estimate_heights'taki yerel-olcek fikrini yeniden kullanir: ayak noktasini
    +1px (u ve v) kaydirip saha-metresindeki yer degisimini olcer. Doner: (N,)
    her gozlem icin 0.5*(|d/du| + |d/dv|).
    """
    p0 = homo.pixel_to_pitch(foot_xy)
    pu = homo.pixel_to_pitch(foot_xy + np.array([1.0, 0.0]))
    pv = homo.pixel_to_pitch(foot_xy + np.array([0.0, 1.0]))
    du = np.hypot(*(pu - p0).T)
    dv = np.hypot(*(pv - p0).T)
    return 0.5 * (du + dv)


def prepare(df, homo, fps: float) -> Prep:
    """Tek geciste hazirlik. Tespit yeniden kosturulmaz.

    pitch_x diskte tamamen NaN VE homo varsa (ve kabul edilmisse), foot_x/foot_y
    pixel_to_pitch ile BIR KEZ vektorize projekte edilir (float64, ~ms). in_pitch
    homo.in_pitch ile; her satir foot_y terciline gore zone alir; tid->t_sec ile
    bir kez np.lexsort; (uniq, starts, counts, codes) np.unique+searchsorted ile.

    homo None ise (bozuk calib / yok): pixel modu -> px=foot_x, py=foot_y,
    unit='px'. Birim, scale_anchor mevcut VE reconcile edilmedikce 'relative_m'
    olur (sadece scale_calibrated bayragindan ASLA 'm' uretilmez).
    """
    # DURUSTLUK KAPISI: bridge_gaps interpolasyon satirlari (interpolated=True,
    # conf=NaN) GERCEK tespit DEGIL -> metre/hiz toplamasina ASLA girmez (mesafe
    # yalani onleme). Satirlar parquet'te KALIR (ham parquet'i okuyan iz/occupancy
    # erisebilir); distance/speed hesabi bu kapidan once onlari duser.
    if "interpolated" in df.columns:
        _keep = ~df["interpolated"].fillna(False).astype(bool).to_numpy()
        if not _keep.all():
            _attrs = dict(getattr(df, "attrs", {}) or {})
            df = df.loc[_keep].reset_index(drop=True)
            try:
                df.attrs.update(_attrs)
            except Exception:
                pass

    n = len(df)
    tid = df["tid"].to_numpy(np.int64)
    t_sec = df["t_sec"].to_numpy(np.float64)
    foot_x = df["foot_x"].to_numpy(np.float64)
    foot_y = df["foot_y"].to_numpy(np.float64)
    box_h = df["box_h"].to_numpy(np.float64)
    conf = df["conf"].to_numpy(np.float64)

    # frame boyutu (meta -> fallback)
    meta = getattr(df, "attrs", {}).get("meta", {}) if hasattr(df, "attrs") else {}
    W = float(meta.get("width") or 1920)
    H = float(meta.get("height") or 1080)
    frame_wh = (W, H)

    disk_pitch = df["pitch_x"].to_numpy(np.float64) if "pitch_x" in df.columns else None
    disk_all_nan = disk_pitch is None or not np.isfinite(disk_pitch).any()

    per_zone_px: dict = {}
    pitch_dims = (34.0, 18.0)

    if homo is not None:
        try:
            pitch_dims = tuple(homo._dims_m())
        except Exception:
            pitch_dims = (34.0, 18.0)
        per_zone_px = dict((homo._qa or {}).get("per_zone", {}) or {})

        # olcek-capasi semasini TEK fonksiyondan oku (generator/eski-JSON farketmez)
        from pitch.scale_snap import scale_state
        st = scale_state(homo)
        force = st.quality != "relative"

        foot_xy = np.column_stack([foot_x, foot_y])
        # KRITIK: capali (relative-disi) calib ayak-pikselinden HER ZAMAN yeniden
        # projekte eder -> diske-pisirilmis v2 pitch_x'in sessiz sahte-metre
        # tuzagini kapatir. Sadece relative + diskte-baked ise diski tekrar kullan.
        if disk_all_nan or force:
            pm = homo.pixel_to_pitch(foot_xy)  # tek vektorize geciste projeksiyon
        else:
            pm = np.column_stack([disk_pitch,
                                  df["pitch_y"].to_numpy(np.float64)])
        px = pm[:, 0].astype(np.float64)
        py = pm[:, 1].astype(np.float64)
        in_pitch = np.asarray(homo.in_pitch(pm), dtype=bool)
        space = "pitch"
        # birim/kalite scale_state'ten: relative_m | m (approx +-N%) | m
        unit = st.unit
        scale_quality = st.quality
        unit_label = st.unit_label
        band_pct_v = st.band_pct
        # approx (snap) olcek SADECE sabit bir carpan; tespit-kalitesi guvenini
        # DEGISTIRMEMELI. w_conf icin pos_sigma'yi kanonik relative cerceveye
        # geri-olceklendir (s'e bol). Boylece v3(approx) mesafe == v2 mesafe * s
        # (raporlanan pos_sigma yine approx-m'de kalir).
        scale_factor_w = (st.scale_factor if (st.quality == "approx"
                          and st.scale_factor) else 1.0)
        jac = _local_jacobian_m_per_px(homo, foot_xy)
    else:
        # fallback: pixel modu, asla sessiz metre
        px = foot_x.copy()
        py = foot_y.copy()
        in_pitch = np.isfinite(px) & np.isfinite(py)
        space = "pixel"
        unit = "px"
        scale_quality = "relative"
        unit_label = "px"
        band_pct_v = None
        scale_factor_w = 1.0
        jac = np.ones(n, dtype=np.float64)  # px modunda Jacobian=1 (px/px)

    # in_frame: ayak noktasi kadraj icinde mi (off-frame kose maskesi icin)
    in_frame = (np.isfinite(foot_x) & np.isfinite(foot_y)
                & (foot_x >= 0) & (foot_x <= W) & (foot_y >= 0) & (foot_y <= H))
    if "bottom_cropped" in df.columns:
        in_frame &= ~df["bottom_cropped"].to_numpy(bool)

    # derinlik zone: foot_y tercilleri (buyuk foot_y = kameraya yakin = near=0)
    finite_fy = foot_y[np.isfinite(foot_y)]
    if finite_fy.size >= 3:
        q1, q2 = np.quantile(finite_fy, [1.0 / 3.0, 2.0 / 3.0])
    else:
        q1 = q2 = np.nanmedian(foot_y) if finite_fy.size else 0.0
    zone = np.full(n, 1, dtype=np.int8)         # mid
    zone[foot_y >= q2] = 0                        # near (alt ucte-bir)
    zone[foot_y < q1] = 2                         # far (ust ucte-bir)

    # pos_sigma_m = hypot(per_zone_reproj_px[zone]*J, FOOT_SIGMA_PX*J) = J*hypot(rep,3)
    zone_to_px = np.array([
        per_zone_px.get("near", np.nan),
        per_zone_px.get("mid", np.nan),
        per_zone_px.get("far", np.nan),
    ], dtype=np.float64)
    med_px = (homo._qa or {}).get("median_px", np.nan) if homo is not None else np.nan
    rep_px = zone_to_px[zone]
    rep_px = np.where(np.isfinite(rep_px), rep_px,
                      med_px if np.isfinite(med_px) else FOOT_SIGMA_PX)
    pos_sigma_m = jac * np.hypot(rep_px, FOOT_SIGMA_PX)

    # w_conf (Design 1), kadraj/saha disinda 0'a zorlanir, [0,1]
    c_term = np.clip(conf / CONF_REF, 0.0, 1.0)
    h_term = np.clip(box_h / BOX_H_REF, BOX_H_FLOOR, 1.0)
    # konum-belirsizligi terimi KANONIK relative cercevede (scale anchor guveni
    # degistirmez); approx modunda pos_sigma s ile olcekli oldugundan s'e bolunur
    pos_sigma_w = pos_sigma_m / scale_factor_w
    s_term = 1.0 / (1.0 + (pos_sigma_w / SIGMA_REF_M) ** 2)
    w_conf = c_term * h_term * s_term
    w_conf = np.where(in_frame & in_pitch, w_conf, 0.0)
    w_conf = np.clip(np.nan_to_num(w_conf, nan=0.0), 0.0, 1.0)

    # tek siralama: tid birincil, t_sec ikincil
    order = np.lexsort((t_sec, tid))
    tid = tid[order]; t_sec = t_sec[order]
    foot_x = foot_x[order]; foot_y = foot_y[order]
    box_h = box_h[order]; conf = conf[order]
    px = px[order]; py = py[order]
    in_pitch = in_pitch[order]; in_frame = in_frame[order]
    zone = zone[order]; pos_sigma_m = pos_sigma_m[order]; w_conf = w_conf[order]

    uniq, starts, counts = np.unique(tid, return_index=True, return_counts=True)
    codes = np.searchsorted(uniq, tid)

    return Prep(
        tid=tid, t_sec=t_sec, foot_x=foot_x, foot_y=foot_y, box_h=box_h,
        conf=conf, px=px, py=py, in_pitch=in_pitch, in_frame=in_frame,
        zone=zone, pos_sigma_m=pos_sigma_m, w_conf=w_conf,
        uniq=uniq, starts=starts, counts=counts, codes=codes,
        fps=float(fps), unit=unit, scale_quality=scale_quality,
        unit_label=unit_label, band_pct=band_pct_v,
        space=space, frame_wh=frame_wh,
        pitch_dims=tuple(pitch_dims), per_zone_px=per_zone_px,
    )


# ===========================================================================
# 3) kinematics_all: TEK kinematik gecis (hiz sprint'e yeniden beslenir)
# ===========================================================================
def _savgol_slice(arr: np.ndarray, win: int) -> np.ndarray:
    """Tek bir contiguous track dilimine Savitzky-Golay; kucukse MA fallback."""
    a = arr.astype(np.float64)
    n = a.size
    if n < 3 or win < 3:
        return a
    w = min(win, n if n % 2 == 1 else n - 1)
    if w % 2 == 0:
        w -= 1
    if w < 3:
        return a
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(a, w, min(2, w - 1))
    except Exception:
        k = np.ones(w) / w
        return np.convolve(a, k, mode="same")


def kinematics_all(prep: Prep, max_speed: float = MAX_SPEED_MPS,
                   max_gap: float = MAX_GAP_S,
                   smooth_win_s: float = SMOOTH_WIN_S) -> dict:
    """Tum track'ler icin tek geciste kinematik (Design 3 motoru).

    Once per-track Savitzky-Golay (tek dongu, n_tracks; contiguous numpy
    dilimleri uzerinde), sonra dx/dy/dt tum dizide np.diff ile; track-arasi
    farklar tid[1:]==tid[:-1] ile sifirlanir; teleport/gap kapisi
    (dt<=0 | dt>max_gap | speed>cap; cap='m' modunda max_speed, aksi halde
    veri-uyarlamali medyan*6); per-track mesafe np.bincount(codes[1:], seg_ok).

    Doner: dist (w_conf-agirlikli), dist_raw (agirliksiz), dropped_far_frac,
    far_frac, pos_sigma_med (hepsi uniq'e hizali) ve spd_row (per-satir hiz
    dizisi, sprint sayimi DOGRUDAN bunu tuketir -> ikinci kinematik gecis YOK).
    """
    n = prep.px.size
    nt = prep.uniq.size
    xs = prep.px.astype(np.float64).copy()
    ys = prep.py.astype(np.float64).copy()
    win = max(3, int(round(smooth_win_s * prep.fps)))

    # tek dongu: per-track contiguous dilim uzerinde smoothing
    for i in range(nt):
        s = int(prep.starts[i]); c = int(prep.counts[i])
        if c >= 3:
            xs[s:s + c] = _savgol_slice(xs[s:s + c], win)
            ys[s:s + c] = _savgol_slice(ys[s:s + c], win)

    # tum dizide diff; track-arasi sifirla
    dx = np.diff(xs); dy = np.diff(ys); dt = np.diff(prep.t_sec)
    same = prep.codes[1:] == prep.codes[:-1]
    seg = np.hypot(dx, dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = seg / dt

    # teleport/gap kapisi; cap KALITEYE bagli (birime degil)
    if getattr(prep, "scale_quality", "relative") == "exact":
        # fiziksel 11 m/s SADECE dogrulanmis (verified) metre icin
        cap = float(max_speed)
    else:
        # relative VE approx -> veri-uyarlamali medyan*6. Izotropik approx,
        # relative*s oldugundan medyan da s ile olceklenir -> ayni segmentler
        # hayatta kalir: v3(approx) mesafe == v2(relative) mesafe * s (cap artefakti yok).
        ok = same & (dt > 0) & np.isfinite(speed)
        med = np.median(speed[ok]) if ok.any() else np.nan
        cap = (med * 6.0) if (np.isfinite(med) and med > 0) else np.inf

    bad = ((~same) | (dt <= 0) | (dt > max_gap)
           | (~np.isfinite(speed)) | (speed > cap))
    seg_ok = np.where(bad, 0.0, seg)

    # per-satir hiz (sprint sayimi bunu tuketir); cross-track/bad -> 0
    spd_row = np.zeros(n, dtype=np.float64)
    spd_row[1:] = np.where(bad, 0.0, speed)

    cseg = prep.codes[1:]  # her segmentin (ileri-uc) track'i
    w_seg = prep.w_conf[1:]
    dist_raw = np.bincount(cseg, weights=seg_ok, minlength=nt)
    dist = np.bincount(cseg, weights=seg_ok * w_seg, minlength=nt)

    # uzak-uc dusurme orani (segment bazli)
    far_seg = (prep.zone[1:] == 2) & same
    far_seg_n = np.bincount(cseg, weights=far_seg.astype(np.float64), minlength=nt)
    far_seg_drop = np.bincount(cseg, weights=(far_seg & bad).astype(np.float64),
                               minlength=nt)
    with np.errstate(divide="ignore", invalid="ignore"):
        dropped_far_frac = np.where(far_seg_n > 0, far_seg_drop / far_seg_n, 0.0)

    # uzak-uc gozlem orani (per-track) + pos_sigma medyani (per-track)
    far_obs = np.bincount(prep.codes, weights=(prep.zone == 2).astype(np.float64),
                          minlength=nt)
    far_frac = far_obs / np.maximum(prep.counts, 1)
    pos_sigma_med = np.empty(nt, dtype=np.float64)
    for i in range(nt):
        s = int(prep.starts[i]); c = int(prep.counts[i])
        seg_sig = prep.pos_sigma_m[s:s + c]
        seg_sig = seg_sig[np.isfinite(seg_sig)]
        pos_sigma_med[i] = float(np.median(seg_sig)) if seg_sig.size else np.nan

    return dict(
        uniq=prep.uniq, dist=dist, dist_raw=dist_raw,
        dropped_far_frac=dropped_far_frac, far_frac=far_frac,
        pos_sigma_med=pos_sigma_med, spd_row=spd_row,
        cap=float(cap) if np.isfinite(cap) else float("inf"), unit=prep.unit,
    )


# ===========================================================================
# 4) parity_qc: 7v7=14 cift-sayi QC sinyali (Alperen #1)
# ===========================================================================
def parity_qc(df, expected: int = 14) -> dict:
    """Parite QC. 7v7 = 14 oyuncu (cift). Tek sayi -> kacan oyuncu var.

    Insight #1: kacanlar UZAK UCTE-BIRDE (kucuk + file arka-plani + dusuk
    kontrast). Uzak-uctaki yoklugu gercek sanma. frac_at_expected ~0.153
    (dogrulandi).
    """
    counts = df.groupby("frame").size().to_numpy()
    if counts.size == 0:
        return dict(frac_at_expected=0.0, median_count=0.0, odd_frame_frac=0.0,
                    note="no frames")
    frac_at = float(np.mean(counts == expected))
    median_count = float(np.median(counts))
    odd_frac = float(np.mean(counts % 2 == 1))
    note = (
        f"7v7={expected} cift sayi olmali; medyan {median_count:.0f} gozlemleniyor. "
        "Tek/eksik sayi => muhtemelen UZAK UCTE-BIR'de kacan oyuncu "
        "(kucuk+dusuk-kontrast); far-zone yoklugunu GERCEK kabul etme.")
    return dict(frac_at_expected=frac_at, median_count=median_count,
                odd_frame_frac=odd_frac, note=note)


# ===========================================================================
# coverage + heatmap
# ===========================================================================
def coverage_mask_pitch(homo, frame_wh, xedges, yedges) -> np.ndarray:
    """Bin MERKEZLERINI homo.pitch_to_pixel ile geri-projekte edip kadraj testi.

    Doner: (ny, nx) bool grid (histogram2d.T konvansiyonu). False binler =>
    'unknown' (kadraj disi / dusuk-kapsama). Sinir binleri konservatif
    (gri'ye kacar, uydurmaz). Alperen #2: off-frame kose yaklasik olarak
    yalnizca kadraj-sinir geri-projeksiyonuyla ele alinir.
    """
    W, Hh = float(frame_wh[0]), float(frame_wh[1])
    xc = 0.5 * (np.asarray(xedges)[:-1] + np.asarray(xedges)[1:])
    yc = 0.5 * (np.asarray(yedges)[:-1] + np.asarray(yedges)[1:])
    XX, YY = np.meshgrid(xc, yc)  # shape (ny, nx)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    # HAM piksele projekte et: kadraj-ici testi GERCEK yakalanan (raw) kare sinirina gore
    # olmali (undistorted-piksel kenar binlerinde kayar -> off-frame maske yanlis).
    px = homo.pitch_to_pixel(pts, distorted=True)
    u, v = px[:, 0], px[:, 1]
    inside = (np.isfinite(u) & np.isfinite(v)
              & (u >= 0) & (u <= W) & (v >= 0) & (v <= Hh))
    return inside.reshape(XX.shape)


def heatmap_grid_equal_area(prep: Prep, homo, frame_wh, bin_m: float = 1.0,
                            smooth: bool = True):
    """Saha-metresinde esit-alan heatmap (perspektif-onyargisi YOK).

    np.histogram2d ile saha metresinde binleme (insaen esit-alan), per-bin
    EXPOSURE (gorunur saniye) ile normalize; gaussian_filter MASKELEMEDEN ONCE
    (maske-bilincli); sonra kadraj-disi ve sifir-exposure hucreler NaN. Eski
    perspektif-onyargili pixel histogram2d + aspect='auto' yalanini degistirir.

    Doner: (grid (ny,nx, NaN'li), edges=(xedges,yedges)).
    """
    L, Wm = prep.pitch_dims
    nx = max(1, int(np.ceil(L / bin_m)))
    ny = max(1, int(np.ceil(Wm / bin_m)))
    xedges = np.linspace(0.0, float(L), nx + 1)
    yedges = np.linspace(0.0, float(Wm), ny + 1)

    use = prep.in_pitch & prep.in_frame & np.isfinite(prep.px) & np.isfinite(prep.py)
    x = prep.px[use]; y = prep.py[use]
    Hcnt, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])  # (nx,ny)
    presence_sec = (Hcnt.T) / max(prep.fps, 1e-9)             # (ny,nx) saniye

    cover = coverage_mask_pitch(homo, frame_wh, xedges, yedges)  # (ny,nx) bool
    total_dur = float(prep.t_sec.max() - prep.t_sec.min()) if prep.t_sec.size else 0.0

    grid = presence_sec.copy()
    if smooth:
        try:
            from scipy.ndimage import gaussian_filter
            sigma = max(0.5, 1.0 / bin_m)
            # maske-bilincli: gorunur bolgede normalize-edilmis konvolusyon
            w = cover.astype(np.float64)
            num = gaussian_filter(presence_sec * w, sigma)
            den = gaussian_filter(w, sigma)
            with np.errstate(divide="ignore", invalid="ignore"):
                grid = np.where(den > 1e-9, num / den, presence_sec)
        except Exception:
            grid = presence_sec.copy()

    # per-bin exposure ile normalize: gorunur=>total_dur, degil=>0 (=> NaN)
    exposure = np.where(cover, total_dur, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        occ = np.where(exposure > 0, grid / exposure, np.nan)

    # kadraj-disi ve sifir-exposure hucreler 'unknown' (NaN), 0 DEGIL
    occ[~cover] = np.nan
    occ = np.where(np.isfinite(occ) & (occ < 0), 0.0, occ)
    return occ, (xedges, yedges)


# ===========================================================================
# 5) generate_topdown_report: orkestrasyon
# ===========================================================================
def _qa_for_report(qa: dict | None) -> dict:
    pz = (qa or {}).get("per_zone", {}) or {}
    return dict(near=pz.get("near"), mid=pz.get("mid"), far=pz.get("far"),
                median_px=(qa or {}).get("median_px"),
                n_landmarks=(qa or {}).get("n_landmarks"))


def _band_by_zone(prep: Prep) -> dict:
    """Olculen pos_sigma'dan derive distance-band (near/mid/far medyani, m)."""
    out = {}
    for name, z in (("near", 0), ("mid", 1), ("far", 2)):
        m = prep.zone == z
        vals = prep.pos_sigma_m[m]
        vals = vals[np.isfinite(vals)]
        out[name] = round(float(np.median(vals)), 3) if vals.size else None
    return out


def _sibling_stitch_report(tracks_path) -> dict:
    """player-keyed/stitched parquet yaninda *_stitch_report.json'u bul (yoksa {})."""
    try:
        p = Path(tracks_path)
        stem = p.stem
        # track_robust ekleri (_bridged) + player-keyed/stitched eklerini soy -> base
        for suf in ("_player_bridged", "_bridged", "_player", "_stitched"):
            if stem.endswith(suf):
                stem = stem[:-len(suf)]
                break
        cand = p.with_name(f"{stem}_stitch_report.json")
        if cand.exists():
            return json.loads(cand.read_text())
    except Exception:  # noqa: BLE001
        pass
    return {}


def generate_topdown_report(tracks_path: str, calib_path: str,
                            out_dir: str) -> dict:
    """Tam top-down rapor; report_topdown.json yazar.

    load_tracks (stats_report yeniden kullanilir) -> homo=PitchHomography.load
    -> accept_calib_qa(homo._qa). Reddedilirse: sebebi logla, homo=None yap ve
    pixel/relative moduna dus (ASLA sessiz metre). prepare + kinematics_all +
    parity_qc + heatmap bir kez.
    """
    import stats_report as SR  # repo-koku yola eklendi
    from pitch.homography import PitchHomography

    df = SR.load_tracks(tracks_path)
    meta = df.attrs.get("meta", {})
    fps = float(meta.get("fps") or 25.0)

    # calib yukle + kapi
    homo = None
    calib_qa = None
    gate_ok, gate_reasons = False, ["calib not loaded"]
    try:
        loaded = PitchHomography.load(calib_path)
        calib_qa = loaded._qa
        gate_ok, gate_reasons = accept_calib_qa(calib_qa)
        if gate_ok:
            homo = loaded
        else:
            sys.stderr.write(
                f"[topdown] calib REJECTED ({calib_path}): {gate_reasons}; "
                "falling back to pixel/relative mode (NO meters)\n")
    except Exception as e:  # pragma: no cover
        gate_reasons = [f"calib load failed: {e}"]
        sys.stderr.write(f"[topdown] {gate_reasons[0]}\n")

    prep = prepare(df, homo, fps)
    kin = kinematics_all(prep)
    parity = parity_qc(df, expected=14)

    heatmap_info = None
    if homo is not None:
        try:
            grid, (xe, ye) = heatmap_grid_equal_area(prep, homo, prep.frame_wh,
                                                     bin_m=1.0)
            heatmap_info = dict(
                shape=list(grid.shape), bin_m=1.0,
                unknown_frac=float(np.mean(np.isnan(grid))),
                occupied_frac=float(np.mean(np.nan_to_num(grid) > 0)),
                note="equal-area pitch-meter bins; off-frame/zero-exposure=NaN")
        except Exception as e:  # pragma: no cover
            heatmap_info = dict(error=str(e))

    # per-player (tracklet) ciktilari
    players: dict[int, dict] = {}
    band_by_zone = _band_by_zone(prep)
    # sprint SADECE dogrulanmis-exact metrede; approx (snap) ASLA sprint vermez
    is_metric_exact = (getattr(prep, "scale_quality", "relative") == "exact")
    for i, tid in enumerate(prep.uniq):
        c = int(prep.counts[i])
        if c < MIN_TRACK_FRAMES:
            continue
        s = int(prep.starts[i])
        spd = kin["spd_row"][s:s + c]
        moving = spd[spd > 0]
        sp_avg = float(np.mean(moving)) if moving.size else 0.0
        sp_p99 = float(np.percentile(moving, 99)) if moving.size else 0.0
        dur = float(prep.t_sec[s + c - 1] - prep.t_sec[s]) if c >= 2 else 0.0
        far_f = float(kin["far_frac"][i])
        band = kin["pos_sigma_med"][i]
        rec = dict(
            n_obs=c,
            duration_s=round(dur, 1),
            distance=round(float(kin["dist"][i]), 2),
            distance_unweighted=round(float(kin["dist_raw"][i]), 2),
            distance_unit=prep.unit,
            distance_unit_label=prep.unit_label,
            distance_band_pm=(round(float(band), 3)
                              if np.isfinite(band) else None),
            far_third_frac=round(far_f, 3),
            dropped_far_frac=round(float(kin["dropped_far_frac"][i]), 3),
            speed_avg=round(sp_avg, 3),
            speed_max_p99=round(sp_p99, 3),
            speed_unit=prep.unit_label + "/s",
            speed_conf=("low" if far_f > FAR_CONF_FRAC else "ok"),
        )
        if is_metric_exact:
            # sprint sayisi YALNIZCA dogrulanmis-exact 'm' modunda anlamli
            rec["sprint_count"] = _count_sprints(spd, prep.t_sec[s:s + c])
        else:
            # relative_m: 18-20km/h esiginin metrik anlami yok -> persentiller
            rec["speed_pctl"] = dict(
                p50=round(float(np.percentile(moving, 50)), 3) if moving.size else 0.0,
                p90=round(float(np.percentile(moving, 90)), 3) if moving.size else 0.0,
                p99=round(sp_p99, 3))
        players[int(tid)] = rec

    # DURUSTLUK bloku (TUREVSEL, iddia degil)
    n_tracklets = int(prep.uniq.size)
    # KIMLIK: girdi STITCHED (player-keyed) mi? frag_tid kolonu stitcher sinyalidir.
    _stitched = ("frag_tid" in df.columns) or ("player_id" in df.columns)
    if _stitched:
        _sr = _sibling_stitch_report(tracks_path)
        _floor = _sr.get("clique_floor")
        _core = _sr.get("n_core_players")
        _partial = _sr.get("n_partial_players")
        identity_note = (
            f"per-OYUNCU (STITCHED): {n_tracklets} distinct kimlik"
            + (f" = {_core} CORE (tum-mac) + {_partial} kismi (yedek/dusuk-recall)"
               if _core is not None else "")
            + (f"; clique_floor={_floor}" if _floor is not None else "")
            + ". Asiri-kume HAYALET DEGIL: co-located/birlesecek cift yok -> GERCEK kisiler "
              "+ recall acigi (per-frame algilanan < mevcut). mesafe OYUNCU bazli; 14'e ZORLAMA yok.")
        non_goals = ["absolute meters/km-h/sprint counts", "ball/possession",
                     "team assignment", "PNG rendering"]
    else:
        identity_note = (
            f"per-tid == per-FRAGMENT ({n_tracklets} tids / ~14 players); "
            "distance UNDERCOUNTS until tracklet stitcher "
            "(detect/track_stitch.py) is wired in -- dominant caveat")
        non_goals = ["tracklet/identity stitching",
                     "absolute meters/km-h/sprint counts", "ball/possession",
                     "team assignment", "PNG rendering"]
    # OLCEK belirsizligi: capa-bilincli (hardcoded 'scale_anchor=null' YOK)
    from pitch.scale_snap import scale_state as _scale_state
    _st = _scale_state(homo)
    if _st.quality == "approx":
        _sa = getattr(homo, "scale_anchor", {}) or {}
        _asserted = (_sa.get("asserted_dims_LW")
                     or [_sa.get("scaled_dims_m", {}).get("L"),
                         _sa.get("scaled_dims_m", {}).get("W")])
        scale_unc = (
            f"standart-boyut snap ({_st.catalog_label} katalog, izotropik "
            f"s={_st.scale_factor}); asserted {_asserted} m, +-{int(_st.band_pct or 10)}% "
            "band; OLCULMEDI/YAYINLANMADI -> 'm (approx)', sprint YOK. "
            "Bkz docs/SCALE_NOTES.md")
    elif _st.quality == "exact":
        scale_unc = "verified scale -> m (~%1-2)"
    else:
        scale_unc = (
            "scale_anchor=null -> relative_m (metre DEGIL); pitch/scale_snap.py "
            "make_v3 ile approx-m uretilebilir")
    honesty = dict(
        per_zone=_qa_for_report(calib_qa),
        scale_uncertainty=scale_unc,
        distance_band=dict(
            measured_pos_sigma_m=band_by_zone,
            note="distance band = measured pos_sigma (near small, far larger)"),
        acceleration="NOT reported (frame-to-frame accel is noise-dominated)",
        possession="None (no ball layer)",
        identity=identity_note,
        coverage=(
            "off-frame near-bottom corner + far third masked as unknown (NaN), "
            "not zero; corner-post wedge approximated by frame-bounds "
            "back-projection only (conservative: greys too much, never fabricates)"),
        calib_gate=dict(ok=bool(gate_ok), reasons=gate_reasons),
        non_goals=non_goals,
    )

    report = dict(
        camera_id=meta.get("camera_id"),
        source_clip=meta.get("source_clip"),
        tracks_path=str(tracks_path),
        calib_path=str(calib_path),
        coordinate=dict(
            unit=prep.unit, unit_label=prep.unit_label,
            scale_quality=prep.scale_quality, band_pct=prep.band_pct,
            space=prep.space,
            pitch_dims_m=dict(L=prep.pitch_dims[0], W=prep.pitch_dims[1]),
            scale_anchor=(getattr(homo, "scale_anchor", None)
                          if homo is not None else None)),
        fps=fps,
        n_tracklets=n_tracklets,
        parity_qc=parity,
        heatmap=heatmap_info,
        players=players,
        possession_pct=None,
        honesty=honesty,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "report_topdown.json", "w") as f:
        json.dump(report, f, indent=2, default=_json_default)
    return report


def _count_sprints(spd: np.ndarray, t: np.ndarray,
                   thr_mps: float = 5.5, min_s: float = 0.6) -> int:
    """Suren sprint epizotlari (yalnizca metrik 'm' modunda cagrilir)."""
    above = spd >= thr_mps
    n = spd.size
    cnt = 0; i = 1
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            dur = t[min(j, n - 1)] - t[i - 1]
            if dur >= min_s:
                cnt += 1
            i = j
        else:
            i += 1
    return int(cnt)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# ===========================================================================
# selftest CLI
# ===========================================================================
def _selftest() -> int:
    real = "raw/tracks_cankaya_cam2_clip2400.parquet"
    good = "calib/cankaya_cam2_v2.json"
    bad = "calib/cankaya_cam2.json"
    from pitch.homography import PitchHomography
    import pandas as pd

    print("== accept_calib_qa ==")
    ok_b, why_b = accept_calib_qa(PitchHomography.load(bad)._qa)
    ok_g, why_g = accept_calib_qa(PitchHomography.load(good)._qa)
    print("  bad ->", ok_b, why_b)
    print("  good->", ok_g, why_g)

    if Path(real).exists() and Path(good).exists():
        df = pd.read_parquet(real)
        homo = PitchHomography.load(good)
        prep = prepare(df, homo, fps=25.0)
        print("== prepare ==")
        print("  rows", prep.px.size, "finite",
              round(float(np.isfinite(prep.px).mean()), 4),
              "in_pitch", round(float(prep.in_pitch.mean()), 4),
              "unit", prep.unit)
        print("  band_by_zone", _band_by_zone(prep))
        k = kinematics_all(prep)
        print("== kinematics ==")
        print("  spd_row len", k["spd_row"].size, "== rows", prep.px.size)
        print("  dist[0:3]", np.round(k["dist"][:3], 2),
              "dist_raw[0:3]", np.round(k["dist_raw"][:3], 2))
        print("== parity ==", parity_qc(df))
        rep = generate_topdown_report(real, good, "stats_out/_selftest")
        print("== report ==")
        print("  unit", rep["coordinate"]["unit"],
              "n_players", len(rep["players"]),
              "possession", rep["possession_pct"])
        print("  honesty.acceleration", rep["honesty"]["acceleration"][:24])
        repb = generate_topdown_report(real, bad, "stats_out/_selftest_bad")
        print("  bad-calib unit", repb["coordinate"]["unit"],
              "gate.ok", repb["honesty"]["calib_gate"]["ok"])
    else:
        print("(real artifacts missing; ran calib-gate only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
