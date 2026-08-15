#!/usr/bin/env python3
"""Halısaha pozisyon-istatistik raporu (per-oyuncu).

Girdi: export_tracks.py'nin urettigi per-frame track kayitlari (Parquet/CSV).
       Sema (track_record_schema): bir satir = bir (tid, frame) gozlemi.
       Sutunlar: tid, frame, t_sec, foot_x, foot_y, box_h, box_w, conf,
                 bottom_cropped, pitch_x, pitch_y, in_pitch.
       Dosya-seviyesi metadata: camera_id, source_clip, fps, width, height,
                 calib_path, scale_calibrated, schema_version.

Cikti: report.json (per-oyuncu metrikler + DURUST hata-sinir notlari) +
       opsiyonel matplotlib heatmap PNG'leri + (varsa) qa_overlay kopyasi.

Tasarim ilkeleri (INTERFACE SPEC ile birebir):
- MEASURE THE PITCH: scale_calibrated=false ise mesafe/hiz METRE DEGIL,
  RELATIVE (normalize) olarak verilir. Asla uydurma km/saat uretme.
- Per-zone / horizon belirsizligi: tek bir QA sayisi yalandir; metrikler
  uzaklik-bandina gore guven etiketi tasir.
- Sadece numpy/scipy/pandas/cv2/matplotlib (BSD/MIT). GPU YOK.

HONEST hata siniri ozeti (bu modulun urettigi her metrik icin):
- toplam mesafe       : ~%2-5 (smoothing + ID-swap teleport gate sonrasi).
- hiz (max/ort)       : ortalama ~%5; ANLIK max gurultuye duyarli -> p95 raporlanir.
- ivme/yavaslama      : ASLA raporlanmaz (tek-kamera + pixel gurultusu guvenilmez).
- heatmap             : pozisyon dagilimi saglam; bin kenarlarinda quantization.
- bolge hakimiyeti    : takim atamasi YOK -> per-tid; takim-duzeyi DEFERRED.
- sahiplik %          : top katmani yok -> None (durust).
- boy tahmini         : DENEYSEL, +-%30; vertical~=ground-plane yerel olcek varsayimi.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --- sabitler / varsayilan parametreler -------------------------------------
SPRINT_THR_MPS = 5.5          # ~20 km/h; halisaha sprint esigi (parametrik)
MIN_SPRINT_S = 0.6            # bir sprint sayilmasi icin min sureli kosu
MAX_SPEED_MPS = 11.0          # bunun ustu = ID-swap/teleport; mesafeye katilmaz
MAX_GAP_S = 1.0              # bu kadar bosluktan sonra segment kopuk sayilir
MIN_TRACK_FRAMES = 12        # bir tid'in raporlanmasi icin min gozlem
SMOOTH_WIN_S = 0.5           # trajektori smoothing penceresi (saniye)

# pixel-domain (kalibrasyonsuz) icin bir "relative" sprint esigi: medyan
# kare-basi-yer-degistirmenin katı. Metre olmadigi icin sadece goreli.
REL_SPRINT_FACTOR = 3.0


# ============================================================================
# Yukleme + metadata
# ============================================================================
def _read_parquet_meta(path: str) -> dict:
    """Parquet schema key-value metadata'sini cozumle (defensive)."""
    try:
        import pyarrow.parquet as pq
    except Exception:
        return {}
    try:
        schema = pq.read_schema(path)
    except Exception:
        return {}
    md = schema.metadata or {}
    # 1) tek JSON blob aranir (export_tracks bunu yazabilir)
    for key in (b"halisaha", b"halisaha_meta", b"meta"):
        if key in md:
            try:
                return json.loads(md[key])
            except Exception:
                pass
    # 2) tek tek byte-anahtarlardan topla
    out: dict = {}
    for k, v in md.items():
        try:
            ks = k.decode()
            vs = v.decode()
        except Exception:
            continue
        if ks.lower().startswith("arrow") or ks.lower().startswith("pandas"):
            continue
        out[ks] = vs
    return out


def _read_sidecar_meta(path: str) -> dict:
    """CSV icin yan-dosya JSON metadata (cesitli aday yollar)."""
    p = Path(path)
    for cand in (p.with_suffix(p.suffix + ".meta.json"),
                 p.with_suffix(".meta.json"),
                 p.with_suffix(".json")):
        if cand.exists():
            try:
                return json.loads(cand.read_text())
            except Exception:
                pass
    return {}


def _coerce_meta(raw: dict) -> dict:
    """Metadata alanlarini dogru tiplere cevir (bytes/str -> python)."""
    def _b(v, default=False):
        if isinstance(v, bool):
            return v
        if v is None:
            return default
        return str(v).strip().lower() in ("1", "true", "yes")

    def _f(v, default=None):
        try:
            return float(v)
        except Exception:
            return default

    def _i(v, default=None):
        try:
            return int(float(v))
        except Exception:
            return default

    m = dict(raw or {})
    return {
        "camera_id": m.get("camera_id"),
        "source_clip": m.get("source_clip"),
        "fps": _f(m.get("fps"), None),
        "width": _i(m.get("width"), None),
        "height": _i(m.get("height"), None),
        "calib_path": m.get("calib_path") or None,
        "scale_calibrated": _b(m.get("scale_calibrated"), False),
        "schema_version": _i(m.get("schema_version"), None),
    }


def load_tracks(path: str) -> "pd.DataFrame":
    """Per-frame track kayitlarini yukle; metadata'yi df.attrs['meta'] altinda tasi.

    Parquet birincil, CSV (metadata yan-dosyadan) fallback. Eksik opsiyonel
    sutunlar (box_w, bottom_cropped, pitch_*, in_pitch) makul varsayilanlarla
    doldurulur ki rapor calib-oncesi dosyalarda da uretilebilsin.
    """
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(path)
        meta = _read_parquet_meta(path)
        if not meta:
            meta = _read_sidecar_meta(path)
    else:
        df = pd.read_csv(path)
        meta = _read_sidecar_meta(path)

    # zorunlu sutun kontrolu
    required = ["tid", "frame", "t_sec", "foot_x", "foot_y"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"track kaydinda eksik zorunlu sutun(lar): {missing}")

    # opsiyonel sutunlari guvenli doldur
    if "box_h" not in df.columns:
        df["box_h"] = np.float32(np.nan)
    if "box_w" not in df.columns:
        df["box_w"] = np.float32(np.nan)
    if "conf" not in df.columns:
        df["conf"] = np.float32(np.nan)
    if "bottom_cropped" not in df.columns:
        df["bottom_cropped"] = False
    if "pitch_x" not in df.columns:
        df["pitch_x"] = np.float32(np.nan)
    if "pitch_y" not in df.columns:
        df["pitch_y"] = np.float32(np.nan)
    if "in_pitch" not in df.columns:
        # pitch yoksa FP gate uygulanamaz -> tumu kabul (gurultuyu ham birak)
        df["in_pitch"] = df[["pitch_x", "pitch_y"]].notna().all(axis=1)

    # fps metadata'da yoksa t_sec/frame'den tahmin et
    cm = _coerce_meta(meta)
    if cm["fps"] is None:
        cm["fps"] = _infer_fps(df)
    # pitch koordinati gercekten dolu mu?
    cm["has_pitch"] = bool(df[["pitch_x", "pitch_y"]].notna().any().any())

    df = df.sort_values(["tid", "frame"]).reset_index(drop=True)
    df.attrs["meta"] = cm
    return df


def _infer_fps(df: "pd.DataFrame") -> float:
    """t_sec ve frame'den fps tahmini (metadata fps yoksa)."""
    try:
        g = df.groupby("tid")
        dt = (g["t_sec"].diff() / g["frame"].diff()).replace([np.inf, -np.inf], np.nan)
        med = float(np.nanmedian(dt.values))
        if med > 0:
            return 1.0 / med
    except Exception:
        pass
    return 25.0


# ============================================================================
# Domain secimi: metre (kalibre+olcekli) / relative-meters / pixel
# ============================================================================
def _resolve_domain(df: "pd.DataFrame") -> dict:
    """Hangi koordinat alaninda calisacagimizi ve birim etiketini belirle.

    -> {'space':'pitch'|'pixel', 'scale_calibrated':bool, 'unit':str,
        'xcol':..., 'ycol':...}
    """
    meta = df.attrs.get("meta", {})
    has_pitch = meta.get("has_pitch", False)
    scale_cal = bool(meta.get("scale_calibrated", False))
    if has_pitch:
        return dict(space="pitch", scale_calibrated=scale_cal,
                    unit=("m" if scale_cal else "relative_m"),
                    xcol="pitch_x", ycol="pitch_y")
    return dict(space="pixel", scale_calibrated=False, unit="px",
                xcol="foot_x", ycol="foot_y")


# ============================================================================
# Kinematik cekirdek (smoothing + teleport gate)
# ============================================================================
def _savgol_or_ma(arr: np.ndarray, win: int) -> np.ndarray:
    """Savitzky-Golay smoothing; pencere kucukse hareketli ortalama fallback."""
    n = len(arr)
    if n < 3 or win < 3:
        return arr.astype(float)
    win = min(win, n if n % 2 == 1 else n - 1)
    if win % 2 == 0:
        win -= 1
    if win < 3:
        return arr.astype(float)
    try:
        from scipy.signal import savgol_filter
        poly = min(2, win - 1)
        return savgol_filter(arr.astype(float), win, poly)
    except Exception:
        k = np.ones(win) / win
        return np.convolve(arr.astype(float), k, mode="same")


def _track_kinematics(track_df: "pd.DataFrame", domain: dict, fps: float) -> dict:
    """Tek bir tid icin smoothed pozisyon, hiz, mesafe; teleport gate uygulanir.

    Donen dict: t, x, y (smoothed), speed (m/s veya px/s veya relative),
    seg (kabul edilen segment uzunluklari), dist (toplam), n_valid,
    clipped_frac (teleport/gap nedeniyle atilan oran).
    """
    xcol, ycol = domain["xcol"], domain["ycol"]
    sub = track_df[["t_sec", xcol, ycol, "in_pitch"]].copy()
    # gecersiz/FP noktalari ele
    sub = sub[np.isfinite(sub[xcol]) & np.isfinite(sub[ycol])]
    if domain["space"] == "pitch":
        sub = sub[sub["in_pitch"].astype(bool)]
    sub = sub.sort_values("t_sec")
    t = sub["t_sec"].to_numpy(float)
    x = sub[xcol].to_numpy(float)
    y = sub[ycol].to_numpy(float)
    n = len(t)
    if n < 2:
        return dict(t=t, x=x, y=y, speed=np.zeros(n), seg=np.zeros(0),
                    dist=0.0, n_valid=n, clipped_frac=0.0, n_seg_used=0)

    win = max(3, int(round(SMOOTH_WIN_S * fps)))
    xs = _savgol_or_ma(x, win)
    ys = _savgol_or_ma(y, win)

    dt = np.diff(t)
    dt[dt <= 0] = np.nan
    seg = np.hypot(np.diff(xs), np.diff(ys))
    inst_speed = seg / dt  # birim/sn

    # teleport/gap gate: buyuk bosluk VEYA fiziksel-olmayan hiz -> at
    if domain["unit"] == "m":
        speed_cap = MAX_SPEED_MPS
    else:
        # pixel/relative: cap'i veri-uyarlamali yap (medyan hizin katı)
        med_sp = np.nanmedian(inst_speed[np.isfinite(inst_speed)]) if np.isfinite(inst_speed).any() else np.nan
        speed_cap = (med_sp * 6.0) if (med_sp and med_sp > 0) else np.inf
    bad = (~np.isfinite(inst_speed)) | (dt > MAX_GAP_S) | (inst_speed > speed_cap)
    n_bad = int(bad.sum())
    seg_used = seg.copy()
    seg_used[bad] = 0.0
    dist = float(np.nansum(seg_used))
    clipped_frac = n_bad / max(len(seg), 1)

    speed = np.zeros(n)
    good = ~bad
    speed[1:][good] = inst_speed[good]
    return dict(t=t, x=xs, y=ys, speed=speed, seg=seg_used, dist=dist,
                n_valid=n, clipped_frac=clipped_frac,
                n_seg_used=int((~bad).sum()))


# ============================================================================
# INTERFACE SPEC fonksiyonlari
# ============================================================================
def compute_distance_m(track_df: "pd.DataFrame") -> float:
    """Tek oyuncunun toplam kosu mesafesi.

    DURUST: scale_calibrated=true ise METRE; degilse RELATIVE (saha-birimi
    veya pixel). Smoothing + teleport gate uygulanir (~%2-5 hata). Birimi
    bilmek icin domain'i load_tracks meta'sindan kontrol edin; tek-float
    imzasi spec geregi korunur.
    """
    meta = track_df.attrs.get("meta")
    if meta is None and hasattr(track_df, "_parent_meta"):
        meta = track_df._parent_meta
    domain = _resolve_domain_from_meta(track_df, meta)
    fps = (meta or {}).get("fps") or _infer_fps(track_df)
    k = _track_kinematics(track_df, domain, fps)
    return float(k["dist"])


def _resolve_domain_from_meta(track_df, meta) -> dict:
    """compute_distance_m gibi tekil-df cagrilari icin domain cozumu."""
    has_pitch = False
    if {"pitch_x", "pitch_y"}.issubset(track_df.columns):
        has_pitch = bool(track_df[["pitch_x", "pitch_y"]].notna().any().any())
    scale_cal = bool((meta or {}).get("scale_calibrated", False))
    if has_pitch:
        return dict(space="pitch", scale_calibrated=scale_cal,
                    unit=("m" if scale_cal else "relative_m"),
                    xcol="pitch_x", ycol="pitch_y")
    return dict(space="pixel", scale_calibrated=False, unit="px",
                xcol="foot_x", ycol="foot_y")


def sprint_count(track_df: "pd.DataFrame", speed_thr_mps: float = SPRINT_THR_MPS) -> int:
    """Esigi gecen suren kosu epizotlarinin sayisi.

    DURUST: anlamli bir sprint sayisi icin scale_calibrated=true gerekir.
    Kalibrasyonsuz (pixel/relative) modda esik metre/sn cinsinden anlam
    tasimaz; bu durumda veri-uyarlamali RELATIVE esik kullanilir ve sonuc
    yalnizca GORELI yorumlanmalidir.
    """
    meta = track_df.attrs.get("meta")
    domain = _resolve_domain_from_meta(track_df, meta)
    fps = (meta or {}).get("fps") or _infer_fps(track_df)
    k = _track_kinematics(track_df, domain, fps)
    speed = k["speed"]
    t = k["t"]
    if len(speed) < 2:
        return 0

    if domain["unit"] == "m":
        thr = speed_thr_mps
    else:
        moving = speed[speed > 0]
        med = np.median(moving) if moving.size else 0.0
        thr = med * REL_SPRINT_FACTOR if med > 0 else np.inf

    above = speed >= thr
    # suren epizotlari say: run-length, min sure MIN_SPRINT_S
    count = 0
    i = 1
    n = len(speed)
    while i < n:
        if above[i]:
            j = i
            while j < n and above[j]:
                j += 1
            dur = t[min(j, n - 1)] - t[i - 1]
            if dur >= MIN_SPRINT_S:
                count += 1
            i = j
        else:
            i += 1
    return int(count)


def heatmap_grid(df: "pd.DataFrame", bin_m: float = 1.0) -> np.ndarray:
    """2D pozisyon histogrami (saha-koord veya pixel).

    numpy.histogram2d ile. Saha alanindaysa bin kenarlari metre cinsinden
    (kalibre degilse relative-metre). Pixel modunda bin_m pixel olarak
    yorumlanir (cozunurluk olcekli). Donen: (ny, nx) sayim matrisi.
    """
    domain = _resolve_domain(df)
    xcol, ycol = domain["xcol"], domain["ycol"]
    sub = df[np.isfinite(df[xcol]) & np.isfinite(df[ycol])]
    if domain["space"] == "pitch" and "in_pitch" in sub.columns:
        sub = sub[sub["in_pitch"].astype(bool)]
    x = sub[xcol].to_numpy(float)
    y = sub[ycol].to_numpy(float)
    if x.size == 0:
        return np.zeros((1, 1), dtype=float)

    ext = _domain_extent(df, domain)
    x0, x1, y0, y1 = ext
    nx = max(1, int(np.ceil((x1 - x0) / bin_m)))
    ny = max(1, int(np.ceil((y1 - y0) / bin_m)))
    xedges = np.linspace(x0, x1, nx + 1)
    yedges = np.linspace(y0, y1, ny + 1)
    Hh, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
    # (nx,ny)->(ny,nx) gorsel konvansiyon (satir=Y/width, sutun=X/length)
    return Hh.T


def _domain_extent(df, domain) -> tuple:
    """Heatmap/territory icin (x0,x1,y0,y1) sinirlari.

    Saha alaninda: kalibrasyon metadata'sindaki L,W (varsa) yoksa veri menzili.
    Pixel: goruntu boyutu (varsa) yoksa veri menzili.
    """
    meta = df.attrs.get("meta", {})
    xcol, ycol = domain["xcol"], domain["ycol"]
    sub = df[np.isfinite(df[xcol]) & np.isfinite(df[ycol])]
    if sub.empty:
        return (0.0, 1.0, 0.0, 1.0)
    if domain["space"] == "pitch":
        LW = _pitch_dims_from_calib(meta.get("calib_path"))
        if LW is not None:
            L, W = LW
            return (0.0, float(L), 0.0, float(W))
        # veri menzili (biraz pad)
        return (float(sub[xcol].min()), float(sub[xcol].max()),
                float(sub[ycol].min()), float(sub[ycol].max()))
    w = meta.get("width")
    h = meta.get("height")
    if w and h:
        return (0.0, float(w), 0.0, float(h))
    return (float(sub[xcol].min()), float(sub[xcol].max()),
            float(sub[ycol].min()), float(sub[ycol].max()))


def _pitch_dims_from_calib(calib_path):
    """calib JSON'undan (L,W) oku; yoksa None."""
    if not calib_path:
        return None
    p = Path(calib_path)
    if not p.exists():
        return None
    try:
        j = json.loads(p.read_text())
        d = j.get("pitch_dims_m") or {}
        if "L" in d and "W" in d:
            return float(d["L"]), float(d["W"])
    except Exception:
        return None
    return None


def territory_share(df: "pd.DataFrame") -> dict:
    """Saha hakimiyeti dagilimi.

    DURUST: oyuncu->takim atamasi YOK (renk/kume katmani henuz yok), bu yuzden
    TAKIM-duzeyi hakimiyet DEFERRED. Burada per-tid olarak sahanin uzunluk
    ekseni (X) uzerindeki uctelik (savunma/orta/hucum) zaman paylasimini
    ve genel uctelik dolulugunu doneriz. 'note' alaninda kisit acikca yazilir.
    """
    domain = _resolve_domain(df)
    xcol, ycol = domain["xcol"], domain["ycol"]
    ext = _domain_extent(df, domain)
    x0, x1 = ext[0], ext[1]
    span = (x1 - x0) or 1.0

    def thirds(xvals):
        f = (np.asarray(xvals, float) - x0) / span
        f = np.clip(f, 0, 0.999999)
        idx = (f * 3).astype(int)
        c = np.bincount(idx, minlength=3).astype(float)
        s = c.sum() or 1.0
        return (c / s).round(3).tolist()  # [def, mid, att] paylari

    per_tid = {}
    for tid, g in df.groupby("tid"):
        sub = g[np.isfinite(g[xcol]) & np.isfinite(g[ycol])]
        if domain["space"] == "pitch" and "in_pitch" in sub.columns:
            sub = sub[sub["in_pitch"].astype(bool)]
        if len(sub) < MIN_TRACK_FRAMES:
            continue
        per_tid[int(tid)] = dict(thirds_share=thirds(sub[xcol].values),
                                 n=int(len(sub)))

    allx = df[np.isfinite(df[xcol])][xcol].values
    overall = thirds(allx) if allx.size else [0, 0, 0]
    return dict(
        axis="length_thirds[def,mid,att]",
        space=domain["space"],
        overall_thirds_share=overall,
        per_tid=per_tid,
        note=("Takim atamasi yok -> takim-duzeyi hakimiyet DEFERRED; "
              "yon (savunma/hucum) operator orientation_hint'e bagli ve "
              "takim kimligi olmadan mutlak degildir."),
    )


def possession_pct(df: "pd.DataFrame", ball_df=None) -> "dict | None":
    """Top sahipligi yuzdesi.

    DURUST: top tespiti/takip katmani henuz yok. ball_df verilmediyse None
    doneriz (uydurma sahiplik uretmeyiz). ball_df ileride eklenince en-yakin
    oyuncu atamasiyla doldurulacak.
    """
    if ball_df is None:
        return None
    # ileri-uyumlu iskelet: top noktasina en yakin tid'e kare-bazli atama
    domain = _resolve_domain(df)
    xcol, ycol = domain["xcol"], domain["ycol"]
    bx = ball_df.get("pitch_x", ball_df.get("foot_x"))
    by = ball_df.get("pitch_y", ball_df.get("foot_y"))
    if bx is None or by is None:
        return None
    counts: dict = {}
    by_frame = {int(r.frame): (float(r_x), float(r_y))
                for r, r_x, r_y in zip(ball_df.itertuples(), bx, by)
                if np.isfinite(r_x) and np.isfinite(r_y)}
    for fr, (bxx, byy) in by_frame.items():
        cand = df[df["frame"] == fr]
        if cand.empty:
            continue
        d = np.hypot(cand[xcol].values - bxx, cand[ycol].values - byy)
        if not np.isfinite(d).any():
            continue
        owner = int(cand.iloc[int(np.nanargmin(d))]["tid"])
        counts[owner] = counts.get(owner, 0) + 1
    total = sum(counts.values()) or 1
    return {tid: round(100.0 * c / total, 1) for tid, c in counts.items()}


# ============================================================================
# Boy tahmini (DENEYSEL)
# ============================================================================
class _CalibProjector:
    """calib JSON'undan pixel->pitch projeksiyonu (cv2; sadece boy yerel-olcek icin).

    pitch/homography.py'ye bagimli OLMADAN calisir ki bu modul tek basina
    test edilebilsin. Mevcutsa o sinif tercih edilebilir; burada minimal+durust.
    """

    def __init__(self, calib_path: str):
        import cv2  # noqa: F401 (BSD)
        self.cv2 = cv2
        j = json.loads(Path(calib_path).read_text())
        self.K = np.array(j.get("K"), float) if j.get("K") is not None else None
        self.dist = np.array(j.get("dist"), float) if j.get("dist") is not None else None
        self.H = np.array(j["H_img2pitch"], float)
        self.undistort_applied = bool(j.get("undistort_applied", self.K is not None))

    def pixel_to_pitch(self, pts_px: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_px, float).reshape(-1, 1, 2)
        if self.undistort_applied and self.K is not None and self.dist is not None:
            und = self.cv2.undistortPoints(pts, self.K, self.dist, P=self.K)
        else:
            und = pts
        u = und.reshape(-1, 2)
        ones = np.ones((u.shape[0], 1))
        hom = np.hstack([u, ones]) @ self.H.T
        w = hom[:, 2:3]
        w[w == 0] = np.nan
        return hom[:, :2] / w


def _estimate_heights(df, domain, calib_path) -> dict:
    """Per-tid boy tahmini (metre). DENEYSEL, +-%30.

    Yontem: ayak noktasinda yerel ground-plane olcegi (m/px) x box_h(px).
    Foreshortening (vertical != ground) duzeltilmez -> sistematik sapma var.
    Yalnizca scale_calibrated ve calib mevcutsa anlamli.
    """
    out = dict(available=False, note="", per_tid={})
    if domain["space"] != "pitch" or not domain["scale_calibrated"]:
        out["note"] = ("Boy tahmini icin olcekli kalibrasyon (scale_calibrated) "
                       "gerekir; mevcut degil -> atlandi.")
        return out
    if not calib_path or not Path(calib_path).exists():
        out["note"] = "calib JSON yok -> boy tahmini atlandi."
        return out
    try:
        proj = _CalibProjector(calib_path)
    except Exception as e:
        out["note"] = f"calib yuklenemedi ({e}) -> boy atlandi."
        return out

    per = {}
    for tid, g in df.groupby("tid"):
        sub = g[np.isfinite(g["foot_x"]) & np.isfinite(g["foot_y"])
                & np.isfinite(g["box_h"])]
        if "bottom_cropped" in sub.columns:
            sub = sub[~sub["bottom_cropped"].astype(bool)]
        if len(sub) < MIN_TRACK_FRAMES:
            continue
        fx = sub["foot_x"].to_numpy(float)
        fy = sub["foot_y"].to_numpy(float)
        bh = sub["box_h"].to_numpy(float)
        foot = np.stack([fx, fy], 1)
        foot_up = np.stack([fx, fy - 1.0], 1)  # 1px yukari (yerel dusey olcek)
        pf = proj.pixel_to_pitch(foot)
        pu = proj.pixel_to_pitch(foot_up)
        m_per_px = np.hypot(*(pf - pu).T)  # her gozlem icin m/px
        h_m = bh * m_per_px
        h_m = h_m[np.isfinite(h_m)]
        if h_m.size == 0:
            continue
        per[int(tid)] = dict(
            height_m_est=round(float(np.median(h_m)), 2),
            mad_m=round(float(np.median(np.abs(h_m - np.median(h_m)))), 2),
            n=int(h_m.size),
        )
    out["available"] = True
    out["per_tid"] = per
    out["note"] = ("DENEYSEL +-%30: vertical~=ground-plane yerel olcek varsayimi; "
                   "foreshortening duzeltilmez, uzak/grazing acida sapar. "
                   "Dogrulanmamis -> sadece kaba siralama icin.")
    return out


# ============================================================================
# Rapor uretimi
# ============================================================================
def _per_player_report(df, domain, fps) -> dict:
    """Her tid icin mesafe/hiz/sprint metrikleri (durust etiketli)."""
    players = {}
    for tid, g in df.groupby("tid"):
        n_obs = int(len(g))
        if n_obs < MIN_TRACK_FRAMES:
            continue
        k = _track_kinematics(g, domain, fps)
        if k["n_valid"] < 2:
            continue
        speed = k["speed"]
        moving = speed[speed > 0]
        sp_max = float(np.percentile(moving, 99)) if moving.size else 0.0
        sp_max_raw = float(np.max(speed)) if speed.size else 0.0
        sp_avg = float(np.mean(moving)) if moving.size else 0.0
        dur = float(k["t"][-1] - k["t"][0]) if k["t"].size >= 2 else 0.0
        players[int(tid)] = dict(
            n_obs=n_obs,
            duration_s=round(dur, 1),
            n_valid_points=int(k["n_valid"]),
            distance=round(k["dist"], 2),
            distance_unit=domain["unit"],
            speed_avg=round(sp_avg, 3),
            speed_max_p99=round(sp_max, 3),
            speed_max_raw=round(sp_max_raw, 3),
            speed_unit=(domain["unit"] + "/s"),
            sprint_count=sprint_count(_attach_meta(g, df), SPRINT_THR_MPS),
            teleport_gate_clipped_frac=round(k["clipped_frac"], 3),
        )
    return players


def _attach_meta(g, parent):
    """Grup df'ine parent meta'yi tasi (sprint_count tekil cagrisi icin)."""
    try:
        g.attrs["meta"] = parent.attrs.get("meta", {})
    except Exception:
        pass
    return g


def generate_report(tracks_path: str, calib_path: str | None, out_dir: str) -> dict:
    """Tam rapor: report.json + heatmap PNG'leri + (varsa) qa bilgisi.

    Her metrik scale_calibrated + uzaklik-bandi belirsizligi tasir. Olcek
    kalibre degilse mesafe/hiz RELATIVE; asla metre/km diye iddia edilmez.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_tracks(tracks_path)
    meta = df.attrs["meta"]
    # CLI calib_path metadata'dakini override edebilir
    if calib_path:
        meta["calib_path"] = calib_path
    domain = _resolve_domain(df)
    fps = meta.get("fps") or _infer_fps(df)

    players = _per_player_report(df, domain, fps)
    terr = territory_share(df)
    poss = possession_pct(df, ball_df=None)
    heights = _estimate_heights(df, domain, meta.get("calib_path"))

    # heatmap PNG'leri (opsiyonel; matplotlib varsa)
    heatmap_files = _write_heatmaps(df, domain, players, out)

    report = dict(
        schema_version=1,
        source=dict(
            tracks_path=str(tracks_path),
            camera_id=meta.get("camera_id"),
            source_clip=meta.get("source_clip"),
            fps=round(float(fps), 3) if fps else None,
            frame_size=[meta.get("width"), meta.get("height")],
            calib_path=meta.get("calib_path"),
        ),
        coordinate=dict(
            space=domain["space"],
            scale_calibrated=domain["scale_calibrated"],
            unit=domain["unit"],
        ),
        honesty=dict(
            distance_err="~%2-5 (smoothing + teleport gate sonrasi)",
            speed_err="ort ~%5; anlik max gurultulu -> p99 raporlandi",
            acceleration="RAPORLANMADI (tek-kamera pixel gurultusu guvenilmez)",
            scale_note=(
                "OLCEKLI metre" if domain["scale_calibrated"]
                else "RELATIVE: mesafe/hiz normalize-birim; metre/km DEGIL "
                     "(scale_calibrated=false)"),
            per_zone="Uzak/grazing acida hata buyur; bkz calib QA per_zone.",
            possession="None: top tespit katmani yok (durust).",
            territory="Takim atamasi yok -> takim-duzeyi DEFERRED.",
        ),
        n_players=len(players),
        players=players,
        territory=terr,
        possession_pct=poss,
        height_estimate=heights,
        heatmaps=heatmap_files,
    )

    rep_path = out / "report.json"
    rep_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    report["_report_path"] = str(rep_path)
    return report


def _write_heatmaps(df, domain, players, out_dir) -> dict:
    """Genel + (en uzun birkac) oyuncu heatmap PNG'leri. matplotlib opsiyonel."""
    files = {}
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return {"note": "matplotlib yok -> heatmap PNG atlandi"}

    ext = _domain_extent(df, domain)

    def _plot(grid, title, path):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.imshow(grid, origin="lower", aspect="auto",
                  extent=[ext[0], ext[1], ext[2], ext[3]], cmap="hot")
        ax.set_title(title)
        ax.set_xlabel(f"X ({domain['unit']})")
        ax.set_ylabel(f"Y ({domain['unit']})")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)

    bin_size = 1.0 if domain["space"] == "pitch" else 20.0
    grid = heatmap_grid(df, bin_m=bin_size)
    p = out_dir / "heatmap_all.png"
    _plot(grid, f"Tum oyuncular ({domain['space']}/{domain['unit']})", p)
    files["all"] = str(p)

    # en cok gozlemli 4 oyuncu
    top = sorted(players.items(), key=lambda kv: -kv[1]["n_obs"])[:4]
    for tid, _ in top:
        sub = df[df["tid"] == tid]
        sub.attrs["meta"] = df.attrs.get("meta", {})
        g = heatmap_grid(sub, bin_m=bin_size)
        pp = out_dir / f"heatmap_tid{tid}.png"
        _plot(g, f"tid #{tid}", pp)
        files[f"tid_{tid}"] = str(pp)
    return files


# ============================================================================
# Sentetik test
# ============================================================================
def _synth_tracks(out_path: str, with_pitch=True, scale_calibrated=True,
                  fps=25.0, n_players=4, seconds=30.0):
    """Sentetik per-frame track parquet uret (sema-uyumlu)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    rng = np.random.default_rng(7)
    L, W = 28.0, 18.0
    nfr = int(seconds * fps)
    rows = []
    for tid in range(1, n_players + 1):
        # rastgele yumusak gezinti + arada bir sprint
        x = rng.uniform(3, L - 3)
        y = rng.uniform(3, W - 3)
        vx = rng.uniform(-1, 1)
        vy = rng.uniform(-1, 1)
        for f in range(nfr):
            t = f / fps
            # ara sira sprint (hiz patlamasi)
            if rng.random() < 0.01:
                vx, vy = rng.uniform(-6, 6), rng.uniform(-6, 6)
            vx += rng.normal(0, 0.3)
            vy += rng.normal(0, 0.3)
            vx = np.clip(vx, -7, 7)
            vy = np.clip(vy, -7, 7)
            x = np.clip(x + vx / fps, 0.2, L - 0.2)
            y = np.clip(y + vy / fps, 0.2, W - 0.2)
            # sahte pixel (basit dogrusal harita) + gurultu
            foot_x = 100 + x * 60 + rng.normal(0, 1.5)
            foot_y = 900 - y * 40 + rng.normal(0, 1.5)
            box_h = 70 - y * 1.2 + rng.normal(0, 2)  # uzakta kucuk
            rows.append((tid, f, t, float(foot_x), float(foot_y),
                         float(box_h), float(box_h * 0.4), float(rng.uniform(0.3, 0.6)),
                         False,
                         float(x) if with_pitch else np.nan,
                         float(y) if with_pitch else np.nan,
                         bool(with_pitch)))
    cols = ["tid", "frame", "t_sec", "foot_x", "foot_y", "box_h", "box_w",
            "conf", "bottom_cropped", "pitch_x", "pitch_y", "in_pitch"]
    df = pd.DataFrame(rows, columns=cols)
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = {
        b"camera_id": b"synth_cam",
        b"source_clip": b"synth_clip.mp4",
        b"fps": str(fps).encode(),
        b"width": b"1920", b"height": b"1080",
        b"calib_path": b"",
        b"scale_calibrated": (b"true" if scale_calibrated else b"false"),
        b"schema_version": b"1",
    }
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **meta})
    pq.write_table(table, out_path)
    return out_path


def _selftest():
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="statsrep_"))
    print(f"[selftest] tmp={d}")
    # 1) kalibre+olcekli
    tp = str(d / "synth_cal.parquet")
    _synth_tracks(tp, with_pitch=True, scale_calibrated=True)
    rep = generate_report(tp, None, str(d / "out_cal"))
    print(f"[selftest] kalibreli: {rep['n_players']} oyuncu, unit={rep['coordinate']['unit']}")
    p0 = next(iter(rep["players"].values()))
    print(f"  ornek oyuncu: dist={p0['distance']} {p0['distance_unit']}, "
          f"sprint={p0['sprint_count']}, max_p99={p0['speed_max_p99']} {p0['speed_unit']}")
    print(f"  territory overall={rep['territory']['overall_thirds_share']}")
    print(f"  possession={rep['possession_pct']} (None beklenir)")

    # 2) kalibresiz (relative)
    tp2 = str(d / "synth_rel.parquet")
    _synth_tracks(tp2, with_pitch=True, scale_calibrated=False)
    rep2 = generate_report(tp2, None, str(d / "out_rel"))
    print(f"[selftest] olceksiz: unit={rep2['coordinate']['unit']} "
          f"(relative_m beklenir), scale_calibrated={rep2['coordinate']['scale_calibrated']}")

    # 3) calib-oncesi (pitch yok)
    tp3 = str(d / "synth_nopitch.parquet")
    _synth_tracks(tp3, with_pitch=False, scale_calibrated=False)
    rep3 = generate_report(tp3, None, str(d / "out_px"))
    print(f"[selftest] pitch-yok: space={rep3['coordinate']['space']} (pixel beklenir), "
          f"unit={rep3['coordinate']['unit']}")

    # assert'ler
    assert rep["coordinate"]["unit"] == "m"
    assert rep["coordinate"]["scale_calibrated"] is True
    assert rep["possession_pct"] is None
    assert rep2["coordinate"]["unit"] == "relative_m"
    assert rep3["coordinate"]["space"] == "pixel"
    assert all(p["distance"] >= 0 for p in rep["players"].values())
    print("[selftest] OK — tum assert'ler gecti.")
    print(f"[selftest] report ornegi: {rep['_report_path']}")


# ============================================================================
# CLI
# ============================================================================
def main(argv):
    args = argv[1:]
    if not args or args[0] in ("--selftest", "-t"):
        _selftest()
        return
    tracks_path = args[0]
    calib_path = None
    out_dir = "stats_out"
    i = 1
    while i < len(args):
        if args[i] == "--calib" and i + 1 < len(args):
            calib_path = args[i + 1]
            i += 2
        elif args[i] == "--out" and i + 1 < len(args):
            out_dir = args[i + 1]
            i += 2
        else:
            i += 1
    rep = generate_report(tracks_path, calib_path, out_dir)
    print(json.dumps({k: v for k, v in rep.items() if k != "players"},
                     indent=2, ensure_ascii=False))
    print(f"-> {rep['_report_path']}  ({rep['n_players']} oyuncu)")


if __name__ == "__main__":
    main(sys.argv)
