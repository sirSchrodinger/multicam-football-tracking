#!/usr/bin/env python3
"""recall_qc.py — Parity-QC + parite-kapili uzak-ucte-bir recall recovery.

Tasarim sentezi (3 bagimsiz tasarimin olculmus en iyi parcalari):
  - Parite bir SINYALDIR, clamp DEGIL: per-frame sayiyi asla 14'e zorla-ma.
    Oyuncu mesru olarak kadraj disinda olabilir (korner) veya hakem olabilir.
    Budget yalnizca aday SIRALAMA onceligidir; overshoot QC bayragi ile raporlanir.
  - Tiling YALNIZCA uzak-bant + dusuk-cozunur deficit frame'lerinde ve cadence
    kapali (her N. frame) -> 4GB GTX1650Ti'de hesap sinirli; ByteTrack araligi tasir.
  - TILE_THRESH=0.40 (olculdu: 0.15-0.20 FP seli -> 22/frame; 0.40 cogu 13-14'te kalir).
  - Pitch-metre dedup base tespitlerine karsi (kucuk uzak kutularda IoU-NMS basarisiz;
    upscale-edilmis kutular alt-piksel kayikligi yuzunden merge OLMAZ).
  - Recovered tespitler PRESENCE-ONLY: low_conf=True, metrik mesafe/heatmap'ten HARIC.
  - Provenance Detections.data ile tasinir (sv 0.28.0'da with_nms+ByteTrack uzerinden
    KORUNDUGU dogrulandi) AMA conf<THRESH + foot_y-bant deterministik fallback'i her
    zaman vardir; bir sv upgrade data-carry'i dusurse de parquet kolonlari gecerli kalir.
  - Calib-QA YUKLEME KAPISI: kanonik calib/cankaya_cam2.json BOZUK (median 65px,
    near 111px); calib/cankaya_cam2_v2.json saglikli (median ~10px). Bozuk calib
    geometriyi sessizce bozar -> yukleme aninda median_px esigi asilirsa REDDET.
  - Cift-yonlu QC: deficit(<14) -> 'uzak-ucte-bir eksik oyuncu' + recovery tetigi;
    surplus(>14) -> 'hakem/cift FP' bayragi (BU katman recovery YAPMAZ; class-filtre isi).
  - GPU-suz parite QC mevcut parquet uzerinde HEMEN calisir (audit + held-out recall taban).

Opt-in: recovery yalnizca acikca cagrildiginda calisir. recovery KAPALI iken
export_tracks.py tespit+takip dongusu BIRE-BIR mevcut ciktiyi uretir.

Olculen (bu footage, gercek RF-DETR; smoke degil):
  parquet 2985 frame: complete(=14) %15.3 | tek %48.9 | <14 %83.6 | deficit 6086 oyuncu-frame
  far tercile: n=24121 box_h_med=70.9px conf_med=0.655 frac<0.4=%8.0 (eksikler burada)
  calib v2 QA: median 9.87px (kanonik .json 65px -> yukleme kapisi gerekcesi)

DURUSTLUK: hicbir recall % iddiasi DONDURULMUS insan-GT olmadan yapilmaz.
Bu katmanin uretebildigi rapor 'parite-tamamlanma orani' ve 'uretilen aday sayisi'dir.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# export_tracks.py / track_smoke.py ile BIRE-BIR ayni sabitler
THRESH = 0.3            # base full-frame floor (degismez)
MIN_H = 25
NMS_THRESH = 0.6
EXPECTED_PLAYERS = 14   # 7v7 (sadece tetik+QC; hedef DEGIL)

# recovery sabitleri (olculmus)
TILE_THRESH = 0.40      # uzak-bant crop floor (0.15-0.20 FP seli; 0.40 secildi)
FAR_MARGIN_M = 4.0      # uzak-bantta grazing-aci hatasi -> daha genis in_pitch margin
DEDUP_M = 1.5           # ayni-oyuncu pitch-metre yaricapi
TILE_EVERY = 6          # ~4Hz keyframe tiling (hesap kapisi); ByteTrack arasini tasir
DEFICIT_WIN = 13        # ~0.5s @25fps yuvarlanan pencere
LOW_CONF_THR = 0.40     # bu altindaki base/far dets low_conf

# calib yukleme kapisi: bu medyan reproj hatasini asan calib REDDEDILIR
CALIB_MAX_MEDIAN_PX = 20.0


# =========================================================================== #
# 1) Calib-QA yukleme kapisi  (bozuk kanonik .json tuzagina karsi)
# =========================================================================== #
def load_calibrated_homography(calib_path: str,
                               max_median_px: float = CALIB_MAX_MEDIAN_PX):
    """PitchHomography yukle AMA QA medyan reproj hatasi esigi asarsa REDDET.

    Doner: (homo, qa_dict) veya (None, qa_dict). homo None ise recovery in_pitch
    kapisi guvenilmez -> cagiran taraf recovery'yi devre disi birakmali.
    """
    p = Path(calib_path)
    if not p.exists():
        return None, {"error": f"calib yok: {calib_path}"}
    try:
        from pitch.homography import PitchHomography
        homo = PitchHomography.load(str(p))
    except Exception as e:  # noqa: BLE001
        return None, {"error": f"yuklenemedi: {e}"}
    qa = homo._qa or {}
    median_px = float(qa.get("median_px", float("nan")))
    if not np.isnan(median_px) and median_px > max_median_px:
        return None, {"rejected": True, "median_px": median_px,
                      "threshold": max_median_px,
                      "reason": "calib reproj hatasi cok yuksek (bozuk calib tuzagi)"}
    return homo, {"accepted": True, "median_px": median_px, **qa}


# =========================================================================== #
# 2) GPU-suz parite QC  (mevcut parquet uzerinde HEMEN; held-out recall taban)
# =========================================================================== #
def _zone_thresholds(foot_y: np.ndarray):
    """foot_y terciline gore (far_t1, mid_t2) esikleri. far=ust, near=alt."""
    y0, y1 = float(np.min(foot_y)), float(np.max(foot_y))
    return y0 + (y1 - y0) / 3.0, y0 + 2.0 * (y1 - y0) / 3.0


def zone_of(foot_y, t1: float, t2: float) -> np.ndarray:
    """0=far(ust) 1=mid 2=near(alt). foot_y skaler ya da diziyle calisir."""
    fy = np.asarray(foot_y, float)
    z = np.where(fy < t1, 0, np.where(fy < t2, 1, 2)).astype(np.int8)
    return z


def parity_qc_from_parquet(df, expected: int = EXPECTED_PLAYERS) -> dict:
    """GPU yok. Mevcut tracks_*.parquet uzerinde cift-yonlu parite QC.

    Doner: ozet metrikler + per_frame Series + per-frame flag listesi.
    deficit(<14) -> uzak-ucte-bir eksik (corr(deficit,far) negatif); surplus -> FP.
    """
    cnt = df.groupby("frame").size()
    t1, t2 = _zone_thresholds(df["foot_y"].to_numpy(float))
    zser = zone_of(df["foot_y"].to_numpy(float), t1, t2)
    far_by_frame = (df.assign(_far=(zser == 0))
                      .groupby("frame")["_far"].sum()
                      .reindex(cnt.index, fill_value=0))
    flags = []
    for f, n in cnt.items():
        if n < expected:
            flags.append(dict(frame=int(f), state="deficit",
                              missing=int(expected - n),
                              likely_zone="far", far_count=int(far_by_frame[f])))
        elif n > expected:
            flags.append(dict(frame=int(f), state="surplus",
                              extra=int(n - expected),
                              note="hakem/cift FP -> class-filtre, recovery degil"))
    return dict(
        n_frames=int(len(cnt)),
        pct_complete=round(float((cnt == expected).mean()) * 100, 1),
        pct_deficit=round(float((cnt < expected).mean()) * 100, 1),
        pct_surplus=round(float((cnt > expected).mean()) * 100, 1),
        pct_odd=round(float((cnt % 2 == 1).mean()) * 100, 1),
        deficit_player_frames=int((expected - cnt).clip(lower=0).sum()),
        per_frame=cnt,
        flags=flags,
    )


# =========================================================================== #
# 3) Downstream kolon turetme  (schema v1 -> v2; deterministik, data-carry'den bagimsiz)
# =========================================================================== #
def derive_qc_columns(df, thresh: float = THRESH,
                      low_conf_thr: float = LOW_CONF_THR):
    """df'e zone/recovered/low_conf kolonlari ekle (yerinde df doner).

    recovered  : df'te 'recovered' kolonu varsa onu kullan; yoksa conf<thresh
                 (base floor 0.30 oldugu icin altindaki her kutu recovery'den gelir).
    zone       : foot_y tercili (0 far/1 mid/2 near).
    low_conf   : recovered OR (zone==far AND conf<low_conf_thr) OR bottom_cropped.
                 PRESENCE-ONLY: stats_report bu satirlari mesafe/heatmap'ten haric tutar.
    Eski v1 okuyucular ek kolonlari yok sayar (additive sema).
    """
    fy = df["foot_y"].to_numpy(float)
    t1, t2 = _zone_thresholds(fy)
    df = df.copy()
    df["zone"] = zone_of(fy, t1, t2)
    if "recovered" in df.columns:
        rec = df["recovered"].to_numpy(bool)
    else:
        rec = df["conf"].to_numpy(float) < thresh
    df["recovered"] = rec
    botc = (df["bottom_cropped"].to_numpy(bool)
            if "bottom_cropped" in df.columns else np.zeros(len(df), bool))
    df["low_conf"] = (rec
                      | ((df["zone"].to_numpy() == 0)
                         & (df["conf"].to_numpy(float) < low_conf_thr))
                      | botc)
    return df


# =========================================================================== #
# 4) Uzak-bant aday cikarimi  (saf geometri; GPU predict enjekte edilir)
# =========================================================================== #
def far_band_from_density(npz_path: str, lo_pct: float = 1.0,
                          hi_pct: float = 40.0, pad: float = 0.04):
    """density_<cam>.npz foot_y dagiliminin yuzdeliklerinden far-bant satirlari.

    Sabit-kamerada uzak oyuncular ust satirlarda toplanir; bandi VERIDEN turet
    (baska venue/kadrajda elle ayar gerekmesin). Doner: (y0, y1) tamsayi satir.
    """
    d = np.load(npz_path, allow_pickle=True)
    # density npz: tespit kutularinin altlari (foot_y) bekleniyor; esnek oku.
    # Sira: (a) dogrudan foot_y/ys/y2 dizisi; (b) 2B points + columns adlandirmasi
    # (uretici format: points[N,K] + columns[K] -> foot_y sutununu adindan bul);
    # (c) son care, ilk dizi xyxy ise alt kenar.
    fy = None
    for key in ("foot_y", "ys", "y2"):
        if key in d:
            fy = np.asarray(d[key]).ravel().astype(float)
            break
    if fy is None and "points" in d and "columns" in d:
        cols = [str(c) for c in np.asarray(d["columns"]).ravel()]
        pts = np.asarray(d["points"], dtype=float).reshape(len(cols) and -1, len(cols))
        for name in ("foot_y", "y2", "ys"):
            if name in cols:
                fy = pts[:, cols.index(name)].astype(float)
                break
    if fy is None:  # xyxy stilinde ise alt kenari al
        arr = np.asarray(d[d.files[0]], dtype=float)
        if arr.ndim == 2 and arr.shape[1] >= 4:
            fy = arr[:, 3].astype(float)
        else:
            fy = np.concatenate(
                [np.asarray(a).reshape(-1, 4) for a in arr])[:, 3].astype(float)
    y0 = np.percentile(fy, lo_pct)
    y1 = np.percentile(fy, hi_pct)
    span = y1 - y0
    return int(max(0, y0 - pad * span)), int(y1 + pad * span)


class FarBandRecovery:
    """Stateful, parite-kapili uzak-bant recall recovery (per-stream).

    predict_fn(img_rgb, thr) -> (xyxy[N,4], conf[N]) cagirana aittir (RF-DETR sarmali);
    bu sinif GPU/torch import ETMEZ -> GPU-suz import + test mumkun.

    process() base sv.Detections alir, base+recovered sv.Detections doner;
    data['recovered'] bool dizisi tasir, ayrica parity_log sidecar biriktirir.
    """

    def __init__(self, predict_fn: Callable, homo, band: tuple[int, int],
                 expected: int = EXPECTED_PLAYERS, up: float = 2.0,
                 tile_thresh: float = TILE_THRESH, tile_every: int = TILE_EVERY,
                 deficit_win: int = DEFICIT_WIN, dedup_m: float = DEDUP_M,
                 far_margin_m: float = FAR_MARGIN_M, clahe: bool = False):
        self.predict = predict_fn
        self.homo = homo
        self.band = band
        self.expected = expected
        self.up = up
        self.tile_thresh = tile_thresh
        self.tile_every = tile_every
        self.deficit_win = deficit_win
        self.dedup_m = dedup_m
        self.far_margin_m = far_margin_m
        self.clahe = clahe          # OFF default, OLCULMEMIS (no-unproven-claims)
        self._recent: list[int] = []
        self.parity_log: list[dict] = []

    # ---- hesap kapisi: kalici deficit AND her N. frame ----
    def _should_tile(self, frame_idx: int, base_count: int) -> bool:
        self._recent.append(base_count)
        if len(self._recent) > self.deficit_win:
            self._recent.pop(0)
        persistent = float(np.median(self._recent)) < self.expected
        return persistent and (frame_idx % self.tile_every == 0)

    def _enhance(self, crop_bgr):
        if not self.clahe:
            return crop_bgr
        import cv2
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _tile_detect(self, frame_bgr):
        """Far-bandi CLAHE(ops)+upscale ile re-detect; kutulari TAM-FRAME ham px'e dondur."""
        import cv2
        y0, y1 = self.band
        crop = self._enhance(frame_bgr[y0:y1].copy())
        big = cv2.resize(crop, None, fx=self.up, fy=self.up,
                         interpolation=cv2.INTER_CUBIC)
        xy, conf = self.predict(big[:, :, ::-1], self.tile_thresh)  # RGB ver
        xy = np.asarray(xy, float).reshape(-1, 4).copy()
        conf = np.asarray(conf, float).reshape(-1)
        if len(xy):
            xy[:, [0, 2]] /= self.up
            xy[:, [1, 3]] = xy[:, [1, 3]] / self.up + y0
            keep = (xy[:, 3] - xy[:, 1]) > MIN_H
            xy, conf = xy[keep], conf[keep]
        return xy, conf

    def process(self, frame_idx: int, t_sec: float, frame_bgr, base_dets):
        """base_dets: post-NMS, pre-tracker sv.Detections. Doner: merged sv.Detections.

        recovery KAPALIysa (should_tile False / homo None) base_dets'i AYNEN doner
        (data['recovered'] hepsi False) -> bire-bir baseline korunur.
        """
        import supervision as sv
        bx = base_dets.xyxy
        bc = base_dets.confidence
        n_base = len(bx)
        rec_flag = np.zeros(n_base, bool)
        n_rec = 0

        do = (self.homo is not None) and self._should_tile(frame_idx, n_base)
        if do:
            # base ayak noktalari -> pitch metre (in_pitch sayim + dedup referansi)
            bfoot = np.stack([(bx[:, 0] + bx[:, 2]) / 2.0, bx[:, 3]], 1) if n_base else np.empty((0, 2))
            bm = self.homo.pixel_to_pitch(bfoot) if n_base else np.empty((0, 2))
            in_b = (self.homo.in_pitch(bm, margin_m=self.far_margin_m)
                    if n_base else np.zeros(0, bool))
            budget = max(0, self.expected - int(in_b.sum()))  # SOFT oncelik, clamp DEGIL
            tx, tc = self._tile_detect(frame_bgr)
            if len(tx):
                tfoot = np.stack([(tx[:, 0] + tx[:, 2]) / 2.0, tx[:, 3]], 1)
                tm = self.homo.pixel_to_pitch(tfoot)
                in_t = self.homo.in_pitch(tm, margin_m=self.far_margin_m)
                kept_m = [bm[i] for i in range(n_base) if in_b[i]]
                cand = []
                for i in np.argsort(-tc):           # en yuksek-conf adaylar once
                    if not in_t[i]:                  # off-pitch (file/tribun) -> red
                        continue
                    if kept_m and min(np.linalg.norm(
                            np.asarray(kept_m) - tm[i], axis=1)) < self.dedup_m:
                        continue                     # base/onceki ile cift
                    cand.append(i)
                    kept_m.append(tm[i])
                    # SOFT budget: budget'a kadar olanlar 'beklenen'; ustu overshoot
                    # bayrakli kabul edilir (clamp YOK) -> insight#2 mesru >14'e izin
                if cand:
                    ci = np.array(cand, int)
                    bx = np.vstack([bx, tx[ci]])
                    bc = np.concatenate([bc, tc[ci]])
                    rec_flag = np.concatenate([rec_flag, np.ones(len(ci), bool)])
                    n_rec = len(ci)

        merged = sv.Detections(
            xyxy=bx, confidence=bc,
            class_id=np.zeros(len(bx), int),
            data={"recovered": rec_flag})

        n_merged = len(bx)
        self.parity_log.append(dict(
            frame=int(frame_idx), t_sec=float(t_sec),
            n_base=int(n_base), n_recovered=int(n_rec), n_merged=int(n_merged),
            expected=int(self.expected),
            deficit=int(max(self.expected - n_merged, 0)),
            odd=bool(n_merged % 2 == 1),
            overshoot=bool(n_merged > self.expected),   # >14 -> hakem/FP QC
            tiled=bool(do),
        ))
        return merged


def write_parity_sidecar(parity_log: list[dict], out_path: str) -> str:
    """parity_<stem>.parquet sidecar yaz (pyarrow yoksa csv)."""
    import pandas as pd
    df = pd.DataFrame(parity_log)
    p = Path(out_path)
    try:
        df.to_parquet(p, index=False)
    except Exception:  # noqa: BLE001
        p = p.with_suffix(".csv")
        df.to_csv(p, index=False)
    return str(p)


# =========================================================================== #
# export_tracks.py ENTEGRASYON (opt-in; KAPALI iken bire-bir baseline)
# =========================================================================== #
# 1) CLI: ap.add_argument("--recover-far", action="store_true")
#         ap.add_argument("--far-band", default=None, help="y0,y1; yoksa density'den")
# 2) run_tracking_export icinde model kurulduktan sonra:
#       recov = None
#       if args.recover_far:
#           from detect.recall_qc import (load_calibrated_homography,
#                                          FarBandRecovery, far_band_from_density,
#                                          write_parity_sidecar)
#           homo_qc, qa = load_calibrated_homography(calib_path)
#           if homo_qc is None:
#               print(f"[uyari] calib QA gecmedi ({qa}); recovery DEVRE DISI")
#           else:
#               band = (tuple(map(int,args.far_band.split(",")))
#                       if args.far_band else far_band_from_density(density_npz))
#               def _pred(img_rgb, thr):
#                   det = model.predict(Image.fromarray(img_rgb), threshold=thr)
#                   return det.xyxy.copy(), det.confidence.copy()
#               recov = FarBandRecovery(_pred, homo_qc, band)
# 3) tracked_stream() icinde, `d = d.with_nms(...)` SONRASI, tracker.update ONCESI:
#       if recov is not None:
#           d = recov.process(i, i/fps, frame, d)   # KAPALI/budget=0 -> bire-bir d
#       d = tracker.update_with_detections(d)        # data['recovered'] tasinir (sv0.28 dogrulandi)
# 4) iter_track_rows(): d.data.get('recovered') oku -> row['recovered'] yaz
#       (yoksa False); pitch post-pass'ten SONRA derive_qc_columns ile
#       zone/low_conf turet. SCHEMA_VERSION = 2; ROW_COLUMNS += [recovered].
# 5) export sonunda: write_parity_sidecar(recov.parity_log, f"parity_{stem}.parquet")
#
# track_smoke.py + export_tracks.py tespit blogu birebir kopya; ileride
# detect/detect_frame() olarak ortaklastirilmali (drift onleme) — bu dosyanin
# disinda ayri refactor.
