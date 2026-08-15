#!/usr/bin/env python3
"""Per-frame ayak-noktasi (foot point) export modulu.

track_smoke.py ile BIRE-BIR ayni RF-DETR + ByteTrack kurulumunu kullanir,
ama per-track ozet yerine her (track, frame) gozlemi icin tek satir yazar
(interface spec'teki TRACK KAYIT SEMASI). Stats katmaninin on-kosulu.

Cikti:
  tracks_<stem>.parquet   (varsayilan; pyarrow varsa)
  tracks_<stem>.csv        (--format csv ile; yaninda .meta.json sidecar)

Pitch koordinatlari (pitch_x, pitch_y) yalnizca bir PitchHomography/calib JSON
verildiginde doldurulur; aksi halde NaN kalir ve dosya yine de gecerlidir
(ham video ~15 gunde silinse de bu parquet kalici, sonradan yeniden doldurulabilir).

CLI:
  venv/bin/python export_tracks.py <clip.mp4> [--calib calib/<cam>.json]
                                   [--homography H.json] [--out tracks.parquet]
                                   [--format parquet|csv] [--camera-id <id>]

NOT: agir inference (GPU ~6 saat/mac) burada calisir; sadece kod + GPU-suz
import sanity icin modul lazy-import eder (rfdetr/torch yalnizca calisma aninda).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np

SCHEMA_VERSION = 1

# track_smoke.py ile birebir ayni sabitler
# 1 Tem: GT-anchored olcum ft'yi reg'e ustun buldu (far-band base tespit 74->82,
# +8/92 far oyuncu, precision temiz/gorsel-dogrulandi, yakin-zon degismedi) ->
# production default ft. Eski davranis: HALISAHA_WEIGHTS=regular
import os as _os
DEFAULT_WEIGHTS = ("models/weights/checkpoint_best_regular.pth"
                   if _os.environ.get("HALISAHA_WEIGHTS", "ft").lower() in ("regular", "reg", "base")
                   else "models/weights/checkpoint_ft.pth")
THRESH = 0.3
MIN_H = 25
MIN_ASPECT = 1.3   # h/w alt-siniri: altindakiler ball/kare-FP (insan degil), elenir
NMS_THRESH = 0.6
BOTTOM_CROP_PX = 2.0  # y2 frame altina bu kadar yakinsa foot=box-bottom GECERSIZ (A-graft)

# Parquet kolon sirasi ve dtype'lari (interface spec semasi)
ROW_COLUMNS = [
    "tid", "frame", "t_sec", "foot_x", "foot_y", "box_h", "box_w",
    "conf", "bottom_cropped", "pitch_x", "pitch_y", "in_pitch",
    "zone", "metric_lowconf", "static_susp",
]

# Goruntu-satiri zon esikleri — eval/recall_eval.zone_of ile AYNI tutulmali.
# far-zon: px-QA metrik hatayi 50-90x gizler (0.4-0.55 m/px) -> metric_lowconf.
ZONE_T1, ZONE_T2 = 360.0, 560.0


# --------------------------------------------------------------------------- #
# Yardimcilar
# --------------------------------------------------------------------------- #
def derive_camera_id(clip_path: str | Path) -> str:
    """Klip stem'inden sondaki _clip\\d+ ekini soyup camera_id uretir.

    cankaya_cam2_clip2400.mp4 -> cankaya_cam2
    """
    stem = Path(clip_path).stem
    return re.sub(r"_clip\d+$", "", stem)


def iter_track_rows(detections_per_frame, fps: float,
                    frame_height: int | None = None) -> Iterator[dict]:
    """Takip edilmis per-frame sv.Detections akisini per-satir dict'lere cevirir.

    Yeniden kullanilabilir; track_smoke.py kod tekrari olmadan cagirabilir.

    detections_per_frame : her elemani tracker_id'si atanmis bir sv.Detections
                           olan iterable (frame indexi enumerate ile uretilir).
    fps                  : t_sec = frame / fps.
    frame_height         : verilirse bottom_cropped bayragi hesaplanir.

    Pitch kolonlari (pitch_x/pitch_y/in_pitch) BURADA doldurulmaz; tek bir
    vektorize post-pass'te eklenir (bkz. _fill_pitch_columns).
    """
    for frame_idx, d in enumerate(detections_per_frame):
        if d is None or len(d) == 0:
            continue
        xyxy = d.xyxy
        conf = d.confidence
        tids = d.tracker_id
        if tids is None:
            continue
        t_sec = frame_idx / fps
        for j in range(len(xyxy)):
            tid = tids[j]
            if tid is None:
                continue
            x1, y1, x2, y2 = (float(xyxy[j][0]), float(xyxy[j][1]),
                              float(xyxy[j][2]), float(xyxy[j][3]))
            foot_x = (x1 + x2) / 2.0
            foot_y = y2
            box_h = y2 - y1
            box_w = x2 - x1
            # ball/non-human aspect filtresi: insan kutusu UZUN (h/w~2.4), top ~kare.
            # Alperen gozlemi + olcum: ~457-671 kare top/kare-FP -> sahte track + teleport.
            if box_h / max(box_w, 1.0) < MIN_ASPECT:
                continue
            bottom_cropped = bool(
                frame_height is not None and (frame_height - y2) <= BOTTOM_CROP_PX)
            yield {
                "tid": int(tid),
                "frame": int(frame_idx),
                "t_sec": float(t_sec),
                "foot_x": float(foot_x),
                "foot_y": float(foot_y),
                "box_h": float(box_h),
                "box_w": float(box_w),
                "conf": float(conf[j]) if conf is not None else float("nan"),
                "bottom_cropped": bottom_cropped,
                # pitch kolonlari post-pass'te doldurulur:
                "pitch_x": float("nan"),
                "pitch_y": float("nan"),
                "in_pitch": False,
                "zone": ("far" if foot_y < ZONE_T1
                         else ("mid" if foot_y < ZONE_T2 else "near")),
                "metric_lowconf": bool(foot_y < ZONE_T1),
                "static_susp": False,  # post-pass'te doldurulur
            }


def _try_load_homography(calib_path: str | None):
    """PitchHomography modulu + calib JSON mevcutsa yukler, yoksa None doner.

    pitch/ modulu paralel kodlanyor; burada sert bagimlilik YOK. Modul veya
    dosya yoksa pitch kolonlari NaN birakilir (dosya yine gecerli).
    """
    if not calib_path:
        return None, False
    p = Path(calib_path)
    if not p.exists():
        print(f"[uyari] calib bulunamadi: {calib_path} -> pitch kolonlari NaN",
              flush=True)
        return None, False
    try:
        from pitch.homography import PitchHomography  # lazy, opsiyonel
    except Exception as e:  # noqa: BLE001 - modul henuz yoksa graceful
        print(f"[uyari] pitch.homography import edilemedi ({e}); "
              f"pitch kolonlari NaN", flush=True)
        return None, False
    try:
        homo = PitchHomography.load(str(p))
    except Exception as e:  # noqa: BLE001
        print(f"[uyari] calib yuklenemedi ({e}); pitch kolonlari NaN", flush=True)
        return None, False
    # QA-KAPISI: bozuk calib (median_px>20 / near>far ters / n_landmarks<6) SESSIZCE
    # metrik uretmesin. status=='manual' guvenmek yeterli degil; calib/cankaya_cam2.json
    # (65px, ters-H) bu kapidan gecemez, _v2.json (9.87px) gecer. Dosya adina degil QA'ya bak.
    try:
        from stats.topdown_stats import accept_calib_qa  # lazy, opsiyonel
        ok, reasons = accept_calib_qa(getattr(homo, "_qa", None))
        if not ok:
            print(f"[UYARI] calib QA-kapisindan GECMEDI ({calib_path}): {reasons}; "
                  f"bozuk homografi REDDEDILDI -> pitch kolonlari NaN", flush=True)
            return None, False
    except Exception as e:  # noqa: BLE001 - kapi modulu yoksa yumusak gec
        print(f"[uyari] calib QA-kapisi atlandi ({e}); homografi QA'siz kabul edildi",
              flush=True)
    # olcek kalibreli mi? scale_anchor varsa true.
    scale_calibrated = False
    try:
        data = json.loads(p.read_text())
        scale_calibrated = bool(data.get("scale_anchor"))
    except Exception:  # noqa: BLE001
        pass
    return homo, scale_calibrated


def _fill_pitch_columns(rows: list[dict], homo) -> None:
    """Tek vektorize post-pass: tum foot noktalarini pitch metreye cevir.

    cv2.undistortPoints -> H_img2pitch matmul (PitchHomography icinde). Sparse,
    tum mac icin ~mikrosaniye, ek GPU yok. rows yerinde guncellenir.
    """
    if homo is None or not rows:
        return
    foot = np.array([[r["foot_x"], r["foot_y"]] for r in rows], dtype=np.float64)
    pitch = homo.pixel_to_pitch(foot)               # (N,2) metre
    in_pitch = homo.in_pitch(pitch)                 # (N,) bool
    for r, (px, py), inp in zip(rows, pitch, in_pitch):
        r["pitch_x"] = float(px)
        r["pitch_y"] = float(py)
        r["in_pitch"] = bool(inp)
        # bottom_cropped foot noktasi guvenilmez -> pitch dusuk guvenli isaretle
        if r["bottom_cropped"]:
            r["in_pitch"] = False


# --------------------------------------------------------------------------- #
# Yazma (parquet birincil, csv fallback)
# --------------------------------------------------------------------------- #
def _write_parquet(rows: list[dict], out_path: Path, meta: dict) -> None:
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    df = pd.DataFrame(rows, columns=ROW_COLUMNS)
    # dtype'lari semaya sabitle
    df = df.astype({
        "tid": "int32", "frame": "int32", "t_sec": "float32",
        "foot_x": "float32", "foot_y": "float32", "box_h": "float32",
        "box_w": "float32", "conf": "float32", "bottom_cropped": "bool",
        "pitch_x": "float32", "pitch_y": "float32", "in_pitch": "bool",
        "zone": "str", "metric_lowconf": "bool", "static_susp": "bool",
    })
    table = pa.Table.from_pandas(df, preserve_index=False)
    # dosya-seviye metadata key-value (parquet schema metadata)
    kv = {k: json.dumps(v) for k, v in meta.items()}
    table = table.replace_schema_metadata({
        **{k.encode(): v.encode() for k, v in kv.items()},
        b"halisaha_schema_version": str(SCHEMA_VERSION).encode(),
    })
    pq.write_table(table, str(out_path))


def _write_csv(rows: list[dict], out_path: Path, meta: dict) -> None:
    import pandas as pd

    df = pd.DataFrame(rows, columns=ROW_COLUMNS)
    df.to_csv(out_path, index=False)
    # CSV dosya-seviye sabitler icin sidecar JSON
    side = out_path.with_suffix(out_path.suffix + ".meta.json")
    side.write_text(json.dumps({**meta, "schema_version": SCHEMA_VERSION},
                               indent=2))


# --------------------------------------------------------------------------- #
# Ana akis
# --------------------------------------------------------------------------- #
def run_tracking_export(clip_path: str,
                        weights_path: str = DEFAULT_WEIGHTS,
                        out_path: str | None = None,
                        camera_id: str | None = None,
                        calib_path: str | None = None,
                        fmt: str = "parquet",
                        max_frames: int | None = None,
                        recover_far: bool = False,
                        far_band: tuple[int, int] = (110, 365)) -> str:
    """RF-DETR + ByteTrack ile per-frame ayak-noktasi export.

    Tespit+takip dongusu track_smoke.py ile BIRE-BIR ayni (detektore dokunma).
    Donus: yazilan dosya yolu (str).
    """
    # Agir/GPU bagimliliklari lazy import (GPU-suz import sanity icin)
    import cv2
    from PIL import Image
    import supervision as sv
    from rfdetr import RFDETRLargeDeprecated

    clip = Path(clip_path)
    cam_id = camera_id or derive_camera_id(clip)

    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"clip {clip.name}: {W}x{H} fps={fps:.2f} frames={n_frames} "
          f"camera_id={cam_id}", flush=True)

    # Opsiyonel calib (yoksa pitch kolonlari NaN)
    homo, scale_calibrated = _try_load_homography(calib_path)

    # --- detektor + tracker: track_smoke.py ile birebir ---
    model = RFDETRLargeDeprecated(
        pretrain_weights=weights_path, device="cuda", num_classes=4)
    tracker = sv.ByteTrack(frame_rate=int(round(fps)),
                           track_activation_threshold=0.25,
                           lost_track_buffer=int(round(fps)) * 2,
                           minimum_matching_threshold=0.8)

    # --- opsiyonel far-band tiling recovery (recall_qc.FarBandRecovery) ---
    recov = None
    if recover_far:
        try:
            from detect.recall_qc import (load_calibrated_homography,
                                          FarBandRecovery)
            homo_qc, qa = load_calibrated_homography(calib_path)
            if homo_qc is None:
                print(f"[uyari] far-recovery: calib QA gecmedi ({qa}); KAPALI", flush=True)
            else:
                def _pred(img_rgb, thr):
                    det = model.predict(Image.fromarray(img_rgb), threshold=thr)
                    return det.xyxy.copy(), det.confidence.copy()
                recov = FarBandRecovery(_pred, homo_qc, tuple(far_band))
                print(f"[bilgi] far-band tiling AKTIF, band={far_band}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[uyari] far-recovery kurulamadi ({e}); KAPALI", flush=True)
            recov = None

    def tracked_stream():
        """track_smoke.py donusunun birebir aynisi; tracked Detections yield eder."""
        i = 0
        t0 = time.time()
        while True:
            if max_frames is not None and i >= max_frames:
                break
            ok, frame = cap.read()
            if not ok:
                break
            det = model.predict(Image.fromarray(frame[:, :, ::-1]), threshold=THRESH)
            _cls = getattr(det, "class_id", None)
            cls = (np.asarray(_cls, dtype=int).reshape(-1) if _cls is not None
                   else np.ones(len(det.xyxy), dtype=int))
            hh = det.xyxy[:, 3] - det.xyxy[:, 1]
            ww = np.maximum(det.xyxy[:, 2] - det.xyxy[:, 0], 1e-6)
            # cls==0 = ball (SoccerNet ckpt) — AMA class-id'ler 1-4'e sacilabiliyor
            # (guvenilmez etiket). Guvenli drop: cls==0 VE kutu top-bicimli (h/w<1.6).
            # Yanlislikla ball-etiketli UZUN kutu (gercek oyuncu) korunur.
            ball_like = (cls == 0) & ((hh / ww) < 1.6)
            keep = (hh > MIN_H) & ~ball_like
            d = sv.Detections(xyxy=det.xyxy[keep],
                              confidence=det.confidence[keep],
                              class_id=cls[keep])
            d = d.with_nms(threshold=NMS_THRESH, class_agnostic=True)
            if recov is not None:
                d = recov.process(i, i / fps, frame, d)   # far-band tiling: kurtarilan dets eklenir
            d = tracker.update_with_detections(d)
            yield d
            i += 1
            if i % 200 == 0:
                el = time.time() - t0
                print(f"  {i}/{n_frames} ({i/el:.1f} fps proc, "
                      f"eta {(n_frames-i)/(i/el)/60:.1f}min)", flush=True)

    t0 = time.time()
    rows = list(iter_track_rows(tracked_stream(), fps, frame_height=H))
    cap.release()

    # far-band tiling parity sidecar (varsa)
    if recov is not None and recov.parity_log:
        try:
            from detect.recall_qc import write_parity_sidecar
            pp = write_parity_sidecar(
                recov.parity_log, str(clip.parent / f"parity_{clip.stem}.parquet"))
            n_rec = sum(p["n_recovered"] for p in recov.parity_log)
            print(f"far-band tiling: {n_rec} kurtarilan tespit -> {pp}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[uyari] parity sidecar yazilamadi ({e})", flush=True)

    # Tek vektorize post-pass: pitch metre doldur
    _fill_pitch_columns(rows, homo)

    # Statik-hayalet suphesi (bicme-seridi/pano FP'leri, AUDIT_ROADMAP fix#2):
    # >4s yasayan + goruntu-uzayinda ~kilitli (foot std<5px) track ISARETLENIR (drop degil).
    # Gercek hareketsiz kaleci >6px gezinir (detect/static_fp_filter piksel-kilit bulgusu).
    try:
        from collections import defaultdict
        _by_tid: dict = defaultdict(list)
        for r in rows:
            _by_tid[r["tid"]].append(r)
        n_susp = 0
        for _tid, rr in _by_tid.items():
            if len(rr) < 2:
                continue
            if max(r["t_sec"] for r in rr) - min(r["t_sec"] for r in rr) < 4.0:
                continue
            fx = np.array([r["foot_x"] for r in rr])
            fy = np.array([r["foot_y"] for r in rr])
            if float(fx.std()) < 5.0 and float(fy.std()) < 5.0:
                n_susp += 1
                for r in rr:
                    r["static_susp"] = True
        if n_susp:
            print(f"static_susp: {n_susp} track isaretlendi (statik-hayalet suphesi)",
                  flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[uyari] static_susp hesaplanamadi ({e})", flush=True)

    # Metrik-QA (px-QA Goodhart fix): per-zone px residual x lokal m/px = metre residual.
    # FINAL'de far ~6px 'iyi' gorunur ama 0.4-0.55 m/px ile ~0.7-2.4m gercek belirsizlik.
    metric_qa = None
    try:
        qa = getattr(homo, "_qa", None) if homo is not None else None
        if qa and isinstance(qa.get("per_zone"), dict):
            zone_rep_y = {"far": 240.0, "mid": 460.0, "near": 800.0}
            metric_qa = {}
            for z, yv in zone_rep_y.items():
                a = np.asarray(homo.pixel_to_pitch(
                    np.array([[W / 2.0, yv]], dtype=np.float64))).reshape(2)
                b = np.asarray(homo.pixel_to_pitch(
                    np.array([[W / 2.0, yv + 1.0]], dtype=np.float64))).reshape(2)
                m_per_px = float(np.hypot(*(b - a)))
                zpx = float(qa["per_zone"].get(z, float("nan")))
                metric_qa[z] = {"px": round(zpx, 2), "m_per_px": round(m_per_px, 4),
                                "residual_m": round(zpx * m_per_px, 3)}
            print(f"metrik-QA (m): { {z: v['residual_m'] for z, v in metric_qa.items()} }",
                  flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[uyari] metrik-QA hesaplanamadi ({e})", flush=True)

    # Cikti yolu
    if out_path is None:
        ext = "csv" if fmt == "csv" else "parquet"
        out = clip.parent / f"tracks_{clip.stem}.{ext}"
    else:
        out = Path(out_path)
        # format'i uzantidan tahmin et (acik fmt override etmez)
        if out.suffix.lower() == ".csv":
            fmt = "csv"
        elif out.suffix.lower() in (".parquet", ".pq"):
            fmt = "parquet"

    meta = {
        "camera_id": cam_id,
        "source_clip": clip.name,
        "fps": float(fps),
        "width": int(W),
        "height": int(H),
        "calib_path": str(calib_path) if calib_path else None,
        "scale_calibrated": bool(scale_calibrated),
        "schema_version": SCHEMA_VERSION,
        "metric_qa_m_per_zone": metric_qa,
    }

    if fmt == "csv":
        _write_csv(rows, out, meta)
    else:
        try:
            _write_parquet(rows, out, meta)
        except Exception as e:  # noqa: BLE001 - pyarrow yoksa csv'e dus
            print(f"[uyari] parquet yazilamadi ({e}); CSV fallback", flush=True)
            out = out.with_suffix(".csv")
            _write_csv(rows, out, meta)

    n_tracks = len({r["tid"] for r in rows})
    n_pitch = sum(1 for r in rows if not np.isnan(r["pitch_x"]))
    print(f"\n{len(rows)} satir, {n_tracks} track yazildi -> {out}", flush=True)
    print(f"pitch dolu satir: {n_pitch}/{len(rows)} "
          f"(scale_calibrated={scale_calibrated})", flush=True)
    print(f"sure: {(time.time()-t0)/60:.1f} dk", flush=True)
    return str(out)


def _parse_args(argv):
    ap = argparse.ArgumentParser(description="Per-frame foot-point track export.")
    ap.add_argument("clip", help="girdi klip (.mp4)")
    ap.add_argument("--calib", "--homography", dest="calib", default=None,
                    help="calib/<camera_id>.json (PitchHomography); verilirse "
                         "pitch_x/pitch_y doldurulur")
    ap.add_argument("--out", default=None, help="cikti yolu (uzanti format belirler)")
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet",
                    dest="fmt")
    ap.add_argument("--camera-id", dest="camera_id", default=None,
                    help="camera_id override (varsayilan: stem'den turetilir)")
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--max-frames", dest="max_frames", type=int, default=None,
                    help="yalnizca ilk N frame'i isle (hizli smoke / kismi export)")
    ap.add_argument("--recover-far", dest="recover_far", action="store_true",
                    help="far-band tiling recovery aktif (recall_qc.FarBandRecovery)")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    run_tracking_export(
        clip_path=args.clip,
        weights_path=args.weights,
        out_path=args.out,
        camera_id=args.camera_id,
        calib_path=args.calib,
        fmt=args.fmt,
        max_frames=args.max_frames,
        recover_far=args.recover_far,
    )


if __name__ == "__main__":
    main()
