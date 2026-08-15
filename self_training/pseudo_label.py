"""pseudo_label.py — yuksek-guven track-lerinden pseudo-label uretimi (DEFERRED).

Iki tur pseudo-label uretilir:

  A) DETECTION (oyuncu bbox) pseudo-label'lari:
     export_tracks.py'in yazdigi per-frame parquet'ten (track_record_schema), sadece
     yuksek-guvenli + uzun-omurlu + zaman-tutarli tracklet'leri secip COCO/YOLO formatina
     cevirir. Amac: dedektoru (RF-DETR) amator sabit-kamera dagilimina drift-guard
     altinda yeniden ince-ayarlamak icin ucuz, otomatik etiket toplamak.

  B) LINE-SEGMENTATION pseudo-label'lari:
     Kabul edilmis/manuel bir homografi H, parametrik saha sablonunu goruntuye geri
     yansitir -> sinir cizgilerinin yogun (dense) ikili maskesi = bedava segmentasyon GT.
     Bu, ileride lisans-temiz bir seg agi (SegFormer-b0) egitmek icin korpus uretir.

Interface spec'e uyumlu imzalar:
    detection_pseudo_labels(tracks_df, conf_thr=0.6, min_dur_s=10.0) -> list[dict]
    line_seg_labels_from_H(homo, img_shape) -> np.ndarray

Bu modul CPU-only ve import-edilebilir (torch/cv2 hot-path yok). cv2 sadece
line_seg ciziminde, fonksiyon icinde lazy import edilir.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

# ---------------------------------------------------------------------------
# Track schema kolonlari (export_tracks.py ile birebir; track_record_schema)
# ---------------------------------------------------------------------------
_REQUIRED_COLS = ("tid", "frame", "foot_x", "foot_y", "box_h", "box_w", "conf")
PERSON_CLASS_ID = 0  # tek sinif: person (sinif id dedektorde guvenilmez, hepsi person)


# ---------------------------------------------------------------------------
# bbox yeniden insasi
# ---------------------------------------------------------------------------
def reconstruct_bbox_xyxy(df) -> np.ndarray:
    """Per-frame foot-noktasi semasindan tam bbox (x1,y1,x2,y2) geri uret.

    Sema yalniz ayak-noktasi + kutu boyutlari tutuyor:
        foot_x = (x1+x2)/2,  foot_y = y2,  box_w = x2-x1,  box_h = y2-y1
    Dolayisiyla:
        x1 = foot_x - box_w/2 ; x2 = foot_x + box_w/2 ; y2 = foot_y ; y1 = foot_y - box_h
    Donen: (N,4) float32.
    """
    fx = df["foot_x"].to_numpy(dtype=np.float32)
    fy = df["foot_y"].to_numpy(dtype=np.float32)
    bw = df["box_w"].to_numpy(dtype=np.float32)
    bh = df["box_h"].to_numpy(dtype=np.float32)
    x1 = fx - bw / 2.0
    x2 = fx + bw / 2.0
    y2 = fy
    y1 = fy - bh
    return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# tracklet kalite kapisi (confidence + temporal tutarlilik + min uzunluk)
# ---------------------------------------------------------------------------
def _tracklet_quality(
    g,
    fps: float,
    conf_thr: float,
    min_dur_s: float,
    min_conf_frac: float,
    max_box_cv: float,
    max_speed_px_per_frame: float,
    require_in_pitch: bool,
) -> dict:
    """Tek bir tracklet (tek tid'in tum satirlari) icin kabul kararini dondur.

    Kontroller (hepsi ucuz, sabit-kamera sinyallerinden):
      - sure (saniye)        >= min_dur_s
      - conf >= conf_thr olan frame orani  >= min_conf_frac
      - kutu boyutu varyasyon katsayisi (std/mean of box_h) <= max_box_cv
        (ani buyuyup-kuculen kutu = ID-switch / merge artefakti)
      - ardisik ayak-noktasi sicramasi medyani <= max_speed_px_per_frame
        (teleport = takip hatasi)
      - require_in_pitch ise: satirlarin cogu in_pitch=True (tribun/yansima FP eler)
      - bottom_cropped satirlar etiketten DUSURULUR (ayak gecersiz)

    Donen: {"accept": bool, "reasons": [...], "n_label_rows": int, "stats": {...}}
    """
    reasons: list[str] = []
    g = g.sort_values("frame")
    n = len(g)
    frames = g["frame"].to_numpy()
    dur_s = (frames[-1] - frames[0]) / fps if n > 1 else 0.0
    conf = g["conf"].to_numpy(dtype=np.float32)
    conf_frac = float((conf >= conf_thr).mean()) if n else 0.0

    bh = g["box_h"].to_numpy(dtype=np.float32)
    box_cv = float(np.std(bh) / (np.mean(bh) + 1e-6)) if n > 1 else 0.0

    fx = g["foot_x"].to_numpy(dtype=np.float32)
    fy = g["foot_y"].to_numpy(dtype=np.float32)
    step = np.hypot(np.diff(fx), np.diff(fy)) if n > 1 else np.array([0.0])
    med_step = float(np.median(step)) if step.size else 0.0

    if dur_s < min_dur_s:
        reasons.append(f"too_short({dur_s:.1f}s<{min_dur_s})")
    if conf_frac < min_conf_frac:
        reasons.append(f"low_conf_frac({conf_frac:.2f}<{min_conf_frac})")
    if box_cv > max_box_cv:
        reasons.append(f"box_jitter(cv={box_cv:.2f}>{max_box_cv})")
    if med_step > max_speed_px_per_frame:
        reasons.append(f"teleport(med_step={med_step:.1f}px>{max_speed_px_per_frame})")

    # etikete giren satirlar: kirpilmamis + (istenirse) saha-ici + conf>=thr
    row_mask = (conf >= conf_thr)
    if "bottom_cropped" in g.columns:
        row_mask &= ~g["bottom_cropped"].to_numpy(dtype=bool)
    if require_in_pitch and "in_pitch" in g.columns:
        ip = g["in_pitch"].to_numpy(dtype=bool)
        # cogu satir saha disindaysa tracklet tribun/yansima adayi
        if ip.mean() < 0.5:
            reasons.append(f"mostly_off_pitch({ip.mean():.2f})")
        row_mask &= ip
    n_label_rows = int(row_mask.sum())
    if n_label_rows == 0 and not reasons:
        reasons.append("no_valid_rows")

    return {
        "accept": len(reasons) == 0 and n_label_rows > 0,
        "reasons": reasons,
        "row_mask": row_mask,
        "n_label_rows": n_label_rows,
        "stats": {
            "n": n, "dur_s": round(dur_s, 1), "conf_frac": round(conf_frac, 3),
            "box_cv": round(box_cv, 3), "med_step_px": round(med_step, 2),
        },
    }


def detection_pseudo_labels(
    tracks_df,
    conf_thr: float = 0.6,
    min_dur_s: float = 10.0,
    *,
    fps: float | None = None,
    min_conf_frac: float = 0.7,
    max_box_cv: float = 0.45,
    max_speed_px_per_frame: float = 60.0,
    require_in_pitch: bool = False,
    img_w: int | None = None,
    img_h: int | None = None,
) -> list[dict]:
    """Yuksek-guvenli tracklet'lerden per-frame detection pseudo-label'lari uret.

    Girdi: export_tracks.py parquet'inden okunmus pandas DataFrame (track_record_schema).
    Cikti: kabul edilen her (tracklet, frame) icin bir kayit listesi. Her kayit:
        {
          "frame": int, "tid": int, "class_id": 0,
          "bbox_xyxy": [x1,y1,x2,y2],   # ham (distorted) piksel
          "conf": float,
          "img_w": int|None, "img_h": int|None,   # to_yolo/to_coco normalize icin
        }
    to_yolo() / to_coco() bu listeyi diske yazar.

    Parametreler interface spec uyumlu: conf_thr, min_dur_s pozisyonel. Geri kalan
    temporal-tutarlilik esikleri makul default'larla keyword.

    fps verilmezse DataFrame'den (frame/t_sec) tahmin edilir.
    img_w/img_h verilmezse file-level metadata'dan disaridan gecirilmeli (normalize icin).
    """
    df = tracks_df
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"tracks_df eksik kolon(lar): {missing}; "
                         f"export_tracks.py track_record_schema bekleniyor")

    if fps is None:
        fps = _infer_fps(df)

    records: list[dict] = []
    bbox_all = reconstruct_bbox_xyxy(df)
    df = df.assign(_x1=bbox_all[:, 0], _y1=bbox_all[:, 1],
                   _x2=bbox_all[:, 2], _y2=bbox_all[:, 3])

    accepted = rejected = 0
    for tid, g in df.groupby("tid", sort=False):
        q = _tracklet_quality(
            g, fps=fps, conf_thr=conf_thr, min_dur_s=min_dur_s,
            min_conf_frac=min_conf_frac, max_box_cv=max_box_cv,
            max_speed_px_per_frame=max_speed_px_per_frame,
            require_in_pitch=require_in_pitch,
        )
        if not q["accept"]:
            rejected += 1
            continue
        accepted += 1
        gg = g.sort_values("frame")
        mask = q["row_mask"]
        sub = gg[mask]
        for _, r in sub.iterrows():
            records.append({
                "frame": int(r["frame"]),
                "tid": int(tid),
                "class_id": PERSON_CLASS_ID,
                "bbox_xyxy": [float(r["_x1"]), float(r["_y1"]),
                              float(r["_x2"]), float(r["_y2"])],
                "conf": float(r["conf"]),
                "img_w": img_w, "img_h": img_h,
            })
    # ozet bilgiyi log-disi tasimak icin liste attribute'u olarak ekleyemeyiz;
    # cagiran isterse summarize_pseudo_labels() kullanir.
    records_meta = {"tracklets_accepted": accepted, "tracklets_rejected": rejected,
                    "label_rows": len(records)}
    # records listesine hafif bir metadata sarmali eklemek yerine, ilk kayitta
    # _summary tutmak kirli olur; bunun yerine modul-seviyesi son-ozet sakla.
    _LAST_SUMMARY.clear()
    _LAST_SUMMARY.update(records_meta)
    return records


_LAST_SUMMARY: dict = {}


def last_summary() -> dict:
    """En son detection_pseudo_labels() cagrisinin ozet sayaclarini dondur."""
    return dict(_LAST_SUMMARY)


def _infer_fps(df) -> float:
    """frame ve t_sec kolonlarindan fps tahmin et (export semasinda t_sec=frame/fps)."""
    if "t_sec" in df.columns:
        f = df["frame"].to_numpy(dtype=np.float64)
        t = df["t_sec"].to_numpy(dtype=np.float64)
        m = t > 0
        if m.sum() > 1:
            return float(np.median(f[m] / t[m]))
    return 25.0  # makul default (klipler ~24.9 fps)


# ---------------------------------------------------------------------------
# COCO / YOLO yazicilar
# ---------------------------------------------------------------------------
def to_yolo(records: list[dict], img_w: int, img_h: int, out_dir: str,
            class_id: int = PERSON_CLASS_ID) -> dict:
    """Detection kayitlarini YOLO txt formatinda yaz (frame basina bir .txt).

    Her satir:  <class_id> <cx_norm> <cy_norm> <w_norm> <h_norm>
    Dosya adi:  <out_dir>/frame_<frame:06d>.txt  (harvest pipeline ayni adla
    frame_<frame:06d>.jpg kaydeder).
    Donen: {"files": int, "boxes": int, "out_dir": str}.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_frame: dict[int, list[str]] = {}
    for r in records:
        x1, y1, x2, y2 = r["bbox_xyxy"]
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        # kirp [0,1]
        cx, cy, w, h = (min(max(v, 0.0), 1.0) for v in (cx, cy, w, h))
        by_frame.setdefault(int(r["frame"]), []).append(
            f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    n_boxes = 0
    for frame, lines in by_frame.items():
        (out / f"frame_{frame:06d}.txt").write_text("\n".join(lines) + "\n")
        n_boxes += len(lines)
    return {"files": len(by_frame), "boxes": n_boxes, "out_dir": str(out)}


def to_coco(records: list[dict], img_w: int, img_h: int, out_path: str,
            image_name_fmt: str = "frame_{frame:06d}.jpg",
            category_name: str = "person") -> dict:
    """Detection kayitlarini tek bir COCO instances JSON dosyasina yaz.

    images[].file_name = image_name_fmt.format(frame=frame). Harvest pipeline
    cercevleri ayni isimle kaydetmeli. bbox COCO formati [x,y,w,h] (ham piksel).
    Donen: {"images": int, "annotations": int, "out_path": str}.
    """
    frames = sorted({int(r["frame"]) for r in records})
    frame_to_imgid = {f: i + 1 for i, f in enumerate(frames)}
    images = [{
        "id": frame_to_imgid[f], "file_name": image_name_fmt.format(frame=f),
        "width": int(img_w), "height": int(img_h),
    } for f in frames]
    annotations = []
    for ann_id, r in enumerate(records, start=1):
        x1, y1, x2, y2 = r["bbox_xyxy"]
        w = max(x2 - x1, 0.0)
        h = max(y2 - y1, 0.0)
        annotations.append({
            "id": ann_id,
            "image_id": frame_to_imgid[int(r["frame"])],
            "category_id": 1,
            "bbox": [float(x1), float(y1), float(w), float(h)],
            "area": float(w * h),
            "iscrowd": 0,
            "score": float(r["conf"]),  # pseudo-label guven skoru (gostergeyle)
            "attributes": {"tid": int(r["tid"]), "pseudo": True},
        })
    coco = {
        "info": {"description": "halisaha pseudo-labels (DEFERRED self-training)",
                 "pseudo_label": True},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": category_name, "supercategory": "person"}],
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(coco))
    return {"images": len(images), "annotations": len(annotations),
            "out_path": str(out_path)}


# ---------------------------------------------------------------------------
# B) LINE-SEGMENTATION GT (manuel/kabul H'den)
# ---------------------------------------------------------------------------
def line_seg_labels_from_H(homo: Any, img_shape: tuple[int, int],
                           line_thickness_px: int = 3) -> np.ndarray:
    """Kabul/manuel homografiden yogun cizgi-segmentasyon GT maskesi uret.

    Parametrik saha sablonunun line_segments_m'i H_pitch2img ile goruntuye yansitilir
    ve cizilir. Donen: (H, W) uint8 ikili maske (1 = saha cizgisi pikseli).

    `homo` PitchHomography ornegi olmali (pitch/homography.py):
        - homo.template : line_segments_m / line_points_m saglar
        - homo.pitch_to_pixel(pts_m) : (N,2) metre -> ham/undist piksel
    Bu fonksiyon B2B primary path'te DEGIL; sadece DEFERRED seg-egitim korpusu icin.
    cv2 lazy import (CPU).
    """
    import cv2  # lazy; sadece burada gerekli

    h, w = int(img_shape[0]), int(img_shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)

    template = getattr(homo, "template", None)
    if template is None:
        raise ValueError("homo.template yok; PitchHomography ornegi bekleniyor")

    segs = getattr(template, "line_segments_m", None) or []
    if not segs:
        # segment yoksa yogun cizgi noktalarini kullan (fallback)
        if hasattr(template, "line_points_m"):
            pts_m = np.asarray(template.line_points_m(), dtype=np.float64)
            px = np.asarray(homo.pitch_to_pixel(pts_m), dtype=np.float64)
            for (u, v) in px:
                if np.isfinite(u) and np.isfinite(v):
                    cv2.circle(mask, (int(round(u)), int(round(v))),
                               max(line_thickness_px // 2, 1), 1, -1)
        return mask

    for (p0_m, p1_m) in segs:
        ends_m = np.asarray([p0_m, p1_m], dtype=np.float64)
        px = np.asarray(homo.pitch_to_pixel(ends_m), dtype=np.float64)
        if not np.all(np.isfinite(px)):
            continue
        (u0, v0), (u1, v1) = px
        cv2.line(mask, (int(round(u0)), int(round(v0))),
                 (int(round(u1)), int(round(v1))), 1, line_thickness_px,
                 lineType=cv2.LINE_AA)
    mask = (mask > 0).astype(np.uint8)
    return mask


def line_seg_corpus_from_calib(calib_path: str, frame_index: Iterable[int],
                               img_shape: tuple[int, int], out_dir: str,
                               line_thickness_px: int = 3) -> dict:
    """Tek bir kalibrasyon icin, secilen frame'lere ayni cizgi-GT maskesini yaz.

    Kamera SABIT oldugundan H tum mac boyunca degismez -> tek maske, frame'lere
    sembolik kopya/yaz. Harvest pipeline ayni frame'lerin .jpg'lerini kaydeder.
    DEFERRED; PitchHomography.load mevcut oldugunda calisir (pitch/homography.py).
    Donen: {"frames": int, "out_dir": str, "calib": str}.
    """
    import cv2

    # lazy: pitch paketi bu asamada mevcut olmali
    try:
        from pitch.homography import PitchHomography
    except Exception as e:  # pragma: no cover - pitch henuz yazilmamis olabilir
        raise RuntimeError(
            "pitch/homography.py yok ya da import edilemiyor; line-seg korpus "
            "kalibrasyon paketinden sonra calisir") from e

    homo = PitchHomography.load(calib_path)
    mask = line_seg_labels_from_H(homo, img_shape, line_thickness_px)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in frame_index:
        cv2.imwrite(str(out / f"frame_{int(f):06d}_lineseg.png"), mask * 255)
        n += 1
    return {"frames": n, "out_dir": str(out), "calib": str(calib_path)}
