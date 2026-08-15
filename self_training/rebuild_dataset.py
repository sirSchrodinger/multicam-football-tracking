#!/usr/bin/env python3
"""rebuild_dataset.py — çok-tesis COCO dataset'i YENİDEN kur (8 tesis).

Mevcut (Çankaya tiling-export + Kıbrıs) + 6 YENİ harvest tesisi birleştirir.
Tesis-split korunur (held-out tesis = overfit guard). Far/küçük-kutu (uzak,
zor recall) oranı ölçülür ve stats'a eklenir.

Önce CPU-ucuz yield taraması (tracklet/label sayısı, far oranı) -> val seçimi ->
sonra build_dataset.build() ile görüntü çıkarımı + COCO yazımı.

DÜRÜST: pseudo-label gürültülü; far oranı = kutu-yüksekliği/konum PROXY'si
(per-venue kalibrasyon yok), recall İDDİASI DEĞİL. Çankaya far-band tiling'li,
yeni tesisler base-only (no-calib) -> not edilir.
"""
import sys, os, json
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from self_training.pseudo_label import detection_pseudo_labels, last_summary, reconstruct_bbox_xyxy  # noqa
from self_training import build_dataset as bd  # noqa

RAW = Path("self_training/data/raw")
OUT = Path("self_training/data/coco")
REPORT = Path(os.environ.get("HARVEST_REPORT_DIR", "scratchpad/wf_dataset"))
FAR_BOX_H = 70.0           # px; box_h < bu = uzak/küçük oyuncu (zor recall) PROXY
FAR_BAND = (110.0, 365.0)  # foot_y bandı (Çankaya-kalib; diğer tesiste rough proxy)

# (name, tracks_parquet, video, resolution, far_mode)
# Çankaya = özel (raw/ altında, tiling-export'lu). Diğer her şey self_training/data/raw/
# içindeki tracks_*.parquet'ten OTO-KEŞİF (harvest/probe ne eklerse dahil olur).
TILING_VENUES = {"cankaya_active", "KıbrısDorukHalıS"}  # per-venue calib var, far-tiling'li
SOURCES = [dict(name="cankaya_active", tracks="raw/tracks_active_game.parquet",
                video="raw/_active_game.mp4", far_mode="tiling")]
for tp in sorted(RAW.glob("tracks_*.parquet")):
    name = tp.stem[len("tracks_"):]
    vid = RAW / f"{name}.mp4"
    if not vid.exists():
        continue
    SOURCES.append(dict(name=name, tracks=str(tp), video=str(vid),
                        far_mode="tiling" if name in TILING_VENUES else "base"))
for s in SOURCES:
    s.setdefault("res", "?")


SMALL_NORM = 0.06   # box_h/H < bu = uzak/küçük oyuncu (çözünürlükten BAĞIMSIZ, zor recall)


def far_stats(df, img_h, conf_thr=0.3):
    """Kabul edilen tracklet'lerin etiket kutularında uzak/küçük oran (H-normalize).

    box_h MUTLAK piksel çözünürlüğe bağlı (720p'de yakın oyuncu da küçük). Bu yüzden
    box_h/img_h ile NORMALIZE: small = box_h/H < 0.06 -> çözünürlükten bağımsız
    'uzak/küçük oyuncu' (zor recall) proxy'si.
    """
    recs = detection_pseudo_labels(df, conf_thr=conf_thr, min_dur_s=5.0,
                                   min_conf_frac=0.4, max_box_cv=0.5,
                                   max_speed_px_per_frame=60.0, require_in_pitch=False)
    summ = last_summary()  # tracklets_accepted/rejected + label_rows
    if not recs:
        return dict(**summ, small_frac=0.0, med_box_h_norm=0.0, med_box_h=0.0, n_frames=0)
    bx = np.array([r["bbox_xyxy"] for r in recs], float)
    box_h = bx[:, 3] - bx[:, 1]
    hn = box_h / float(img_h)
    return dict(**summ, small_frac=round(float((hn < SMALL_NORM).mean()), 3),
                med_box_h_norm=round(float(np.median(hn)), 4),
                med_box_h=round(float(np.median(box_h)), 1),
                n_frames=len({r["frame"] for r in recs}))


def _video_hw(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    return h, w


def main():
    # 1) CPU-ucuz yield taraması (img_h video'dan -> H-normalize küçük-kutu oranı)
    yields = {}
    print("=== per-venue yield taraması (CPU) ===")
    for s in SOURCES:
        tp = Path(s["tracks"])
        if not tp.exists():
            print(f"  {s['name']}: tracks YOK ({tp}) -> atla"); continue
        df = pd.read_parquet(tp)
        if not len(df):
            print(f"  {s['name']}: 0 satir -> atla"); continue
        H, Wv = _video_hw(s["video"])
        s["res"] = f"{Wv}x{H}"
        fs = far_stats(df, img_h=H)
        fs["img_h"] = H
        yields[s["name"]] = fs
        print(f"  {s['name']:18s} {s['res']:9s} far_mode={s['far_mode']:6s} "
              f"trk_acc={fs['tracklets_accepted']:3d} rows={fs['label_rows']:6d} "
              f"frames={fs['n_frames']:4d} small_frac={fs['small_frac']:.2f} "
              f"med_h/H={fs['med_box_h_norm']:.3f}")

    usable = [s for s in SOURCES if yields.get(s["name"], {}).get("tracklets_accepted", 0) > 0]
    print(f"\nkullanılabilir tesis: {[s['name'] for s in usable]}")

    # 2) val seçimi: held-out tesis = overfit guard. Kıbrıs (kanıtlı, görünmeyen
    #    tesis) val'de tutulur; TÜM yeni tesisler + Çankaya train'de (en zengin
    #    far-domain veriyi -Aydınoğlu- val'e harcamamak için). Kıbrıs yoksa en
    #    küçük yield'li yeni tesisi held-out yap.
    usable_names = {s["name"] for s in usable}
    if "KıbrısDorukHalıS" in usable_names:
        val_venues = ["KıbrısDorukHalıS"]
    else:
        cands = [s["name"] for s in usable if s["name"] != "cankaya_active"]
        cands.sort(key=lambda n: yields[n]["label_rows"])  # en küçük held-out
        val_venues = cands[:1] if cands else []
    print(f"val (held-out tesis, overfit guard): {val_venues}")

    # 3) görüntü çıkarımı + COCO (build_dataset.build, stride seyrek örnek)
    stride = int(os.environ.get("STRIDE", "20"))
    print(f"\n=== build (stride={stride}) ===")
    stats = bd.build(usable, str(OUT), stride=stride, conf_thr=0.3, val_venues=val_venues)

    # 4) küçük-kutu (uzak) analizini stats'a ekle (COCO'dan, per-image H ile normalize)
    def coco_far(path):
        c = json.loads(Path(path).read_text())
        if not c["annotations"]:
            return dict(boxes=0, small_frac=0.0, med_box_h_norm=0.0, med_box_h=0.0)
        ih = {im["id"]: im["height"] for im in c["images"]}
        h = np.array([a["bbox"][3] for a in c["annotations"]], float)
        hn = np.array([a["bbox"][3] / ih[a["image_id"]] for a in c["annotations"]], float)
        return dict(boxes=len(h), small_frac=round(float((hn < SMALL_NORM).mean()), 3),
                    med_box_h_norm=round(float(np.median(hn)), 4),
                    med_box_h=round(float(np.median(h)), 1))
    far_analysis = dict(train=coco_far(OUT / "train.json"),
                        val=coco_far(OUT / "val.json"),
                        small_norm_thr=SMALL_NORM,
                        note="small = box_h/H < 0.06 (uzak/küçük oyuncu, çözünürlük-NORMALİZE) "
                             "PROXY; recall İDDİASI DEĞİL. Çankaya+Kıbrıs far-band tiling'li, "
                             "yeni tesisler base-only (no-calib).")
    # res/far_mode meta'sı
    venue_meta = {s["name"]: dict(res=s["res"], far_mode=s["far_mode"]) for s in SOURCES}
    stats["far_analysis"] = far_analysis
    stats["venue_meta"] = venue_meta
    stats["n_venues_usable"] = len(usable)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print("\n=== STATS ===")
    print(json.dumps(dict(total_images=stats["total_images"], total_boxes=stats["total_boxes"],
                          train_venues=stats["train_venues"], val_venues=stats["val_venues"],
                          far_analysis=far_analysis), indent=2, ensure_ascii=False))
    # rebuild özeti REPORT'a
    (REPORT / "rebuild_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    (REPORT / "yields.json").write_text(json.dumps(yields, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
