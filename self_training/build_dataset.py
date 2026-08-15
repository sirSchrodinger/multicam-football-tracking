#!/usr/bin/env python3
"""build_dataset.py — pseudo-label COCO dataset'i kur (RF-DETR fine-tune için).

Tasarım (overfit-siz + temiz-label):
  - KAYNAK çok-tesis (her tesis ayrı kamera) -> model Çankaya'ya overfit olmaz.
  - LABEL = track-doğrulanmış kutular (pseudo_label.detection_pseudo_labels:
    tracklet süre/box-cv/TELEPORT/in-pitch kapısı). conf-eşik düşük (0.3) ama
    tracklet-kapısı sıkı -> tiling-kurtarılan FAR oyuncular (base'in kaçırdığı zor
    durumlar = öğretmek istediğimiz) dahil; tek-kare gürültü/FP elenir.
  - TRAIN/VAL split TESİS-bazlı (held-out venue = generalization ölçümü, overfit
    guard). Tek tesis varsa hepsi train; harvest gelince held-out val.
Çıktı: <out>/images/*.jpg + train.json + val.json (COCO) + stats.json.
Lisans: pandas+cv2. GPU yok (fine-tune ayrı, RunPod).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import cv2
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from self_training.pseudo_label import detection_pseudo_labels, last_summary


def _extract_and_label(name, tracks_path, video, img_dir, stride, conf_thr):
    df = pd.read_parquet(tracks_path)
    cap = cv2.VideoCapture(str(video))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    recs = detection_pseudo_labels(df, conf_thr=conf_thr, min_dur_s=5.0,
                                   min_conf_frac=0.4, max_box_cv=0.5,
                                   max_speed_px_per_frame=60.0, require_in_pitch=False,
                                   img_w=W, img_h=H)
    summ = last_summary()
    by_frame = {}
    for r in recs:
        by_frame.setdefault(r["frame"], []).append(r)
    frames = sorted(by_frame)[::stride]          # seyrek örnek (çeşitlilik + boyut)
    images, anns = [], []
    img_dir.mkdir(parents=True, exist_ok=True)
    for fi in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, fr = cap.read()
        if not ok:
            continue
        fn = f"{name}_f{fi:06d}.jpg"
        cv2.imwrite(str(img_dir / fn), fr)
        images.append(dict(file_name=fn, width=W, height=H, _frame=int(fi)))
        for r in by_frame[fi]:
            x1, y1, x2, y2 = r["bbox_xyxy"]
            anns.append(dict(_img=fn, bbox=[float(x1), float(y1),
                             float(max(x2 - x1, 1)), float(max(y2 - y1, 1))],
                             score=r["conf"]))
    cap.release()
    return images, anns, dict(venue=name, **summ, n_images=len(images), n_boxes=len(anns))


def build(sources, out_dir, stride=25, conf_thr=0.3, val_venues=None):
    out = Path(out_dir); (out / "images").mkdir(parents=True, exist_ok=True)
    val_venues = set(val_venues or [])
    per = {}
    for s in sources:
        imgs, anns, st = _extract_and_label(s["name"], s["tracks"], s["video"],
                                            out / "images", stride, conf_thr)
        per[s["name"]] = (imgs, anns, st)
        print(f"  {s['name']}: {st['n_images']} kare, {st['n_boxes']} kutu "
              f"({st.get('tracklets_accepted','?')} tracklet kabul)")

    def to_coco(names):
        images, annotations = [], []; iid = aid = 1; nmeta = {}
        for nm in names:
            if nm not in per:
                continue
            imgs, anns, _ = per[nm]
            for im in imgs:
                nmeta[im["file_name"]] = iid
                images.append(dict(id=iid, file_name=im["file_name"],
                                   width=im["width"], height=im["height"])); iid += 1
            for a in anns:
                images_id = nmeta.get(a["_img"])
                if images_id is None:
                    continue
                x, y, w, h = a["bbox"]
                annotations.append(dict(id=aid, image_id=images_id, category_id=1,
                                        bbox=[x, y, w, h], area=w * h, iscrowd=0,
                                        score=a["score"])); aid += 1
        return dict(info=dict(description="halisaha pseudo-labels (multi-venue)"),
                    images=images, annotations=annotations,
                    categories=[dict(id=1, name="person")])

    all_names = [s["name"] for s in sources]
    train_names = [n for n in all_names if n not in val_venues]
    val_names = [n for n in all_names if n in val_venues] or train_names[-1:]  # en az 1 val
    (out / "train.json").write_text(json.dumps(to_coco(train_names)))
    (out / "val.json").write_text(json.dumps(to_coco(val_names)))
    stats = dict(per_venue={k: v[2] for k, v in per.items()},
                 train_venues=train_names, val_venues=val_names,
                 total_images=sum(v[2]["n_images"] for v in per.values()),
                 total_boxes=sum(v[2]["n_boxes"] for v in per.values()),
                 design="track-validated pseudo-labels; venue-split (overfit guard); "
                        "low conf_thr+strict tracklet gate (includes tiling far players)")
    (out / "stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    return stats


if __name__ == "__main__":
    # şimdilik mevcut tek tesis (Çankaya aktif-oyun, tiling export'lu); harvest gelince eklenir
    srcs = [dict(name="cankaya_active", tracks="raw/tracks_active_game.parquet",
                 video="raw/_active_game.mp4")]
    st = build(srcs, "self_training/data/coco", stride=20)
    print(json.dumps(st, indent=2, ensure_ascii=False))
