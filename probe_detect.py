#!/usr/bin/env python3
"""Week-1 GO/NO-GO probe: far-half detection recall on sosyalhalisaha footage.

Extracts frames from downloaded match video(s), runs RF-DETR SoccerNet
checkpoint, saves annotated frames + a per-frame detection report.

Usage: python probe_detect.py <video.mp4> [--times 300,900,1500,2100] [--device auto]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


def extract_frames(video_path, times_s, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dur = n / fps if fps else 0
    print(f"video: {video_path.name} fps={fps:.2f} frames={n:.0f} dur={dur/60:.1f}min")
    frames = []
    for t in times_s:
        if dur and t >= dur:
            print(f"  skip t={t}s (past end)")
            continue
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, img = cap.read()
        if not ok:
            print(f"  read fail t={t}s")
            continue
        p = out_dir / f"{video_path.stem}_t{t:05d}.jpg"
        cv2.imwrite(str(p), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        frames.append((t, p, img.shape))
        print(f"  extracted t={t}s -> {p.name} {img.shape}")
    cap.release()
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--times", default="300,600,900,1200,1500,1800,2400,3000")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threshold", type=float, default=0.4)
    ap.add_argument("--resolution", type=int, default=0, help="override model resolution (0=default)")
    args = ap.parse_args()

    video = Path(args.video)
    times = [int(x) for x in args.times.split(",")]
    frames_dir = Path(__file__).parent / "frames"
    ann_dir = Path(__file__).parent / "annotated"
    ann_dir.mkdir(exist_ok=True)

    frames = extract_frames(video, times, frames_dir)
    if not frames:
        sys.exit("no frames extracted")

    import torch
    from rfdetr import RFDETRLargeDeprecated
    import supervision as sv
    from PIL import Image

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    kw = {"pretrain_weights": str(Path(__file__).parent / "models/weights/checkpoint_best_regular.pth"),
          "device": device, "num_classes": 4}
    if args.resolution:
        kw["resolution"] = args.resolution
    print(f"loading RF-DETR-Large(deprecated arch) on {device} kw={ {k: v for k, v in kw.items() if k != 'pretrain_weights'} }")
    model = RFDETRLargeDeprecated(**kw)

    CLASS_NAMES = ["ball", "player", "referee", "goalkeeper"]
    box_annot = sv.BoxAnnotator(thickness=1)
    label_annot = sv.LabelAnnotator(text_scale=0.3, text_thickness=1, text_padding=1)

    report = []
    for t, p, shape in frames:
        img = Image.open(p)
        det = model.predict(img, threshold=args.threshold)
        h = shape[0]
        ys = det.xyxy[:, 3]  # bottom edge = feet position
        heights = det.xyxy[:, 3] - det.xyxy[:, 1]
        is_person = det.class_id != 0
        far = is_person & (ys < h * 0.45)   # upper image band ~ far half
        near = is_person & (ys >= h * 0.45)
        rec = {
            "t": t, "file": p.name,
            "n_total": int(is_person.sum()),
            "n_far": int(far.sum()), "n_near": int(near.sum()),
            "far_heights_px": [round(float(x), 1) for x in heights[far]],
            "near_heights_px": [round(float(x), 1) for x in heights[near]],
            "confidences": [round(float(c), 2) for c in det.confidence[is_person]],
        }
        report.append(rec)
        print(f"t={t:5d}s persons={rec['n_total']:2d} far={rec['n_far']:2d} near={rec['n_near']:2d} "
              f"far_h_px={rec['far_heights_px']}")
        labels = [f"{CLASS_NAMES[c]} {conf:.2f}" for c, conf in zip(det.class_id, det.confidence)]
        ann = np.array(img.convert("RGB"))[:, :, ::-1].copy()
        ann = box_annot.annotate(ann, det)
        ann = label_annot.annotate(ann, det, labels)
        cv2.imwrite(str(ann_dir / p.name), ann, [cv2.IMWRITE_JPEG_QUALITY, 95])

    out = Path(__file__).parent / f"probe_report_{video.stem}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"report -> {out}")


if __name__ == "__main__":
    main()
