#!/usr/bin/env python3
"""Tracking smoke test: RF-DETR + ByteTrack on a short clip, native fps.

Outputs:
  dets_<clip>.npz        per-frame detections (xyxy, conf)
  tracks_<clip>.json     per-track stats (frames, px path length, mean height)
  overlay_<clip>.mp4     first N seconds with track-id overlay
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import supervision as sv
from rfdetr import RFDETRLargeDeprecated

CLIP = Path(sys.argv[1] if len(sys.argv) > 1 else "raw/cankaya_cam2_clip2400.mp4")
THRESH = 0.3
MIN_H = 25
OVERLAY_SECONDS = 30

cap = cv2.VideoCapture(str(CLIP))
fps = cap.get(cv2.CAP_PROP_FPS)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"clip {CLIP.name}: {W}x{H} fps={fps:.2f} frames={n_frames}", flush=True)

model = RFDETRLargeDeprecated(
    pretrain_weights="models/weights/checkpoint_best_regular.pth",
    device="cuda", num_classes=4)

tracker = sv.ByteTrack(frame_rate=int(round(fps)),
                       track_activation_threshold=0.25,
                       lost_track_buffer=int(round(fps)) * 2,
                       minimum_matching_threshold=0.8)

box_a = sv.BoxAnnotator(thickness=2)
lab_a = sv.LabelAnnotator(text_scale=0.5, text_thickness=1, text_padding=2)
trace_a = sv.TraceAnnotator(thickness=2, trace_length=int(fps) * 3)

writer = cv2.VideoWriter(str(CLIP.parent / f"overlay_{CLIP.stem}.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
overlay_frames = int(OVERLAY_SECONDS * fps)

all_dets = []
tracks = defaultdict(list)  # tid -> [(frame_idx, foot_x, foot_y, h)]
t0 = time.time()
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    det = model.predict(Image.fromarray(frame[:, :, ::-1]), threshold=THRESH)
    hh = det.xyxy[:, 3] - det.xyxy[:, 1]
    keep = hh > MIN_H
    d = sv.Detections(xyxy=det.xyxy[keep], confidence=det.confidence[keep],
                      class_id=np.zeros(int(keep.sum()), dtype=int))
    d = d.with_nms(threshold=0.6, class_agnostic=True)
    all_dets.append((d.xyxy.copy(), d.confidence.copy()))
    d = tracker.update_with_detections(d)
    for xyxy, tid in zip(d.xyxy, d.tracker_id):
        foot = ((xyxy[0] + xyxy[2]) / 2, xyxy[3])
        tracks[int(tid)].append((i, float(foot[0]), float(foot[1]),
                                 float(xyxy[3] - xyxy[1])))
    if i < overlay_frames:
        ann = frame.copy()
        ann = trace_a.annotate(ann, d)
        ann = box_a.annotate(ann, d)
        ann = lab_a.annotate(ann, d, [f"#{t}" for t in d.tracker_id])
        writer.write(ann)
    i += 1
    if i % 200 == 0:
        el = time.time() - t0
        print(f"  {i}/{n_frames} ({i/el:.1f} fps proc, eta {(n_frames-i)/(i/el)/60:.1f}min)", flush=True)
writer.release()
cap.release()

np.savez_compressed(
    CLIP.parent / f"dets_{CLIP.stem}.npz",
    xyxy=np.array([x for x, _ in all_dets], dtype=object),
    conf=np.array([c for _, c in all_dets], dtype=object),
    allow_pickle=True)

# track istatistikleri
stats = []
for tid, pts in tracks.items():
    pts = sorted(pts)
    arr = np.array([(x, y) for _, x, y, _ in pts])
    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    stats.append(dict(
        tid=tid, n=len(pts),
        dur_s=round((pts[-1][0] - pts[0][0]) / fps, 1),
        px_path=round(float(seg.sum()), 0),
        mean_h=round(float(np.mean([h for _, _, _, h in pts])), 0),
    ))
stats.sort(key=lambda s: -s["n"])
out = CLIP.parent / f"tracks_{CLIP.stem}.json"
out.write_text(json.dumps(stats, indent=1))

long_tracks = [s for s in stats if s["dur_s"] >= 10]
print(f"\nTOPLAM {len(stats)} track; >=10sn yaşayan: {len(long_tracks)}")
for s in stats[:20]:
    print(s)
print(f"-> {out}")
print(f"süre: {(time.time()-t0)/60:.1f} dk")
