#!/usr/bin/env python3
"""Tam-maç örneklenmiş yoğunluk pass'i (detect-only, takipsiz).

Heatmap için TAKİP gerekmez — sadece tespit ayak-noktaları. Bu yüzden her frame
yerine her STRIDE'inci frame işlenir (oyuncu ~1-2 m/sn; 1 sn örnekleme yoğunluk
için fazlasıyla yeter). Tüm maç boyunca örnekleyince oyun tüm sahaya yayılır →
2 dakikalık tek-faz dilimin tek-yanlı heatmap'i sorunu kalkar.

Tespit yığını track_smoke.py / export_tracks.py ile BİRE-BİR aynı (RFDETRLargeDeprecated,
threshold 0.3, min_h 25, nms 0.6 class_agnostic). Çıktı: foot-point npz (frame, t_sec,
foot_x, foot_y, box_h, box_w, conf). Heatmap render'ı AYRI (GPU'suz, yeniden çalıştırılabilir).

CLI:
  venv/bin/python heatmap_match.py <full_match.mp4> [--stride 25] [--out points.npz]
                                   [--weights ...] [--max-samples N]
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np

DEFAULT_WEIGHTS = "models/weights/checkpoint_best_regular.pth"
THRESH, MIN_H, NMS_THRESH = 0.3, 25, 0.6


def run(video, stride=25, out=None, weights=DEFAULT_WEIGHTS, max_samples=None):
    import cv2
    from PIL import Image
    import supervision as sv
    from rfdetr import RFDETRLargeDeprecated

    vid = Path(video)
    cap = cv2.VideoCapture(str(vid))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{vid.name}: {n} frame, fps={fps:.1f}, stride={stride} "
          f"-> ~{n//stride} ornek", flush=True)

    model = RFDETRLargeDeprecated(pretrain_weights=weights, device="cuda", num_classes=4)

    rows = []  # (frame, t_sec, foot_x, foot_y, box_h, box_w, conf)
    i, done, t0 = 0, 0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % stride == 0:
            det = model.predict(Image.fromarray(frame[:, :, ::-1]), threshold=THRESH)
            hh = det.xyxy[:, 3] - det.xyxy[:, 1]
            keep = hh > MIN_H
            d = sv.Detections(xyxy=det.xyxy[keep], confidence=det.confidence[keep],
                              class_id=np.zeros(int(keep.sum()), dtype=int))
            d = d.with_nms(threshold=NMS_THRESH, class_agnostic=True)
            for (x1, y1, x2, y2), c in zip(d.xyxy, d.confidence):
                rows.append((i, i / fps, (x1 + x2) / 2.0, float(y2),
                             float(y2 - y1), float(x2 - x1), float(c)))
            done += 1
            if done % 100 == 0:
                el = time.time() - t0
                print(f"  ornek {done}/{n//stride} ({done/el:.1f} fps, "
                      f"eta {((n//stride)-done)/(done/el)/60:.1f}min, "
                      f"{len(rows)} tespit)", flush=True)
            if max_samples and done >= max_samples:
                break
        i += 1
    cap.release()

    arr = np.array(rows, dtype=np.float32)
    out = out or str(vid.parent / f"density_{vid.stem}.npz")
    np.savez_compressed(out, points=arr,
                        columns=np.array(["frame", "t_sec", "foot_x", "foot_y",
                                          "box_h", "box_w", "conf"]),
                        fps=np.float32(fps), n_frames=np.int64(n),
                        stride=np.int64(stride))
    print(f"\n{len(rows)} tespit, {done} ornek frame -> {out}", flush=True)
    print(f"foot_y {arr[:,3].min():.0f}-{arr[:,3].max():.0f}, "
          f"box_h {arr[:,4].min():.0f}-{arr[:,4].max():.0f}, "
          f"sure {(time.time()-t0)/60:.1f}dk", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Tam-mac orneklenmis yogunluk (detect-only).")
    ap.add_argument("video")
    ap.add_argument("--stride", type=int, default=25, help="her N'inci frame (default 25 ~1sn)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--weights", default=DEFAULT_WEIGHTS)
    ap.add_argument("--max-samples", dest="max_samples", type=int, default=None)
    a = ap.parse_args(argv if argv is not None else sys.argv[1:])
    run(a.video, a.stride, a.out, a.weights, a.max_samples)


if __name__ == "__main__":
    main()
