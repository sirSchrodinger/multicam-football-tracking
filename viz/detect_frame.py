#!/usr/bin/env python3
"""Tek-kare oyuncu tespiti — RF-DETR SoccerNet, 4GB-GPU dostu, tekrar-kullanılabilir.

Model bir kez yüklenir (FrameDetector), sonra frame-frame çağrılır. GPU OOM'da
otomatik CPU'ya düşer. Ham BGR kare -> (boxes xyxy, feet, conf, cls) döner.
Sınıflar: 0=ball 1=player 2=referee 3=goalkeeper. Varsayılan: player+goalkeeper.
"""
from __future__ import annotations

import numpy as np


class FrameDetector:
    def __init__(self, weights="models/weights/checkpoint_best_regular.pth",
                 device="auto", min_h=18):
        import torch
        from rfdetr import RFDETRLargeDeprecated
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.torch = torch
        self.min_h = min_h
        # export_tracks.py ile BİREBİR: Large-deprecated, num_classes=4
        self.model = RFDETRLargeDeprecated(
            pretrain_weights=weights, device=device, num_classes=4)

    def detect(self, frame_bgr, threshold=0.5, nms=0.5):
        """Ham BGR kare -> dict(boxes(N,4), feet(N,2), conf(N,), cls(N,)).

        export_tracks.py güvenli ball-drop kuralı: cls==0 & h/w<1.6 = top -> at.
        Uzun yanlış-ball-etiketli kutu (gerçek oyuncu) korunur. GPU OOM -> CPU.
        """
        from PIL import Image
        pil = Image.fromarray(frame_bgr[:, :, ::-1])
        try:
            det = self.model.predict(pil, threshold=threshold)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and self.device == "cuda":
                self.torch.cuda.empty_cache()
                self.model.model.model.to("cpu"); self.device = "cpu"
                det = self.model.predict(pil, threshold=threshold)
            else:
                raise
        if det is None or len(det.xyxy) == 0:
            return dict(boxes=np.empty((0, 4)), feet=np.empty((0, 2)),
                        conf=np.empty(0), cls=np.empty(0, int))
        xy = np.asarray(det.xyxy, float)
        _c = getattr(det, "class_id", None)
        cls = (np.asarray(_c, int).reshape(-1) if _c is not None
               else np.ones(len(xy), int))
        conf = np.asarray(det.confidence, float)
        hh = xy[:, 3] - xy[:, 1]
        ww = np.maximum(xy[:, 2] - xy[:, 0], 1e-6)
        ball_like = (cls == 0) & ((hh / ww) < 1.6)
        keep = (hh > self.min_h) & ~ball_like
        xy, cls, conf = xy[keep], cls[keep], conf[keep]
        # basit sınıf-agnostik NMS
        order = conf.argsort()[::-1]
        picked = []
        for i in order:
            ok = True
            for j in picked:
                xx1 = max(xy[i, 0], xy[j, 0]); yy1 = max(xy[i, 1], xy[j, 1])
                xx2 = min(xy[i, 2], xy[j, 2]); yy2 = min(xy[i, 3], xy[j, 3])
                iw = max(0, xx2 - xx1); ih = max(0, yy2 - yy1)
                inter = iw * ih
                a_i = (xy[i, 2] - xy[i, 0]) * (xy[i, 3] - xy[i, 1])
                a_j = (xy[j, 2] - xy[j, 0]) * (xy[j, 3] - xy[j, 1])
                if inter / (a_i + a_j - inter + 1e-6) > nms:
                    ok = False; break
            if ok:
                picked.append(i)
        picked = np.array(sorted(picked)) if picked else np.array([], int)
        xy, cls, conf = xy[picked], cls[picked], conf[picked]
        feet = np.c_[(xy[:, 0] + xy[:, 2]) / 2.0, xy[:, 3]] if len(xy) else np.empty((0, 2))
        return dict(boxes=xy, feet=feet, conf=conf, cls=cls)


if __name__ == "__main__":
    import sys
    import time
    import cv2
    p = sys.argv[1] if len(sys.argv) > 1 else "calib/cankaya_cam2_frame_raw.png"
    im = cv2.imread(p)
    t0 = time.time()
    d = FrameDetector()
    t1 = time.time()
    r = d.detect(im)
    t2 = time.time()
    print(f"yükleme {t1-t0:.1f}s · çıkarım {t2-t1:.2f}s · device={d.device}")
    print(f"{len(r['boxes'])} oyuncu (player+gk), conf med "
          f"{np.median(r['conf']) if len(r['conf']) else float('nan'):.2f}")
