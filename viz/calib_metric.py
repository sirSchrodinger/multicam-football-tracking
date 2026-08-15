#!/usr/bin/env python3
"""Kalibrasyon çizgi-artığı — öz-geliştirmenin SAYISAL hedefi (metre).

İki yön (ikisi de metre, gerçek birim):
  template_support_m  (BİRİNCİL) — her template çizgi-noktasının EN YAKIN tespit-
      edilen beyaz-çizgi pikseline metre mesafesi. Küçük = template çizgileri
      gerçekten tespit-edilen çizginin olduğu yerde => doğru H. Fazladan boyalı
      çizgilere (ceza sahası vb, template'te yok) DAYANIKLI.
  detection_explained_m (İKİNCİL) — tespit-edilen her beyaz pikselin en yakın
      template çizgisine mesafesi. Büyükse: ya kötü H ya modellenmemiş çizgi çok.

drift_check ile aynı ruh ama metre-domeninde ve iki-yönlü. Loop bunu KÜÇÜLTÜR.
Sadece numpy — scipy/GPU yok.
"""
from __future__ import annotations

import numpy as np


def _sample_segments_m(segs, circle=None, step=0.20):
    """Template segment + orta-yuvarlağı yoğun metre-noktalarına örnekle."""
    pts = []
    for (p0, p1) in segs:
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
        d = float(np.linalg.norm(p1 - p0))
        n = max(2, int(d / step))
        for t in np.linspace(0, 1, n):
            pts.append(p0 + t * (p1 - p0))
    if circle is not None:
        cx, cy, r = circle
        n = max(12, int(2 * np.pi * r / step))
        for a in np.linspace(0, 2 * np.pi, n, endpoint=False):
            pts.append([cx + r * np.cos(a), cy + r * np.sin(a)])
    return np.asarray(pts, float) if pts else np.empty((0, 2))


def _nearest(A, B):
    """A (N,2) her noktasının B (M,2) içindeki en yakın komşusuna mesafesi (N,)."""
    if len(A) == 0 or len(B) == 0:
        return np.full(len(A), np.nan)
    out = np.empty(len(A))
    # bloklu (bellek güvenli): 512'lik parçalar
    for i in range(0, len(A), 512):
        blk = A[i:i + 512]
        d = np.linalg.norm(blk[:, None, :] - B[None, :, :], axis=2)
        out[i:i + 512] = d.min(axis=1)
    return out


def line_residual_m(homo, dl, L=None, W=None, circle_r=3.0,
                    prob_thr=0.5, max_det=3000, margin_m=1.0):
    """Çizgi-artığı (metre). homo: PitchHomography, dl: DetectedLines.

    Döner: dict(template_support_m, detection_explained_m, coverage_0p5,
                n_det, n_tmpl). Kalibrasyon yoksa NaN.
    """
    out = dict(template_support_m=float("nan"), detection_explained_m=float("nan"),
               coverage_0p5=float("nan"), n_det=0, n_tmpl=0)
    if homo is None or homo.H_img2pitch is None:
        return out
    if L is None or W is None:
        L, W = homo._dims_m()

    # tespit-edilen beyaz-çizgi pikselleri (ham px) -> metre
    ys, xs = np.where(dl.prob_map > prob_thr)
    if len(xs) == 0:
        return out
    if len(xs) > max_det:
        sel = np.random.RandomState(0).choice(len(xs), max_det, replace=False)
        xs, ys = xs[sel], ys[sel]
    det_m = homo.pixel_to_pitch(np.c_[xs, ys].astype(float))
    inb = ((det_m[:, 0] >= -margin_m) & (det_m[:, 0] <= L + margin_m) &
           (det_m[:, 1] >= -margin_m) & (det_m[:, 1] <= W + margin_m))
    det_m = det_m[inb]
    if len(det_m) < 10:
        return out

    circle = (L / 2.0, W / 2.0, circle_r) if circle_r else None
    tmpl = _sample_segments_m(homo._template_segments_m(), circle=circle)
    if len(tmpl) == 0:
        return out

    d_ts = _nearest(tmpl, det_m)     # template -> tespit
    d_de = _nearest(det_m, tmpl)     # tespit -> template
    out["template_support_m"] = float(np.nanmedian(d_ts))
    out["detection_explained_m"] = float(np.nanmedian(d_de))
    out["coverage_0p5"] = float(np.mean(d_ts < 0.5))   # template'in %'si tespit-destekli
    out["n_det"] = int(len(det_m))
    out["n_tmpl"] = int(len(tmpl))
    return out


if __name__ == "__main__":
    import sys
    import cv2
    sys.path.insert(0, ".")
    from pitch.homography import PitchHomography
    from viz import line_detect
    homo = PitchHomography.load("calib/cankaya_cam2_FINAL.json")
    fr = cv2.imread("calib/cankaya_cam2_frame_raw.png")
    dl = line_detect.detect(fr)
    r = line_residual_m(homo, dl)
    print("cankaya FINAL (gold manuel):")
    for k, v in r.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
