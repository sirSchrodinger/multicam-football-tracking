#!/usr/bin/env python3
"""Sentetik halısaha çizgi-MASKESİ + keypoint üretici (otomatik calib eğitimi için).

YARATICI İÇGÖRÜ: RGB'de eğitmek sim-to-real uçurumuna takılır (gece tint, file, çim
dokusu sahadan sahaya değişir). Ama BEYAZ-ÇİZGİ MASKESİ görünümden bağımsızdır —
sentetik bir sahanın çizgi-maskesi ile gerçek sahanınki BENZER görünür. Bu yüzden
modeli çizgi-maskesi uzayında eğitiyoruz: maske -> keypoint ısı-haritaları.
Çıkarımda gerçek kareden turf-maskeli beyaz-çizgi çıkarımı maskeyi verir.

Üretir: (line_mask HxW uint8, keypoints {name:(x,y,vis)}). Rastgele saha boyutu,
kamera pozu, focal, fisheye, çizgi-dropout + gürültü (gerçek maske kalitesini taklit).
"""
import numpy as np
import cv2

# Çankaya ile aynı keypoint şeması (15)
KP_NAMES = ["corner_nl", "corner_nr", "corner_fl", "corner_fr", "mid_l", "mid_r",
            "center", "boxN_l", "boxN_r", "boxF_l", "boxF_r",
            "goalN_l", "goalN_r", "goalF_l", "goalF_r"]


def _rng(rs, a, b):
    return a + (b - a) * rs.random()


def pitch_geometry(L, W):
    bd, bhw = L / 8.0, 5.0
    kps = {
        "corner_nl": (0, 0), "corner_nr": (0, W), "corner_fl": (L, 0), "corner_fr": (L, W),
        "mid_l": (L / 2, 0), "mid_r": (L / 2, W), "center": (L / 2, W / 2),
        "boxN_l": (bd, W / 2 - bhw), "boxN_r": (bd, W / 2 + bhw),
        "boxF_l": (L - bd, W / 2 - bhw), "boxF_r": (L - bd, W / 2 + bhw),
        "goalN_l": (0, W / 2 - 1.5), "goalN_r": (0, W / 2 + 1.5),
        "goalF_l": (L, W / 2 - 1.5), "goalF_r": (L, W / 2 + 1.5),
    }
    segs = [[(0, 0), (L, 0)], [(L, 0), (L, W)], [(L, W), (0, W)], [(0, W), (0, 0)],   # sınır
            [(L / 2, 0), (L / 2, W)],                                                 # orta çizgi
            [(bd, W / 2 - bhw), (bd, W / 2 + bhw)], [(0, W / 2 - bhw), (bd, W / 2 - bhw)],
            [(0, W / 2 + bhw), (bd, W / 2 + bhw)],                                    # near box
            [(L - bd, W / 2 - bhw), (L - bd, W / 2 + bhw)],
            [(L, W / 2 - bhw), (L - bd, W / 2 - bhw)], [(L, W / 2 + bhw), (L - bd, W / 2 + bhw)]]
    return kps, segs, bd, bhw


def lookat_R(C, target):
    f = np.array(target) - np.array(C); f = f / np.linalg.norm(f)
    up = np.array([0, 0, 1.0])
    r = np.cross(f, up); r = r / np.linalg.norm(r)
    d = np.cross(f, r)
    return np.stack([r, d, f], 0)   # rows: x-right, y-down, z-forward


def project(P3, C, R, K, k1, k2):
    Xc = (R @ (P3 - C).T).T
    z = Xc[:, 2]
    xn = Xc[:, 0] / z; yn = Xc[:, 1] / z
    r2 = xn * xn + yn * yn
    fac = 1 + k1 * r2 + k2 * r2 * r2
    u = K[0, 0] * xn * fac + K[0, 2]
    v = K[1, 1] * yn * fac + K[1, 2]
    return np.stack([u, v], 1), z


def sample(seed, H=288, Wd=512):
    rs = np.random.RandomState(seed)
    L = _rng(rs, 25, 42); W = _rng(rs, 16, 25)
    kps, segs, bd, bhw = pitch_geometry(L, W)
    # kamera near-kale arkasi, yukselti
    C = np.array([_rng(rs, -4, 1), W / 2 + _rng(rs, -3, 3), _rng(rs, 2.5, 5.2)])
    target = np.array([L * _rng(rs, 0.45, 0.7), W / 2 + _rng(rs, -2, 2), 0])
    R = lookat_R(C, target)
    f = _rng(rs, 0.55, 1.05) * Wd
    K = np.array([[f, 0, Wd / 2], [0, f, H / 2], [0, 0, 1.0]])
    k1 = _rng(rs, -0.32, -0.05); k2 = _rng(rs, 0, 0.12)

    def to_img(pts3):
        return project(np.asarray(pts3, float), C, R, K, k1, k2)

    mask = np.zeros((H, W if False else Wd), np.uint8) if False else np.zeros((H, Wd), np.uint8)
    # çizgileri çiz (segment başına çok-nokta örnekle -> fisheye eğrisi)
    for (p0, p1) in segs:
        t = np.linspace(0, 1, 40)
        pts = np.stack([p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]),
                        np.zeros_like(t)], 1)
        uv, z = to_img(pts)
        uv = uv[z > 0.2]
        lw = max(1, int(_rng(rs, 1, 3)))
        for i in range(len(uv) - 1):
            a, b = uv[i], uv[i + 1]
            if (np.all(np.isfinite(a)) and np.all(np.isfinite(b))
                    and np.all(np.abs(a) < 4000) and np.all(np.abs(b) < 4000)):
                cv2.line(mask, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 255, lw)
    # orta yuvarlak
    th = np.linspace(0, 2 * np.pi, 60)
    circ = np.stack([L / 2 + 3 * np.cos(th), W / 2 + 3 * np.sin(th), np.zeros_like(th)], 1)
    uv, z = to_img(circ); uv = uv[z > 0.2]
    for i in range(len(uv) - 1):
        a, b = uv[i], uv[i + 1]
        if (np.all(np.isfinite(a)) and np.all(np.isfinite(b))
                and np.all(np.abs(a) < 4000) and np.all(np.abs(b) < 4000)):
            cv2.line(mask, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 255, 2)

    # keypoint görüntü konumları + görünürlük
    KP = {}
    for name in KP_NAMES:
        x, y = kps[name]
        uv, z = to_img([[x, y, 0]])
        u, v = uv[0]
        vis = (z[0] > 0.1) and (0 <= u < Wd) and (0 <= v < H)
        KP[name] = (float(u), float(v), int(vis))

    # GERÇEK MASKE KALİTESİNİ TAKLİT: çizgi-dropout + gürültü + morfoloji
    if rs.random() < 0.6:  # bazı çizgileri sil (örtülü/soluk)
        ys = int(rs.randint(0, H)); xs = int(rs.randint(0, Wd))
        cv2.rectangle(mask, (xs, ys), (xs + int(rs.randint(40, 160)), ys + int(rs.randint(20, 90))), 0, -1)
    # rastgele kısa-çizgi gürültü (file/yansıma sahte-pozitif)
    for _ in range(int(rs.randint(0, 14))):
        x0, y0 = int(rs.randint(0, Wd)), int(rs.randint(0, H))
        ang = rs.random() * np.pi; ln = int(rs.randint(8, 40))
        x1, y1 = int(x0 + ln * np.cos(ang)), int(y0 + ln * np.sin(ang))
        cv2.line(mask, (x0, y0), (x1, y1), 255, 1)
    mask = cv2.dilate(mask, np.ones((2, 2), np.uint8))
    meta = dict(L=L, W=W, f=f, k1=k1, k2=k2)
    return mask, KP, meta


if __name__ == "__main__":
    # 6 örnek montaj — gerçek line-mask'a benziyor mu gözle bak
    panels = []
    for s in range(6):
        m, KP, meta = sample(s)
        vis = cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)
        for name, (u, v, vi) in KP.items():
            if vi:
                cv2.circle(vis, (int(u), int(v)), 4, (0, 255, 255), -1)
        cv2.putText(vis, f"L={meta['L']:.0f} W={meta['W']:.0f} k1={meta['k1']:.2f}",
                    (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
        panels.append(vis)
    grid = np.vstack([np.hstack(panels[:3]), np.hstack(panels[3:])])
    cv2.imwrite("scratchpad/autorun/synth_samples.jpg", grid)
    print("yazıldı synth_samples.jpg", grid.shape)
