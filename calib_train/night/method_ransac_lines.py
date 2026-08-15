#!/usr/bin/env python3
"""METHOD: ransac_lines
Baseline groups_from_prob largest-component yerine: her sınır-rol kanalının
maskesindeki TÜM noktalara RANSAC robust çizgi-fit uygula. Fence/direk/gürültü
dış-noktalarını ele, baskın collinear inlier setine fit. Inlier noktaları group
olarak döner; solve_calib (joint lens+H) bunları kullanır.

Orijinal auto_calib BOZULMAZ; sadece groups üretimi değişir.
"""
import numpy as np, cv2
from calib_train import auto_calib as AC


def ransac_line(pts, iters=300, thr=3.0, seed=0):
    """pts (N,2) -> (inlier_mask, line[a,b,c]) en kalabalık collinear set.
    thr piksel-tolerans çözünürlüğe göre ölçeklenir (çağıran ayarlar)."""
    n = len(pts)
    if n < 2:
        return np.ones(n, bool), None
    rs = np.random.RandomState(seed)
    best_in = None; best_cnt = -1
    P = pts.astype(np.float64)
    for _ in range(iters):
        i, j = rs.randint(0, n, 2)
        if i == j:
            continue
        p, q = P[i], P[j]
        d = q - p; nrm = np.hypot(d[0], d[1])
        if nrm < 1e-6:
            continue
        # çizgi normali (a,b), c
        a, b = -d[1] / nrm, d[0] / nrm
        c = -(a * p[0] + b * p[1])
        dist = np.abs(a * P[:, 0] + b * P[:, 1] + c)
        inl = dist < thr
        cnt = int(inl.sum())
        if cnt > best_cnt:
            best_cnt = cnt; best_in = inl
    if best_in is None or best_cnt < 2:
        return np.ones(n, bool), None
    # inlier'larla rafine (TLS via fitLine)
    line = AC.fitL(P[best_in])
    a, b, c = line
    nn = np.hypot(a, b) + 1e-12
    dist = np.abs(a * P[:, 0] + b * P[:, 1] + c) / nn
    inl = dist < thr
    if inl.sum() >= 2:
        line = AC.fitL(P[inl])
        best_in = inl
    return best_in, line


def groups_from_prob_ransac(prob, w, h, thr=0.5, minpts=25, maxpts=220, seed=0):
    """prob (>=4,Hs,Ws) -> {role:inlier_pts}. Largest-component yerine RANSAC.
    Eksikse (None,reason)."""
    rs = np.random.RandomState(seed); groups = {}
    px_thr = max(2.5, w / 500.0)  # çözünürlüğe ölçekli inlier toleransı
    for ci, role in enumerate(["goalN", "goalF", "touchN", "touchF"]):
        pc = cv2.resize(prob[ci], (w, h)); mk = (pc > thr).astype(np.uint8)
        mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        pts = np.column_stack(np.where(mk > 0))[:, ::-1].astype(float)
        if len(pts) < minpts:
            return None, f"{role} bulunamadi({len(pts)})"
        # çok büyük maske -> hız için alt-örnekle (RANSAC öncesi)
        work = pts
        if len(work) > 4000:
            work = work[rs.choice(len(work), 4000, replace=False)]
        inl, line = ransac_line(work, iters=400, thr=px_thr, seed=seed + ci)
        in_pts = work[inl]
        if len(in_pts) < minpts:
            # RANSAC çöktüyse largest-component'e düş (güvenli geri)
            ncc, lbl, st, _ = cv2.connectedComponentsWithStats(mk)
            if ncc > 1:
                mk2 = (lbl == (1 + np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
                in_pts = np.column_stack(np.where(mk2 > 0))[:, ::-1].astype(float)
            else:
                in_pts = pts
            if len(in_pts) < minpts:
                return None, f"{role} ransac-zayif({len(in_pts)})"
        if len(in_pts) > maxpts:
            in_pts = in_pts[rs.choice(len(in_pts), maxpts, replace=False)]
        groups[role] = in_pts
    return groups, None


def calib_from_pred_ransac(prob, w, h, thr=0.5, minpts=25):
    cx, cy, s = w / 2, h / 2, w / 2
    groups, reason = groups_from_prob_ransac(prob, w, h, thr=thr, minpts=minpts)
    if groups is None:
        return dict(ok=False, reason=reason, fit=None)
    r = AC.solve_calib(groups, cx, cy, s)
    if r is None:
        return dict(ok=False, reason="H tekil", fit=None)
    r.update(ok=True, groups=groups)
    return r
