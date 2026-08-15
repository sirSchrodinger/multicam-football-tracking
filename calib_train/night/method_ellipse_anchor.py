#!/usr/bin/env python3
"""ELLIPSE-ANCHOR yontemi: prob[6] (merkez-yuvarlak) kanalindan cv2.fitEllipse ile
elips cikar, sinir piksellerini ornekle ve solve_calib residual'ina EK KISIT ekle:
elips pikselleri image->metrik haritalandiginda saha-merkezli R=3m cembere otursun.
Bu, head-on/zayif-perspektif dejenereligini kiran BAGIMSIZ bir kisit ekler.
circle kanali zayifsa anchor'siz (=baseline) coz; geometriyi bozma.

method_calib(prob,w,h,img) -> (rec, used_anchor)  ; rec baseline ile ayni format.
"""
import os, numpy as np, cv2
from scipy.optimize import least_squares
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from calib_train import auto_calib as AC

K1S, K2S = AC.K1S, AC.K2S
L = AC.L
Wp = 18.0
CX_RAD = 3.0  # merkez-yuvarlak yaricapi (m)


def ellipse_pts_from_prob(prob, w, h, thr=0.5, minpts=60, nsamp=60):
    """prob[6] -> elips sinir pikselleri (image uzayi) | None (zayif sinyal)."""
    pc = cv2.resize(prob[6], (w, h))
    mk = (pc > thr).astype(np.uint8)
    mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    ncc, lbl, st, _ = cv2.connectedComponentsWithStats(mk)
    if ncc <= 1:
        return None
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    area = st[big, cv2.CC_STAT_AREA]
    if area < minpts:
        return None
    comp = (lbl == big).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if len(c) < 5:
        return None
    try:
        (ex, ey), (MA, ma), ang = cv2.fitEllipse(c)
    except cv2.error:
        return None
    # dejenere elips reddi
    if MA < 4 or ma < 4 or MA / max(ma, 1e-6) > 12:
        return None
    # elips sinirini parametrik ornekle
    t = np.linspace(0, 2 * np.pi, nsamp, endpoint=False)
    a, b = MA / 2.0, ma / 2.0
    ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    xs = ex + a * np.cos(t) * ca - b * np.sin(t) * sa
    ys = ey + a * np.cos(t) * sa + b * np.sin(t) * ca
    pts = np.stack([xs, ys], 1)
    # goruntu-ici tut
    ins = (pts[:, 0] >= 0) & (pts[:, 0] < w) & (pts[:, 1] >= 0) & (pts[:, 1] < h)
    pts = pts[ins]
    if len(pts) < 20:
        return None
    return pts


def solve_calib_anchor(groups, ell_pts, cx, cy, s, Wp=18.0, ew=1.0):
    """solve_calib + elips-anchor residual blok."""
    CON = [("goalN", "X", 0.0), ("goalF", "X", L), ("touchN", "Y", 0.0), ("touchF", "Y", Wp)]
    fitL, und_pix, und_norm, aH, inter = AC.fitL, AC.und_pix, AC.und_norm, AC.aH, AC.inter

    def cor(k1, k2):
        fl = {r: fitL(und_pix(groups[r], k1, k2, cx, cy, s)) for r in groups}
        return np.array([inter(fl["goalN"], fl["touchN"]), inter(fl["goalN"], fl["touchF"]),
                         inter(fl["goalF"], fl["touchN"]), inter(fl["goalF"], fl["touchF"])])
    c0 = cor(K1S, K2S)
    H0, _ = cv2.findHomography(c0, np.array([[0, 0], [0, Wp], [L, 0], [L, Wp]], float))
    if H0 is None:
        return None
    H0 = H0 / H0[2, 2]
    p0 = [K1S, K2S, *H0.ravel()[:8]]
    cen = np.array([L / 2.0, Wp / 2.0])

    def resid(pp):
        k1, k2 = pp[0], pp[1]
        H = np.array([[pp[2], pp[3], pp[4]], [pp[5], pp[6], pp[7]], [pp[8], pp[9], 1.0]])
        out = []
        for r, kind, tg in CON:
            m = aH(und_norm(groups[r], k1, k2, cx, cy, s), H)
            out.append((m[:, 0] - tg) if kind == "X" else (m[:, 1] - tg))
        if ell_pts is not None:
            me = aH(und_norm(ell_pts, k1, k2, cx, cy, s), H)
            d = np.sqrt(((me - cen) ** 2).sum(1))
            out.append(ew * (d - CX_RAD))
        return np.concatenate(out)

    sol = least_squares(resid, p0, method="trf",
                        bounds=([0, 0] + [-np.inf] * 8, [0.45, 0.7] + [np.inf] * 8), max_nfev=4000)
    k1, k2 = sol.x[0], sol.x[1]
    H = np.array([[sol.x[2], sol.x[3], sol.x[4]], [sol.x[5], sol.x[6], sol.x[7]], [sol.x[8], sol.x[9], 1.0]])
    fit = float(np.sqrt((resid(sol.x) ** 2).mean()))

    def u2pix(P):
        un = und_norm(P, k1, k2, cx, cy, s)
        return np.stack([cx + un[:, 0] / 1.5 * s, cy + un[:, 1] / 1.5 * s], 1)
    fl = {r: fitL(u2pix(groups[r])) for r in groups}
    c_nN = inter(fl["goalN"], fl["touchN"]); c_nF = inter(fl["goalN"], fl["touchF"]); c_fN = inter(fl["goalF"], fl["touchN"])
    A = c_fN - c_nN; B = c_nF - c_nN; cross = A[0] * B[1] - A[1] * B[0]
    return dict(k1=k1, k2=k2, H=H, fit=fit, camside="SAG" if cross > 0 else "SOL")


def method_calib(prob, w, h, thr=0.5, minpts=25, img=None):
    """FALLBACK tasarimi: once sinir-only (baseline) coz. Sanity gecerse ONU kullan
    (regresyon yok). Gecmezse ve elips varsa anchor-coz dene; anchor sanity gecerse
    onu kullan (RECOVERY). Net etki monoton >= baseline."""
    from calib_train import auto_clean2d as AC2
    cx, cy, s = w / 2, h / 2, w / 2
    groups, reason = AC.groups_from_prob(prob, w, h, thr=thr, minpts=minpts)
    if groups is None:
        return dict(ok=False, reason=reason, fit=None), False
    # 1) baseline (sinir-only)
    base = AC.solve_calib(groups, cx, cy, s)
    if base is not None:
        base.update(ok=True, groups=groups)
        ok_b, _ = AC2.sanity(base, w, h, img=img)
        if ok_b:
            return base, False  # baseline yeterli, anchor'a gerek yok
    # 2) baseline yok/sanity-fail -> elips-anchor fallback
    ell = ellipse_pts_from_prob(prob, w, h, thr=thr)
    if ell is None:
        if base is not None:
            return base, False
        return dict(ok=False, reason="H tekil", fit=None), False
    r = solve_calib_anchor(groups, ell, cx, cy, s, ew=1.0)
    if r is None:
        return (base if base is not None else dict(ok=False, reason="H tekil", fit=None)), True
    r.update(ok=True, groups=groups)
    return r, True
