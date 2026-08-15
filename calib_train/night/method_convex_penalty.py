#!/usr/bin/env python3
"""YÖNTEM: convex_penalty.
solve_calib least_squares residual'ına, reprojekte kanonik dikdörtgenin (köşeler ->
ileri model -> piksel) KONVEKS olmamasını cezalandiran terim ekler. Optimizer bowtie/
çöküş çözüme kayamaz. ORİJİNAL auto_calib BOZULMADI; burada solve_calib yeniden yazildi.
calib_from_pred_cp(prob,w,h) -> rec (AC.calib_from_pred ile ayni şema).
"""
import numpy as np, cv2
from calib_train import auto_calib as AC
from calib_train.auto_calib import und_pix, und_norm, aH, fitL, inter, project_metric, L, K1S, K2S


def _corners_px(k1, k2, H, cx, cy, s):
    """kanonik dikdörtgen köşeleri (metrik) -> ileri-model piksel."""
    cm = np.array([[0, 0], [L, 0], [L, 18.0], [0, 18.0]], float)
    return project_metric(cm, k1, k2, H, cx, cy, s)


def _convex_resid(pix, w, h, weight):
    """4 köşe piksel -> konvekslik-ihlal residual (4 değer). Konveks ise ~0.
    non-finite ise büyük ceza. Normalize: piksel/w."""
    if not np.isfinite(pix).all():
        return np.full(4, weight * 10.0)
    c = pix / float(w)
    d = np.diff(np.vstack([c, c[0]]), axis=0)
    cr = d[:, 0] * np.roll(d[:, 1], -1) - d[:, 1] * np.roll(d[:, 0], -1)  # 4 çapraz-çarpim
    am = np.abs(cr)
    sgn = np.sign(cr[int(np.argmax(am))]) if am.max() > 1e-12 else 1.0
    # baskin işarete UYMAYAN köşeleri cezalandir (relu(-sgn*cr))
    pen = np.maximum(0.0, -sgn * cr)
    return weight * pen


def solve_calib_cp(groups, cx, cy, s, w, h, Wp=18.0, weight=3.0):
    CON = [("goalN", "X", 0.0), ("goalF", "X", L), ("touchN", "Y", 0.0), ("touchF", "Y", Wp)]
    from scipy.optimize import least_squares

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

    def Hof(pp):
        return np.array([[pp[2], pp[3], pp[4]], [pp[5], pp[6], pp[7]], [pp[8], pp[9], 1.0]])

    def resid(pp):
        k1, k2 = pp[0], pp[1]
        H = Hof(pp)
        out = []
        for r, kind, tg in CON:
            m = aH(und_norm(groups[r], k1, k2, cx, cy, s), H)
            out.append((m[:, 0] - tg) if kind == "X" else (m[:, 1] - tg))
        # KONVEKSLİK cezasi
        pix = _corners_px(k1, k2, H, cx, cy, s)
        out.append(_convex_resid(pix, w, h, weight))
        return np.concatenate(out)

    sol = least_squares(resid, p0, method="trf",
                        bounds=([0, 0] + [-np.inf] * 8, [0.45, 0.7] + [np.inf] * 8),
                        max_nfev=4000)
    k1, k2 = sol.x[0], sol.x[1]
    H = Hof(sol.x)
    # fit = SADECE çizgi-residual (konvekslik-terimi hariç) -> baseline ile kiyaslanabilir
    line_r = []
    for r, kind, tg in CON:
        m = aH(und_norm(groups[r], k1, k2, cx, cy, s), H)
        line_r.append((m[:, 0] - tg) if kind == "X" else (m[:, 1] - tg))
    fit = float(np.sqrt((np.concatenate(line_r) ** 2).mean()))

    def u2pix(P):
        un = und_norm(P, k1, k2, cx, cy, s)
        return np.stack([cx + un[:, 0] / 1.5 * s, cy + un[:, 1] / 1.5 * s], 1)
    fl = {r: fitL(u2pix(groups[r])) for r in groups}
    c_nN = inter(fl["goalN"], fl["touchN"]); c_nF = inter(fl["goalN"], fl["touchF"]); c_fN = inter(fl["goalF"], fl["touchN"])
    A = c_fN - c_nN; B = c_nF - c_nN; cross = A[0] * B[1] - A[1] * B[0]
    return dict(k1=k1, k2=k2, H=H, fit=fit, camside="SAG" if cross > 0 else "SOL")


def calib_from_pred_cp(prob, w, h, thr=0.5, minpts=25, weight=3.0):
    cx, cy, s = w / 2, h / 2, w / 2
    groups, reason = AC.groups_from_prob(prob, w, h, thr=thr, minpts=minpts)
    if groups is None:
        return dict(ok=False, reason=reason, fit=None)
    r = solve_calib_cp(groups, cx, cy, s, w, h, weight=weight)
    if r is None:
        return dict(ok=False, reason="H tekil", fit=None)
    r.update(ok=True, groups=groups)
    return r
