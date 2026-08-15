"""Halısaha-özel keypoint şablonu (SoccerNet'in 57-kp full-pitch şeması UYMAZ —
ceza sahası yok, farklı oran). HER 5-a-side sahada var olan landmark'lar:
4 köşe + 2 orta-çizgi×kenar + orta-nokta + 4 çember-kardinal = 11 keypoint.
Kaleler opsiyonel (varsa +2). Metre-koordinatları saha (L,W) parametreli.
"""
import numpy as np

# keypoint tanımları: (isim, lambda(L,W,r)->(x,y) metre)
KP_DEFS = [
    ("corner_bl",     lambda L, W, r: (0.0, 0.0)),
    ("corner_br",     lambda L, W, r: (L, 0.0)),
    ("corner_tr",     lambda L, W, r: (L, W)),
    ("corner_tl",     lambda L, W, r: (0.0, W)),
    ("halfway_bottom", lambda L, W, r: (L / 2, 0.0)),
    ("halfway_top",   lambda L, W, r: (L / 2, W)),
    ("center",        lambda L, W, r: (L / 2, W / 2)),
    ("circle_top",    lambda L, W, r: (L / 2, W / 2 + r)),
    ("circle_bottom", lambda L, W, r: (L / 2, W / 2 - r)),
    ("circle_left",   lambda L, W, r: (L / 2 - r, W / 2)),
    ("circle_right",  lambda L, W, r: (L / 2 + r, W / 2)),
]
KP_NAMES = [d[0] for d in KP_DEFS]
NUM_KP = len(KP_DEFS)  # 11


def keypoints_m(L, W, circle_r=3.0):
    """(NUM_KP, 2) metre-koordinatı keypoint dizisi."""
    return np.array([fn(L, W, circle_r) for _, fn in KP_DEFS], float)


# SoccerNet-uyumlu çizgi seti (annotation üretimi için; halısahada VAR olanlar)
def line_segments_m(L, W, circle_r=3.0, n_circle=24):
    """İsimli saha çizgileri -> {isim: [(x,y)m, ...]}. Boundary+halfway+circle."""
    segs = {
        "Side line bottom": [(0, 0), (L, 0)],
        "Side line top":    [(0, W), (L, W)],
        "Side line left":   [(0, 0), (0, W)],
        "Side line right":  [(L, 0), (L, W)],
        "Middle line":      [(L / 2, 0), (L / 2, W)],
    }
    circ = [(L / 2 + circle_r * np.cos(a), W / 2 + circle_r * np.sin(a))
            for a in np.linspace(0, 2 * np.pi, n_circle, endpoint=False)]
    segs["Circle central"] = circ
    return segs
