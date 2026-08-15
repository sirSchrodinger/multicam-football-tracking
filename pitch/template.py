"""pitch/template.py — Halısaha saha şablonu (pitch template).

Saha-koordinat (metre) referansı: homografi + chamfer kayıt + manuel kalibrasyon
hepsi buradan beslenir. INTERFACE SPEC §3'e birebir uyar.

Koordinat konvansiyonu (tüm modüllerle ortak):
    origin (0,0) = sol-alt köşe.
    X ekseni  -> saha BOYU (length L) yönünde, [0, L]
    Y ekseni  -> saha ENİ  (width  W) yönünde, [0, W]
    polygon_m() = (0,0)-(L,0)-(L,W)-(0,W)   (saat yönünün tersi, BL->BR->TR->TL)

Amatör halısaha gerçeği: en güvenilir işaretler kenar çizgileri (duvar/çit kenarı),
orta çizgi ve orta yuvarlaktır. Ceza sahası çoğu sahada yok/soluk -> has_penalty_box
varsayılan False. Boyutlar PARAMETRİK; gerçek saha ölçülünce five_a_side(L=..,W=..)
ile override edilir (ölçmeden metre/km istatistiği iddia etme — scale_anchor bunun içindir).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import math

import numpy as np

# Saha-koordinat sistemi adı (kalibrasyon JSON'una yazılır, ileride farklı conv. ayırt etmek için)
COORD_CONVENTION = "origin_bottom_left_X_length_Y_width_meters"


def _sample_segment(p0, p1, step_m: float):
    """Bir doğru parçasını step_m aralıkla örnekle (uç noktalar dahil)."""
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    L = float(np.linalg.norm(p1 - p0))
    if L < 1e-9:
        return p0[None, :]
    n = max(1, int(math.ceil(L / max(step_m, 1e-6))))
    ts = np.linspace(0.0, 1.0, n + 1)[:, None]
    return p0[None, :] * (1.0 - ts) + p1[None, :] * ts


@dataclass
class PitchTemplate:
    """Metre cinsinden saha şablonu. Tüm fiziksel-boyalı işaretler burada tanımlı."""

    dims_m: tuple[float, float]                      # (L, W) metre; L=boy(X), W=en(Y)
    landmarks_m: dict[str, tuple[float, float]] = field(default_factory=dict)
    line_segments_m: list[tuple[tuple[float, float], tuple[float, float]]] = field(default_factory=list)
    has_penalty_box: bool = False
    center_circle_r_m: Optional[float] = None
    optimize_LW: bool = False                        # True ise chamfer L,W'yi de optimize eder (auto_calib TODO)
    scale_anchor: Optional[tuple[str, float]] = None  # ("goal_width", 3.0) gibi; ölçek doğrulaması için

    # ---- yapıcılar ----
    @classmethod
    def five_a_side(cls, L: float = 28.0, W: float = 18.0,
                    center_circle_r_m: Optional[float] = 3.0,
                    has_penalty_box: bool = False) -> "PitchTemplate":
        """5v5 halısaha tipik ~25-30m boy x ~15-20m en. Ölçülmemişse default; ölçünce override et."""
        return cls(dims_m=(float(L), float(W)),
                   landmarks_m=cls._default_landmarks(L, W),
                   line_segments_m=cls._default_segments(L, W, has_penalty_box),
                   has_penalty_box=has_penalty_box,
                   center_circle_r_m=center_circle_r_m)

    @classmethod
    def seven_a_side(cls, L: float = 40.0, W: float = 24.0,
                     center_circle_r_m: Optional[float] = 4.0,
                     has_penalty_box: bool = False) -> "PitchTemplate":
        """7v7 tipik ~38-45m x ~22-28m."""
        return cls(dims_m=(float(L), float(W)),
                   landmarks_m=cls._default_landmarks(L, W),
                   line_segments_m=cls._default_segments(L, W, has_penalty_box),
                   has_penalty_box=has_penalty_box,
                   center_circle_r_m=center_circle_r_m)

    # ---- varsayılan geometri ----
    @staticmethod
    def _default_landmarks(L: float, W: float) -> dict[str, tuple[float, float]]:
        return {
            "corner_bl": (0.0, 0.0),
            "corner_br": (L, 0.0),
            "corner_tr": (L, W),
            "corner_tl": (0.0, W),
            "half_bottom": (L / 2.0, 0.0),
            "half_top": (L / 2.0, W),
            "center": (L / 2.0, W / 2.0),
        }

    @staticmethod
    def _default_segments(L: float, W: float, has_penalty_box: bool):
        segs: list[tuple[tuple[float, float], tuple[float, float]]] = [
            ((0.0, 0.0), (L, 0.0)),     # alt kenar
            ((L, 0.0), (L, W)),         # sağ kenar
            ((L, W), (0.0, W)),         # üst kenar
            ((0.0, W), (0.0, 0.0)),     # sol kenar
            ((L / 2.0, 0.0), (L / 2.0, W)),  # orta çizgi
        ]
        if has_penalty_box:
            # Küçük, sahaya oranlı ceza alanı (her iki kale); amatörde çoğu zaman yok.
            bw = min(W * 0.55, W - 1.0)            # box width (Y)
            bd = min(L * 0.16, 6.0)               # box depth (X)
            y0, y1 = (W - bw) / 2.0, (W + bw) / 2.0
            for x_goal, xd in ((0.0, bd), (L, -bd)):
                segs += [
                    ((x_goal, y0), (x_goal + xd, y0)),
                    ((x_goal + xd, y0), (x_goal + xd, y1)),
                    ((x_goal + xd, y1), (x_goal, y1)),
                ]
        return segs

    # ---- tüketici API (auto_calib + homography çağırır) ----
    def polygon_m(self) -> np.ndarray:
        """(4,2) saha dikdörtgeni: BL, BR, TR, TL (metre)."""
        L, W = self.dims_m
        return np.array([[0.0, 0.0], [L, 0.0], [L, W], [0.0, W]], dtype=np.float64)

    def line_points_m(self, step_m: float = 0.25) -> np.ndarray:
        """(M,2) yoğun örneklenmiş çizgi noktaları (chamfer maliyeti için).

        Tüm line_segments_m + (varsa) orta yuvarlak örneklenir. line_segments_m boşsa
        en azından sınır + orta çizgi default'una düşülür (chamfer hiç boş kalmasın)."""
        segs = self.line_segments_m
        if not segs:
            L, W = self.dims_m
            segs = self._default_segments(L, W, self.has_penalty_box)
        pts = [_sample_segment(p0, p1, step_m) for (p0, p1) in segs]
        if self.center_circle_r_m and self.center_circle_r_m > 0:
            cx, cy = self.dims_m[0] / 2.0, self.dims_m[1] / 2.0
            r = float(self.center_circle_r_m)
            n = max(8, int(math.ceil(2.0 * math.pi * r / max(step_m, 1e-6))))
            th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
            pts.append(np.stack([cx + r * np.cos(th), cy + r * np.sin(th)], axis=1))
        return np.concatenate(pts, axis=0) if pts else np.empty((0, 2), dtype=np.float64)

    def goal_centers_m(self) -> dict[str, tuple[float, float]]:
        """İki kale orta noktası (Y simetri / yön ipucu için)."""
        L, W = self.dims_m
        return {"goal_left": (0.0, W / 2.0), "goal_right": (L, W / 2.0)}

    # ---- serileştirme (PitchHomography.save/load buradan geçer) ----
    def to_dict(self) -> dict:
        return {
            "dims_m": list(self.dims_m),
            "landmarks_m": {k: list(v) for k, v in self.landmarks_m.items()},
            "line_segments_m": [[list(p0), list(p1)] for (p0, p1) in self.line_segments_m],
            "has_penalty_box": bool(self.has_penalty_box),
            "center_circle_r_m": self.center_circle_r_m,
            "optimize_LW": bool(self.optimize_LW),
            "scale_anchor": list(self.scale_anchor) if self.scale_anchor else None,
            "coord_convention": COORD_CONVENTION,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "PitchTemplate":
        return cls(
            dims_m=tuple(d["dims_m"]),
            landmarks_m={k: tuple(v) for k, v in (d.get("landmarks_m") or {}).items()},
            line_segments_m=[(tuple(p0), tuple(p1)) for (p0, p1) in (d.get("line_segments_m") or [])],
            has_penalty_box=bool(d.get("has_penalty_box", False)),
            center_circle_r_m=d.get("center_circle_r_m"),
            optimize_LW=bool(d.get("optimize_LW", False)),
            scale_anchor=tuple(d["scale_anchor"]) if d.get("scale_anchor") else None,
        )


if __name__ == "__main__":
    # Hızlı self-test (GPU yok, ağ yok)
    t = PitchTemplate.five_a_side()
    poly = t.polygon_m()
    lp = t.line_points_m(step_m=0.5)
    assert poly.shape == (4, 2)
    assert lp.shape[0] > 20 and lp.shape[1] == 2
    assert t.dims_m == (28.0, 18.0)
    d = t.to_dict()
    t2 = PitchTemplate.from_dict(d)
    assert t2.dims_m == t.dims_m
    assert np.allclose(t2.polygon_m(), t.polygon_m())
    print(f"template self-test OK: polygon{poly.shape}, line_points={lp.shape[0]}, "
          f"segments={len(t.line_segments_m)}, landmarks={list(t.landmarks_m)}")
