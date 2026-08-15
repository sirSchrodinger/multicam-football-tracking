#!/usr/bin/env python3
"""pitch<->pixel distorsiyon-konvansiyon tutarliligi (saha-koordinat footgun fix).

pitch_to_pixel(distorted=True) = pixel_to_pitch'in TAM TERSI olmali (raw->pitch->raw ~0).
default (distorted=False) UNDISTORTED dondurur -> onu pixel_to_pitch'e vermek CIFT-undistort
(eski 7m 'hata' artefakti); test bunun da gercekten saptigini dogrular (fix'in onemi).
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

CALIB = os.path.join(ROOT, "calib", "cankaya_cam2_v2.json")
_REAL = os.path.exists(CALIB)


@pytest.mark.skipif(not _REAL, reason="v2 calib yok")
def test_distorted_inverse_roundtrip_tight():
    from pitch.homography import PitchHomography
    h = PitchHomography.load(CALIB)
    L, W = h._dims_m()
    P = np.array([(x, y) for x in np.linspace(1, L - 1, 12)
                  for y in np.linspace(1, W - 1, 7)], float)
    raw = h.pitch_to_pixel(P, distorted=True)     # pitch -> HAM px (tam ters)
    back = h.pixel_to_pitch(raw)                   # HAM -> pitch (tek undistort)
    err = np.hypot(*(back - P).T)
    assert np.median(err) < 0.02, f"medyan {np.median(err)}"
    assert err.max() < 0.25, f"max {err.max()} (far-third dahil tutarli olmali)"


@pytest.mark.skipif(not _REAL, reason="v2 calib yok")
def test_undistorted_path_is_double_undistort_artifact():
    # default (undistorted) ciktiyi pixel_to_pitch'e vermek CIFT-undistort -> kenarda sapar.
    # (Bu testi gecmek 'fix gerekliydi'yi belgeler; distorted=True yolu yukarida ~0.)
    from pitch.homography import PitchHomography
    h = PitchHomography.load(CALIB)
    L, W = h._dims_m()
    far = np.array([[L - 1, W - 1], [L - 2, 1.0]], float)   # uzak-uc koseler
    und = h.pitch_to_pixel(far, distorted=False)            # UNDISTORTED (yanlis besleme)
    back = h.pixel_to_pitch(und)
    assert np.hypot(*(back - far).T).max() > 1.0  # cift-undistort belirgin sapar


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
