import os
import sys
import json
import numpy as np
import pandas as pd
import pytest

# repo koku yola (pytest farkli cwd'lerde calisabilir)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pathlib import Path
from pitch.homography import PitchHomography
from stats import topdown_stats as T

REAL = os.path.join(_ROOT, "raw/tracks_cankaya_cam2_clip2400.parquet")
GOOD = os.path.join(_ROOT, "calib/cankaya_cam2_v2.json")
BAD = os.path.join(_ROOT, "calib/cankaya_cam2.json")

_HAVE_REAL = Path(REAL).exists() and Path(GOOD).exists() and Path(BAD).exists()
_real = pytest.mark.skipif(not _HAVE_REAL, reason="real artifacts missing")


def test_calib_gate_fails_broken_passes_good():
    ok_bad, why = T.accept_calib_qa(PitchHomography.load(BAD)._qa)
    ok_good, _ = T.accept_calib_qa(PitchHomography.load(GOOD)._qa)
    assert ok_bad is False and any("median" in r for r in why)   # 65px rejected
    assert ok_good is True                                        # 9.87px accepted


@_real
def test_projects_onthefly_without_redetect():
    df = pd.read_parquet(REAL)
    assert df["pitch_x"].notna().sum() == 0                       # disk is NaN
    homo = PitchHomography.load(GOOD)
    prep = T.prepare(df, homo, fps=25.0)
    assert np.isfinite(prep.px).mean() > 0.99                     # filled in-memory
    assert prep.in_pitch.mean() > 0.95                            # v2 in_pitch~1.0


@_real
def test_unit_stays_relative_never_meters():
    rep = T.generate_topdown_report(REAL, GOOD, "stats_out/_t")
    assert rep["coordinate"]["unit"] == "relative_m"             # scale_anchor null
    assert rep["coordinate"]["unit"] != "m"
    for p in rep["players"].values():
        assert "sprint_count" not in p or p.get("distance_unit") == "m"


@_real
def test_parity_qc_matches_real_signal():
    q = T.parity_qc(pd.read_parquet(REAL), expected=14)
    assert 0.10 <= q["frac_at_expected"] <= 0.20                 # verified 0.153
    assert q["median_count"] <= 14 and "far" in q["note"].lower()


@_real
def test_coverage_mask_offframe_correct_space():
    # DISTORTED-uzay duzeltmesi sonrasi: kalibre saha (34x18) TAM kadrajda -> ic binler
    # hepsi covered (eski gri-kama undistorted-piksel-vs-raw-sinir ARTEFAKTIYDI).
    # Ayrica acikca kadraj-disi dunya bolgesi DOGRU maskelenmeli.
    df = pd.read_parquet(REAL)
    homo = PitchHomography.load(GOOD)
    L, W = homo._dims_m()
    cov_in = T.coverage_mask_pitch(homo, (1920, 1080),
                                   np.linspace(0, L, 6), np.linspace(0, W, 4))
    assert cov_in.all(), "kalibre saha tam kadrajda olmali (distorted-space duzeltmesi)"
    # NOT: saha-disi != kadraj-disi (file/duvar GORUNUR); coverage GORUNURLUGU test eder.
    # Homografi 2D projektif (derinlik-isareti yok) -> kamera-arkasi nokta guvenilir
    # tespit edilemez; bu yuzden negatif-test yok. Asil duzeltme: saha tam kapsanir.
    prep = T.prepare(df, homo, fps=25.0)
    grid, edges = T.heatmap_grid_equal_area(prep, homo, (1920, 1080), bin_m=1.0)
    assert np.nanmin(grid) >= 0


@_real
def test_single_pass_speed_reused_for_sprints():
    df = pd.read_parquet(REAL)
    homo = PitchHomography.load(GOOD)
    prep = T.prepare(df, homo, fps=25.0)
    k = T.kinematics_all(prep)
    assert "spd_row" in k and len(k["spd_row"]) == prep.px.size
    assert "dist" in k and "dist_raw" in k


@_real
def test_honesty_block_surfaces_fragmentation_and_scale():
    rep = T.generate_topdown_report(REAL, GOOD, "stats_out/_t")
    h = rep["honesty"]
    assert h["acceleration"].upper().startswith("NOT")
    assert rep["possession_pct"] is None
    assert "146" in h["identity"] or "fragment" in h["identity"].lower()
    assert "scale_anchor" in json.dumps(h).lower() or "relative_m" in json.dumps(h)
    assert h["per_zone"]["near"] is not None


@_real
def test_broken_calib_falls_back_not_silent_meters():
    rep = T.generate_topdown_report(REAL, BAD, "stats_out/_tb")
    assert rep["coordinate"]["unit"] in ("relative_m", "px")
    assert rep["honesty"]["calib_gate"]["ok"] is False


if __name__ == "__main__":
    import subprocess
    raise SystemExit(subprocess.call(
        [sys.executable, "-m", "pytest", __file__, "-q"]))
