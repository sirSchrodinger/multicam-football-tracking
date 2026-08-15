import os, sys, numpy as np, pandas as pd, pytest
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)
from pathlib import Path
from pitch.homography import PitchHomography
from pitch import scale_snap as S
from stats import topdown_stats as T

RAW    = os.path.join(_ROOT, 'raw/tracks_cankaya_cam2_clip2400.parquet')
PLAYER = os.path.join(_ROOT, 'stats_out/pipeline_demo/tracks_cankaya_cam2_clip2400_player.parquet')
V2 = os.path.join(_ROOT, 'calib/cankaya_cam2_v2.json')
V3 = os.path.join(_ROOT, 'calib/cankaya_cam2_v3.json')
HAVE = all(Path(p).exists() for p in (RAW, PLAYER, V2, V3))
real = pytest.mark.skipif(not HAVE, reason='real artifacts missing')

# 1) Schema normalizer accepts BOTH concurrent schemas -> both classify 'approx'
def test_normalizer_handles_both_schemas():
    gen = dict(kind='standard_size', reconciled=True, approximate=True, verified=False, band_pct=10, scale_factor=1.3558, catalog_label='25x45')
    jsn = dict(type='standard_size', reconciled=True, approximate=True, passed=False, band_pct=10, scale_rel_to_m=1.33784, snapped_label='25x45')
    for a in (gen, jsn):
        st = S.scale_state(a)
        assert st.quality == 'approx'
        assert st.unit == 'm' and st.unit_label == 'm (approx ±10%)'
        assert st.band_pct == 10 and st.scale_factor and st.catalog_label == '25x45'
    assert S.scale_state(None).quality == 'relative'
    ver = dict(kind='standard_size', reconciled=True, approximate=False, verified=True, band_pct=2)
    assert S.scale_state(ver).quality == 'exact' and S.scale_state(ver).unit_label == 'm'

# 2) QA invariance: v3 reprojection median == v2 median (scale absorbed by H)
@real
def test_v3_qa_px_invariant():
    q2 = PitchHomography.load(V2)._qa['median_px']; q3 = PitchHomography.load(V3)._qa['median_px']
    assert abs(q3 - q2) < 0.05  # ~9.875 both

# 3) force_reproject closes the silent fake-metre trap on baked-pitch parquet
@real
def test_force_reproject_breaks_disk_reuse():
    dfp = pd.read_parquet(PLAYER)
    assert np.isfinite(dfp['pitch_x']).mean() > 0.99      # disk IS baked (the trap)
    h2 = PitchHomography.load(V2); h3 = PitchHomography.load(V3)
    p2 = T.prepare(dfp, h2, fps=25.0); p3 = T.prepare(dfp, h3, fps=25.0)
    span2 = float(np.nanmax(p2.px) - np.nanmin(p2.px))
    span3 = float(np.nanmax(p3.px) - np.nanmin(p3.px))
    assert abs(span2 - 33.19) < 1.0                       # v2 unchanged (relative_m disk-reuse ok)
    assert span3 > 40.0                                   # v3 RE-PROJECTED (~44.4), NOT 33.19 silent-baked
    assert p2.scale_quality == 'relative' and p2.unit == 'relative_m'
    assert p3.scale_quality == 'approx'   and p3.unit_label == 'm (approx ±10%)'
    assert p3.unit_label != 'm'                           # never bare m

# 4) cap bound to quality -> v3 distance == v2 distance * s (clean isotropic, no cap artifact)
@real
def test_approx_is_relative_times_scale():
    dfp = pd.read_parquet(PLAYER)
    h2 = PitchHomography.load(V2); h3 = PitchHomography.load(V3)
    s = S.scale_state(h3).scale_factor
    k2 = T.kinematics_all(T.prepare(dfp, h2, 25.0)); k3 = T.kinematics_all(T.prepare(dfp, h3, 25.0))
    m = k2['dist'] > 1.0
    ratios = (k3['dist'][m] / k2['dist'][m])
    assert abs(float(np.median(ratios)) - s) / s < 0.03   # uniform multiplier ~= scale
    assert float(np.std(ratios)) < 0.05                    # axis-independent (isotropic), no cap confound

# 5) sprint only in exact; approx -> percentiles, no sprint; default v2 -> relative_m
@real
def test_report_labels_and_no_sprint_in_approx(tmp_path):
    r2 = T.generate_topdown_report(PLAYER, V2, str(tmp_path/'v2'))
    r3 = T.generate_topdown_report(PLAYER, V3, str(tmp_path/'v3'))
    assert r2['coordinate']['unit'] == 'relative_m'
    assert r3['coordinate']['scale_quality'] == 'approx'
    assert r3['coordinate']['unit_label'] == 'm (approx ±10%)'
    for p in r3['players'].values():
        assert 'sprint_count' not in p
        assert 'speed_pctl' in p
        assert p['distance_unit_label'] == 'm (approx ±10%)'
    for p in r2['players'].values():
        assert 'sprint_count' not in p   # relative also no sprint

# 6) generator is reproducible + fail-closed on broken calib
@real
def test_generator_reproducible_and_fail_closed(tmp_path):
    out = str(tmp_path/'v3_regen.json')
    res = S.make_v3(V2, out, fmt='7v7', band_pct=10)
    assert res['gate_ok'] and res['in_band']['L'] and res['in_band']['W']
    h = PitchHomography.load(out)
    assert S.scale_state(h).quality == 'approx'
    with pytest.raises(Exception):                       # broken calib rejected (fail-closed)
        S.make_v3(os.path.join(_ROOT,'calib/cankaya_cam2.json'), str(tmp_path/'bad.json'))
