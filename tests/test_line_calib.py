import sys; from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, cv2
from pitch.template import PitchTemplate
from pitch.homography import PitchHomography
from pitch.line_calib import (fit_line_tls, line_intersection, corners_from_lines,
    build_homography_from_lines, line_point_residual_qa, coverage_confidence_map)

L, W = 34.0, 18.0
img = np.array([[204.448,599.13],[128.019,477.03],[1775.79,160.75],[1661.34,152.14],
                [533.549,1110.77],[55.94,349.02],[2095.49,190.04],[1422.44,133.62]])
world = np.array([[0,7.5],[0,10.5],[34,7.5],[34,10.5],[0,0],[0,18],[34,0],[34,18.]])
Hwp,_ = cv2.findHomography(world, img)                       # world(m) -> und px (no distortion)
def w2p(P): return cv2.perspectiveTransform(np.asarray(P,float).reshape(-1,1,2), Hwp).reshape(-1,2)
def samp(p0,p1,n=6):
    t=np.linspace(0,1,n)[:,None]; return w2p(np.array(p0)*(1-t)+np.array(p1)*t)
clicks = {"endline_x0": samp((0,0),(0,W)), "endline_xL": samp((L,0),(L,W)),
          "touchline_y0": samp((0,0),(L,0)), "touchline_yW": samp((0,W),(L,W))}

# 1. line fit returns a normalized homogeneous line
f = fit_line_tls(clicks["touchline_y0"]); l = np.asarray(f["l"])
assert abs(np.hypot(l[0],l[1]) - 1.0) < 1e-6, "line not normalized"

# 2. intersection conditioning is |sin theta| in [0,1]
pt, s = line_intersection(clicks["endline_xL"] is not None and fit_line_tls(clicks["endline_xL"])["l"],
                          fit_line_tls(clicks["touchline_y0"])["l"])
assert pt is not None and 0.0 <= s <= 1.0

# 3. OFF-FRAME corner synthesized natively and recovered to sub-pixel
fits = {k: fit_line_tls(v) for k,v in clicks.items()}
cs = corners_from_lines(fits, L, W)
true_br = w2p([[L,0.0]])[0]
assert true_br[0] > 1920, "fixture should place far-bottom off-frame"     # ~2101
got_br = np.asarray(cs["br"]["img_und"])
assert np.linalg.norm(got_br - true_br) < 1.5, f"off-frame corner err {got_br} vs {true_br}"
assert 0.0 <= cs["br"]["sin_angle"] <= 1.0 and "ill" in cs["br"] and "extrap" in cs["br"]

# 4. full build + METER QA (un-gameable, honest)
homo, cs2, qa = build_homography_from_lines("test_cam", PitchTemplate.seven_a_side(L=L,W=W), clicks)
assert homo.H_img2pitch is not None and homo.calib_method == "manual_line"
assert qa["median_m"] < 0.5, f"meter residual too high: {qa['median_m']}"
qa2 = line_point_residual_qa(homo, clicks, L, W)
assert "per_line" in qa2 and qa2["median_m"] < 0.5

# 5. coverage-confidence map: shape, range, off-frame cells -> 0, some covered
cov = coverage_confidence_map(homo, (1080, 1920))
conf = np.asarray(cov["conf"])
assert conf.shape == (len(cov["ys"]), len(cov["xs"]))
assert conf.min() >= 0.0 and conf.max() <= 1.0
assert (conf == 0).any() and (conf > 0).any(), "expected mix of off-frame and covered cells"
assert 0.0 < cov["covered_frac"] < 1.0 and np.asarray(cov["covered_polygon_m"]).shape == (4,2)

# 6. honesty gate: too few lines -> refuse, do not fabricate an H
try:
    build_homography_from_lines("x", PitchTemplate.seven_a_side(L=L,W=W),
                                {"endline_x0": clicks["endline_x0"]})
    raise AssertionError("should have refused with <4 corners")
except RuntimeError:
    pass

# 7. backward-compat: calibrate_manual default unchanged + accepts new flag
h = PitchHomography("bc", PitchTemplate.seven_a_side(L=L,W=W))
h.calibrate_manual(img, world)                                  # old call still works (identity undistort)
base = h.pixel_to_pitch(img[:1])
h2 = PitchHomography("bc2", PitchTemplate.seven_a_side(L=L,W=W))
h2.calibrate_manual(img, world, already_undistorted=True)       # new kwarg accepted
assert np.allclose(base, h2.pixel_to_pitch(img[:1], already_undistorted=True), atol=1e-6)

print("OK: line_calib acceptance passed")
