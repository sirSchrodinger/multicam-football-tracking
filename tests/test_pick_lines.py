import sys, json, tempfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import numpy as np, cv2
from pitch.template import PitchTemplate
from pitch.homography import PitchHomography
from pitch.line_calib import (fit_line_tls, corners_from_lines,
    coverage_confidence_map, build_homography_from_lines)
import pick_lines as PL

L, W = 34.0, 18.0

def _real():
    d = json.load(open(ROOT/"calib/cankaya_cam2_v2.json"))
    img = np.array(d["_img_pts_und"]); w = np.array(d["_world_pts"])
    K = np.array(d["K"]); dist = np.array(d["dist"])
    clicks = {"endline_x0": img[w[:,0]==0], "endline_xL": img[w[:,0]==34],
              "touchline_y0": img[w[:,1]==0], "touchline_yW": img[w[:,1]==18]}
    M = {tuple(np.round(ww,3)): u for ww,u in zip(w,img)}
    return d, img, w, K, dist, clicks, M

# 1. picks list -> dict; singletons dropped
def test_assemble_drops_singletons():
    picks=[("touchline_y0",(10,20)),("touchline_y0",(30,40)),("endline_x0",(5,5))]
    lc=PL.assemble_line_clicks(picks)
    assert set(lc)=={"touchline_y0"} and lc["touchline_y0"].shape==(2,2)

# 2. menu = full boundary key set; halfway last (optional, not a corner parent)
def test_line_menu_keys():
    keys={k for k,_ in PL.line_menu(L,W)}
    assert keys>={"endline_x0","endline_xL","touchline_y0","touchline_yW"}
    assert PL.line_menu(L,W)[-1][0]=="halfway"

# 3. REAL: off-frame corner recovery + honest conditioning (Alperen insight #2)
def test_real_offframe_and_conditioning():
    d,img,w,K,dist,clicks,M=_real()
    cs=corners_from_lines({k:fit_line_tls(v) for k,v in clicks.items()},L,W)
    assert M[(34,0)][0]>1920 and M[(0,0)][1]>1080            # br off-right, bl off-bottom (LEGAL)
    assert np.linalg.norm(cs["br"]["img_und"]-M[(34,0)])<3.0 # measured 1.66px
    assert np.linalg.norm(cs["bl"]["img_und"]-M[(0,0)])<3.0  # measured 1.56px
    assert cs["tr"]["ill"] is True and cs["tr"]["sin_angle"]<0.26  # far-upper grazing sin=0.238
    assert cs["bl"]["ill"] is False

# 4. REAL full pipeline: meter-QA gate + reproj~0 degeneracy + cross-check + scale=null + save round-trip
def test_real_full_pipeline_meterqa():
    d,img,w,K,dist,clicks,M=_real()
    with tempfile.TemporaryDirectory() as td:
        res=PL.run_line_calibration(None,K,dist,clicks,L,W,"cankaya_cam2",outdir=td,tag="t")
        homo=res["homo"]
        assert homo.calib_method=="manual_line"
        assert res["qa_line"]["median_m"]<2.0                # measured 0.023
        assert homo._qa["n_finite_corr"]==4 and homo._qa["n_well_conditioned"]==3
        assert homo.scale_anchor is None                     # INSIGHT(5) metre kilidi kapali
        assert homo.reprojection_error()<1e-3                # DEGENERACY: ~0, must NOT be a gate
        XY=homo.pixel_to_pitch(img,already_undistorted=True)
        assert float(np.median(np.linalg.norm(XY-w,axis=1)))<1.0   # un-gameable cross-check, measured 0.258m
        assert res["saved"] and "cal" in res["paths"]
        h2=PitchHomography.load(res["paths"]["cal"])
        assert np.allclose(h2.H_img2pitch,homo.H_img2pitch)

# 5. point_lms mirror-breaker: scale stays null by default, only seeded when explicitly locked
def test_pointlms_scale_lock():
    d,img,w,K,dist,clicks,M=_real()
    plm=PL.goalpost_point_lms({"near_low":M[(0,7.5)],"near_high":M[(0,10.5)],
                               "far_low":M[(34,7.5)],"far_high":M[(34,10.5)]},L,W)
    with tempfile.TemporaryDirectory() as td:
        r0=PL.run_line_calibration(None,K,dist,clicks,L,W,"c",outdir=td,point_lms=plm,lock_scale=False)
        assert r0["homo"]._qa["n_finite_corr"]==8 and r0["homo"].scale_anchor is None
        r1=PL.run_line_calibration(None,K,dist,clicks,L,W,"c",outdir=td,point_lms=plm,lock_scale=True)
        assert r1["homo"].scale_anchor==("goal_width",3.0)

# 6. REAL coverage OVER-OPTIMISM documented: covered_frac==1.0 here -> only assert [0,1]+variance, never frac<1
def test_real_coverage_overoptimism_documented():
    d,img,w,K,dist,clicks,M=_real()
    homo,_,_=build_homography_from_lines("c",PitchTemplate.seven_a_side(L=L,W=W),clicks,K=K,dist=dist)
    cov=coverage_confidence_map(homo,(1080,1920)); conf=np.asarray(cov["conf"])
    assert 0.0<=conf.min() and conf.max()<=1.0 and conf.min()<conf.max()
    assert 0.0<=cov["covered_frac"]<=1.0
    assert np.asarray(cov["covered_polygon_m"]).shape==(4,2)

# 7. SYNTHETIC (distortion-free): off-frame cells are GENUINELY 0 and covered_frac<1
def test_synthetic_offframe_coverage():
    img=np.array([[204,599],[1775,160],[533,1110],[55,349],[2095,190],[1422,133]],float)
    world=np.array([[0,7.5],[34,7.5],[0,0],[0,18],[34,0],[34,18.]])
    Hwp,_=cv2.findHomography(world,img)
    w2p=lambda P: cv2.perspectiveTransform(np.asarray(P,float).reshape(-1,1,2),Hwp).reshape(-1,2)
    samp=lambda a,b,n=6:w2p(np.array(a)*(1-np.linspace(0,1,n)[:,None])+np.array(b)*np.linspace(0,1,n)[:,None])
    clicks={"endline_x0":samp((0,0),(0,W)),"endline_xL":samp((L,0),(L,W)),
            "touchline_y0":samp((0,0),(L,0)),"touchline_yW":samp((0,W),(L,W))}
    res=PL.run_line_calibration(None,None,None,clicks,L,W,"syn",outdir=tempfile.mkdtemp())
    cov=coverage_confidence_map(res["homo"],(1080,1920)); conf=np.asarray(cov["conf"])
    assert (conf==0).any() and (conf>0).any() and 0.0<cov["covered_frac"]<1.0

# 8. HONESTY GATE: 3 boundary lines -> <4 finite corners -> RuntimeError (no fabricated H)
def test_refuse_three_lines():
    d,img,w,K,dist,clicks,M=_real()
    three={k:clicks[k] for k in ("endline_x0","touchline_y0","touchline_yW")}
    try:
        PL.run_line_calibration(None,K,dist,three,L,W,"x",outdir=tempfile.mkdtemp())
        assert False,"3-line input should be refused"
    except RuntimeError:
        pass

if __name__=="__main__":
    for n,fn in sorted((k,v) for k,v in globals().items() if k.startswith("test_")):
        fn(); print("OK",n)
    print("OK: pick_lines acceptance passed")
