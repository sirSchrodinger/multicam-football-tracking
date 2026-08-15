import json, math, os, sys
# allow `venv/bin/python ingest/test_venue_registry.py` (plain script) from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.venue_registry import (parse_video_url, classify_environment,
    attach_calibration, build_from_scan, resolve_camera, rank, CameraRecord)

# 1) URL PARSING on the 3 real special rows (verified ground truth)
a = parse_video_url("https://s1.sosyalhalisaha.com/matches/disk4/shst373s12/build/20260608233002.A2-2.mp4")
assert a["facility_token"] == "shst373" and a["saha_code"] == "A2" and a["cam_index"] == 2
k = parse_video_url("https://s2.sosyalhalisaha.com/matches/disk4/shst373s12/build/20260608233002.K2-1.mp4")
assert k["facility_token"] == "shst373" and k["saha_code"] == "K2" and k["cam_index"] == 1
p = parse_video_url("https://s2.sosyalhalisaha.com/matches/disk8/shst399s12/build/20260608233001.1-1.mp4")
assert p["facility_token"] == "shst399" and p["saha_code"] == "1" and p["cam_index"] == 1

# 2) ENVIRONMENT cascade (token, then title fallback, then unknown — never default outdoor)
assert classify_environment("A","A2","Açık Saha 2")[0] == "outdoor"
assert classify_environment("K","K2","Kapalı Saha 2")[0] == "indoor"
# Pınar: filename token absent (.1-1), title 'Kapalı Saha' must still classify indoor
assert classify_environment(None,"1","Kapalı Saha")[0] == "indoor"
# bare numeric saha + empty title -> unknown, NOT outdoor
env, src, conf = classify_environment(None,"1","")
assert env == "unknown" and conf == 0.0

# 3) CALIB ACCEPTANCE GATE on the 3 real calib files
def link(path):
    r = CameraRecord(camera_id="t", venue_id="t"); attach_calibration(r, path); return r
good = link("calib/cankaya_cam2_v2.json")          # median 9.87, near 9.24<far*1.5
assert good.calib_status == "linked"
bad = link("calib/cankaya_cam2.json")               # median 65, near 111>far*1.5 -> backwards-H
assert bad.calib_status == "rejected"
draft = link("calib/cankaya_cam2_DRAFT.json")       # n_landmarks=4, per_zone.mid=NaN
assert draft.calib_status == "rejected"

# 4) HONESTY GATE: standard_7v7 guess can never claim meters; real dims+passing anchor can
g = CameraRecord(camera_id="g", venue_id="g"); g.resolve_dims()
assert g.dims_source == "standard_7v7" and abs(g.dims_confidence-0.25) < 1e-9
assert g.metric_claim_allowed is False                       # guess + no anchor -> relative
g.dims_source = "physical_pacing"; g.scale_anchor = {"type":"goal_width","passed":True}; g.resolve_dims()
assert g.metric_claim_allowed is True                        # 0.6 dims + passing anchor -> meters
g.scale_anchor = {"type":"goal_width","passed":False}; g.resolve_dims()
assert g.metric_claim_allowed is False                       # anchor fails -> back to relative

# 5) BUILD on the real 31-row scan: Serdivan yields 2 records (açık+kapalı) under 1 facility;
#    indoor excluded from measurable rank
reg = build_from_scan("cams_08062026.tsv", "list_0608.json", "/tmp/claude-1000/venues_test.json")
serd = [r for r in reg["cameras"].values() if r["facility_token"] == "shst373"]
envs = sorted(r["environment"] for r in serd)
assert envs == ["indoor","outdoor"]                          # both sahas, distinct env
top = rank(reg, measurable_only=True, top=20)
assert all(r["environment"] != "indoor" for r in top)        # indoor disqualified from meters demo
assert all(0.0 <= r["calibratability_score"] <= 1.5 for r in reg["cameras"].values())

# 6) BC ALIAS bridge to the existing short id used by export_tracks --calib
any_rec = next(iter(reg["cameras"].values()))
any_rec.setdefault("aliases", []).append("cankaya_cam2")
assert resolve_camera("cankaya_cam2", reg) is not None
print("venue_registry acceptance: ALL PASS")
