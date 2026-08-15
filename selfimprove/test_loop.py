"""test_loop.py — kapalı-döngü kontrolörün DÜRÜSTLÜK invariantları.

Bu testler "geliştirdi" iddiasının gerçekten gate'e bağlı olduğunu ve route'un
recall-öncelikli + kök-neden-farkında olduğunu kilitler. CPU-only, GPU yok.
"""
from __future__ import annotations

from selfimprove import diagnose as D
from selfimprove import gate as G
from selfimprove import recovery_search as RS


def _sc(recall_base, far_base, axes):
    return {"target": "t", "proxy_axes": axes,
            "proxy_overall": 70, "proxy_limiting_factor": min(axes, key=lambda k: axes[k]["score"]),
            "recall_heldout": {"base": recall_base, "far_base": far_base,
                               "base_tile": recall_base, "far_base_tile": far_base,
                               "precision_fp": 0, "n_gt_confident": 107},
            "gate_ready": False}


def test_recall_primary_overrides_proxy():
    # proxy 'identity' en düşük (61) AMA held-out recall hedefin altında -> recall objective
    axes = {"detection": {"score": 79}, "identity": {"score": 61},
            "calibration": {"score": 80},
            "motion": {"score": 61, "off_field_pct": 9, "impossible_step_pct": 2}}
    act = D.diagnose(_sc(0.73, 0.68, axes))
    assert act["objective"] == "recall", act
    assert act["needs_gpu_train"] is True
    assert "far" in act["corrective_module"]  # far<0.85 -> far-band vurgusu


def test_recall_met_falls_to_proxy_root():
    # recall hedefte -> proxy zayıf köke düş
    axes = {"detection": {"score": 92}, "identity": {"score": 55},
            "calibration": {"score": 88},
            "motion": {"score": 70, "off_field_pct": 3, "impossible_step_pct": 1}}
    act = D.diagnose(_sc(0.93, 0.90, axes))
    assert act["objective"] == "identity"
    assert act["root_cause"] == "identity_stitch"


def test_motion_symptom_splits_to_root():
    # motion en düşük; off_field baskın -> KÖK calibration (semptom değil)
    axes = {"detection": {"score": 92}, "identity": {"score": 80},
            "calibration": {"score": 75},
            "motion": {"score": 40, "off_field_pct": 12, "impossible_step_pct": 2}}
    act = D.diagnose(_sc(0.93, 0.90, axes))
    assert act["root_cause"] == "calibration", act
    # teleport baskın olursa kök identity
    axes2 = dict(axes); axes2["motion"] = {"score": 40, "off_field_pct": 2,
                                           "impossible_step_pct": 9}
    act2 = D.diagnose(_sc(0.93, 0.90, axes2))
    assert act2["root_cause"] == "identity_stitch", act2


def test_gate_blocks_without_loso():
    base = {"overall": {"ap": 0.71, "recall": 0.74}, "per_stadium": {}}
    new = {"overall": {"ap": 0.80, "recall": 0.85}, "per_stadium": {}}
    v = G.decide(new, base, ["cankaya_cam2"])  # tek saha
    assert v["decision"] == "blocked"
    assert v["gate_ready"] is False


def test_gate_catches_per_stadium_collapse():
    base = {"overall": {"ap": 0.71, "recall": 0.74},
            "per_stadium": {"a": {"ap": 0.70}, "b": {"ap": 0.72}}}
    collapse = {"overall": {"ap": 0.74, "recall": 0.78},
                "per_stadium": {"a": {"ap": 0.62}, "b": {"ap": 0.86}}}
    v = G.decide(collapse, base, ["a_venue", "b_venue"])
    assert v["decision"] == "rollback"
    assert any("stadium_collapse" in r for r in v["reasons"])


def test_gate_keeps_clean_improvement():
    base = {"overall": {"ap": 0.71, "recall": 0.74},
            "per_stadium": {"a": {"ap": 0.70}, "b": {"ap": 0.72}},
            "calib": {"median_reproj_px": 9.9}}
    good = {"overall": {"ap": 0.76, "recall": 0.80},
            "per_stadium": {"a": {"ap": 0.75}, "b": {"ap": 0.77}},
            "calib": {"median_reproj_px": 9.7}}
    v = G.decide(good, base, ["a_venue", "b_venue"])
    assert v["decision"] == "keep"
    assert v["deltas"]["ap"] > 0


# --------------------------------------------------------------------------- #
# Track A graft (recovery_search / config_gate / 3-yol routing) dürüstlük testleri
# --------------------------------------------------------------------------- #
def test_config_gate_keeps_far_gain_no_regression():
    base = {"far_recall": 0.68, "rest_recall": 1.0, "fp": 0, "gt_n": 40, "n_frames": 3}
    cand = {"far_recall": 0.79, "rest_recall": 1.0, "fp": 0, "gt_n": 40, "n_frames": 3}
    v = G.config_gate(cand, base)
    assert v["decision"] == "keep"
    assert v["deltas"]["far_recall"] > 0


def test_config_gate_rolls_back_rest_drop():
    # far ARTSA bile rest düşerse REDDET (B2B güven > marjinal kazanç)
    base = {"far_recall": 0.68, "rest_recall": 1.0, "fp": 0}
    cand = {"far_recall": 0.85, "rest_recall": 0.90, "fp": 0}
    v = G.config_gate(cand, base)
    assert v["decision"] == "rollback"
    assert any("rest_recall_drop" in r for r in v["reasons"])


def test_config_gate_rolls_back_fp_worse():
    base = {"far_recall": 0.68, "rest_recall": 1.0, "fp": 0}
    cand = {"far_recall": 0.85, "rest_recall": 1.0, "fp": 3}
    v = G.config_gate(cand, base)
    assert v["decision"] == "rollback"
    assert any("fp_worse" in r for r in v["reasons"])


def test_config_gate_rolls_back_no_gain():
    base = {"far_recall": 0.79, "rest_recall": 1.0, "fp": 0}
    cand = {"far_recall": 0.79, "rest_recall": 1.0, "fp": 0}
    v = G.config_gate(cand, base)
    assert v["decision"] == "rollback"
    assert any("no_far_gain" in r for r in v["reasons"])


def test_recovery_search_split_is_disjoint_and_real():
    # dev/test ayrık + recover_far ON dev'de OFF'tan iyi (gerçek frozen-GT)
    out = RS.search()
    dev, test = set(out["split"]["dev"]), set(out["split"]["test"])
    assert dev.isdisjoint(test)
    assert len(dev) == 5 and len(test) == 3
    assert out["best_cfg"]["recover_far"] is True
    assert out["dev_far_recall"] > out["off_dev_metrics"]["far_recall"]
    assert out["gt_n"] == 107  # 92 far + 15 rest confident-GT


def test_recovery_search_box_knobs_not_scored():
    # box-GT gerektiren ince knob'lar count-GT'de SKIP (sahte delta YASAK)
    out = RS.search(grid=RS.default_grid() + RS.default_grid_box())
    assert len(out["skipped_box_gt"]) == len(RS.default_grid_box())
    assert all(s["reason"] == "needs_point_gt" for s in out["skipped_box_gt"])


def test_diagnose_routes_to_track_a_when_recover_not_promoted():
    sc = {"target": "t", "proxy_axes": {"detection": {"score": 79}},
          "recall_heldout": {"base": 0.73, "far_base": 0.68}, "gate_ready": False}
    ctx = {"recover_far_promoted": False, "frozen_gt_available": True, "n_gt_venues": 1}
    act = D.diagnose(sc, context=ctx)
    assert act["corrective_module"] == "recovery_search"
    assert act["ray"] == "A" and act["needs_gpu_train"] is False


def test_diagnose_routes_to_gt_request_when_track_a_exhausted():
    sc = {"target": "t", "proxy_axes": {"detection": {"score": 79}},
          "recall_heldout": {"base": 0.82, "far_base": 0.79}, "gate_ready": False}
    ctx = {"recover_far_promoted": True, "frozen_gt_available": True, "n_gt_venues": 1}
    act = D.diagnose(sc, context=ctx)
    assert act["corrective_module"] == "gt_request"
    assert act["ray"] == "gt"


def test_diagnose_backward_compatible_without_context():
    # tek-arg çağrı (eski davranış) korunur: recall -> Track B çerçevesi
    sc = {"target": "t", "proxy_axes": {"detection": {"score": 79}},
          "recall_heldout": {"base": 0.73, "far_base": 0.68}, "gate_ready": False}
    act = D.diagnose(sc)
    assert act["objective"] == "recall"
    assert act["needs_gpu_train"] is True
    assert "far" in act["corrective_module"]
