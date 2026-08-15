"""loop.py — kapalı-döngü orkestratör (kendi-kendini geliştiren modül).

Tek cycle:  MEASURE -> DIAGNOSE -> (MINE+IMPROVE.stage) -> (GATE) -> LEDGER.

Çalıştırma modları:
  measure   : measure+diagnose+ledger (model değiştirmez; cycle-0 için güvenli).
  stage     : yukarısı + recall objective ise korpus yenile + RunPod iş-spec'i emit
              (LEDGER'a 'staged'; HİÇBİR iyileşme iddiası yok).
  gate      : RunPod'dan dönen aday ağırlığı held-out LOSO'da kıyasla -> keep/rollback.
  status    : ledger ilerleme özeti.

DÜRÜSTLÜK: hiçbir mod "geliştirdi" demez; sadece gate KEEP + held-out delta LEDGER'a
yazıldığında ilerleme kanıtlanmış olur. Route proxy'den, gate insan-GT'den.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from selfimprove import diagnose as D  # noqa: E402
from selfimprove import gate as G  # noqa: E402
from selfimprove import improve as I  # noqa: E402
from selfimprove import ledger as L  # noqa: E402
from selfimprove import measure as M  # noqa: E402
from selfimprove import mine as MINE  # noqa: E402
from selfimprove import recovery_search as RS  # noqa: E402

# Varsayılan target: aktif-pencere Çankaya koşusu (mevcut artefaktlar).
ACTIVE = ROOT / "stats_out/active_game"
GT_LABELS = ROOT / "recall_val/gt_labels.json"
DEFAULT_TARGET = {
    "target": "cankaya_active",
    "tracks_path": str(ACTIVE / "tracks_active_game_presplit_player_bridged.parquet"),
    "state_path": str(ACTIVE / "state14_clean.parquet"),
    "calib_path": str(ROOT / "calib/cankaya_cam2_FINAL.json"),
    "recall_summary": str(ROOT / "recall_val/score_summary.json"),
}


def _diag_context(state: dict) -> dict:
    """diagnose 3-yol yönlendirmesi için loop state'inden bağlam."""
    pc = state.get("promoted_config") or {}
    return {
        "recover_far_promoted": bool(pc.get("recover_far")),
        "frozen_gt_available": GT_LABELS.exists(),
        "n_gt_venues": len(set(state.get("gt_venues", []))),
    }


def _measure_and_diagnose(tgt: dict, state: dict) -> tuple[dict, dict]:
    sc = M.measure(gt_venues=state.get("gt_venues", []), **tgt)
    act = D.diagnose(sc, context=_diag_context(state))
    return sc, act


def run_cycle(mode: str, tgt: dict | None = None,
              candidate: str | None = None,
              candidate_metrics: str | None = None,
              force_mine: bool = False) -> dict:
    tgt = tgt or DEFAULT_TARGET
    state = L.load_state()
    gt_venues = state.get("gt_venues", [])

    sc, act = _measure_and_diagnose(tgt, state)
    entry: dict = {"mode": mode, "scorecard": sc, "action": act}

    # ---- IMPROVE-LOCAL (RAY A): eğitimsiz config araması (frozen-GT dev/test gate)
    if mode == "improve-local":
        cand = RS.search(venue=(gt_venues[0] if gt_venues else "cankaya_cam2"))
        entry["recovery"] = cand
        cand_test = cand.get("best_test_metrics")
        # taban = hâlihazırda promote edilmiş config (yoksa recover_far OFF)
        base_test = (cand.get("best_test_metrics") if (state.get("promoted_config") or {}).get("recover_far")
                     else cand.get("off_test_metrics"))
        if cand_test is None:
            entry["gate"] = {"decision": "blocked", "reasons": ["no_scoreable_candidate"]}
        else:
            v = G.config_gate(cand_test, base_test)
            entry["gate"] = v
            if v["decision"] == "keep":
                L.promote_config(cand["best_cfg"], cand,
                                 venue=cand.get("venue", ""),
                                 reason=act.get("rationale", ""))
                entry["promoted_config"] = True
                entry["reversible"] = True

    # ---- GT-REQUEST: 2. saha blind-GT etiket kuyruğu (Track B kilidini açar)
    elif mode == "gt-request":
        budget = 20
        q = MINE.label_queue(budget=budget)
        entry["gt_request"] = q
        entry["gate"] = {"decision": "not_run", "gate_ready": False,
                         "reasons": ["awaiting_2nd_venue_blind_gt"]}

    # ---- STAGE (RAY B): recall objective ise korpus + RunPod iş-spec'i
    elif mode == "stage" and act.get("needs_gpu_train"):
        corpus = MINE.far_band_corpus(
            emphasize_far=act.get("params", {}).get("emphasize_far", True),
            force=force_mine)
        job = I.stage_runpod_job(corpus, reason=act.get("rationale", ""))
        entry["mine"] = corpus
        entry["improve"] = job
        entry["gate"] = {"decision": "not_run", "gate_ready": sc.get("gate_ready"),
                         "reasons": ["loso_not_ready"]}

    # ---- GATE (RAY B): RunPod'dan dönen aday ağırlık var mı?
    elif mode == "gate":
        new_metrics = json.loads(Path(candidate_metrics).read_text()) if candidate_metrics else {}
        verdict = G.decide(new_metrics, state.get("baseline_metrics"), gt_venues)
        entry["gate"] = verdict
        entry["candidate"] = candidate
        if verdict["decision"] == "keep" and candidate:
            L.promote(candidate, new_metrics, reason=act.get("rationale", ""))
            entry["promoted"] = candidate

    else:
        # measure modu (ya da stage'de gpu_train gerekmiyorsa): gate çalışmaz
        entry["gate"] = {"decision": "not_run", "gate_ready": sc.get("gate_ready"),
                         "reasons": ["no_candidate" if sc.get("gate_ready")
                                     else "loso_not_ready"]}

    rec = L.append(entry)
    return rec


def _print_cycle(rec: dict) -> None:
    sc = rec["scorecard"]; act = rec["action"]; gt = rec.get("gate", {})
    rh = sc.get("recall_heldout") or {}
    print("=" * 64)
    print(f"  CYCLE {rec['cycle']}  [{rec['mode']}]  target={sc['target']}  {rec['ts']}")
    print("=" * 64)
    print(f"  proxy_overall={sc.get('proxy_overall')}/100  "
          f"proxy_limiting={sc.get('proxy_limiting_factor')}")
    for ax, d in sc.get("proxy_axes", {}).items():
        print(f"    [{ax:11}] {d.get('score'):>3}/100  {d.get('note','')}")
    if rh:
        print(f"  HELD-OUT RECALL (insan-GT, n={rh.get('n_gt_confident')}): "
              f"base {rh['base']:.0%}  +tile {rh['base_tile']:.0%}  "
              f"far {rh['far_base']:.0%}->+tile {rh['far_base_tile']:.0%}  "
              f"FP={rh['precision_fp']}  venues_with_gt={rh['n_venues_with_gt']}")
    print(f"  DIAGNOSE -> objective={act['objective']}  root={act['root_cause']}  "
          f"module={act['corrective_module']}  ray={act.get('ray','-')}  "
          f"gpu_train={act['needs_gpu_train']}")
    print(f"    {act['rationale']}")
    if "recovery" in rec:
        rv = rec["recovery"]; sp = rv.get("split", {})
        print(f"  RECOVERY-SEARCH (Track A, eğitimsiz; venue={rv.get('venue')} "
              f"gt_n={rv.get('gt_n')}):")
        print(f"    split dev={sp.get('dev')} test={sp.get('test')}")
        off = rv.get("off_test_metrics", {}) or {}
        print(f"    best_cfg={rv.get('best_cfg')}")
        print(f"    DEV far_recall={rv.get('dev_far_recall')}  ->  "
              f"HELD-OUT(test) far={rv.get('heldout_far_recall')} "
              f"(OFF taban far={off.get('far_recall')})  rest={rv.get('heldout_rest_recall')} "
              f"prec={rv.get('heldout_precision')} fp_delta={rv.get('fp_delta')}")
        kf = rv.get("kfold", {})
        print(f"    k-fold far-delta mean={kf.get('far_delta_mean')} "
              f"min={kf.get('far_delta_min')} all_positive={kf.get('all_positive')}")
        if rv.get("skipped_box_gt"):
            print(f"    (box-GT knob'ları SKIP: {len(rv['skipped_box_gt'])} cfg needs_point_gt)")
    if "mine" in rec:
        m = rec["mine"]
        print(f"  MINE: {m.get('status')}  images={m.get('n_images')} boxes={m.get('n_boxes')}"
              f"  val={m.get('val_venues')}")
    if "improve" in rec:
        print(f"  IMPROVE (staged): {rec['improve']['spec_path']}")
        print(f"    deploy: {rec['improve']['deploy_cmd']}")
    if "gt_request" in rec:
        q = rec["gt_request"]
        print(f"  GT-REQUEST: venue={q.get('venue')} status={q.get('status')} "
              f"n_frames={q.get('n_frames')} -> {len(q.get('queue', []))} kare etiketle")
        for r in (q.get("queue") or [])[:6]:
            print(f"    frame {r['frame']:>6}  deficit={r['deficit']} far={r['far_count']} "
                  f"odd={r['odd']} conf={r['median_conf']} score={r['score']}")
    print(f"  GATE: {gt.get('decision')}  gate_ready={gt.get('gate_ready')}  "
          f"{gt.get('reasons')}")
    if gt.get("deltas"):
        print(f"    deltas={gt['deltas']}")
    if rec.get("promoted_config"):
        print(f"    >>> CONFIG PROMOTED (reversible): {L.load_state().get('promoted_config')}")
    if gt.get("note"):
        print(f"    {gt['note']}")


def main():
    ap = argparse.ArgumentParser(description="self-improving control loop")
    ap.add_argument("mode", choices=["measure", "improve-local", "gt-request",
                                     "stage", "gate", "status"])
    ap.add_argument("--candidate", help="aday ağırlık yolu (gate modu)")
    ap.add_argument("--candidate-metrics", help="aday held-out metrik json (gate modu)")
    ap.add_argument("--force-mine", action="store_true", help="korpusu zorla yeniden kur")
    args = ap.parse_args()

    if args.mode == "status":
        print(json.dumps(L.progress_summary(), indent=2, ensure_ascii=False))
        return
    rec = run_cycle(args.mode, candidate=args.candidate,
                    candidate_metrics=args.candidate_metrics,
                    force_mine=args.force_mine)
    _print_cycle(rec)


if __name__ == "__main__":
    main()
