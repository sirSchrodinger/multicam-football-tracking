"""cli.py — ince giriş noktası: tek cycle koş / durum göster.

loop.py orkestratörün kullanıcı-dostu sarmalı. Asıl mantık loop/measure/diagnose/
gate/ledger içinde; bu dosya sadece kısa komutları yönlendirir.

  python -m selfimprove.cli status            # ledger ilerleme özeti (insan-okunur)
  python -m selfimprove.cli run               # bir improve-local cycle (Track A, eğitimsiz)
  python -m selfimprove.cli run --mode measure
  python -m selfimprove.cli run --mode gt-request
  python -m selfimprove.cli run --mode stage  # Track B iş-spec'i (RunPod, GATED)
"""
from __future__ import annotations

import argparse
import json

from selfimprove import ledger as L
from selfimprove import loop


def _status() -> None:
    s = L.progress_summary()
    print("=" * 64)
    print("  SELF-IMPROVE LOOP — DURUM")
    print("=" * 64)
    print(f"  cycles={s['n_cycles']}  keep={s['n_keep']}  rollback={s['n_rollback']}")
    print(f"  promoted_weights = {s['current_promoted_weights']}")
    pc = s.get("current_promoted_config")
    if pc:
        print(f"  promoted_config  = recover_far={pc.get('recover_far')} "
              f"tile_thresh={pc.get('tile_thresh')} venue={pc.get('venue')} "
              f"gt_n={pc.get('gt_n')} far_recall={pc.get('heldout_far_recall')} "
              f"reversible={pc.get('reversible')}")
    else:
        print("  promoted_config  = (yok; recover_far OFF baseline)")
    if s.get("recall_base_trend"):
        print(f"  held-out base recall trend: {s['recall_base_trend']}")
    if s.get("config_promotions"):
        print(f"  config promotions: {json.dumps(s['config_promotions'], ensure_ascii=False)}")


def main() -> None:
    ap = argparse.ArgumentParser(description="self-improve loop CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    rp = sub.add_parser("run")
    rp.add_argument("--mode", default="improve-local",
                    choices=["measure", "improve-local", "gt-request", "stage", "gate"])
    rp.add_argument("--candidate")
    rp.add_argument("--candidate-metrics")
    rp.add_argument("--force-mine", action="store_true")
    args = ap.parse_args()

    if args.cmd == "status":
        _status()
        return
    rec = loop.run_cycle(args.mode, candidate=args.candidate,
                         candidate_metrics=args.candidate_metrics,
                         force_mine=args.force_mine)
    loop._print_cycle(rec)


if __name__ == "__main__":
    main()
