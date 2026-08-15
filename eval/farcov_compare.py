#!/usr/bin/env python3
"""farcov_compare.py — ft vs ft_v2 FAR-band tespit-kapsama karsilastirmasi (ayni harness).

recall_eval.py detect'in urettigi iki detections json'unu (ft, ft_v2) 3-kor-sayici
count-GT'ye (score_summary.json far n_humans) karsi kiyaslar. Nokta-GT yok -> DURUST
proxy: far-band base tespit sayisi (foot_y<360). precision guvencesi: 'coverage'
= min(far_dets, far_humans) frame basina (GT insan sayisini asan tespit = FP riski,
kapsamayi SISIRMEZ). Ham far_dets de yazilir (asim=FP isareti; overlay ile dogrula).

Kullanim: python eval/farcov_compare.py
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "recall_val"
FROZEN = [37307, 44769, 52230, 59692, 67153, 74615, 82076, 85807]
ZONE_T1 = 360.0


def far_counts(det, fi):
    d = det.get(str(fi), {})
    fb = sum(1 for x in d.get("base", []) if x["foot_y"] < ZONE_T1)
    ft = sum(1 for x in d.get("tiled", []) if x["foot_y"] < ZONE_T1)
    return fb, ft


def load(name):
    p = VAL / name
    return json.load(open(p)) if p.exists() else None


def main():
    gt = json.load(open(VAL / "score_summary.json"))["reconciled"]
    ftd = load("detections_ft.json")
    v2d = load("detections_ft_v2.json")
    if ftd is None or v2d is None:
        print("EKSIK: detections_ft.json / detections_ft_v2.json"); sys.exit(1)
    print(f"{'frame':>7} {'GTfar':>5} | {'ft_far':>6} {'ft_cov':>6} | {'v2_far':>6} {'v2_cov':>6} | {'d_far':>5}")
    tot = dict(gt=0, ft=0, ftcov=0, fttile=0, v2=0, v2cov=0, v2tile=0)
    for fi in FROZEN:
        gh = gt[str(fi)]["far"]["n_humans"]
        fb, ftl = far_counts(ftd, fi)
        vb, vtl = far_counts(v2d, fi)
        fcov = min(fb, gh); vcov = min(vb, gh)
        tot["gt"] += gh; tot["ft"] += fb; tot["ftcov"] += fcov; tot["fttile"] += ftl
        tot["v2"] += vb; tot["v2cov"] += vcov; tot["v2tile"] += vtl
        print(f"{fi:>7} {gh:>5} | {fb:>6} {fcov:>6} | {vb:>6} {vcov:>6} | {vb-fb:>+5}")
    g = tot["gt"]
    def pct(n): return f"{100.0*n/g:.1f}%"
    print("-" * 60)
    print(f"{'TOTAL':>7} {g:>5} | {tot['ft']:>6} {tot['ftcov']:>6} | {tot['v2']:>6} {tot['v2cov']:>6} | {tot['v2']-tot['ft']:>+5}")
    print()
    print(f"ft   far-cov (min-capped, precision-guarded): {tot['ftcov']}/{g} = {pct(tot['ftcov'])}   (raw far dets={tot['ft']}, tile+={tot['fttile']})")
    print(f"ft_v2far-cov (min-capped, precision-guarded): {tot['v2cov']}/{g} = {pct(tot['v2cov'])}   (raw far dets={tot['v2']}, tile+={tot['v2tile']})")
    dcov = tot["v2cov"] - tot["ftcov"]
    print(f"\nDELTA far-cov (ft_v2 - ft): {dcov:+d} oyuncu  ({100.0*dcov/g:+.1f} puan)")
    over_ft = tot["ft"] - tot["ftcov"]; over_v2 = tot["v2"] - tot["v2cov"]
    print(f"GT-asimi (FP riski, overlay ile dogrula): ft={over_ft}  ft_v2={over_v2}"
          f"   {'[v2 daha cok asiyor -> FP kontrolu SART]' if over_v2 > over_ft else '[v2 asim artmadi -> kazanc temiz]'}")


if __name__ == "__main__":
    main()
