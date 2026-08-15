#!/usr/bin/env python3
"""recall_score.py — coklu-sayici (blind GT) etiketlerini uzlastir + recall hesapla.

Girdi: recall_val/gt_labels.json  (workflow ciktisi: her frame icin 3 lens-sayici)
       recall_val/detections.json (capraz-kontrol icin; recall etiketlerden gelir)

Her (frame, region={far,rest}) icin 3 sayicinin MEDYANI alinir (tek tembel/asiri
sayici outlier'i bastirilir). recall = base-kutu kaplayan GT-insan / GT-insan.
DURUST belirsizlik bandi: ambiguous (faint/dusuk-kontrast) GT'ler payda
belirsizligi -> [covered/H_all, covered/(H_all - A)].
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAL = ROOT / "recall_val"
FROZEN = [37307, 44769, 52230, 59692, 67153, 74615, 82076, 85807]


def med(xs):
    xs = [x for x in xs if x is not None]
    return None if not xs else st.median(xs)


def reconcile(labels):
    """labels: list of dicts (her biri bir sayici-frame). frame -> medyan-uzlastirilmis."""
    by_frame = {}
    for L in labels:
        by_frame.setdefault(L["_frame"], []).append(L)
    rec = {}
    for fi, ls in by_frame.items():
        out = {"n_counters": len(ls)}
        for reg in ("far", "rest"):
            fields = ["n_humans", "n_ambiguous", "n_base_covered", "n_uncovered", "n_false_pos"]
            if reg == "far":
                fields.append("n_tile_covered")
            out[reg] = {f: med([l[reg].get(f) for l in ls]) for f in fields}
            # ham sayilar (spread gormek icin)
            out[reg]["_raw_humans"] = sorted(l[reg].get("n_humans") for l in ls)
            out[reg]["_raw_base"] = sorted(l[reg].get("n_base_covered") for l in ls)
        rec[fi] = out
    return rec


def band(cov, H, A):
    if not H:
        return None, None
    lo = 100.0 * cov / H
    hi = 100.0 * cov / max(H - A, 1e-9) if (H - A) > 0 else 100.0
    return round(lo, 1), round(min(hi, 100.0), 1)


def main():
    labels = json.loads((VAL / "gt_labels.json").read_text())
    if isinstance(labels, dict) and "labels" in labels:
        labels = labels["labels"]
    rec = reconcile(labels)
    det = json.loads((VAL / "detections.json").read_text()) if (VAL / "detections.json").exists() else {}

    agg = {"far": dict(H=0, A=0, base=0, tile=0, unc=0, fp=0),
           "rest": dict(H=0, A=0, base=0, unc=0, fp=0)}
    print(f"\n{'frame':>7} {'reg':>4} {'GT(med)':>8} {'amb':>4} {'base':>5} "
          f"{'tile':>5} {'unc':>4} {'fp':>3} {'raw_GT':>10} {'detFar':>7}")
    for fi in FROZEN:
        if fi not in rec:
            print(f"{fi:>7}  (GT yok)"); continue
        r = rec[fi]
        det_far = sum(1 for d in det.get(str(fi), {}).get("base", []) if d["zone"] == "far") if det else None
        det_far_tile = len(det.get(str(fi), {}).get("tiled", [])) if det else None
        for reg in ("far", "rest"):
            x = r[reg]
            H = x["n_humans"] or 0; A = x["n_ambiguous"] or 0
            b = x["n_base_covered"] or 0; u = x["n_uncovered"] or 0; fp = x["n_false_pos"] or 0
            t = (x.get("n_tile_covered") or 0) if reg == "far" else 0
            agg[reg]["H"] += H; agg[reg]["A"] += A; agg[reg]["base"] += b
            agg[reg]["unc"] += u; agg[reg]["fp"] += fp
            if reg == "far": agg[reg]["tile"] += t
            extra = f"{det_far}+{det_far_tile}" if (reg == "far" and det) else ""
            print(f"{fi:>7} {reg:>4} {H:>8} {A:>4} {b:>5} {t:>5} {u:>4} {fp:>3} "
                  f"{str(x['_raw_humans']):>10} {extra:>7}")

    print("\n================  AGGREGATE RECALL  (medyan-uzlastirilmis GT)  ================")
    for reg in ("far", "rest"):
        a = agg[reg]
        lo, hi = band(a["base"], a["H"], a["A"])
        print(f"[{reg.upper():4}] GT={a['H']} (ambiguous {a['A']})  base_covered={a['base']}  "
              f"uncovered={a['unc']}  FP={a['fp']}")
        print(f"        base recall = {lo}%–{hi}%   (primary {lo}% = all-counted-real)")
        if reg == "far":
            lo2, hi2 = band(a["base"] + a["tile"], a["H"], a["A"])
            print(f"        +tiling     = {lo2}%–{hi2}%   (tiling net +{a['tile']} recovered, "
                  f"delta {round(lo2-lo,1)} pts)")
    # overall
    H = agg["far"]["H"] + agg["rest"]["H"]; A = agg["far"]["A"] + agg["rest"]["A"]
    base = agg["far"]["base"] + agg["rest"]["base"]
    tile = agg["far"]["tile"]
    lo, hi = band(base, H, A); lo2, hi2 = band(base + tile, H, A)
    print(f"\n[ALL ] GT={H} (amb {A})  base recall = {lo}%–{hi}%  |  base+tiling = {lo2}%–{hi2}%")
    fp = agg["far"]["fp"] + agg["rest"]["fp"]
    det_total = base + fp  # kaba: kaplanan + FP ~ base tespit sayisi
    print(f"        FP toplam={fp}  ->  precision ~{round(100*base/max(det_total,1),1)}% (kaba)")
    (VAL / "score_summary.json").write_text(json.dumps(
        dict(reconciled=rec, aggregate=agg), indent=2, default=str))
    print("\nscore_summary.json yazildi")


if __name__ == "__main__":
    main()
