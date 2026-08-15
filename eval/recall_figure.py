#!/usr/bin/env python3
"""recall_figure.py — per-frame ve toplam recall (base vs +tiling) figürü.

GT = recall_val/gt_labels.json (3 sayıcı/frame). Her frame için TOPLAM kapsama
(far+rest) sayıcı-başına hesaplanıp medyanı alınır (far/rest sınır-belirsizliğine
sağlam). Çıktı: docs/report/figures/recall.{pdf,png}
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent / "Templates/article-twocolumn/figures"))
import paper_style as ps  # noqa: E402

FROZEN = [37307, 44769, 52230, 59692, 67153, 74615, 82076, 85807]


def per_frame(labels):
    by = {}
    for o in labels:
        by.setdefault(o["_frame"], []).append(o)
    rows = {}
    for f, ls in by.items():
        H = [l["far"]["n_humans"] + l["rest"]["n_humans"] for l in ls]
        base = [l["far"]["n_base_covered"] + l["rest"]["n_base_covered"] for l in ls]
        tile = [l["far"]["n_tile_covered"] for l in ls]
        bt = [b + t for b, t in zip(base, tile)]
        rows[f] = dict(H=float(np.median(H)), base=float(np.median(base)),
                       bt=float(np.median(bt)))
    return rows


def main():
    import matplotlib.pyplot as plt
    ps.use()
    labels = json.load(open(ROOT / "recall_val/gt_labels.json"))
    pf = per_frame(labels)
    frames = [f for f in FROZEN if f in pf]
    base_r = [100 * pf[f]["base"] / pf[f]["H"] for f in frames]
    bt_r = [100 * pf[f]["bt"] / pf[f]["H"] for f in frames]
    # aggregate
    Hs = sum(pf[f]["H"] for f in frames)
    base_agg = 100 * sum(pf[f]["base"] for f in frames) / Hs
    bt_agg = 100 * sum(pf[f]["bt"] for f in frames) / Hs

    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    x = np.arange(len(frames)); w = 0.38
    ax.bar(x - w / 2, base_r, w, color=ps.C["slate"], label="base detektör")
    ax.bar(x + w / 2, bt_r, w, color=ps.C["blue"], label="base + far-band tiling")
    # tiling katkısını üstte göster
    for i in range(len(frames)):
        d = bt_r[i] - base_r[i]
        if d > 0.5:
            ax.annotate(f"+{d:.0f}", (x[i] + w / 2, bt_r[i] + 1), ha="center",
                        fontsize=7.5, color=ps.C["blue"])
    ax.axhline(base_agg, color=ps.C["slate"], ls="--", lw=1.0)
    ax.axhline(bt_agg, color=ps.C["blue"], ls="--", lw=1.0)
    ax.annotate(f"toplam base {base_agg:.0f}%", (len(frames) - 0.5, base_agg - 5),
                ha="right", fontsize=8, color=ps.C["slate"])
    ax.annotate(f"toplam +tiling {bt_agg:.0f}%", (len(frames) - 0.5, bt_agg + 1.5),
                ha="right", fontsize=8, color=ps.C["blue"])
    ax.set_xticks(x)
    ax.set_xticklabels([f"f{f}" for f in frames], rotation=30, fontsize=7.5)
    ax.set_ylabel("recall (%)"); ax.set_ylim(0, 105)
    ax.set_title("Tespit recall'ı: base vs far-band tiling (8 frozen kare, 3 sayıcı GT)")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    ps.save(fig, str(ROOT / "docs/report/figures/recall"))
    plt.close(fig)
    print(f"toplam base recall {base_agg:.1f}%  +tiling {bt_agg:.1f}%")
    for f in frames:
        print(f"  f{f}: base {100*pf[f]['base']/pf[f]['H']:.0f}%  +tile {100*pf[f]['bt']/pf[f]['H']:.0f}%  (GT {pf[f]['H']:.0f})")


if __name__ == "__main__":
    main()
