#!/usr/bin/env python3
"""CEPHE3 görsel kanıt: (1) aktif-pencere kare-başı tespit sayısı önce/sonra
(spawn proxy), (2) insan-GT recall base vs +tile bar.  Tek PNG."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "scratchpad/wf2_recall"
st = json.loads((OUT / "active_reexport_stats.json").read_text())
pf = pd.DataFrame(st["per_frame"])
tw = json.loads((OUT / "recall_threeway.json").read_text())["human_gt"]

fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))

# (1) per-frame total count before/after
ax[0].plot(pf.frame, pf.base_total, lw=0.8, color="#888", label=f"base/cache  μ={pf.base_total.mean():.2f} σ={pf.base_total.std():.2f}")
ax[0].plot(pf.frame, pf.reexport_total, lw=0.8, color="#1f77b4", label=f"+tile+CLAHE  μ={pf.reexport_total.mean():.2f} σ={pf.reexport_total.std():.2f}")
ax[0].axhline(14, ls="--", c="r", lw=0.8, label="14 oyuncu")
ax[0].set_title("Aktif-pencere kare-başı tespit (spawn proxy)\nmin floor 8 DEĞİŞMEDİ -> far-tiling spawn'ı çözmüyor")
ax[0].set_xlabel("frame"); ax[0].set_ylabel("tespit / kare"); ax[0].legend(fontsize=7, loc="lower left")
ax[0].set_ylim(6, 18)

# (2) per-frame FAR count before/after
ax[1].plot(pf.frame, pf.base_far, lw=0.8, color="#888", label=f"base far  σ={pf.base_far.std():.2f}")
ax[1].plot(pf.frame, pf.reexport_far, lw=0.8, color="#2ca02c", label=f"+tile+CLAHE far  σ={pf.reexport_far.std():.2f}")
rec = pf[(pf.tile_rec + pf.clahe_rec) > 0]
ax[1].scatter(rec.frame, rec.reexport_far, s=14, color="#d62728", zorder=5, label=f"kurtarma noktası ({len(rec)} kare)")
ax[1].set_title(f"Far-band (y<365) önce/sonra\n+{int(pf.tile_rec.sum())} tile +{int(pf.clahe_rec.sum())} CLAHE = +{int((pf.tile_rec+pf.clahe_rec).sum())} det / {len(pf)} kare")
ax[1].set_xlabel("frame"); ax[1].set_ylabel("far tespit / kare"); ax[1].legend(fontsize=7, loc="lower left")

# (3) human-GT recall bars
zones = ["far", "rest", "all"]
base = [tw["far"]["base_pct"], tw["rest"]["base_pct"], tw["all"]["base_pct"]]
tile = [tw["far"]["tile_pct"], tw["rest"]["base_pct"], tw["all"]["tile_pct"]]
x = np.arange(3); w = 0.38
ax[2].bar(x - w / 2, base, w, color="#888", label="base")
ax[2].bar(x + w / 2, tile, w, color="#1f77b4", label="+tile (insan-doğrulanmış)")
for i, (b, t) in enumerate(zip(base, tile)):
    ax[2].text(i - w / 2, b + 1, f"{b:.0f}", ha="center", fontsize=8)
    ax[2].text(i + w / 2, t + 1, f"{t:.0f}", ha="center", fontsize=8)
ax[2].set_xticks(x); ax[2].set_xticklabels(["far\n(H=92)", "rest\n(H=15)", "all\n(H=107)"])
ax[2].set_ylim(0, 110); ax[2].set_ylabel("recall %")
ax[2].set_title("İNSAN-GT recall (3 kör sayıcı, 8 kare oyun-geneli)\nprecision=100% (fp=0)  CLAHE GT-atıfsız")
ax[2].legend(fontsize=8)

fig.tight_layout()
p = OUT / "recall_evidence.png"
fig.savefig(p, dpi=110)
print("wrote", p)
