#!/usr/bin/env python3
"""zone_figure.py — saha bölgelerini (far/mid/near) NASIL seçtiğimizi gösteren figür.

Bölgeler GÖRÜNTÜ-uzayında (foot_y) tanımlanır, metrik-uzayda değil. Gerekçe:
sabit oblik kamerada oyuncunun GÖRÜNTÜ-BOYU (ve dolayısıyla tespit zorluğu) image
satırıyla (foot_y) tekdüze değişir — uzak oyuncu küçük+yukarıda, yakın oyuncu büyük+altta.
Metrik-zone'dan kaçınıldı çünkü far-third PIKSEL-FAKIRI (Jacobian ~0.12 m/px).

Çıktı: docs/report/figures/zones.{pdf,png}
  (a) gerçek kare + far/mid/near bandları + zone-renkli tespit kutuları
  (b) foot_y dağılımı (eşikler) + binlenmiş medyan kutu-yüksekliği (boy→zorluk gradyanı)
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
FIGDIR = ROOT / "docs/report/figures"
sys.path.insert(0, str(ROOT.parent / "Templates/article-twocolumn/figures"))
import paper_style as ps  # noqa: E402

ZONE_T1, ZONE_T2 = 360.0, 560.0   # eval/recall_eval.py ile birebir
FRAME = 44769                      # t=1800: oyuncular far+mid yayılmış (zone örneklemesi iyi)


def main():
    import matplotlib.pyplot as plt
    import pandas as pd
    FIGDIR.mkdir(parents=True, exist_ok=True)
    ps.use()

    frame = cv2.imread(str(ROOT / f"recall_val/frames/f{FRAME}.png"))[:, :, ::-1]
    det = json.loads((ROOT / "recall_val/detections.json").read_text())
    boxes = det[str(FRAME)]["base"] + det[str(FRAME)]["tiled"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.2, 2.85),
                                   gridspec_kw=dict(width_ratios=[1.55, 1.0]))
    # --- (a) kare + bandlar ---
    axA.imshow(frame)
    H, Wd = frame.shape[:2]
    zcol = {"far": ps.C["red"], "mid": ps.C["amber"], "near": ps.C["teal"]}
    for (y0, y1), name in [((0, ZONE_T1), "far"), ((ZONE_T1, ZONE_T2), "mid"),
                           ((ZONE_T2, H), "near")]:
        axA.axhspan(y0, y1, color=zcol[name], alpha=0.13)
        axA.text(18, (y0 + y1) / 2, name.upper(), color=zcol[name],
                 fontsize=11, va="center", fontweight="bold")
    for b in boxes:
        z = "far" if b["foot_y"] < ZONE_T1 else ("mid" if b["foot_y"] < ZONE_T2 else "near")
        axA.add_patch(plt.Rectangle((b["x1"], b["y1"]), b["x2"] - b["x1"], b["y2"] - b["y1"],
                      fill=False, ec=zcol[z], lw=1.3))
    for yy in (ZONE_T1, ZONE_T2):
        axA.axhline(yy, color="white", lw=0.8, ls="--")
    axA.set_xlim(0, Wd); axA.set_ylim(H, 0)
    axA.set_xticks([]); axA.set_yticks([])
    axA.set_title("(a) görüntü-uzayı bölgeleri (sabit kamera)")
    axA.grid(False)

    # --- (b) foot_y dağılımı + boy gradyanı ---
    df = pd.read_parquet(ROOT / "raw/tracks_cankaya_cam2_clip2400.parquet")
    fy = df["foot_y"].to_numpy(float)
    axB.hist(fy, bins=40, orientation="horizontal", color=ps.C["blue"], alpha=0.55,
             ec="white", label="tespit foot_y")
    axB.set_ylim(H, 0)
    for yy, nm in ((ZONE_T1, "far|mid"), (ZONE_T2, "mid|near")):
        axB.axhline(yy, color=ps.C["ink"], lw=1.0, ls="--")
        axB.text(axB.get_xlim()[1] * 0.98, yy - 8, nm, ha="right", fontsize=7.5, color=ps.C["ink"])
    axB.set_ylabel("foot_y (görüntü satırı)"); axB.set_xlabel("tespit sayısı")
    axB.set_title("(b) zorluk gradyanı")
    # ikiz: medyan box_h vs foot_y (oyuncu görüntü-boyu)
    ax2 = axB.twiny()
    bins = np.linspace(fy.min(), fy.max(), 14)
    bh = df["box_h"].to_numpy(float)
    idx = np.digitize(fy, bins)
    cy, cbh = [], []
    for i in range(1, len(bins)):
        m = idx == i
        if m.sum() > 20:
            cy.append(0.5 * (bins[i - 1] + bins[i])); cbh.append(np.median(bh[m]))
    ax2.plot(cbh, cy, color=ps.C["red"], lw=2.0, marker="o", ms=3)
    ax2.set_xlabel("medyan kutu yüksekliği (px)", color=ps.C["red"])
    ax2.tick_params(axis="x", colors=ps.C["red"])
    ax2.grid(False)

    fig.tight_layout()
    ps.save(fig, str(FIGDIR / "zones"))
    plt.close(fig)
    print(f"foot_y aralığı {fy.min():.0f}-{fy.max():.0f}; eşikler {ZONE_T1:.0f},{ZONE_T2:.0f}")
    print(f"box_h: far-band medyan {np.median(bh[fy<ZONE_T1]):.0f}px, "
          f"near-band medyan {np.median(bh[fy>=ZONE_T2]):.0f}px")


if __name__ == "__main__":
    main()
