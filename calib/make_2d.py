#!/usr/bin/env python3
"""calib/make_2d.py — üstten 2D saha şeması (numaralı landmark + legend).

Tıklayıcı ile AYNI numaralandırma (landmarks.py). Alperen bunu referans alıp
gerçek görüntüde karşılık gelen noktayı basar; görmediğini atlar.
"""
import sys
sys.path.insert(0, ".")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from calib.landmarks import landmarks, BOX_BD_NOM, BOX_BW_NOM

L, W = 33.0, 18.0
LM = landmarks(L, W)

fig, ax = plt.subplots(figsize=(15, 9))
ax.set_facecolor("#0b6b2e")
fig.patch.set_facecolor("#1c1c1e")

# saha
ax.add_patch(Rectangle((0, 0), L, W, fill=False, ec="white", lw=2.5))
ax.plot([L/2, L/2], [0, W], color="white", lw=2)                       # orta çizgi
ax.add_patch(Circle((L/2, W/2), 3.0, fill=False, ec="white", lw=2))    # orta yuvarlak
ax.plot(L/2, W/2, "o", color="white", ms=5)
# kaleler (ağız 3m)
for gx, dx in [(0, 1.0), (L, -1.0)]:
    ax.add_patch(Rectangle((gx + (0 if dx > 0 else dx), W/2 - 1.5), abs(dx), 3.0,
                           fill=False, ec="#ffd60a", lw=2))
# İKİ ceza sahası da — TIKLANACAK (solid yeşil; boyut nominal, solver serbest arar)
ax.add_patch(Rectangle((0, W/2 - BOX_BW_NOM/2), BOX_BD_NOM, BOX_BW_NOM,
                       fill=False, ec="#34c759", lw=2.2))
ax.add_patch(Rectangle((L - BOX_BD_NOM, W/2 - BOX_BW_NOM/2), BOX_BD_NOM, BOX_BW_NOM,
                       fill=False, ec="#34c759", lw=2.2))

# kadraj-dışı köşe bölgeleri (yakın kale tarafı)
ax.add_patch(Rectangle((0, 0), 6, 4, color="red", alpha=0.13))
ax.add_patch(Rectangle((0, W-4), 6, 4, color="red", alpha=0.13))
ax.text(3, 2, "kadraj\ndışı?", color="#ffb3b3", ha="center", va="center", fontsize=8)
ax.text(3, W-2, "kadraj\ndışı?", color="#ffb3b3", ha="center", va="center", fontsize=8)

# landmark noktaları + numara
for i, (lid, label, (x, y), grp, off) in enumerate(LM, 1):
    col = {"post": "#af52de", "center": "#0a84ff", "midline": "#5856d6",
           "corner": "#ff9f0a", "box": "#34c759"}[grp]
    ax.plot(x, y, "o", color=col, ms=13, mec="white", mew=1.5, zorder=5)
    ax.text(x, y, str(i), color="white", ha="center", va="center",
            fontsize=9, fontweight="bold", zorder=6)

# kamera yönü
ax.annotate("KAMERA\n(yakın kenar = ALT)", xy=(2, -0.3), xytext=(2, -2.6),
            color="#0a84ff", ha="center", fontsize=11, fontweight="bold",
            arrowprops=dict(arrowstyle="-|>", color="#0a84ff", lw=2))
ax.text(L/2, -2.4, "← YAKIN kale (X=0)        UZAK kale (X=L) →",
        color="white", ha="center", fontsize=11)
ax.text(-1.2, W/2, "ÜST kenar", color="white", rotation=90, va="center", ha="center", fontsize=9)
ax.text(L/2, W + 0.8, "ÜST / UZAK kenar (Y=W)", color="white", ha="center", fontsize=10)
ax.text(L/2, -0.9, "YAKIN kenar — kameraya yakın (Y=0)", color="white", ha="center", fontsize=10)

# legend (sağda)
lines = []
for i, (lid, label, (x, y), grp, off) in enumerate(LM, 1):
    tag = "  [genelde KADRAJ DIŞI — görüyorsan bas]" if off else ""
    lines.append(f"{i:2d}.  {label}{tag}")
ax.text(L + 1.5, W, "\n".join(lines), color="white", va="top", ha="left",
        fontsize=10, family="monospace",
        bbox=dict(boxstyle="round", fc="#2c2c2e", ec="#3a3a3c"))

ax.set_xlim(-4, L + 22)
ax.set_ylim(-4, W + 2)
ax.set_aspect("equal")
ax.axis("off")
ax.set_title("HALISAHA — üstten 2D referans (numaralar tıklayıcıdaki ile aynı). "
             "Görmediğin noktayı ATLA.", color="white", fontsize=13)
fig.tight_layout()
fig.savefig("calib/field_2d.png", dpi=110, facecolor=fig.get_facecolor())
print("yazıldı: calib/field_2d.png")
