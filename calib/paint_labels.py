#!/usr/bin/env python3
"""calib/paint_labels.py — paint aracı + solver ORTAK etiket/dünya tanımı.

Alperen serbest çizer: ÇİZGİ (görünen beyaz çizgileri trace) + NOKTA (net gördüğü
işaretler). Solver çizgileri kesiştirerek GÖRÜNMEYEN köşeleri (kesik near köşeler)
geri kazanır: near_goalline ∩ near_touchline = (0,0) köşe — kadrajda olmasa bile.

Konvansiyon (template.py/landmarks.py ile aynı): origin sol-alt, X=boy [0,L] (YAKIN
kale X=0 sol, UZAK kale X=L sağ), Y=en [0,W] (YAKIN/kamera kenarı Y=0 alt, UZAK
kenar Y=W üst). Kale ağzı = 3m → direk = merkez ∓1.5. Ceza sahası bd,bw SERBEST.
"""

# ÇİZGİLER — her biri sabit bir dünya-doğrusu üstünde (bd,bw'den bağımsız).
# kind: "X" => sabit X (dik çizgi), val = X katsayısı (L cinsinden: 0, 0.5, 1)
#       "Y" => sabit Y (yatay çizgi), val = Y katsayısı (W cinsinden: 0, 1)
LINES = [
    # id,              etiket_tr,                                            renk,       kind, val
    ("near_goalline",  "YAKIN kale çizgisi (kalenin durduğu dik çizgi)",     "#ff2d55",  "X",  0.0),
    ("midline",        "ORTA saha çizgisi (santra çizgisi)",                 "#0a84ff",  "X",  0.5),
    ("far_goalline",   "UZAK kale çizgisi (karşı kale dik çizgisi)",         "#ff6482",  "X",  1.0),
    ("near_touchline", "YAKIN yan çizgi (sana en yakın uzun kenar, alt)",    "#32d74b",  "Y",  0.0),
    ("far_touchline",  "UZAK yan çizgi (üst uzun kenar)",                    "#64d2ff",  "Y",  1.0),
]

# NOKTALAR — net görünen işaretler. world fn (L,W,bd,bw).
def _pt_world(L, W, bd, bw):
    cx, cy, hg = L / 2.0, W / 2.0, 1.5
    hw = bw / 2.0
    return {
        "center":           (cx, cy),
        "near_post_y0":     (0.0, cy - hg),
        "near_post_yW":     (0.0, cy + hg),
        "far_post_y0":      (L,   cy - hg),
        "far_post_yW":      (L,   cy + hg),
        "box_front_y0":     (bd,  cy - hw),
        "box_front_yW":     (bd,  cy + hw),
    }

POINTS = [
    ("center",       "ORTA NOKTA (santra noktası)",                 "#0a84ff"),
    ("near_post_y0", "YAKIN kale — ALT direk DİBİ (yere değdiği)",  "#af52de"),
    ("near_post_yW", "YAKIN kale — ÜST direk DİBİ",                 "#af52de"),
    ("far_post_y0",  "UZAK kale — ALT direk dibi",                  "#bf5af2"),
    ("far_post_yW",  "UZAK kale — ÜST direk dibi",                  "#bf5af2"),
    ("box_front_y0", "YAKIN ceza sahası — ÖN çizgi ALT köşe",       "#34c759"),
    ("box_front_yW", "YAKIN ceza sahası — ÖN çizgi ÜST köşe",       "#34c759"),
]

# çizgi-kesişim → dünya köşe noktaları (X-çizgisi × Y-çizgisi). GÖRÜNMEYEN köşeyi kurtarır.
def intersections_world(L, W):
    """{(xline_id, yline_id): (Xworld, Yworld)} — her geçerli kesişim."""
    xl = {i: val * L for (i, _, _, k, val) in LINES if k == "X"}
    yl = {i: val * W for (i, _, _, k, val) in LINES if k == "Y"}
    out = {}
    for xi, X in xl.items():
        for yi, Y in yl.items():
            out[(xi, yi)] = (X, Y)
    return out


def point_world(label, L, W, bd, bw):
    return _pt_world(L, W, bd, bw).get(label)
