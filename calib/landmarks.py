#!/usr/bin/env python3
"""calib/landmarks.py — TEK KAYNAK: saha nokta-işaretleri (2D şema + tıklayıcı + solver ortak).

Konvansiyon (template.py ile aynı): origin sol-alt, X=boy [0,L], Y=en [0,W], metre.
Kamera görünümü: YAKIN kale = X=0 (sol), UZAK kale = X=L (sağ);
YAKIN kenar (kameraya yakın) = Y=0 (alt), ÜST/UZAK kenar = Y=W (üst).
Kale ağzı = 3 m (post = merkez ∓1.5). Orta nokta = (L/2, W/2).

CEZA SAHASI (penalty box): amatör sahada ölçü değişir → boyutu (bd=derinlik,
bw=en) SOLVER'da SERBEST aranır; aşağıdaki nominal değerler yalnız 2D-şema ve
tıklayıcıda göstermek için. Kutu çıpası güçlüdür çünkü dikdörtgen + kale-ekseni
etrafında simetrik + gol çizgisine paralel — boyutu bilinmese bile yakın yarıyı
sabitler.

Her işaret: (id, etiket_tr, world_xy, grup, genelde_kadraj_disi?).
grup: post/center/midline/corner/box.
"""

# ceza sahası nominal (yalnız gösterim; solver serbest arar)
BOX_BD_NOM = 5.0   # derinlik (gol çizgisinden öne, m)
BOX_BW_NOM = 9.0   # en (m, kale eksenine simetrik)


def base_landmarks(L: float, W: float):
    """Kutu-DIŞI sabit işaretler (boyut yalnız L,W'ye bağlı)."""
    cx, cy = L / 2.0, W / 2.0
    hg = 1.5  # yarı kale ağzı (3m/2)
    return [
        # id,                etiket,                                   world,         grup,     kadraj_disi
        ("near_post_y0", "YAKIN kale — ALT direk DİBİ (yere değdiği yer)",  (0.0, cy - hg), "post",   False),
        ("near_post_yW", "YAKIN kale — ÜST direk DİBİ (yere değdiği yer)",  (0.0, cy + hg), "post",   False),
        ("far_post_y0",  "UZAK kale — ALT direk dibi",                      (L,   cy - hg), "post",   False),
        ("far_post_yW",  "UZAK kale — ÜST direk dibi",                      (L,   cy + hg), "post",   False),
        ("center",       "ORTA NOKTA (santra)",                             (cx,  cy),      "center", False),
        ("mid_y0",       "ORTA ÇİZGİ × YAKIN kenar (alt uç)",               (cx,  0.0),     "midline",False),
        ("mid_yW",       "ORTA ÇİZGİ × ÜST kenar (üst uç)",                 (cx,  W),       "midline",False),
        ("corner_far_y0","KÖŞE — UZAK kale + YAKIN kenar (sağ-alt)",        (L,   0.0),     "corner", False),
        ("corner_far_yW","KÖŞE — UZAK kale + ÜST kenar (sağ-üst)",          (L,   W),       "corner", False),
        ("corner_near_y0","KÖŞE — YAKIN kale + YAKIN kenar (sol-alt)",      (0.0, 0.0),     "corner", True),
        ("corner_near_yW","KÖŞE — YAKIN kale + ÜST kenar (sol-üst)",        (0.0, W),       "corner", True),
    ]


def box_landmarks(L: float, W: float, bd: float = BOX_BD_NOM, bw: float = BOX_BW_NOM):
    """İki ceza-sahası köşeleri (YAKIN + UZAK). front = gol çizgisinden bd içeride."""
    cy = W / 2.0
    hw = bw / 2.0
    return [
        ("box_near_front_y0", "YAKIN ceza sahası — ÖN çizgi ALT köşe (sahanın içinde, net beyaz)", (bd,    cy - hw), "box", False),
        ("box_near_front_yW", "YAKIN ceza sahası — ÖN çizgi ÜST köşe",                              (bd,    cy + hw), "box", False),
        ("box_near_goal_y0",  "YAKIN ceza sahası — GOL çizgisi ALT köşe (kale dibinde)",            (0.0,   cy - hw), "box", False),
        ("box_near_goal_yW",  "YAKIN ceza sahası — GOL çizgisi ÜST köşe",                           (0.0,   cy + hw), "box", False),
        ("box_far_front_y0",  "UZAK ceza sahası — ÖN çizgi ALT köşe (karşı kale)",                  (L - bd, cy - hw), "box", False),
        ("box_far_front_yW",  "UZAK ceza sahası — ÖN çizgi ÜST köşe",                               (L - bd, cy + hw), "box", False),
        ("box_far_goal_y0",   "UZAK ceza sahası — GOL çizgisi ALT köşe",                            (L,     cy - hw), "box", False),
        ("box_far_goal_yW",   "UZAK ceza sahası — GOL çizgisi ÜST köşe",                            (L,     cy + hw), "box", False),
    ]


def landmarks(L: float, W: float, bd: float = BOX_BD_NOM, bw: float = BOX_BW_NOM):
    """Tüm işaretler (base + box). Tıklayıcı/2D bunu kullanır."""
    return base_landmarks(L, W) + box_landmarks(L, W, bd, bw)


BOX_IDS = ("box_near_front_y0", "box_near_front_yW", "box_near_goal_y0", "box_near_goal_yW",
           "box_far_front_y0", "box_far_front_yW", "box_far_goal_y0", "box_far_goal_yW")

# post çiftleri (ayna/atama için): goal -> (id_y0, id_yW)
POST_PAIRS = {"near": ("near_post_y0", "near_post_yW"),
              "far":  ("far_post_y0", "far_post_yW")}
