#!/usr/bin/env python3
"""height_scale_mode — populasyon boy-MODU metrik olcek + imkansiz-katalog dedektoru.

NEDEN (Alperen'in cizgi-egrisi/bell-curve fikri, DOGRU yapilmis hali):
  Sahadaki on-larca insan evrensel bir olcu-cubugudur. Ama her per-oyuncu
  geri-projekte boy olcumu AYNI degildir: kimi DIK durur (~1.75 m), kimi
  cömelir/egilir/ziplar -> dagilim Gauss DEGIL, SOLA carpiktir (crouch alt-kuyrugu).
  Bu yuzden ORTALAMA (mean) yanli, sistematik olarak DUSUKTUR. Yetiskin-dik
  olcegini dagilimin TEPESI (MODE) verir: en sik gozlemlenen duruslu boy.

YONTEM (gozetim-kamera oto-kalibrasyonu, BMVC 2011):
  Jingchen Liu, Robert T. Collins, Yanxi Liu, "Surveillance Camera Autocalibration
  based on Pedestrian Height Distributions", BMVC 2011. Yaya boylarinin dagiliminin
  TEPESI populasyonun tipik yetiskin-dik boyuna karsilik gelir; bu mod kameranin
  metrik olcegini sabitler. Biz iki yol sunariz:
    (a) MODE = KDE tepesi (scipy.stats.gaussian_kde, mesafe-agirlikli),
    (b) 2-bilesenli GMM (sklearn) -> baskin (en cok agirlikli) bilesenin merkezi =
        'standing' kume; digeri 'non-standing' (cömelme/egilme) kuyrugu.
  Duruslu-kapi (standing-gate): foot velocity ~0 (ayak-noktasi stabil/yerde) +
  dik bbox en-boy orani (upright) verilirse olcumler bu kosullara gore filtrelenir.
  Mesafe-agirligi: kameraya yakin (daha guvenilir) olcumler daha agir tartilir.

IMKANSIZ-KATALOG (impossible catalog) DEDEKTORU:
  Bir aday saha-boyutu katalogu (orn. 46x24) yanlissa, kale-genisliginden cikan
  olcek (scale_from_goal) ile boy-modundan cikan olcek (scale_from_height)
  CELISIR ve/veya implied saha 'imkansiz' insanlara (orn. 2.4 m) yol acar.
  Bayrak: iki olcek orani >%~10 sapma VEYA implied boyut futsal araliginin disinda
  (uzunluk 25-42 m, genislik 16-25 m).

DURUSTLUK / SINIRLAR:
  - scale bir VARSAYIMDAN gelir: populasyon yetiskin-dik boyu ~1.75 m. Bu yuzden
    sonuc 'approximate'tir; mutlak metre iddia edilmeden once on-site bilinen-mesafe
    ile dogrulanmalidir (bkz. pitch/height_scale.scale_from_known_distance).
  - MODE, tepe iyi-orneklenmisse (yeterli DIK olcum) saglamdir; cok az veride
    veya tek-degerli girdide KDE coker -> histogram-modu fallback'i devreye girer.
  - GMM 'standing' secimi BASKIN-AGIRLIK kuralina dayanir: oyuncular sahada
    cogunlukla dik durur/kosar -> en kalabalik kume duruslu boydur. Ziplama gibi
    nadir kuyruklar baskin olmadigi icin secimi bozmaz.

Lisans: yalnizca numpy + scipy + scikit-learn (BSD). GPU/dis-model YOK.
"""
from __future__ import annotations

import numpy as np

# Populasyon yetiskin-dik boyu (karma yetiskin); height_scale.py ile tutarli.
ASSUMED_STANDING_M = 1.75

# Futsal saha sinirlari (FIFA futsal: 25-42 m x 16-25 m). implied-dim sanity.
FUTSAL_L_BOUNDS = (25.0, 42.0)
FUTSAL_W_BOUNDS = (16.0, 25.0)


# --------------------------------------------------------------------------- #
#  Yardimcilar
# --------------------------------------------------------------------------- #
def _clean(arr):
    """np.ndarray(float) + sonlu maske. (degerler, maske) doner."""
    a = np.asarray(arr, dtype=np.float64).ravel()
    m = np.isfinite(a)
    return a, m


def _weighted_mean(h, w):
    s = float(np.sum(w))
    if s <= 0:
        return float(np.mean(h)) if h.size else float("nan")
    return float(np.sum(h * w) / s)


def _kde_mode(h, w, grid_n=1024, bw_method=None):
    """KDE tepesi (mesafe-agirlikli). Tek-degerli/dejenere girdide median fallback.

    gaussian_kde 'weights' destekler -> mesafe-agirligi dogrudan moda yansir.
    """
    from scipy.stats import gaussian_kde

    if h.size == 0:
        return float("nan")
    if h.size < 3 or np.unique(np.round(h, 9)).size < 2:
        # KDE icin yetersiz cesitlilik -> agirlikli median (saglam tepe vekili)
        return _weighted_median(h, w)
    try:
        kde = gaussian_kde(h, weights=w, bw_method=bw_method)
    except (np.linalg.LinAlgError, ValueError):
        return _weighted_median(h, w)
    lo, hi = float(h.min()), float(h.max())
    pad = 0.05 * (hi - lo + 1e-9)
    grid = np.linspace(lo - pad, hi + pad, grid_n)
    dens = kde(grid)
    return float(grid[int(np.argmax(dens))])


def _weighted_median(h, w):
    if h.size == 0:
        return float("nan")
    order = np.argsort(h)
    hs, ws = h[order], w[order]
    cw = np.cumsum(ws)
    if cw[-1] <= 0:
        return float(np.median(h))
    cutoff = 0.5 * cw[-1]
    return float(hs[int(np.searchsorted(cw, cutoff))])


def _gmm_standing_center(h, w=None, n_components=2, seed=0):
    """2-bilesenli GMM -> baskin (en agirlikli) bilesenin merkezi = duruslu boy.

    sklearn GaussianMixture sample-weight desteklemez; mesafe-agirligi varsa
    agirlikla-orantili yeniden-ornekleme ile yaklasilir (deterministik seed).
    """
    from sklearn.mixture import GaussianMixture

    if h.size < n_components:
        return _weighted_median(h, w if w is not None else np.ones_like(h))
    rng = np.random.default_rng(seed)
    if w is not None and np.sum(w) > 0 and np.any(w != w[0]):
        p = w / np.sum(w)
        n_draw = max(h.size, 1000)
        idx = rng.choice(h.size, size=n_draw, p=p)
        Xfit = h[idx].reshape(-1, 1)
    else:
        Xfit = h.reshape(-1, 1)
    gm = GaussianMixture(n_components=n_components, n_init=3, random_state=seed)
    gm.fit(Xfit)
    means = gm.means_.ravel()
    weights = gm.weights_.ravel()
    # 'standing' = en kalabalik kume (oyuncu cogunlukla dik durur/kosar).
    return float(means[int(np.argmax(weights))])


# --------------------------------------------------------------------------- #
#  Ana API
# --------------------------------------------------------------------------- #
def height_mode_scale(heights, weights=None, *, foot_speed=None, aspect=None,
                      assumed_height_m=ASSUMED_STANDING_M, method="kde",
                      speed_max=None, aspect_min=1.6, min_used=5,
                      bw_method=None):
    """Populasyon boy-MODUNDAN metrik olcek (carpik dagilimda dogru olan yol).

    heights : (N,) per-oyuncu geri-projekte boy olcumleri (rel_m ya da m). Bazilari
              dik (~1.75), bazilari cömelmis (~1.3) -> SOLA carpik dagilim.
    weights : (N,) opsiyonel mesafe-agirligi (kameraya yakin = daha guvenilir =
              daha agir). None ise uniform.
    foot_speed : (N,) opsiyonel ayak-noktasi hizi; standing-gate icin. None degilse
              speed_max ustundeki (hareketli/stabil-olmayan) olcumler atilir.
    aspect  : (N,) opsiyonel bbox en-boy orani (box_h/box_w); upright (dik) duruslu
              kapi icin. None degilse aspect_min altindaki (cömelmis) olcumler atilir.
    method  : 'kde' (varsayilan, KDE tepesi) | 'gmm' (2-bilesenli GMM baskin-merkez).

    Doner: {mode_height, scale, n_used, method, mean_height, n_input, gated}
      scale = assumed_height_m / mode_height  (rel_m -> m donusumu / olcek capasi).
    NOT: mean'i de doner ki carpitma (mode != mean) saydam gorulebilsin.
    """
    h, fin = _clean(heights)
    n_input = int(h.size)

    if weights is None:
        w = np.ones_like(h)
    else:
        w, _ = _clean(weights)
        if w.size != h.size:
            raise ValueError("weights uzunlugu heights ile esit olmali")
        w = np.where(np.isfinite(w), w, 0.0)

    # gecerli olcum maskesi: sonlu boy + pozitif boy + sonlu agirlik
    mask = fin & (h > 0)

    # ---- standing-gate (opsiyonel) ----
    gated = False
    if aspect is not None:
        asp, _ = _clean(aspect)
        if asp.size != h.size:
            raise ValueError("aspect uzunlugu heights ile esit olmali")
        mask = mask & (asp >= float(aspect_min))
        gated = True
    if foot_speed is not None:
        spd, _ = _clean(foot_speed)
        if spd.size != h.size:
            raise ValueError("foot_speed uzunlugu heights ile esit olmali")
        if speed_max is None:
            # adaptif: yer-yer hareket var; en stabil (yavas) yarisini tut.
            valid_spd = spd[np.isfinite(spd) & mask]
            thr = float(np.median(valid_spd)) if valid_spd.size else np.inf
        else:
            thr = float(speed_max)
        mask = mask & np.isfinite(spd) & (spd <= thr)
        gated = True

    hu, wu = h[mask], w[mask]
    n_used = int(hu.size)

    out = dict(mode_height=float("nan"), scale=float("nan"), n_used=n_used,
               method=method, mean_height=float("nan"), n_input=n_input,
               gated=gated)
    if n_used < int(min_used):
        return out

    if method == "gmm":
        mode_h = _gmm_standing_center(hu, wu)
    elif method == "kde":
        mode_h = _kde_mode(hu, wu, bw_method=bw_method)
    else:
        raise ValueError(f"bilinmeyen method: {method!r} ('kde' | 'gmm')")

    out["mode_height"] = float(mode_h)
    out["mean_height"] = _weighted_mean(hu, wu)
    out["scale"] = float(assumed_height_m / mode_h) if mode_h > 0 else float("nan")
    return out


def impossible_catalog_flag(scale_from_height, scale_from_goal,
                            implied_L, implied_W, *, ratio_tol=0.10,
                            L_bounds=FUTSAL_L_BOUNDS, W_bounds=FUTSAL_W_BOUNDS):
    """Imkansiz-katalog dedektoru: iki bagimsiz olcek tutarli mi + boyut makul mu.

    scale_from_height : boy-modundan cikan olcek (insanlar ~1.75 m capa).
    scale_from_goal   : kale-genisligi / katalog-snap'inden cikan olcek.
    implied_L, implied_W : o olcekten implied saha boyutlari (m).

    Bayrak (flagged=True) kosullari:
      - |scale_from_height/scale_from_goal - 1| > ratio_tol  (olcekler celisir;
        orn. katalog 2.4 m insan iddia ediyor -> boy-capasiyla %>10 sapar), VEYA
      - implied_L futsal araligi disinda (25-42 m), VEYA
      - implied_W futsal araligi disinda (16-25 m).

    Doner: {ratio, flagged, reason}. ratio = scale_from_height/scale_from_goal.
    """
    sh = float(scale_from_height)
    sg = float(scale_from_goal)
    L = float(implied_L)
    W = float(implied_W)

    if not np.isfinite(sg) or sg == 0.0:
        return dict(ratio=float("inf"), flagged=True,
                    reason="scale_from_goal sifir/gecersiz")

    ratio = sh / sg
    reasons = []
    if not np.isfinite(ratio):
        reasons.append("olcek orani gecersiz")
    elif abs(ratio - 1.0) > float(ratio_tol):
        reasons.append(
            f"olcek celiskisi: height/goal={ratio:.3f} "
            f"(|sapma|={abs(ratio - 1.0) * 100:.1f}% > {ratio_tol * 100:.0f}%)")
    if not (L_bounds[0] <= L <= L_bounds[1]):
        reasons.append(
            f"implied uzunluk {L:.1f} m futsal araligi {L_bounds} disinda")
    if not (W_bounds[0] <= W <= W_bounds[1]):
        reasons.append(
            f"implied genislik {W:.1f} m futsal araligi {W_bounds} disinda")

    flagged = len(reasons) > 0
    return dict(ratio=float(ratio), flagged=bool(flagged),
                reason="; ".join(reasons) if flagged else "tutarli")


if __name__ == "__main__":
    # Hizli kendi-kendine demo: carpik karisim -> mode ~1.75, mean dusuk.
    rng = np.random.default_rng(0)
    standing = rng.normal(1.75, 0.05, 750)
    crouch = rng.normal(1.30, 0.08, 250)
    H = np.concatenate([standing, crouch])
    r = height_mode_scale(H)
    print("mode=%.3f  mean=%.3f  scale=%.3f  n=%d"
          % (r["mode_height"], r["mean_height"], r["scale"], r["n_used"]))
    print(impossible_catalog_flag(1.0, 1.15, 46.0, 24.0))
    print(impossible_catalog_flag(1.0, 1.02, 38.0, 20.0))
