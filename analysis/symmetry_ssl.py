#!/usr/bin/env python3
"""symmetry_ssl — near/far field-symmetry self-supervision (LABEL-FREE recall audit).

Fikir (Alperen): sabit kamerada bir futsal maçı, devre arasında saha
değişimi (side-swap) ile oynanır. AYNI 14 oyuncu, maçın tamamına
zaman-integre edildiğinde, orta çizginin iki yanında (kameraya YAKIN yarı vs
UZAK yarı) istatistiksel olarak SİMETRİK zaman geçirir. Dolayısıyla zaman-
integre, yarı-koşullu dağılımlar (oyuncu sayısı / occupancy / hız) orta çizgi
boyunca AYNALANDIKTAN sonra birbirini tutmalıdır.

Mükemmel tespit altında bu uyumsuzluk (discrepancy) ~0'dır. UZAK yarıda bir
recall açığı (kamera-uzak üçte-birde küçük/düşük-kontrast oyuncuların
kaçırılması) ise POZİTİF bir uyumsuzluk olarak, ve UZAK ÜÇTE-BİRDE yoğunlaşmış
biçimde görünür => ETİKETSİZ bir uzak-recall açığı tahmini.

Simetri önsavı (prior) literatürdeki yansıma-eşdeğerliğine (reflection-
equivariance) dayanır: Wang et al., "TacticAI: an AI assistant for football
tactics", Nature Communications 15, 1906 (2024) — futbol sahnelerinin
yatay/orta-çizgi yansımaları altında değişmezliği. Buradaki yeni kısım: bu
önsavı sabit-kamera spor çekiminde zaman-integre bir DENETİM (audit) ölçütüne
çevirmek; eşdeğerli bir model eğitmek değil, var olan tespit borusunun
recall açığını etiket olmadan ÖLÇMEK.

Üç tamamlayıcı dedektör (hepsi orta çizgi boyunca aynalama/folding sonrası):
  1) occupancy_jsd  — yarı-occupancy 2B histogramlarının Jensen-Shannon
       diverjansı (bit). YEREL (şekil) asimetriyi yakalar: uzak-üçte-bir gibi
       lokalize bir recall açığı normalize occupancy ŞEKLİNİ bozar.
  2) count_emd      — kare-başına (near vs far) oyuncu-sayısı dağılımları
       arası 1B Wasserstein (EMD). BÜYÜKLÜK kayması: uzak yarı sistematik
       daha az oyuncu sayarsa EMD pozitifleşir.
  3) far_deficit_estimate — folding sonrası toplam kütle oranından türetilen
       skaler eksik-tespit kesri  max(0, 1 - N_far/N_near).  TEKDÜZE (uniform)
       bir uzak açığı normalize şekli bozmaz (occupancy_jsd ~0) ama kütle
       oranını düşürür; bu dedektör onu yakalar.
  per_zone — orta çizgiden uzaklığa (u) göre bantlar; her bantta near vs far
       kütle oranı + JSD. Recall açığının UZAK ÜÇTE-BİRDE yerelleştiğini
       gösterir (son bant = uzak üçte-bir).

Koordinat sözleşmesi (repo geneli): saha metrik X = uzunluk (length),
Y = genişlik (width). midline X üzerinde bir skalerdir (tipik L/2). UZAK yarı
kameradan UZAK yarıdır; `far_side` ile hangi X-tarafının uzak olduğu seçilir.
Aynalama orta çizgiye dik yansımadır: (x, y) -> (2*midline - x, y); y korunur.
Folded ortak koordinat u = orta çizgiye olan uzaklık (0 = orta çizgi,
half_length = kale çizgisi).

Lisans/bağımlılık: yalnızca numpy + scipy (BSD). GPU yok, harici model yok.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:  # scipy her zaman var (repo bağımlılığı); yine de net hata verelim
    from scipy.stats import wasserstein_distance
except Exception as _e:  # pragma: no cover
    raise ImportError("symmetry_ssl scipy.stats.wasserstein_distance gerektirir") from _e


# --------------------------------------------------------------------------- #
#  düşük-seviye yardımcılar                                                    #
# --------------------------------------------------------------------------- #
def _js_divergence_bits(p: np.ndarray, q: np.ndarray) -> float:
    """İki (normalize edilmemiş) histogram arası Jensen-Shannon diverjansı (bit).

    JSD = 0.5*KL(p||m) + 0.5*KL(q||m),  m = 0.5*(p+q).  Simetrik, sınırlı
    [0, 1] bit (base-2). p,q olasılığa normalize edilir; sıfır binler maskelenir
    (m, p>0 olan her yerde >0 olduğundan KL sonlu kalır).
    """
    p = np.asarray(p, dtype=np.float64).ravel()
    q = np.asarray(q, dtype=np.float64).ravel()
    sp, sq = p.sum(), q.sum()
    if sp <= 0 or sq <= 0:
        return float("nan")
    p = p / sp
    q = q / sq
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0  # b = m >= 0.5*a > 0 burada -> sıfıra bölme yok
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    jsd_nats = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    jsd_nats = max(jsd_nats, 0.0)  # nümerik negatif sızıntıyı kırp
    return jsd_nats / np.log(2.0)  # nats -> bits


def _robust_extent(vals: np.ndarray, lo_q: float = 0.005, hi_q: float = 0.995):
    """Aykırı-değere dayanıklı (alt, üst) sınır."""
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return (0.0, 1.0)
    return float(np.quantile(vals, lo_q)), float(np.quantile(vals, hi_q))


# --------------------------------------------------------------------------- #
#  ana API                                                                     #
# --------------------------------------------------------------------------- #
def symmetry_discrepancy(
    df_positions,
    midline: float,
    *,
    far_side: str = "high",
    x_col: str = "pitch_x",
    y_col: str = "pitch_y",
    frame_col: str = "frame",
    tid_col: str = "tid",
    t_col: str = "t_sec",
    in_pitch_col: Optional[str] = "in_pitch",
    use_in_pitch: bool = True,
    pitch_length: Optional[float] = None,
    pitch_width: Optional[float] = None,
    half_length: Optional[float] = None,
    y_range: Optional[tuple] = None,
    n_xbins: int = 12,
    n_ybins: int = 8,
    n_zones: int = 3,
    fps: float = 25.0,
    max_speed: Optional[float] = None,
) -> dict:
    """Near/far simetri uyumsuzluğunu hesapla (etiketsiz uzak-recall denetimi).

    Parametreler
    ----------
    df_positions : pandas.DataFrame
        En az `x_col`, `y_col`, `frame_col` sütunları. Hız (speed) için ek
        olarak `tid_col` ve (`t_col` veya `frame_col`+`fps`) gerekir.
    midline : float
        Orta çizginin X (uzunluk) değeri; yarıları ayırır.
    far_side : {'high','low'}
        'high' => X >= midline olan yarı kameradan UZAK yarıdır. 'low' tersi.
    pitch_length, pitch_width : float, opsiyonel
        Verilirse half_length = min(midline, L-midline) ve y aralığı [0, W]
        olarak alınır; yoksa veriden dayanıklı kuantil ile türetilir.
    n_xbins, n_ybins : int
        Occupancy 2B histogram bölme sayıları (u ekseni, y ekseni).
    n_zones : int
        Orta çizgiden uzaklığa göre bant sayısı (son bant = uzak üçte-bir).

    Döner
    -----
    dict (anahtarlar): overall, per_zone, far_deficit_estimate,
        occupancy_jsd, count_emd, speed_emd, far_deficit_count,
        n_near, n_far, half_length, y_range, bins, n_zones, far_side.
        `overall` = occupancy_jsd (birincil uzamsal simetri diverjansı, bit).
    """
    if far_side not in ("high", "low"):
        raise ValueError("far_side 'high' veya 'low' olmalı")
    if n_zones < 1:
        raise ValueError("n_zones >= 1 olmalı")

    # ---- sütunları diziye çek ---------------------------------------------
    x = np.asarray(df_positions[x_col], dtype=np.float64)
    y = np.asarray(df_positions[y_col], dtype=np.float64)
    fr = np.asarray(df_positions[frame_col])

    finite = np.isfinite(x) & np.isfinite(y)
    if use_in_pitch and in_pitch_col is not None and in_pitch_col in df_positions:
        finite = finite & np.asarray(df_positions[in_pitch_col], dtype=bool)

    # ---- yarı üyeliği + folded koordinat u --------------------------------
    if far_side == "high":
        is_far = x >= midline
    else:
        is_far = x < midline
    is_near = ~is_far
    u = np.abs(x - midline)  # orta çizgiye uzaklık (her iki yarı için ortak)

    # ---- half_length (folding ortak menzili) ------------------------------
    if half_length is None:
        if pitch_length is not None:
            half_length = float(min(midline, pitch_length - midline))
        else:
            near_lo = midline - _robust_extent(x[is_near & finite])[0]
            far_hi = _robust_extent(x[is_far & finite])[1] - midline
            half_length = float(min(near_lo, far_hi))
    if not np.isfinite(half_length) or half_length <= 0:
        raise ValueError(f"geçersiz half_length={half_length} (midline yanlış olabilir)")

    valid = finite & (u <= half_length + 1e-9)

    # ---- y aralığı --------------------------------------------------------
    if y_range is not None:
        y_lo, y_hi = float(y_range[0]), float(y_range[1])
    elif pitch_width is not None:
        y_lo, y_hi = 0.0, float(pitch_width)
    else:
        y_lo, y_hi = _robust_extent(y[valid])
    if y_hi <= y_lo:
        y_hi = y_lo + 1.0

    # ---- occupancy 2B histogramlar (folded: ortak u x y) ------------------
    u_edges = np.linspace(0.0, half_length, n_xbins + 1)
    y_edges = np.linspace(y_lo, y_hi, n_ybins + 1)

    sel_near = valid & is_near
    sel_far = valid & is_far
    H_near, _, _ = np.histogram2d(u[sel_near], y[sel_near], bins=[u_edges, y_edges])
    H_far, _, _ = np.histogram2d(u[sel_far], y[sel_far], bins=[u_edges, y_edges])

    N_near = float(H_near.sum())
    N_far = float(H_far.sum())

    occupancy_jsd = _js_divergence_bits(H_near, H_far)

    # ---- kare-başına sayı dağılımı (EMD) ----------------------------------
    uf, inv = np.unique(fr, return_inverse=True)
    near_counts = np.bincount(inv[sel_near], minlength=uf.size).astype(np.float64)
    far_counts = np.bincount(inv[sel_far], minlength=uf.size).astype(np.float64)
    if near_counts.size and far_counts.size:
        count_emd = float(wasserstein_distance(near_counts, far_counts))
    else:
        count_emd = float("nan")

    # ---- uzak-recall açığı (kütle) tahmini --------------------------------
    if N_near > 0:
        far_deficit_estimate = float(np.clip(1.0 - N_far / N_near, 0.0, 1.0))
    else:
        far_deficit_estimate = float("nan")
    far_deficit_count = float(max(N_near - N_far, 0.0))

    # ---- per_zone: orta çizgiden uzaklığa göre bantlar --------------------
    bin_centers = 0.5 * (u_edges[:-1] + u_edges[1:])
    band_w = half_length / n_zones
    band_of_bin = np.clip((bin_centers / band_w).astype(int), 0, n_zones - 1)

    per_zone = []
    for b in range(n_zones):
        cols = np.where(band_of_bin == b)[0]
        near_mass = float(H_near[cols, :].sum())
        far_mass = float(H_far[cols, :].sum())
        ratio = far_mass / near_mass if near_mass > 0 else float("nan")
        deficit = float(np.clip(1.0 - ratio, 0.0, 1.0)) if np.isfinite(ratio) else float("nan")
        band_jsd = _js_divergence_bits(H_near[cols, :], H_far[cols, :])
        per_zone.append(
            dict(
                zone_index=b,
                u_lo=float(b * band_w),
                u_hi=float((b + 1) * band_w),
                near_mass=near_mass,
                far_mass=far_mass,
                mass_ratio=ratio,
                deficit_frac=deficit,
                jsd=band_jsd,
                is_far_third=(b == n_zones - 1),
            )
        )

    # ---- hız (speed) dağılımı simetrisi (opsiyonel kod yolu) --------------
    speed_emd = _speed_emd(
        df_positions, is_far, valid, x, y, fr,
        tid_col=tid_col, t_col=t_col, frame_col=frame_col,
        fps=fps, max_speed=max_speed,
    )

    return dict(
        overall=occupancy_jsd,
        occupancy_jsd=occupancy_jsd,
        count_emd=count_emd,
        speed_emd=speed_emd,
        far_deficit_estimate=far_deficit_estimate,
        far_deficit_count=far_deficit_count,
        per_zone=per_zone,
        n_near=N_near,
        n_far=N_far,
        half_length=float(half_length),
        y_range=(y_lo, y_hi),
        bins=(int(n_xbins), int(n_ybins)),
        n_zones=int(n_zones),
        far_side=far_side,
    )


def _speed_emd(df_positions, is_far, valid, x, y, fr, *,
               tid_col, t_col, frame_col, fps, max_speed):
    """Near vs far hız dağılımları arası Wasserstein (varsa). Yoksa None.

    Hız, AYNI tid'in ardışık karelerinden türetilir (pitch-birim / sn). Her
    hız, başlangıç noktasının yarısına atanır. Veri tid içermiyorsa None döner.
    """
    if tid_col not in df_positions:
        return None
    tid = np.asarray(df_positions[tid_col])
    if t_col in df_positions:
        t = np.asarray(df_positions[t_col], dtype=np.float64)
    else:
        t = np.asarray(fr, dtype=np.float64) / float(fps)

    order = np.lexsort((t, tid))  # tid içinde zamana göre sırala
    tid_s = tid[order]
    t_s = t[order]
    x_s = x[order]
    y_s = y[order]
    far_s = is_far[order]
    val_s = valid[order]

    same = tid_s[1:] == tid_s[:-1]
    dt = t_s[1:] - t_s[:-1]
    dx = x_s[1:] - x_s[:-1]
    dy = y_s[1:] - y_s[:-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        spd = np.sqrt(dx * dx + dy * dy) / dt
    pair_ok = same & (dt > 0) & val_s[1:] & val_s[:-1] & np.isfinite(spd)
    if max_speed is not None:
        pair_ok = pair_ok & (spd <= max_speed)

    start_far = far_s[:-1]
    near_sp = spd[pair_ok & ~start_far]
    far_sp = spd[pair_ok & start_far]
    if near_sp.size == 0 or far_sp.size == 0:
        return None
    return float(wasserstein_distance(near_sp, far_sp))


# --------------------------------------------------------------------------- #
#  CLI self-test (kaba)                                                        #
# --------------------------------------------------------------------------- #
def _selftest() -> int:  # pragma: no cover
    import pandas as pd

    rng = np.random.default_rng(0)
    L, W, mid = 40.0, 20.0, 20.0
    T, P = 1500, 14
    rows = []
    for f in range(T):
        for p in range(P):
            # orta çizgiden uzaklık u: iki-kümeli (yapısal) dağılım
            u = rng.choice([4.0, 15.0])[()] + rng.normal(0, 1.5)
            u = float(np.clip(u, 0.2, 19.5))
            yy = float(np.clip(rng.normal(10, 4), 0.5, 19.5))
            side = rng.random() < 0.5
            xx = mid + u if side else mid - u
            rows.append((p, f, f / 25.0, xx, yy, True))
    df = pd.DataFrame(rows, columns=["tid", "frame", "t_sec", "pitch_x", "pitch_y", "in_pitch"])
    r = symmetry_discrepancy(df, mid, pitch_length=L, pitch_width=W)
    print("symmetric  overall=%.4f far_deficit=%.4f count_emd=%.3f"
          % (r["overall"], r["far_deficit_estimate"], r["count_emd"]))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_selftest())
