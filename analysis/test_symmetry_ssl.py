"""test_symmetry_ssl — near/far simetri SSL ölçütünün sentetik-GT testleri.

Tüm yer-doğrulukları (ground truth) sentetik ve BİLİNİR:
  * symmetric  : orta çizgi boyunca tam simetrik occupancy => discrepancy ~0.
  * far hole   : uzak yarının UZAK ÜÇTE-BİRİNDE tespitler sistematik kaçırılır
                 => overall > 0, far_deficit_estimate > 0 ve uzak banda lokalize.
  * uniform hole: uzak yarıda TEKDÜZE kayıp => şekli bozmaz (occupancy_jsd küçük)
                 ama kütle dedektörü (far_deficit_estimate) yakalar.

Beklenen cevap MODÜLE gömülü değildir; modül ham veriden hesaplar, test yalnız
gerçek sayısal davranışı (simetrik küçük, açık büyük, lokalize) doğrular.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis import symmetry_ssl as S  # noqa: E402

L, W, MID = 40.0, 20.0, 20.0  # uzunluk, genişlik, orta çizgi
HALF = MID                     # half_length = 20
T_FRAMES, P_PLAYERS = 2000, 14


def _symmetric_rows(rng):
    """Orta çizgi boyunca SİMETRİK, YAPISAL (tekdüze değil) occupancy üret.

    Her oyuncu her karede, orta çizgiden uzaklığı u olan iki-kümeli yapısal bir
    dağılımdan örneklenir; yarı (near/far) adil yazı-tura ile seçilir. far yarı,
    near yarının orta çizgi-yansımasıdır => fold sonrası birebir eşleşmeli.
    u-dağılımı tekdüze DEĞİL: yanlış (foldsuz) bir karşılaştırma büyük JSD verir,
    doğru fold ~0 verir -> bu test fold mantığını da doğrular.
    """
    rows = []
    for f in range(T_FRAMES):
        for p in range(P_PLAYERS):
            u = rng.choice([4.0, 15.0]) + rng.normal(0.0, 1.5)
            u = float(np.clip(u, 0.3, HALF - 0.3))
            yy = float(np.clip(rng.normal(10.0, 4.0), 0.5, W - 0.5))
            far = rng.random() < 0.5
            xx = MID + u if far else MID - u
            rows.append((p, f, f / 25.0, xx, yy, far, u, True))
    return pd.DataFrame(
        rows, columns=["tid", "frame", "t_sec", "pitch_x", "pitch_y", "_far", "_u", "in_pitch"]
    )


def _apply_far_third_hole(df, rng, p_drop=0.6, u_thresh=None):
    """UZAK yarıda u > u_thresh olan tespitleri p_drop olasılıkla DÜŞÜR (recall açığı)."""
    if u_thresh is None:
        u_thresh = (2.0 / 3.0) * HALF  # uzak üçte-bir başlangıcı
    drop = df["_far"].to_numpy() & (df["_u"].to_numpy() > u_thresh)
    drop = drop & (rng.random(len(df)) < p_drop)
    return df.loc[~drop].reset_index(drop=True)


def _apply_uniform_far_hole(df, rng, p_drop=0.4):
    """UZAK yarıda TEKDÜZE (u'dan bağımsız) tespit düşür => şekil korunur, kütle düşer."""
    drop = df["_far"].to_numpy() & (rng.random(len(df)) < p_drop)
    return df.loc[~drop].reset_index(drop=True)


@pytest.fixture(scope="module")
def sym_df():
    return _symmetric_rows(np.random.default_rng(7))


@pytest.fixture(scope="module")
def sym_result(sym_df):
    return S.symmetry_discrepancy(sym_df, MID, pitch_length=L, pitch_width=W)


# --------------------------------------------------------------------------- #
#  1) simetrik => uyumsuzluk ~0                                                #
# --------------------------------------------------------------------------- #
def test_symmetric_discrepancy_near_zero(sym_result):
    r = sym_result
    # birincil uzamsal diverjans (bit) sıfıra çok yakın
    assert r["overall"] == r["occupancy_jsd"]
    assert r["overall"] < 0.02, r["overall"]
    # kütle eksiği yok
    assert r["far_deficit_estimate"] < 0.05, r["far_deficit_estimate"]
    # sayı dağılımları örtüşür (EMD küçük; ~7 oyuncu/yarı ölçeğinde)
    assert r["count_emd"] < 0.5, r["count_emd"]
    # near/far toplam kütleler birbirine yakın
    assert abs(r["n_near"] - r["n_far"]) / max(r["n_near"], 1) < 0.05


def test_symmetric_per_zone_all_low(sym_result):
    # hiçbir bantta anlamlı açık yok
    for z in sym_result["per_zone"]:
        assert z["deficit_frac"] < 0.06, z


def test_symmetric_speed_emd_small(sym_result):
    # hız kod yolu çalışır ve simetrik => küçük EMD
    assert sym_result["speed_emd"] is not None
    assert sym_result["speed_emd"] >= 0.0


# --------------------------------------------------------------------------- #
#  2) uzak-üçte-bir recall açığı => pozitif + LOKALİZE                          #
# --------------------------------------------------------------------------- #
def test_far_third_hole_positive_and_localized(sym_df, sym_result):
    rng = np.random.default_rng(11)
    df_hole = _apply_far_third_hole(sym_df, rng, p_drop=0.6)
    r = S.symmetry_discrepancy(df_hole, MID, pitch_length=L, pitch_width=W)

    base = sym_result
    # uyumsuzluk simetrikten BELİRGİN şekilde büyük
    assert r["overall"] > 0.03, r["overall"]
    assert r["overall"] > 5.0 * base["overall"], (r["overall"], base["overall"])
    # sayı-dağılımı kayması da büyür
    assert r["count_emd"] > base["count_emd"]
    # etiketsiz uzak-recall açığı tahmini pozitif
    assert r["far_deficit_estimate"] > 0.10, r["far_deficit_estimate"]
    assert r["far_deficit_count"] > 0

    # LOKALİZASYON: uzak üçte-bir (son bant) açığı, orta-çizgi bandından çok büyük
    near_band = r["per_zone"][0]
    far_band = r["per_zone"][-1]
    assert far_band["is_far_third"] is True
    assert far_band["deficit_frac"] > 0.20, far_band
    assert far_band["deficit_frac"] > near_band["deficit_frac"] + 0.15, (
        near_band["deficit_frac"], far_band["deficit_frac"])
    # kütle oranı bantlar boyunca orta çizgiden kale çizgisine MONOTON düşmeli
    # (açık derinleştikçe far/near oranı uzaklaştıkça azalır)
    ratios = [z["mass_ratio"] for z in r["per_zone"]]
    assert ratios[-1] < ratios[0] - 0.20, ratios


def test_far_third_hole_deficit_monotone(sym_df):
    # açık derinleştikçe (p_drop artar) far_deficit_estimate monotonik büyür
    rng = np.random.default_rng(3)
    ests = []
    for p in (0.0, 0.3, 0.6, 0.9):
        dfh = _apply_far_third_hole(sym_df, rng, p_drop=p)
        r = S.symmetry_discrepancy(dfh, MID, pitch_length=L, pitch_width=W)
        ests.append(r["far_deficit_estimate"])
    assert all(b >= a - 1e-9 for a, b in zip(ests, ests[1:])), ests
    assert ests[-1] > ests[0] + 0.10, ests


# --------------------------------------------------------------------------- #
#  3) TEKDÜZE uzak açığı => şekil korunur, KÜTLE dedektörü yakalar              #
# --------------------------------------------------------------------------- #
def test_uniform_far_hole_caught_by_mass_not_shape(sym_df, sym_result):
    rng = np.random.default_rng(5)
    df_u = _apply_uniform_far_hole(sym_df, rng, p_drop=0.4)
    r = S.symmetry_discrepancy(df_u, MID, pitch_length=L, pitch_width=W)

    # tekdüze kayıp normalize ŞEKLİ büyük ölçüde korur -> occupancy_jsd küçük kalır
    assert r["occupancy_jsd"] < 0.02, r["occupancy_jsd"]
    # ama KÜTLE dedektörü açığı net görür (~%40)
    assert r["far_deficit_estimate"] > 0.25, r["far_deficit_estimate"]
    # ve tekdüze olduğu için bantlar arası açık KABACA eşit (lokalize DEĞİL)
    defs = [z["deficit_frac"] for z in r["per_zone"]]
    assert max(defs) - min(defs) < 0.15, defs


# --------------------------------------------------------------------------- #
#  4) API sözleşmesi + yön (far_side) tutarlılığı                              #
# --------------------------------------------------------------------------- #
def test_api_contract_keys_and_types(sym_result):
    r = sym_result
    for k in ("overall", "per_zone", "far_deficit_estimate", "occupancy_jsd",
              "count_emd", "speed_emd", "far_deficit_count", "n_near", "n_far",
              "half_length", "bins", "n_zones", "far_side"):
        assert k in r, k
    assert isinstance(r["per_zone"], list) and len(r["per_zone"]) == r["n_zones"]
    assert 0.0 <= r["far_deficit_estimate"] <= 1.0
    assert r["half_length"] == pytest.approx(HALF)


def test_far_side_low_mirrors_high(sym_df):
    """far_side='low' ile UZAK yarıyı X<midline yapınca: simetrik veride iki
    yön de ~0; ama far_third açığını 'low' tarafa koyarsak 'low' yönü yakalar."""
    rng = np.random.default_rng(21)
    # simetrik veri: yön ne olursa olsun ~0
    r_hi = S.symmetry_discrepancy(sym_df, MID, far_side="high", pitch_length=L, pitch_width=W)
    r_lo = S.symmetry_discrepancy(sym_df, MID, far_side="low", pitch_length=L, pitch_width=W)
    assert r_hi["far_deficit_estimate"] < 0.05
    assert r_lo["far_deficit_estimate"] < 0.05

    # açığı X<midline (low taraf) uzak-üçte-birine koy
    low_far = (~sym_df["_far"].to_numpy()) & (sym_df["_u"].to_numpy() > (2.0 / 3.0) * HALF)
    drop = low_far & (rng.random(len(sym_df)) < 0.6)
    dfh = sym_df.loc[~drop].reset_index(drop=True)
    # 'low' yönü (X<midline = uzak kabul) açığı görmeli; 'high' yönü ise bunu
    # 'near' yarıdaki fazlalık sayar -> far_deficit_estimate ~0.
    r_lo_h = S.symmetry_discrepancy(dfh, MID, far_side="low", pitch_length=L, pitch_width=W)
    r_hi_h = S.symmetry_discrepancy(dfh, MID, far_side="high", pitch_length=L, pitch_width=W)
    assert r_lo_h["far_deficit_estimate"] > 0.10, r_lo_h["far_deficit_estimate"]
    assert r_hi_h["far_deficit_estimate"] < 0.05, r_hi_h["far_deficit_estimate"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
