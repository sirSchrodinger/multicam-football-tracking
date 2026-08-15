#!/usr/bin/env python3
"""active_select sentetik testleri (bilinen ground-truth ile anlamli iddialar).

Iddialar (modulun cevabi HARDCODE EDILMEZ; sentetik geometriden turetilir):
  * k secince TAM k essiz kare doner;
  * belirsizlik+near-duplicate sahnesinde: yuksek-belirsizlik oncelenir VE
    near-identical kume de-dup edilir (kumeden en fazla ~1 secilir);
  * saf-cesitlilik (tekduze belirsizlik) kaplamasi ardisik VE rastgele secimi
    yener (secilenlerin min ciftli-mesafesi daha buyuk);
  * tohum = en belirsiz kare; zamansal-bosluk bonusu secimi kaydirir;
  * entropi modunda 0.5-conf en belirsiz.
"""
import numpy as np
from scipy.spatial.distance import pdist

from detect.active_select import (
    select_frames, frame_uncertainty, coreset_kcenter)


def _min_pair_dist(E, idx):
    """Secili kumenin min ciftli-mesafesi (kaplama yaricapi olcusu)."""
    sub = np.asarray(E)[list(idx)]
    return float(pdist(sub).min())


# ---------------------------------------------------------------------------
def test_returns_exactly_k_distinct():
    rng = np.random.default_rng(0)
    n, k = 30, 8
    fids = list(range(n))
    conf = [[float(rng.uniform(0.3, 0.9))] for _ in range(n)]
    emb = rng.normal(size=(n, 6))
    sel = select_frames(fids, conf, emb, k)
    assert len(sel) == k
    assert len(set(sel)) == k                 # essiz
    assert all(s in fids for s in sel)        # gecerli kimlikler


def test_k_geq_n_returns_all_distinct():
    n = 10
    fids = list(range(n))
    conf = [[0.5] for _ in range(n)]
    emb = np.random.default_rng(1).normal(size=(n, 4))
    sel = select_frames(fids, conf, emb, k=25)   # k > n => n'e kirpilir
    assert len(sel) == n
    assert set(sel) == set(fids)


# ---------------------------------------------------------------------------
def test_prefers_uncertainty_and_dedups_cluster():
    """B bolgesi: 12 yuksek-belirsizlik + 12 dusuk-belirsizlik (UZAYDA karisik).
    Ayrica (30,30)'da 8 near-identical YUKSEK-belirsizlik kume.
    Beklenti: secimler yuksek-belirsizlikten gelir (dusuk=0), kumeden <=1.
    """
    rng = np.random.default_rng(7)
    # Region B: ayni uzayda yuksek ve dusuk belirsizlik kareleri ic-ice
    B_high = rng.uniform(0, 10, size=(12, 2))   # conf ~0.5  -> yuksek u
    B_low = rng.uniform(0, 10, size=(12, 2))    # conf ~0.97 -> dusuk u
    cluster = np.array([[30.0, 30.0]]) + rng.normal(scale=1e-4, size=(8, 2))

    emb = np.vstack([B_high, B_low, cluster])
    n = emb.shape[0]
    fids = list(range(n))
    idx_high = list(range(0, 12))
    idx_low = list(range(12, 24))
    idx_cluster = list(range(24, 32))

    conf = ([[0.5]] * 12) + ([[0.97]] * 12) + ([[0.5]] * 8)  # kume yuksek-belirsiz
    k = 7
    sel = select_frames(fids, conf, emb, k)

    assert len(sel) == k and len(set(sel)) == k
    n_low = sum(s in idx_low for s in sel)
    n_cluster = sum(s in idx_cluster for s in sel)
    n_high = sum(s in idx_high for s in sel)
    # belirsizligi onceler: dusuk-belirsizlik karesi SECILMEZ
    assert n_low == 0, f"dusuk-belirsizlik secildi: {sel}"
    # near-identical kumeyi de-dup eder: en fazla 1 temsilci
    assert n_cluster <= 1, f"kumeden {n_cluster} secildi (de-dup basarisiz)"
    # geri kalan hepsi yuksek-belirsizlik B karesi
    assert n_high == k - n_cluster


def test_uncertainty_dominates_when_all_diverse():
    """Tum embedding'ler birbirinden cok uzak (cesitlilik esit) => tohum en
    belirsiz kare olmali; en belirsiz mutlaka secimde ve ILK sirada."""
    n = 12
    emb = np.eye(n) * 100.0          # hepsi essiz/uzak (ortonormal eksenler)
    conf = [[0.9]] * n
    conf[5] = [0.5]                  # 5 numarali kare en belirsiz
    sel = select_frames(list(range(n)), conf, emb, k=4)
    assert sel[0] == 5               # tohum = en belirsiz
    assert 5 in sel


# ---------------------------------------------------------------------------
def test_pure_diversity_beats_contiguous_and_random():
    """Tekduze belirsizlik => saf farthest-first. Bir cizgi uzerindeki 40 kare
    icin secimin min-ciftli-mesafesi, ardisik blok VE rastgele secimden buyuk."""
    n, k = 40, 5
    emb = np.arange(n, dtype=np.float64).reshape(n, 1)   # 0..39 cizgi
    conf = [[0.6]] * n                                   # tekduze
    sel = select_frames(list(range(n)), conf, emb, k)

    ours = _min_pair_dist(emb, sel)
    contiguous = _min_pair_dist(emb, list(range(k)))     # 0..4 => min=1
    assert ours > contiguous

    rng = np.random.default_rng(0)
    rand_scores = [_min_pair_dist(emb, rng.choice(n, size=k, replace=False))
                   for _ in range(200)]
    assert ours > float(np.mean(rand_scores))
    # ek: degerlerin > %90'ini de gecsin (sadece ortalama degil)
    assert ours >= np.quantile(rand_scores, 0.90)


def test_coreset_pure_kcenter_coverage():
    """coreset_kcenter (agirliksiz) 2B izgarada iyi yayilim verir: secilen
    kumenin min-ciftli-mesafesi rastgele secimin ortalamasini gecer."""
    xs, ys = np.meshgrid(np.arange(8), np.arange(8))
    emb = np.column_stack([xs.ravel(), ys.ravel()]).astype(np.float64)
    k = 6
    sel = coreset_kcenter(emb, k)
    assert len(sel) == k and len(set(sel)) == k
    ours = _min_pair_dist(emb, sel)
    rng = np.random.default_rng(3)
    rand = np.mean([_min_pair_dist(emb, rng.choice(len(emb), k, replace=False))
                    for _ in range(200)])
    assert ours > rand


# ---------------------------------------------------------------------------
def test_temporal_gap_bonus_shifts_seed():
    """Esit-belirsizlik + uzak embedding'ler; bir karede gap-bayragi => skoru
    artar => tohum o kare olur."""
    n = 8
    emb = np.eye(n) * 50.0
    conf = [[0.7]] * n               # hepsi esit belirsizlik
    gaps = [0] * n
    gaps[3] = 1                      # 3 numarada zamansal bosluk
    sel = select_frames(list(range(n)), conf, emb, k=3,
                        temporal_gaps=gaps, gap_weight=0.5)
    assert sel[0] == 3               # tohum gap karesine kaydi
    # gap olmadan tohum 0 olurdu:
    sel0 = select_frames(list(range(n)), conf, emb, k=3)
    assert sel0[0] == 0


# ---------------------------------------------------------------------------
def test_frame_uncertainty_modes_and_empty():
    # least_confidence: dusuk conf => yuksek u; ortalama dogru
    u = frame_uncertainty([[0.9, 0.7], [0.2], []], mode="least_confidence",
                          empty_value=0.0)
    assert np.isclose(u[0], 1 - 0.8)     # (0.1+0.3)/2 = 0.2
    assert np.isclose(u[1], 0.8)
    assert u[2] == 0.0                   # tespitsiz
    assert u[1] > u[0]                   # dusuk-conf daha belirsiz

    # entropy: 0.5 conf maksimum belirsizlik (=1 bit), 0.99 cok dusuk
    e = frame_uncertainty([[0.5], [0.99], [0.01]], mode="entropy")
    assert np.isclose(e[0], 1.0, atol=1e-6)
    assert e[0] > e[1] and e[0] > e[2]
    assert np.isclose(e[1], e[2], atol=1e-6)   # simetrik


def test_ragged_and_2d_confidence_inputs_agree():
    """Ragged list ve NaN-dolgulu 2B dizi ayni belirsizligi vermeli."""
    ragged = [[0.9, 0.5], [0.3]]
    padded = np.array([[0.9, 0.5], [0.3, np.nan]])
    ur = frame_uncertainty(ragged, mode="least_confidence")
    up = frame_uncertainty(padded, mode="least_confidence")
    assert np.allclose(ur, up)


def test_pool_factor_two_stage_matches_dedup():
    """Iki-asamali (havuz on-filtresi) yol da kumeyi de-dup eder ve dusuk-
    belirsizligi havuz disinda birakir."""
    rng = np.random.default_rng(11)
    B_high = rng.uniform(0, 10, size=(10, 2))
    B_low = rng.uniform(0, 10, size=(20, 2))
    cluster = np.array([[40.0, 0.0]]) + rng.normal(scale=1e-4, size=(8, 2))
    emb = np.vstack([B_high, B_low, cluster])
    n = emb.shape[0]
    conf = ([[0.5]] * 10) + ([[0.98]] * 20) + ([[0.5]] * 8)
    idx_low = set(range(10, 30))
    idx_cluster = set(range(30, 38))
    sel = select_frames(list(range(n)), conf, emb, k=6, pool_factor=3.0)
    assert len(sel) == 6 and len(set(sel)) == 6
    assert sum(s in idx_cluster for s in sel) <= 1
    assert sum(s in idx_low for s in sel) == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
