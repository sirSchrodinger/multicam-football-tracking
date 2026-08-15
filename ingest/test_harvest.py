"""ingest/test_harvest.py — harvest_catalog için anlamlı (sentetik GT) testler.

Kanıtlananlar:
  - build_catalog alanları tip-dönüşümüyle round-trip eder (in-memory + sqlite).
  - stratified_sample istenen sayıda FARKLI sahaya yayılır, hiçbir sahayı per_venue'den
    fazla seçmez, ve çeşitlilik-önce sıralar (ilk K girdi K farklı saha).
  - phash_dedup birebir/near-identical ardışık kareleri atar ama açıkça farklı kareyi tutar;
    Hamming uzaklığı aynı görüntüde 0, farklı görüntüde büyüktür.

Sentetik veriler bilinen ground-truth taşır; beklenen cevap modüle gömülü DEĞİL —
testler gerçek sayısal davranışı assert eder. Gerçek cams_28062026.tsv varsa ek olarak
(katalog kendi içinden türetilmiş dinamik GT ile) doğrulanır.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ingest.harvest import (
    build_catalog, stratified_sample, phash_dedup,
    Catalog, MatchRecord, dhash, ahash, hamming, read_tsv,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REAL_TSV = os.path.join(REPO, "cams_28062026.tsv")


# ----------------------------------------------------------------- fixtures --
def synthetic_rows():
    """Bilinen tabaka dağılımı: A=3 maç, B=2, C=1, D=1, E=1 → 8 maç / 5 farklı saha."""
    return [
        # id, n_cams, date, venue, title, views, first_url
        {"id": 1, "n_cams": 2, "date": "d1", "venue": "Alpha Hali Saha", "title": "1.Saha", "views": 10, "first_url": "u1"},
        {"id": 2, "n_cams": 1, "date": "d2", "venue": "Alpha Hali Saha", "title": "2.Saha", "views": 30, "first_url": "u2"},
        {"id": 3, "n_cams": 2, "date": "d3", "venue": "Alpha Hali Saha", "title": "3.Saha", "views": 5,  "first_url": "u3"},
        {"id": 4, "n_cams": 2, "date": "d4", "venue": "Beta Arena",      "title": "",       "views": 7,  "first_url": "u4"},
        {"id": 5, "n_cams": 1, "date": "d5", "venue": "Beta Arena",      "title": "B2",     "views": 9,  "first_url": "u5"},
        {"id": 6, "n_cams": 2, "date": "d6", "venue": "Gamma Park",      "title": "",       "views": 2,  "first_url": "u6"},
        {"id": 7, "n_cams": 1, "date": "d7", "venue": "Delta Spor",      "title": "",       "views": 0,  "first_url": "u7"},
        {"id": 8, "n_cams": 2, "date": "d8", "venue": "Epsilon Saha",    "title": "",       "views": 1,  "first_url": "u8"},
    ]


def _venue_keys(matches):
    return [m.venue_key for m in matches]


# =========================================================== CATALOG PARSE ===
def test_build_catalog_roundtrips_fields_and_types():
    rows = synthetic_rows()
    cat = build_catalog(rows)
    assert len(cat) == 8

    # ilk kayıt: alanlar ve tipler korunuyor
    m0 = cat[0]
    assert isinstance(m0, MatchRecord)
    assert m0.id == 1 and isinstance(m0.id, int)
    assert m0.n_cams == 2 and isinstance(m0.n_cams, int)
    assert m0.views == 10 and isinstance(m0.views, int)
    assert m0.venue == "Alpha Hali Saha"
    assert m0.title == "1.Saha"
    assert m0.first_url == "u1"

    # tam round-trip: to_rows() girişle (tip-coerce edilmiş) eşleşir
    out = cat.to_rows()
    for src, got in zip(rows, out):
        assert got["id"] == int(src["id"])
        assert got["n_cams"] == int(src["n_cams"])
        assert got["views"] == int(src["views"])
        assert got["venue"] == src["venue"]
        assert got["title"] == src["title"]
        assert got["first_url"] == src["first_url"]


def test_catalog_positional_rows_parse():
    # pozisyonel (dict olmayan) satırlar da COLUMNS düzeninde parse edilmeli
    pos = [(42, "2", "dX", "Zeta Saha", "1.Saha", "99", "uZ")]
    cat = build_catalog(pos)
    m = cat[0]
    assert m.id == 42 and m.n_cams == 2 and m.views == 99
    assert m.venue == "Zeta Saha" and m.first_url == "uZ"


def test_catalog_sqlite_roundtrip(tmp_path):
    cat = build_catalog(synthetic_rows())
    db = str(tmp_path / "cat.db")
    cat.to_sqlite(db)
    back = Catalog.from_sqlite(db)
    assert len(back) == len(cat)
    # sıra ve tüm alanlar korunmalı (rowid sırası)
    assert [r.as_row() for r in back] == [r.as_row() for r in cat]


def test_venue_normalization_merges_truncation():
    # 'X Halı Saha' ile truncated 'X Halı Saha...' aynı tabakaya düşmeli
    rows = [
        {"id": 1, "n_cams": 1, "date": "d", "venue": "Rize Arena Hali Saha", "title": "1", "views": 0, "first_url": "a"},
        {"id": 2, "n_cams": 1, "date": "d", "venue": "rize arena hali saha...", "title": "2", "views": 0, "first_url": "b"},
        {"id": 3, "n_cams": 1, "date": "d", "venue": "Baska Saha", "title": "", "views": 0, "first_url": "c"},
    ]
    cat = build_catalog(rows)
    assert len(cat.venues()) == 2          # Rize iki yazım → tek tabaka, + Baska
    grp = cat.by_venue()
    rize_key = [k for k in grp if "rize" in k][0]
    assert len(grp[rize_key]) == 2


# ====================================================== STRATIFIED SAMPLER ===
def test_stratified_maximizes_distinct_venues():
    cat = build_catalog(synthetic_rows())   # 5 farklı saha
    # n_venues < toplam: tam olarak n_venues farklı saha, hiçbiri tekrar değil
    s = stratified_sample(cat, n_venues=3, per_venue=1)
    assert len(s) == 3
    assert len(set(_venue_keys(s))) == 3     # hepsi farklı


def test_stratified_no_venue_overpicked():
    cat = build_catalog(synthetic_rows())
    # bol bütçe ama per_venue=1: her sahadan EN FAZLA 1 → 5 maç, Alpha (3 maçlı) bir kez
    s = stratified_sample(cat, n_venues=99, per_venue=1)
    keys = _venue_keys(s)
    assert len(s) == 5
    assert len(set(keys)) == 5
    for k in set(keys):
        assert keys.count(k) == 1            # hiçbir saha aşırı-seçilmedi


def test_stratified_round_robin_diversity_first():
    cat = build_catalog(synthetic_rows())
    # per_venue=2: Alpha→2, Beta→2, Gamma/Delta/Epsilon→1 (mevcudu kadar) = 7
    s = stratified_sample(cat, n_venues=99, per_venue=2)
    keys = _venue_keys(s)
    counts = {k: keys.count(k) for k in set(keys)}
    assert sum(counts.values()) == 7
    # hiçbir saha per_venue'yi aşmaz
    assert all(c <= 2 for c in counts.values())
    # 3 tek-maçlı saha 1, iki çok-maçlı saha 2
    assert sorted(counts.values()) == [1, 1, 1, 2, 2]
    # çeşitlilik-önce: ilk 5 girdi (round 0) 5 FARKLI saha
    assert len(set(keys[:5])) == 5


def test_stratified_within_venue_prefers_two_cam():
    cat = build_catalog(synthetic_rows())
    # Alpha'da 3 maç var; per_venue=1 → tek seçim 2-cam (n_cams=2) olmalı, 1-cam değil
    s = stratified_sample(cat, n_venues=99, per_venue=1)
    alpha = [m for m in s if m.venue_key == "alpha hali saha"]
    assert len(alpha) == 1
    assert alpha[0].n_cams == 2              # 2-cam füzyon tercihli; id=2 (n_cams=1) seçilmedi


def test_stratified_degenerate_args():
    cat = build_catalog(synthetic_rows())
    assert stratified_sample(cat, n_venues=0, per_venue=1) == []
    assert stratified_sample(cat, n_venues=3, per_venue=0) == []
    assert stratified_sample(build_catalog([]), n_venues=3, per_venue=1) == []


# ========================================================= PERCEPTUAL HASH ===
def _smooth_image(seed, h=120, w=160):
    """Düşük-frekanslı, monoton-eğimli + birkaç blob içeren sentetik gri kare.

    Eğim belirgin olduğundan küçük gürültü dHash bitlerini çevirmez → near-duplicate.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    base = 0.6 * xx / w * 255 + 0.4 * yy / h * 255      # monoton gradyan
    # birkaç yumuşak blob (seed'e bağlı yapı)
    for _ in range(3):
        cy, cx = rng.uniform(0, h), rng.uniform(0, w)
        sig = rng.uniform(20, 40)
        base += 60.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sig ** 2))
    return np.clip(base, 0, 255)


def test_phash_identical_zero_distance():
    img = _smooth_image(1)
    assert hamming(dhash(img), dhash(img)) == 0
    assert hamming(ahash(img), ahash(img)) == 0


def test_phash_distinct_images_large_distance():
    a = _smooth_image(1)
    b = np.fliplr(a)            # yatay ayna → gradyan yönü tersine döner, dHash büyük flip
    # yapısal olarak farklı görüntüler için Hamming büyük (eşikten çok yüksek)
    assert hamming(dhash(a), dhash(b)) > 30
    # farklı-seed yapı da near-duplicate eşiğinin (5) üstünde ayrışmalı
    assert hamming(dhash(_smooth_image(1)), dhash(_smooth_image(99))) > 5


def test_phash_dedup_drops_near_duplicates_keeps_distinct():
    base = _smooth_image(1)
    rng = np.random.default_rng(7)
    frames = [
        base,                                   # 0  → tutulur (ilk)
        base + rng.normal(0, 1.5, base.shape),  # 1  → near-identical → atılır
        base.copy(),                            # 2  → birebir → atılır
        _smooth_image(99),                      # 3  → açıkça farklı → tutulur
        _smooth_image(99) + rng.normal(0, 1.5, base.shape),  # 4 → 3'e yakın → atılır
    ]
    kept = phash_dedup(frames, threshold=5, method="dhash")
    assert kept == [0, 3]


def test_phash_dedup_accepts_precomputed_int_hashes():
    # önceden hesaplanmış int hash yolu: aynı/yakın art arda atılır, uzak tutulur
    h0 = 0b0000_0000
    h1 = 0b0000_0001          # h0'a 1 bit (≤ threshold) → atılır
    h2 = 0b1111_1111          # h0'a 8 bit (> threshold) → tutulur
    h3 = 0b1111_1110          # h2'ye 1 bit → atılır
    kept = phash_dedup([h0, h1, h2, h3], threshold=3)
    assert kept == [0, 2]


def test_phash_dedup_empty():
    assert phash_dedup([], threshold=5) == []


# =============================================== REAL FILE (varsa, dinamik GT) =
@pytest.mark.skipif(not os.path.exists(REAL_TSV), reason="gerçek cams TSV yok")
def test_real_tsv_parse_and_diversity():
    cat = Catalog.from_tsv(REAL_TSV)
    assert len(cat) > 0

    # round-trip: ham TSV satırı ile parse edilmiş kayıt aynı alanları taşır
    raw = read_tsv(REAL_TSV)
    assert len(raw) == len(cat)
    r0, m0 = raw[0], cat[0]
    assert m0.id == int(r0["id"])
    assert m0.venue == r0["venue"].strip()
    assert m0.first_url == r0["first_url"].strip()

    # dinamik GT: per_venue=1 + bol bütçe → tam olarak #venue maç, hepsi farklı
    n_distinct = len(cat.venues())
    s = stratified_sample(cat, n_venues=10_000, per_venue=1)
    assert len(s) == n_distinct
    assert len(set(_venue_keys(s))) == n_distinct

    # en az bir saha >1 maça sahipse (gerçek veride Rize Arena 2 kez), o saha 1 kez seçilmeli
    grp = cat.by_venue()
    multi = [k for k, v in grp.items() if len(v) > 1]
    if multi:
        keys = _venue_keys(s)
        for k in multi:
            assert keys.count(k) == 1
