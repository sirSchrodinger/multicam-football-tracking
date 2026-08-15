#!/usr/bin/env python3
"""active_select — etiketleme butcesi icin aktif-ogrenme kare secimi.

PROBLEM: 12-dk halisaha macinda ~20k kare var ama elle/yari-elle etiketleyecek
butce kucuk (orn. 200 kare). Hangi kareleri etiketleyelim? Iki kor nokta:
  * BELIRSIZLIK: dedektörün emin olmadigi (dusuk-conf / 0.5'e yakin) kareler
    en cok bilgi tasir -- onlari etiketlemek modeli en cok iyilestirir.
  * COESLILIK (diversity): ardisik kareler neredeyse ayni; sadece belirsizlige
    bakarsak yan-yana 50 near-duplicate kareyi secip butceyi yakariz. Embedding
    uzayinda BIRBIRINE benzer kareleri ELE.

YAKLASIM (literatur):
  * Belirsizlik skoru: MI-AOD (arXiv:2104.02324) / CALD tarzi detektör-ciktisi
    belirsizligi. Kare basina = o karedeki tespitlerin ORTALAMA least-confidence
    (1 - conf) ya da ikili-entropi (0.5'te max) degeri, + opsiyonel zamansal-
    bosluk (temporal-gap) bonusu (tracker'in ID kaybettigi/uzun-bosluk kareleri
    bilgi-yogun olma egiliminde).
  * Cesitlilik: Core-Set k-center greedy (Sener & Savarese, arXiv:1708.00489).
    Klasik farthest-first traversal (Gonzalez 1985) -- her adimda secili-kumeye
    en UZAK kareyi ekle. Bu, secili kumenin "kaplama yaricapini" (max-min
    mesafe) minimize eder => near-duplicate'leri ayni anda secmez.

FUZYON: agirlikli (uncertainty-weighted) Core-Set. Her adimda kazanc
  gain_i = w_i * d_i,  d_i = secili kumeye min embedding-mesafesi,
           w_i = normalize edilmis belirsiklik (+gap) skoru.
  * Bir near-duplicate'in d_i ~ 0 oldugundan kazanci ~ 0 => SECILMEZ (de-dup).
  * Esit-mesafede yuksek-belirsizlik secilir (prefer uncertainty).
  * Belirsizlik tekduze oldugunda gain ~ d_i => saf farthest-first (saf
    cesitlilik): iyi kaplama, rastgele/ardisik secimi yener.
Tohum (ilk secim) = en yuksek belirsizlik => belirsizligi onceler.

DURUSTLUK / SINIRLAR:
  * Embedding'leri DISARIDAN aliyoruz (CNN/DINO/ReID ozellik vektörü). Bu modul
    embedding URETMEZ -- gercek pipeline'da bir backbone forward-pass'i embedding
    saglar; burasi onun ardindaki secim mantigidir (entegrasyon dikisi:
    `embeddings` argumani). Sentetik embedding ile mantik birebir test edilir.
  * Mesafe = ham Euclid (istege bagli L2-normalize = cosine benzeri). Olcek
    embedding'e baglidir; cagiran gerekiyorsa standardize/normalize etmeli.
  * GPL/AGPL yok: yalniz numpy (+ test'te scipy.spatial). GPU gerekmez.

API:
  select_frames(frame_ids, confidences, embeddings, k, temporal_gaps=None, ...)
    -> secim sirasinda k adet (tohum ilk) frame_id listesi.
"""
from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Belirsizlik (uncertainty) skoru
# ---------------------------------------------------------------------------
def _binary_entropy(p: np.ndarray) -> np.ndarray:
    """Ikili entropi (bit), 0.5'te 1.0 ile maksimum; p in [0,1]."""
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))


def frame_uncertainty(confidences, mode: str = "least_confidence",
                      empty_value: float = 0.0) -> np.ndarray:
    """Kare-basi belirsizlik dizisi (n_frames,).

    `confidences` formatlari (hepsi desteklenir):
      * list[seq]  : her eleman o karedeki tespit conf'lari (degisken-uzunluk),
      * 2B ndarray : satir = kare, NaN dolgu yok-sayilir,
      * 1B dizi    : kare basina TEK tespit conf'u.

    mode:
      * 'least_confidence' : per-tespit u = 1 - conf  (dusuk-conf => yuksek u),
      * 'entropy'          : per-tespit u = ikili-entropi (0.5 conf => max u).
    Kare degeri = o karedeki tespitlerin ORTALAMASI. Tespitsiz kare => empty_value.
    """
    if mode not in ("least_confidence", "entropy"):
        raise ValueError(f"bilinmeyen mode: {mode!r}")
    out = np.empty(len(confidences), dtype=np.float64)
    for i, c in enumerate(confidences):
        arr = np.atleast_1d(np.asarray(c, dtype=np.float64)).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            out[i] = empty_value
            continue
        if mode == "least_confidence":
            u = 1.0 - np.clip(arr, 0.0, 1.0)
        else:  # entropy
            u = _binary_entropy(arr)
        out[i] = float(u.mean())
    return out


def _gap_norm(temporal_gaps, n: int) -> np.ndarray:
    """Zamansal-bosluk bayraklarini [0,1]'e normalize et (bool => 0/1)."""
    if temporal_gaps is None:
        return np.zeros(n, dtype=np.float64)
    g = np.asarray(temporal_gaps, dtype=np.float64).ravel()
    if g.shape[0] != n:
        raise ValueError("temporal_gaps uzunlugu frame_ids ile eslesmiyor")
    g = np.where(np.isfinite(g), g, 0.0)
    mx = g.max() if g.size else 0.0
    return g / mx if mx > 0 else g


def _score_to_weights(score_raw: np.ndarray, floor: float) -> np.ndarray:
    """Ham skoru greedy-agirligina [floor, 1] araliginda normalize et.

    Tek-deger (range=0) durumunda hepsi 1.0 => saf cesitlilik (farthest-first).
    floor > 0 olmasi, butce zorlarsa dusuk-belirsizlik karelerinin de
    secilebilmesini saglar (ama yuksek-belirsizlik kuvvetle oncelenir).
    """
    smin = float(score_raw.min())
    smax = float(score_raw.max())
    if smax > smin:
        return floor + (1.0 - floor) * (score_raw - smin) / (smax - smin)
    return np.ones_like(score_raw)


# ---------------------------------------------------------------------------
# Core-Set k-center greedy (agirlikli farthest-first)
# ---------------------------------------------------------------------------
def coreset_kcenter(embeddings: np.ndarray, k: int,
                    weights: np.ndarray | None = None,
                    seed_idx: int | None = None) -> list[int]:
    """Agirlikli Core-Set greedy (Sener & Savarese 2018; Gonzalez 1985).

    Her adimda gain_i = weights_i * min_mesafe(i, secili) maksimize edilir.
    weights=None ise saf farthest-first (max-min mesafe kaplamasi).
    Tohum: seed_idx, yoksa argmax(weights) (en yuksek belirsizlik).
    Doner: secim SIRASINDA index listesi (uzunluk min(k, n)).
    """
    E = np.asarray(embeddings, dtype=np.float64)
    if E.ndim == 1:
        E = E.reshape(-1, 1)
    n = E.shape[0]
    k = int(min(max(k, 0), n))
    if k == 0:
        return []
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).ravel()

    selected: list[int] = []
    chosen = np.zeros(n, dtype=bool)

    s = int(np.argmax(weights)) if seed_idx is None else int(seed_idx)
    selected.append(s)
    chosen[s] = True
    diff = E - E[s]
    min_dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))

    while len(selected) < k:
        gain = weights * min_dist
        gain[chosen] = -np.inf
        nxt = int(np.argmax(gain))
        if not np.isfinite(gain[nxt]):  # k > essiz nokta sayisi: kalan distinct
            nxt = int(np.argmax(~chosen))
        selected.append(nxt)
        chosen[nxt] = True
        diff = E - E[nxt]
        d = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        min_dist = np.minimum(min_dist, d)
    return selected


# ---------------------------------------------------------------------------
# Ana API
# ---------------------------------------------------------------------------
def select_frames(frame_ids, confidences, embeddings, k,
                  temporal_gaps=None, mode: str = "least_confidence",
                  gap_weight: float = 0.5, score_floor: float = 0.05,
                  pool_factor: float | None = None,
                  normalize_emb: bool = False, empty_value: float = 0.0):
    """Etiketleme icin k kareyi sec (belirsizlik x cesitlilik).

    Parametreler
    ------------
    frame_ids   : (n,) her karenin kimligi (int/str). Donus bunlardan secilir.
    confidences : kare-basi tespit conf'lari (bkz. frame_uncertainty formatlari).
    embeddings  : (n, d) kare embedding'leri (DISARIDAN; backbone forward-pass).
    k           : secilecek kare sayisi (k>n ise n'e kirpilir).
    temporal_gaps : opsiyonel (n,) bosluk-bayragi/degeri; belirsizlige bonus.
    mode        : 'least_confidence' | 'entropy'.
    gap_weight  : zamansal-bosluk bonus agirligi (skor = u + gap_weight*gap_norm).
    score_floor : agirlik taban degeri [0,1); 0 = dusuk-belirsizligi tamamen ele.
    pool_factor : verilirse, ONCE belirsizlikten top-(pool_factor*k) havuz, SONRA
                  bu havuzda Core-Set (klasik iki-asamali Suggestive-Annotation).
                  None = tum kareler aday; agirlik tercihini saglar.
    normalize_emb : True => embedding satirlarini L2-normalize (cosine benzeri).
    empty_value : tespitsiz karenin belirsizligi.

    Doner: secim sirasinda (tohum ilk) k adet frame_id listesi.
    """
    fids = np.asarray(frame_ids)
    n = fids.shape[0]
    E = np.asarray(embeddings, dtype=np.float64)
    if E.ndim == 1:
        E = E.reshape(-1, 1)
    if E.shape[0] != n:
        raise ValueError("embeddings satir sayisi frame_ids ile eslesmiyor")
    if len(confidences) != n:
        raise ValueError("confidences uzunlugu frame_ids ile eslesmiyor")
    k = int(min(max(k, 0), n))
    if k == 0:
        return []

    if normalize_emb:
        norms = np.sqrt(np.einsum("ij,ij->i", E, E))
        norms = np.where(norms > 0, norms, 1.0)
        E = E / norms[:, None]

    unc = frame_uncertainty(confidences, mode=mode, empty_value=empty_value)
    gapn = _gap_norm(temporal_gaps, n)
    score_raw = unc + gap_weight * gapn

    # Asama 1 (opsiyonel): belirsizlik havuzu on-filtresi.
    if pool_factor is not None:
        pool_size = int(min(n, max(k, np.ceil(pool_factor * k))))
        pool = np.argsort(-score_raw, kind="stable")[:pool_size]
        Ep = E[pool]
        weights = _score_to_weights(score_raw[pool], score_floor)
        sel_local = coreset_kcenter(Ep, k, weights=weights)
        sel = [int(pool[j]) for j in sel_local]
    else:
        weights = _score_to_weights(score_raw, score_floor)
        sel = coreset_kcenter(E, k, weights=weights)

    return [fids[i].item() if hasattr(fids[i], "item") else fids[i]
            for i in sel]
