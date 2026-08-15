#!/usr/bin/env python3
"""team_color2 — 2-takim jersey-renk kumeleme + capraz-takim ASSOCIATION-BAN.

Yontem grounding (paper):
  SoccerNet Game State Reconstruction (GSR) 2025 ve KIST-GSR sinifindaki kazanan
  yaklasimlar: oyuncu govde-bolgesi (torso) appearance betimleyicilerini IKI takima
  kumeler, takim kimligini saha-x ortalamasi (cluster mean pitch-x) ile dengeler ve
  TRACKING association maliyetine "capraz-takim CANNOT-LINK" (sonsuz maliyet) enjekte
  eder: iki tespit FARKLI takimsa AYNI iz olamaz. Bu modul o adimlari ucretsiz/lisans-
  temiz (sklearn BSD) parcalarla yeniden kurar.

  Gercek GSR sistemleri torso betimleyicisini derin appearance ag'indan (jersey-renk
  CNN / OSNet) cikarir. Bu MODUL betimleyiciyi DISARIDAN bir OZELLIK MATRISI olarak
  alir (HSV histogram, ortalama LAB, vb. — caller ne uretmisse). Yani ag/GPU bu modulun
  ICINDE DEGIL; betimleyici cikarimi entegrasyon dikisidir (caller'a ait). Modulun
  kendi mantigi (kumeleme + ban) sentetik betimleyicilerle TAM test edilir.

ALPEREN OTORITE DOMAIN KURALI (amator gece halisahasi):
  Bir takim TEK-DUZE YELEK giyer (cogu sari/yesil) = YUKSEK-KESINLIKLI TEK-SINIF capa.
  Digeri KARISIK sahsi kiyafet = bu capanin TUMLEYENI (complement). Bu yuzden simetrik
  2-means HER ZAMAN dogru degildir: dogru yapi "tek SIKI kume (yelek) + dagilmis geri
  kalan (other)". `fit_uniform_anchor` tam bunu yapar: yelek takimini TEK siki kume
  olarak bulur, geri kalani "other"a atar. Iki takim da ayirt-edilebilir tek-duze renkse
  (or. sari vs koyu, ikisi de toplu) `cluster_jersey` (KMeans/GMM) yeter; `team_label
  method='auto'` asimetriye bakip ikisi arasinda secer.

Etiket sozlesmesi:
  0 = takim A,  1 = takim B,  GK_LABEL(2) = kaleci (opsiyonel),  UNKNOWN(-1) = abstain.
  uniform-anchor modunda: 0 = YELEK (tek-duze) takim, 1 = OTHER (karisik).

Capraz-takim maliyeti (hard cannot-link):
  cross_team_cost(li, lj) = inf  (li!=lj ve ikisi de BILINEN takim)  else 0.0
  UNKNOWN(-1) taraf -> 0.0 (belirsizken ZORLA yasaklamayiz; yanlis-negatif tracking
  kirilmasi yaratmamak icin). cross_team_cost_matrix(...) tracker'in maliyet matrisine
  eklenecek (Na,Nb) 0/inf maskesini verir = entegrasyon dikisi.

Lisans: numpy + scipy + sklearn (hepsi BSD). GPU yok, AGPL yok.
"""
from __future__ import annotations

import numpy as np

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


# ------------------------------------------------------------- etiket sabitleri --
UNKNOWN = -1     # abstain / belirsiz -> capraz-takim kisiti UYGULANMAZ
GK_LABEL = 2     # kaleci (opsiyonel ucuncu sinif)

DEFAULTS = dict(
    normalize="zscore",   # 'zscore' | 'l2' | None : heterojen olcekli ozellikleri esitle
    k_density=None,       # uniform-anchor kNN-yogunluk komsu sayisi (None -> veri-suru)
    seed_frac=0.2,        # uniform capa: en-yogun bu oran nokta = tohum (center tahmini)
    min_seed=3,           # tohum minimum nokta sayisi
    auto_asym_ratio=1.8,  # auto: spread_other/spread_uniform bunun ustu -> uniform_anchor
    auto_min_sep=0.6,     # auto: en-buyuk-gap / uniform-spread bunun ustu olmali
    eps=1e-9,
)


# --------------------------------------------------------------- yardimcilar -----
def _as_matrix(features):
    """Ozellik girdisini (N,D) float64 2D dizisine cevirir; NaN/sonsuz reddeder."""
    X = np.asarray(features, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"ozellik matrisi 2D olmali, geldi: shape={X.shape}")
    if X.shape[0] < 2:
        raise ValueError("en az 2 tespit gerekli")
    if not np.isfinite(X).all():
        raise ValueError("ozellik matrisinde NaN/inf var (temizle, sonra cagir)")
    return X


def _normalize(X, mode, eps=1e-9):
    """Ozellikleri esitler. zscore: kolon-bazli z; l2: satir-bazli birim norm."""
    if mode in (None, "none", "None"):
        return X
    if mode == "zscore":
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd = np.where(sd < eps, 1.0, sd)
        return (X - mu) / sd
    if mode == "l2":
        n = np.linalg.norm(X, axis=1, keepdims=True)
        n = np.where(n < eps, 1.0, n)
        return X / n
    raise ValueError(f"bilinmeyen normalize modu: {mode}")


# ---------------------------------------------- (1) tek-duze YELEK capa kumesi ----
def fit_uniform_anchor(features, cfg=None):
    """Tek SIKI kume (YELEK takimi) + dagilmis geri-kalan (other) ayrimi.

    Alperen domain kurali: yelek takimi renk-uzayinda SIKI tek-sinif capa; other karisik
    (yuksek varyans). Yontem: (a) kNN-yogunlukla en-yogun bolgeyi tohumla, (b) tohum
    merkezine uzakliklari sirala, (c) siralanmis uzakliklarda EN-BUYUK-GAP esigi ile siki
    kumeyi (yakin olanlar) other'dan ayir. Esik VERI-SURUMLU; sabit yaricap UYDURMAZ.

    Doner dict: labels(0=uniform,1=other), team_conf, center, threshold,
                spread_uniform, spread_other, separation, n_uniform.
    """
    cfg = dict(DEFAULTS, **(cfg or {}))
    eps = cfg["eps"]
    X = _normalize(_as_matrix(features), cfg["normalize"], eps)
    N = X.shape[0]

    # kNN-yogunluk: yogun nokta = kucuk k.komsu-uzakligi
    k = cfg["k_density"] or max(2, min(10, N // 3))
    k = min(k, N - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)   # +1: kendisi dahil
    dists, _ = nn.kneighbors(X)
    kdist = dists[:, -1]                              # k. komsuya uzaklik

    # tohum = en yogun (en kucuk kdist) noktalar -> robust merkez tahmini
    n_seed = max(cfg["min_seed"], int(np.ceil(cfg["seed_frac"] * N)))
    n_seed = int(min(n_seed, N - 1))
    seed = np.argsort(kdist)[:n_seed]
    center = X[seed].mean(axis=0)

    # merkeze uzakliklar + siralama
    d = np.linalg.norm(X - center, axis=1)
    order = np.argsort(d)
    ds = d[order]

    # en-buyuk-gap esigi: split i -> uyeler={ilk i+1 nokta}. En az n_seed uye, en az 1 other.
    gaps = np.diff(ds)
    lo = max(0, n_seed - 1)         # uyeler >= n_seed
    hi = N - 2                      # other >= 1
    if hi < lo:                     # cok kucuk N -> hepsi uniform say
        labels = np.zeros(N, dtype=np.int64)
        return dict(labels=labels, team_conf=np.ones(N), center=center,
                    threshold=float("inf"), spread_uniform=float(ds.mean()),
                    spread_other=0.0, separation=0.0, n_uniform=int(N),
                    note="N cok kucuk: gap aranamadi, hepsi uniform")
    istar = lo + int(np.argmax(gaps[lo:hi + 1]))
    threshold = 0.5 * (ds[istar] + ds[istar + 1])
    members = d <= threshold

    # bir-gecis rafinasyon: merkezi uyelerden tekrar hesapla (kararlilik)
    center = X[members].mean(axis=0)
    d = np.linalg.norm(X - center, axis=1)

    labels = np.where(members, 0, 1).astype(np.int64)
    spread_u = float(d[members].mean())            # yelek kumesi INTRINSIK yayilim
    others = ~members
    if others.any():
        # other'in INTRINSIK yayilimi = KENDI merkezine uzaklik (uniform-merkezine DEGIL).
        # Boyle iki ESIT-siki kumeyi (her ikisi de tek-duze) yanlislikla asimetrik gormeyiz.
        oc = X[others].mean(axis=0)
        spread_o = float(np.linalg.norm(X[others] - oc, axis=1).mean())
    else:
        spread_o = 0.0
    sep = float(gaps[istar] / (spread_u + eps))   # gap'in uniform-yayilima gore buyuklugu

    # per-nokta guven: uye ise merkeze yakinlik, other ise esige uzaklik
    conf = np.empty(N)
    conf[members] = np.clip(1.0 - d[members] / (threshold + eps), 0.0, 1.0)
    conf[others] = np.clip((d[others] - threshold) / (threshold + eps), 0.0, 1.0)

    return dict(labels=labels, team_conf=conf, center=center,
                threshold=float(threshold), spread_uniform=spread_u,
                spread_other=spread_o, separation=sep, n_uniform=int(members.sum()))


# --------------------------------------- (2) simetrik 2-takim (KMeans / GMM) ------
def cluster_jersey(features, n_teams=2, with_gk=False, method="kmeans",
                   cfg=None, random_state=0):
    """Ayirt-edilebilir tek-duze renklerde simetrik kumeleme (KMeans/GMM).

    n_teams=2 (+ with_gk -> ekstra kume). GK := EN KUCUK kume (kaleci tek/farkli renk,
    az kisi) -> GK_LABEL(2)'ye yeniden eslenir; kalan iki kume 0/1. Doner dict:
    labels, team_conf, centers, method.
    """
    cfg = dict(DEFAULTS, **(cfg or {}))
    eps = cfg["eps"]
    X = _normalize(_as_matrix(features), cfg["normalize"], eps)
    N = X.shape[0]
    k = n_teams + (1 if with_gk else 0)
    k = min(k, N)

    if method == "kmeans":
        km = KMeans(n_clusters=k, n_init=10, random_state=random_state).fit(X)
        raw = km.labels_
        centers = km.cluster_centers_
    elif method == "gmm":
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             n_init=5, random_state=random_state).fit(X)
        raw = gm.predict(X)
        centers = gm.means_
    else:
        raise ValueError(f"bilinmeyen method: {method} ('kmeans'|'gmm')")

    labels = raw.astype(np.int64).copy()
    if with_gk and k >= 3:
        sizes = np.bincount(raw, minlength=k)
        gk_cl = int(np.argmin(sizes))           # en az kisi = kaleci
        team_cls = [c for c in range(k) if c != gk_cl]
        remap = {gk_cl: GK_LABEL}
        for new, old in enumerate(team_cls[:2]):
            remap[old] = new
        labels = np.array([remap.get(int(c), int(c)) for c in raw], dtype=np.int64)

    # guven: atanan merkez ile en-yakin DIGER takim-merkezi arasi margin (0..1)
    team_mask = labels >= 0
    conf = np.zeros(N)
    cen_for = {}
    for lab in np.unique(labels):
        if lab < 0:
            continue
        cen_for[lab] = X[labels == lab].mean(axis=0)
    cen_keys = list(cen_for.keys())
    for i in range(N):
        li = int(labels[i])
        if li < 0 or li == GK_LABEL or len(cen_keys) < 2:
            conf[i] = 1.0 if li >= 0 else 0.0
            continue
        d_self = np.linalg.norm(X[i] - cen_for[li])
        d_other = min(np.linalg.norm(X[i] - cen_for[l]) for l in cen_keys if l != li)
        conf[i] = float(np.clip((d_other - d_self) / (d_other + d_self + eps), 0.0, 1.0))
    _ = team_mask
    return dict(labels=labels, team_conf=conf, centers=centers, method=method)


# --------------------------------- (3) saha-x ile takim kimligini KANONIKLE -------
def order_labels_by_x(labels, pitch_x):
    """KIST-GSR: cluster mean pitch-x ile takim 0/1'i kanoniklestir.

    Kucuk ortalama-x'li takim 0, digeri 1 olur (GK_LABEL/UNKNOWN korunur). Boyle iki
    kareyi/maci tutarli karsilastirabilirsin (kume-kimligi keyfiligi kalkar).
    """
    labels = np.asarray(labels, dtype=np.int64).copy()
    px = np.asarray(pitch_x, dtype=np.float64)
    teams = [l for l in np.unique(labels) if l in (0, 1)]
    if len(teams) < 2:
        return labels
    mean_x = {}
    for l in teams:
        m = (labels == l) & np.isfinite(px)
        mean_x[l] = float(px[m].mean()) if m.any() else np.inf
    # ortalama-x'e gore sirala -> en kucuk 0
    order = sorted(teams, key=lambda l: mean_x[l])
    remap = {order[0]: 0, order[1]: 1}
    out = labels.copy()
    for old, new in remap.items():
        out[labels == old] = new
    return out


# ---------------------------------------------------- ust seviye dispatcher -------
def fit_teams(features, method="auto", n_teams=2, with_gk=False,
              pitch_x=None, cfg=None, random_state=0):
    """Ust seviye: ozellik matrisinden takim etiketleri + tani.

    method:
      'uniform_anchor' -> tek-siki-yelek vs other (Alperen domain kurali)
      'kmeans'/'gmm'   -> simetrik 2 (+GK) kume
      'auto'           -> asimetriye bak: bir kume digerinden COK daha siki + ayrik ise
                          uniform_anchor; degilse kmeans.
    Doner dict: labels, team_conf, method, + secilen yontemin tanilari.
    """
    cfg = dict(DEFAULTS, **(cfg or {}))
    chosen = method
    if method == "auto":
        ua = fit_uniform_anchor(features, cfg=cfg)
        asym = ua["spread_other"] / (ua["spread_uniform"] + cfg["eps"])
        if (asym >= cfg["auto_asym_ratio"]) and (ua["separation"] >= cfg["auto_min_sep"]):
            chosen = "uniform_anchor"
            res = ua
        else:
            chosen = "kmeans"
            res = cluster_jersey(features, n_teams=n_teams, with_gk=with_gk,
                                 method="kmeans", cfg=cfg, random_state=random_state)
    elif method == "uniform_anchor":
        res = fit_uniform_anchor(features, cfg=cfg)
    elif method in ("kmeans", "gmm"):
        res = cluster_jersey(features, n_teams=n_teams, with_gk=with_gk,
                             method=method, cfg=cfg, random_state=random_state)
    else:
        raise ValueError(f"bilinmeyen method: {method}")

    res = dict(res)
    res["method"] = chosen
    if pitch_x is not None:
        res["labels"] = order_labels_by_x(res["labels"], pitch_x)
    return res


def team_label(features, method="auto", n_teams=2, with_gk=False,
               pitch_x=None, cfg=None, random_state=0):
    """Ozellik matrisi -> takim etiketleri (np.ndarray). fit_teams'in kisa yolu."""
    return fit_teams(features, method=method, n_teams=n_teams, with_gk=with_gk,
                     pitch_x=pitch_x, cfg=cfg, random_state=random_state)["labels"]


# ------------------------------------------- (4) capraz-takim CANNOT-LINK ----------
def cross_team_cost(label_i, label_j, unknown=UNKNOWN):
    """Hard cannot-link: farkli BILINEN takim -> inf; ayni veya belirsiz -> 0.0.

    Tracker association maliyetine eklenince iki FARKLI takim tespiti AYNI iz olamaz.
    UNKNOWN(-1) taraf -> 0.0 (belirsizken zorla yasaklamayiz; yanlis-negatif kirilma yok).
    """
    li, lj = int(label_i), int(label_j)
    if li == unknown or lj == unknown:
        return 0.0
    return float("inf") if li != lj else 0.0


def cross_team_cost_matrix(labels_a, labels_b=None, unknown=UNKNOWN):
    """Vektorize cannot-link maskesi (Na,Nb): farkli-takim -> inf, else 0.

    labels_b None ise labels_a ile kendisi (NxN, kose koselik tracking). Tracker bunu
    Hungarian/greedy maliyet matrisine EKLER (entegrasyon dikisi).
    """
    a = np.asarray(labels_a, dtype=np.int64).reshape(-1, 1)
    b = (a.reshape(1, -1) if labels_b is None
         else np.asarray(labels_b, dtype=np.int64).reshape(1, -1))
    diff = (a != b)
    known = (a != unknown) & (b != unknown)
    cost = np.zeros((a.shape[0], b.shape[1]), dtype=np.float64)
    cost[diff & known] = np.inf
    return cost


# ------------------------------------------------------------- tani: saflik -------
def cluster_purity(labels, ground_truth):
    """Sentetik dogrulama icin kume-saflik orani (Hungarian-esi en-iyi eslesme).

    Her tahmin-kumesini en cok ortustugu GT sinifina atar, dogru-oran doner [0,1].
    """
    labels = np.asarray(labels)
    gt = np.asarray(ground_truth)
    total = 0
    for cl in np.unique(labels):
        mask = labels == cl
        if not mask.any():
            continue
        gt_here = gt[mask]
        vals, counts = np.unique(gt_here, return_counts=True)
        total += counts.max()
    return float(total) / len(labels)


if __name__ == "__main__":   # kucuk sentetik demo (video gerekmez)
    rng = np.random.default_rng(0)
    uni = rng.normal([5, 5, 5], 0.25, size=(7, 3))         # siki yelek kumesi
    oth = rng.uniform(-3, 3, size=(7, 3))                  # dagilmis karisik
    X = np.vstack([uni, oth])
    r = fit_teams(X, method="auto")
    print("method:", r["method"])
    print("labels:", r["labels"])
    print("n_uniform:", r.get("n_uniform"))
