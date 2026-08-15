"""ingest/harvest.py — Çok-saha HASAT KATALOĞU + tabakalı örnekleyici + algısal-hash dedup.

Bu modül, futsal CV temel-modeli (foundation) hattının VERİ KATMANIDIR. Ücretsiz
katmanda sosyalhalisaha tarama artefaktlarından (cams_*.tsv / list_*.json) yapılandırılmış
bir maç KATALOĞU kurar ve "75k maç" ölçeğine giden iki ölçeklenme problemini çözer:

  (1) ÇEŞİTLİLİK > HACİM. Bir temel-model için 1000 maçın 1 sahadan gelmesi, 1000
      sahadan 1'er maç gelmesinden çok daha az değerlidir (domain çeşitliliği =
      genelleme). stratified_sample() bu yüzden farklı VENUE (saha/tesis) sayısını
      MAKSİMİZE eder: önce her sahadan bir maç (round-robin), ancak ondan sonra
      saha başına ikinci/üçüncü maç. Hiçbir saha aşırı-seçilmez.

  (2) ZAMANSAL FAZLALIK. Sabit kamerada ardışık kareler neredeyse aynıdır; etiket/eğitim
      için bunları taşımak israftır. phash_dedup() küçük gri görüntü üzerinde basit bir
      average-hash / dHash hesaplar ve son TUTULAN kareye Hamming uzaklığı eşiğin altında
      kalan ardışık (near-duplicate) kareleri atar.

DÜRÜSTLÜK / KAPSAM:
  - DRY-RUN ONLY. Burada AĞ ÇAĞRISI YOKTUR. Modül yalnızca kendisine verilen satırlar/
    kareler üzerinde çalışır; gerçek indirme/decoding çağıran tarafın işidir. Bu, projedeki
    venue_registry'nin bilinçli "harvest henüz yok" bağımlılığını kapatan parçadır.
  - Saf stdlib + numpy. cv2/torch/pandas import edilmez (resize bile numpy bilinear).

LİTERATÜR / TEMEL:
  - dHash (difference hash): N. Krawetz, "Looks Like It", The Hacker Factor Blog (2011)
    — komşu piksel parlaklık karşılaştırması, küçük bozulmalara/aydınlatmaya gürbüz.
  - aHash / pHash ailesi: C. Zauner, "Implementation and Benchmarking of Perceptual
    Image Hash Functions", MSc thesis, Upper Austria Univ. of Applied Sciences (2010).
  - Tabakalı örnekleme (stratified sampling) — tabaka = venue; veri-küme çeşitliliği için
    standart varyans-azaltıcı tasarım (Cochran, Sampling Techniques, 1977).
"""
from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

# ============================================================== CONSTANTS ====
# scan_cams satır şeması — venue_registry/scan ile aynı sıralı kolonlar.
COLUMNS = ["id", "n_cams", "date", "venue", "title", "views", "first_url"]

# Bazı kaynaklar (ham site list_*.json) farklı anahtarlar kullanır; köprü için takma adlar.
_ALIASES = {
    "venue": ("venue", "place_name", "place"),
    "views": ("views", "watch_count", "view_count"),
    "first_url": ("first_url", "url", "video_url"),
    "n_cams": ("n_cams", "cams", "n_cameras"),
}


# ============================================================ ROW PARSING ====
def _to_int(v, default: int = 0) -> int:
    """Boş/None/ondalık-string güvenli int çevirimi (tarama verisi düzensiz olabilir)."""
    if v is None:
        return default
    if isinstance(v, (int, np.integer)):
        return int(v)
    s = str(v).strip()
    if not s:
        return default
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return default


def _to_str(v) -> str:
    return "" if v is None else str(v).strip()


def _pick(d: dict, field: str):
    """Bir alanı kanonik adı veya takma adlarından ilk bulunanla çek."""
    for k in _ALIASES.get(field, (field,)):
        if k in d and d[k] not in (None, ""):
            return d[k]
        if k in d:                       # var ama boş: yine de döndür (boş geçerli)
            return d[k]
    # iç-içe place.name desteği
    if field == "venue" and isinstance(d.get("place"), dict):
        return d["place"].get("name")
    return None


def _normalize_row(row: Union[dict, Sequence]) -> "MatchRecord":
    """Bir ham satırı (dict veya pozisyonel sıra) tipli MatchRecord'a çevir."""
    if isinstance(row, dict):
        return MatchRecord(
            id=_to_int(row.get("id")),
            n_cams=_to_int(_pick(row, "n_cams"), default=1),
            date=_to_str(row.get("date")),
            venue=_to_str(_pick(row, "venue")),
            title=_to_str(row.get("title")),
            views=_to_int(_pick(row, "views")),
            first_url=_to_str(_pick(row, "first_url")),
        )
    # pozisyonel sıra: COLUMNS düzeninde
    vals = list(row)
    vals += [None] * (len(COLUMNS) - len(vals))
    return MatchRecord(
        id=_to_int(vals[0]),
        n_cams=_to_int(vals[1], default=1),
        date=_to_str(vals[2]),
        venue=_to_str(vals[3]),
        title=_to_str(vals[4]),
        views=_to_int(vals[5]),
        first_url=_to_str(vals[6]),
    )


def _norm_venue(venue: str) -> str:
    """Gruplama anahtarı: aynı tesisin truncated/varyant yazımlarını birleştir.

    'Rize Arena Halı Saha' iki satırda aynı; 'X Halı Saha' ile 'X Halı Saha...'
    (truncated) aynı tabakaya düşmeli. casefold + boşluk daraltma + sondaki '...'/
    noktalama temizliği.
    """
    s = _to_str(venue).casefold()
    s = " ".join(s.split())              # iç boşlukları daralt
    while s.endswith(".") or s.endswith("…"):
        s = s[:-1].rstrip()
    return s


@dataclass
class MatchRecord:
    id: int
    n_cams: int
    date: str
    venue: str
    title: str
    views: int
    first_url: str

    @property
    def venue_key(self) -> str:
        """Tabakalama / dedup için normalize edilmiş venue anahtarı."""
        return _norm_venue(self.venue)

    def as_row(self) -> dict:
        """Kanonik kolon düzeninde dict (round-trip için)."""
        return {c: getattr(self, c) for c in COLUMNS}


# ================================================================ CATALOG ====
class Catalog:
    """Bellekte yapılandırılmış maç kataloğu (+ opsiyonel stdlib sqlite3 kalıcılığı)."""

    columns = COLUMNS

    def __init__(self, matches: Iterable[MatchRecord]):
        self.matches: List[MatchRecord] = list(matches)

    # --- temel diziliş ----------------------------------------------------
    def __len__(self) -> int:
        return len(self.matches)

    def __iter__(self):
        return iter(self.matches)

    def __getitem__(self, i) -> MatchRecord:
        return self.matches[i]

    # --- gruplama ---------------------------------------------------------
    def by_venue(self) -> Dict[str, List[MatchRecord]]:
        """venue_key -> bu sahadaki maçların listesi (giriş sırası korunur)."""
        groups: Dict[str, List[MatchRecord]] = {}
        for m in self.matches:
            groups.setdefault(m.venue_key, []).append(m)
        return groups

    def venues(self) -> List[str]:
        """Sıralı, tekil venue_key listesi."""
        return sorted(self.by_venue().keys())

    # --- round-trip / serileştirme ---------------------------------------
    def to_rows(self) -> List[dict]:
        return [m.as_row() for m in self.matches]

    def to_sqlite(self, path: str) -> str:
        """Kataloğu stdlib sqlite3 ile tek tabloya yaz. Giriş sırası rowid ile korunur."""
        con = sqlite3.connect(path)
        try:
            con.execute("DROP TABLE IF EXISTS matches")
            con.execute(
                "CREATE TABLE matches ("
                "id INTEGER, n_cams INTEGER, date TEXT, venue TEXT, "
                "title TEXT, views INTEGER, first_url TEXT)"
            )
            con.executemany(
                "INSERT INTO matches (id,n_cams,date,venue,title,views,first_url) "
                "VALUES (?,?,?,?,?,?,?)",
                [(m.id, m.n_cams, m.date, m.venue, m.title, m.views, m.first_url)
                 for m in self.matches],
            )
            con.commit()
        finally:
            con.close()
        return path

    @classmethod
    def from_sqlite(cls, path: str) -> "Catalog":
        con = sqlite3.connect(path)
        try:
            cur = con.execute(
                "SELECT id,n_cams,date,venue,title,views,first_url "
                "FROM matches ORDER BY rowid"
            )
            recs = [
                MatchRecord(id=r[0], n_cams=r[1], date=r[2], venue=r[3],
                            title=r[4], views=r[5], first_url=r[6])
                for r in cur.fetchall()
            ]
        finally:
            con.close()
        return cls(recs)

    # --- dosyadan kurma ---------------------------------------------------
    @classmethod
    def from_tsv(cls, path: str) -> "Catalog":
        return build_catalog(read_tsv(path))

    @classmethod
    def from_json(cls, path: str) -> "Catalog":
        return build_catalog(read_json(path))


def build_catalog(rows: Iterable[Union[dict, Sequence]]) -> Catalog:
    """Ham satır dizisini (dict veya pozisyonel) yapılandırılmış Catalog'a çevir.

    rows: scan_cams kolonlarını (id,n_cams,date,venue,title,views,first_url) taşıyan
          dict'ler ya da aynı sıradaki pozisyonel diziler. Site-stili list_*.json
          anahtarları (place.name, watch_count, url) takma adlarla köprülenir.
    """
    return Catalog(_normalize_row(r) for r in rows)


# --- ham dosya okuyucular (DRY-RUN: yalnız diskten, ağ yok) ----------------
def read_tsv(path: str) -> List[dict]:
    """TSV'yi başlık satırına göre dict listesine oku."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(r) for r in reader]


def read_json(path: str) -> List[dict]:
    """JSON'u satır listesine oku. list / {'data':[...]} / {'rows':[...]} destekler."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return list(obj)
    if isinstance(obj, dict):
        for k in ("rows", "data", "matches"):
            if isinstance(obj.get(k), list):
                return list(obj[k])
    raise ValueError("read_json: tanınmayan JSON yapısı")


# ====================================================== STRATIFIED SAMPLER ===
def _venue_priority(matches: List[MatchRecord]) -> tuple:
    """n_venues bütçesi tabaka sayısından küçükken HANGİ sahalar seçilsin?

    Çeşitlilik açısından hangi sahanın seçildiği eşit-değerlidir; deterministiklik için
    'zengin' sahaları öne alıyoruz: daha çok maç, daha çok kamera (2-cam füzyon için iyi),
    daha çok izlenme. (Tümü desc; venue_key tie-break dışarıda eklenir.)
    """
    return (len(matches),
            max(m.n_cams for m in matches),
            max(m.views for m in matches))


def stratified_sample(catalog: Catalog,
                      n_venues: int,
                      per_venue: int = 1) -> List[MatchRecord]:
    """Çeşitliliği maksimize eden tabakalı örnekleme (tabaka = venue).

    DİZAYN: çeşitlilik > hacim. En fazla `n_venues` farklı sahaya kadar yayıl ve her
    sahadan en fazla `per_venue` maç al. Round-robin ile doldurulur: önce her seçili
    sahadan 1'er (round 0 → sonuçtaki ilk K girdi K farklı sahadır), ANCAK ondan sonra
    saha başına 2., 3., ... maç. Böylece hiçbir saha aşırı-seçilmez ve dönen liste mümkün
    olan en çok sayıda farklı sahayı kapsar.

    Bir saha içinde maç seçimi deterministiktir: önce daha çok kameralı (2-cam füzyon
    avantajı), sonra daha çok izlenen, sonra id.

    Döner: MatchRecord listesi (uzunluk ≤ min(n_venues, #venues) * per_venue, ayrıca her
    sahanın gerçek maç sayısıyla sınırlı).
    """
    if n_venues <= 0 or per_venue <= 0 or len(catalog) == 0:
        return []

    groups = catalog.by_venue()
    # Seçilecek sahaları sırala (öncelik desc, sonra anahtar asc → kararlı).
    venue_order = sorted(groups.keys(),
                         key=lambda v: (tuple(-x for x in _venue_priority(groups[v])), v))
    chosen = venue_order[:n_venues]

    # Her saha içinde maçları sırala.
    ranked = {
        v: sorted(groups[v], key=lambda m: (-m.n_cams, -m.views, m.id))
        for v in chosen
    }

    # Round-robin: çeşitlilik-önce doldurma.
    out: List[MatchRecord] = []
    for r in range(per_venue):
        for v in chosen:
            if r < len(ranked[v]):
                out.append(ranked[v][r])
    return out


# ====================================================== PERCEPTUAL HASHING ===
def _to_gray(img: np.ndarray) -> np.ndarray:
    """Görüntüyü float64 gri seviyeye indir. 2D gri ya da (...,C) renk kabul eder.

    NOT: BGR/RGB ayrımı algısal-hash için önemsizdir (yakın-kopya tespiti renk-sırasına
    duyarlı değil); luminosity ağırlıkları simetrik uygulanır.
    """
    a = np.asarray(img)
    if a.ndim == 2:
        return a.astype(np.float64)
    if a.ndim == 3:
        c = a.shape[2]
        if c >= 3:
            return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float64)
        return a[..., 0].astype(np.float64)
    raise ValueError(f"_to_gray: beklenmeyen şekil {a.shape}")


def _resize_bilinear(img: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Saf-numpy bilinear yeniden boyutlandırma (hash için küçük gri görüntüye indirger)."""
    g = img.astype(np.float64)
    in_h, in_w = g.shape
    if in_h == out_h and in_w == out_w:
        return g
    ys = (np.arange(out_h) + 0.5) * in_h / out_h - 0.5
    xs = (np.arange(out_w) + 0.5) * in_w / out_w - 0.5
    ys = np.clip(ys, 0.0, in_h - 1)
    xs = np.clip(xs, 0.0, in_w - 1)
    y0 = np.floor(ys).astype(int); y1 = np.minimum(y0 + 1, in_h - 1)
    x0 = np.floor(xs).astype(int); x1 = np.minimum(x0 + 1, in_w - 1)
    wy = (ys - y0)[:, None]; wx = (xs - x0)[None, :]
    Ia = g[np.ix_(y0, x0)]; Ib = g[np.ix_(y0, x1)]
    Ic = g[np.ix_(y1, x0)]; Id = g[np.ix_(y1, x1)]
    top = Ia * (1.0 - wx) + Ib * wx
    bot = Ic * (1.0 - wx) + Id * wx
    return top * (1.0 - wy) + bot * wy


def _bits_to_int(bits: np.ndarray) -> int:
    """Boolean bit dizisini (MSB-first) tek bir Python int hash'e paketle."""
    v = 0
    for b in np.asarray(bits).ravel():
        v = (v << 1) | int(bool(b))
    return v


def dhash(img: np.ndarray, hash_size: int = 8) -> int:
    """Difference hash (Krawetz 2011): yatay komşu piksel parlaklık karşılaştırması.

    hash_size x (hash_size+1) gri görüntüye indirger, yan-yana farkların işaretini bit
    olarak paketler → hash_size^2 bitlik tamsayı. Aydınlatma/küçük gürültüye gürbüz.
    """
    g = _to_gray(img)
    small = _resize_bilinear(g, hash_size, hash_size + 1)
    diff = small[:, 1:] > small[:, :-1]
    return _bits_to_int(diff)


def ahash(img: np.ndarray, hash_size: int = 8) -> int:
    """Average hash (Zauner 2010): piksel > ortalama bit maskesi."""
    g = _to_gray(img)
    small = _resize_bilinear(g, hash_size, hash_size)
    bits = small > small.mean()
    return _bits_to_int(bits)


def hamming(a: int, b: int) -> int:
    """İki tamsayı hash arasındaki Hamming (farklı bit) uzaklığı."""
    x = int(a) ^ int(b)
    try:
        return x.bit_count()           # Python 3.10+
    except AttributeError:             # güvenli geri-düşüş
        return bin(x).count("1")


def _coerce_to_hash(item, method: str, hash_size: int) -> int:
    """Bir öğeyi hash int'e çevir: skaler int → hash; 1D bit-vektör → paketle; 2D/3D → görüntü."""
    if isinstance(item, (int, np.integer)):
        return int(item)
    arr = np.asarray(item)
    if arr.ndim <= 1:                  # önceden hesaplanmış bit vektörü
        return _bits_to_int(arr.astype(bool))
    return ahash(arr, hash_size) if method == "ahash" else dhash(arr, hash_size)


def phash_dedup(frames_or_hashes: Iterable,
                threshold: int = 5,
                method: str = "dhash",
                hash_size: int = 8) -> List[int]:
    """Ardışık near-duplicate kareleri at; tutulan karelerin indekslerini döndür.

    Sabit kamerada ardışık kareler neredeyse aynıdır. İlk kareyi tut; sonraki her kare
    için son TUTULAN kareye Hamming uzaklığını ölç — eşiğin (`threshold`) altındaysa kopya
    sayılır ve atılır, üstündeyse tutulur ve yeni referans olur.

    frames_or_hashes: görüntü dizisi (2D gri / (H,W,C) renk), ya da önceden hesaplanmış
                      hash'ler (int) ya da bit-vektörleri. method: 'dhash' | 'ahash'.
    Döner: tutulan indekslerin artan listesi.
    """
    items = list(frames_or_hashes)
    if not items:
        return []
    hashes = [_coerce_to_hash(it, method, hash_size) for it in items]
    kept = [0]
    last = hashes[0]
    for i in range(1, len(hashes)):
        if hamming(hashes[i], last) > threshold:
            kept.append(i)
            last = hashes[i]
    return kept
