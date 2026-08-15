"""ingest/venue_registry.py — Venue-agnostik kamera kayıt defteri + ölçü-kaynağı.

TEK GERÇEK KAYNAĞI (single source of truth). Üç soruyu cevaplar:
  (1) Bir kamera için AKTİF/SAĞLAM kalibrasyon dosyası hangisi?
  (2) Boyutlar ne, ne kadar güvenilir — metre iddia edebilir miyiz?
  (3) Hangi sahaya manuel kalibrasyon yatırımı yapmalı / self_training LOSO
      doğrulama setine hangisini seçmeli?

Venue-agnostiktir: tek sahaya bağlı değildir. Ücretsiz katmanda yalnızca
tarama artefaktlarıyla (cams_*.tsv + list_*.json) ve diskteki kalibrasyon
JSON'larıyla çalışır — YENİ AĞ ÇAĞRISI YOK.

İçe-aktarma yüzeyi (import-time): json, re, math, csv, dataclasses, pathlib,
datetime, numpy. torch YOK; cv2 YOK. (cv2/frame sezgileri yalnızca opsiyonel,
tembel/çağrı-zamanı hook'larıdır — goal_width_crosscheck'e geçirilen bir homo
nesnesi pixel_to_pitch çağırdığında cv2'yi O kullanır, biz değil.)

pitch.template.PitchTemplate ve PitchHomography.save'in yazdığı kalibrasyon JSON
şemasıyla uyumludur (doğrulanmış alanlar: qa.{median_px, p95_px,
per_zone.{near,mid,far}, n_landmarks, both_halves, outside_hull_flag,
horizon_warn}, pitch_dims_m, template, scale_anchor, H_img2pitch/H_pitch2img).

DÜRÜSTLÜK DURUŞU:
  - Hiçbir metrik uydurulmaz. standard_7v7 saf bir TAHMİNdir (conf 0.25) ve geometri
    ne kadar güzel olursa olsun sonsuza dek RELATİF-birim kalır.
  - environment tek zayıf sinyalden ASLA kesin iddia edilmez; "unknown" geçerli ve
    yaygın bir cevaptır. Asla varsayılan olarak "outdoor"a düşmeyiz.
  - uydu (satellite) boyutu yalnızca ÇATISIZ (covered=="open") sahalar için açıktır;
    çatı sahayı kapatır (içgörü #4). 0.75 ile sınırlıdır ve metre için 3m kale
    çapraz-kontrolünden geçmek zorundadır.

BİLİNEN BAĞIMLILIK (docstring'e bilerek konuldu): bu defter SIRALAR ve İŞARET EDER,
ama ona göre HAREKET ETMEK için henüz-inşa-edilmemiş harvest/downloader gerekir.
Ham video ~15 günde silinir; bu yüzden `available` + `last_seen_utc`
eyleme-geçirilebilirliği kapılar. Buradaki sıralama "hangi sahayı ölçmeye yatırım
yapayım" sorusunun operasyonel cevabıdır, indirme garantisi değil.
"""
from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

# ============================================================== CONSTANTS ====
REGISTRY_PATH = "registry/venues.json"
STD_7V7 = (40.0, 24.0)              # PitchTemplate.seven_a_side defaults; GUESS, conf 0.25
GOAL_WIDTH_M = 3.0                  # halısahanın değişmez tek fiziksel sabiti
GOAL_RESID_MAX_FRAC = 0.05         # 3m çapraz-kontrol kabul bandı
QA_MAX_MEDIAN_PX = 15.0            # kabul kapısı (v2=9.87 geçer, canonical=65 kalır)
QA_INVERSION_FACTOR = 1.5         # near > far*1.5 => ters-H (backwards homography)
DIMS_CONF = {"published": 0.9, "satellite_measured": 0.75,
             "physical_pacing": 0.6, "standard_7v7": 0.25}
METRIC_DIMS_CONF_GATE = 0.6       # altında: metre için geçen kale-çıpası şart

SCHEMA_VERSION = 1

# ============================================================= URL PARSING ===
# Design 2 regex — gerçek veride doğrulandı (3 özel satır + numerik sahalar).
#   /shst373s12/build/20260608233002.A2-2.mp4  -> facility shst373, saha A2, cam 2
#   /shst373s12/build/20260608233002.K2-1.mp4  -> facility shst373, saha K2, cam 1
#   /shst399s12/build/20260608233001.1-1.mp4   -> facility shst399, saha 1,  cam 1
#   /shst775s1/build/20260608234248.1.mp4      -> facility shst775, saha 1,  cam 1
_URL_RE = re.compile(r"/(shst\d+)(?:s\d+)?/build/\d+\.([AKak]?)(\d+)(?:-(\d+))?\.mp4")


def parse_video_url(url):
    """Video URL'inden tesis/saha/kamera bileşenlerini çıkar.

    Döner: {facility_token, saha_code, cam_index, ak_token}
      - facility_token: "shst373" (grup1; sondaki saha-LİSTE 'sXYZ' tesisin parçası DEĞİL).
      - saha_code: ak_token+rakamlar, ör. "A2","K2","1","2" (grup2.upper()+grup3).
      - cam_index: int(grup4) yoksa 1.
      - ak_token: "A"|"K"|None.
    Eşleşme yoksa hepsi None (saha_code=None) — güvenli.
    """
    if not url:
        return {"facility_token": None, "saha_code": None, "cam_index": None,
                "ak_token": None}
    m = _URL_RE.search(str(url))
    if not m:
        return {"facility_token": None, "saha_code": None, "cam_index": None,
                "ak_token": None}
    facility = m.group(1)
    ak_raw = (m.group(2) or "").upper()
    ak_token = ak_raw if ak_raw in ("A", "K") else None
    digits = m.group(3)
    saha_code = (ak_raw + digits) if ak_raw else digits
    cam_index = int(m.group(4)) if m.group(4) else 1
    return {"facility_token": facility, "saha_code": saha_code,
            "cam_index": cam_index, "ak_token": ak_token}


# ====================================================== ENV CLASSIFIER =======
def classify_environment(ak_token, saha_code, title, frame=None):
    """3 katmanlı, ucuzdan-pahalıya environment sınıflandırıcı.

    Döner: (env, source, confidence)   env in {"outdoor","indoor","unknown"}.

    Tier A (BEDAVA, deterministik — saha kodu jetonu, sonra başlık anahtar kelimesi):
        K jetonu  -> indoor (0.9)
        A jetonu  -> outdoor (0.9)
        başlıkta "kapal" -> indoor (0.85)   # Pınar'ı yakalar (.1-1 jetonsuz)
        başlıkta "açık"/"acik" -> outdoor (0.85)
    Tier B (uydu çatı — yalnız OUTDOOR-doğrulama): operatör-destekli hook; operatör
        vermediyse None döner, asla uydurmaz.
    Tier C (frame gökyüzü/çatı sezgisi): opsiyonel tembel cv2 hook; düşük güven, yalnız
        harvest edilmiş bir frame geçirilirse.
    Aksi halde -> ("unknown", None, 0.0). ASLA varsayılan outdoor DEĞİL.
    """
    sc = (saha_code or "")
    tok = (ak_token or "").upper()

    # --- Tier A: saha-kodu jetonu (en güçlü, bedava) ---
    if tok == "K" or sc.startswith("K"):
        return ("indoor", "saha_token_K", 0.9)
    if tok == "A" or sc.startswith("A"):
        return ("outdoor", "saha_token_A", 0.9)

    # --- Tier A: başlık anahtar kelimesi (jeton yoksa) ---
    t = (title or "").lower()
    if "kapal" in t:                       # "Kapalı Saha" -> indoor
        return ("indoor", "title_keyword", 0.85)
    if "açık" in t or "acik" in t:         # "Açık Saha" -> outdoor
        return ("outdoor", "title_keyword", 0.85)

    # --- Tier B: uydu çatı (yalnız operatör-destekli, outdoor-confirm) ---
    # manuel-destek hook'u: operatör vermediyse fabrikasyon yok.
    # (placeholder; bedava katmanda devre-dışı.)

    # --- Tier C: frame sezgisi (opsiyonel, tembel cv2) ---
    if frame is not None:
        guess = _frame_env_hint(frame)     # düşük güven, uydurma yok
        if guess is not None:
            return guess

    return ("unknown", None, 0.0)


def _frame_env_hint(frame):
    """Opsiyonel tembel cv2 hook — harvest edilmiş bir frame'den gök/çatı sezgisi.

    cv2 yalnızca BURADA, çağrı-zamanında içe aktarılır (import-time temiz kalır).
    Güvenli/temkinli: emin değilse None döner (asla uydurmaz)."""
    try:
        import cv2  # noqa: F401  (tembel, opsiyonel)
        import numpy as _np
        arr = _np.asarray(frame)
        if arr.ndim != 3 or arr.shape[0] < 8:
            return None
        # üst-şerit parlaklık vs. mavilik kaba sezgisi — düşük güven.
        top = arr[: max(1, arr.shape[0] // 6)].astype("float64")
        b, g, r = top[..., 0].mean(), top[..., 1].mean(), top[..., 2].mean()
        if b > r + 18 and b > 90:          # mavi-baskın aydınlık şerit ~ gökyüzü
            return ("outdoor", "frame_sky_heuristic", 0.4)
        return None
    except Exception:
        return None


# ====================================================== GOAL ANCHOR ==========
def goal_width_crosscheck(homo, goalpost_px):
    """3m kale genişliği çapraz-kontrolü — ÖLÇEĞİN tek fiziksel çıpası.

    homo: pixel_to_pitch(.) sağlayan bir PitchHomography (cv2'yi O kullanır, biz değil).
    goalpost_px: [[uL,vL],[uR,vR]] iki kale direği ayak pikseli (HAM).

    Döner: {"type":"goal_width","expected_m":3.0,"measured_m":..,"residual_frac":..,
            "passed":bool} veya geçersizse None.

    NOT: Kale çıpası öncelikle GENİŞLİK (Y) eksenini sabitler. BOY ekseni mesafesi,
    L bir tahminse düşük-güvende kalır — geçen bir çıpa TEK BAŞINA tahmini L'yi tam
    metreye YÜKSELTMEZ; yalnızca ÖLÇEĞİ açar, boyut kaynağı (dims_source) yine yönetir.
    """
    if homo is None or goalpost_px is None:
        return None
    pm = homo.pixel_to_pitch(np.asarray(goalpost_px, dtype=float))
    pm = np.asarray(pm, dtype=float).reshape(-1, 2)
    if pm.shape[0] < 2:
        return None
    measured = float(np.linalg.norm(pm[0] - pm[1]))
    resid_frac = abs(measured - GOAL_WIDTH_M) / GOAL_WIDTH_M
    return {"type": "goal_width", "expected_m": GOAL_WIDTH_M,
            "measured_m": measured, "residual_frac": resid_frac,
            "passed": bool(resid_frac < GOAL_RESID_MAX_FRAC)}


# ====================================================== CAMERA RECORD ========
def _utc_now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class CameraRecord:
    """Kamera-başına kayıt; tesis/saha hiyerarşisini iç-içe taşır.

    (Serdivan tek tesis altında İKİ kayıt gerektirir: açık A2 + kapalı K2.)
    """
    camera_id: str = ""
    venue_id: str = ""
    venue_name: str = ""
    facility_token: Optional[str] = None
    saha_code: Optional[str] = None
    cam_index: int = 1
    n_cams: int = 1
    aliases: list = field(default_factory=list)

    environment: str = "unknown"
    environment_source: Optional[str] = None
    environment_confidence: float = 0.0
    covered: str = "unknown"            # roof status: "open"|"covered"|"unknown"

    dims_m: list = field(default_factory=list)         # [L, W]
    dims_source: Optional[str] = None
    dims_confidence: float = 0.0
    scale_anchor: Optional[dict] = None
    metric_claim_allowed: bool = False
    meters_ready: bool = False                          # alias of metric_claim_allowed

    # frame_coverage: kadraj kapsaması (içgörü #2/#3) — ÇATI `covered`'la KARIŞTIRILMAZ
    frame_coverage: dict = field(default_factory=lambda: {
        "full_pitch": None, "off_frame_corners": [],
        "low_conf_zones": [], "covered_fraction_est": None})

    calib_path: Optional[str] = None
    calib_qa: Optional[dict] = None
    calib_status: str = "uncalibrated"   # uncalibrated|linked|rejected

    sample_count: int = 0
    views_total: int = 0
    calibratability_score: float = 0.0
    score_components: dict = field(default_factory=dict)
    last_seen_utc: Optional[str] = None
    available: bool = True
    notes: str = ""

    # ----------------------------------------------------------- methods ----
    def resolve_dims(self):
        """Boyut kaynağı öncelik zinciri + dürüstlük kapısı yeniden-hesabı.

        published(0.9) > satellite_measured (OUTDOOR/çatısız ONLY, 0.75) >
        physical_pacing(0.6) > standard_7v7(0.25).
        - satellite_measured covered!="open" ise SERT-DEVRE-DIŞI (çatı sahayı kapatır).
        - üst katmanı sessizce alt katmana DÜŞÜRME (yalnız uydu-gate zorunlu istisna).
        - dims_m/dims_source/dims_confidence ayarlar, metric_claim'i yeniden hesaplar.
        """
        src = self.dims_source

        # uydu sert-gate: çatısız değilse devre-dışı, tahmine düş
        if src == "satellite_measured" and self.covered != "open":
            self.notes = (self.notes + " | satellite disabled (covered!=open)").strip(" |")
            src = None

        if src not in DIMS_CONF:
            # bilinen gerçek kaynak yok -> standart 7v7 TAHMİNi (relatif sonsuza dek)
            src = "standard_7v7"
            if not self.dims_m:
                self.dims_m = list(STD_7V7)
        else:
            if not self.dims_m:
                # gerçek kaynak iddia edildi ama boyut yoksa makul iskele (yine de
                # güven kaynaktan gelir; metre kapısı çıpaya bağlı kalır)
                self.dims_m = list(STD_7V7)

        self.dims_source = src
        self.dims_confidence = DIMS_CONF[src]
        self._recompute_metric_claim()
        return self

    def _recompute_metric_claim(self):
        """metric_claim_allowed = (dims_conf >= 0.6) AND scale_anchor.passed.

        scale_anchor.passed YALNIZCA goal_width_crosscheck'ten gelir ("bir dict var"
        demek YETMEZ). Saf standart_7v7 tahmini (0.25) sonsuza dek relatif kalır.
        """
        anchor_ok = bool(self.scale_anchor) and bool(self.scale_anchor.get("passed"))
        allowed = (self.dims_confidence >= METRIC_DIMS_CONF_GATE) and anchor_ok
        self.metric_claim_allowed = bool(allowed)
        self.meters_ready = bool(allowed)
        return self.metric_claim_allowed

    def to_template(self):
        """PitchTemplate üret (7v7 default; W<=20 ise 5v5). cv2 İÇE-AKTARMAZ."""
        from pitch.template import PitchTemplate  # tembel; numpy-only, cv2 yok
        L, W = (self.dims_m or list(STD_7V7))[:2]
        L, W = float(L), float(W)
        if W <= 20.0 and L <= 32.0:
            return PitchTemplate.five_a_side(L=L, W=W)
        return PitchTemplate.seven_a_side(L=L, W=W)

    def to_dict(self):
        return asdict(self)


# ====================================================== CALIB ACCEPTANCE =====
def _is_nan_or_none(v):
    if v is None:
        return True
    try:
        return isinstance(v, float) and math.isnan(v)
    except Exception:
        return False


def attach_calibration(rec, calib_path):
    """Kalibrasyon JSON'unu headless oku, qa'yı incele, AKTİF-kalibrasyon işaretçisi kur.

    "Kanonik dosya bozuk olan" hatasını düzeltir: dosya ADINA değil QA'ya bakar.

    REJECT (calib_status="rejected", nedeni notes'a) — şunlardan herhangi biri:
      - median_px > QA_MAX_MEDIAN_PX (15)
      - per_zone.near > per_zone.far * QA_INVERSION_FACTOR (ters-H)
      - n_landmarks < 6
      - per_zone değerlerinden biri NaN/None (dejenere tam-uyum, ör. DRAFT mid=NaN)
    Aksi halde calib_status="linked": calib_path + calib_qa anlık görüntü saklanır;
    JSON'da GÜVENİLİR bir scale_anchor varsa kopyalanır; frame_coverage low_conf_zones
    qa'dan türetilir (horizon_warn veya far>2*near -> ["far_third"];
    outside_hull_flag -> full_pitch tam-saha değil).
    """
    p = Path(calib_path)
    if not p.exists():
        rec.calib_status = "rejected"
        rec.notes = (rec.notes + f" | calib not found: {calib_path}").strip(" |")
        return rec
    try:
        data = json.loads(p.read_text())
    except Exception as e:
        rec.calib_status = "rejected"
        rec.notes = (rec.notes + f" | calib unreadable: {e}").strip(" |")
        return rec

    qa = data.get("qa") or {}
    pz = qa.get("per_zone") or {}
    near, mid, far = pz.get("near"), pz.get("mid"), pz.get("far")
    median_px = qa.get("median_px")
    n_landmarks = qa.get("n_landmarks", 0)

    reasons = []
    if _is_nan_or_none(median_px) or (median_px is not None and not _is_nan_or_none(median_px)
                                      and median_px > QA_MAX_MEDIAN_PX):
        reasons.append(f"median_px={median_px}>{QA_MAX_MEDIAN_PX}")
    if any(_is_nan_or_none(v) for v in (near, mid, far)):
        reasons.append("per_zone NaN/None (degenerate fit)")
    else:
        if near > far * QA_INVERSION_FACTOR:
            reasons.append(f"backwards-H near={near}>far*{QA_INVERSION_FACTOR}")
    if (n_landmarks or 0) < 6:
        reasons.append(f"n_landmarks={n_landmarks}<6")

    rec.calib_path = str(p)
    if reasons:
        rec.calib_status = "rejected"
        rec.calib_qa = {"median_px": median_px, "per_zone": pz,
                        "n_landmarks": n_landmarks}
        rec.notes = (rec.notes + " | calib rejected: " + "; ".join(reasons)).strip(" |")
        return rec

    # ---- kabul edildi: linked ----
    rec.calib_status = "linked"
    rec.calib_qa = {
        "median_px": median_px, "p95_px": qa.get("p95_px"),
        "per_zone": {"near": near, "mid": mid, "far": far},
        "n_landmarks": n_landmarks,
        "both_halves": qa.get("both_halves"),
        "outside_hull_flag": qa.get("outside_hull_flag"),
        "horizon_warn": qa.get("horizon_warn"),
    }

    # güvenilir scale_anchor JSON'da varsa kopyala (çıpa = ölçek kanıtı)
    sa = data.get("scale_anchor")
    if isinstance(sa, dict):
        rec.scale_anchor = sa

    # frame_coverage düşük-güven bölgeleri qa'dan türet (içgörü #2/#3)
    low = []
    horizon_warn = bool(qa.get("horizon_warn"))
    far_gt_2near = (not _is_nan_or_none(near) and not _is_nan_or_none(far)
                    and near > 0 and far > 2 * near)
    if horizon_warn or far_gt_2near:
        low.append("far_third")
    outside = bool(qa.get("outside_hull_flag"))
    rec.frame_coverage = {
        "full_pitch": (None if outside else (rec.frame_coverage or {}).get("full_pitch", True)),
        "off_frame_corners": list((rec.frame_coverage or {}).get("off_frame_corners", [])),
        "low_conf_zones": low,
        "covered_fraction_est": (rec.frame_coverage or {}).get("covered_fraction_est"),
    }
    if outside:
        rec.frame_coverage["full_pitch"] = False

    # boyut/metre kapısını çıpa kopyalandıysa yeniden değerlendir
    rec._recompute_metric_claim()
    return rec


# ====================================================== SCORE / RANK =========
_ENV_MEASURABILITY = {"outdoor": 1.0, "unknown": 0.4, "indoor": 0.15}


def calibratability_score(rec):
    """Kamera-başına 'ölçmeye yatırım yap' skoru. Ölçülebilirlik bir KAPIdır, yalnız ağırlık değil.

    Bileşenler rec.score_components'a yazılır. Skor [0, ~1] aralığındadır.
    """
    env = rec.environment if rec.environment in _ENV_MEASURABILITY else "unknown"
    measurability = _ENV_MEASURABILITY[env] * (0.4 + 0.6 * float(rec.dims_confidence or 0.0))

    fc = rec.frame_coverage or {}
    coverage = fc.get("covered_fraction_est")
    coverage = float(coverage) if coverage else 0.5

    sample_volume = min(math.log1p(max(0, rec.sample_count)) / math.log(30), 1.0)

    if rec.calib_status == "linked" and rec.calib_qa:
        med = rec.calib_qa.get("median_px")
        med = float(med) if (med is not None and not _is_nan_or_none(med)) else QA_MAX_MEDIAN_PX
        calib_quality = max(0.0, 1.0 - med / QA_MAX_MEDIAN_PX)
    else:
        calib_quality = 0.0

    n_cams_bonus = 0.1 if (rec.n_cams or 0) >= 2 else 0.0
    availability = 1.0 if rec.available else 0.3

    score = availability * (0.35 * measurability + 0.20 * coverage
                            + 0.20 * sample_volume + 0.15 * calib_quality
                            + n_cams_bonus)
    rec.score_components = {
        "measurability": measurability, "coverage": coverage,
        "sample_volume": sample_volume, "calib_quality": calib_quality,
        "n_cams_bonus": n_cams_bonus, "availability": availability,
    }
    rec.calibratability_score = float(score)
    return rec.calibratability_score


def rank(reg, measurable_only=True, top=10):
    """Sıralı (desc) kamera listesi — 'hangi sahayı ölçmeye yatırım yapayım' + LOSO seçici.

    measurable_only=True iken environment=="indoor" VE available==False olanlar ELENİR
    (kapalı saha asla uydudan ölçülemez -> güzel kalibrasyon olsa bile metre demosundan
    diskalifiye). reg ham defter dict'i veya {"cameras": {...}} olabilir.
    """
    cams = reg.get("cameras", reg) if isinstance(reg, dict) else {}
    out = []
    for r in cams.values():
        if measurable_only:
            if r.get("environment") == "indoor":
                continue
            if r.get("available") is False:
                continue
        out.append(r)
    out.sort(key=lambda r: r.get("calibratability_score", 0.0), reverse=True)
    return out[: int(top)] if top else out


# ====================================================== SLUG / LISTING =======
_TR_MAP = str.maketrans({
    "ı": "i", "İ": "i", "ş": "s", "Ş": "s", "ç": "c", "Ç": "c",
    "ğ": "g", "Ğ": "g", "ü": "u", "Ü": "u", "ö": "o", "Ö": "o",
})


def _slugify(name):
    if not name:
        return "venue"
    s = name.translate(_TR_MAP).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    return s or "venue"


def _listing_name_to_slug(listing_json):
    """list_*.json'dan venue_name -> place-slug haritası (place.url son bileşeni).

    Tarama TSV'sindeki kısaltılmış adlar listedekilerle birebir eşleşir; eşleşmeyen
    sahalar için ad slugify edilir (build_from_scan'de)."""
    mapping = {}
    if not listing_json:
        return mapping
    try:
        d = json.loads(Path(listing_json).read_text()) if isinstance(listing_json, str) \
            else listing_json
    except Exception:
        return mapping
    for item in (d.get("data") or []):
        place = item.get("place") or {}
        name = place.get("name")
        url = place.get("url") or ""
        if not name:
            continue
        slug = url.rstrip("/").rsplit("/", 1)[-1] if url else _slugify(name)
        mapping[name.strip()] = slug or _slugify(name)
    return mapping


def _env_to_covered(env):
    return {"outdoor": "open", "indoor": "covered"}.get(env, "unknown")


# ====================================================== BUILD / PERSIST ======
def build_from_scan(cams_tsv, listing_json=None, registry_path=REGISTRY_PATH):
    """Tarama artefaktlarından kamera defteri kur (idempotent upsert).

    - TSV başlığı: id,n_cams,date,venue,title,views,first_url.
    - opsiyonel listing_json venue_name->place-slug eşler (venue_id).
    - parse_video_url -> facility/saha/cam; classify_environment; (venue_slug, saha_code,
      cam_index) başına bir CameraRecord. Serdivan İKİ kayıt verir (A2 açık + K2 kapalı)
      tek tesis shst373 altında.
    - sample_count += 1, views_total +=, last_seen_utc damgalanır, available=True.
      Her kayıt için resolve_dims + calibratability_score.
    - registry varsa idempotent upsert. registry/venues.json yazılır.

    Döner: registry dict {schema_version, built_utc, cameras{cid: record-dict}}.
    """
    name2slug = _listing_name_to_slug(listing_json)

    # mevcut defteri yükle (idempotent upsert)
    reg = {"schema_version": SCHEMA_VERSION, "built_utc": _utc_now_iso(),
           "cameras": {}}
    if registry_path and Path(registry_path).exists():
        try:
            reg = json.loads(Path(registry_path).read_text())
            reg.setdefault("cameras", {})
            reg["schema_version"] = SCHEMA_VERSION
        except Exception:
            pass
    cameras = reg["cameras"]

    # bu dosyadaki kayıtları topla (dosya-içi sayım = idempotent re-run)
    current = {}
    with open(cams_tsv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            url = (row.get("first_url") or "").strip()
            venue_name = (row.get("venue") or "").strip()
            title = (row.get("title") or "").strip()
            try:
                views = int((row.get("views") or "0").strip() or 0)
            except ValueError:
                views = 0
            try:
                n_cams = int((row.get("n_cams") or "1").strip() or 1)
            except ValueError:
                n_cams = 1

            parsed = parse_video_url(url)
            if not parsed["facility_token"]:
                continue  # eşleşmeyen URL -> sessizce atla (güvenli)

            venue_slug = name2slug.get(venue_name) or _slugify(venue_name)
            saha_code = parsed["saha_code"]
            cam_index = parsed["cam_index"] or 1
            cid = f"{venue_slug}__s{saha_code or 'x'}__c{cam_index}"

            env, env_src, env_conf = classify_environment(
                parsed["ak_token"], saha_code, title)

            if cid in current:
                rec = current[cid]
                rec.sample_count += 1
                rec.views_total += views
                rec.n_cams = max(rec.n_cams, n_cams)
            else:
                rec = CameraRecord(
                    camera_id=cid, venue_id=venue_slug, venue_name=venue_name,
                    facility_token=parsed["facility_token"], saha_code=saha_code,
                    cam_index=cam_index, n_cams=n_cams,
                    environment=env, environment_source=env_src,
                    environment_confidence=env_conf, covered=_env_to_covered(env),
                    sample_count=1, views_total=views,
                    last_seen_utc=_utc_now_iso(), available=True,
                )
                current[cid] = rec

    # her kayıt için boyut çöz + skor, dict'e çevir, idempotent upsert
    for cid, rec in current.items():
        rec.resolve_dims()
        calibratability_score(rec)
        new_d = rec.to_dict()
        if cid in cameras and isinstance(cameras[cid], dict):
            old = cameras[cid]
            # operatör/önceki taramadan gelen kalibrasyon + alias'ları koru
            for k in ("aliases", "calib_path", "calib_qa", "calib_status",
                      "scale_anchor", "frame_coverage", "notes"):
                if k == "aliases":
                    merged = list(dict.fromkeys(
                        (old.get("aliases") or []) + (new_d.get("aliases") or [])))
                    new_d["aliases"] = merged
                elif old.get(k) not in (None, "", [], {}, "uncalibrated") \
                        and new_d.get(k) in (None, "", [], {}, "uncalibrated"):
                    new_d[k] = old.get(k)
            # kalibrasyon korunduysa metre kapısını da koru (çıpa kopyalanmış olabilir)
            if new_d.get("calib_status") == "linked" and isinstance(
                    new_d.get("scale_anchor"), dict):
                anchor_ok = bool(new_d["scale_anchor"].get("passed"))
                new_d["metric_claim_allowed"] = bool(
                    new_d.get("dims_confidence", 0.0) >= METRIC_DIMS_CONF_GATE
                    and anchor_ok)
                new_d["meters_ready"] = new_d["metric_claim_allowed"]
        cameras[cid] = new_d

    reg["built_utc"] = _utc_now_iso()
    reg["schema_version"] = SCHEMA_VERSION

    if registry_path:
        outp = Path(registry_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(reg, indent=2, ensure_ascii=False))

    return reg


def load_registry(path=REGISTRY_PATH):
    """Defteri JSON'dan yükle."""
    return json.loads(Path(path).read_text())


def resolve_camera(camera_id, reg):
    """camera_id'yi çöz: önce üretilen id, sonra alias'lar (BC köprüsü).

    TÜM tüketiciler buradan çözmeli — üretilen id'yi ASLA string-eşleştirme yapma.
    Döner: kamera kayıt-dict'i veya None.
    """
    if not isinstance(reg, dict):
        return None
    cams = reg.get("cameras", reg)
    if camera_id in cams:
        return cams[camera_id]
    for r in cams.values():
        if camera_id in (r.get("aliases") or []):
            return r
    return None


def sync_calib_json(rec, calib_path=None):
    """Çözülmüş boyutları kalibrasyon JSON'una yaz; scale_anchor SADECE metre izinliyken.

    pitch_dims_m + template.dims_m güncellenir. scale_anchor YALNIZCA
    rec.metric_claim_allowed iken yazılır (export_tracks.py satır ~144'teki
    scale_calibrated doğru flip etsin); aksi halde scale_anchor=null bırakılır.
    "Tahmin üzerinde metre iddiası" denetim hatasını düzelten sözleşme budur.

    rec: CameraRecord veya kayıt-dict. Döner: yazılan JSON dict'i (veya None).
    """
    if isinstance(rec, CameraRecord):
        d = rec.to_dict()
    elif isinstance(rec, dict):
        d = rec
    else:
        return None
    path = calib_path or d.get("calib_path")
    if not path or not Path(path).exists():
        return None
    data = json.loads(Path(path).read_text())

    L, W = (d.get("dims_m") or list(STD_7V7))[:2]
    L, W = float(L), float(W)
    data["pitch_dims_m"] = {"L": L, "W": W}
    if isinstance(data.get("template"), dict):
        data["template"]["dims_m"] = [L, W]

    if d.get("metric_claim_allowed"):
        data["scale_anchor"] = d.get("scale_anchor") or {
            "type": "goal_width", "expected_m": GOAL_WIDTH_M, "passed": True}
    else:
        data["scale_anchor"] = None  # tahmin -> ölçek iddia etme

    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data


__all__ = [
    "REGISTRY_PATH", "STD_7V7", "GOAL_WIDTH_M", "GOAL_RESID_MAX_FRAC",
    "QA_MAX_MEDIAN_PX", "QA_INVERSION_FACTOR", "DIMS_CONF", "METRIC_DIMS_CONF_GATE",
    "parse_video_url", "classify_environment", "goal_width_crosscheck",
    "CameraRecord", "attach_calibration", "calibratability_score", "rank",
    "build_from_scan", "load_registry", "resolve_camera", "sync_calib_json",
]
