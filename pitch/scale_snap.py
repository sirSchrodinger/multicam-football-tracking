#!/usr/bin/env python3
"""scale_snap — olcek-kilidini DURUSTCE acmanin TEK kanonik evi.

Iki sorumluluk, tek dosya, boylece topdown_stats ve topdown_viz JSON'u kim
yazmis olursa olsun (generator semasi mi, eski JSON aliasi mi) TEK fonksiyona
baglanir:

  (1) GENERATOR  make_v3(): QA-gecmis bir v2 calib'i 'standart-boyut' olcek
      capasiyla v3'e snap'ler. Tek-kameradan mutlak olcek YAYINLANMAMIS ve
      Cankaya KAPALI (uydu olcemez) oldugu icin olcek bir VARSAYIMDAN gelir ->
      'm (approx +-N%)' olarak ETIKETLENIR, asla ciplak 'm'.

  (2) READER     scale_state(): hem generator'un yazdigi kanonik semayi hem
      eski JSON aliaslarini (type/scale_rel_to_m/snapped_label/passed) ayni
      sekilde sınıflandirir. Tuketiciler (topdown_stats/topdown_viz) yalnizca
      bunu cagirir.

DEGISMEZ KURALLAR:
  - Default v2 (scale_anchor=null -> relative_m) HER YERDE kanonik ve DOKUNULMAZ.
  - v3 ayri, opt-in; olcek capasi tasir ve 'm (approx +-N%)' olarak yuzeylenir.
  - 'verified' (olculmus/yayinli) olmadan ASLA ciplak 'm', ASLA sprint sayisi.
  - Capali calib (herhangi bir capa) ayak-pikselinden YENIDEN projekte edilmeli;
    diske-pisirilmis v2 pitch_x'i ASLA yeniden kullanma (sessiz sahte-metre tuzagi).

Lisans: numpy + OpenCV (homography.py uzerinden), BSD/MIT. GPU yok.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from pitch.homography import PitchHomography           # noqa: E402
from pitch.template import PitchTemplate               # noqa: E402
from stats.topdown_stats import accept_calib_qa        # noqa: E402

# format -> ayrik standart boyut kataloğu, (W, L) metre
STD_CATALOG = {
    "7v7": [(20.0, 40.0), (25.0, 45.0), (30.0, 50.0)],
    "5v5": [(15.0, 25.0), (18.0, 28.0), (20.0, 30.0)],
}
DEFAULT_BY_FORMAT = {"7v7": (25.0, 45.0), "5v5": (18.0, 28.0)}


# ===========================================================================
# (1) GENERATOR
# ===========================================================================
def snap_size(fit_aspect: float, fmt: str = "7v7") -> tuple[float, float]:
    """fit-aspect (L/W) -> kataloğdaki aspect'i en yakin (W, L). Esitlikte default."""
    cat = STD_CATALOG[fmt]
    best = None
    best_d = np.inf
    for (Wc, Lc) in cat:
        d = abs((Lc / Wc) - fit_aspect)
        if d < best_d - 1e-12:          # kesin daha-iyi
            best, best_d = (Wc, Lc), d
        elif abs(d - best_d) <= 1e-12:  # esitlik -> format default'u tercih et
            if (Wc, Lc) == DEFAULT_BY_FORMAT.get(fmt):
                best, best_d = (Wc, Lc), d
    return best


def make_v3(calib_in: str, calib_out: str, fmt: str = "7v7",
            band_pct: float = 10.0, anisotropic: bool = False) -> dict:
    """QA-gecmis calib_in'i standart-boyut snap'iyle calib_out'a (v3) yaz.

    Izotropik (default): s = sqrt((Lc/L0)*(Wc/W0)); dunya noktalari s ile
    olceklenir -> fitlenmis aspect KORUNUR, hata iki eksene esit bolunur.
    Izotropik dunya-olceklemesi H tarafindan TAM absorbe edilir -> reprojection
    (piksel) QA INVARYANT kalir (median 9.875 -> 9.875).

    Anizotropik (--anisotropic): X,Y ayri olceklenip dims TAM (Lc,Wc) yapilir;
    ama fitlenmis aspect'i katalog aspect'ine zorlar = ~%5 UYDURMA bozulma.

    Fail-closed: kaynak QA reddedilirse veya karsiliklar yoksa hata firlatir.
    Doner: ozet dict (gate_ok, in_band, asserted dims, scale_factor, qa).
    """
    src = PitchHomography.load(calib_in)
    ok, why = accept_calib_qa(src._qa)
    if not ok:
        raise ValueError(f"kaynak calib QA reddedildi (fail-closed): {why}")
    if src._img_pts_und is None or src._world_pts is None:
        raise ValueError("kaynak calib'te kalibrasyon karsiliklari (_img_pts_und/"
                         "_world_pts) yok; snap icin gerekli")

    L0, W0 = src._dims_m()
    fit_aspect = L0 / W0
    Wc, Lc = snap_size(fit_aspect, fmt)

    img_und = src._img_pts_und
    wpts = src._world_pts.astype(np.float64)
    if anisotropic:
        # dims TAM (Lc, Wc) ama fitlenmis aspect'i bozar (~%5 uydurma). Default degil.
        wnew = wpts.copy()
        wnew[:, 0] = wpts[:, 0] * (Lc / L0)
        wnew[:, 1] = wpts[:, 1] * (Wc / W0)
        Ln, Wn = float(Lc), float(Wc)
        method = "aspect_snap_anisotropic"
        s = None
        sys.stderr.write(
            "[scale_snap] WARN anizotropik: fitlenmis aspect "
            f"{fit_aspect:.3f} -> katalog {Lc/Wc:.3f} zorlandi (~%5 uydurma "
            "aspect bozulmasi)\n")
    else:
        # IZOTROPIK geometrik-ortalama: fitlenmis aspect KORUNUR (default, honest)
        s = float(np.sqrt((Lc / L0) * (Wc / W0)))
        wnew = wpts * s
        Ln, Wn = round(L0 * s, 2), round(W0 * s, 2)
        method = "aspect_snap_isotropic"

    tmpl = (PitchTemplate.seven_a_side if fmt == "7v7"
            else PitchTemplate.five_a_side)(L=float(Ln), W=float(Wn))
    v3 = PitchHomography(src.camera_id, tmpl)
    if src.K is not None and src.dist is not None:
        v3.set_distortion(src.K, src.dist)
    qa = v3.calibrate_manual(img_und, wnew, already_undistorted=True)
    gate_ok, gate_why = accept_calib_qa(v3._qa)
    if not gate_ok:                       # izotropik olcek QA'yi bozmamali
        raise RuntimeError(f"v3 QA gate fail (beklenmedik): {gate_why}")

    band = float(band_pct)
    catalog_label = f"{int(Wc)}x{int(Lc)}"
    provenance = ("no published dims; no goal-width crosscheck; closed venue "
                  "(no satellite); nearest-standard-size snap from fit aspect")
    # KANONIK sema + geriye-donuk eski-JSON aliaslari ayni dict'te
    v3.scale_anchor = {
        # --- kanonik (generator) ---
        "kind": "standard_size",
        "schema": "standard_size/v2",
        "method": method,
        "format": fmt,
        "fit_aspect": round(fit_aspect, 4),
        "catalog_label": catalog_label,
        "catalog_wl": [Wc, Lc],
        "scale_factor": (round(s, 5) if s is not None else None),
        "asserted_dims_LW": [Ln, Wn],
        "band_pct": band,
        "reconciled": True,
        "approximate": True,    # -> birim 'm (approx +-N%)', sprint YOK
        "verified": False,      # olculmedi/yayinlanmadi; True olunca ciplak 'm'
        "label": f"approximate standart-boyut, +-{int(band)}%",
        "provenance": provenance,
        # --- eski-JSON aliaslari (back-compat readers) ---
        "type": "standard_size",
        "scale_rel_to_m": (round(s, 5) if s is not None else None),
        "snapped_label": catalog_label,
        "standard_size_m": {"L": float(Lc), "W": float(Wc)},
        "scaled_dims_m": {"L": Ln, "W": Wn},
        "passed": False,
    }
    v3.source_clip = src.source_clip
    v3.status = "manual_scaled_approx"
    v3.save(calib_out)

    return dict(
        camera_id=src.camera_id, fit_aspect=round(fit_aspect, 4),
        catalog=catalog_label, method=method, asserted_dims_LW=[Ln, Wn],
        scale_factor=s, qa_median_px=round(qa["median_px"], 3),
        gate_ok=bool(gate_ok), gate_reasons=gate_why,
        in_band={
            "L": (Lc * (1 - band / 100) <= Ln <= Lc * (1 + band / 100)),
            "W": (Wc * (1 - band / 100) <= Wn <= Wc * (1 + band / 100)),
        },
        out=calib_out)


# ===========================================================================
# (2) SCHEMA-NORMALIZING READER  (tum tuketiciler bunu cagirir)
# ===========================================================================
@dataclass
class ScaleState:
    quality: str          # 'relative' | 'approx' | 'exact'
    unit: str             # 'relative_m' | 'm'
    unit_label: str       # 'relative_m' | 'm (approx +-N%)' | 'm'
    band_pct: float | None
    scale_factor: float | None
    catalog_label: str | None


def _as_anchor(homo_or_anchor) -> dict | None:
    """homo, ham dict veya None -> scale_anchor dict (None-safe)."""
    if homo_or_anchor is None:
        return None
    if isinstance(homo_or_anchor, dict):
        return homo_or_anchor
    return getattr(homo_or_anchor, "scale_anchor", None)


def scale_state(homo_or_anchor) -> ScaleState:
    """scale_anchor'i (hangi sema olursa olsun) ScaleState'e cevir.

    HER IKI anahtar ailesini kabul eder:
      verified := sa['verified']  (generator) | sa['passed']  (eski JSON)
      sf       := sa['scale_factor']          | sa['scale_rel_to_m']
      label    := sa['catalog_label']         | sa['snapped_label']

    Siniflandirma:
      sa yok / reconciled=False        -> relative (relative_m, metre DEGIL)
      approximate=False & verified     -> exact    (ciplak 'm', sprint serbest)
      aksi (reconciled & approximate)  -> approx   ('m (approx +-N%)', sprint YOK)
    """
    sa = _as_anchor(homo_or_anchor)

    reconciled = bool(sa.get("reconciled", False)) if sa else False
    approximate = bool(sa.get("approximate", False)) if sa else False
    verified = bool(sa.get("verified", sa.get("passed", False))) if sa else False
    band = sa.get("band_pct") if sa else None
    sf = sa.get("scale_factor", sa.get("scale_rel_to_m")) if sa else None
    label = sa.get("catalog_label", sa.get("snapped_label")) if sa else None

    if sa is None or not reconciled:
        return ScaleState("relative", "relative_m", "relative_m",
                          None, None, None)
    if (not approximate) and verified:
        return ScaleState("exact", "m", "m", band, sf, label)
    return ScaleState("approx", "m", f"m (approx ±{int(band or 10)}%)",
                      band, sf, label)


def should_force_reproject(homo) -> bool:
    """Capali (relative-disi) herhangi bir calib ayak-pikselinden YENIDEN
    projekte etmeli; diske-pisirilmis v2 pitch_x'e asla guvenme."""
    return scale_state(homo).quality != "relative"


# ------------------------------------------------------------------- CLI ----
def _main(argv) -> int:
    ap = argparse.ArgumentParser(
        description="standart-boyut snap ile v2 calib'i v3'e olcekle",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("calib_in", help="QA-gecmis kaynak (or. calib/cankaya_cam2_v2.json)")
    ap.add_argument("calib_out", help="cikti v3 (or. calib/cankaya_cam2_v3.json)")
    ap.add_argument("--format", default="7v7", choices=list(STD_CATALOG))
    ap.add_argument("--band-pct", type=float, default=10.0)
    ap.add_argument("--anisotropic", action="store_true",
                    help="dims'i TAM katalog boyutuna zorla (aspect'i bozar; default izotropik)")
    a = ap.parse_args(argv)
    res = make_v3(a.calib_in, a.calib_out, fmt=a.format,
                  band_pct=a.band_pct, anisotropic=a.anisotropic)
    import json
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0 if res["gate_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
