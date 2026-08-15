#!/usr/bin/env python3
"""roster — amator sabit-kamera halisaha icin roster-dogru stitch yardimcilari.

SAF FONKSIYONLAR (I/O yok, izole unit-test edilebilir). track_stitch.py bunlari
import edip cagirir; track_stitch diff'i minimal ve geriye-uyumlu kalir.

DOMAIN GERCEGI (Alperen duzeltmesi): AMATOR macta HAKEM YOK, SEYIRCI YOK. Sahada
sadece oyuncular var. Oyun 12/14/16 KISI (saha boyutu; 7v7=14 en yaygin). Onceki
"18 = 14 + hakem/seyirci" varsayimi YANLIS idi -> 18 kume asiri-bolunme. Hedef:
hayalet/oyuncu-disi tespitleri DURUST sekilde temizleyerek mevcut 18'i gercek
~15-17'ye cekmek; roster=14'u HEADLINE olarak raporlamak ama kume sayisini ASLA
14'e zorlamamak; over-merge'i yapisal olarak imkansiz tutmak.

DURUSTLUK DEGISMEZLERI:
  * Uydurma metre yok: unit relative_m, scale_anchor null, sprint/accel/km-h yok.
  * Over-merge imkansiz: cannot-link DSU + temporal_violations==0 +
    yalnizca-frame-disjoint cluster-merge.
  * Asla 14'e zorlama: sert taban (15) > headline (14) gercek bir bosluk olarak
    raporlanir, collapse edilmez. Belirgin kimlikler (yedek rotasyonu) 14'u
    asabilir ve raporlanir, merge edilmez.
  * Sifir satir dusurulmez; hayaletler etiketlenir, sessizce silinmez.

Lisans: numpy + pandas (BSD/MIT). AGPL yok.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# saha standartlari (kisa_kenar_m, uzun_kenar_m, etiket, roster). ASPECT bazli;
# scale_anchor=null oldugu icin ALAN/mutlak-metre ASLA kullanilmaz.
STD_FORMATS = [
    (20.0, 40.0, "6v6", 12),
    (25.0, 45.0, "7v7", 14),
    (30.0, 50.0, "8v8", 16),
]


# ------------------------------------------------------- detection_floor -----
def detection_floor(df_kept: pd.DataFrame) -> int:
    """SURVIVOR'lar uzerinden DURUST over-merge sert alt siniri.

    int(df_kept.groupby('frame')['tid'].nunique().max()): herhangi bir frame'deki
    es-zamanli AYAKTA-KALAN tracklet sayisinin maksimumu. (Span-araligi
    _clique_floor'un yerine gecer; o gappy tracklet'leri span-ortusmesiyle
    fazla sayiyordu.)

    Guard: saymadan once (tid,frame) cifti tekrar etmemeli.
    """
    assert not df_kept.duplicated(["tid", "frame"]).any(), \
        "detection_floor: tekrar eden (tid,frame) satiri var"
    if len(df_kept) == 0:
        return 0
    return int(df_kept.groupby("frame")["tid"].nunique().max())


# ------------------------------------------------------- concurrency_stats ---
def _dedup_concurrency_series(df: pd.DataFrame) -> np.ndarray:
    """Her frame icin 1m tek-baglanti (single-linkage) bagli-bilesen sayisi.

    Design 3 _dedup_concurrency: ayni oyuncu uzerindeki iki kutu (<1m) tek
    bilesene collapse olur. pitch_x/pitch_y gerekir; yoksa ham unique-tid'e
    geri duser.
    """
    if "pitch_x" not in df.columns or "pitch_y" not in df.columns:
        return df.groupby("frame")["tid"].nunique().to_numpy()
    px = df["pitch_x"].to_numpy(np.float64)
    py = df["pitch_y"].to_numpy(np.float64)
    if not np.isfinite(px).any():
        return df.groupby("frame")["tid"].nunique().to_numpy()
    fr = df["frame"].to_numpy(np.int64)
    order = np.argsort(fr, kind="stable")
    fr_s = fr[order]; px_s = px[order]; py_s = py[order]
    uniq, starts = np.unique(fr_s, return_index=True)
    starts = list(starts) + [len(fr_s)]
    counts = []
    for i in range(len(uniq)):
        a, b = starts[i], starts[i + 1]
        X = px_s[a:b]; Y = py_s[a:b]
        ok = np.isfinite(X) & np.isfinite(Y)
        X = X[ok]; Y = Y[ok]
        m = len(X)
        if m == 0:
            counts.append(0); continue
        # 1m single-linkage bagli-bilesenler (union-find, kucuk m -> O(m^2) yeter)
        parent = list(range(m))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]; x = parent[x]
            return x
        for ii in range(m):
            for jj in range(ii + 1, m):
                if (X[ii] - X[jj]) ** 2 + (Y[ii] - Y[jj]) ** 2 <= 1.0:
                    ri, rj = find(ii), find(jj)
                    if ri != rj:
                        parent[ri] = rj
        counts.append(len({find(x) for x in range(m)}))
    return np.asarray(counts, dtype=np.int64)


def concurrency_stats(df: pd.DataFrame) -> dict:
    """Ham per-frame unique-tid istatistikleri + dedup_p99 (hepsi DIAGNOSTIC).

    Hicbiri floor olarak KULLANILMAZ. Doner: max (hayalet-sismis, sadece teshis),
    p95, median, mode, dedup_p99.
    """
    g = df.groupby("frame")["tid"].nunique()
    arr = g.to_numpy()
    mode_val = int(pd.Series(arr).mode().iloc[0]) if len(arr) else 0
    dedup = _dedup_concurrency_series(df)
    return dict(
        max=int(arr.max()) if len(arr) else 0,
        p95=float(np.percentile(arr, 95)) if len(arr) else 0.0,
        median=float(np.median(arr)) if len(arr) else 0.0,
        mode=mode_val,
        dedup_p99=float(np.percentile(dedup, 99)) if len(dedup) else 0.0,
    )


# ------------------------------------------------------------- flag_spurious -
def _inside_quad(pts: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Vektorize dis-bukey (convex) quad icindelik (winding-bagimsiz; centroid referansli)."""
    pts = np.asarray(pts, float); q = np.asarray(quad, float); c = q.mean(0)
    inside = np.ones(len(pts), bool)
    for i in range(4):
        a = q[i]; b = q[(i + 1) % 4]; e = b - a
        ref = np.sign(e[0] * (c[1] - a[1]) - e[1] * (c[0] - a[0]))
        cr = e[0] * (pts[:, 1] - a[1]) - e[1] * (pts[:, 0] - a[0])
        inside &= (cr * ref >= -1e-6)
    return inside


def flag_spurious(df: pd.DataFrame, summaries: list, pm: np.ndarray, homo, *,
                  n_eph: int = 4, r_dd_m: float = 2.0, dd_frac: float = 0.6,
                  dd_longer_mult: float = 3.0, off_inp: float = 0.5,
                  off_spread_px: float = 8.0, n_offmax: int = 15,
                  static_min_n: int = 40, static_diag_m: float = 1.5,
                  static_out_m: float = 0.5, off_field_frac: float = 0.35,
                  off_field_m: float = 1.0, max_speed_mps: float = 8.0,
                  max_gap_s: float = 8.0) -> dict:
    """KONSERVATIF, COK-KANITLI hayalet/oyuncu-disi isaretleme. tid->reason.

    Kisa-yasam TEK BASINA asla yeterli degil. Kurallar:
      RULE-1 double_detection: n<=n_eph VE ortusen frame'lerinin >=dd_frac'inda
        es-zamanli, >=dd_longer_mult kat daha uzun bir track'in <r_dd_m yakininda.
      RULE-2 off_pitch_stationary: n<=n_offmax VE in_pitch(margin=0) orani
        <off_inp VE ayak-noktasi piksel yayilimi <off_spread_px (direk/canta).
      RULE-3 orphan_blip: n<=2 VE max_gap_s icinde ileri/geri stitch_motion_gate_m
        gecen komsu YOK (stitch'lenemeyen izole blip). Stitch'lenebilir komsusu
        OLAN kisa fragment SILINMEZ (stitch icin tutulur).
      RULE-4 static_foreign: n>=static_min_n VE yorunge cap'i <static_diag_m VE
        medyan konum saha DISINDA out_dist>static_out_m. KALECI GUARDI: cizgi
        ICINDE (out_dist=0) neredeyse-statik kaleci ISARETLENMEZ.
      RULE-5 off_field_persistent: in_pitch(margin=0) orani <off_field_frac VE medyan
        konum saha DISINDA out_dist>off_field_m. HAREKETTEN BAGIMSIZ -> sahada GEZINEN
        kenar-taraftarini da yakalar (RULE-4 statik sart kosuyor, bunu kaciriyordu).
        Oyuncu/kaleci cogunlukla saha-ICI (in_pitch_frac yuksek) -> ISARETLENMEZ.
    """
    tid_arr = df["tid"].to_numpy(np.int64)
    frame_arr = df["frame"].to_numpy(np.int64)
    foot = df[["foot_x", "foot_y"]].to_numpy(np.float64)

    by_tid = {s["tid"]: s for s in summaries}
    life = {s["tid"]: s["n"] for s in summaries}

    # per-frame indeks (es-zamanlilar icin) — pm konumlari ile
    frame_to_rows: dict = {}
    for ridx in range(len(frame_arr)):
        frame_to_rows.setdefault(int(frame_arr[ridx]), []).append(ridx)

    L, W = homo._dims_m()
    # saha-sinir GORUNTU polygonu (margin off_field_m), ILERI-projekte -> far-third'de
    # KARARLI (pixel_to_pitch'in uzak-uc patlamasindan etkilenmez).
    _m = off_field_m
    _quad_img = homo.pitch_to_pixel(np.array(
        [[-_m, -_m], [L + _m, -_m], [L + _m, W + _m], [-_m, W + _m]], float))

    def _out_dist(x, y):
        dx = max(0.0, -x, x - L)
        dy = max(0.0, -y, y - W)
        return float(np.hypot(dx, dy))

    flags: dict = {}

    for s in summaries:
        tid = s["tid"]
        n = s["n"]
        sel = np.where(tid_arr == tid)[0]
        my_pm = pm[sel]
        my_foot = foot[sel]
        my_frames = frame_arr[sel]

        # --- RULE-1 double_detection ---
        if n <= n_eph:
            overlap_frames = 0
            dd_hits = 0
            for r in range(len(sel)):
                fr = int(my_frames[r])
                rows = frame_to_rows.get(fr, [])
                others = [o for o in rows if tid_arr[o] != tid]
                if not others:
                    continue
                overlap_frames += 1
                p = my_pm[r]
                best = False
                for o in others:
                    op = pm[o]
                    d = np.hypot(p[0] - op[0], p[1] - op[1])
                    if d < r_dd_m and life.get(int(tid_arr[o]), 0) >= dd_longer_mult * n:
                        best = True
                        break
                if best:
                    dd_hits += 1
            if overlap_frames > 0 and dd_hits / overlap_frames >= dd_frac:
                flags[tid] = "double_detection"
                continue

        # --- RULE-4 static_foreign (kaleci guardli) ---
        if n >= static_min_n:
            fin = np.isfinite(my_pm[:, 0]) & np.isfinite(my_pm[:, 1])
            if fin.any():
                X = my_pm[fin, 0]; Y = my_pm[fin, 1]
                diag = float(np.hypot(X.max() - X.min(), Y.max() - Y.min()))
                medx = float(np.median(X)); medy = float(np.median(Y))
                od = _out_dist(medx, medy)
                # KALECI GUARDI: cizgi icinde (od~0) statik -> kaleci, ISARETLEME
                if diag < static_diag_m and od > static_out_m:
                    flags[tid] = "static_foreign"
                    continue

        # --- RULE-5 off_field_persistent (GORUNTU-uzayi; hareketten BAGIMSIZ; gezinen taraftar) ---
        # pitch-koordinati far-third'de patladigi icin saha-disiligi GORUNTU polygonunda olc.
        # UZAY TUTARLILIGI: quad undistorted (pitch_to_pixel default) -> foot'u da undistort et
        # (yoksa ham-foot vs undistorted-quad distorsiyon-kaymasi olur).
        foot_und = homo.undistort_points(my_foot)
        inside5 = _inside_quad(foot_und, _quad_img)
        if len(inside5) and float(inside5.mean()) < off_field_frac:
            flags[tid] = "off_field_persistent"
            continue

        # --- RULE-2 off_pitch_stationary ---
        if n <= n_offmax:
            inp = np.asarray(homo.in_pitch(my_pm, margin_m=0.0), dtype=bool)
            inp_frac = float(inp.mean()) if len(inp) else 0.0
            fx = my_foot[:, 0]; fy = my_foot[:, 1]
            spread = float(np.hypot(fx.max() - fx.min(), fy.max() - fy.min())) \
                if len(fx) else 0.0
            if inp_frac < off_inp and spread < off_spread_px:
                flags[tid] = "off_pitch_stationary"
                continue

        # --- RULE-3 orphan_blip ---
        if n <= 2:
            head = s["head"]; tail = s["tail"]
            t0 = s["t0"]; t1 = s["t1"]
            stitchable = False
            for s2 in summaries:
                if s2["tid"] == tid:
                    continue
                # ileri: bu A, s2 B (A.t1 < B.t0)
                if t1 < s2["t0"]:
                    gap = s2["t0"] - t1
                    if 0.0 < gap <= max_gap_s and \
                            homo.stitch_motion_gate_m(tail, s2["head"], gap, max_speed_mps):
                        stitchable = True
                        break
                # geri: s2 A, bu B (s2.t1 < A.t0)
                if s2["t1"] < t0:
                    gap = t0 - s2["t1"]
                    if 0.0 < gap <= max_gap_s and \
                            homo.stitch_motion_gate_m(s2["tail"], head, gap, max_speed_mps):
                        stitchable = True
                        break
            if not stitchable:
                flags[tid] = "orphan_blip"
                continue

    return flags


# --------------------------------------------------------------- field_format
def field_format(L: float, W: float) -> dict:
    """ASPECT-bazli saha-format cikarimi (mutlak-metre iddiasi YOK).

    aspect = uzun_kenar/kisa_kenar. STD_FORMATS icinden min aspect-hatasi secilir.
    ALAN ASLA kullanilmaz (scale_anchor=null).
    """
    longd = max(float(L), float(W))
    shortd = min(float(L), float(W))
    aspect = longd / shortd if shortd > 0 else float("inf")
    cands = []
    for (s_m, l_m, label, roster) in STD_FORMATS:
        fa = l_m / s_m
        err = abs(aspect - fa)
        cands.append(dict(format=label, format_roster=roster,
                          format_aspect=round(fa, 4), match_err=round(err, 4)))
    best = min(cands, key=lambda c: c["match_err"])
    return dict(
        format=best["format"], format_roster=best["format_roster"],
        aspect=round(aspect, 4), match_err=best["match_err"],
        candidates=cands,
    )


# ------------------------------------------------------------ roster_estimate
def roster_estimate(df_kept: pd.DataFrame, homo, expected_players=None) -> dict:
    """3 bagimsiz sinyalden roster headline; ASLA 14'e zorlamaz.

    p95 := round(concurrency p95); dedup_p99; fmt := field_format(*dims). agree:=
    ucu de <=1 icinde. roster_on_field := expected_players verildiyse o, yoksa
    (agree ise fmt.format_roster, degilse p95) — veri primary, prior uzlasir.
    confidence 'high' iff agree (veya beyan edilen expected ile esleserse).
    p95/dedup ALT sinirdir (recall<1); format prior headline'i sabitler.
    """
    conc = concurrency_stats(df_kept)
    p95 = int(round(conc["p95"]))
    dedup_p99 = int(round(conc["dedup_p99"]))
    fmt = field_format(*homo._dims_m())
    fmt_roster = fmt["format_roster"]

    vals = [p95, dedup_p99, fmt_roster]
    agree = (max(vals) - min(vals)) <= 1

    if expected_players is not None:
        roster_on_field = int(expected_players)
        matches_declared = abs(fmt_roster - int(expected_players)) <= 1 or \
            abs(p95 - int(expected_players)) <= 1
        confidence = "high" if (agree or matches_declared) else "medium"
    else:
        roster_on_field = fmt_roster if agree else p95
        confidence = "high" if agree else "low"

    if agree:
        note = ("3 bagimsiz sinyal (p95, dedup_p99, format-prior) <=1 icinde "
                "uyumlu; headline guvenilir.")
    else:
        note = ("sinyaller UYUSMUYOR; veri (p95/dedup) tercih edildi, format "
                "prior raporlandi, 14'e ZORLAMA yok.")

    return dict(
        roster_on_field=roster_on_field,
        confidence=confidence,
        p95=p95,
        dedup_p99=dedup_p99,
        format_roster=fmt_roster,
        format_aspect=fmt["aspect"],
        agreement=bool(agree),
        note=note,
    )


# -------------------------------------------------------- classify_presence
def classify_presence(df_clustered: pd.DataFrame, total_frames: int,
                      core_span: float = 0.85, core_density: float = 0.5,
                      min_obs: int = 60) -> dict:
    """Her player_id'yi 'core' (tum-mac oyuncu) vs 'partial' (yedek/dusuk-recall) sinifla.

    DERIN BULGU (Cankaya): asiri-kume sayisi 'hayalet/parca' degil — co-located cift YOK
    (min 4.5m), birlesecek frame-ayrik cift YOK. 17 kume = 15 KALICI + 2 kismi GERCEK kisi.
    Per-frame algilanan medyan (~12) < mevcut kalici (~15) farki = RECALL acigi (uzak-uc).
    Bu yuzden 14'e ZORLAMAK uydurmadir; dururst olan core/partial ayrimi.

    core := span_frac>=core_span VE density>=core_density. partial := digerleri.
    Doner: dict(per_player{pid:{...}}, n_core, n_partial, note).
    """
    key = "player_id" if "player_id" in df_clustered.columns else "tid"
    per = {}
    n_core = n_partial = 0
    for pid, g in df_clustered.groupby(key):
        if int(pid) < 0 or len(g) < min_obs:
            continue
        f0 = int(g["frame"].min()); f1 = int(g["frame"].max())
        span = f1 - f0 + 1
        span_frac = span / float(max(total_frames, 1))
        density = len(g) / float(max(span, 1))   # span icinde gozlem yogunlugu (1=her frame)
        is_core = (span_frac >= core_span and density >= core_density)
        per[int(pid)] = dict(n_obs=int(len(g)), f0=f0, f1=f1,
                             span_frac=round(span_frac, 3), density=round(density, 3),
                             role="core" if is_core else "partial")
        n_core += int(is_core); n_partial += int(not is_core)
    note = (f"{n_core} core (tum-mac) + {n_partial} kismi (yedek/dusuk-recall). "
            "Asiri-kume 'hayalet' DEGIL: co-located/birlesecek cift yok -> GERCEK kisiler "
            "+ recall acigi. 14'e zorlanmadi (uydurma onleme).")
    return dict(per_player=per, n_core=n_core, n_partial=n_partial, note=note)


# --------------------------------------------------------- cluster_merge_pass
def cluster_merge_pass(cl: list, fps: float, cluster_max_gap_s: float = 30.0,
                       cap: float = 8.0) -> int:
    """Design 2 frame-disjoint under-merge bosluk-doldurma (iteratif).

    cl: cluster dict listesi; her biri: tids(list), frame_set(set), t0, t1,
    head(np(2)), tail(np(2)). A,B yalnizca su durumda merge edilir:
      frame_set'ler DISJOINT (cannot-link bozulmaz) VE span'ler ic-ice-degil
      (A.t1 < B.t0) VE seam hizi = ||B.head - A.tail|| / gap <= cap,
      0 < gap <= cluster_max_gap_s.
    Disjoint birlesim max-es-zamanli'yi ASLA artiramaz -> over-merge imkansiz.
    Doner: n_merges. Her seam hizi loglanir.
    """
    import sys
    n_merges = 0
    merged = True
    while merged:
        merged = False
        m = len(cl)
        for i in range(m):
            for j in range(m):
                if i == j:
                    continue
                A = cl[i]; B = cl[j]
                # ic-ice-degil: A tamamen B'den once
                if not (A["t1"] < B["t0"]):
                    continue
                if not A["frame_set"].isdisjoint(B["frame_set"]):
                    continue
                gap = B["t0"] - A["t1"]
                if not (0.0 < gap <= cluster_max_gap_s):
                    continue
                seam_speed = float(np.linalg.norm(
                    np.asarray(B["head"]) - np.asarray(A["tail"])) / gap)
                if seam_speed > cap:
                    continue
                # MERGE B -> A
                print(f"[cluster_merge] A.tids={A['tids']} <- B.tids={B['tids']} "
                      f"gap={gap:.2f}s seam_speed={seam_speed:.2f} m/s",
                      file=sys.stderr)
                A["tids"] = list(A["tids"]) + list(B["tids"])
                A["frame_set"] = set(A["frame_set"]) | set(B["frame_set"])
                A["t1"] = B["t1"]
                A["tail"] = B["tail"]
                del cl[j]
                n_merges += 1
                merged = True
                break
            if merged:
                break
    return n_merges
