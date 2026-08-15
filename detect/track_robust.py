#!/usr/bin/env python3
"""track_robust — takip-saglamligi: ID-swap / occlusion-bosluk / uzak-uc / kadraj-disi.

KONUM: track_stitch.py'nin SAHIBI roster_correct; bu modul ONA DOKUNMAZ. Iki temiz
dikis ile baglanir (run_pipeline icinde, bkz. modul sonu ENTEGRASYON NOTU):

  [PRE-stitch]  presplit(raw_parquet, calib) -> raw_clean_parquet
      Tracklet-ICI teleport (ID-swap imzasi) noktasinda tracklet'i BOLER. stitch()
      kendi cannot-link + tail->head motion-gate'i ile temiz uclari yeniden birlestirir;
      kotu sicrama uzerinden ASLA kopru kurmaz. Sessiz swap'i durust bir dikise cevirir.
      stitch() degismeden ham-temiz parquet'i tuketir.

  [POST-stitch] bridge_gaps(player_keyed_parquet, calib) -> *_bridged.parquet
      Occlusion-farkinda bosluk-koprusu: KISA boslugu (intra-tracklet delik + <=1s
      inter-tracklet bosluk) geometrik interpolasyonla doldurur, interpolated=True +
      in_pitch ile ISARETLER. UZUN bosluk (>1s) doldurulMAZ (mesafe yalani olur) —
      bridged_skipped_long'da RAPORLANIR. stats/topdown_viz degismeden tuketir.

Ayrica diagnose() ham/stitched veride dort hata-modunu NICEL olcer (asagidaki gercek
clip sayilari koddan uretildi, uydurma yok):

  cankaya_cam2_clip2400 (146 tracklet, 35738 satir, fps 24.87, 1920x1080):
    MODE1a tracklet-ici teleport (swap suspect)  : 39 olay / 23 tracklet (jump 2.8-3.1m, 35-78 m/s)
    MODE1b yakin-karsilasma (<1.5m, swap-risk)   : 1507 frame, ~50 olay, 133 tracklet-cifti;
                                                    seam'e <=10 frame denk gelen: 67
    MODE2a tracklet-ici delik (kisa, guvenli)    : 110 tracklet, 802 delik, 2498 frame (med 2, p95 11, max 36)
    MODE2b oyuncu-ici inter-bosluk (post-stitch) : 128 bosluk, 10594 frame (med 70 ~2.8s);
                                                    <=25f(~1s): 26 -> kopru-uygun; >75f: 59 -> doldurulMAZ
    MODE3 uzak-uc piksel-fakiri (foot_y<230)     : satirlarin %33.3'u; box_h med 61.8 vs yakin 92.8; conf 0.648 vs 0.694
    MODE4 kadraj-disi/korner                      : bu klipte LITERAL kirpma YOK (foot x[222,1873] y[161,540],
                                                    bottom_cropped=0); cikis-curumesi uzak-touchline (90/134 olum
                                                    foot_y<300) + sag-yan (x>1670: 24) bolgesinde toplaniyor.

DURUSTLUK: interpolasyon ISARETLI; uzun bosluk doldurulmaz; teleport-bolme yalniz
yuksek-guven (uzak-uc jitter'i haric); per-OYUNCU renk-swap duzeltmesi YAPILMAZ
(olculen birey-appearance sans seviyesinde, intra 0.59~inter 0.58); yalniz TAKIM-renk
seviyesinde cross-team swap flag'lenir. Sayilar gizlenmez.

Lisans: numpy + pandas + (opsiyonel) cv2 — hepsi BSD/MIT/Apache. AGPL yok. GPU yok.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import stats_report
from pitch.homography import PitchHomography

# ----------------------------------------------------------------- esikler ---
# Hepsi statik-kamera + saha-metre domeninde; far_relax uzak-uc jitter'ini korur.
# --- ID-swap (MODE1) ---
# PRIMARY sinyal piksel-uzayinda, OLCEK-BAGIMSIZ: uzak-ucte metre-domeni grazing-acili
# projeksiyon sismesiyle 2417 yanlis-teleport uretiyor (919 far-uc, medyan 80 px/s);
# piksel-uzayi ~77 gercek aday verir. Bunlardan SADECE crossing'e denk gelenler (baska
# bir tracklet'le ENC_DIST_M icinde, COINC_FR frame'de) BOLUNUR; gerisi jitter olarak
# raporlanir, bolunmez (asiri-temizlik karsiti).
PX_SPEED_GATE = 600.0     # PRIMARY: piksel-uzayi anlik hiz tavani (px/s); olcek-bagimsiz
COINC_FR = 2              # teleport frame'i bir crossing'e bu kadar frame icinde denk gelmeli
MIN_SEG_OBS = 5           # bolme sonrasi her iki splinter >= k_end gozlem (gurultu head/tail engeli)
SWAP_SPEED_MPS = 9.0      # CROSS-CHECK (rapor): metre-domeni anlik hiz tavani
SWAP_JUMP_M = 2.0         # CROSS-CHECK: tek-adim siframa esigi (metre)
SWAP_MAX_GAP_FR = 3       # yalniz neredeyse-ardisik frame'lerde teleport say
FAR_FOOT_Y = 230.0        # uzak-uc piksel esigi (rapor far_threshold ile uyumlu)
FAR_SWAP_RELAX = 1.6      # CROSS-CHECK: uzak-ucte foot-noktasi gurultusu -> esigi gevset
ENC_DIST_M = 1.5          # yakin-karsilasma (crossing) yaricapi
ENC_MERGE_FR = 5          # ardisik encounter frame'lerini tek olaya birlestir
BRIDGE_MAX_FR = 25        # ~1s: bunun ustunde inter-bosluk interpolasyonU YOK
BORDER_PX = 40            # kadraj-kenari yakinlik (literal kirpma riski)
SEAM_MARGIN = 0.20        # audit_seams: top-2 aday maliyet marji bunun altinda -> belirsiz


# ============================================================= yardimcilar ===
def _fps(df: pd.DataFrame) -> float:
    tmax = float(df["t_sec"].max())
    return float(df["frame"].max()) / tmax if tmax > 0 else 25.0


def _project(df: pd.DataFrame, homo: PitchHomography) -> np.ndarray:
    """foot piksel -> saha-metre (N,2). Disk pitch_x/y NaN olabilir; her zaman yeniden-projekte."""
    foot = df[["foot_x", "foot_y"]].to_numpy(np.float64)
    return homo.pixel_to_pitch(foot)


# ===================================================== MODE1: ID-swap tespit ==
def _encounter_pair_frames(df, homo):
    """Ham yapi: her tracklet-cifti -> ayni frame'de <ENC_DIST_M oldugu frame kumesi.

    find_encounters (rapor) + find_intra_swaps (crossing-coincidence kapisi) ortak kullanir.
    """
    pm = _project(df, homo)
    fr = df["frame"].to_numpy(np.int64)
    tid = df["tid"].to_numpy(np.int64)
    pair_frames: dict[tuple, set] = {}
    for f in np.unique(fr):
        sel = fr == f
        P = pm[sel]; T = tid[sel]
        n = len(P)
        if n < 2:
            continue
        for a in range(n):
            for b in range(a + 1, n):
                if np.hypot(*(P[a] - P[b])) < ENC_DIST_M:
                    key = (int(min(T[a], T[b])), int(max(T[a], T[b])))
                    pair_frames.setdefault(key, set()).add(int(f))
    return pair_frames


def find_intra_swaps(df, homo, fps=None, pair_frames=None):
    """Tracklet-ICI teleport (ID-swap imzasi).

    PRIMARY = piksel-uzayi, OLCEK-BAGIMSIZ hiz (px_speed > PX_SPEED_GATE) ardisik
    frame'de. Metre-domeni grazing-acili projeksiyon sismesiyle sahte teleport uretir
    (uzak-uc 80 px/s -> metre yalani); piksel-uzayi bunu eler. BOLME yalniz crossing'e
    denk gelen (baska tracklet'le COINC_FR frame icinde encounter) teleport'ta yapilir;
    crossing'siz teleport JITTER olarak raporlanir, BOLUNMEZ (asiri-temizlik karsiti).

    Doner: (swaps, split_at, metric_swaps)
      swaps        : list[dict] tum piksel-teleport adaylari (crossing bayrakli)
      split_at     : {tid: [frame,...]} yalniz crossing-coincident -> presplit tuketir
      metric_swaps : list[dict] metre-domeni cross-check (yalniz RAPOR)
    """
    fps = fps or _fps(df)
    pm = _project(df, homo)
    fx = df["foot_x"].to_numpy(np.float64)
    fy = df["foot_y"].to_numpy(np.float64)
    tid = df["tid"].to_numpy(np.int64)
    fr = df["frame"].to_numpy(np.int64)

    order = np.lexsort((fr, tid))
    tid_s, fr_s = tid[order], fr[order]
    pmx, pmy = pm[order, 0], pm[order, 1]
    px_s, py_s, fy_s = fx[order], fy[order], fy[order]
    uniq, starts, counts = np.unique(tid_s, return_index=True, return_counts=True)

    if pair_frames is None:
        pair_frames = _encounter_pair_frames(df, homo)
    tid_enc: dict[int, set] = {}
    for (a, b), frs in pair_frames.items():
        tid_enc.setdefault(a, set()).update(frs)
        tid_enc.setdefault(b, set()).update(frs)

    swaps = []
    metric_swaps = []
    split_at: dict[int, list[int]] = {}
    for i in range(len(uniq)):
        s, c = int(starts[i]), int(counts[i])
        if c < 2:
            continue
        sl = slice(s, s + c)
        f = fr_s[sl]
        dfr = np.diff(f)
        dt = dfr / fps
        dpx = np.hypot(np.diff(px_s[sl]), np.diff(py_s[sl]))
        dm = np.hypot(np.diff(pmx[sl]), np.diff(pmy[sl]))
        with np.errstate(divide="ignore", invalid="ignore"):
            sp_px = np.where(dt > 0, dpx / dt, 0.0)
            sp_m = np.where(dt > 0, dm / dt, 0.0)
        far_step = (fy_s[sl][:-1] < FAR_FOOT_Y) | (fy_s[sl][1:] < FAR_FOOT_Y)
        enc = tid_enc.get(int(uniq[i]), set())

        # PRIMARY: piksel-uzayi teleport
        mpx = (dfr <= SWAP_MAX_GAP_FR) & (sp_px > PX_SPEED_GATE)
        for k in np.where(mpx)[0]:
            frm = int(f[k + 1])
            crossing = any(abs(e - frm) <= COINC_FR for e in enc)
            swaps.append(dict(tid=int(uniq[i]), frame=frm,
                              jump_px=round(float(dpx[k]), 1),
                              px_speed=round(float(sp_px[k]), 1),
                              jump_m=round(float(dm[k]), 2),
                              speed_mps=round(float(sp_m[k]), 1),
                              far=bool(far_step[k]), crossing=bool(crossing)))
            if crossing:
                split_at.setdefault(int(uniq[i]), []).append(frm)

        # CROSS-CHECK: metre-domeni (yalniz rapor; grazing sismesini gosterir)
        cap = np.where(far_step, SWAP_SPEED_MPS * FAR_SWAP_RELAX, SWAP_SPEED_MPS)
        jcap = np.where(far_step, SWAP_JUMP_M * FAR_SWAP_RELAX, SWAP_JUMP_M)
        mm = (dfr <= SWAP_MAX_GAP_FR) & (sp_m > cap) & (dm > jcap)
        for k in np.where(mm)[0]:
            metric_swaps.append(dict(tid=int(uniq[i]), frame=int(f[k + 1]),
                                     speed_mps=round(float(sp_m[k]), 1),
                                     far=bool(far_step[k])))
    return swaps, split_at, metric_swaps


def find_encounters(df, homo, pair_frames=None):
    """MODE1b: yakin-karsilasma (crossing) pencereleri — in-tracker swap RISKI.

    Ayni frame'de iki tracklet <ENC_DIST_M ise swap-riski penceresi. Doner:
    dict(n_frames, n_events, pairs, seam_risk) — seam_risk: bir uye tracklet'in
    karsilasmaya <=10 frame icinde BITTIGI cift sayisi (en akut swap noktasi).
    """
    if pair_frames is None:
        pair_frames = _encounter_pair_frames(df, homo)
    ends = {int(t): int(g["frame"].max()) for t, g in df.groupby("tid")}
    enc_frames = set()
    for frs in pair_frames.values():
        enc_frames.update(frs)

    ef = sorted(enc_frames)
    events = 0; last = -10**9
    for f in ef:
        if f - last > ENC_MERGE_FR:
            events += 1
        last = f
    seam = sum(1 for key, frs in pair_frames.items()
               if any(abs(ends[t] - max(frs)) <= 10 for t in key))
    return dict(n_frames=len(enc_frames), n_events=events,
                n_pairs=len(pair_frames), seam_risk=seam)


# =============================================== MODE2: occlusion bosluklari ==
def find_intra_holes(df):
    """MODE2a: tracklet-ICI eksik-frame delikleri (tracker ID'yi tuttu, det yok)."""
    holes = []
    n_tids = 0
    for tid, g in df.groupby("tid"):
        f = np.sort(g["frame"].to_numpy(np.int64))
        h = np.diff(f) - 1
        h = h[h > 0]
        if h.size:
            n_tids += 1
            holes.extend(int(x) for x in h)
    holes = np.array(holes, dtype=np.int64)
    if holes.size == 0:
        return dict(n_tids=0, n_holes=0, missed_frames=0)
    return dict(n_tids=n_tids, n_holes=int(holes.size),
                missed_frames=int(holes.sum()),
                med=int(np.median(holes)), p95=int(np.percentile(holes, 95)),
                max=int(holes.max()))


def find_player_gaps(df, mapping):
    """MODE2b: oyuncu-ici inter-tracklet bosluklari (stitch'in bridge'leyecegi).

    mapping: tid->player_id (track_stitch tid2player.json). Doner kova-dagilimi +
    kopru-uygun (<=BRIDGE_MAX_FR) vs uzun (doldurulmaz) ayrimi.
    """
    pid = df["tid"].map(mapping)
    gaps = []
    for p, g in df.assign(_pid=pid).groupby("_pid"):
        segs = sorted((int(gg["frame"].min()), int(gg["frame"].max()))
                      for _, gg in g.groupby("tid"))
        for (a0, a1), (b0, b1) in zip(segs[:-1], segs[1:]):
            gp = b0 - a1 - 1
            if gp > 0:
                gaps.append(gp)
    gaps = np.array(gaps, dtype=np.int64)
    if gaps.size == 0:
        return dict(n_gaps=0, missed_frames=0, bridgeable=0, long_skip=0)
    return dict(n_gaps=int(gaps.size), missed_frames=int(gaps.sum()),
                med=int(np.median(gaps)), p95=int(np.percentile(gaps, 95)),
                max=int(gaps.max()),
                bridgeable=int((gaps <= BRIDGE_MAX_FR).sum()),
                long_skip=int((gaps > BRIDGE_MAX_FR).sum()))


# =============================================== MODE3/4: uzak-uc + kadraj ====
def find_far_and_border(df, homo):
    """MODE3 uzak-uc piksel-fakiri + MODE4 kadraj-disi/korner curumesi (gercek geometriye gore)."""
    foot_y = df["foot_y"].to_numpy(np.float64)
    foot_x = df["foot_x"].to_numpy(np.float64)
    bh = df["box_h"].to_numpy(np.float64)
    conf = df["conf"].to_numpy(np.float64)
    far = foot_y < FAR_FOOT_Y
    W, H = 1920, 1080

    births, deaths = [], []
    for tid, g in df.groupby("tid"):
        g = g.sort_values("frame")
        births.append(g.iloc[0][["foot_x", "foot_y", "frame"]].to_numpy(float))
        deaths.append(g.iloc[-1][["foot_x", "foot_y", "frame"]].to_numpy(float))
    B = np.array(births); D = np.array(deaths)
    # klip-basi/sonu (dogal dogum/olum) haric: gercek saha-ici cikis/giris
    mid_d = D[(D[:, 2] > 30) & (D[:, 2] < df["frame"].max() - 30)]
    mid_b = B[(B[:, 2] > 30) & (B[:, 2] < df["frame"].max() - 30)]
    border_d = int(((D[:, 0] < BORDER_PX) | (D[:, 0] > W - BORDER_PX) |
                    (D[:, 1] < BORDER_PX) | (D[:, 1] > H - BORDER_PX)).sum())
    return dict(
        far_row_frac=round(float(far.mean()), 4),
        box_h_far_med=round(float(np.median(bh[far])), 1),
        box_h_near_med=round(float(np.median(bh[~far])), 1),
        conf_far_med=round(float(np.median(conf[far])), 3),
        conf_near_med=round(float(np.median(conf[~far])), 3),
        literal_border_deaths=border_d,
        bottom_cropped_rows=int(df.get("bottom_cropped",
                                       pd.Series(False, index=df.index)).sum()),
        midtime_deaths=int(len(mid_d)),
        far_touchline_deaths=int((mid_d[:, 1] < 300).sum()) if len(mid_d) else 0,
        side_deaths=int(((mid_d[:, 0] < 250) | (mid_d[:, 0] > 1670)).sum()) if len(mid_d) else 0,
        midtime_births=int(len(mid_b)),
        far_touchline_births=int((mid_b[:, 1] < 300).sum()) if len(mid_b) else 0,
    )


# ============================================================== diagnose() ===
def diagnose(tracks_path, calib_path, mapping_path=None, out_dir=None):
    """Dort hata-modunu ham/stitched veride NICEL olc; JSON rapor uret (uydurma yok)."""
    df = stats_report.load_tracks(tracks_path)
    homo = PitchHomography.load(calib_path)
    fps = _fps(df)

    gpf = df.groupby("frame")["tid"].nunique()
    pair_frames = _encounter_pair_frames(df, homo)
    swaps, split_at, metric_swaps = find_intra_swaps(df, homo, fps, pair_frames)
    n_cross = sum(1 for s in swaps if s["crossing"])
    rep = dict(
        clip=Path(tracks_path).stem, fps=round(fps, 3),
        n_rows=int(len(df)), n_tracklets=int(df["tid"].nunique()),
        per_frame_tids=dict(min=int(gpf.min()), med=float(gpf.median()),
                            max=int(gpf.max()),
                            frac_below_16=round(float((gpf < 16).mean()), 4)),
        mode1a_intra_swaps=dict(
            n=len(swaps),                                   # PRIMARY piksel-teleport
            n_tids=len({s["tid"] for s in swaps}),
            n_crossing=n_cross,                             # crossing-coincident
            n_split_candidates=sum(len(v) for v in split_at.values()),
            n_far=sum(1 for s in swaps if s["far"]),
            metric_n=len(metric_swaps),                     # CROSS-CHECK (rapor)
            metric_n_far=sum(1 for s in metric_swaps if s["far"]),
            top=sorted(swaps, key=lambda s: -s["px_speed"])[:6]),
        mode1b_encounters=find_encounters(df, homo, pair_frames),
        mode2a_intra_holes=find_intra_holes(df),
        mode3_4_far_border=find_far_and_border(df, homo),
    )
    if mapping_path and os.path.exists(mapping_path):
        mapping = {int(k): int(v) for k, v in json.load(open(mapping_path)).items()}
        rep["mode2b_player_gaps"] = find_player_gaps(df, mapping)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        p = os.path.join(out_dir, f"{Path(tracks_path).stem}_robust_diag.json")
        json.dump(rep, open(p, "w"), indent=2)
        rep["_path"] = p
    return rep


# ===================================================== PRE-stitch: presplit ===
def presplit(tracks_path, calib_path, out_path=None):
    """Teleport noktalarinda tracklet'i BOL -> ham-temiz parquet (stitch girdisi).

    Yeni tid'ler eski-max+1'den verilir; bolunen ucler artik AYRI tracklet, stitch()
    bunlari kendi motion-gate'i ile yeniden birlestirir AMA kotu sicrama uzerinden
    KOPRU KURMAZ (cunku iki yan ayni frame-komsulugunda hiz-kapisini gecemez).
    Doner: out_path. df.attrs['meta'] korunur.
    """
    df = stats_report.load_tracks(tracks_path)
    homo = PitchHomography.load(calib_path)
    _, split_at, _ = find_intra_swaps(df, homo)
    if not split_at:
        out_path = out_path or tracks_path
        if out_path != tracks_path:
            df.to_parquet(out_path, index=False)
        return out_path

    next_tid = int(df["tid"].max()) + 1
    new_tid = df["tid"].to_numpy(np.int64).copy()
    fr = df["frame"].to_numpy(np.int64)
    tid = df["tid"].to_numpy(np.int64)
    for t, cuts in split_at.items():
        for cut in sorted(set(cuts)):
            sel = (tid == t) & (fr >= cut) & (new_tid == t)
            left = (new_tid == t) & (fr < cut)
            # MIN-SEG GUARD: her iki splinter >= MIN_SEG_OBS olmali; degilse bolme
            # YAPILMAZ (gurultu head/tail yanlis-dikise yol acar -> asiri-temizlik).
            if sel.sum() >= MIN_SEG_OBS and left.sum() >= MIN_SEG_OBS:
                new_tid[sel] = next_tid
                next_tid += 1
    out = df.copy()
    out["tid"] = new_tid
    out["frag_split"] = (new_tid != df["tid"].to_numpy(np.int64))
    out_path = out_path or str(Path(tracks_path).with_name(
        Path(tracks_path).stem + "_presplit.parquet"))
    out.to_parquet(out_path, index=False)
    return out_path


# ================================================ POST-stitch: bridge_gaps ===
def _interp_segment(f0, f1, p0, p1, fps):
    """[f0+1, f1-1] frame'leri icin lineer ara-konum (saha-metre)."""
    frames = np.arange(f0 + 1, f1)
    if frames.size == 0:
        return frames, np.empty((0, 2))
    a = (frames - f0) / float(f1 - f0)
    pos = p0[None, :] * (1 - a[:, None]) + p1[None, :] * a[:, None]
    return frames, pos


def bridge_gaps(player_keyed_path, calib_path, out_path=None):
    """KISA boslugu interpolasyonla doldur (ISARETLI); UZUN boslugu birak (RAPORLA).

    Girdi: track_stitch *_player.parquet (tid=player_id). Hem tracklet-ici delikleri
    hem <=BRIDGE_MAX_FR inter-tracklet bosluklari oyuncu-ici doldurur. Eklenen satir:
    interpolated=True, conf=NaN, in_pitch yeniden-hesaplanir. Doner: out_path.
    """
    df = stats_report.load_tracks(player_keyed_path)
    homo = PitchHomography.load(calib_path)
    fps = _fps(df)
    if "interpolated" not in df.columns:
        df = df.assign(interpolated=False)

    add_rows = []
    skipped_long = 0
    for pid, g in df.groupby("tid"):  # tid == player_id (player-keyed)
        g = g.sort_values("frame")
        f = g["frame"].to_numpy(np.int64)
        px = g["pitch_x"].to_numpy(np.float64)
        py = g["pitch_y"].to_numpy(np.float64)
        gap_idx = np.where(np.diff(f) > 1)[0]
        for k in gap_idx:
            gap = int(f[k + 1] - f[k] - 1)
            if gap > BRIDGE_MAX_FR or not (np.isfinite(px[k]) and np.isfinite(px[k + 1])):
                skipped_long += 1
                continue
            frames, pos = _interp_segment(int(f[k]), int(f[k + 1]),
                                          np.array([px[k], py[k]]),
                                          np.array([px[k + 1], py[k + 1]]), fps)
            inp = np.asarray(homo.in_pitch(pos), dtype=bool)
            for fi, (X, Y), ip in zip(frames, pos, inp):
                add_rows.append(dict(tid=int(pid), frame=int(fi),
                                     t_sec=float(fi) / fps,
                                     pitch_x=float(X), pitch_y=float(Y),
                                     in_pitch=bool(ip), interpolated=True,
                                     conf=np.nan))
    if add_rows:
        df = pd.concat([df, pd.DataFrame(add_rows)], ignore_index=True)
        df = df.sort_values(["tid", "frame"]).reset_index(drop=True)
    df.attrs.setdefault("robust", {})
    df.attrs["robust"] = dict(bridged_rows=len(add_rows),
                              bridged_skipped_long=skipped_long,
                              bridge_max_fr=BRIDGE_MAX_FR)
    out_path = out_path or str(Path(player_keyed_path).with_name(
        Path(player_keyed_path).stem + "_bridged.parquet"))
    df.to_parquet(out_path, index=False)
    return out_path, df.attrs["robust"]


# ================================================= TAKIM ETIKETI (team_label) ==
def _kmeans2_color(descs, valid):
    """PRIMARY: jersey-renk 2-means (normalize HS-hist vektoru, Euclidean).

    cok-restart (en dusuk inertia). Bhattacharyya-medoid uzak-uc outlier'ina
    yapisip 16-1 dejenere veriyor; Euclidean 2-means dengeli SARI/KOYU ayrimi
    verir (olculen: 11-6). Renk SOFT prior'dur, ASLA sert-zorlanmaz.

    Doner: {player_idx: 0|1} yalniz valid icin.
    """
    v = list(valid)
    if len(v) < 2:
        return {i: 0 for i in v}
    X = np.array([descs[i].flatten().astype(np.float64) for i in v])

    def _run(seed):
        rng = np.random.default_rng(seed)
        c = X[rng.choice(len(X), 2, replace=False)].copy()
        lab = np.zeros(len(X), dtype=int)
        for _ in range(25):
            d0 = ((X - c[0]) ** 2).sum(1)
            d1 = ((X - c[1]) ** 2).sum(1)
            new = (d1 < d0).astype(int)
            if np.array_equal(new, lab) and _ > 0:
                lab = new
                break
            lab = new
            for j in (0, 1):
                if (lab == j).any():
                    c[j] = X[lab == j].mean(0)
        inertia = float(np.minimum(((X - c[0]) ** 2).sum(1),
                                   ((X - c[1]) ** 2).sum(1)).sum())
        return lab, inertia

    best_lab, best_inr = None, np.inf
    for s in range(12):
        lab, inr = _run(s)
        if len(set(lab.tolist())) == 2 and inr < best_inr:
            best_lab, best_inr = lab, inr
    if best_lab is None:                       # tum restart'lar dejenere -> ilk cut
        best_lab = (np.arange(len(X)) % 2)
    return {v[k]: int(best_lab[k]) for k in range(len(v))}


def _motion_community_2cut(df, players):
    """SECONDARY (Alperen pas/etkilesim proxy'si): koordineli-hareket grafigi 2-kesim.

    Top yok -> possession-proxy = oyuncularin hiz-vektoru korelasyonu (es-zamanli
    koordineli hareket). Affinity = pozitif korelasyon; Fiedler vektor isareti 2-cut.
    Doner: {player: 0|1} (ham etiket; team_label renk ile hizalar).
    """
    P = list(players)
    idx = {p: k for k, p in enumerate(P)}
    n = len(P)
    if n < 2:
        return {p: 0 for p in P}
    # per-oyuncu per-frame hiz vektoru (saha-metre)
    vel: dict[int, dict[int, tuple]] = {}
    for p, g in df.groupby("tid"):
        g = g.sort_values("frame")
        f = g["frame"].to_numpy(np.int64)
        x = g["pitch_x"].to_numpy(np.float64)
        y = g["pitch_y"].to_numpy(np.float64)
        d = {}
        for k in range(1, len(f)):
            if f[k] - f[k - 1] <= 3 and np.isfinite(x[k]) and np.isfinite(x[k - 1]):
                d[int(f[k])] = (x[k] - x[k - 1], y[k] - y[k - 1])
        vel[int(p)] = d
    A = np.zeros((n, n), dtype=np.float64)
    for ia in range(n):
        for ib in range(ia + 1, n):
            va, vb = vel[P[ia]], vel[P[ib]]
            common = set(va) & set(vb)
            if len(common) < 5:
                continue
            cs = []
            for fr in common:
                ax, ay = va[fr]; bx, by = vb[fr]
                na = np.hypot(ax, ay); nb = np.hypot(bx, by)
                if na > 1e-6 and nb > 1e-6:
                    cs.append((ax * bx + ay * by) / (na * nb))  # cosine
            if cs:
                w = max(0.0, float(np.mean(cs)))  # pozitif koordinasyon
                A[ia, ib] = A[ib, ia] = w
    deg = A.sum(1)
    if deg.sum() <= 0:
        return {P[k]: (k % 2) for k in range(n)}  # affinity yok -> nondegenerate fallback
    # normalize Laplacian Fiedler vektoru
    L = np.diag(deg) - A
    try:
        w_, V = np.linalg.eigh(L)
        fied = V[:, 1] if V.shape[1] > 1 else V[:, 0]
    except Exception:
        return {P[k]: (k % 2) for k in range(n)}
    return {P[k]: (0 if fied[k] <= 0 else 1) for k in range(n)}


def team_label(player_keyed_df, calib_path, video_path):
    """Oyuncu -> takim (0/1). PRIMARY jersey-renk (2-means HS-hist), SECONDARY
    koordineli-hareket dogrulamasi.

    Renk SOFT prior'dur, ASLA sert-zorlanmaz. Uzak-uc baskin / dusuk-kalite oyuncu
    dusuk-conf isaretlenir. conf = color_quality x (anlasma yoksa 0.5). Ayni-takim
    yumusak swap KURTARILAMAZ -> raporlanir, uydurulmaz.

    Doner: {player_id: {team_id, color_quality, motion_agree, conf}}
    """
    import detect.track_stitch as TS
    df = player_keyed_df
    players = sorted(int(p) for p in df["tid"].unique())
    summaries = [dict(tid=p) for p in players]

    descs, qs = [None] * len(players), np.zeros(len(players))
    if video_path and os.path.exists(video_path):
        try:
            descs, qs = TS._appearance_descriptors(df, summaries, video_path)
        except Exception:
            descs, qs = [None] * len(players), np.zeros(len(players))
    valid = [i for i, d in enumerate(descs) if d is not None]

    # PRIMARY renk 2-kume
    color_lab = _kmeans2_color(descs, valid)    # {idx:0/1} valid icin

    # SECONDARY koordineli-hareket 2-kesim
    motion_lab_p = _motion_community_2cut(df, players)  # {player:0/1}
    motion_lab = {i: motion_lab_p[players[i]] for i in range(len(players))}

    # renk etiketini referans al; hareket etiketini renge HIZALA (en iyi permutasyon)
    both = [i for i in valid if i in color_lab]
    agree_same = sum(1 for i in both if color_lab[i] == motion_lab[i])
    flip = agree_same < (len(both) - agree_same)  # ters daha tutarliysa hareketi cevir
    if flip:
        motion_lab = {i: 1 - v for i, v in motion_lab.items()}

    # uzak-uc baskinligi (dusuk-conf isareti icin)
    far_frac = {}
    for i, p in enumerate(players):
        fy = df.loc[df["tid"] == p, "foot_y"].to_numpy(np.float64)
        fy = fy[np.isfinite(fy)]
        far_frac[i] = float(np.mean(fy < FAR_FOOT_Y)) if fy.size else 1.0

    out = {}
    for i, p in enumerate(players):
        cq = float(qs[i]) if i < len(qs) else 0.0
        if i in color_lab:
            team = int(color_lab[i])                       # renk PRIMARY
            agree = bool(color_lab[i] == motion_lab[i])
            conf = cq * (1.0 if agree else 0.5)
        else:
            team = int(motion_lab[i])                       # renk yok -> hareket
            agree = False
            conf = 0.15 * (1.0 - far_frac[i])               # dusuk, renk-prior'siz
        conf = float(np.clip(conf, 0.0, 1.0))
        out[p] = dict(team_id=team, color_quality=round(cq, 3),
                      motion_agree=bool(agree),
                      far_dominant=bool(far_frac[i] > 0.5),
                      conf=round(conf, 3))
    return out


# ================================================= DIKIS DENETIMI (audit_seams) =
def audit_seams(stitched_df, calib_path, video_path=None):
    """Post-hoc, YIKICI-DEGIL: kabul edilmis dikislerde (player_id ici ardisik
    fragment'lar) renk+sabit-hiz en-iyi-aday belirsizse swap-suspect isaretler.

    top-2 aday maliyet marji < SEAM_MARGIN ise belirsiz. SADECE raporlar; stitch
    ciktisini ASLA sessizce yeniden-yazmaz (durustluk).

    Doner: dict(n_seams, n_ambiguous_seams, suspects=[...])
    """
    df = stitched_df
    homo = PitchHomography.load(calib_path)
    fps = _fps(df)
    pid_col = "player_id" if "player_id" in df.columns else "tid"

    # fragment uc-ozetleri (orijinal tracker tid)
    foot = df[["foot_x", "foot_y"]].to_numpy(np.float64)
    pm = homo.pixel_to_pitch(foot)
    fr = df["frame"].to_numpy(np.int64)
    tcol = df["tid"].to_numpy(np.int64)
    frag = {}
    for t in np.unique(tcol):
        sel = np.where(tcol == t)[0]
        o = sel[np.argsort(fr[sel])]
        frag[int(t)] = dict(
            f0=int(fr[o[0]]), f1=int(fr[o[-1]]),
            head=pm[o[:min(5, len(o))]].mean(0),
            tail=pm[o[-min(5, len(o)):]].mean(0),
            vel=((pm[o[-1]] - pm[o[-min(5, len(o))]]) /
                 max(1, (fr[o[-1]] - fr[o[-min(5, len(o))]]))),
            pid=int(df[pid_col].to_numpy()[o[0]]))

    # opsiyonel renk descriptor (orijinal tid)
    descs = {}
    if video_path and os.path.exists(video_path):
        try:
            import detect.track_stitch as TS
            tids = sorted(frag.keys())
            dl, _ = TS._appearance_descriptors(df, [dict(tid=t) for t in tids], video_path)
            descs = {t: dl[k] for k, t in enumerate(tids)}
        except Exception:
            descs = {}

    def cost(a, b):
        ga, gb = frag[a], frag[b]
        dt = (gb["f0"] - ga["f1"]) / fps
        if dt <= 0:
            return np.inf
        pred = ga["tail"] + ga["vel"] * (gb["f0"] - ga["f1"])  # sabit-hiz tahmini
        pos = float(np.hypot(*(pred - gb["head"])))
        col = 0.0
        if descs.get(a) is not None and descs.get(b) is not None:
            import detect.track_stitch as TS
            col = TS._hist_dist(descs[a], descs[b])
        return pos + 3.0 * col + 0.5 * dt

    # her oyuncu icin zaman-sirali fragment dizisi -> ardisik dikisler
    seams = 0
    suspects = []
    frags_by_pid: dict[int, list] = {}
    for t, g in frag.items():
        frags_by_pid.setdefault(g["pid"], []).append(t)
    all_frags = list(frag.keys())
    for pid, ts in frags_by_pid.items():
        ts.sort(key=lambda t: frag[t]["f0"])
        for a, b in zip(ts[:-1], ts[1:]):
            if frag[b]["f0"] <= frag[a]["f1"]:
                continue  # es-zamanli (dikis degil)
            seams += 1
            # a'nin tail'inden sonra baslayan, frame-ayrik tum adaylar
            cands = []
            for c in all_frags:
                if c == a:
                    continue
                if frag[c]["f0"] > frag[a]["f1"]:
                    cands.append((cost(a, c), c))
            cands.sort()
            if len(cands) < 2:
                continue
            chosen = next((cc for cc in cands if cc[1] == b), None)
            if chosen is None:
                continue
            best, second = cands[0], cands[1]
            margin = float(second[0] - best[0])
            ambiguous = (chosen[1] != best[1]) or (margin < SEAM_MARGIN)
            if ambiguous:
                suspects.append(dict(
                    player_id=int(pid), frag_a=int(a), frag_b=int(b),
                    chosen_cost=round(float(chosen[0]), 3),
                    best_alt=int(best[1]), best_cost=round(float(best[0]), 3),
                    margin=round(margin, 3),
                    chosen_is_best=bool(chosen[1] == best[1])))
    return dict(n_seams=int(seams), n_ambiguous_seams=len(suspects),
                suspects=sorted(suspects, key=lambda s: s["margin"])[:20])


# ============================================== RAPOR (JSON + kisa markdown) ===
def robust_report(diag, bmeta, teams, audit, out_dir, stem="robust"):
    """Per-mode olculen sayilari + auto-duzeltilen vs yalniz-isaretlenen + durustluk
    notlarini tek JSON + kisa markdown olarak yazar. Doner: (json_path, md_path)."""
    os.makedirs(out_dir, exist_ok=True)
    n_low = sum(1 for t in teams.values() if t["conf"] < 0.3) if teams else 0
    teams_split = {}
    if teams:
        for lab in (0, 1):
            teams_split[lab] = sum(1 for t in teams.values() if t["team_id"] == lab)
    rep = dict(
        clip=diag.get("clip"), fps=diag.get("fps"),
        n_rows=diag.get("n_rows"), n_tracklets=diag.get("n_tracklets"),
        auto_corrected=dict(
            presplit_splits=diag["mode1a_intra_swaps"].get("n_split_candidates"),
            bridged_rows=(bmeta or {}).get("bridged_rows"),
        ),
        only_flagged=dict(
            jitter_teleports_not_split=(diag["mode1a_intra_swaps"]["n"]
                                        - diag["mode1a_intra_swaps"]["n_crossing"]),
            long_holes_skipped=(bmeta or {}).get("bridged_skipped_long"),
            swap_suspect_seams=(audit or {}).get("n_ambiguous_seams"),
            low_conf_team_ids=n_low,
        ),
        team_split=teams_split,
        modes=dict(
            mode1a=diag["mode1a_intra_swaps"], mode1b=diag["mode1b_encounters"],
            mode2a=diag["mode2a_intra_holes"],
            mode2b=diag.get("mode2b_player_gaps"),
            mode3_4=diag["mode3_4_far_border"]),
        honesty_notes=[
            "interpolasyonlu satirlar metre/hiz metriginden HARIC (topdown_stats kapisi).",
            "ayni-takim yumusak swap KURTARILAMAZ; raporlanir, uydurulmaz.",
            "uzak-uc team_id dusuk-guven (kucuk kutu, dusuk color_quality).",
            "MODE4 literal kadraj-kirpma bu klipte YOK (literal_border_deaths=0).",
            "esikler klip/calib-ozgu; yeni mekanda diagnose dry-run ile yeniden dogrula.",
        ],
    )
    jp = os.path.join(out_dir, f"{stem}_report.json")
    json.dump(rep, open(jp, "w"), indent=2)
    md = [
        f"# Takip-saglamligi raporu — {rep['clip']}",
        "",
        f"- satir {rep['n_rows']} | tracklet {rep['n_tracklets']} | fps {rep['fps']}",
        "",
        "## Auto-duzeltildi",
        f"- presplit bolme (crossing-coincident teleport): "
        f"{rep['auto_corrected']['presplit_splits']}",
        f"- kopru kurulan (interpolasyon) satir: {rep['auto_corrected']['bridged_rows']}",
        "",
        "## Yalniz isaretlendi (otomatik degistirilmedi)",
        f"- jitter teleport (BOLUNMEDI): "
        f"{rep['only_flagged']['jitter_teleports_not_split']}",
        f"- doldurulmayan uzun bosluk: {rep['only_flagged']['long_holes_skipped']}",
        f"- swap-suspect dikis: {rep['only_flagged']['swap_suspect_seams']}",
        f"- dusuk-guven team_id: {rep['only_flagged']['low_conf_team_ids']}",
        "",
        f"## Takim dagilimi: {teams_split}",
        "",
        "## Durustluk notlari",
    ] + [f"- {h}" for h in rep["honesty_notes"]]
    mp = os.path.join(out_dir, f"{stem}_report.md")
    open(mp, "w").write("\n".join(md) + "\n")
    return jp, mp


# ===================================================================== CLI ===
if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "diag":
        mp = sys.argv[4] if len(sys.argv) > 4 else None
        out = sys.argv[5] if len(sys.argv) > 5 else None
        print(json.dumps(diagnose(sys.argv[2], sys.argv[3], mp, out), indent=2))
    else:
        print(__doc__)
