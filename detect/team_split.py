#!/usr/bin/env python3
"""team_split — oyunculari iki takima ayir (jersey-renk PRIMARY + pas/etkilesim DOGRULAMA).

DOMAIN GERCEKLERI (Alperen, amator halisaha):
  * Sahada SADECE oyuncular var (hakem/seyirci YOK). Cift roster (12/14/16).
  * Takim ayrimi NET renk sinyalli: bir takim acik/yelek (SARI), digeri KOYU.
    Bu yuzden jersey-renk PRIMARY. Sabit "sari bandi" YOK (bu gece kaydinda
    16-2 patliyor) -> DATA-DRIVEN 2-means renk uzayinda (dengeli 9-9 verir).
  * Top yok -> possession-proxy: en sik kumelenen oyuncularin EMA-merkezi
    pseudo-top, en yakin oyuncu tasiyici; tasiyici degisimi proxy-pas. Pas
    grafigi (normalized-cut Fiedler) renk kumelerini DOGRULAR; ASLA yuksek-guven
    rengi ezmez (sadece dusuk-guven rengi duzeltir).

DURUSTLUK DEGISMEZLERI:
  * Additive: girdi df'ye SADECE team_id (int64) + team_conf (float32) eklenir;
    satir sayisi ve tid==player_id korunur -> topdown_stats/topdown_viz/stats_report
    DEGISMEDEN calisir.
  * Asla 9-9'a ZORLAMAZ; takim dengesizligi balance_note ile RAPORLANIR.
  * Bozuk calib (median>20px) fail-closed reddedilir (uydurma yok).
  * Proxy ground-truth DEGIL; sadece dusuk-guven rengi duzeltir, caveat'ta yazar.
  * stitch raporundaki residual_under_merge oyunculari dusuk-guven isaretlenir.

NON-DESTRUCTIVE: identity layer [C]; track_stitch.stitch() SONRASI calisir, omurgaya
dokunmaz (track_stitch.py + paylasilanlar DUZENLENMEZ).

Lisans: numpy + pandas + cv2 + scipy (kmeans2, linalg.eigh). Hepsi BSD/Apache.
GPU yok. AGPL yok (ultralytics/boxmot KULLANILMAZ).

TODO(dedup): torso-crop + cim/golge maskesi gecmisi su an track_stitch._appearance_descriptors
ile BURADA aynen tekrar ediyor (ayni geometri: cy1=fy-bh, cy2=fy-0.45*bh, cx=fx+-0.30*bw).
Ileride detect/jersey_color.py'ye CIKARILIP iki modul de oradan cagirmali ki geometri
kaymasi (drift) olmasin. Simdilik bilincli kopya: track_stitch.py DUZENLENMEME kisiti var.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import stats_report
from pitch.homography import PitchHomography
from stats.topdown_stats import accept_calib_qa


# --------------------------------------------------------------- jersey color --
def _player_color_features(df, video_path, max_samples=14, min_keep_px=20):
    """Her player_id icin renk-ozelligi (cos2h, sin2h, S/255, V/255) + q + disp.

    Govde kirpma geometrisi track_stitch._appearance_descriptors ile AYNEN ayni:
    cy1=fy-bh, cy2=fy-0.45*bh, cx=fx+-0.30*bw. HSV'ye cevir, cim (H35-85 & S>60 &
    V>40) ve golge (V<=25) at, >=min_keep_px piksel sart, ornek basina median HSV.

    Doner: dict pid -> dict(feat(4,), q, disp, n_samples, med_hsv(3,)).
    """
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"video acilamadi: {video_path}")

    fcol = df["frame"].to_numpy(np.int64)
    xcol = df["foot_x"].to_numpy(np.float64)
    ycol = df["foot_y"].to_numpy(np.float64)
    hcol = df["box_h"].to_numpy(np.float64)
    wcol = (df["box_w"].to_numpy(np.float64) if "box_w" in df.columns
            else hcol * 0.45)
    pcol = df["player_id"].to_numpy(np.int64)

    out = {}
    for pid in np.unique(pcol):
        sel = np.where(pcol == pid)[0]
        if sel.size == 0:
            continue
        # en buyuk box'lara yanli (en guvenilir govde pikseli)
        order = sel[np.argsort(-hcol[sel])][:max_samples]

        sample_meds = []   # ornek basina median HSV (3,)
        size_acc = 0.0
        for ridx in order:
            fr = int(fcol[ridx])
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, img = cap.read()
            if not ok or img is None:
                continue
            H, W = img.shape[:2]
            bh = hcol[ridx] if np.isfinite(hcol[ridx]) else 0
            bw = wcol[ridx] if np.isfinite(wcol[ridx]) else 0
            if bh <= 4 or bw <= 2:
                continue
            fx, fy = xcol[ridx], ycol[ridx]
            cy1 = int(max(0, fy - bh))
            cy2 = int(max(0, fy - 0.45 * bh))
            cx1 = int(max(0, fx - 0.30 * bw))
            cx2 = int(min(W, fx + 0.30 * bw))
            if cy2 <= cy1 + 1 or cx2 <= cx1 + 1:
                continue
            crop = img[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            hh = hsv[:, :, 0].astype(np.float64)
            ss = hsv[:, :, 1].astype(np.float64)
            vv = hsv[:, :, 2].astype(np.float64)
            grass = (hh >= 35) & (hh <= 85) & (ss > 60) & (vv > 40)
            shadow = vv <= 25
            keep = (~grass) & (~shadow)
            if int(keep.sum()) < min_keep_px:
                continue
            med = np.array([np.median(hh[keep]), np.median(ss[keep]),
                            np.median(vv[keep])], dtype=np.float64)
            sample_meds.append(med)
            size_acc += float(bh)

        if not sample_meds:
            out[int(pid)] = dict(feat=None, q=0.0, disp=np.nan,
                                 n_samples=0, med_hsv=None)
            continue

        S = np.asarray(sample_meds, dtype=np.float64)  # (k,3)
        # median-of-medians (golge/cim sonrasi dayanikli merkez)
        med_hsv = np.median(S, axis=0)
        h = med_hsv[0]
        # OpenCV H in [0,180); cift-aci (2h) kirmizi 0/180 sarmasini onler
        theta = h * (np.pi / 90.0)  # h/180*2pi
        feat = np.array([np.cos(2 * theta), np.sin(2 * theta),
                         med_hsv[1] / 255.0, med_hsv[2] / 255.0],
                        dtype=np.float64)
        # disp = ornekler-arasi HSV yayilim (yuksek -> bimodal renk -> ID-swap riski)
        # H'yi cift-aci vektorde olc (sarma-bagimsiz), S/V'yi normalize.
        th_s = S[:, 0] * (np.pi / 90.0)
        hv = np.stack([np.cos(2 * th_s), np.sin(2 * th_s)], axis=1)
        comp = np.concatenate([hv, S[:, 1:3] / 255.0], axis=1)
        disp = float(np.mean(np.std(comp, axis=0))) if len(comp) > 1 else 0.0
        size_term = min(1.0, (size_acc / len(sample_meds)) / 120.0)
        cover_term = len(sample_meds) / float(max_samples)
        q = float(np.clip(size_term * cover_term, 0.0, 1.0))
        out[int(pid)] = dict(feat=feat, q=q, disp=disp,
                             n_samples=len(sample_meds), med_hsv=med_hsv)

    cap.release()
    return out


def _kmeans2_teams(feat_mat):
    """Standardize -> kmeans2 k=2, 8 seed, en dusuk inertia. Doner labels, centroids,
    d_self, d_other, degenerate(bool) (standardize uzayinda)."""
    from scipy.cluster.vq import kmeans2
    X = feat_mat.astype(np.float64)
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Z = (X - mu) / sd

    best = None
    for s in range(8):
        try:
            cent, lab = kmeans2(Z, 2, iter=50, minit="++", seed=s)
        except TypeError:
            np.random.seed(s)
            cent, lab = kmeans2(Z, 2, iter=50, minit="++")
        if len(np.unique(lab)) < 2:
            continue
        inertia = float(np.sum((Z - cent[lab]) ** 2))
        if best is None or inertia < best[0]:
            best = (inertia, cent, lab)

    if best is None:
        # degenerate: tek kume; herkes label 0, mesafeler esit
        d0 = np.linalg.norm(Z - Z.mean(axis=0), axis=1)
        return (np.zeros(len(Z), dtype=int), np.stack([Z.mean(0), Z.mean(0)]),
                d0, d0, True)

    _, cent, lab = best
    d0 = np.linalg.norm(Z - cent[0], axis=1)
    d1 = np.linalg.norm(Z - cent[1], axis=1)
    d_self = np.where(lab == 0, d0, d1)
    d_other = np.where(lab == 0, d1, d0)
    return lab, cent, d_self, d_other, False


# ------------------------------------------------------- possession-proxy graph --
def _proxy_pass_graph(df, homo, pid_list, fps, hz=5.0, pass_gap_s=2.0):
    """Top-suz possession-proxy pas matrisi P (simetrik) + carrier dizisi.

    ~hz Hz'de in_pitch & sonlu pitch_x satirlari; pseudo-top = en sik kumelenen
    4 oyuncunun EMA-merkezi; carrier = en yakin oyuncu; <=pass_gap_s icinde carrier
    degisimi = proxy-pas. Doner P (n,n) + total_pass.
    """
    pid_index = {int(p): i for i, p in enumerate(pid_list)}
    n = len(pid_list)
    P = np.zeros((n, n), dtype=np.float64)

    mask = df["in_pitch"].to_numpy(dtype=bool)
    px = df["pitch_x"].to_numpy(np.float64)
    py = df["pitch_y"].to_numpy(np.float64)
    # finite-frac < 0.5 ise yeniden re-projeksiyon (burada 1.0, ama genel)
    finite_frac = float(np.mean(np.isfinite(px)))
    if finite_frac < 0.5:
        foot = df[["foot_x", "foot_y"]].to_numpy(np.float64)
        pm = homo.pixel_to_pitch(foot)
        px, py = pm[:, 0], pm[:, 1]
    good = mask & np.isfinite(px) & np.isfinite(py)

    fcol = df["frame"].to_numpy(np.int64)
    pcol = df["player_id"].to_numpy(np.int64)

    stride = max(1, int(round(fps / hz)))
    frames = np.unique(fcol[good])
    frames = frames[::stride]

    ema = None
    alpha = 0.4
    last_carrier = None
    last_t = None

    for fr in frames:
        rows = np.where(good & (fcol == fr))[0]
        if rows.size < 2:
            continue
        pts = np.stack([px[rows], py[rows]], axis=1)
        pids = pcol[rows]
        # en sik kumelenen 4 oyuncu: en kucuk 3-NN mesafe toplami
        k = min(4, len(pts))
        D = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=2)
        np.fill_diagonal(D, np.inf)
        nn3 = np.sort(D, axis=1)[:, :min(3, len(pts) - 1)].sum(axis=1)
        dense = np.argsort(nn3)[:k]
        ball = pts[dense].mean(axis=0)
        ema = ball if ema is None else alpha * ball + (1 - alpha) * ema
        # carrier = pseudo-topa en yakin oyuncu
        d_ball = np.linalg.norm(pts - ema, axis=1)
        carrier_pid = int(pids[int(np.argmin(d_ball))])
        t_now = float(fr) / fps if fps > 0 else float(fr)

        if (last_carrier is not None and carrier_pid != last_carrier
                and (t_now - last_t) <= pass_gap_s):
            a = pid_index.get(last_carrier)
            b = pid_index.get(carrier_pid)
            if a is not None and b is not None:
                P[a, b] += 1.0
                P[b, a] += 1.0
        last_carrier = carrier_pid
        last_t = t_now

    total_pass = float(P.sum() / 2.0)
    return P, total_pass


def _fiedler_cut(P):
    """Normalized Laplacian Fiedler-vektor isareti -> 2 topluluk (per player)."""
    from scipy.linalg import eigh
    n = P.shape[0]
    deg = P.sum(axis=1)
    if np.all(deg == 0):
        return np.zeros(n, dtype=int)
    d_inv_sqrt = np.zeros(n)
    nz = deg > 0
    d_inv_sqrt[nz] = 1.0 / np.sqrt(deg[nz])
    Dm = np.diag(d_inv_sqrt)
    L = np.eye(n) - Dm @ P @ Dm
    L = 0.5 * (L + L.T)
    try:
        w, v = eigh(L)
    except Exception:
        return np.zeros(n, dtype=int)
    order = np.argsort(w)
    fiedler = v[:, order[1]] if n > 1 else v[:, 0]
    return (fiedler >= 0).astype(int)


def _align(labels_a, labels_b):
    """labels_b'yi labels_a'ya en iyi 2-permutasyonla hizala. Doner (aligned_b, agreement)."""
    a = np.asarray(labels_a)
    b = np.asarray(labels_b)
    agree_id = float(np.mean(a == b))
    agree_sw = float(np.mean(a == (1 - b)))
    if agree_sw > agree_id:
        return (1 - b), agree_sw
    return b, agree_id


# ------------------------------------------------------------------ ana akis ---
def assign_teams(player_tracks_path, calib_path, out_dir, video_path,
                 min_conf=0.35, outlier_thr=2.0, swap_disp_thr=0.18,
                 agree_disable=0.55) -> dict:
    """Stitched player-keyed parquet -> per-oyuncu team_id (0/1) + guven.

    Bkz. modul docstring'i. Renk PRIMARY (data-driven 2-means), pas-grafigi yalnizca
    dusuk-guven rengi DUZELTIR. Additive cikti: {base}_team.parquet + iki JSON.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) yukle + STITCHED dogrula
    df = stats_report.load_tracks(player_tracks_path)
    if "player_id" not in df.columns:
        raise ValueError(
            "girdi STITCHED degil; once track_stitch.stitch() (player_id kolonu yok)")

    fps = (float(df["frame"].max()) / float(df["t_sec"].max())
           if float(df["t_sec"].max()) > 0 else 25.0)

    # 2) calib QA kapisi — FAIL-CLOSED
    homo = PitchHomography.load(calib_path)
    ok, reasons = accept_calib_qa(homo._qa)
    if not ok:
        raise ValueError(
            "calib QA reddedildi (fail-closed): " + "; ".join(reasons))

    # stitch raporundan residual_under_merge oyuncularini + clique_floor oku (varsa)
    residual_pids = set()
    clique_floor = None
    base = Path(player_tracks_path).stem
    stitch_base = base[:-7] if base.endswith("_player") else base
    src_dir = os.path.dirname(os.path.abspath(player_tracks_path))
    for cand in (os.path.join(src_dir, f"{stitch_base}_stitch_report.json"),
                 os.path.join(out_dir, f"{stitch_base}_stitch_report.json")):
        if os.path.exists(cand):
            try:
                sr = json.load(open(cand))
                for r in sr.get("residual_under_merge", []):
                    residual_pids.add(int(r["player_id"]))
                if sr.get("clique_floor") is not None:
                    clique_floor = int(sr["clique_floor"])
            except Exception:
                pass
            break

    pid_list = sorted(int(p) for p in df["player_id"].unique())
    n = len(pid_list)
    pid_pos = {p: i for i, p in enumerate(pid_list)}

    # 3) PRIMARY: jersey renk
    colinfo = _player_color_features(df, video_path)
    feat_mat = np.zeros((n, 4), dtype=np.float64)
    q_arr = np.zeros(n)
    disp_arr = np.zeros(n)
    has_feat = np.zeros(n, dtype=bool)
    nsamp = np.zeros(n, dtype=int)
    for p, i in pid_pos.items():
        ci = colinfo.get(p, dict(feat=None, q=0.0, disp=np.nan, n_samples=0))
        nsamp[i] = ci["n_samples"]
        q_arr[i] = ci["q"]
        disp_arr[i] = ci["disp"] if np.isfinite(ci["disp"]) else 0.0
        if ci["feat"] is not None:
            feat_mat[i] = ci["feat"]
            has_feat[i] = True
    # ozelliksiz oyunculari sutun-ortalamasiyla imputla (atama garantisi; q=0->dusuk guven)
    if has_feat.any():
        col_mean = feat_mat[has_feat].mean(axis=0)
        feat_mat[~has_feat] = col_mean

    labels, cent, d_self, d_other, degenerate = _kmeans2_teams(feat_mat)

    # kararli isimlendirme: yuksek-ortalama-V kume = team_id 1 ("parlak/yelek")
    method_primary = "jersey_color_kmeans2_data_driven"
    if degenerate:
        color_team = np.zeros(n, dtype=int)  # tek renk; herkes belirsiz
        margin = np.zeros(n)
        outlier = np.ones(n)
    else:
        v_by_lab = [feat_mat[labels == c, 3].mean() for c in (0, 1)]
        bright = int(np.argmax(v_by_lab))
        color_team = np.where(labels == bright, 1, 0).astype(int)
        denom = (d_other + d_self)
        denom[denom < 1e-9] = 1e-9
        margin = (d_other - d_self) / denom
        med_self = np.median(d_self[d_self > 0]) if np.any(d_self > 0) else 1.0
        outlier = d_self / (med_self if med_self > 0 else 1.0)

    # 4) SECONDARY: possession-proxy grafik
    P, total_pass = _proxy_pass_graph(df, homo, pid_list, fps)
    graph_labels = _fiedler_cut(P)
    aligned_graph, agreement = _align(color_team, graph_labels)

    # proxy_pass_intra_rate: renk-takimina gore takim-ici pas orani
    intra = 0.0
    if total_pass > 0:
        same = color_team[:, None] == color_team[None, :]
        intra = float(P[same].sum() / 2.0 / total_pass)

    graph_usable = (intra > agree_disable) and (total_pass >= 30)

    # 5) FUSION (renk primary; grafik yalniz dusuk-guven duzeltir)
    final_team = color_team.copy()
    conf = np.clip(0.5 * margin + 0.5 * q_arr, 0.0, 1.0)
    swap_flags, gk_cands, corrected, flagged_low = [], [], [], []

    for i, p in enumerate(pid_list):
        c = float(conf[i])
        if disp_arr[i] > swap_disp_thr:
            c *= 0.5
            swap_flags.append(p)
        if outlier[i] > outlier_thr:
            c *= 0.6
            gk_cands.append(p)
        # grafik DUZELTME: yalniz usable + uyusmazlik + dusuk renk-guveni
        if (graph_usable and aligned_graph[i] != color_team[i]
                and float(conf[i]) < min_conf):
            final_team[i] = int(aligned_graph[i])
            c = max(c, 0.45)
            corrected.append(p)
        # residual under-merge: dusuk-guven isaretle
        if p in residual_pids or not has_feat[i]:
            c *= 0.5
            if p not in flagged_low:
                flagged_low.append(p)
        if c < min_conf and p not in flagged_low:
            flagged_low.append(p)
        conf[i] = c

    # 6) takim boyutlari + denge raporu (ASLA 9-9'a zorlanmaz)
    sizes = {0: int(np.sum(final_team == 0)), 1: int(np.sum(final_team == 1))}
    imbalance = abs(sizes[0] - sizes[1])
    balance_ok = imbalance <= 2
    balance_note = (None if balance_ok else
                    f"takim dengesizligi |{sizes[0]}-{sizes[1]}|={imbalance}>2 "
                    "(renk-data-driven; 9-9'a ZORLANMADI, raporlandi)")

    color_sep_conf = float(np.mean(margin[has_feat])) if has_feat.any() else 0.0

    per_player = []
    for i, p in enumerate(pid_list):
        per_player.append(dict(
            player_id=int(p), team_id=int(final_team[i]),
            team_conf=round(float(conf[i]), 4),
            color_team=int(color_team[i]),
            graph_team=int(aligned_graph[i]),
            margin=round(float(margin[i]), 4),
            color_disp=round(float(disp_arr[i]), 4),
            outlier=round(float(outlier[i]), 4),
            color_q=round(float(q_arr[i]), 4),
            n_color_samples=int(nsamp[i]),
            corrected=bool(p in corrected),
            swap_flag=bool(p in swap_flags),
            gk_candidate=bool(p in gk_cands),
            low_conf=bool(p in flagged_low),
        ))

    report = dict(
        method_primary=method_primary,
        method_secondary="possession_proxy_normalized_cut_fiedler",
        n_players=n,
        team_sizes=sizes,
        clique_floor=clique_floor,
        balance_ok=bool(balance_ok),
        balance_note=balance_note,
        color_separation_confidence=round(color_sep_conf, 4),
        color_interaction_agreement=round(float(agreement), 4),
        proxy_pass_intra_rate=round(float(intra), 4),
        proxy_pass_total=int(total_pass),
        graph_used=bool(graph_usable),
        corrected_players=[int(p) for p in corrected],
        flagged_low_conf=[int(p) for p in flagged_low],
        gk_candidates=[int(p) for p in gk_cands],
        swap_flags=[int(p) for p in swap_flags],
        degenerate_color=bool(degenerate),
        fps=round(fps, 4),
        per_player=per_player,
        caveats=[
            "additive: girdiye sadece team_id+team_conf eklendi; tuketiciler etkilenmedi.",
            "data-driven 2-means renk (sabit sari-bandi DEGIL; bu kayitta 16-2 patlardi).",
            "possession-proxy ground-truth pas DEGIL; sadece dusuk-guven rengi duzeltir.",
            "takim dengesizligi balance_note'ta raporlanir, 9-9'a ZORLANMAZ.",
            "stitch residual_under_merge + renk-ornegisiz oyuncular dusuk-guven isaretli.",
        ],
    )

    # 7) ciktilar (additive; satir sayisi + tid==player_id korunur)
    out_df = df.copy()
    team_map = {int(p): int(final_team[i]) for i, p in enumerate(pid_list)}
    conf_map = {int(p): float(conf[i]) for i, p in enumerate(pid_list)}
    out_df["team_id"] = out_df["player_id"].map(team_map).astype("int64")
    out_df["team_conf"] = out_df["player_id"].map(conf_map).astype("float32")

    team_parquet_path = os.path.join(out_dir, f"{base}_team.parquet")
    out_df.to_parquet(team_parquet_path, index=False)

    mapping = {int(p): int(final_team[i]) for i, p in enumerate(pid_list)}
    mapping_path = os.path.join(out_dir, f"{base}_pid2team.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)

    report_path = os.path.join(out_dir, f"{base}_team_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    conf_out = {int(p): round(float(conf[i]), 4) for i, p in enumerate(pid_list)}
    return dict(report=report, mapping=mapping, conf=conf_out,
                team_parquet_path=team_parquet_path, report_path=report_path,
                mapping_path=mapping_path)


if __name__ == "__main__":
    res = assign_teams(
        "stats_out/pipeline_demo/tracks_cankaya_cam2_clip2400_player.parquet",
        "calib/cankaya_cam2_v2.json",
        "stats_out/team_demo",
        "raw/cankaya_cam2_clip2400.mp4")
    import pprint
    pprint.pprint(res["report"])
