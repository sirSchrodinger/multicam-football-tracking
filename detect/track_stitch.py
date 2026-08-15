#!/usr/bin/env python3
"""track_stitch — fragment tracklet'leri oyuncu kumelerine birlestir (statik kamera).

PROBLEM: export_tracks 146 parcali tracklet uretti ama saha ~14 oyuncu. Bu modul
tracklet'leri YALNIZCA zaman-bosluklari boyunca birlestirir, ASLA es-zamanli
(concurrent) iki tracklet'i ayni oyuncuya katmaz. Cikti: tid->player_id eslemesi
+ player_id kolonu eklenmis stitched parquet (sifir satir kaybi) + durust JSON rapor.

TASARIM GERCEKLERI (statik kamera bedava sinyalleri):
  * Kamera SABIT -> saha-metresi domeninde hiz-kapisi anlamli (broadcast'te yok).
  * Birlestirme YONLU: yalnizca A biter, sonra B baslar (forward-only).
  * Es-zamanlilik = cannot-link: iki tracklet ayni frame'de gorunuyorsa ASLA
    ayni oyuncu olamaz (over-merge'a karsi yapisal koruma).
  * clique_floor = herhangi bir frame'deki es-zamanli tracklet sayisinin MAKSIMUMU
    = oyuncu sayisinin SERT alt siniri. Kume sayisi bunun altina ASLA inmez.

DURUSTLUK DEGISMEZLERI:
  * Mesafe relative_m'de kalir (scale_anchor=null). metre/km-h/accel/sprint YOK.
  * Asla 14'e zorlamaz; clique_floor'un altina inmez (mesafe yalani uretmez).
  * Under-merge (uzak-uc >max_gap parcalari) residual_under_merge'de RAPORLANIR,
    gizlenmez. Over-merge cannot-link + atama-sonrasi yeniden-kontrol ile imkansiz.
  * Appearance (renk) yalnizca veto/yeniden-siralama yapar; ASLA kenar URETMEZ
    (olculen: intra 0.59 vs inter 0.58 ~ sans seviyesi, bu yuzden demote edildi).
  * Bozuk kalibrasyon (65px) fail-closed reddedilir; uydurma yok.

Lisans: numpy + scipy.optimize + pandas + cv2 (hepsi BSD/MIT/Apache). GPU yok.
AGPL yok (ultralytics/boxmot/PnLCalib KULLANILMAZ).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# omurga modulleri (paylasilan, DUZENLENMEDI)
import stats_report
from pitch.homography import PitchHomography
from stats.topdown_stats import accept_calib_qa
# roster yardimcilari (saf, I/O yok)
from detect import roster as _roster

_BIG = 1.0e9  # yasak kenar maliyeti (scipy.inf yerine sonlu; QA-temiz)


# ---------------------------------------------------------------- union-find --
class _DSU:
    """Cannot-link yeniden-kontrollu union-find (transitive over-merge koruma).

    Her kok icin frame-kumesi tutulur; merge(a,b) birlesik frame-kumeleri
    kesisiyorsa REDDEDILIR (yaklasim 3: atama-sonrasi transitive cannot-link).
    """

    def __init__(self, frame_sets: list[frozenset]):
        self.parent = list(range(len(frame_sets)))
        self.rank = [0] * len(frame_sets)
        self.frames = [set(fs) for fs in frame_sets]  # kok-bazli frame kumesi

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return True
        # transitive cannot-link: birlesik frame-kumeleri kesisiyorsa reddet
        sa, sb = self.frames[ra], self.frames[rb]
        small, big = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
        if not small.isdisjoint(big):
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.frames[ra].update(self.frames[rb])
        self.frames[rb] = set()
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


# --------------------------------------------------------- tracklet ozetleri --
def _summaries(df, homo, k_end: int):
    """Her tracklet icin uc-ozetleri (topdown_stats.prepare deseni).

    Doner: list[dict] tid sirasinda; her biri:
      tid, n, f0, f1, t0, t1, head(med ilk k_end), tail(med son k_end),
      frame_set(frozenset), far_frac (uzak-ucte gozlem orani).
    + global far esigi (piksel foot_y'nin alt-uctebiri = uzak uc).
    """
    foot = df[["foot_x", "foot_y"]].to_numpy(np.float64)
    # TEK vektorize re-projeksiyon (diskteki pitch_x/pitch_y %100 NaN)
    pm = homo.pixel_to_pitch(foot)  # relative_m

    foot_y = foot[:, 1]
    finite = foot_y[np.isfinite(foot_y)]
    # uzak uc = goruntu ust-uctebiri = en kucuk foot_y (kameradan uzak, piksel-fakiri)
    far_thr = float(np.quantile(finite, 1.0 / 3.0)) if finite.size >= 3 else -np.inf

    tid_arr = df["tid"].to_numpy(np.int64)
    frame_arr = df["frame"].to_numpy(np.int64)
    t_arr = df["t_sec"].to_numpy(np.float64)

    order = np.lexsort((frame_arr, tid_arr))
    tid_s = tid_arr[order]
    uniq, starts, counts = np.unique(tid_s, return_index=True, return_counts=True)

    summaries = []
    for i in range(len(uniq)):
        s = int(starts[i]); c = int(counts[i])
        idx = order[s:s + c]
        # frame'e gore sirali (lexsort zaten frame ikincil)
        pm_t = pm[idx]
        fr = frame_arr[idx]
        tt = t_arr[idx]
        fy = foot_y[idx]
        k = min(int(k_end), c)
        head = np.median(pm_t[:k], axis=0)   # ilk k_end gozlem (baslangic konumu)
        tail = np.median(pm_t[-k:], axis=0)  # son k_end gozlem (bitis konumu)
        far_frac = float(np.mean(fy < far_thr)) if c else 0.0
        summaries.append(dict(
            tid=int(uniq[i]), n=c,
            f0=int(fr.min()), f1=int(fr.max()),
            t0=float(tt.min()), t1=float(tt.max()),
            head=head, tail=tail,
            frame_set=frozenset(int(x) for x in fr),
            far_frac=far_frac,
        ))
    return summaries, far_thr


def _clique_floor(summaries) -> int:
    """Herhangi bir frame'deki maksimum es-zamanli tracklet sayisi (SERT alt sinir).

    Sweep-line: her tracklet [f0, f1] araliginda +1/-1; tepe degeri = floor.
    """
    events = []
    for s in summaries:
        events.append((s["f0"], 1))
        events.append((s["f1"] + 1, -1))
    events.sort()
    cur = peak = 0
    for _, d in events:
        cur += d
        if cur > peak:
            peak = cur
    return int(peak)


# ----------------------------------------------------------------- ana akis ---
def stitch(tracks_path: str, calib_path: str, out_dir: str,
           video_path: str | None = None,
           max_speed_mps: float = 8.0, max_gap_s: float = 8.0,
           far_relax: float = 2.0, w_appear: float = 1.5,
           gap_pen: float = 0.15, no_link: float = 0.9, k_end: int = 5,
           n_eph: int = 4, r_dd_m: float = 2.0, dd_frac: float = 0.6,
           off_inp: float = 0.5, static_min_n: int = 40,
           static_diag_m: float = 1.5, static_out_m: float = 0.5,
           cluster_max_gap_s: float = 30.0,
           expected_players: int | None = None) -> dict:
    """146 parcali tracklet -> oyuncu kumeleri. Tek-giris API.

    Bkz. modul docstring'i. Doner: dict(report, mapping, stitched, stitched_path,
    mapping_path, report_path).
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1) yukle (df.attrs['meta'] korunur)
    df = stats_report.load_tracks(tracks_path)
    fps = (float(df["frame"].max()) / float(df["t_sec"].max())
           if float(df["t_sec"].max()) > 0 else 25.0)

    # 2) calib QA kapisi — FAIL-CLOSED (65px reddedilir; uydurma yok)
    homo = PitchHomography.load(calib_path)
    ok, reasons = accept_calib_qa(homo._qa)
    if not ok:
        raise ValueError(
            "calib QA reddedildi (fail-closed, uydurma yok): " + "; ".join(reasons))

    # 4) tracklet uc-ozetleri (3. re-projeksiyon adimi _summaries icinde)
    summaries_all, far_thr = _summaries(df, homo, k_end)

    # 4b) pm BIR KEZ (flag_spurious + concurrency dedup + roster). df'e pitch ekle.
    pm = homo.pixel_to_pitch(df[["foot_x", "foot_y"]].to_numpy(np.float64))
    df["pitch_x"] = pm[:, 0]
    df["pitch_y"] = pm[:, 1]

    # 4c) hayalet/oyuncu-disi tespitleri KONSERVATIF cok-kanitli isaretle.
    #     Hakem/seyirci YOK; hedef cift roster (12/14/16). Etiketlenir, SILINMEZ.
    spurious = _roster.flag_spurious(
        df, summaries_all, pm, homo,
        n_eph=n_eph, r_dd_m=r_dd_m, dd_frac=dd_frac, off_inp=off_inp,
        static_min_n=static_min_n, static_diag_m=static_diag_m,
        static_out_m=static_out_m, max_speed_mps=max_speed_mps, max_gap_s=max_gap_s)
    spurious_set = set(spurious.keys())
    spurious_lifetimes = {s["tid"]: s["n"] for s in summaries_all}
    df_kept = df[~df["tid"].isin(spurious_set)]

    # YALNIZCA hayatta-kalanlar uzerinde stitch (cost/LSAP/DSU N=len(kept))
    summaries = [s for s in summaries_all if s["tid"] not in spurious_set]
    N = len(summaries)

    # 5) clique_floor — SURVIVOR'lar uzerinden DURUST sert alt sinir (15)
    clique_floor = _roster.detection_floor(df_kept)
    # diagnostic concurrency (raw max=16 hayalet-sismis; floor DEGIL) + roster
    conc = _roster.concurrency_stats(df)
    roster_info = _roster.roster_estimate(df_kept, homo, expected_players)

    # 10) appearance descriptor (yalnizca video verildiyse; demote: veto/tie-break)
    appdesc = [None] * N
    appq = np.zeros(N, dtype=np.float64)
    appearance_used_frac = 0.0
    if video_path is not None and os.path.exists(video_path):
        try:
            appdesc, appq = _appearance_descriptors(df, summaries, video_path)
            appearance_used_frac = float(np.mean(appq > 0))
        except Exception:
            appdesc = [None] * N
            appq = np.zeros(N, dtype=np.float64)

    # 6) maliyet matrisi C[i,j] (i'nin tail'i -> j'nin head'i), SERT kapilar
    C = np.full((N, N), _BIG, dtype=np.float64)
    for i in range(N):
        Ai = summaries[i]
        ti_far = Ai["far_frac"] > 0.5
        for j in range(N):
            if i == j:
                continue
            Bj = summaries[j]
            # forward-only: A biter, sonra B baslar
            if not (Ai["t1"] < Bj["t0"]):
                continue
            # cannot-link: ayni frame'de gorunuyorlarsa ASLA birlesemez
            if not Ai["frame_set"].isdisjoint(Bj["frame_set"]):
                continue
            dt = Bj["t0"] - Ai["t1"]
            if not (0.0 < dt <= max_gap_s):
                continue
            # motion gate (metre-domeninde hiz); uzak-ucte cap gevsetilir (icgoru #3)
            link_far = ti_far or (Bj["far_frac"] > 0.5)
            far_term = (far_relax - 1.0) if link_far else 0.0
            cap = max_speed_mps * (1.0 + far_term)
            if not homo.stitch_motion_gate_m(Ai["tail"], Bj["head"], dt, cap):
                continue
            # appearance cross-team veto (yalnizca her iki taraf yuksek-kalite+marjin)
            adist = 0.0
            if appdesc[i] is not None and appdesc[j] is not None:
                adist = _hist_dist(appdesc[i], appdesc[j])
                if (appq[i] > 0.5 and appq[j] > 0.5 and adist > 0.85):
                    continue  # cross-team veto (konservatif)
            # SOFT maliyet (appearance ASLA kenar uretmez, yalniz yeniden-siralar)
            speed = float(np.linalg.norm(Bj["head"] - Ai["tail"]) / dt)
            cost = (speed / max_speed_mps
                    + gap_pen * (dt / max_gap_s)
                    + w_appear * adist * min(appq[i], appq[j]))
            C[i, j] = cost

    # 7) GLOBAL one-to-one path-cover (2N x 2N, no_link kosegende). GREEDY DEGIL.
    from scipy.optimize import linear_sum_assignment
    M = np.full((2 * N, 2 * N), _BIG, dtype=np.float64)
    M[:N, :N] = C
    # tail i eslesmeyebilir -> sag-ust kosegen no_link
    for i in range(N):
        M[i, N + i] = no_link
    # head j eslesmeyebilir -> sol-alt kosegen no_link
    for j in range(N):
        M[N + j, j] = no_link
    # dummy-dummy blok sifir
    M[N:, N:] = 0.0
    rows, cols = linear_sum_assignment(M)

    # kabul edilen kenarlar: tail i -> head j (j<N ve maliyet sonlu)
    links = []
    for r, cc in zip(rows, cols):
        if r < N and cc < N and C[r, cc] < _BIG / 2:
            links.append((float(C[r, cc]), int(r), int(cc)))
    links.sort()  # maliyet artan; en-iyi kenar once birlestirilir

    # 8) union-find + transitive cannot-link yeniden-kontrol
    frame_sets = [summaries[i]["frame_set"] for i in range(N)]
    dsu = _DSU(frame_sets)
    n_links = 0
    for _cost, i, j in links:
        if dsu.union(i, j):
            n_links += 1

    # kume kimlikleri (DSU sonrasi)
    roots = [dsu.find(i) for i in range(N)]
    uniq_roots = sorted(set(roots))
    obs = np.array([summaries[i]["n"] for i in range(N)], dtype=np.int64)
    dsu_members: dict = {}
    for i, r in enumerate(roots):
        dsu_members.setdefault(r, []).append(i)

    # 8b) cluster_merge_pass — Design 2 frame-disjoint under-merge bosluk-doldurma
    #     (cluster_max_gap_s>pairwise gap; uzak-uc/uzun-bosluk parcalarini birlestir).
    #     Disjoint birlesim max-es-zamanli'yi artiramaz -> over-merge imkansiz.
    cl = []
    for r in uniq_roots:
        members = dsu_members[r]
        fs: set = set()
        for i in members:
            fs |= summaries[i]["frame_set"]
        t0 = min(summaries[i]["t0"] for i in members)
        t1 = max(summaries[i]["t1"] for i in members)
        head_i = min(members, key=lambda i: summaries[i]["f0"])
        tail_i = max(members, key=lambda i: summaries[i]["f1"])
        cl.append(dict(tids=list(members), frame_set=fs, t0=t0, t1=t1,
                       head=summaries[head_i]["head"], tail=summaries[tail_i]["tail"]))
    n_cluster_merges = _roster.cluster_merge_pass(
        cl, fps, cluster_max_gap_s=cluster_max_gap_s, cap=max_speed_mps)
    n_clusters = len(cl)

    # 9) ASSERT: over-merge koruma (merge sonrasi kume-ici frame-cakismasi)
    temporal_violations = 0
    for c in cl:
        seen: set = set()
        bad = False
        for i in c["tids"]:
            fsi = summaries[i]["frame_set"]
            if not seen.isdisjoint(fsi):
                bad = True
            seen |= fsi
        if bad:
            temporal_violations += 1
    assert temporal_violations == 0, "OVER-MERGE: kume-ici zaman cakismasi"
    assert n_clusters >= clique_floor, (
        f"floor ihlali: {n_clusters} < clique_floor {clique_floor}")

    # player_id: toplam gozleme gore AZALAN sirayla (rol/takim etiketi YOK)
    cl_obs = []
    for idx, c in enumerate(cl):
        members = c["tids"]
        tot = int(obs[members].sum())
        cl_obs.append((tot, idx, members))
    cl_obs.sort(key=lambda x: (-x[0], x[1]))

    tid_to_pid = {}
    cluster_sizes = []
    cluster_timelines = []
    for pid, (tot, idx, members) in enumerate(cl_obs):
        for i in members:
            tid_to_pid[summaries[i]["tid"]] = pid
        cluster_sizes.append(tot)
        f0 = min(summaries[i]["f0"] for i in members)
        f1 = max(summaries[i]["f1"] for i in members)
        span = max(1, f1 - f0 + 1)
        covered = sum(len(summaries[i]["frame_set"]) for i in members)
        cluster_timelines.append(dict(
            player_id=pid, n_tracklets=len(members), obs=tot,
            f0=int(f0), f1=int(f1),
            timeline_coverage=round(float(covered) / span, 4)))

    total_obs = int(obs.sum())
    top16 = sorted(cluster_sizes, reverse=True)[:clique_floor]
    top16_obs_coverage = round(float(sum(top16)) / total_obs, 4) if total_obs else 0.0

    # residual_under_merge: floor uzerindeki fazla parcalar (under-merge, GIZLENMEZ)
    residual = []
    for pid, (tot, idx, members) in enumerate(cl_obs):
        if pid >= clique_floor:
            residual.append(dict(
                player_id=pid, obs=tot, n_tracklets=len(members),
                tids=[summaries[i]["tid"] for i in members],
                far_frac=round(float(np.mean([summaries[i]["far_frac"]
                                              for i in members])), 3)))

    # 11) ciktilar (yikici-degil): stitched parquet = df + player_id + pitch (self-contained)
    #     SIFIR satir kaybi: st = TAM df; hayalet tid'ler player_id=-1 + reason ile
    #     ISARETLENIR (silinmez). Tuketiciler player_id>=0 filtreler.
    st = df.copy()
    st["player_id"] = st["tid"].map(tid_to_pid).fillna(-1).astype("int64")
    st["spurious_reason"] = st["tid"].map(spurious).fillna("").astype(str)
    # pitch'i BURADA yaz (disk export'u NaN birakmis olabilir; QA'dan gecmis v2 homo
    # ile tek vektorize re-projeksiyon) -> stitched parquet tek-basina tuketilebilir.
    _foot = st[["foot_x", "foot_y"]].to_numpy(dtype=np.float64)
    _pm = homo.pixel_to_pitch(_foot)
    st["pitch_x"] = _pm[:, 0].astype("float32")
    st["pitch_y"] = _pm[:, 1].astype("float32")
    _inp = np.asarray(homo.in_pitch(_pm), dtype=bool)
    if "bottom_cropped" in st.columns:
        _inp = _inp & (~st["bottom_cropped"].to_numpy(dtype=bool))
    st["in_pitch"] = _inp
    base = Path(tracks_path).stem
    stitched_path = os.path.join(out_dir, f"{base}_stitched.parquet")
    st.to_parquet(stitched_path, index=False)

    # 11b) PLAYER-KEYED gorunum: tid := player_id (frag_tid = orijinal tracker tid).
    #      Boylece per-tid gruplayan tuketiciler (stats_report / topdown_stats / topdown_viz)
    #      KOD DEGISMEDEN per-OYUNCU gruplar. frag_tid kolonu 'stitched' sinyalidir.
    #      Hayaletler (player_id=-1) HARIC tutulur -> downstream per-oyuncu stats temiz.
    pk = st[st["player_id"] >= 0].copy()
    pk["frag_tid"] = pk["tid"]
    pk["tid"] = pk["player_id"].astype("int64")
    player_keyed_path = os.path.join(out_dir, f"{base}_player.parquet")
    pk.to_parquet(player_keyed_path, index=False)

    mapping = {int(k): int(v) for k, v in tid_to_pid.items()}
    mapping_path = os.path.join(out_dir, f"{base}_tid2player.json")
    with open(mapping_path, "w") as f:
        json.dump(mapping, f, indent=2)

    # hayalet ozeti (her biri insan-denetimi icin reason ile loglandi)
    from collections import Counter
    n_spurious_obs = int(df["tid"].isin(spurious_set).sum())
    spurious_list = [dict(tid=int(t), reason=spurious[t],
                          n=int(spurious_lifetimes.get(t, 0)))
                     for t in sorted(spurious)]
    spurious_by_reason = dict(Counter(spurious.values()))

    # core (tum-mac oyuncu) vs partial (yedek/dusuk-recall) -- asiri-kume 'hayalet' DEGIL,
    # GERCEK kisiler + recall acigi; 14'e zorlama yok (derin co-location analizi: min 4.5m).
    presence = _roster.classify_presence(st, int(df["frame"].max()) + 1)

    report = dict(
        n_tracklets=N,
        n_distinct_identities=n_clusters,
        n_clusters=n_clusters,
        n_core_players=presence["n_core"],
        n_partial_players=presence["n_partial"],
        presence=presence,
        clique_floor=clique_floor,
        clique_floor_basis="detection_same_frame_kept",
        concurrency_raw_max=conc["max"],
        concurrency_raw_max_note="ghost-inflated diagnostic, NOT the floor",
        concurrency_p95=conc["p95"],
        concurrency_median=conc["median"],
        concurrency_mode=conc["mode"],
        dedup_concurrency_p99=conc["dedup_p99"],
        roster=roster_info,
        on_field_vs_distinct_note=(
            "clique_floor=es-zamanli sahadaki oyuncu (survivors=%d); "
            "roster_on_field=%d HEADLINE; n_distinct=%d yedek rotasyonuyla daha "
            "fazla olabilir -- ucu de ayri/durust raporlanir, 14'e ZORLAMA yok."
            % (clique_floor, roster_info["roster_on_field"], n_clusters)),
        n_spurious=len(spurious),
        n_spurious_obs=n_spurious_obs,
        obs_lost_frac=round(n_spurious_obs / total_obs, 6) if total_obs else 0.0,
        spurious=spurious_list,
        spurious_by_reason=spurious_by_reason,
        n_cluster_merges=n_cluster_merges,
        temporal_violations=temporal_violations,
        over_merge_violations=temporal_violations,
        n_links=n_links,
        cluster_sizes=sorted(cluster_sizes, reverse=True),
        cluster_timelines=cluster_timelines,
        top16_obs_coverage=top16_obs_coverage,
        appearance_used_frac=round(appearance_used_frac, 4),
        residual_under_merge=residual,
        n_residual_fragments=len(residual),
        total_obs=total_obs,
        fps=round(fps, 4),
        unit="relative_m",
        scale_anchor=homo.scale_anchor,
        far_threshold_foot_y_px=round(far_thr, 2),
        params=dict(max_speed_mps=max_speed_mps, max_gap_s=max_gap_s,
                    far_relax=far_relax, w_appear=w_appear, gap_pen=gap_pen,
                    no_link=no_link, k_end=k_end, n_eph=n_eph, r_dd_m=r_dd_m,
                    dd_frac=dd_frac, off_inp=off_inp, static_min_n=static_min_n,
                    static_diag_m=static_diag_m, static_out_m=static_out_m,
                    cluster_max_gap_s=cluster_max_gap_s,
                    expected_players=expected_players),
        caveats=[
            "relative_m: scale_anchor=null, mutlak metre/hiz/ivme/kosu iddiasi YOK.",
            "seam identity-swap riski: ardisik birlesmelerde kimlik takasi olabilir.",
            "under-merge (uzak-uc >max_gap parcalari) residual_under_merge'de "
            "raporlandi -- bu durustluktur, mesafe yalani DEGIL.",
            "olcek dogrulanmadi (PROVISIONAL L=34 W=18); mutlak metre iddia edilmez.",
            "appearance yalnizca veto/yeniden-siralama yapti, kenar URETMEDI "
            "(intra~inter ~ sans seviyesi).",
            "spurious tracklet'ler player_id=-1 + reason ile isaretlendi "
            "(silinmedi); tuketiciler player_id>=0 filtreler.",
            "clique_floor=EsZAMANLI sahadaki oyuncu (cannot-link sert siniri, "
            "survivors=15); roster_on_field=14 HEADLINE; n_distinct yedek "
            "rotasyonuyla daha fazla olabilir; ikisi de ayri/durust raporlanir, "
            "14'e ZORLAMA yok.",
        ],
    )
    report_path = os.path.join(out_dir, f"{base}_stitch_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    return dict(report=report, mapping=mapping, stitched=st,
                stitched_path=stitched_path, player_keyed_path=player_keyed_path,
                mapping_path=mapping_path, report_path=report_path)


# ----------------------------------------------------- appearance (opsiyonel) -
def _hist_dist(h1, h2) -> float:
    """Iki normalize H-S histogrami arasi mesafe (Bhattacharyya-benzeri, [0,1])."""
    import cv2
    return float(cv2.compareHist(h1.astype(np.float32), h2.astype(np.float32),
                                 cv2.HISTCMP_BHATTACHARYYA))


def _appearance_descriptors(df, summaries, video_path):
    """Ornek-frame'lerden cim-bastirilmis maskeli-govde H-S histogrami / tracklet.

    DEMOTE: olculen ayrim sans seviyesinde (intra 0.59 vs inter 0.58). Yalnizca
    konservatif cross-team veto + soft tie-break icin. q in [0,1] = box-boyutu *
    gecerli-piksel-orani * ornek-kapsama.
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
    tcol = df["tid"].to_numpy(np.int64)

    N = len(summaries)
    descs = [None] * N
    qs = np.zeros(N, dtype=np.float64)
    n_samp = 6  # tracklet basina ornek frame

    for k, s in enumerate(summaries):
        tid = s["tid"]
        sel = np.where(tcol == tid)[0]
        if sel.size == 0:
            continue
        # en buyuk box'lara yanli (en guvenilir govde), n_samp ornek
        order = sel[np.argsort(-hcol[sel])]
        order = order[:max(n_samp * 3, n_samp)]
        order = order[np.argsort(fcol[order])]
        step = max(1, len(order) // n_samp)
        picks = order[::step][:n_samp]

        hists = []
        valid_frac_acc = 0.0
        size_acc = 0.0
        for ridx in picks:
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
            # govde (torso): ayak noktasinin ustunde, box ust-yarisinin orta seridi
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
            # cim bastirma: yesil ton (H~35-85) maskele
            hh = hsv[:, :, 0]
            ss = hsv[:, :, 1]
            vv = hsv[:, :, 2]
            grass = (hh >= 35) & (hh <= 85) & (ss > 60) & (vv > 40)
            mask = (~grass).astype(np.uint8) * 255
            valid_frac = float(mask.mean() / 255.0)
            if valid_frac < 0.15:
                continue
            hist = cv2.calcHist([hsv], [0, 1], mask, [16, 16],
                                [0, 180, 0, 256])
            cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
            hists.append(hist)
            valid_frac_acc += valid_frac
            size_acc += float(bh)

        if not hists:
            continue
        descs[k] = np.mean(hists, axis=0)
        size_term = min(1.0, (size_acc / len(hists)) / 120.0)  # ~box yuksekligi ref
        cover_term = len(hists) / float(n_samp)
        valid_term = valid_frac_acc / len(hists)
        qs[k] = float(np.clip(size_term * cover_term * valid_term, 0.0, 1.0))

    cap.release()
    return descs, qs
