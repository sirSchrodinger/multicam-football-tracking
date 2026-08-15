#!/usr/bin/env python3
"""team_vest — amator gece halisahasinda YELEK kurallariyla takim ayrimi (CEPHE B).

Alperen'in OTORITE domain kurallari (bu modul bunlara gore yazildi):
  (1) ILK GOLDEN ONCE kimse yelek giymez -> renkle ayrim IMKANSIZ. Ilk golden SONRA
      gol-YIYEN takim yelek giyer -> "yelek vs yeleksiz". goal_time verilmezse modul
      AYIRMAYI REDDEDER ("pre_goal_unsplittable") -- uydurma yapmaz.
  (2) KALECI yelekli takimda olsa bile yelek GIYMEZ -> GK istisnasi: GK id'leri
      ASLA yelek-takimina atanmaz (cephe A gk_anchor'i veya ceza-sahasi heuristik).
  (3) Yelek disinda takimlari ayiran HICBIR SEY yok (karisik sahsi kiyafet) -> bu yuzden
      ayrim TEK-YONLU: "bu oyuncu yelekli mi?" Yeleksiz takimin ortak imzasi YOKTUR.
  (4) Yelek genelde SARI (bazen yesil). En zor edge: rakipte FENERBAHCE formasi
      (sari-lacivert) sari-yelekle karisir -> govde-ici hue yayilimi yuksekse
      "fener_suspect" dusuk-guven isaretlenir, yelek diye SAYILMAZ.

YONTEM (govde-merkez ROI sariligi):
  * ROI = govde (gogus->karin): foot'tan fy-0.90*bh .. fy-0.40*bh, fx +-0.32*bw.
    (Sort/bacak HARIC -- gece koyu-sort yanlis-pozitif okuyor; bkz docs/TEAM_NOTES.md.)
  * cim (HSV H35-85,S>50,V>40) ve golge (V<=25) maskelenir.
  * Sarilik (illumination-normalize): y = (R+G-2B)/(R+G+B)  (= 1.5*[(R+G)/2-B]/ortalama-yogunluk).
    Aydinlatmadan bagimsiz sari-mavi opponent ekseni. Yine de tek basina ZAYIF ayirir
    (gece desaturasyon) -> sari-HUE piksel orani (H16-35,S>55,V>55) ana is yuku.
  * Oyuncu basina ZAMANA-YAYILI ornekler (sadece near/mid box_h>=min_box_h; far ABSTAIN).

NEDEN TEK ESIK YETMEZ (olculen, durust):
  player_id izleri ID-SWAP ile KIRLI -- tek bir track sari-yelek + mavi + koyu
  oyunculari KARISTIRABILIR (28 Haz olcum: 19 track'in ~yarisi swap-kirli). Bu yuzden
  modul renk-KARARLILIGINI kullanir: yalniz TUTARLI sari (yelek) veya TUTARLI sari-degil
  (yeleksiz) track'ler GUVENLE etiketlenir; her iki rengi de goren (kirli) veya orta-bant
  track'ler ABSTAIN eder. SAHTE 7-7 URETILMEZ -- zorla-2-means YOK.

Karar (tek-yonlu + abstain-bandi):
  far/az-ornek               -> ABSTAIN (band=far)
  fYel>=vest_fyel & fDark<=vest_fdark   -> VEST     (tutarli sari)
  fYel<=novest_fyel                     -> NO_VEST  (hic sari okumadi)
  digerleri (orta-bant / cift-renk kirli) -> ABSTAIN
  GK id'leri -> 'gk' (yelek-takimina ASLA atanmaz, sayima girmez)
  fYel>=vest_fyel ama govde-ici hue yayilimi yuksek -> fener_suspect, dusuk-guven (yelek DEGIL)

CIKTI (additive, satir-sayisi + player_id korunur): {base}_vest.parquet
  (+vest_label, +vest_conf, +team_id[1=vest,0=novest,-1=abstain/gk]) + report JSON + histogram PNG.

Lisans: numpy + pandas + opencv(BSD) + matplotlib(PSF) + scipy. GPU yok, AGPL yok.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


# ------------------------------------------------------------------ defaults ---
DEFAULTS = dict(
    min_box_h=58.0,        # near/mid esigi (px). altinda far -> ABSTAIN (gece 25-50px guvenilmez)
    max_samples=34,        # oyuncu basina zamana-yayili ornek
    min_keep_px=15,        # govde ROI'de cim/golge sonrasi min piksel
    roi_top=0.90, roi_bot=0.40, roi_halfw=0.32,   # govde-merkez ROI (foot-orani)
    yel_h_lo=16, yel_h_hi=35, yel_s=55, yel_v=55, # sari-HUE piksel testi
    frame_yel_frac=0.22,   # bir KARE "sari" sayilir esigi (govde sari-piksel orani)
    frame_dark_yness=-0.04,# bir KARE "mavi/koyu" sayilir esigi (yness median)
    vest_fyel=0.45,        # VEST: karelerin >=%45'i sari
    vest_fdark=0.08,       # VEST: karelerin <=%8'i koyu (cift-renk kirliligi engeli)
    novest_fyel=0.05,      # NO_VEST: karelerin <=%5'i sari
    contam_yel=0.15, contam_dark=0.15,   # her ikisi de >= -> 'contaminated' isareti
    fener_hue_disp=28.0,   # govde-ici hue circ-std (derece) bunun ustu + sari -> fener_suspect
    gk_goal_margin_m=5.0,  # GK heuristik: kale cizgisinden bu kadar yakin
    gk_roam_max_m=6.5,     # GK heuristik: pitch_x yayilimi (std) bunun altinda
)


# --------------------------------------------------------- ilk-gol kapisi ------
def first_goal_gate(goal_time):
    """Kural (1). goal_time None/'unknown' -> ayirma REDDEDILIR.

    Doner dict(mode, can_split, goal_time, note).
    """
    if goal_time is None or (isinstance(goal_time, str) and
                             goal_time.strip().lower() in ("", "unknown", "none", "na")):
        return dict(
            mode="pre_goal_unsplittable", can_split=False, goal_time=None,
            note=("ilk-gol zamani bilinmiyor: ilk golden ONCE kimse yelek giymez, "
                  "renkle takim ayrimi ILKESEL olarak imkansiz -> AYIRMA REDDEDILDI "
                  "(uydurma yok). goal_time saniye verilirse golden-sonra orneklenir."))
    gt = float(goal_time)
    return dict(
        mode="post_goal", can_split=True, goal_time=gt,
        note=(f"ilk-gol t={gt:.1f}s varsayildi: yalniz golden-SONRAKI kareler "
              "orneklenir (yelek-vs-yeleksiz)."))


# ---------------------------------------------------- govde-sariligi ozelligi ---
def _circ_std_deg(hue_deg):
    """OpenCV hue (0-180) -> daire-istatistigi std (derece, 0-180 olcekli)."""
    if len(hue_deg) < 2:
        return 0.0
    ang = hue_deg.astype(np.float64) * (np.pi / 90.0)   # 0..2pi
    C = np.mean(np.cos(ang)); S = np.mean(np.sin(ang))
    R = np.hypot(C, S)
    R = min(1.0, max(1e-9, R))
    return float(np.sqrt(-2.0 * np.log(R)) * (90.0 / np.pi))


def torso_features(img, fx, fy, bh, bw, cfg=DEFAULTS):
    """Tek karede govde-merkez ROI'den (yness_median, yfrac, n_keep, hue_disp) doner.

    yness = (R+G-2B)/(R+G+B) illumination-normalize; yfrac = sari-HUE piksel orani;
    hue_disp = govde-ici doygun-piksel hue daire-std (Fener-formasi edge icin).
    Yetersiz govde pikseli -> None.
    """
    import cv2
    H, W = img.shape[:2]
    if not (np.isfinite(bh) and np.isfinite(bw)) or bh <= 4 or bw <= 2:
        return None
    cy1 = int(max(0, fy - cfg["roi_top"] * bh))
    cy2 = int(max(0, fy - cfg["roi_bot"] * bh))
    cx1 = int(max(0, fx - cfg["roi_halfw"] * bw))
    cx2 = int(min(W, fx + cfg["roi_halfw"] * bw))
    if cy2 <= cy1 + 1 or cx2 <= cx1 + 1:
        return None
    crop = img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    b = crop[:, :, 0].astype(np.float64)
    g = crop[:, :, 1].astype(np.float64)
    r = crop[:, :, 2].astype(np.float64)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hh = hsv[:, :, 0].astype(np.float64)
    ss = hsv[:, :, 1].astype(np.float64)
    vv = hsv[:, :, 2].astype(np.float64)
    grass = (hh >= 36) & (hh <= 85) & (ss > 50) & (vv > 40)
    shadow = vv <= 25
    keep = (~grass) & (~shadow)
    nkeep = int(keep.sum())
    if nkeep < cfg["min_keep_px"]:
        return None
    inten = (r + g + b) + 1e-6
    yness = (r + g - 2 * b) / inten
    yel = ((hh >= cfg["yel_h_lo"]) & (hh <= cfg["yel_h_hi"]) &
           (ss > cfg["yel_s"]) & (vv > cfg["yel_v"]))
    ymed = float(np.median(yness[keep]))
    yfrac = float(np.mean(yel[keep]))
    sat = keep & (ss > 60) & (vv > 50)
    hue_disp = _circ_std_deg(hh[sat]) if int(sat.sum()) >= 8 else 0.0
    return dict(yness=ymed, yfrac=yfrac, n_keep=nkeep, hue_disp=hue_disp)


def player_vest_features(df, video_path, goal_time=None, fps=None, cfg=DEFAULTS):
    """Her player_id icin zamana-yayili govde-sariligi ozet ozellikleri.

    Sadece near/mid (box_h>=min_box_h) ve goal_time verilmisse golden-SONRAKI kareler
    orneklenir. Doner dict pid -> dict(band, fYel, fDark, yness_med, yfrac_med, yness_std,
    within_hue_disp, n_samples, box_h_med).
    """
    import cv2
    if "player_id" not in df.columns:
        raise ValueError("girdi STITCHED degil (player_id kolonu yok)")
    if fps is None:
        tmax = float(df["t_sec"].max())
        fps = (float(df["frame"].max()) / tmax) if tmax > 0 else 25.0

    gt_frame = None
    if goal_time is not None and not (isinstance(goal_time, str)):
        gt_frame = int(round(float(goal_time) * fps))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"video acilamadi: {video_path}")

    bw_col = "box_w" in df.columns
    out = {}
    for pid, g in df.groupby("player_id"):
        g = g.copy()
        if gt_frame is not None:
            g = g[g["frame"] >= gt_frame]
        g = g[g["box_h"] >= cfg["min_box_h"]].sort_values("frame")
        box_h_med = float(g["box_h"].median()) if len(g) else np.nan
        if len(g) < 6:
            out[int(pid)] = dict(band="far", fYel=np.nan, fDark=np.nan,
                                 yness_med=np.nan, yfrac_med=np.nan, yness_std=np.nan,
                                 within_hue_disp=np.nan, n_samples=0,
                                 box_h_med=box_h_med)
            continue
        idx = np.linspace(0, len(g) - 1, min(cfg["max_samples"], len(g))).astype(int)
        sub = g.iloc[idx]
        ynl, yfl, disp = [], [], []
        for _, rr in sub.iterrows():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(rr["frame"]))
            ok, img = cap.read()
            if not ok or img is None:
                continue
            bw = float(rr["box_w"]) if bw_col else float(rr["box_h"]) * 0.45
            f = torso_features(img, float(rr["foot_x"]), float(rr["foot_y"]),
                               float(rr["box_h"]), bw, cfg)
            if f is None:
                continue
            ynl.append(f["yness"]); yfl.append(f["yfrac"]); disp.append(f["hue_disp"])
        if len(yfl) < 5:
            out[int(pid)] = dict(band="far", fYel=np.nan, fDark=np.nan,
                                 yness_med=np.nan, yfrac_med=np.nan, yness_std=np.nan,
                                 within_hue_disp=np.nan, n_samples=len(yfl),
                                 box_h_med=box_h_med)
            continue
        yn = np.asarray(ynl); yf = np.asarray(yfl)
        out[int(pid)] = dict(
            band="nearmid",
            fYel=float(np.mean(yf > cfg["frame_yel_frac"])),
            fDark=float(np.mean(yn < cfg["frame_dark_yness"])),
            yness_med=float(np.median(yn)),
            yfrac_med=float(np.median(yf)),
            yness_std=float(np.std(yn)),
            within_hue_disp=float(np.median(disp)),
            n_samples=int(len(yf)),
            box_h_med=box_h_med,
        )
    cap.release()
    return out


# ---------------------------------------------------- GK (ceza-sahasi heuristik) ---
def detect_gk_candidates(df, L=34.0, cfg=DEFAULTS):
    """Ceza-sahasi/kale-cizgisi heuristigi ile GK adaylari (cephe A gk_anchor yoksa).

    GK := median pitch_x kale cizgisine yakin (X<margin VEYA X>L-margin) VE pitch_x
    yayilimi (std) dusuk (az dolasir). DUSUK-GUVEN: cephe A gk_anchor varsa onu kullan.
    Doner dict pid -> reason.
    """
    if "pitch_x" not in df.columns:
        return {}
    gk = {}
    for pid, g in df.groupby("player_id"):
        x = g["pitch_x"].to_numpy(np.float64)
        x = x[np.isfinite(x)]
        if len(x) < 20:
            continue
        mx = float(np.median(x)); sx = float(np.std(x))
        near_goal = (mx < cfg["gk_goal_margin_m"]) or (mx > L - cfg["gk_goal_margin_m"])
        if near_goal and sx < cfg["gk_roam_max_m"]:
            side = "left" if mx < L / 2 else "right"
            gk[int(pid)] = f"heuristic: median_x={mx:.1f}m ({side}-goal), roam_std={sx:.1f}m"
    return gk


# --------------------------------------------------------------- siniflama -----
def classify(features, gk_ids=None, cfg=DEFAULTS):
    """Ozelliklerden tek-yonlu yelek siniflamasi (zorla-2-means YOK).

    Doner dict pid -> dict(label, conf, flags...) + ozet sayimlar ayri.
    """
    gk_ids = set(int(g) for g in (gk_ids or []))
    per = {}
    for pid, f in features.items():
        pid = int(pid)
        flags = dict(contaminated=False, fener_suspect=False, gk=False, band=f["band"])
        if pid in gk_ids:
            flags["gk"] = True
            per[pid] = dict(label="gk", conf=0.0, **flags,
                            reason="kaleci: yelek-takimina ASLA atanmaz")
            continue
        if f["band"] == "far" or not np.isfinite(f.get("fYel", np.nan)):
            per[pid] = dict(label="abstain", conf=0.0, **flags,
                            reason="far/az-ornek: gece uzak govde guvenilmez")
            continue
        fy, fd = f["fYel"], f["fDark"]
        contaminated = (fy >= cfg["contam_yel"]) and (fd >= cfg["contam_dark"])
        flags["contaminated"] = bool(contaminated)
        # tek-yonlu karar
        if fy >= cfg["vest_fyel"] and fd <= cfg["vest_fdark"]:
            fener = f.get("within_hue_disp", 0.0) is not None and \
                    np.isfinite(f["within_hue_disp"]) and \
                    f["within_hue_disp"] >= cfg["fener_hue_disp"]
            if fener:
                # sari-baskin AMA govde-ici hue cift-modlu -> Fenerbahce formasi suphesi
                flags["fener_suspect"] = True
                per[pid] = dict(label="abstain", conf=round(0.30, 3), **flags,
                                reason=("sari-baskin ama govde-ici hue yayilimi yuksek "
                                        f"({f['within_hue_disp']:.0f}deg) -> Fener-formasi "
                                        "suphesi, yelek SAYILMADI"))
                continue
            # guven: sari-baskinligi + cift-renk-temizligi
            conf = float(np.clip(0.5 + 0.5 * (fy - cfg["vest_fyel"]) /
                                 max(1e-6, 1 - cfg["vest_fyel"]) - 0.5 * fd, 0.3, 0.95))
            per[pid] = dict(label="vest", conf=round(conf, 3), **flags,
                            reason=f"tutarli sari: fYel={fy:.2f} fDark={fd:.2f}")
        elif fy <= cfg["novest_fyel"]:
            conf = float(np.clip(0.4 + 0.6 * fd, 0.4, 0.95))
            per[pid] = dict(label="no_vest", conf=round(conf, 3), **flags,
                            reason=f"hic sari okumadi: fYel={fy:.2f} fDark={fd:.2f}")
        else:
            why = ("cift-renk kirli (ID-swap suphesi)" if contaminated
                   else "orta-bant: yelek/yeleksiz belirsiz")
            per[pid] = dict(label="abstain", conf=round(float(0.5 * fy), 3), **flags,
                            reason=f"{why}: fYel={fy:.2f} fDark={fd:.2f}")
    return per


# ------------------------------------------------------------- histogram PNG ---
def _write_histogram(features, per, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pids = [p for p in features if features[p]["band"] == "nearmid"
            and np.isfinite(features[p]["yness_med"])]
    yness = np.array([features[p]["yness_med"] for p in pids])
    fyel = np.array([features[p]["fYel"] for p in pids])
    fdark = np.array([features[p]["fDark"] for p in pids])
    labels = [per[p]["label"] for p in pids]
    cmap = {"vest": "#e6b800", "no_vest": "#3060c0", "abstain": "#888888",
            "gk": "#22aa55"}
    cols = [cmap.get(l, "#888888") for l in labels]

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

    # (1) (R+G)/2-B normalize edilmis dagilim + en-buyuk-gap (BIMODAL DEGIL -> durust)
    order = np.argsort(yness)
    ys = yness[order]
    ax[0].hist(yness, bins=14, color="#bbbbbb", edgecolor="#444", alpha=0.7)
    for y, c in zip(yness, cols):
        ax[0].axvline(y, color=c, alpha=0.85, lw=1.6)
    if len(ys) > 1:
        gaps = np.diff(ys); gi = int(np.argmax(gaps))
        gap_mid = 0.5 * (ys[gi] + ys[gi + 1])
        ax[0].axvline(gap_mid, color="k", ls="--", lw=1.2)
        ax[0].text(gap_mid, ax[0].get_ylim()[1] * 0.92,
                   f" en-buyuk gap={gaps[gi]:.3f}\n (median gap={np.median(gaps):.3f})",
                   fontsize=8, va="top")
    ax[0].set_title("(R+G-2B)/(R+G+B) govde-sariligi (oyuncu-medyan)\n"
                    "SUREKLI dagilim -> tek esik ZAYIF (durust)", fontsize=9)
    ax[0].set_xlabel("illumination-normalize sarilik")
    ax[0].set_ylabel("oyuncu sayisi")

    # (2) fYel vs fDark: kararli-yelek / kararli-koyu / kirli yapi
    for x, y, c, p in zip(fyel, fdark, cols, pids):
        ax[1].scatter(x, y, c=c, s=70, edgecolor="k", lw=0.5, zorder=3)
        ax[1].text(x + 0.012, y, f"p{p}", fontsize=7, va="center")
    ax[1].axvspan(DEFAULTS["vest_fyel"], 1.0, ymin=0, ymax=DEFAULTS["vest_fdark"],
                  color="#e6b800", alpha=0.10)
    ax[1].axvline(DEFAULTS["vest_fyel"], color="#e6b800", ls=":", lw=1)
    ax[1].axhline(DEFAULTS["vest_fdark"], color="#e6b800", ls=":", lw=1)
    ax[1].axvline(DEFAULTS["novest_fyel"], color="#3060c0", ls=":", lw=1)
    ax[1].set_xlabel("fYel = sari-kare orani")
    ax[1].set_ylabel("fDark = koyu/mavi-kare orani")
    ax[1].set_title("kararlilik haritasi: sag-alt=yelek, sol=yeleksiz,\n"
                    "yuksek-fDark+yuksek-fYel = ID-swap kirli -> abstain", fontsize=9)
    ax[1].set_xlim(-0.03, 1.0); ax[1].set_ylim(-0.03, 1.0)

    from matplotlib.patches import Patch
    leg = [Patch(color=cmap[k], label=k) for k in ("vest", "no_vest", "abstain", "gk")]
    ax[1].legend(handles=leg, fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ ana akis ---
def assign_teams_vest(player_tracks_path, out_dir, video_path,
                      goal_time=None, gk_ids=None, calib_path=None, cfg=None):
    """Stitched player-keyed parquet -> yelek-kuralli takim siniflamasi (CEPHE B).

    goal_time None -> 'pre_goal_unsplittable' (ayirma reddedilir). Aksi halde golden-sonra
    govde-sariligindan tek-yonlu yelek/yeleksiz/abstain. Additive cikti + report + PNG.
    """
    cfg = dict(DEFAULTS, **(cfg or {}))
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_parquet(player_tracks_path)
    base = Path(player_tracks_path).stem

    # saha boyu L (GK heuristik icin; calib varsa oradan, yoksa pitch'ten/varsayilan)
    L = 34.0
    if calib_path and os.path.exists(calib_path):
        try:
            cj = json.load(open(calib_path))
            L = float(cj.get("pitch_dims_m", {}).get("L")
                      or cj["template"]["dims_m"][0])
        except Exception:
            pass

    gate = first_goal_gate(goal_time)

    # GK adaylari (acik gk_ids verilmezse heuristik)
    if gk_ids is None:
        gk_map = detect_gk_candidates(df, L=L, cfg=cfg)
        gk_ids = sorted(gk_map.keys())
        gk_source = "penalty_box_heuristic"
    else:
        gk_ids = [int(g) for g in gk_ids]
        gk_map = {g: "explicit (cephe A gk_anchor)" for g in gk_ids}
        gk_source = "explicit"

    if not gate["can_split"]:
        # KURAL (1): ayirma REDDEDILDI. Herkes abstain; sahte takim URETILMEZ.
        pids = sorted(int(p) for p in df["player_id"].unique())
        per = {p: dict(label="abstain", conf=0.0, band="pre_goal",
                       reason="pre_goal_unsplittable") for p in pids}
        report = dict(
            front="B_team_vest", gate=gate, n_players=len(pids),
            counts=dict(vest=0, no_vest=0, abstain=len(pids), gk=0),
            gk_candidates=gk_map, gk_source=gk_source,
            per_player=[dict(player_id=p, **per[p]) for p in pids],
            caveats=[
                "ilk-gol bilinmiyor -> renkle takim ayrimi ILKESEL imkansiz, REDDEDILDI.",
                "goal_time (saniye) verilince golden-sonra yelek-vs-yeleksiz calisir.",
                "SAHTE 7-7 URETILMEDI.",
            ])
        out_df = df.copy()
        out_df["vest_label"] = "abstain"
        out_df["vest_conf"] = np.float32(0.0)
        out_df["team_id"] = np.int64(-1)
        return _write_outputs(out_df, report, None, out_dir, base, features=None, per=per)

    # KURAL post-goal: govde-sariligi
    features = player_vest_features(df, video_path, goal_time=gate["goal_time"], cfg=cfg)
    per = classify(features, gk_ids=gk_ids, cfg=cfg)

    counts = dict(vest=0, no_vest=0, abstain=0, gk=0)
    for p in per.values():
        counts[p["label"]] += 1
    n = len(per)
    abstain_rate = round(counts["abstain"] / n, 3) if n else 0.0

    pp = []
    for pid in sorted(per):
        f = features[pid]
        pp.append(dict(
            player_id=int(pid), label=per[pid]["label"],
            conf=per[pid]["conf"], band=f["band"],
            fYel=None if not np.isfinite(f.get("fYel", np.nan)) else round(f["fYel"], 3),
            fDark=None if not np.isfinite(f.get("fDark", np.nan)) else round(f["fDark"], 3),
            yness_med=None if not np.isfinite(f.get("yness_med", np.nan)) else round(f["yness_med"], 4),
            yfrac_med=None if not np.isfinite(f.get("yfrac_med", np.nan)) else round(f["yfrac_med"], 3),
            within_hue_disp=None if not np.isfinite(f.get("within_hue_disp", np.nan)) else round(f["within_hue_disp"], 1),
            box_h_med=None if not np.isfinite(f.get("box_h_med", np.nan)) else round(f["box_h_med"], 1),
            n_samples=int(f.get("n_samples", 0)),
            contaminated=bool(per[pid].get("contaminated", False)),
            fener_suspect=bool(per[pid].get("fener_suspect", False)),
            gk=bool(per[pid].get("gk", False)),
            reason=per[pid].get("reason", ""),
        ))

    n_contam = sum(1 for x in pp if x["contaminated"])
    report = dict(
        front="B_team_vest", gate=gate, n_players=n,
        method="one_sided_vest_yellowness (torso ROI, illumination-normalized, "
               "near/mid only, color-stability gated)",
        counts=counts, abstain_rate=abstain_rate,
        n_contaminated_tracks=n_contam,
        gk_candidates=gk_map, gk_source=gk_source,
        thresholds={k: cfg[k] for k in (
            "min_box_h", "frame_yel_frac", "frame_dark_yness",
            "vest_fyel", "vest_fdark", "novest_fyel",
            "contam_yel", "contam_dark", "fener_hue_disp")},
        per_player=pp,
        honest_findings=[
            "yness ((R+G)/2-B) dagilimi SUREKLI (bimodal degil) -> tek esik ZAYIF; "
            "is yukunu sari-HUE piksel orani (fYel) + kararlilik (fDark) tasiyor.",
            f"{n_contam} track ID-SWAP kirli (hem sari hem koyu kare goruyor) -> "
            "per-track team ATANAMAZ, abstain edildi.",
            "DIKKAT (over-claim degil): bir track swap ile TEK renge baskinsa (or. cogu "
            "kare sari) o renkte okunur; bu yuzden 'vest' guveni TRACK temizligi kadardir, "
            "fiziksel-oyuncu temizligi DEGIL. team_id ancak tracking kadar temiz.",
            "tek-yonlu: yalniz YELEK guvenle bulunur; yeleksiz takimin ortak imzasi yok.",
            "GK yelek-takimina ASLA atanmadi (kural 2).",
            "Fener-formasi suphesi: sari-baskin+yuksek-hue-yayilimi -> yelek SAYILMADI.",
            "SAHTE 7-7 URETILMEDI; dengeye zorlanmadi.",
        ],
        caveats=[
            "additive: girdiye sadece vest_label+vest_conf+team_id eklendi.",
            "goal_time bu kosuda VARSAYILDI (operator teyidi gerek); golden-once kareler atilmadi varsayilirsa abstain artar.",
            "ABSTAIN dogru davranis: sahte takim uretmemek icin.",
        ])

    out_df = df.copy()
    lab_map = {int(p): per[p]["label"] for p in per}
    conf_map = {int(p): float(per[p]["conf"]) for p in per}
    team_map = {int(p): (1 if per[p]["label"] == "vest"
                         else 0 if per[p]["label"] == "no_vest" else -1) for p in per}
    out_df["vest_label"] = out_df["player_id"].map(lab_map).astype("object")
    out_df["vest_conf"] = out_df["player_id"].map(conf_map).astype("float32")
    out_df["team_id"] = out_df["player_id"].map(team_map).astype("int64")

    return _write_outputs(out_df, report, features, out_dir, base, features_full=features, per=per)


def _write_outputs(out_df, report, _unused, out_dir, base, features=None,
                   features_full=None, per=None):
    parquet_path = os.path.join(out_dir, f"{base}_vest.parquet")
    out_df.to_parquet(parquet_path, index=False)
    report_path = os.path.join(out_dir, f"{base}_vest_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    png_path = None
    if features_full and per:
        png_path = os.path.join(out_dir, f"{base}_vest_hist.png")
        try:
            _write_histogram(features_full, per, png_path)
        except Exception as e:
            png_path = f"(histogram atlandi: {e})"
    return dict(report=report, parquet_path=parquet_path,
               report_path=report_path, hist_png=png_path)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="CEPHE B: yelek-kuralli takim ayrimi")
    ap.add_argument("--tracks", default="stats_out/full_demo/"
                    "tracks_cankaya_cam2_clip2400_presplit_player.parquet")
    ap.add_argument("--video", default="raw/cankaya_cam2_clip2400.mp4")
    ap.add_argument("--out", default="scratchpad/wf_team")
    ap.add_argument("--goal-time", default=None,
                    help="ilk-gol saniyesi; verilmezse pre_goal_unsplittable")
    ap.add_argument("--calib", default="calib/cankaya_cam2_v2.json")
    args = ap.parse_args()
    gt = None if args.goal_time in (None, "", "unknown") else float(args.goal_time)
    res = assign_teams_vest(args.tracks, args.out, args.video,
                            goal_time=gt, calib_path=args.calib)
    import pprint
    pprint.pprint(res["report"]["counts"])
    print("gate:", res["report"]["gate"]["mode"])
    print("parquet:", res["parquet_path"])
    print("report:", res["report_path"])
    print("hist:", res["hist_png"])
