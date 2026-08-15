#!/usr/bin/env python3
"""overnight_runner.py — Alperen uyurken çalışan, TOKEN HARCAMAYAN lokal GPU/CPU
araştırma runner'ı (8 saatlik pencere, saatlik checkpoint).

Felsefe: ağır iş tamamen LOKAL (Claude yok). Her aşama izole (try/except), zaman
bütçeli, STATUS.md'ye saatlik yazar. Bir aşama patlasa diğerleri devam eder.
Claude sadece bittiğinde (kota resetlenmiş olur) sonucu sentezleyip commit'ler.

Aşamalar (öncelik sırası):
  1. extended_sims      (CPU)  — derin Monte Carlo + duyarlılık (dimension_sim)
  2. centercircle_data  (CPU)  — merkez-yuvarlak ROI topla + Hough dene (focal-bağımsız ölçek hazırlığı)
  3. fullgame_export    (GPU)  — aktif oyun penceresi (t=1500-3450) tiling'le export (ANA ürün)
  4. fullgame_analysis  (CPU)  — run_pipeline.py ile roster+konum-haritası+figür
  5. clahe_ablation     (GPU)  — karanlık karelerde CLAHE tespit-sayısı deltası (vakit varsa)

Kullanım:
  venv/bin/python eval/overnight_runner.py            # gerçek 8h koşu
  venv/bin/python eval/overnight_runner.py --smoke    # hızlı plumbing testi (~3dk)
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
PY = str(ROOT / "venv/bin/python")
STATUS = ROOT / "docs/OVERNIGHT_STATUS.md"
RESULTS = ROOT / "docs/OVERNIGHT_RESULTS.json"
OUTDIR = ROOT / "overnight_out"
OUTDIR.mkdir(exist_ok=True)

T0 = time.time()
RESULTS_D = {"started_epoch": T0, "stages": {}}


def hrs():
    return (time.time() - T0) / 3600.0


def log(msg: str):
    line = f"- [{hrs():4.2f}h] {msg}"
    print(line, flush=True)
    with open(STATUS, "a") as f:
        f.write(line + "\n")


def stage(name, fn, deadline_h, smoke):
    if hrs() > deadline_h:
        log(f"SKIP {name}: zaman bütçesi aşıldı ({hrs():.2f}h > {deadline_h}h)")
        RESULTS_D["stages"][name] = {"status": "skipped_time"}
        return RESULTS_D["stages"][name]
    log(f"BAŞLA {name}")
    t = time.time()
    try:
        info = fn(smoke) or {}
        RESULTS_D["stages"][name] = {"status": "ok", "mins": round((time.time()-t)/60, 1), **info}
        log(f"BİTTİ {name} ({(time.time()-t)/60:.1f}dk) {info}")
    except Exception as e:  # noqa: BLE001
        RESULTS_D["stages"][name] = {"status": "error", "error": str(e),
                                     "mins": round((time.time()-t)/60, 1)}
        log(f"HATA {name}: {e}")
        (OUTDIR / f"err_{name}.txt").write_text(traceback.format_exc())
    RESULTS.write_text(json.dumps(RESULTS_D, indent=2))
    return RESULTS_D["stages"][name]


# ----------------------------------------------------------------------------- #
def s_sims(smoke):
    """Derin Monte Carlo + duyarlılık (gerçek dimension_sim kodu)."""
    import eval.dimension_sim as ds
    K = 50 if smoke else 2000
    ideal = ds.run_ideal()
    mc = ds.run_montecarlo(K=K)
    se = ds.run_sensitivity()
    out = dict(ideal_err=ideal["err_pct"], mc=mc["stats"],
               real_focal_band=se["real_focal_band_pct"],
               syn_focal_band=se["syn_focal_band_pct"],
               height_band=se["height_band_pct"])
    (OUTDIR / "sims_deep.json").write_text(json.dumps(out, indent=2))
    return dict(mc_trials=K, mc_bias_pct=round(mc["stats"]["L_bias_pct"], 3),
                mc_spread_pct=round(mc["stats"]["L_spread_pct"], 3))


def s_centercircle(smoke):
    """Merkez-yuvarlak ROI'sini birçok kareden medyanla topla + beyaz-çizgi mask +
    HoughCircles/fitEllipse dene. Focal-BAĞIMSIZ ölçek-çıpası için veri hazırlığı.
    Best-effort: bulamasa da ROI + mask'ı diske yazar (sonraki oturum analizi)."""
    import cv2, numpy as np
    src = ROOT / "raw/cankaya_cam2.mp4"
    cap = cv2.VideoCapture(str(src)); fps = cap.get(cv2.CAP_PROP_FPS)
    n = 6 if smoke else 120
    # merkez-yuvarlak bu kamerada ~ust-orta (distorted ~x1357,y226). Genis ROI al.
    x0, x1, y0, y1 = 1150, 1750, 150, 430
    acc = []
    step = max(1, int(60 * fps))  # ~her 60s bir kare
    for k in range(n):
        cap.set(cv2.CAP_PROP_POS_FRAMES, k * step)
        ok, fr = cap.read()
        if not ok:
            break
        acc.append(fr[y0:y1, x0:x1].astype(np.float32))
    cap.release()
    if not acc:
        return {"note": "kare okunamadi"}
    med = np.median(np.stack(acc), axis=0).astype(np.uint8)
    cv2.imwrite(str(OUTDIR / "centercircle_roi_median.png"), med)
    # beyaz-cizgi mask (yuksek parlaklik) + edge
    g = cv2.cvtColor(med, cv2.COLOR_BGR2GRAY)
    mask = (g > np.percentile(g, 92)).astype(np.uint8) * 255
    cv2.imwrite(str(OUTDIR / "centercircle_linemask.png"), mask)
    edges = cv2.Canny(g, 40, 120)
    cv2.imwrite(str(OUTDIR / "centercircle_edges.png"), edges)
    circles = cv2.HoughCircles(cv2.medianBlur(g, 3), cv2.HOUGH_GRADIENT, dp=1.2,
                               minDist=80, param1=120, param2=40, minRadius=20, maxRadius=200)
    found = [] if circles is None else circles[0].tolist()
    json.dump({"roi_xyxy": [x0, y0, x1, y1], "hough_circles_roi": found, "n_frames": len(acc)},
              open(OUTDIR / "centercircle_fit.json", "w"), indent=2)
    return {"hough_n": len(found), "roi_saved": True}


def s_export(smoke):
    """ANA ÜRÜN: aktif oyun penceresini far-band tiling'le tam export et."""
    import cv2
    import export_tracks as et
    calib = str(ROOT / "calib/cankaya_cam2_v2.json")
    if smoke:
        clip = str(ROOT / "raw/cankaya_cam2_clip2400.mp4")
        out = str(OUTDIR / "tracks_smoke.parquet")
        et.run_tracking_export(clip, calib_path=calib, camera_id="cankaya_cam2",
                               out_path=out, max_frames=80, recover_far=True)
        return {"out": out, "smoke": True}
    # aktif pencere t=1500..2220 (12 dk) — DEPTH örneği (breadth'e zaman bırakmak için kısa)
    aw = ROOT / "raw/_active_window.mp4"
    if not aw.exists():
        subprocess.run(["ffmpeg", "-y", "-ss", "1500", "-t", "720",
                        "-i", str(ROOT / "raw/cankaya_cam2.mp4"), "-c", "copy", str(aw)],
                       check=True, capture_output=True)
    out = str(ROOT / "raw/tracks_cankaya_cam2_fullgame.parquet")
    et.run_tracking_export(str(aw), calib_path=calib, camera_id="cankaya_cam2",
                           out_path=out, recover_far=True)
    import pandas as pd
    df = pd.read_parquet(out)
    return {"out": out, "rows": len(df), "tracks": int(df["tid"].nunique())}


def s_analysis(smoke):
    """run_pipeline.py ile roster + sürekli konum haritası + figür (tested pipeline)."""
    parq = (OUTDIR / "tracks_smoke.parquet") if smoke else (ROOT / "raw/tracks_cankaya_cam2_fullgame.parquet")
    if not Path(parq).exists():
        return {"note": "parquet yok, export atlandi/patladi"}
    calib = str(ROOT / "calib/cankaya_cam2_v2.json")
    cmd = [PY, str(ROOT / "scripts/run_pipeline.py"), str(parq), calib,
           "--scale-height", "--expected-players", "14"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    (OUTDIR / "analysis_stdout.txt").write_text(r.stdout + "\n---STDERR---\n" + r.stderr)
    return {"returncode": r.returncode, "tail": r.stdout.strip().splitlines()[-3:] if r.stdout else []}


def s_clahe(smoke):
    """Karanlık karede CLAHE far-band tespit-sayısı deltası (GT'siz relatif sinyal)."""
    import cv2, numpy as np
    from rfdetr import RFDETRLargeDeprecated
    from PIL import Image
    model = RFDETRLargeDeprecated(pretrain_weights=str(ROOT / "models/weights/checkpoint_best_regular.pth"),
                                  device="cuda", num_classes=4)
    frames = [37307, 52230] if smoke else [37307, 44769, 52230, 82076]
    band = (110, 365)
    def det_count(img_rgb, thr=0.40):
        d = model.predict(Image.fromarray(img_rgb), threshold=thr)
        xy = np.asarray(d.xyxy, float).reshape(-1, 4)
        return int(((xy[:, 3] - xy[:, 1]) > 25).sum()) if len(xy) else 0
    res = {}
    for fi in frames:
        fr = cv2.imread(str(ROOT / f"recall_val/frames/f{fi}.png"))
        if fr is None:
            continue
        crop = fr[band[0]:band[1]]
        big = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 0])
        cl = cv2.resize(cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), None, fx=2.0, fy=2.0,
                        interpolation=cv2.INTER_CUBIC)
        res[fi] = dict(plain=det_count(big[:, :, ::-1]), clahe=det_count(cl[:, :, ::-1]))
    json.dump(res, open(OUTDIR / "clahe_ablation.json", "w"), indent=2)
    return {"frames": len(res), "result": res}


# ============================ ÇOK-TESİS BOY DAĞILIMI ============================ #
SOSYAL_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/125 Safari/537.36"


def _sget(url, xhr=False, timeout=25):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": SOSYAL_UA})
    if xhr:
        req.add_header("X-Requested-With", "XMLHttpRequest")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


DATES_ALL = [f"{d:02d}.06.2026" for d in range(27, 12, -1)]  # 27.06..13.06 (15-gün pencere)


def discover_venues(n_venues, dates, skip=None):
    """Son tarihlerden FARKLI tesis başına bir maç video URL'i topla (skip'tekiler hariç)."""
    import json, re, time
    skip = set(skip or [])
    seen = {}
    for date in dates:
        if len(seen) >= n_venues:
            break
        try:
            d = json.loads(_sget(f"https://sosyalhalisaha.com/xhr/filtre/___{date}__", xhr=True))
        except Exception as e:  # noqa: BLE001
            log(f"  discover {date}: liste hata {e}"); continue
        if d.get("status") != "success":
            continue
        for m in d.get("data", []):
            venue = (m.get("place") or {}).get("name", "?")
            if venue in seen or venue in skip:
                continue
            try:
                html = _sget(m["url"])
                mm = re.search(r"videoSrc\s*=\s*(\[.*?\]);", html, re.S)
                if not mm:
                    continue
                arr = json.loads(mm.group(1))
                if arr:
                    seen[venue] = arr[0]["url"]
                    log(f"  venue '{venue}' -> {arr[0]['url'][:64]}")
            except Exception:  # noqa: BLE001
                continue
            if len(seen) >= n_venues:
                break
            time.sleep(0.3)
    return seen


def _ff_frames(url, outdir, start, dur, fps):
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    rc = ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]
    out = str(outdir / "f_%04d.jpg")
    # 1) hızlı: -ss girdiden ÖNCE (range seek). 2) fallback: -ss girdiden SONRA (decode).
    for cmd in (["ffmpeg", "-y", *rc, "-ss", str(start), "-i", url, "-t", str(dur),
                 "-vf", f"fps={fps}", "-q:v", "3", out],
                ["ffmpeg", "-y", *rc, "-i", url, "-ss", str(start), "-t", str(dur),
                 "-vf", f"fps={fps}", "-q:v", "3", out]):
        r = subprocess.run(cmd, capture_output=True, timeout=1200)
        frames = sorted(outdir.glob("f_*.jpg"))
        if frames:
            return frames
    return []


def _detect_boxes(model, frame_paths, clean_upright=True):
    """(foot_y, box_h) topla. clean_upright=True: height_scale._clean_upright ile AYNI
    filtre (conf>0.6, h>50, 1.7<en-boy<5.0) -> bükülen/örtülü/FP kutuları eler;
    box_h'i gerçek-boy proxy'sine yaklaştırır (çan-eğrisi testi için kritik)."""
    import cv2, numpy as np
    from PIL import Image
    rows = []
    for p in frame_paths:
        fr = cv2.imread(str(p))
        if fr is None:
            continue
        det = model.predict(Image.fromarray(fr[:, :, ::-1]), threshold=0.30)
        xy = np.asarray(det.xyxy, float).reshape(-1, 4)
        cf = np.asarray(det.confidence, float).reshape(-1)
        if not len(xy):
            continue
        h = xy[:, 3] - xy[:, 1]; w = xy[:, 2] - xy[:, 0]
        if clean_upright:
            ar = h / np.clip(w, 1.0, None)
            keep = (cf > 0.6) & (h > 50) & (ar > 1.7) & (ar < 5.0)
        else:
            keep = h > 25
        for a, hh in zip(xy[keep], h[keep]):
            rows.append((float(a[3]), float(hh)))   # (foot_y, box_h)
    return np.array(rows) if rows else np.empty((0, 2))


def _band_norm_heights(arr, n_bands=8, min_per=40):
    """foot_y bantları içinde box_h'i band-medyanına böl -> ÖLÇEK-BAĞIMSIZ boy proxy
    (ince bantta perspektif ~sabit -> box_h ∝ gerçek boy). Kalibrasyon GEREKMEZ."""
    import numpy as np
    if len(arr) < min_per * 2:
        return np.array([])
    fy, bh = arr[:, 0], arr[:, 1]
    qs = np.quantile(fy, np.linspace(0, 1, n_bands + 1))
    out = []
    for i in range(n_bands):
        hi = fy <= qs[i + 1] if i == n_bands - 1 else fy < qs[i + 1]
        m = (fy >= qs[i]) & hi
        if m.sum() < min_per:
            continue
        med = np.median(bh[m])
        if med > 0:
            out.append(bh[m] / med)
    return np.concatenate(out) if out else np.array([])


def _normality(x):
    import numpy as np
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) < 30:
        return {"n": int(len(x)), "ok": False}
    try:
        from scipy import stats as st
        sk = float(st.skew(x)); ku = float(st.kurtosis(x))
        xs = x if len(x) <= 5000 else np.random.default_rng(0).choice(x, 5000, replace=False)
        try:
            sh_p = float(st.shapiro(xs).pvalue)
        except Exception:  # noqa: BLE001
            sh_p = float("nan")
        osm, osr = st.probplot(x, dist="norm")[0]
        qq_r = float(np.corrcoef(osm, osr)[0, 1])
    except Exception:  # noqa: BLE001 - scipy yoksa elle skew/kurt
        m, s = np.mean(x), np.std(x)
        sk = float(np.mean(((x - m) / s) ** 3)); ku = float(np.mean(((x - m) / s) ** 4) - 3)
        sh_p = float("nan"); qq_r = float("nan")
    cv = float(np.std(x) / np.mean(x)) if np.mean(x) else float("nan")
    bell = bool(abs(sk) < 0.5 and abs(ku) < 1.0 and (np.isnan(qq_r) or qq_r > 0.985))
    return {"n": int(len(x)), "mean": round(float(np.mean(x)), 3), "cv": round(cv, 3),
            "skew": round(sk, 3), "excess_kurt": round(ku, 3),
            "qq_r": None if np.isnan(qq_r) else round(qq_r, 4),
            "shapiro_p": None if np.isnan(sh_p) else round(sh_p, 4), "bell_curve": bell}


def s_multivenue(smoke, offset=0, accumulate=False):
    """ÇOK-TESİS boy-dağılımı çan-eğrisi testi (Alperen ana isteği). Kalibrasyon-
    bağımsız band-normalize box_h + her tesiste normallik; + Çankaya kalibre metrik boy.
    accumulate=True + offset: doldurma turlarında YENİ tesisler ekler (tekrar yok)."""
    import json, shutil
    import numpy as np
    from rfdetr import RFDETRLargeDeprecated
    n_venues = 2 if smoke else 6
    seen_path = OUTDIR / "seen_venues.json"
    all_path = OUTDIR / "height_distribution_all.json"
    seen = set(json.loads(seen_path.read_text())) if (accumulate and seen_path.exists()) else set()
    per = json.loads(all_path.read_text()) if (accumulate and all_path.exists()) else {}
    dates = DATES_ALL[:3] if smoke else (DATES_ALL[offset % len(DATES_ALL):] + DATES_ALL[:offset % len(DATES_ALL)])
    venues = discover_venues(n_venues, dates, skip=seen)
    log(f"  keşfedilen {len(venues)} YENİ tesis (offset={offset}, zaten görülen {len(seen)})")
    model = RFDETRLargeDeprecated(pretrain_weights=str(ROOT / "models/weights/checkpoint_best_regular.pth"),
                                  device="cuda", num_classes=4)
    vdir = OUTDIR / "venues"; vdir.mkdir(exist_ok=True)
    for venue, url in venues.items():
        seen.add(venue)   # başarısız olsa da işaretle (kırık URL'yi tekrar deneme)
        try:
            fdir = vdir / ("".join(c for c in venue if c.isalnum())[:18] or "v")
            frames = _ff_frames(url, fdir, start=(60 if smoke else 600),
                                dur=(40 if smoke else 240), fps=(1.0 if smoke else 1.5))
            if smoke:
                frames = frames[:12]
            arr = _detect_boxes(model, frames)
            norm = _band_norm_heights(arr)
            per[venue] = {"raw_n": int(len(arr)), "norm_stats": _normality(norm),
                          "_norm": norm.tolist()[:6000]}
            log(f"  '{venue}': {len(arr)} dets, bell={per[venue]['norm_stats'].get('bell_curve')} "
                f"skew={per[venue]['norm_stats'].get('skew')}")
            shutil.rmtree(fdir, ignore_errors=True)
        except Exception as e:  # noqa: BLE001
            log(f"  '{venue}': hata {e}")
    # Çankaya kalibre METRİK boy (gerçek metre) — referans çan eğrisi (bir kez)
    if "Cankaya_kalibre_metre" not in per:
      try:
        import pandas as pd
        from pitch.height_scale import estimate_heights_relm, _pose_at_f
        calib = json.loads((ROOT / "calib/cankaya_cam2_v2.json").read_text())
        H = np.array(calib["H_img2pitch"]); K = np.array(calib["K"]); dist = np.array(calib["dist"])
        df = pd.read_parquet(ROOT / "raw/tracks_cankaya_cam2_clip2400.parquet")
        Rb, tb, _, _ = _pose_at_f(np.linalg.inv(H), float(K[0, 0]), 960, 540)
        Z = estimate_heights_relm(df, H, K, dist, Rb, tb) * 0.9566
        per["Cankaya_kalibre_metre"] = {"raw_n": int(len(Z)),
            "metric_height_stats": _normality(Z),
            "norm_stats": _normality(_band_norm_heights(df[["foot_y", "box_h"]].to_numpy(float))),
            "_metric": Z.tolist()[:6000]}
        log(f"  Çankaya metrik boy: {per['Cankaya_kalibre_metre']['metric_height_stats']}")
        seen.add("Cankaya_kalibre_metre")
      except Exception as e:  # noqa: BLE001
        log(f"  Çankaya metrik: hata {e}")
    # kaydet: tam (figür+accumulate için) + insan-okur özet + seen
    all_path.write_text(json.dumps(per, ensure_ascii=False))
    json.dump({k: {kk: vv for kk, vv in v.items() if not kk.startswith('_')} for k, v in per.items()},
              open(OUTDIR / "height_distribution.json", "w"), indent=2, ensure_ascii=False)
    seen_path.write_text(json.dumps(sorted(seen), ensure_ascii=False))
    try:
        _fig_heightdist(per)
    except Exception as e:  # noqa: BLE001
        log(f"  figür hata {e}")
    n_bell = sum(1 for v in per.values() if v.get("norm_stats", {}).get("bell_curve"))
    return {"venues_total": len(per), "new_this_round": len(venues), "bell_curve_count": n_bell}


def _fig_heightdist(per):
    import numpy as np, matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT.parent / "Templates/article-twocolumn/figures"))
    import paper_style as ps; ps.use()
    items = [(k, v) for k, v in per.items() if v.get("_norm") or v.get("_metric")]
    # metrik referans (Çankaya) önce, sonra en çok veri olanlar; en fazla 12 panel
    items.sort(key=lambda kv: (0 if kv[1].get("_metric") else 1, -kv[1].get("raw_n", 0)))
    items = items[:12]
    if not items:
        return
    cols = min(3, len(items)); rows = (len(items) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(7.0, 2.5 * rows), squeeze=False)
    for ax, (k, v) in zip(axes.ravel(), items):
        x = np.array(v.get("_metric") or v.get("_norm"), float)
        s = v.get("metric_height_stats") or v.get("norm_stats")
        ax.hist(x, bins=40, density=True, color=ps.C["blue"], alpha=0.7, ec="white")
        mu, sd = np.mean(x), np.std(x)
        xs = np.linspace(x.min(), x.max(), 200)
        ax.plot(xs, np.exp(-(xs - mu) ** 2 / (2 * sd ** 2)) / (sd * (2 * np.pi) ** .5),
                color=ps.C["red"], lw=1.8)
        ax.set_title(f"{k[:20]}\nskew {s.get('skew')} kurt {s.get('excess_kurt')} "
                     f"bell={s.get('bell_curve')}", fontsize=7)
        ax.set_yticks([])
    for ax in axes.ravel()[len(items):]:
        ax.axis("off")
    fig.suptitle("Tesis-başı boy dağılımı (kırmızı = Gauss fit) — çan eğrisi testi", fontsize=10)
    fig.tight_layout()
    ps.save(fig, str(ROOT / "docs/report/figures/height_distribution"))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--refine", action="store_true",
                    help="SADECE clean-upright-filtreli çok-tesis breadth (boy çan-eğrisi düzeltme, ~3h)")
    args = ap.parse_args()
    smoke = args.smoke
    if args.refine:
        STATUS.write_text(f"# Overnight runner — REFINE çok-tesis (clean-upright filtered) "
                          f"(epoch {int(T0)})\n\n")
        log("REFINE: clean-upright filtreli çok-tesis breadth (box_h boy-proxy düzeltme)")
        stage("refine_baseline", s_multivenue, 0.6, smoke)
        FLOOR = 0.05 if smoke else 3.0
        rnd = 0; dry = 0
        while hrs() < FLOOR and rnd < 200:
            rnd += 1
            r = stage(f"refine_breadth_{rnd}",
                      (lambda s, k=rnd: s_multivenue(s, offset=k, accumulate=True)),
                      FLOOR + 0.3, smoke)
            if (r or {}).get("new_this_round", 0) == 0:
                dry += 1
                if dry >= 5:
                    log("  tesisler tükendi -> refine bitti"); break
            else:
                dry = 0
        RESULTS_D["finished_epoch"] = time.time(); RESULTS_D["total_hours"] = round(hrs(), 2)
        RESULTS.write_text(json.dumps(RESULTS_D, indent=2))
        log(f"REFINE bitti — {hrs():.2f}h, breadth turu {rnd}")
        return
    STATUS.write_text(f"# Overnight runner — {'SMOKE' if smoke else 'GERÇEK 8h'} "
                      f"(başlangıç epoch {int(T0)})\n\n")
    log(f"runner başladı (smoke={smoke}). plan: sims→çok-tesis BREADTH(birincil)→merkez-yuvarlak→export(12dk)→analiz→topup")
    # Alperen önceliği: ÇOK-TESİS breadth (çan-eğrisi) BİRİNCİL ve ÖNCE; tek-tesis depth ikincil+kısa.
    if smoke:
        BREADTH_H, FLOOR, DLcc, DLexp, DLan = 0.04, 0.06, 0.9, 1.1, 1.3
    else:
        BREADTH_H, FLOOR, DLcc, DLexp, DLan = 5.5, 7.5, 6.0, 7.0, 7.5
    stage("extended_sims", s_sims, 0.2 if smoke else 0.6, smoke)
    stage("multivenue_heightdist", s_multivenue, 0.8 if smoke else 6.0, smoke)  # baseline + Çankaya metrik
    stage("centercircle_data", s_centercircle, DLcc, smoke)
    # === BREADTH (BİRİNCİL): BREADTH_H'e kadar mümkün olduğunca çok YENİ tesis (çan-eğrisi gücü) ===
    rnd = 0; dry = 0
    while hrs() < BREADTH_H and rnd < 200:
        rnd += 1
        r = stage(f"breadth_{rnd}", (lambda s, k=rnd: s_multivenue(s, offset=k, accumulate=True)),
                  BREADTH_H + 0.3, smoke)
        if (r or {}).get("new_this_round", 0) == 0:
            dry += 1
            if dry >= 4:
                log("  tesisler tükendi -> breadth bitti, depth'e geçiliyor"); break
        else:
            dry = 0
    # === DEPTH (ikincil, kısa): tam-oyun 12dk export (tiling) + analiz ===
    stage("fullgame_export", s_export, DLexp, smoke)
    stage("fullgame_analysis", s_analysis, DLan, smoke)
    # === TOP-UP: TABAN süreye kadar garanti (sims; network'süz, her zaman iş var) ===
    r2 = 0
    while hrs() < FLOOR and r2 < 80:
        r2 += 1
        stage(f"topup_sims_{r2}", s_sims, FLOOR + 0.5, smoke)
    log(f"TABAN süreye ulaşıldı ({hrs():.2f}h ≥ {FLOOR}h); breadth turu={rnd}")
    RESULTS_D["finished_epoch"] = time.time()
    RESULTS_D["total_hours"] = round(hrs(), 2)
    RESULTS.write_text(json.dumps(RESULTS_D, indent=2))
    log(f"runner BİTTİ — toplam {hrs():.2f}h. Özet: "
        + ", ".join(f"{k}={v['status']}" for k, v in RESULTS_D["stages"].items()))


if __name__ == "__main__":
    main()
