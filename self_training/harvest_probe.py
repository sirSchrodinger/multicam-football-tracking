#!/usr/bin/env python3
"""harvest_probe.py — AKTİF-oyun ön-filtreli çok-tesis harvest (GPU verimli).

Sorun: sabit -ss offset sık sık BOŞ saha karesine düşüyor (booking arası /
maç-öncesi). Boş klipte 7dk'lık tam export = GPU israfı.

Çözüm: her aday tesiste önce ~6 kareye RF-DETR at (ucuz, ~2s), oyuncu sayısı
>= MIN_PLAYERS ise AKTİF say -> SADECE aktif tesislerde tam export çalıştır.
Model bir kez yüklenir (probe + export aynı süreçte).

skip listesi: zaten indirilmiş/işlenmiş tesisler.
Çıktı: aktif tesisler için self_training/data/raw/{<name>.mp4, tracks_<name>.parquet}
       + scratchpad/wf_dataset/probe_manifest.json
"""
import sys, os, json, time, subprocess
from pathlib import Path
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, "eval")
from overnight_runner import discover_venues  # noqa
import export_tracks as et  # noqa

RAW = Path("self_training/data/raw"); RAW.mkdir(parents=True, exist_ok=True)
REPORT = Path(os.environ.get("HARVEST_REPORT_DIR", "scratchpad/wf_dataset"))
N_WANT = int(os.environ.get("N_WANT", "4"))       # hedef AKTİF yeni tesis
DUR = int(os.environ.get("CLIP_DUR", "70"))
SS = int(os.environ.get("CLIP_SS", "900"))        # 15dk -> maç ortası daha olası
MAXF = int(os.environ.get("MAX_FRAMES", "1400"))
MIN_PLAYERS = int(os.environ.get("MIN_PLAYERS", "5"))
DATES = ["26.06.2026", "24.06.2026", "22.06.2026", "21.06.2026", "19.06.2026",
         "17.06.2026", "15.06.2026", "13.06.2026"]
# zaten elimizdekiler (tekrar harvest etme)
SKIP_NAMES = set((os.environ.get("SKIP", "")).split(",")) | {
    "KıbrısDorukHalıS", "SporlandHalıSaha", "AydınoğluHalıSah", "Atakum3MSporiumH",
    "AtlantikOsmangaz", "AyazmaİMKBHalıSa", "RizeArenaHalıSah", "DarıcaOlimpiaHal",
    "GardenParkHalıSa", "cankaya_active"}

t0 = time.time()
def log(m): print(f"[{(time.time()-t0)/60:.1f}dk] {m}", flush=True)
def norm(v): return "".join(c for c in v if c.isalnum())[:16]

def download(url, mp4):
    rc = ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]
    for ss in (SS, 300):
        if mp4.exists(): mp4.unlink()
        try:
            subprocess.run(["ffmpeg", "-y", *rc, "-ss", str(ss), "-t", str(DUR),
                            "-i", url, "-c", "copy", str(mp4)],
                           capture_output=True, timeout=900)
        except Exception:
            continue
        if mp4.exists() and mp4.stat().st_size > 2e6:
            return True
    return False

def main():
    import cv2
    from PIL import Image
    from rfdetr import RFDETRLargeDeprecated
    model = RFDETRLargeDeprecated(
        pretrain_weights="models/weights/checkpoint_best_regular.pth",
        device="cuda", num_classes=4)
    THRESH = float(os.environ.get("THRESH", "0.5")); MIN_H = 25

    def probe_active(mp4):
        cap = cv2.VideoCapture(str(mp4)); N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        counts = []
        for fi in np.linspace(30, max(N - 5, 31), 6).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, fr = cap.read()
            if not ok: continue
            det = model.predict(Image.fromarray(fr[:, :, ::-1]), threshold=THRESH)
            xy = np.asarray(det.xyxy, float).reshape(-1, 4)
            h = (xy[:, 3] - xy[:, 1]) if len(xy) else np.array([])
            counts.append(int((h > MIN_H).sum()))
        cap.release()
        return int(np.median(counts)) if counts else 0, counts

    venues = discover_venues(N_WANT + 14, DATES, skip=set())
    log(f"{len(venues)} aday: {list(venues)}")
    manifest = []; n_active = 0
    for venue, url in venues.items():
        name = norm(venue)
        if name in SKIP_NAMES:
            continue
        mp4 = RAW / f"{name}.mp4"
        if not download(url, mp4):
            log(f"{venue}: indirilemedi (s1/non-faststart?) -> atla")
            manifest.append(dict(venue=venue, name=name, status="download_fail", url=url))
            continue
        med, counts = probe_active(mp4)
        if med < MIN_PLAYERS:
            log(f"{venue}: PASİF (median {med} oyuncu, counts={counts}) -> sil")
            manifest.append(dict(venue=venue, name=name, status="passive",
                                 median_players=med, counts=counts))
            mp4.unlink(); continue
        log(f"{venue}: AKTİF (median {med}) -> tam export...")
        out = RAW / f"tracks_{name}.parquet"
        try:
            te = time.time()
            et.run_tracking_export(str(mp4), out_path=str(out), camera_id=name,
                                   calib_path=None, max_frames=MAXF, recover_far=False)
            n_active += 1
            manifest.append(dict(venue=venue, name=name, status="active",
                                 median_players=med, counts=counts,
                                 minutes=round((time.time() - te) / 60, 1)))
            log(f"{venue}: export OK [{n_active}/{N_WANT}]")
        except Exception as e:
            log(f"{venue}: export HATA {repr(e)[:120]}")
            manifest.append(dict(venue=venue, name=name, status="export_error",
                                 median_players=med, err=repr(e)[:160]))
        if n_active >= N_WANT:
            break
    (REPORT / "probe_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    log(f"BİTTİ — {n_active} aktif yeni tesis export edildi")

if __name__ == "__main__":
    main()
