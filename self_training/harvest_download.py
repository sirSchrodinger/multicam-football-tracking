#!/usr/bin/env python3
"""harvest_download.py — çok-tesis GECE klip indirici (GPU'suz, sadece ağ).

discover_venues ile sosyalhalisaha tesislerini enumerate eder, her YENİ tesisten
~70s gece klip indirir. CDN'deki bazı mp4'ler faststart DEĞİL (moov sonda) ->
ffmpeg moov-seek'te 404 alır; bunları PROBE edip atlar (per-file, evrensel değil).

Çıktı: self_training/data/raw/<name>.mp4  (+ download_manifest.json)
GPU export AYRI (export_tracks). Bu script sadece indirir + doğrular.
"""
import sys, os, json, time, subprocess, re
from pathlib import Path
sys.path.insert(0, ".")
sys.path.insert(0, "eval")
from overnight_runner import discover_venues  # noqa: E402

RAW = Path("self_training/data/raw")
RAW.mkdir(parents=True, exist_ok=True)
REPORT = Path(os.environ.get("HARVEST_REPORT_DIR", "scratchpad/wf_dataset"))
REPORT.mkdir(parents=True, exist_ok=True)

# zaten elimizde olan tesisler (yeniden indirme): isim-normalize ile eşle
HAVE = {"KıbrısDorukHalıS", "cankaya_active"}
# Darıca daha önce 8s bozuk indi -> yeniden dene (HAVE'e koymuyoruz)

N_WANT = int(os.environ.get("N_WANT", "6"))   # hedef YENİ tesis sayısı
DUR = int(os.environ.get("CLIP_DUR", "70"))   # klip uzunluğu (s)
SS = int(os.environ.get("CLIP_SS", "600"))    # maç içine atla (s) -> orta-oyun, far aktif
DATES = ["27.06.2026", "25.06.2026", "23.06.2026", "20.06.2026",
         "18.06.2026", "16.06.2026", "14.06.2026"]

t0 = time.time()
def log(m): print(f"[{(time.time()-t0)/60:.1f}dk] {m}", flush=True)

def norm(v): return "".join(c for c in v if c.isalnum())[:16]

def probe(mp4):
    """(ok, dur_s, size_mb). ok = ffprobe okunabilir + dur>=DUR*0.6 + size>2MB."""
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "format=duration", "-of", "csv=p=0", str(mp4)],
                           capture_output=True, timeout=30, text=True)
        dur = float(r.stdout.strip() or 0)
    except Exception:
        dur = 0.0
    size_mb = mp4.stat().st_size / 1e6 if mp4.exists() else 0.0
    ok = dur >= DUR * 0.6 and size_mb > 2.0
    return ok, dur, size_mb

def try_download(url, mp4):
    """faststart -> -ss önce kopya. Başarısızsa baştan kopya dener. Probe ile doğrular."""
    rc = ["-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5"]
    strategies = [
        ["ffmpeg", "-y", *rc, "-ss", str(SS), "-t", str(DUR), "-i", url, "-c", "copy", str(mp4)],
        ["ffmpeg", "-y", *rc, "-ss", "30", "-t", str(DUR), "-i", url, "-c", "copy", str(mp4)],
    ]
    for ci, cmd in enumerate(strategies):
        if mp4.exists():
            mp4.unlink()
        try:
            subprocess.run(cmd, capture_output=True, timeout=900)
        except Exception as e:
            log(f"    strat{ci} exc {repr(e)[:60]}")
            continue
        ok, dur, mb = probe(mp4)
        if ok:
            return True, dur, mb, ci
    return False, 0.0, 0.0, -1

def main():
    log(f"keşif başlıyor (hedef {N_WANT} yeni tesis)...")
    venues = discover_venues(N_WANT + 8, DATES, skip=set())  # bol aday topla
    log(f"{len(venues)} aday tesis: {list(venues)}")
    manifest = []
    n_new = 0
    for venue, url in venues.items():
        name = norm(venue)
        if name in HAVE:
            log(f"{venue}: zaten var, atla")
            continue
        mp4 = RAW / f"{name}.mp4"
        # önceden başarılı indirilmişse atla
        if mp4.exists():
            ok, dur, mb = probe(mp4)
            if ok:
                log(f"{venue}: mevcut OK ({dur:.0f}s {mb:.1f}MB)")
                manifest.append(dict(venue=venue, name=name, ok=True, dur=dur,
                                     size_mb=mb, url=url, cached=True))
                n_new += 1
                if n_new >= N_WANT:
                    break
                continue
        log(f"{venue}: indiriliyor {url[:70]}")
        ok, dur, mb, strat = try_download(url, mp4)
        manifest.append(dict(venue=venue, name=name, ok=ok, dur=round(dur, 1),
                             size_mb=round(mb, 1), url=url, strategy=strat))
        if ok:
            n_new += 1
            log(f"{venue}: OK {dur:.0f}s {mb:.1f}MB (strat{strat}) [{n_new}/{N_WANT}]")
        else:
            log(f"{venue}: BAŞARISIZ (faststart değil / 404 / ağ) -> atla")
            if mp4.exists():
                mp4.unlink()
        if n_new >= N_WANT:
            break
        time.sleep(0.5)
    (REPORT / "download_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False))
    oks = [m for m in manifest if m["ok"]]
    log(f"BİTTİ — {len(oks)}/{len(manifest)} tesis indirildi: {[m['name'] for m in oks]}")

if __name__ == "__main__":
    main()
