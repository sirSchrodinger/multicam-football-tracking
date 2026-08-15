#!/usr/bin/env python3
"""harvest_export.py — indirilen YENİ tesis kliplerinde base tracking export.

download_manifest.json'daki ok=True tesisleri okur, her biri için
export_tracks.run_tracking_export çalıştırır. recover_far=False çünkü YENİ
tesiste per-venue kalibrasyon YOK (FarBandRecovery in_pitch filtresi yanlış
geometriyle güvenilmez). pitch kolonları NaN olur (build_dataset require_in_pitch
=False kullanıyor, sorun değil). max_frames ile GPU süresi sınırlanır.

Çıktı: self_training/data/raw/tracks_<name>.parquet (her tesis için).
"""
import sys, os, json, time
from pathlib import Path
sys.path.insert(0, ".")
import export_tracks as et  # noqa: E402

RAW = Path("self_training/data/raw")
REPORT = Path(os.environ.get("HARVEST_REPORT_DIR", "scratchpad/wf_dataset"))
MAXF = int(os.environ.get("MAX_FRAMES", "1500"))   # GPU bütçesi (~50s @30fps)

t0 = time.time()
def log(m): print(f"[{(time.time()-t0)/60:.1f}dk] {m}", flush=True)

def main():
    man_path = REPORT / "download_manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else []
    todo = [m for m in manifest if m.get("ok")]
    log(f"{len(todo)} tesis export edilecek (max_frames={MAXF})")
    results = []
    for m in todo:
        name = m["name"]
        mp4 = RAW / f"{name}.mp4"
        out = RAW / f"tracks_{name}.parquet"
        if not mp4.exists():
            log(f"{name}: mp4 yok, atla"); continue
        if out.exists():
            log(f"{name}: tracks zaten var, atla")
            results.append(dict(name=name, status="cached", tracks=str(out)))
            continue
        try:
            log(f"{name}: export başlıyor (recover_far=False, no-calib)...")
            te = time.time()
            p = et.run_tracking_export(str(mp4), out_path=str(out),
                                       camera_id=name, calib_path=None,
                                       max_frames=MAXF, recover_far=False)
            results.append(dict(name=name, status="ok", tracks=p,
                                minutes=round((time.time()-te)/60, 1)))
            log(f"{name}: OK -> {p} ({(time.time()-te)/60:.1f}dk)")
        except Exception as e:
            log(f"{name}: HATA {repr(e)[:160]}")
            results.append(dict(name=name, status="error", err=repr(e)[:200]))
    (REPORT / "export_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False))
    oks = [r for r in results if r["status"] in ("ok", "cached")]
    log(f"BİTTİ — {len(oks)}/{len(results)} export başarılı")

if __name__ == "__main__":
    main()
