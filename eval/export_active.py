"""Aktif-oyun segmentini (cam2 t=2300-2800) far-band tiling'le export + pipeline.
Hem aktivite hem recall limiter'ını çözer. Arka plan GPU işi (~2-3h)."""
import subprocess, sys, time, json
sys.path.insert(0,".")
from pathlib import Path
t0=time.time()
def log(m): print(f"[{(time.time()-t0)/60:.1f}dk] {m}",flush=True)
aw=Path("raw/_active_game.mp4")
if not aw.exists():
    log("aktif segment kesiliyor (t=2300, 500s)...")
    subprocess.run(["ffmpeg","-y","-ss","2300","-t","500","-i","raw/cankaya_cam2.mp4","-c","copy",str(aw)],check=True,capture_output=True)
log("export_tracks --recover-far (tiling)...")
import export_tracks as et
out=et.run_tracking_export(str(aw),calib_path="calib/cankaya_cam2_FINAL.json",camera_id="cankaya_cam2",
                           out_path="raw/tracks_active_game.parquet",recover_far=True)
log(f"export bitti -> {out}")
log("run_pipeline (scale-height)...")
r=subprocess.run(["venv/bin/python","scripts/run_pipeline.py","raw/tracks_active_game.parquet",
                  "calib/cankaya_cam2_FINAL.json","--scale-height","--expected-players","14"],
                 capture_output=True,text=True,timeout=3600)
Path("overnight_out/active_export_log.txt").write_text(r.stdout+"\n---\n"+r.stderr)
log(f"pipeline rc={r.returncode}")
log("BİTTİ — replay yeniden render'a hazır (stats_out/tracks_active_game/)")
