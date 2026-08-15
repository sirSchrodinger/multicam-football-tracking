"""Multi-venue pseudo-label kaynağı: tesis keşfi -> klip indir -> tiling export.
Çıktı: self_training/data/raw/<venue>.mp4 + tracks_<venue>.parquet. Arka plan GPU."""
import sys, time, json, subprocess; sys.path.insert(0,".")
from pathlib import Path
sys.path.insert(0,"eval")
from overnight_runner import discover_venues, _ff_frames  # discovery reuse
import export_tracks as et
t0=time.time(); D=Path("self_training/data/raw"); D.mkdir(parents=True,exist_ok=True)
def log(m): print(f"[{(time.time()-t0)/60:.1f}dk] {m}",flush=True)
DATES=["27.06.2026","25.06.2026","23.06.2026","20.06.2026","18.06.2026"]
venues=discover_venues(4, DATES)  # 4 farklı tesis
log(f"keşfedilen: {list(venues.keys())}")
calib="calib/cankaya_cam2_v2.json"  # NOT: başka tesis için calib yok -> pitch NaN olur ama bbox/track geçerli
done=[]
for venue,url in list(venues.items())[:4]:
    name="".join(c for c in venue if c.isalnum())[:16]
    mp4=D/f"{name}.mp4"; out=str(D/f"tracks_{name}.parquet")
    try:
        if not mp4.exists():
            log(f"{venue}: 90s klip indiriliyor...")
            subprocess.run(["ffmpeg","-y","-reconnect","1","-reconnect_streamed","1","-ss","600","-t","90","-i",url,"-c","copy",str(mp4)],check=True,capture_output=True,timeout=600)
        log(f"{venue}: tiling export...")
        et.run_tracking_export(str(mp4),calib_path=calib,camera_id=name,out_path=out,recover_far=True)
        done.append(name); log(f"{venue}: OK -> {out}")
    except Exception as e:
        log(f"{venue}: HATA {e}")
json.dump(done, open(D/"harvested.json","w"))
log(f"BİTTİ — {len(done)} tesis: {done}")
