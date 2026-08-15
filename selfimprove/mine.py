"""mine.py — düzeltici KORPUS üretimi (recall objective için).

diagnose 'recall' dediğinde: track-doğrulanmış pseudo-label korpusunu (çok-tesis,
tesis-split) üretir/yeniler. Far-band vurgusu açıksa uzak/küçük kutuların payını
artıracak şekilde örnekler (build_dataset zaten tiling-kurtarılan far kutuları
dahil ediyor; burada SADECE freshness + emphasis sarmalıyoruz).

DÜRÜSTLÜK: mine HİÇBİR iyileşme iddia etmez. Sadece etiket sayar + tesis-split
verir. İyileşme yalnız gate'ten (held-out) okunur. CPU-only (frame decode cv2).
İdempotent: korpus güncelse yeniden-decode yapmaz.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

RAW_DIR = ROOT / "self_training/data/raw"
COCO_DIR = ROOT / "self_training/data/coco"


def _discover_venues() -> list[dict]:
    """self_training/data/raw içinde (tracks_<venue>.parquet + <venue>.mp4) çiftleri."""
    out = []
    for tp in sorted(RAW_DIR.glob("tracks_*.parquet")):
        venue = tp.stem[len("tracks_"):]
        vid = RAW_DIR / f"{venue}.mp4"
        if vid.exists():
            out.append({"name": venue, "tracks": str(tp), "video": str(vid)})
    # cankaya aktif-oyun ayrı dizinde
    ca_tr = ROOT / "stats_out/active_game/tracks_active_game_presplit_player_bridged.parquet"
    ca_vid = ROOT / "raw/_active_game.mp4"
    if ca_tr.exists() and ca_vid.exists():
        out.insert(0, {"name": "cankaya_active", "tracks": str(ca_tr), "video": str(ca_vid)})
    return out


def _corpus_fresh() -> bool:
    """stats.json mevcut + en yeni ham-track'ten daha yeni mi?"""
    stats = COCO_DIR / "stats.json"
    if not stats.exists():
        return False
    cm = stats.stat().st_mtime
    raws = list(RAW_DIR.glob("tracks_*.parquet"))
    if not raws:
        return True
    return cm >= max(r.stat().st_mtime for r in raws)


def far_band_corpus(emphasize_far: bool = True, stride: int = 20,
                    val_venues: list[str] | None = None, force: bool = False) -> dict:
    """Düzeltici korpusu üret/yenile (recall objective).

    Far-band vurgusu için uzak kutu-yoğun karelerde stride'ı yarıya indirir
    (çeşitlilik + zor-örnek payı). force=False ve korpus güncelse decode atlanır.
    Döndürür: korpus manifesti {n_images, n_boxes, train_venues, val_venues, far_analysis}.
    """
    if not force and _corpus_fresh():
        st = json.loads((COCO_DIR / "stats.json").read_text())
        return {"status": "fresh_reused", "corpus_dir": str(COCO_DIR),
                "n_images": st.get("total_images"), "n_boxes": st.get("total_boxes"),
                "train_venues": st.get("train_venues"), "val_venues": st.get("val_venues"),
                "far_analysis": st.get("far_analysis"),
                "note": "korpus güncel; yeniden decode YAPILMADI (idempotent)"}

    from self_training.build_dataset import build
    sources = _discover_venues()
    if not sources:
        return {"status": "no_sources", "corpus_dir": str(COCO_DIR)}
    # held-out val: blind-GT'siz bir tesis (overfit-guard); verilmezse sonuncusu.
    vv = val_venues or (["KıbrısDorukHalıS"] if any(s["name"] == "KıbrısDorukHalıS"
                                                    for s in sources) else None)
    t0 = time.time()
    st = build(sources, str(COCO_DIR), stride=(stride // 2 if emphasize_far else stride),
               conf_thr=0.3, val_venues=vv)
    return {"status": "rebuilt", "corpus_dir": str(COCO_DIR),
            "n_images": st.get("total_images"), "n_boxes": st.get("total_boxes"),
            "train_venues": st.get("train_venues"), "val_venues": st.get("val_venues"),
            "far_analysis": st.get("far_analysis"),
            "elapsed_s": round(time.time() - t0, 1), "emphasize_far": emphasize_far}


detection_corpus = far_band_corpus  # diagnose 'mine.detection_corpus' eşanlamı


# =========================================================================== #
# label_queue — aktif-öğrenme GT-talebi (gate_ready blokerini kıran tek şey)
# =========================================================================== #
# Track B (RunPod fine-tune) gate'i >=2 saha blind-GT olmadan KİLİTLİ. O kilidi
# açmanın TEK yolu 2. saha için insan-GT toplamak. Naif "rastgele kare etiketle"
# israf; bunun yerine EN BELİRSİZ kareleri (deficit + far-yoğun + tek-parite) öne
# koy -> sınırlı insan bütçesi en çok bilgi taşıyan karelere harcanır.
def label_queue(parity_parquet: str | None = None, det_cache=None,
                budget: int = 20, venue: str = "KıbrısDorukHalıS") -> dict:
    """2. saha için insan-etiketleme kuyruğu üret (parity-belirsizlik sıralı).

    parity_parquet : tracks_<venue>.parquet (yoksa self_training/data/raw'dan venue).
    det_cache      : (ops; imza uyumu) — sıralama parite/far/conf'tan; kutu-cache şart değil.
    budget         : kaç kare istenecek (insan bütçesi).
    Döndürür: {venue, n_frames, queue:[{frame,deficit,far_count,odd,score}], note}.
    """
    import pandas as pd
    from detect.recall_qc import parity_qc_from_parquet
    p = Path(parity_parquet) if parity_parquet else (RAW_DIR / f"tracks_{venue}.parquet")
    if not p.exists():
        return {"status": "no_tracks", "venue": venue, "looked": str(p),
                "note": "2. saha track parquet yok; ham video -> export_tracks gerekir"}
    df = pd.read_parquet(p)
    qc = parity_qc_from_parquet(df)
    cnt = qc["per_frame"]
    # far-yoğunluk per-frame (recall_qc zone): foot_y üst-tercil
    from detect.recall_qc import _zone_thresholds, zone_of
    t1, t2 = _zone_thresholds(df["foot_y"].to_numpy(float))
    z = zone_of(df["foot_y"].to_numpy(float), t1, t2)
    far_by_frame = (df.assign(_far=(z == 0)).groupby("frame")["_far"].sum())
    conf_by_frame = df.groupby("frame")["conf"].median() if "conf" in df else None

    rows = []
    for f, n in cnt.items():
        deficit = max(0, 14 - int(n))
        far_c = int(far_by_frame.get(f, 0))
        odd = int(int(n) % 2 == 1)
        low_conf = float(conf_by_frame.get(f, 1.0)) if conf_by_frame is not None else 1.0
        # belirsizlik skoru: eksik oyuncu + far-yoğunluk + tek-parite + düşük-conf
        score = deficit * 2.0 + far_c * 0.5 + odd * 1.0 + (1.0 - low_conf) * 3.0
        rows.append({"frame": int(f), "deficit": deficit, "far_count": far_c,
                     "odd": bool(odd), "median_conf": round(low_conf, 3),
                     "score": round(score, 3)})
    rows.sort(key=lambda r: -r["score"])
    # çeşitlilik: ardışık-yakın kareleri seyrelt (>= ~1s ara)
    picked, last = [], {}
    for r in rows:
        seg = r["frame"] // 25
        if seg in last:
            continue
        last[seg] = True
        picked.append(r)
        if len(picked) >= budget:
            break
    return {"status": "ok", "venue": venue, "tracks": str(p),
            "n_frames": qc["n_frames"], "pct_deficit": qc["pct_deficit"],
            "budget": budget, "queue": picked,
            "note": f"{venue} için {len(picked)} belirsiz-kare etiketle "
                    "(3-kör-sayıcı blind-GT) -> gt_venues'a ekle -> Track B gate açılır"}
