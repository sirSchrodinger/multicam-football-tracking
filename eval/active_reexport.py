#!/usr/bin/env python3
"""CEPHE3 — aktif-pencere (66905-67451) FULL re-export: base(cache) + far-tile + CLAHE-far-tile.

Spawn kökü: det_cache BASE-ONLY (tiling/CLAHE yok) -> uzak-bant oyuncular kare
kaçınca kaybolur. Bu script aktif-pencerenin TÜM 547 karesini yeniden export eder:

  base   : det_cache.parquet (RF-DETR thr=0.30, MIN_H=25, NMS=0.6) — GPU atla, cache
  +tile  : far-band (rows 110-365) x3 upscale -> re-detect -> base'e IoU-dedup
           (eval/recall_eval._tile_far: foot<28px VEYA IoU>0.3 VEYA merkez-içinde)
  +CLAHE : aynı far-tile ama CLAHE-L (clip=3, 8x8) ön-işlenmiş kare üstünde; base
           VE raw-tile recoveries'e karşı dedup.  (Global CLAHE-base ELENDI: 8-kare
           ölçümünde far 73->70 düşürüyor; CLAHE SADECE additive far-tile için.)

Çıktı: scratchpad/wf2_recall/active_tiled.parquet  (frame,x1,y1,x2,y2,conf,foot_x,foot_y,src)
       scratchpad/wf2_recall/active_reexport_stats.json  (önce/sonra spawn proxy)
       scratchpad/wf2_recall/active_reexport_overlay_f*.png  (görsel kanıt)

DÜRÜSTLÜK: bu DETEKSIYON-seviye kurtarma; oyuncu/roster atfı downstream'in işi.
recall %% insan-GT'ye karşı eval/recall_eval score (recall_val/score_summary) ile;
burada raporlanan kare-başı sayı/varyans = spawn proxy, GT değil.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np, cv2, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
import recall_eval as R

# İNSAN-GT'nin doğruladığı config = recall_eval defaultları (TILE_UP=2.0,
# TILE_THRESH=0.40, aspect-filtre YOK). O config 8-karede +tile far-recall
# 68.5->79.3%% + precision 100%% (fp=0) verdi. Daha sıkı (TILE_UP=3.0+aspect)
# bir sapma denedim ama daha AZ kurtarıyor ve GT-doğrulanmamış -> validated'a dön.
# (R.TILE_UP override YOK; aspect-filtre KALDIRILDI.)
FAR = 365.0
DEDUP_CLAHE_PX = 22.0
F0, F1 = 66905, 67451
OUT = ROOT / "scratchpad/wf2_recall"
OUT.mkdir(parents=True, exist_ok=True)
OVERLAY_FRAMES = {66905, 67005, 67153, 67305}


def clahe_bgr(img, clip=3.0, grid=8):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l)
    return cv2.cvtColor(cv2.merge([cl, a, b]), cv2.COLOR_LAB2BGR)


def foots(xy):
    return np.stack([(xy[:, 0] + xy[:, 2]) / 2, xy[:, 3]], 1) if len(xy) else np.empty((0, 2))


def main():
    t_start = time.time()
    dc = pd.read_parquet(ROOT / "scratchpad/det_cache.parquet")
    model = R._load_model()
    cap = cv2.VideoCapture(str(ROOT / "raw/cankaya_cam2.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, F0)

    out_rows = []
    stats = []
    overlays = {}
    n_frames = F1 - F0 + 1
    for k, fi in enumerate(range(F0, F1 + 1)):
        ok, fr = cap.read()
        if not ok:
            print("read FAIL", fi); break
        bsub = dc[dc.frame == fi]
        base_xy = bsub[["x1", "y1", "x2", "y2"]].to_numpy(float)
        base_cf = bsub["conf"].to_numpy(float)

        # raw far-tile (validated config; _tile_far içi IoU-dedup vs base)
        t_xy, t_cf = R._tile_far(model, fr, base_xy)

        # CLAHE far-tile, dedup vs base AND vs raw-tile recoveries
        frC = clahe_bgr(fr)
        tc_xy, tc_cf = R._tile_far(model, frC, base_xy)
        if len(tc_xy):
            tfoot = foots(t_xy)
            keep = []
            tcf = foots(tc_xy)
            for i in range(len(tc_xy)):
                if len(tfoot) and np.linalg.norm(tfoot - tcf[i], axis=1).min() < DEDUP_CLAHE_PX:
                    continue
                keep.append(i)
            keep = np.array(keep, int)
            tc_xy, tc_cf = (tc_xy[keep], tc_cf[keep]) if len(keep) else (tc_xy[:0], tc_cf[:0])

        # write base + recoveries
        for a, c in zip(base_xy, base_cf):
            out_rows.append((fi, *a, c, (a[0] + a[2]) / 2, a[3], "base"))
        for a, c in zip(t_xy, t_cf):
            out_rows.append((fi, *a, c, (a[0] + a[2]) / 2, a[3], "tile"))
        for a, c in zip(tc_xy, tc_cf):
            out_rows.append((fi, *a, c, (a[0] + a[2]) / 2, a[3], "clahe_tile"))

        base_far = int((base_xy[:, 3] < FAR).sum()) if len(base_xy) else 0
        rec_far = len(t_xy) + len(tc_xy)
        stats.append(dict(frame=fi, base_total=len(base_xy), base_far=base_far,
                          tile_rec=len(t_xy), clahe_rec=len(tc_xy),
                          reexport_total=len(base_xy) + rec_far,
                          reexport_far=base_far + rec_far))

        if fi in OVERLAY_FRAMES:
            vis = fr.copy()
            for a in base_xy:
                cv2.rectangle(vis, (int(a[0]), int(a[1])), (int(a[2]), int(a[3])), (0, 255, 0), 2)
            for a in t_xy:
                cv2.rectangle(vis, (int(a[0]), int(a[1])), (int(a[2]), int(a[3])), (0, 255, 255), 2)
            for a in tc_xy:
                cv2.rectangle(vis, (int(a[0]), int(a[1])), (int(a[2]), int(a[3])), (255, 0, 255), 2)
            cv2.rectangle(vis, (0, 110), (1919, int(FAR)), (255, 0, 0), 1)
            cv2.putText(vis, f"f{fi}  green=base({len(base_xy)})  yellow=tile(+{len(t_xy)})  magenta=CLAHE-tile(+{len(tc_xy)})",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            p = OUT / f"active_reexport_overlay_f{fi}.png"
            cv2.imwrite(str(p), vis); overlays[fi] = str(p)

        if k % 50 == 0:
            print(f"[{k}/{n_frames}] f{fi} base={len(base_xy)} +tile={len(t_xy)} +clahe={len(tc_xy)} ({time.time()-t_start:.0f}s)")
    cap.release()

    cols = ["frame", "x1", "y1", "x2", "y2", "conf", "foot_x", "foot_y", "src"]
    df = pd.DataFrame(out_rows, columns=cols)
    df.to_parquet(OUT / "active_tiled.parquet", index=False)

    sdf = pd.DataFrame(stats)
    # spawn proxy: per-frame total count, before (base/cache) vs after (reexport)
    def stat(s):
        return dict(mean=round(float(s.mean()), 2), std=round(float(s.std()), 2),
                    min=int(s.min()), max=int(s.max()), var=round(float(s.var()), 2))
    summary = dict(
        n_frames=int(len(sdf)),
        before_total=stat(sdf.base_total),
        after_total=stat(sdf.reexport_total),
        before_far=stat(sdf.base_far),
        after_far=stat(sdf.reexport_far),
        recovered=dict(tile_total=int(sdf.tile_rec.sum()),
                       clahe_total=int(sdf.clahe_rec.sum()),
                       per_frame_mean=round(float((sdf.tile_rec + sdf.clahe_rec).mean()), 3),
                       frames_with_recovery=int(((sdf.tile_rec + sdf.clahe_rec) > 0).sum())),
        rows=dict(base=int((df.src == "base").sum()),
                  tile=int((df.src == "tile").sum()),
                  clahe_tile=int((df.src == "clahe_tile").sum())),
        runtime_s=round(time.time() - t_start, 1),
        overlays=overlays)
    (OUT / "active_reexport_stats.json").write_text(json.dumps(
        dict(summary=summary, per_frame=stats), indent=2))
    print("\n==== SPAWN PROXY (per-frame detection count, active window) ====")
    print("BEFORE (base/cache) total:", summary["before_total"])
    print("AFTER  (+tile+CLAHE) total:", summary["after_total"])
    print("BEFORE far:", summary["before_far"])
    print("AFTER  far:", summary["after_far"])
    print("recovered:", summary["recovered"])
    print("wrote", OUT / "active_tiled.parquet", "and active_reexport_stats.json")


if __name__ == "__main__":
    main()
