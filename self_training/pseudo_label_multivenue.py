import os
#!/usr/bin/env python3
"""Faz 2 — çok-tesis pseudo-label: RF-DETR-ft + GENEL üst-bant tiling -> COCO + görsel.

Mevcut eval/recall_eval detect+tiling mantığını yeniden kullanır ama (a) checkpoint_ft.pth,
(b) Çankaya-özel FAR_BAND yerine GENEL üst-%45 band tiling (her tesis geometrisine uyar).
Çıktı: self_training/data/pseudo/ (coco ann + montaj). Bu set RunPod self-train'i besler.
DÜRÜST: pseudo-label = mükemmel değil; far-tiling base'in kaçırdığı küçük/uzak oyuncuyu
ekler (domain-çeşitliliği + zor-pozitif). FP riski conf+NMS+boyut ile sınırlı.
"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, cv2
from pathlib import Path

THRESH, MIN_H, NMS, TILE_UP, TILE_THR, DEDUP = 0.30, 25, 0.6, 2.0, 0.40, 28.0
WEIGHTS = "models/weights/checkpoint_ft.pth"
OUT = Path("self_training/data/pseudo"); OUT.mkdir(parents=True, exist_ok=True)
(OUT / "vis").mkdir(exist_ok=True)


def load_model():
    from rfdetr import RFDETRLargeDeprecated
    return RFDETRLargeDeprecated(pretrain_weights=WEIGHTS, device="cuda", num_classes=4)


def predict(model, bgr, thr):
    from PIL import Image
    det = model.predict(Image.fromarray(bgr[:, :, ::-1]), threshold=thr)
    return (np.asarray(det.xyxy, float).reshape(-1, 4),
            np.asarray(det.confidence, float).reshape(-1))


def base_detect(model, bgr):
    import supervision as sv
    xy, cf = predict(model, bgr, THRESH)
    if len(xy):
        k = (xy[:, 3] - xy[:, 1]) > MIN_H; xy, cf = xy[k], cf[k]
    if not len(xy):
        return np.empty((0, 4)), np.empty((0,))
    d = sv.Detections(xyxy=xy, confidence=cf, class_id=np.zeros(len(xy), int)).with_nms(
        threshold=NMS, class_agnostic=True)
    return np.asarray(d.xyxy, float).reshape(-1, 4), np.asarray(d.confidence, float)


def tile_far_generic(model, bgr, base_xy, band_frac=0.45):
    """Üst band_frac (genel: uzak oyuncular kameradan-uzak = kare üstü) x2 upscale re-detect."""
    H = bgr.shape[0]; y0, y1 = int(0.04 * H), int(band_frac * H)
    big = cv2.resize(bgr[y0:y1], None, fx=TILE_UP, fy=TILE_UP, interpolation=cv2.INTER_CUBIC)
    xy, cf = predict(model, big, TILE_THR)
    if not len(xy):
        return np.empty((0, 4)), np.empty((0,))
    xy = xy.copy(); xy[:, [0, 2]] /= TILE_UP; xy[:, [1, 3]] = xy[:, [1, 3]] / TILE_UP + y0
    k = (xy[:, 3] - xy[:, 1]) > MIN_H; xy, cf = xy[k], cf[k]
    new_xy, new_cf = [], []
    for i in range(len(xy)):
        tb = xy[i]; tcx, tcy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2; dup = False
        for b in base_xy:
            if b[0] <= tcx <= b[2] and b[1] <= tcy <= b[3]:
                dup = True; break
            fx = (b[0] + b[2]) / 2
            if abs(fx - tcx) < DEDUP and abs(b[3] - tb[3]) < DEDUP:
                dup = True; break
        if not dup:
            new_xy.append(tb); new_cf.append(cf[i])
    return (np.array(new_xy).reshape(-1, 4), np.array(new_cf).reshape(-1))


def main():
    vids = sorted(Path("self_training/data/raw").glob("*.mp4"))
    model = load_model()
    images, annos = [], []
    img_id, ann_id = 0, 0
    summary = []
    for vp in vids:
        venue = vp.stem
        cap = cv2.VideoCapture(str(vp))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1500
        picks = np.linspace(int(0.06 * n), int(0.94 * n), 32).astype(int)
        nb_tot, nt_tot, n_active = 0, 0, 0
        montage = None
        for j, fi in enumerate(picks):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi)); ok, fr = cap.read()
            if not ok:
                continue
            bxy, bcf = base_detect(model, fr)
            txy, tcf = tile_far_generic(model, fr, bxy)
            allxy = np.vstack([bxy, txy]) if len(txy) else bxy
            allcf = np.concatenate([bcf, tcf]) if len(txy) else bcf
            if len(allxy) < 8:        # AKTİF-OYUN filtresi: boş/ısınma karelerini at
                continue
            n_active += 1
            nb_tot += len(bxy); nt_tot += len(txy)
            H, W = fr.shape[:2]
            fn = f"{venue}_f{int(fi)}.jpg"
            cv2.imwrite(str(OUT / "vis" / fn), fr)
            images.append(dict(id=img_id, file_name=fn, width=W, height=H, venue=venue))
            for k in range(len(allxy)):
                x1, y1, x2, y2 = allxy[k]
                annos.append(dict(id=ann_id, image_id=img_id, category_id=1,
                                  bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                                  area=float((x2 - x1) * (y2 - y1)), iscrowd=0,
                                  score=float(allcf[k]),
                                  src="tile" if k >= len(bxy) else "base"))
                ann_id += 1
            if j == 5:  # orta kareyi görselle
                vis = fr.copy()
                for b in bxy:
                    cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 220, 0), 2)
                for b in txy:
                    cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 165, 255), 2)
                cv2.rectangle(vis, (0, 0), (W, 30), (0, 0, 0), -1)
                cv2.putText(vis, f"{venue}  base(yesil)={len(bxy)} +tile(turuncu)={len(txy)}",
                            (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                montage = cv2.resize(vis, (640, 360))
            img_id += 1
        if montage is not None:
            cv2.imwrite(str(OUT / f"vis_{venue}.jpg"), montage)
        summary.append((venue, nb_tot, nt_tot))
        cap.release()
        print(f"[{venue}] aktif={n_active}/32 kare, base={nb_tot} +tile={nt_tot}", flush=True)
    coco = dict(images=images, annotations=annos,
                categories=[dict(id=1, name="player")])
    json.dump(coco, open(OUT / "pseudo_coco.json", "w"))
    tb = sum(s[1] for s in summary); tt = sum(s[2] for s in summary)
    print(f"\nTOPLAM: {len(images)} kare, {len(annos)} kutu (base {tb} + tile {tt}, "
          f"tile-kazanç +%{100*tt/max(tb,1):.0f}). -> {OUT}/pseudo_coco.json")


if __name__ == "__main__":
    main()
