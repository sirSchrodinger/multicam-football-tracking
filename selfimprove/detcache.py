"""detcache.py — 1x GPU forward, sonra SAF-CPU config sweep (Track A makinesi).

Öneri 2 çekirdeği. Recall-recovery config araması (recovery_search) çok sayıda
FarBandRecovery knob kombinasyonunu denemek ister; her kombinasyon için yeniden
GPU inference yapmak israf + 4GB 1650Ti'de paralel-cleanup ile yarışır. Çözüm:

  build()  : DONMUŞ frozen kareleri BİR KEZ düşük eşikle (thr=0.15) + far-band crop
             × {up=2.0,2.5} × {clahe off,on} ile decode et -> ham kutu parquet'i.
             GPU; sadece açıkça çağrılınca (GPU müsaitse). İdempotent: cache varsa atlar.
  apply()  : ham cache + cfg -> {frame:{base:[...],tiled:[...]}}. SAF-CPU filtre/NMS/dedup.
             Hiç GPU yok; recovery_search bunu yüzlerce kez çağırabilir.

DÜRÜSTLÜK SINIRI (kritik): apply() bir KUTU üretir; o kutuların insan-GT'ye karşı
recall'i ANCAK nokta-konumlu GT ile ölçülebilir. Mevcut GT sayı-tabanlı (3-kör-sayıcı
n_base_covered/n_tile_covered), kutu-konumlu değil. Bu yüzden:
  - recover_far BAYRAĞI (ON/OFF) sayı-GT'den DOĞRUDAN ölçülür (recovery_search,
    count-GT yolu) -> bugün gate'lenebilir gerçek kazanç.
  - tile_thresh/up/clahe gibi İNCE knob'lar kutu-konumlu GT ister; o gelene dek
    apply() makinesi hazır ama gate'e SOKULMAZ (sahte knob-delta YASAK).

mevcut scratchpad/det_cache.parquet thr=0.30'da (full-game) -> recall-recovery için
ÇOK yüksek floor; bu yüzden AYRI 0.15 frozen-kare cache (out_parquet) tutulur.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# recall_eval.py ile BİRE-BİR base sabitler (üretim ile aynı tespit hattı)
THRESH = 0.30          # üretim base floor
MIN_H = 25
NMS_THRESH = 0.6
FAR_BAND = (110, 365)  # recall_eval.FAR_BAND ile aynı (cam2 geometrisi)
CACHE_THR = 0.15       # build floor: CPU sweep base/tile eşiğini buradan yukarı süzer

FROZEN = [37307, 44769, 52230, 59692, 67153, 74615, 82076, 85807]
FRAMES_DIR = ROOT / "recall_val/frames"
DEFAULT_CACHE = ROOT / "selfimprove/_state/detcache_frozen_thr015.parquet"
DETECTIONS_JSON = ROOT / "recall_val/detections.json"


# --------------------------------------------------------------------------- #
# saf-CPU yardımcılar (apply'da GPU/torch/sv import YOK)
# --------------------------------------------------------------------------- #
def _nms(xy: np.ndarray, cf: np.ndarray, thr: float) -> np.ndarray:
    """class-agnostic greedy NMS; tutulan indeksleri (conf-azalan) döndür."""
    if len(xy) == 0:
        return np.empty(0, int)
    x1, y1, x2, y2 = xy[:, 0], xy[:, 1], xy[:, 2], xy[:, 3]
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = np.argsort(-cf)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        ix1 = np.maximum(x1[i], x1[rest]); iy1 = np.maximum(y1[i], y1[rest])
        ix2 = np.minimum(x2[i], x2[rest]); iy2 = np.minimum(y2[i], y2[rest])
        iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
        inter = iw * ih
        iou = inter / (area[i] + area[rest] - inter + 1e-6)
        order = rest[iou <= thr]
    return np.asarray(keep, int)


def _foot(xy: np.ndarray) -> np.ndarray:
    if len(xy) == 0:
        return np.empty((0, 2))
    return np.stack([(xy[:, 0] + xy[:, 2]) / 2.0, xy[:, 3]], 1)


def _dedup_box(tb, base_xy) -> bool:
    """recall_eval._tile_far._dup_box ile aynı: merkez-içinde veya IoU>0.3 -> kopya."""
    tcx, tcy = (tb[0] + tb[2]) / 2, (tb[1] + tb[3]) / 2
    for b in base_xy:
        if b[0] <= tcx <= b[2] and b[1] <= tcy <= b[3]:
            return True
        ix1, iy1 = max(tb[0], b[0]), max(tb[1], b[1])
        ix2, iy2 = min(tb[2], b[2]), min(tb[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        ua = ((tb[2] - tb[0]) * (tb[3] - tb[1])
              + (b[2] - b[0]) * (b[3] - b[1]) - inter)
        if ua > 0 and inter / ua > 0.3:
            return True
    return False


def _pack(xy, cf):
    return [dict(x1=float(a[0]), y1=float(a[1]), x2=float(a[2]), y2=float(a[3]),
                 foot_x=float((a[0] + a[2]) / 2), foot_y=float(a[3]), conf=float(c))
            for a, c in zip(xy, cf)]


# --------------------------------------------------------------------------- #
# build  (GPU; 1x forward, idempotent)
# --------------------------------------------------------------------------- #
def build(frames_dir: str | Path = FRAMES_DIR, weights: str | None = None,
          thr: float = CACHE_THR, out_parquet: str | Path = DEFAULT_CACHE,
          ups=(2.0, 2.5), clahes=(False, True), force: bool = False) -> str:
    """Frozen kareleri DÜŞÜK eşikle decode et -> ham kutu cache (parquet).

    Her kare için: full-frame base @thr + far-band-crop @thr × {up} × {clahe}.
    KUTULAR ham olarak saklanır (filtre/NMS/dedup apply'da, CPU). GPU gerektirir;
    sadece açıkça çağrılınca koş (paralel-cleanup ile GPU yarışı). force=False ve
    cache varsa atlanır (idempotent). Döndürür: out_parquet yolu.
    """
    import pandas as pd
    out_parquet = Path(out_parquet)
    if out_parquet.exists() and not force:
        return str(out_parquet)

    import cv2
    from PIL import Image
    from rfdetr import RFDETRLargeDeprecated
    w = str(weights or (ROOT / "models/weights/checkpoint_best_regular.pth"))
    model = RFDETRLargeDeprecated(pretrain_weights=w, device="cuda", num_classes=4)

    def _predict(img_bgr, t):
        det = model.predict(Image.fromarray(img_bgr[:, :, ::-1]), threshold=t)
        xy = np.asarray(det.xyxy, float).reshape(-1, 4)
        cf = np.asarray(det.confidence, float).reshape(-1)
        return xy, cf

    def _enhance(crop_bgr, clahe):
        if not clahe:
            return crop_bgr
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    rows = []
    y0, y1 = FAR_BAND
    for fi in FROZEN:
        fp = Path(frames_dir) / f"f{fi}.png"
        if not fp.exists():
            continue
        fr = cv2.imread(str(fp))
        bxy, bcf = _predict(fr, thr)
        for a, c in zip(bxy, bcf):
            rows.append(dict(frame=fi, source="base", up=0.0, clahe=False,
                             x1=a[0], y1=a[1], x2=a[2], y2=a[3], conf=c))
        for up in ups:
            for cl in clahes:
                crop = _enhance(fr[y0:y1].copy(), cl)
                big = cv2.resize(crop, None, fx=up, fy=up, interpolation=cv2.INTER_CUBIC)
                txy, tcf = _predict(big, thr)
                if len(txy):
                    txy = txy.copy()
                    txy[:, [0, 2]] /= up
                    txy[:, [1, 3]] = txy[:, [1, 3]] / up + y0
                for a, c in zip(txy, tcf):
                    rows.append(dict(frame=fi, source="tiled", up=float(up), clahe=bool(cl),
                                     x1=a[0], y1=a[1], x2=a[2], y2=a[3], conf=c))
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out_parquet, index=False)
    return str(out_parquet)


# --------------------------------------------------------------------------- #
# apply  (SAF-CPU; recovery_search yüzlerce kez çağırır)
# --------------------------------------------------------------------------- #
def apply(cache, cfg: dict) -> dict:
    """Ham cache + cfg -> {frame:{base:[...],tiled:[...]}}. GPU YOK.

    cfg knob'ları (FarBandRecovery ile uyumlu):
      base_thr   (default 0.30), tile_thresh (default 0.40), up (default 2.0),
      clahe (default False), dedup_px (default 28.0), min_h (default 25),
      nms (default 0.6), recover_far (default True; False -> tiled boş).
    """
    import pandas as pd
    if isinstance(cache, (str, Path)):
        cache = pd.read_parquet(cache)
    base_thr = cfg.get("base_thr", THRESH)
    tile_thresh = cfg.get("tile_thresh", 0.40)
    up = cfg.get("up", 2.0)
    clahe = cfg.get("clahe", False)
    dedup_px = cfg.get("dedup_px", 28.0)
    min_h = cfg.get("min_h", MIN_H)
    nms = cfg.get("nms", NMS_THRESH)
    recover_far = cfg.get("recover_far", True)

    out = {}
    for fi in sorted(cache["frame"].unique()):
        sub = cache[cache["frame"] == fi]
        b = sub[sub["source"] == "base"]
        bxy = b[["x1", "y1", "x2", "y2"]].to_numpy(float)
        bcf = b["conf"].to_numpy(float)
        if len(bxy):
            m = (bcf >= base_thr) & ((bxy[:, 3] - bxy[:, 1]) > min_h)
            bxy, bcf = bxy[m], bcf[m]
            k = _nms(bxy, bcf, nms)
            bxy, bcf = bxy[k], bcf[k]
        txy, tcf = np.empty((0, 4)), np.empty(0)
        if recover_far:
            t = sub[(sub["source"] == "tiled") & (sub["up"] == up) & (sub["clahe"] == clahe)]
            txy_all = t[["x1", "y1", "x2", "y2"]].to_numpy(float)
            tcf_all = t["conf"].to_numpy(float)
            if len(txy_all):
                m = (tcf_all >= tile_thresh) & ((txy_all[:, 3] - txy_all[:, 1]) > min_h)
                txy_all, tcf_all = txy_all[m], tcf_all[m]
                bfoot = _foot(bxy)
                tfoot = _foot(txy_all)
                keep = []
                for i in range(len(txy_all)):
                    if len(bfoot) and np.min(np.linalg.norm(bfoot - tfoot[i], axis=1)) < dedup_px:
                        continue
                    if len(bxy) and _dedup_box(txy_all[i], bxy):
                        continue
                    keep.append(i)
                keep = np.asarray(keep, int)
                txy, tcf = txy_all[keep], tcf_all[keep]
        out[int(fi)] = {"base": _pack(bxy, bcf), "tiled": _pack(txy, tcf)}
    return out


def from_detections_json(path: str | Path = DETECTIONS_JSON) -> dict:
    """Mevcut recall_val/detections.json'u apply()-uyumlu cache görünümüne çevir.

    Bu ÜRETİM-CONFIG cache'idir (base thr=0.30, tiled thr=0.40, up=2.0/2.5).
    recovery_search'in count-GT yolu kutu-cache'e ihtiyaç duymaz (sayı-GT zaten
    n_base_covered/n_tile_covered tutar); bu fonksiyon cache soyutlamasının
    doğrulanması + ileride kutu-konumlu GT geldiğinde köprü içindir.
    """
    import json
    d = json.loads(Path(path).read_text())
    return {int(k): {"base": v.get("base", []), "tiled": v.get("tiled", [])}
            for k, v in d.items()}
