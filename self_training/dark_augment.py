#!/usr/bin/env python3
"""dark_augment.py — POD-uzeri offline karanlik-augmentation + rfdetr dataset kur.

v2 amaci: FAR + KARANLIK oyuncu recall'ini hedeflemek. Uzak-kale onunde koyu-formali
oyuncu = detektörün en zayif modu (a45: ~%27 kacan far-dark etiketsiz). Cozum:
her TRAIN karesinin gece/karanlik bir KOPYASINI uret (AYNI box annotation) ->
model dusuk-isik + dusuk-kontrast dagilimini ogrenir. rfdetr internals'a DOKUNMAZ
(en guvenli yol): sadece pikselleri karartir, kutular birebir korunur.

Karartma (fiziksel-akla-yatkin gece dönüşümü, per-image randomize):
  - dikey gradyan: ust (uzak/far band) daha koyu, alt (yakin) daha az -> far-dark'i
    ozellikle vurgular (uzak-kale geometrisi cam2'de ustte).
  - global gamma>1 (mid-ton karart) + parlaklik olcek + hafif sensor gurultusu.
  - VAL karartilmaz (held-out venue = gercek dagilim, overfit/ölçüm bozulmaz).

Cikti: /workspace/dataset/{train,valid}/  her birinde _annotations.coco.json + jpg.
       train = orijinal + dark_ kopyalar. Kategori tek 'person' (id 1), yerel ft ile birebir.
Kullanim (pod): python dark_augment.py /workspace/coco /workspace/dataset
"""
from __future__ import annotations
import json, sys, shutil
from pathlib import Path
import numpy as np
import cv2

SRC = Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/coco")
DST = Path(sys.argv[2] if len(sys.argv) > 2 else "/workspace/dataset")
TRAIN_JSON = "train_multivenue.json"
VAL_JSON = "val.json"
IMG_OFFSET = 1_000_000          # dark image id ofset (orijinallerle cakisma yok)
ANN_OFFSET = 10_000_000
SEED = 1234


def night_transform(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Fiziksel-akla-yatkin gece dönüşümü. Kutulari DEGISTIRMEZ (sadece piksel)."""
    h, w = img.shape[:2]
    f = img.astype(np.float32) / 255.0
    # 1) dikey karartma gradyani: ust (far) koyu, alt (near) daha az. randomize.
    top = rng.uniform(0.42, 0.58)      # ust satir carpani (en koyu)
    bot = rng.uniform(0.72, 0.86)      # alt satir carpani
    col = np.linspace(top, bot, h, dtype=np.float32)[:, None, None]
    f = f * col
    # 2) global gamma>1 (mid-ton karart)
    gamma = rng.uniform(1.6, 2.1)
    f = np.power(np.clip(f, 0, 1), gamma)
    # 3) hafif genel parlaklik dususu
    f = f * rng.uniform(0.85, 1.0)
    # 4) dusuk-isik sensor gurultusu (far-dark'ta gercekci)
    noise = rng.normal(0.0, rng.uniform(0.006, 0.016), f.shape).astype(np.float32)
    f = f + noise
    out = np.clip(f * 255.0, 0, 255).astype(np.uint8)
    return out


def build():
    tr = json.load(open(SRC / TRAIN_JSON))
    va = json.load(open(SRC / VAL_JSON))
    cats = [{"id": 1, "name": "person"}]
    rng = np.random.default_rng(SEED)

    # ---- VALID (karartma YOK) ----
    dval = DST / "valid"; dval.mkdir(parents=True, exist_ok=True)
    n = 0
    for im in va["images"]:
        s = SRC / "images" / im["file_name"]
        if s.exists():
            shutil.copy(s, dval / im["file_name"]); n += 1
    va_out = {"images": va["images"], "annotations": va["annotations"], "categories": cats}
    json.dump(va_out, open(dval / "_annotations.coco.json", "w"))
    print(f"valid: {n} kare -> {dval}", flush=True)

    # ---- TRAIN (orijinal + dark kopya) ----
    dtr = DST / "train"; dtr.mkdir(parents=True, exist_ok=True)
    images = list(tr["images"])
    anns = list(tr["annotations"])
    # image_id -> anns eslesmesi
    by_img = {}
    for a in tr["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)

    kept = darkened = missing = 0
    for im in tr["images"]:
        s = SRC / "images" / im["file_name"]
        if not s.exists():
            missing += 1; continue
        # orijinali kopyala
        shutil.copy(s, dtr / im["file_name"]); kept += 1
        # dark kopya uret
        img = cv2.imread(str(s))
        if img is None:
            continue
        dark = night_transform(img, rng)
        dfn = "dark_" + im["file_name"]
        cv2.imwrite(str(dtr / dfn), dark, [cv2.IMWRITE_JPEG_QUALITY, 92])
        dim = {"id": im["id"] + IMG_OFFSET, "file_name": dfn,
               "width": im.get("width", img.shape[1]), "height": im.get("height", img.shape[0])}
        images.append(dim)
        for a in by_img.get(im["id"], []):
            na = dict(a)
            na["id"] = a["id"] + ANN_OFFSET
            na["image_id"] = im["id"] + IMG_OFFSET
            anns.append(na)
        darkened += 1

    tr_out = {"images": images, "annotations": anns, "categories": cats}
    json.dump(tr_out, open(dtr / "_annotations.coco.json", "w"))
    print(f"train: orijinal={kept} + dark={darkened} = {len(images)} kare, "
          f"{len(anns)} box (missing_src={missing}) -> {dtr}", flush=True)
    # ozet
    print(json.dumps({"train_images": len(images), "train_boxes": len(anns),
                      "valid_images": n, "valid_boxes": len(va["annotations"]),
                      "dark_copies": darkened}), flush=True)


if __name__ == "__main__":
    build()
