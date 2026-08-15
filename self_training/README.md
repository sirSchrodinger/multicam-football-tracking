# self_training/ — DEFERRED self-improvement loop

Bu paket **ertelenmiş**tir. Sistemin asıl başarı kriteri (görülmemiş yeni sahada
otomatik kalibrasyonun tutması) **klasik-CV** ile çözülür (`pitch/auto_calib.py` +
manuel `pitch/homography.py`). `self_training/` ancak elde yeterli kalibre saha
biriktikten sonra, dedektörü ve (ileride) çizgi-segmentasyon ağını ucuza
iyileştirmek için devreye girer.

> **Derin ağ ile BAŞLAMA.** Önce manuel + chamfer auto-calib K≈5+ saha biriktirsin,
> sonra burası o veriyi madenler.

## Ön koşullar (ikisi de zorunlu)

1. **K ≥ ~5 elle (manuel-H) kalibre edilmiş ayrı saha** (`calib/<camera_id>.json`,
   `status="manual"` veya `auto_accepted`). Bunlar pseudo-label kaynağı.
2. **Donmuş leave-one-stadium-out validasyon seti.** İnsan-doğrulamalı küçük kümeler;
   her saha sırayla held-out olur. Generalization SADECE buradan ölçülür. Tek-saha QA
   hiçbir şey kanıtlamaz.

## Döngü

```
harvest ──▶ pseudolabel ──▶ drift_guard(BEFORE) ──▶ train(4090) ──▶ drift_guard(AFTER) ──▶ keep | rollback
 (ham video)  (pseudo_label)   (baseline ölç)         (stub)          (regresyon kapısı)
```

### 1. harvest
Ham video ~15 günde silindiği için **sürekli harvest hattı şart**. Her maçta:
- `export_tracks.py` ile per-frame parquet üret (kalıcı; ham video silinse de kalır).
- Düşük-yoğunluklu birkaç frame'i kalibrasyon/seg-GT için JPG kaydet
  (`pitch/auto_calib.select_calib_frame`).

### 2. pseudolabel (`pseudo_label.py`)
```python
import pandas as pd
from self_training import pseudo_label as pl

df = pd.read_parquet("tracks_cankaya_cam2_clipNNNN.parquet")
recs = pl.detection_pseudo_labels(
    df, conf_thr=0.6, min_dur_s=10.0,        # interface-spec eşikleri
    require_in_pitch=True, img_w=1920, img_h=1080,
)
print(pl.last_summary())                      # {tracklets_accepted, tracklets_rejected, label_rows}
pl.to_coco(recs, 1920, 1080, "corpus/cankaya/instances.json")
pl.to_yolo(recs, 1920, 1080, "corpus/cankaya/labels")
```
Sadece **yüksek-güven + uzun-ömür + zaman-tutarlı** tracklet'ler etikete dönüşür
(box-jitter / teleport / mostly-off-pitch elenir). `bottom_cropped` satırlar düşürülür.

Çizgi-seg GT'si (kabul/manuel H'den, kamera sabit → tek maske tüm maça):
```python
mask = pl.line_seg_labels_from_H(homo, img_shape=(1080, 1920))   # (H,W) uint8
# veya kalibrasyon dosyasından doğrudan:
pl.line_seg_corpus_from_calib("calib/cankaya_cam2.json", frame_index=[100, 800, 1500],
                              img_shape=(1080, 1920), out_dir="corpus/cankaya/lineseg")
```

### 3. drift_guard — BEFORE (baseline) (`drift_guard.py`)
Fine-tune öncesi mevcut modeli held-out sette ölç:
```python
from self_training import drift_guard as dg

def model_fn(image):                 # RF-DETR sarmalayıcı; {"boxes":(N,4),"scores":(N,)}
    ...
baseline = dg.evaluate_held_out(model_fn, frozen_loso_val, iou_thr=0.5)
```
`baseline["overall"]` (AP/recall/precision), `baseline["per_stadium"]`,
`baseline["calib"]` (median_reproj_px, auto_accept_rate).

### 4. train — KİRALANAN 4090 (`train.py`, STUB)
Lokal GTX 1650 Ti 4GB **yetmez**; ağır iş kiralanır.

```bash
# RunPod RTX 4090 (~$0.34/saat). Vast.ai genelde daha ucuz.
runpodctl create pod \
  --gpuType "NVIDIA GeForce RTX 4090" --imageName "runpod/pytorch:2.4.0-py3.11-cuda12.4" \
  --containerDiskInGb 40 --volumeInGb 60 --ports "22/tcp"
# pod hazır olunca:
scp -r corpus/ frozen_loso_val/ root@<pod-ip>:/workspace/
ssh root@<pod-ip>
#   pip install -e .   (BSD/MIT bağımlılıklar; PnLCalib/Ultralytics ASLA)
#   python -m self_training.train_entry --corpus /workspace/corpus --val /workspace/frozen_loso_val
scp root@<pod-ip>:/workspace/out/weights.pth models/weights/
runpodctl remove pod <pod-id>            # işi bitince kapat; maliyet ~mac başına $0.2-0.5
```
- **Mean-Teacher EMA** (`drift_guard.ema_update` mantığı): teacher pseudo-label
  üretir, student gradyan alır, teacher = EMA(student). Pseudo-label gürültüsünü
  bastırır.
- Çizgi-seg eklenirse **yalnız BSD/MIT backbone** (SegFormer-b0 gövde lisansını
  doğrula; tercih Apache/MIT). `finetune_segnet` / `finetune_detector` şu an
  `NotImplementedError` — gerçek torch döngüsü buraya yazılır.

### 5. drift_guard — AFTER + kapı (`gate_or_rollback`)
```python
new = dg.evaluate_held_out(new_model_fn, frozen_loso_val)
verdict = dg.gate_or_rollback(new, baseline)
if verdict["keep"]:
    promote("models/weights/weights.pth")        # yeni ağırlığı yayına al
else:
    print("ROLLBACK:", verdict["reasons"])        # eski ağırlıkta kal
```
Kapı **muhafazakâr**: held-out AP/recall gerilerse, **herhangi bir** sahada AP
çökerse veya median reproj_px artarsa **reddet**. Şüphede iken yeni modeli reddet
(B2B güveni > marjinal kazanç).

## Ne zaman İNSAN doğrulaması şart
- Donmuş val setini kurarken (kutu/landmark elle onayı) — **bir kez, az sayıda frame**.
- Yeni bir saha tipine (örn. ilk 7v7) geçerken pseudo-label kalitesini spot-check.
- `gate_or_rollback` ardışık turlarda metriği yatay/aşağı çekiyorsa: pseudo-label
  eşiklerini insan gözüyle gözden geçir (otomatik gevşetme YOK).
- `drift_guard.vibration_sentinel_ok` RECALIBRATE derse: kamera oynamış olabilir,
  o sahayı elle yeniden kalibre et.

## Lisans (paket geneli, pazarlık edilemez)
Primary path **yalnız** OpenCV(BSD) + scipy(BSD) + numpy + pandas. Öğrenilmiş ağ
eklenirse **BSD/MIT backbone**. **ASLA** PnLCalib (GPL-2.0) / Ultralytics (AGPL-3.0)
kapalı üründe. SoccerNet-winner repo lisansı doğrulanmadan kullanılmaz.

## Dosyalar
| dosya | iş |
|---|---|
| `pseudo_label.py` | yüksek-güven tracklet → COCO/YOLO detection pseudo-label; H → dense çizgi-seg GT |
| `pseudolabel.py` | `pseudo_label`'ın ince re-export shim'i (görev adı uyumu) |
| `drift_guard.py` | held-out mAP/recall/reproj ölç; regresyon kapısı + rollback; EMA + titreşim nöbetçisi |
| `train.py` | fine-tune girişi (STUB; kiralanan 4090; BSD/MIT) |
