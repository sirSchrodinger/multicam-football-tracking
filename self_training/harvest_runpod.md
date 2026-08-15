# Çok-tesis harvest + pseudo-label — RunPod GPU reçetesi

**Neden GPU'da:** yerel 1650 Ti pseudo-label hızı ~2.8-3.3 fps. RunPod 4090 ~30-50
fps → 10-15× hızlı. 70s klip (2100 kare) yerelde ~7 dk, 4090'da ~1 dk. Çok-tesis
dataset'i (10-20 tesis) yerelde saatler, 4090'da ~20-30 dk.

## 0) Yerelde ne HAZIR (bu cephe, 29 Haz)
- `self_training/harvest_download.py` — sosyalhalisaha tesis keşfi + gece klip
  indirici (probe+doğrula). 6 YENİ tesis indirdi (s2 sunucu = faststart).
- `self_training/harvest_export.py` — base tracking export (recover_far=False,
  no-calib), `tracks_<name>.parquet` üretir.
- `self_training/rebuild_dataset.py` — 8 tesisi birleştirip COCO yeniden kurar,
  tesis-split + far-band ölçümü.
- Mevcut COCO: `self_training/data/coco/` (Çankaya tiling + Kıbrıs + 6 yeni).

## 1) İNDİRME tuzağı — s1 vs s2 (ÖNEMLİ bulgu)
sosyalhalisaha CDN'de **iki sunucu** var:
- `s2.sosyalhalisaha.com` → mp4 **faststart** (moov başta). ffmpeg `-ss N -t D -i
  URL -c copy` ÇALIŞIR. 6/6 s2 tesisi indi.
- `s1.sosyalhalisaha.com` → mp4 **faststart DEĞİL** (moov sonda). ffmpeg moov'u
  okumak için dosya-sonuna byte-range atar → CDN **404** döner (ama curl ile aynı
  range 206 çalışır → ffmpeg keep-alive/backend sorunu). 5/5 s1 tesisi atlandı.

**RunPod fix (s1'i de kurtar):** curl byte-range çalıştığı için tüm dosyayı
indir + faststart'a remux et, sonra kes:
```bash
# s1 dosyası: önce tamamını (veya yeterli baş kısmı) çek — 4090 podunda bant geniş
curl -A "Mozilla/5.0" -o full.mp4 "$URL"            # 4090 pod hızlı indirir
ffmpeg -y -i full.mp4 -ss 600 -t 70 -c copy -movflags +faststart clip.mp4
rm full.mp4
```
(Alternatif: `aria2c -x8` parça-paralel indirme; ya da yt-dlp.) Bu, tesis
havuzunu ~2×'e çıkarır (s1 tesisleri de erişilebilir olur).

## 2) RunPod adımları
```bash
# pod: Community 4090, PyTorch image, ~$0.34/sa
git clone <repo> halisaha-stats && cd halisaha-stats
pip install rfdetr==1.7.1 supervision==0.28.0 opencv-python pandas pyarrow  # (yerel venv eşleği)
# weights kopyala: models/weights/checkpoint_best_regular.pth

# A) keşif + indir (s1 remux fallback eklenmiş harvest_download ile)
N_WANT=20 CLIP_DUR=80 venv/bin/python self_training/harvest_download.py

# B) base tracking export (GPU) — MAX_FRAMES daha yüksek tut (4090 hızlı)
MAX_FRAMES=2400 venv/bin/python self_training/harvest_export.py

# C) COCO yeniden kur (tesis-split + far ölçümü)
STRIDE=15 venv/bin/python self_training/rebuild_dataset.py
```

## 3) Far-band tiling'i YENİ tesislerde açmak (recall'ın asıl açığı)
`recover_far=True` `FarBandRecovery` per-venue **kalibrasyon** ister (`in_pitch`
filtresi). Yeni tesiste calib yok → şimdilik base-only. İki yol:
- **(a) Hızlı, calib'siz:** `FarBandRecovery`'ye `homo=None` modu ekle →
  in_pitch filtresini ATLA, sadece boyut-kapısı (MIN_H) + tracklet-kapısı ile far
  kutuları kabul et. Statik tribün gürültüsü tracklet-kapısında (5s süre + düşük
  box-cv) çoğunlukla elenir. RunPod'da ucuz dene, held-out recall ile doğrula.
- **(b) Doğru:** her tesise tek-kare çizgi-kalibrasyonu (`calib/` line_calib) →
  in_pitch geçerli → tiling güvenilir. Emek pahalı, az tesis için.

## 4) Fine-tune (mevcut finetune_runpod.md ile aynı)
COCO hazır olunca `reorg_for_rfdetr.py` → `RFDETRLarge.train(...)`. Sonra YERELDE
`eval/recall_eval.py` (8 insan-GT) + held-out tesis recall ile doğrula.

## DÜRÜST sınırlar
- Pseudo-label gürültülü; base-only yeni tesislerde far oyuncu eksik etiketlenir
  (tam da öğretmek istediğimiz zor durum) → tiling açılana kadar far-zayıf.
- Box-count ile recall İDDİA ETME (tiling dedup geçmişte şişirdi). Recall = insan-GT.
- s1 tesisleri remux fallback olmadan atlanır (havuzun ~yarısı).
