# halisaha-stats — mimari

Sabit-kamera amatör halısaha (7v7 = 14 oyuncu) maçlarından **oyuncu-bazlı
istatistik** (mesafe, heatmap, hız zonları, territory) çıkarıp sosyalhalisaha.com
gibi platformlara B2B satmak. Tek elden geliştirilir; gerçek-veri ve dürüstlük
önceliklidir — *ölçemediğimiz hiçbir şeyi uydurmuyoruz*.

## Veri akışı (uçtan uca)

```
maç videosu (sabit kamera, gece, 1080p)
   │
   ├─[A] KALİBRASYON  (maç/kamera başına BİR KEZ)
   │      fisheye undistort (k1) → manuel/çizgi homografi → calib/<cam>.json
   │
   ├─[B] TESPİT+TAKİP  export_tracks.py
   │      RF-DETR (SoccerNet ckpt) + ByteTrack → ayak noktaları parquet
   │      + [B'] recall_qc: parite-QC + uzak-bant recovery + low_conf/zone
   │
   ├─[C] KİMLİK  detect/track_stitch.py
   │      146 tracklet → ~14 oyuncu (metrik hareket + zaman + jersey füzyon)
   │
   ├─[D] İSTATİSTİK  stats/topdown_stats.py
   │      homografi ile saha-metresi → equal-area heatmap, mesafe, hız, territory
   │      per-zone güven + kapsama maskesi + relative_m/m dürüst etiket
   │
   └─[E] GÖRSEL/ÜRÜN  stats/topdown_viz.py
          top-down heatmap PNG, per-oyuncu, yörünge overlay
```

## Katmanlar

### [A] Kalibrasyon — `pitch/`, `calibrate_field.py`, `pick_points.py`, `calib/`
Sabit kamera → homografi maç boyunca DEĞİŞMEZ, bir kez kurulur.
- **Lens distorsiyonu (fisheye):** `cv2.undistort`, tek-parametre `k1` bölme modeli;
  gerçekte-düz saha çizgilerini düzleştirerek bulunur. Çankaya cam2: **k1=-0.13, fx=1000**
  (`calib/cankaya_cam2_distortion.npy`). *Bug dersi:* zayıf bir optimizasyon (boundary
  fit) `best_k1`'i +0.02'ye ezmişti → her şey kavisli kareye oturmuştu; düzeltildi.
- **Homografi:** `PitchHomography` (`pitch/homography.py`) — `calibrate_manual` (HAM-piksel
  tıklama, içeride undistort), `pixel_to_pitch` (sıcak yol, vektörize), `pitch_to_pixel`,
  `qa_overlay`, `save/load`. Yön: UNDISTORTED piksel → saha metresi.
- **Manuel tıklama:** `pick_points.py` (rehberli, 4x büyüteç, atla/geri); kale direkleri
  (3 m), saha köşeleri, orta yuvarlak işaretlenir. `calibrate_field.run_calibration`
  homografi + QA overlay + top-down üretir.
- **Görünmeyen/kadraj-dışı köşe:** `pitch/line_calib.py` — köşe yerine her sınır ÇİZGİSİ
  üstünden ≥2 nokta → RANSAC fit → KESİŞİM ile görünmeyen köşe sentezi (saha dikdörtgen).
  `coverage_confidence_map` (kadraj-dışı/düşük-kapsama maskesi) + `scale_anchor('goal_width',3.0)`.
- **Kanonik calib QA ile seçilir, dosya adıyla DEĞİL:** `calib/cankaya_cam2_v2.json`
  (reproj 9.87px) İYİ; `calib/cankaya_cam2.json` (65px, ters-H) BOZUK. `export_tracks` ve
  `topdown_stats` `accept_calib_qa` ile bozuğu **reddeder** (median_px>20 / near>far / n<6).

### [B] Tespit + Takip — `export_tracks.py`, `track_smoke.py`, `detect/recall_qc.py`
- **Dedektör:** `RFDETRLargeDeprecated` (SoccerNet ckpt `models/weights/checkpoint_best_regular.pth`,
  Apache-2.0). THRESH=0.3, MIN_H=25, NMS 0.6 class_agnostic; tüm sınıflar=person.
  *Uyarı:* 85.7 mAP BROADCAST'tir, amatör/gece recall'ı düşük (uzak üçte-bir).
- **Takip:** supervision ByteTrack (MIT). Ultralytics/boxmot AGPL → **kullanılmaz**.
- **Çıktı:** per-frame ayak noktası parquet. Kolonlar: `tid,frame,t_sec,foot_x,foot_y,
  box_h,box_w,conf,bottom_cropped,pitch_x,pitch_y,in_pitch`. Pitch kolonları tek vektörize
  post-pass ile dolar (calib varsa); yoksa NaN (dürüst).
- **recall_qc:** parite-QC (7v7=14, tek→kaçan oyuncu sinyali), uzak-bant tiling recovery,
  `low_conf`/`zone` kolonları (kadraj-dışı/uzak-üçte metrik uydurmama garantisi).

### [C] Kimlik — `detect/track_stitch.py`
ByteTrack 2 dk'da yüzlerce ID-switch üretir (146 tracklet). Mesafe/heatmap/territory'nin
*hepsi* oyuncu-bazlı kimliğe bağlı. Stitching: metrik hareket-süreklilik (saha-metresi hız
gate) + zaman-örtüşme (örtüşen tracklet ≠ aynı oyuncu) + jersey/appearance füzyonu.
**Aşırı-birleşme (2 oyuncu→1) = mesafe yalanı → yasak.** Hedef ~14-16 küme.

### [D] İstatistik — `stats/topdown_stats.py` (+ eski `stats_report.py`)
- **Tek-kaynak projeksiyon:** sadece `PitchHomography`. `prepare` → `kinematics_all` →
  `parity_qc` → `heatmap_grid_equal_area`.
- **Equal-area heatmap:** saha-metresinde binleme, per-bin exposure normalize, kadraj-dışı
  hücre NaN ('bilinmiyor', 0 değil). Eski perspektif-önyargılı piksel-heatmap'i değiştirir.
- **Dürüstlük:** metre kilidi `scale_anchor` ile açılır; yoksa çıktı `relative_m` (metre
  DEĞİL). Mesafe ~%2-5, hız ~%5 *ancak* ölçek+stitching tamamsa; ivme **asla**.

### [E] Self-training — `self_training/` (uykuda)
Pseudo-label + drift-guard (Mean Teacher, leave-one-stadium-out). Far-third negatif-sızıntısı
(eksik oyuncuyu arka-plan sayma) ve K≥2 saha + donmuş val olmadan etkin değil. `train.py` stub.

### [F] Ingest + Venue — `ingest/venue_registry.py`, `scan_cams.py`
sosyalhalisaha XHR'den maç/kamera listesi; ham video ~15 günde silinir. `venue_registry`:
venue-agnostik, kapalı/açık tespiti, ölçü-kaynağı hiyerarşisi (yayınlı > uydu(açık) >
adımlama > standart+3m kale), en kolay kalibre edilen sahayı sıralar. Çankaya muhtemelen
kapalı (uydu çatı gösterir).

## Dürüstlük duruşu (sözleşme)
- **relative_m vs m:** gerçek saha ölçüsü (L) yok → `scale_anchor=null` → her çıktı `relative_m`.
  Metre iddiası ancak 3m kale-ağzı çapası VEYA ölçülmüş L ile.
- **Kapsama maskesi:** kadraj-dışı (yakın-alt köşe, korner) + uzak-üçte düşük-piksel → gri
  'bilinmiyor', metrik üretilmez.
- **Parite-QC:** tek sayım → kaçan oyuncu raporlanır, gizlenmez.
- **Aşırı-birleşme yasağı:** stitching iki oyuncuyu birleştirmemeli.
- **Broadcast≠amatör:** mAP broadcast'ten, amatör recall ayrı ölçülür.

## Çalıştırma
```bash
# 1) kalibrasyon (kamera başına bir kez) — GUI
venv/bin/python pick_points.py calib/<cam>_frame_raw.png \
  --k1npy calib/<cam>_distortion.npy --cam <cam> --L 40 --W 25
# görünmeyen köşe varsa: pick_lines.py (çizgi-modu)

# 2) tespit+takip → parquet  (calib QA-kapısından geçmeli)
venv/bin/python export_tracks.py <clip.mp4> --calib calib/<cam>_v2.json

# 3) top-down istatistik (gerçek veri demosu)
venv/bin/python -c "from stats.topdown_stats import generate_topdown_report as g; \
  g('raw/tracks_<cam>_clip.parquet','calib/<cam>_v2.json','stats_out/demo')"

# testler
venv/bin/python tests/test_line_calib.py && venv/bin/python -m ingest.test_venue_registry \
  && venv/bin/python detect/test_recall_qc.py && venv/bin/python tests/test_topdown_stats.py
```

## Açık işler (öncelik sırası)
1. **Gerçek saha ölçüsü (L):** 34 vs 40 m belirsizliği → metre kilidi kapalı. Çözüm: 3m
   kale-ağzı çapası / sahayı adımla / açık-saha (Serdivan) + uydu.
2. **Tracklet stitching (146→14):** oyuncu-bazlı her metriğin ön-koşulu.
3. **Hot-path wiring:** recall_qc + low_conf maskesi `export_tracks`/`stats_report`'a.
4. **Harvest downloader:** 15-gün silme penceresine karşı otomatik indirme/kuyruk.
5. **Self-training:** K≥2 saha + donmuş val + far-third ignore-region.
6. **Throughput:** RF-DETR Large 4GB'da ~6h/maç → resume/checkpoint + nano-distill.
