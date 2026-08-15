# Saha kalibrasyonu — cankaya_cam2

Tek-kamera (sabit, köşe-montaj, gece) halısaha görüntüsünden metrik üst-bakış için
kalibrasyon kaydı. Homografi maç başına **bir kez** kurulur (kamera sabit).

## Tesis
- **Çankaya Halı Saha**, Rabindranath Tagore Cad. No:24, Çankaya-Ankara.
  sosyalhalisaha.com platformu, cam2. (Watermark + dosya adı + adres teyitli.)
- Listeleme sayfasında saha ölçüsü **yayınlanmamış** → ölçü standarttan + ölçek-çapasından.

## Gerçek saha ölçüsü (araştırma)
- TR halısaha **7v7 = 14 oyuncu** (2×7, kaleciler dahil). Bu görüntüde sayım da 14'e oturuyor.
- 7v7 saha tipik: **uzunluk 38–50 m, genişlik 25–30 m** (ideal 28×48 / 30×50, küçük 25×45).
- **Kale ağzı 3 m genişlik × 2 m yükseklik (standart)** → en güvenilir ölçek-çapası.
  İki kale de görünür; homografi ölçeği 3 m kale ağzından çapraz-kontrol edilebilir.
- Default varsayım: **L=40 m (yakın→uzak kale), W=25 m (touchline arası)** — yaklaşık,
  kullanıcı/maps ile netleşir. Kaynaklar: reformsports.com, nursanspor.com, sakaryahalisaha.com.tr.

## Lens distorsiyonu (fisheye/barrel) — ÇÖZÜLDÜ
- `cankaya_cam2_distortion.npy` = `[k1, fx, cx, cy] = [-0.18, 1000, 960, 540]`.
- k1, gerçekte-düz saha çizgilerini düzleştirerek nicel bulundu (göz-tahmini değil;
  scratchpad/undistort_fit.py). Düzeltilmiş kare: `cankaya_cam2_undistorted.png`.
- Pipeline ham frame'de tespit yapar; sadece seyrek ayak-noktaları undistort + H.

## Köşe konvansiyonu (saha = dikdörtgen, origin yakın-kale alt köşesi)
```
yakın-kale end-line x=0   |   uzak-kale end-line x=L     (uzunluk)
touchline'lar y=0 ve y=W                                  (genişlik)
image saat-yönü: FL, FR, NR, NL
FL=(0,W)   FR=(L,W)   NR=(L,0)   NL=(0,0)
```

## Köşe durumu (TEK kareden tam otomatik çıkmıyor — dürüst)
3 bağımsız yöntem denendi (Hough; yeşil-maske 4-kenar RANSAC; süreklilik-kısıtlı
touchline fit). Sonuç:
- **Görülebilen:** FL (sol, yakın kale üstü), NL (alt-sol), FR (uzak kale, sağ-üst).
- **Görülemeyen:** **NR (alt-sağ köşe FRAME DIŞI** — yeşil sağ-alt kenardan taşıyor);
  uzak touchline file/arka-planla karışıyor; end-line'lar kalelerin arkasında.
- Bu yüzden hassas köşe = kullanıcı tıklaması (ground-truth) + dikdörtgen kısıtıyla
  görünmeyen köşenin uzatma/kesişimle tamamlanması.

## Yeniden kalibrasyon (turnkey)
1. `calib/cankaya_cam2_grid_ref.png` aç (undistorted + 100px etiketli grid; kırmızı = kaba tahminim).
2. 4 köşe pikselini oku: **FL FR NR NL** (undistorted-grid koordinatı).
3. Çalıştır:
   ```
   venv/bin/python calibrate_field.py calib/cankaya_cam2_frame_raw.png \
     --k1npy calib/cankaya_cam2_distortion.npy --cam cankaya_cam2 \
     --L 40 --W 25 --corners "FLx,FLy FRx,FRy NRx,NRy NLx,NLy"
   ```
   (GUI tercih: `python -m pitch.homography` annotate yolu.)
4. Çıktı: `calib/cankaya_cam2.json` (H + distorsiyon), `qa_overlay_*.png`
   (turuncu 5 m grid saha çizgilerine UYMALI), `topdown_*.png`.
5. QA overlay'de grid saha çizgilerine oturana kadar köşeleri ince-ayar yap.

## DRAFT (kaba köşelerle, ölçek doğrulaması)
`*_DRAFT.json/png` benim göz-tahmini köşelerimle. Pipeline'ın çalıştığını gösterir;
metrik doğruluk için kullanıcı köşeleriyle DEĞİŞTİR.

## Dürüst sınırlar
- **Uzak üçte-bir piksel-fakiri:** köşe-montaj tek kamera uzak yarıyı az piksele
  sıkıştırır → uzak-üçte-bir metrik/heatmap düşük-güven (per-zone QA bunu işaretler).
- Homografi düzlem (zemin) varsayar; ayak-noktası tabanlı → ölçüm ~%2–5 (mesafe),
  ~%5 (hız) bandında; ivme RAPOR EDİLMEZ.
- k1 tek-parametre (bölme modeli); satranç-tahtası kalibrasyonu daha sağlam olur.
