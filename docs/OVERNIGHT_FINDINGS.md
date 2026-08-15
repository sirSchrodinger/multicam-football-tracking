# Gece koşusu bulguları (28 Haz → 29 Haz, ~6.9h otonom + refine)

Runner `eval/overnight_runner.py` **6.89h** çalıştı (101 aşama, hepsi ok) — erken-bitme
problemi çözüldü. Breadth-first öncelik + taban-süre doldurma. Sonuçlar (hepsi gerçek,
uydurma yok):

## 1. Tespit recall'ı (önceki oturum, doğrulandı)
8 frozen kare × 3 kör sayıcı GT: **base ~%73, +far-band tiling ~%82 (+9-10p),
precision ~%100** (0 FP). Kayıp tamamen far/karanlık-kümede; near/mid %100.
`docs/RECALL_NOTES.md`. FarBandRecovery doğrulandı → `export_tracks --recover-far`
ile pipeline'a wire edildi (bu gece full-game'de kullanıldı).

## 2. Boy-ölçeği çapraz-doğrulama
- **Boy-yöntemi:** Çankaya saha 32.5×17.2 m, kamera 3.28 m, band ±13% (focal
  belirsizliği; yöntem değil — simülasyon kanıtı).
- **Katalog elendi:** 46×24 → medyan insan boyu 2.48 m (fizik-dışı); 40×20 → 2.15 m.
  Saha ~32-34 m'de, boy + landmark f*≈1091'de uyuşuyor (`eval/scale_crosscheck.py`).
- **Merkez-yuvarlak çapası — NEGATİF sonuç (dürüst):** gece medyan-ROI ile yuvarlak
  net görüldü AMA homografi onu pitch'te **çembere değil 1.57-aspect elipse** taşıyor.
  Sebep: merkez-yuvarlak far/piksel-fakiri bölgede; oradaki homografi-aspect
  güvenilmez (projenin bilinen far-third limiti). → merkez-yuvarlak bu kamerada
  ölçek çapası DEĞİL. Ölçek için tek güvenilir yol: boy-yöntemi (tüm-saha, sağlam)
  + on-site tek ölçüm (±2%). (`eval/centercircle_scale.py`, negatif kayıt.)

## 3. İlk TAM-OYUN dataset (12 dk, tiling) — YENİ
`export_tracks --recover-far` ile aktif pencere (t=1500-2220): **178.749 satır,
17.902 frame, 1034 ham track → 33 tracklet**. Gerçek equal-area ısı haritası
(34×18 bin, %100 saha kapsama), per-oyuncu mesafe, ölçek "m (approx ±14%)".
parity-QC: medyan 12/frame (recall açığıyla tutarlı; far-küme eksikleri). Şimdiye
dek sadece 2 dk klip vardı → ilk anlamlı tam-oyun çıktısı.
`raw/tracks_cankaya_cam2_fullgame.parquet`, `stats_out/cankaya_cam2_fullgame/`.

## 4. ÇAN EĞRİSİ — boy dağılımı çok-tesis testi (Alperen ana sorusu)
**Soru:** birçok halısahanın verisinde oyuncu boyları beklenen çan eğrisini
veriyor mu?

- **Kalibre Çankaya METRİK boy: EVET, temiz Gauss.** n=21.973, ortalama 1.752 m,
  CV 0.085, skew **0.19**, excess-kurtosis **0.08**, QQ-r **0.998**. (Ortalamanın
  1.75 çıkması tanım gereği; ASIL kanıt şeklin Gauss olması — bu gerçek.)
- **Kalibrasyon-bağımsız çok-tesis proxy (ham box_h):** 72 tesis keşfedildi, 23'ü
  yeterli-veri. Ham box_h band-normalize → **çan eğrisi DEĞİL** (medyan skew +0.49,
  ağır kuyruk). **Sebep teşhis edildi:** ham box_h gerçek-boy değil — bükülen/örtülü/
  düşük-conf/FP kutuları kirletiyor. Çankaya'da clean-upright filtre uygulanınca
  skew **−0.30 → −0.007** (neredeyse mükemmel simetri) → kirlilik kanıtlandı.
- **REFINED filtreli çok-tesis koşusu (clean-upright filtre) DEVAM EDİYOR**
  (`--refine`, ~3h). Adil çok-tesis simetri testi: filtreyle her tesiste skew→0
  beklenir. [Sonuçlar tamamlandığında bu bölüm + rapor güncellenecek.]

**Ön sonuç:** Boy dağılımı, DÜZGÜN ölçülünce (kalibre + clean-upright) çan eğrisidir
(Çankaya kanıtı kesin). Ucuz ham-box_h kısayolu, çan-eğrisini göstermek için
clean-upright filtre gerektirir; filtresiz kontaminasyon onu çarpıtır.

## 5. Diğer
- Derin Monte Carlo (2000+ koşu, sigma süpürmeli): yöntem yansız (sapma ~%0.05),
  ideal-koşulda hatasız (doğruluk kanıtı sağlam).
- CLAHE: breadth önceliği için final koşudan çıkarıldı (smoke'ta sinyal yoktu).

## Sıradaki (refine bitince)
1. Refined filtreli çok-tesis çan-eğrisi sonucu → bu doc + rapora.
2. Raporu yeniden yaz: recall + boy-çan-eğrisi (çok-tesis) + ölçek-çapraz + tam-oyun
   bölümleri; yazar Alperen Uğur Erden.
3. Memory güncelle + commit.
