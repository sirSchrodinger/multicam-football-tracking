# Halısaha öğrenilmiş kalibrasyon — dürüst plan (18 Tem 2026)

## Neden buradayız
Klasik tek-kare auto-calib (turf-mask + LSD + homografi) ÇALIŞMIYOR — kanıtlandı:
reprojekte template hiçbir gerçek boyalı çizginin üstüne oturmuyor (bkz
`scratchpad/brutal_*.jpg`). Residual metriği bunu yakalamıyor (Goodhart: düşük
residual ↔ yanlış calib bir arada). Manuel calib ÇALIŞIYOR (cankaya orta-yuvarlak
testi geçti) ama sadece orta-yuvarlak için; köşeler cankaya'nın 36.5×17 template'i
gerçek sahaya oturmadığından ŞÜPHELİ.

Alperen kararı (18 Tem): **sıfır-kurulum, her kamera** → öğrenilmiş pitch-keypoint
modeli. (Bu "SONRAKİ keypoint detektör" notu 30 Haz'dan beri bekliyordu.)

## Pivotal deney (YAPILDI)
PnLCalib (SOTA, HRNet keypoint+line, pretrained broadcast) halısahaya SIFIR-SHOT
transfer ETMİYOR: eşik 0.15'te sadece 2-6 gürültü noktası çimende, orta-yuvarlak/
kenar/köşede HİÇBİR şey. Domain gap çok büyük (balıkgözü CCTV + 5v5 + gece).
→ **Finetune ŞART.** Mimari (HRNet heatmap) + pipeline (train.py) yeniden kullanılır.
Ama 57-kp SoccerNet şeması full-pitch'e bağlı; halısaha KENDİ şablonunu gerektirir
(`learned/kp_template.py`: 11 kp — 4 köşe + 2 orta×kenar + merkez + 4 çember).

## GERÇEK darboğaz: VERİ (etiketleme)
Genelleşen model için ÇOK sayıda çeşitli sahadan etiketli veri gerekir. Etiket =
gerçek görünür özellikleri (köşe/çember) tıklamak. **Ben tıklayamam (GUI yok).**
Ya Alperen tıklar (`calib/make_clicker_html.py` hazır) ya da bir etiketleme çabası.
Statik kamera avantajı: saha başına BİR calib → binlerce etiketli kare (farklı
oyuncu/ışık, augment'lenebilir). ~15-20 çeşitli saha yeterli olabilir.

- cankaya calib'inden auto-etiket denendi (`learned/gen_labels.py`): çember-kp'ler
  doğru, köşe-kp'ler YANLIŞ (template≠gerçek sınır). Yani cankaya bile temiz seed
  için calib'i yeniden-doğrulama/tıklama ister.

## Yol (dürüst milestone'lar)
1. **Etiketleme** (Alperen-gated): 15-20 saha, `make_clicker_html` ile 11-kp tıkla
   → her saha calib + binlerce etiketli kare. cankaya'yı ÖNCE düzgün yeniden-tıkla
   (mevcut 36.5×17 template'i gerçek sınıra oturt).
2. **Etiket→heatmap** pipeline (ben): calib → 11-kp Gaussian heatmap eğitim çifti.
3. **Finetune** (ben, RunPod — yerel 4GB yetmez): HRNet, pretrained init, 11-kp head.
4. **Inference + solve**: keypoint → homografi (statik kamera: çok-kare agrega ile
   sağlamlaştır) → orta-yuvarlak testiyle DOĞRULA (sayı değil GÖZLE).
5. **Genelleşme testi**: eğitimde OLMAYAN sahada çalışıyor mu.

## Ne ben yaparım / ne Alperen'den gerekir
- Ben: pipeline, heatmap-üretim, finetune (RunPod), inference, doğrulama, tüm kod.
- Alperen: sahaları etiketle (tıkla) — genelleşmenin tek gerçek maliyeti. Kaç saha
  = ne kadar genelleşir. Bunu ben yapamam; dürüst gerçek bu.

## Dosyalar
- `learned/kp_template.py` — 11-kp halısaha şablonu
- `learned/gen_labels.py` — calib→etiket (cankaya; köşe-kp doğrulaması BEKLİYOR)
- `external/PnLCalib/` — SOTA repo + pretrained ağırlıklar + `diag_raw.py` (transfer testi)
- `scratchpad/brutal_*.jpg` — klasik calib'in çuvalladığının kanıtı
