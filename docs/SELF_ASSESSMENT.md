# Öz-değerlendirme + yol haritası (29 Haz) — halisaha-stats

Dürüst bakış: ne sağlam, ne zayıf, en yüksek kaldıraç ne, nasıl ilerlemeli.

## Nerede SAĞLAMIZ (kanıtlı, test'li)
- **Tespit + takip güçlü.** Recall ÖLÇÜLDÜ (eyeball değil, 3-kör-sayıcı GT):
  base ~%73, far-band tiling ile ~%82, **precision ~%100** (dedektör uydurmuyor).
  Far-band tiling doğrulandı + pipeline'a wire edildi.
- **Boy-ölçeği yöntemi matematiksel olarak doğru.** Simülasyon: ideal koşulda %0
  hata, gürültü altında yansız. ±13% band = tek-görüntü odak belirsizliği
  (yöntem kusuru değil; çözülebilir).
- **Çekirdek pipeline çalışıyor:** sürekli konum haritası, roster (14-core/partial),
  occupancy, equal-area ısı haritası, tek-komut export. İlk **tam-oyun (12 dk)
  tiling dataset'i** üretildi (178k satır).
- **Dürüstlük kültürü işliyor:** negatif sonuçları (merkez-yuvarlak çapası,
  kalibrasyonsuz box_h proxy, tek-kareden takım-renk) zorlama-pozitife çevirmeden
  raporluyoruz. Bu, sonuçlara güveni artırıyor.

## Nerede ZAYIFIZ (dürüst boşluklar)
1. **TEK TESİS.** Her metrik sonuç Çankaya cam2'ye bağlı. Genellenebilirlik
   KANITLANMADI — başka tesiste çalışır mı bilmiyoruz.
2. **TEK MAÇ örnekleri.** Çok-tesis denemesi tesis başına bir maç (~14 kişi) =
   popülasyon değil; çan-eğrisi testi için çok küçük/yanlı. (Alperen yakaladı.)
3. **Kalibrasyon MANUEL.** Çankaya elle tıklandı. Otomatik değil → ürün
   ölçeklenmiyor (her yeni tesis elle iş ister).
4. **Ölçek ±13% hâlâ açık** (focal). Merkez-yuvarlak çapası bu kamerada başarısız
   (far-field). On-site ölçüm veya çok-veri tutarlılığı gerek.
5. **Recall pilot N=8 kare, tek maç.** Yön-gösterici; tesis/ışık çeşitliliği yok.
6. **Takım-ayrımı ve top-tespiti çözülmedi** (dinamik yelek + gece zor).

## EN YÜKSEK KALDIRAÇ — tek hamle çoğu boşluğu kapatıyor
**PER-TESİS OTO-KALİBRASYON + ÇOK-MAÇ HAVUZLAMA.**

Mantık: kamera SABİT → homografi o tesisin TÜM maçlarında aynı → tesisi **bir kez**
(otomatik) kalibre et, sonra o tesisin **onlarca maçını** aynı kalibrasyonla işle.
Tek hamlede:
- (a) **Çan-eğrisini binlerce oyuncuyla** düzgün test eder (14 → binlerce; gerçek
  popülasyon). Boşluk #2 kapanır.
- (b) Metrik istatistiği **HER tesise** açar → ürün tek-tesis demosundan
  ölçeklenebilir sisteme döner. Boşluk #1, #3 kapanır.
- (c) Ölçeği binlerce-boy medyanı + tesisler-arası tutarlılıkla **daha sıkı**
  kestirir → ±13% daralabilir. Boşluk #4'e kısmi.

Üstelik kilit kolaylık: oto-kalibrasyon sadece **homografiyi (aspect)** doğru
kursa yeter — **ölçeği boy-yöntemi** zaten veriyor. Yani "tam metrik kalibrasyon"
değil, "doğru-perspektif homografi + gate" yeterli. Bu, problemi ciddi küçültür.

## RİSKLER (ve karşılığı)
- **Oto-kalibrasyon gece-720p'de kırılgan.** `auto_calib` iskeleti var ama
  per-tesis fisheye çıkarmıyor, sol-sağ+180° belirsizliğini tam çözmüyor, sabit
  saha-boyutu varsayıyor. → Bunları kapatan `venue_autocalib` inşa edilecek;
  **QA-gate dürüst reddetmeli** (sessiz-yanlış homografi YAYMA — kabul-oranı düşük
  olsa bile temiz).
- **İndirme verimi düşük** (çok "0 dets"). → çok-tarih + retry + reconnect; tesis
  başına birden çok maç dene (zaten istiyoruz).
- **Aynı kamera mı?** Bir tesiste birden çok saha/kamera olabilir. → maçı
  havuzlamadan önce statik çizgi-haritası örtüşmesiyle **view-consistency** doğrula.
- **Kabul-oranı belirsiz:** belki tesislerin %30-50'si oto-kalibre olur. Dürüst:
  olanları raporla, olmayanları "manuel/atla" işaretle (zorlama yok).

## YOL HARİTASI (faz, öncelik sırası)
1. **`pitch/venue_autocalib.py`** — per-tesis fisheye-tahmin + chamfer kayıt
   (mevcut `register_template_chamfer`) + güçlendirilmiş gate (gol-tarafı/aspect)
   + boy-ölçeği entegrasyonu. Çıktı: kabul/red + calib JSON. (calib tasarım
   workflow'u spec'i besledi.)
2. **Çok-maç havuzlama runner** — `discover_venues` → tesis→maç-listesi (çok tarih);
   tesis-başı-bir-kez kalibre; view-consistency; tüm maçları havuzla → metrik boy.
3. **Çan-eğrisi v2** — kalibre-geçen tesislerde **metrik** boy popülasyonu
   (binlerce oyuncu) → tesis-içi + tesisler-arası normallik. Asıl cevap bu.
4. **Ölçeği sıkıştır** — çok-veri medyanı + cross-venue tutarlılık → ±13% daralt;
   on-site tek ölçüm opsiyonel rötuş.
5. **Recall'ı genişlet** — çok-tesis/ışık, GPU; far-band tiling'i tüm-maçta ölç.
6. **Ayrı hat:** takım (yelek-tek-yönlü + ilk-gol) + top (tespit).

## Tek cümle
Şu ana kadarki iş **bir tesiste sağlam ve dürüst bir temel**; sıradaki tek-en-değerli
hamle, bunu **oto-kalibrasyonla çok-tesis/çok-maça** taşımak — hem çan-eğrisi
sorusunu binlerce oyuncuyla cevaplar hem de ürünü ölçeklenebilir kılar.
