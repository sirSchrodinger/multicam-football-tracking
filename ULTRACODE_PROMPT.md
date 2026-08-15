# Halısaha Maç Analizi AI — Ana Görev Promptu

> Bu görevi **ultracode** ile, çok-ajanlı bir workflow olarak işle. Aşağıda projenin
> tüm bağlamı, kısıtları, yol haritası ve bu session'da üretmeni istediğim somut
> çıktılar var. Hiçbir önceki konuşmaya erişimin yok — ihtiyacın olan her şey burada.

---

## 0. Rolün ve çalışma bağlamı

Türkiye'deki halısahaların (5v5/7v7 amatör futbol) sabit güvenlik/maç kameralarından
gelen videoları işleyip **profesyonel seviyeye yakın maç analizi** üreten bir bilgisayarlı
görü (CV) sistemi geliştiriyoruz. Nihai hedef: bunu **sosyalhalisaha gibi platformlara
B2B bir eklenti/ürün** olarak satmak.

Kod burada: `~/halisaha-stats`. Mevcut durum:
- **Tespit:** RF-DETR Large (özel eğitilmiş checkpoint: `models/weights/checkpoint_best_regular.pth`, 4 sınıf, CUDA). Oyuncu recall ~%85, kapsama %84-98.
- **Takip:** ByteTrack (`supervision` kütüphanesi üzerinden). Smoke test: `track_smoke.py`.
- **Diğer:** `probe_detect.py` (tespit ölçümü), `scan_cams.py`, `models/inference.py`, `raw/` ham klipler, `frames/`, `annotated/`, `overlay_demo_30s.mp4`.
- **Gözlem:** Uzak oyuncular 55-80px, conf 0.3-0.5 — yani **küçük/uzak oyuncu + amatör çözünürlük** asıl zorluk.
- **Sıradaki adım:** stitcher + **homografi** (piksel → saha koordinatı). Bütün gerçek istatistikler bunun arkasında kilitli.

**Donanım/bütçe gerçeği:** Eğitme gücü sınırlı. GPU ve RAM **kiralanacak** (RunPod ~$0.34/saat RTX 4090; Vast.ai daha ucuz; A100 ~$0.5-0.6/saat). Maç başına inference maliyeti kuruşlar (~$0.2-0.5). Yani darboğaz compute değil; **mühendislik + veri + sahaya özel kalibrasyon**.

---

## 1. Ürün vizyonu

Bir halısaha maçından, mevcut kamerayı **olduğu gibi kabul ederek** (sahibine "kameranı yükselt" demeden) şu çıktıları üret:

**Konum katmanı (omurga, yüksek doğruluk):** ısı haritası, koşu mesafesi, sprint sayısı, hız profilleri, bölge hâkimiyeti, sahiplik %, oyuncu boyu tahmini (kalibrasyondan).

**Top/olay katmanı (daha zor):** pas sayısı, pas isabeti, şut, top kimde — top takibi + oyuncu-top etkileşiminden.

**Premium katmanlar (en sona):** maç-içi an analizi (taktik hata: geri koşmama, atılmayan açık pas), 2D→3D stilize tekrar (Dünya Kupası tarzı ama tek kameradan "yeterince iyi" replika), simültane/real-time işleme.

---

## 2. Pazarlık edilemez mimari sürücüler

Bunlar tasarımı baştan belirler; sonradan eklenecek detay değiller:

1. **Ham video ~15 günde siliniyor.** "Önce büyük veri topla, sonra eğit" YOK. Sistem **sürekli çalışan bir harvest hattı** olmak zorunda: maçlar olurken çek, silinmeden frame/track çıkar, biriktir. Eğitim verisini hasat etmek, ürünün maç işleme akışının yan ürünü.

2. **Elle etiketleme ölçeklenmez.** Pipeline'ın yüksek-güven track'leri **pseudo-label** olarak kullanılır (self-training). ZORUNLU muhafız: pseudo-label denetlenmezse kendi hatasını büyütür (drift). Her tur küçük bir **insan-doğrulamalı validation set**'inde sapma kontrolü.

3. **Genelleme = ürünün varlık şartı.** Bir sahaya kalibre edilmiş sistem diğerine bedavaya geçmez (homografi sahaya özel). Çözüm: birkaç sahayı **bizzat ölçüp** (sahibinden saha boyutları + referans noktaları) doğru homografi kur; bunları **otomatik çizgi-tabanlı kalibrasyonu** eğitmek/doğrulamak için seed yap. Gerçek başarı kriteri: **hiç dokunulmamış yeni bir sahada** otomatik kalibrasyon + rapor tutuyor mu.

4. **Satış offline batch ile.** Raporlar ertesi sabah hazır olsun yeter (B2B'ye fazlasıyla satılır). Maçlar gece boyunca **ucuz spot GPU**'da batch işlenir → akşam pik yükü maliyeti çöker. Real-time = pahalı **premium upsell**, varsayılan değil.

---

## 3. Dürüst mühendislik sınırları (overpromise = güven kaybı)

Her katmanın "neyi iddia edebilir, neyi edemez" sınırı net olmalı; ürünün güveni bu sınırda yaşıyor:

- **Kadraj dışı oyuncu:** Kısa kapanmalar (1-2 sn) Kalman + dizilim önceliğiyle doldurulur. Onbinlerce maçtan öğrenilmiş öncülükle re-entry tahmini **olasılıksal** veri verir — ısı haritası/dizilim/toplam için harika, ama o anki tartışmalı pozisyonda "tam buradaydı" diye **kesin iddia ettiremez**. Tek dar kamera tüm sahayı görmeyebilir; çift-kameralı sahalar bu kör noktayı çözer → **premium katman**.
- **Top:** En zayıf halka (minik, bulanık, kapanan). Doğrudan tespit (göründüğünde) + oyuncu hareketinden öncülük (görünmediğinde) **füzyonu**; biri diğerini doğrular, biri diğerinin yerine geçmez.
- **Takım ataması (yeleksiz):** Forma rengi kümeleme **birincil** + paslaşma grafiği/sahiplik sürekliliği **düzeltici**. Cold-start (ilk saniyeler) belirsiz, maç ilerledikçe oturur. İki takım da benzer renk + yeleksiz ise gerçekten zor.
- **Kimlik sürekliliği:** Re-ID olmadan "yeni oyuncu ekleniyor gibi" görünür. Sahada ~10-12 oyuncu/2 renk → pro kalabalığından **daha kolay** (renk + konum sürekliliğiyle kümele).
- **Maç-içi taktik eleştirisi:** En değerli ama en hassas katman. "Bu pası atmalıydı" demek tüm takım arkadaşlarının + sahadaki boşluğun bilinmesini ister (kısmen kadraj dışı). **Yanlış bir eleştiri, hiç eleştiri yapmamaktan daha hızlı güven öldürür.** En sona, konum istatistikleri kaya gibi olduktan sonra.

---

## 4. Fazlı yol haritası

- **Faz 0 — Veri zemini (en kritik):** 2-3 işbirlikçi saha ölç, homografi referansları al (hem kalibrasyon tohumu hem ilk demo). Otomatik ingest (15-gün penceresine karşı). Mevcut RF-DETR+ByteTrack çıktısını homografiye bağla → **saha-koordinatlı track**.
- **Faz 1 — Kendini besleyen etiketleme:** Pseudo-label biriktir, periyodik küçük fine-tune, **drift muhafızı** ile. Günde 4+ maç → döngü hızlı dolar; kısıt veri hacmi değil, **erişim + 15-gün penceresi**.
- **Faz 2 — Kimlik + takım + top:** Re-ID; renk+pas füzyonuyla takım; top doğrudan-tespit + hareket-öncülük füzyonu.
- **Faz 3 — İlk satılabilir ürün:** Konum istatistikleri + per-maç rapor (gece batch).
- **Faz 4 — Genelleme testi:** Görülmemiş sahada otomatik kalibrasyon tutuyor mu = gerçek kilometre taşı.
- **Faz 5 — Premium cila:** Maç-içi an analizi → simültane → 2D→3D tekrar.
- **Paralel iş tarafı:** Ar-Ge (SoccerNet SOTA taraması), iş modeli (per-maç maliyet + batch stratejisi + platform teklifi/rev-share + tek-alıcı riskine plan).

---

## 5. BU SESSION'DA ÜRETMENİ İSTEDİĞİM SOMUT ÇIKTILAR

Önceliklendirilmiş; üstten başla, derinlemesine git. Kod yazıyorsan `~/halisaha-stats` yapısına ve mevcut bağımlılıklara (rfdetr, supervision, opencv, numpy) uy.

1. **Homografi modülü (Faz 0 çekirdeği).** Bir sahanın ölçülü boyutları + kullanıcının işaretlediği (veya otomatik tespit edilen) saha çizgisi referans noktalarından piksel↔saha koordinat dönüşümü kuran, `track_smoke.py` çıktısındaki track'leri saha koordinatına çeviren bir modül tasarla ve yaz. Hem manuel-4-nokta hem otomatik-çizgi-tespiti yolunu planla; otomatik için SoccerNet'teki pitch-calibration yöntemlerini araştırıp en uygununu öner.
2. **Otomatik kalibrasyon araştırması.** SoccerNet / sports-field-registration literatüründen homografi + çizgi tespiti için uygulanabilir, hafif (kısıtlı GPU'ya uygun) yöntemleri tara, 2-3 aday karşılaştır, birini gerekçeyle seç.
3. **Self-training + drift-guard tasarımı.** Pseudo-label biriktirme, güven eşiği, periyodik fine-tune ve drift kontrolü için somut mimari + minimal iskelet kod.
4. **Konum istatistik katmanı.** Saha-koordinatlı track'lerden ısı haritası, mesafe, sprint, sahiplik %, bölge hâkimiyeti üreten rapor üreticisinin tasarımı.
5. **İş/maliyet modeli.** Per-maç inference maliyeti + akşam pik yükü için gece-batch/spot-GPU stratejisi + platforma B2B teklif taslağı + tek-alıcı ve in-house-DIY risklerine somut karşı plan.

Her çıktı için: ne yaptığını, hangi varsayımları aldığını, sınırlarını ve bir sonraki adımı açıkça yaz.

---

## 6. Bu görevi nasıl parçala (ultracode fan-out)

- **Araştırma fazı (paralel):** SoccerNet pitch-calibration, sports re-ID, ball-tracking-from-pose, trajectory-imputation — her biri ayrı ajan, yapılandırılmış özet döndürsün.
- **Tasarım fazı (jüri paneli):** Homografi/kalibrasyon için N bağımsız yaklaşım üret, bağımsız juri ajanlarıyla skorla, kazananı sentezle (runner-up'lardan iyi fikirleri devşir).
- **Uygulama fazı:** Seçilen tasarımı kodla (`~/halisaha-stats` içine), worktree izolasyonuyla çakışmasız.
- **Adversaryal doğrulama:** Her teknik iddiayı (örn. "bu yöntem amatör 720p'de tutar") çürütmeye çalışan bağımsız bir ajan; çoğunluk çürütürse iddiayı düşür.
- **İş departmanı (paralel, $0 metin):** maliyet modeli + teklif + risk planı.

---

## 7. Kalite çıtası ve anti-pattern'ler

- **Foundation-first.** Faz 0-1 (homografi + veri döngüsü) sağlam olmadan parlak katmanlara (3D, real-time, taktik eleştiri) ATLAMA. Cesaret = sıkıcı temele güvenmek.
- **Overpromise yok.** Bölüm 3'teki sınırları her iddiada koru. "Yüksek doğrulukla X" diyorsan hangi katmanda, hangi koşulda olduğunu söyle.
- **Uydurma yok.** Literatür yöntemi öneriyorsan gerçek olsun (citation doğrula); rakam veriyorsan kaynağı belli olsun. "Havada" / "obvious" / temelsiz iddia reddedilir.
- **Compute'u doğru yere harca.** Detector'ı sıfırdan eğitme; hazırı al, kıt GPU'yu **genelleme/kalibrasyon** problemine harca.
- **Gerçek kilometre taşı = görülmemiş saha.** "Kendi sahanda iyi sonuç" başarı değil; her değerlendirmeyi bu lensle yap.
- **Türkçe yaz**, teknik terimler İngilizce kalabilir.

---

## 8. Beklenen çıktı formatı

Workflow bitince bana şunları döndür: (a) yazılan/değişen dosyaların listesi ve ne yaptıkları, (b) homografi + kalibrasyon için seçilen yaklaşım ve gerekçesi, (c) araştırma bulgularının kaynaklı özeti, (d) self-training mimarisi, (e) iş/maliyet modeli, (f) net "sıradaki adım" listesi — hangi sahayı ölçmem, hangi GPU'yu kiralamam, ilk fine-tune'da ne beklemem gerektiği dahil.
