# Vision-Grounded Saha Kalibrasyonu — Sentez (2026-07-01)

Otomatik `seg → 4-çizgi → solve` pipeline'ının BOWTIE / çöküş verdiği **14 sahaya**
ajanlar **gözle** baktı, standart-saha şablonunu oturttu, ortak-lens undistort + homografi
ile kalibre etti. Otomatik pipeline bu 14 sahada **%0** başarılıydı. Aşağısı dürüst sonuç.

## 1. Kaç saha gözle kurtarıldı?

| Metrik | Sonuç |
|---|---|
| Toplam saha | 14 |
| `sanity_pass = true` (dış-sınır SANE, bowtie YOK) | **14 / 14 (%100)** |
| `visual_quality = good` (dış-sınır + iç-işaretler oturuyor) | **7 / 14 (%50)** |
| `visual_quality = rough` (dış-sınır oturuyor, iç-işaretler kayık) | 7 / 14 (%50) |
| Otomatik pipeline başarısı (aynı 14 saha) | 0 / 14 (%0) |

- **good (7):** idx 12, 13, 16, 22, 28, 29, 36
- **rough (7):** idx 0, 1, 9, 10, 19, 31, 33

**İki seviyeli kazanım var, karıştırmamak şart:**
1. **Dış-sınır / bowtie kurtarma: 14/14.** Her sahada saha-dikdörtgeni SANE oturdu, çöküş yok.
   Otomatik %0'dan buraya geçiş tamamen gözle-etiketlemenin eseri.
2. **Tam (iç-işaret dahil) kalibrasyon: 7/14.** Center-circle + halfway + penalty-box gerçek
   markalara oturan saha yarısı.

İterasyon: ortalama ~2.3 deneme/saha (1–4). En kritik düzeltme tekrar tekrar **oryantasyon**
oldu: "goller sol/sağ mı yoksa üst/alt mı" yanlış etiketlenince center-circle boş çime düşüyor;
köşeleri yeniden etiketleyince (idx 28, 13, 22) her şey yerine oturdu. Bu, otomatik seg
pipeline'ının **yapamadığı** ayrım — gören bir ajanın getirdiği asıl katma değer.

## 2. Hangi köşe-tipleri zordu?

Üç bağımsız zorluk ekseni, çoğu sahada birden fazlası üst üste:

- **Occluded köşeler (en yaygın).** Yakın köşeler (BL/BR) neredeyse her sahada frame-dışı veya
  ön-plandaki **file/çit/kale-çerçevesi/bina/yaprak** arkasında. Çözüm: touchline + goal-line
  yönlerini uzatıp kesişimi **extrapole etmek** (insan gibi). idx 10'da yaprak sağ yarının
  TAMAMINI kapatıyordu → sağ köşeler saf tahmin, doğrulanamaz.
- **Fisheye-kavisli köşeler.** Saha kenarı köşeye doğru kavislenir; köşenin "gittiği yer"
  tahmin edilir. Ortak-lens (k1=0.167, k2=0.240) bunu kısmen düzeltir ama head-on'da yetmez.
- **Karanlık (gece) köşeler.** idx 36 (çok karanlık) ve 19/16'da köşe + iç-işaretler agda
  kayboluyor; metrik ölçek/oryantasyon tam doğrulanamıyor.

**rough'un kök-nedeni köşe değil:** Dış-sınır 4 köşesi doğru seçildiğinde bile (idx 0,1,9,19,31,33)
iç-işaretler kayıyor çünkü:
- **(a) Model limiti:** head-on geometride tek-düzlemsel homografi + ortak-lens, dış-sınır VE
  iç-işaretleri AYNI ANDA oturtamıyor. Far köşeleri daraltıp center-circle'ı oturtursan far-touchline
  gerçek sınıra ulaşmıyor — trade-off. Bu klasik bowtie/foreshortening-eksikliği imzası.
- **(b) Şablon limiti (gerçek hata değil):** araç jenerik 12m/6m kutu çiziyor; gerçek halısaha
  ceza-sahası oranı farklı → magenta kutu "küçük/kaymış" görünüyor ama bu kalibrasyon hatası değil
  (idx 16, 22, 29, 31 dürüstçe not düştü).

## 3. "Limit yöntemdeydi, bilgide değil" tezi doğrulanıyor mu?

**Kısmen — ve dürüst ayrım önemli.**

**DOĞRULANAN kısım (güçlü):** Otomatik pipeline'ın çöküş nedeni **bilgi eksikliği değil yöntem
darlığıydı.** Aynı pikseller, aynı lens, aynı homografi-çözücü ile, sadece köşeleri/oryantasyonu
**gören bir aktör** etiketleyince **14/14 SANE dış-sınır** çıktı (otomatik %0). seg→4-çizgi
pipeline'ı görünmeyen-köşe extrapolasyonunu ve standart-saha önbilgisini kullanmadığı için
geometri-limitli sahalarda bowtie veriyordu; vision-grounded etiketleme tam da bunu sokuyor.
**Vision-grounded kapsama, dış-sınır/oryantasyon için otomatiğin kesinlikle ÜSTÜNDE.**

**DOĞRULANMAYAN kısım (overclaim olur):** "Tam çözüldü" denemez. rough yarısı (7/14) gerçek bir
**ikinci limiti** ortaya çıkardı ki bu artık daha-iyi-köşe-seçmeyle çözülmüyor: head-on fisheye'da
tek-homografi modeli iç-işaret sadakatini veremiyor. Bu yöntem/bilgi değil **model-kapasitesi**
limiti — per-saha fisheye katsayısı veya çizgi-tabanlı (sadece köşe değil) kalibrasyon gerekiyor.

**Net verdict:** Tez "bowtie çöküşü = yöntem limiti, gören aktör çözer" iddiası için **doğru ve
kanıtlı (14/14 sane)**. "Standart-saha bilgisini sokmak her şeyi çözer" genişletmesi için
**eksik kanıtlı**: iç-işaret sadakati head-on'da hâlâ açık problem.

## 4. Ölçek planı (1000s saha)

Üç seçenek tek başına yetmez; **bootstrap pipeline** olarak birleştir:

**A. Alperen etiketler** — en yüksek kalite, ÖLÇEKLENMEZ. İnsan darboğazı. Rolü: **altın-küme
denetimi** (good iç-işaret GT'si + zor vakaların final onayı), 1000s için ana iş gücü DEĞİL.

**B. Ben-ajan (vision) etiketler** — Max planında ~$0 marjinal, concurrent sınırsız. Dış-sınır SANE
**%100 güvenilir**, iç-işaret ~%50 head-on'da rough. Rolü: **toplu dış-sınır GT üretimi** +
oryantasyon (goller LR/TB) etiketi. 14→1000s'e taşınabilir.

**C. Yarı-otomatik köşe-detektörü** — ÖLÇEKLENEBİLİR omurga. Şu mimari:
1. **Bootstrap GT:** B-ajan 14→birkaç-yüz sahayı etiketler (sane dış-sınır). 7 good + Alperen-onaylı
   küme iç-işaret GT'si olur.
2. **Detektör eğit:** 4 köşe (BL/TL/BR/TR) regresyonu + **oryantasyon başlığı** (goller LR vs TB —
   ayrı, ucuz, discrete; bu sahaların asıl çuvallama eksenini ayıklar) + **occlusion/güven skoru**.
3. **Inference + routing:** detektör köşe önerir; düşük-güven (head-on / ağır-occluded / karanlık —
   yani rough-prediktörleri) vakalar **B-ajan veya Alperen review**'a düşer. Yüksek-güven otomatik geçer.
4. **Head-on iç-işaret düzeltmesi:** rough-kümesi için köşe-detektörünün üzerine **çizgi-tabanlı
   kalibrasyon** (iç-işaretleri de kısıt olarak kullanan, per-saha barrel serbest) — model-kapasitesi
   limitini kapatan tek yol.

**Öncelik sırası:** (1) B-ajanla 200-300 saha sane-dış-sınır + oryantasyon etiketle → (2) köşe+oryantasyon
detektörü eğit, hard-case routing ile → (3) Alperen yalnız flagged + altın-küme denetler → (4) head-on
rezidüel için çizgi-tabanlı kalibrasyonu wire et. Böylece insan eforu O(1000)'den O(zor-vakalar)'a düşer.

---
*Veri: 14-saha gözle-kalibrasyon ajan turu (sanity 14/14, good 7/14). Araç: `calib_train/night/vcal_tool.py`.*
