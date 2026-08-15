# Takım ayrımı — derin araştırma + dürüst sonuç

Çankaya cam2 (gece) klibinde **otomatik takım ayrımı GÜVENİLMEZ**. Bu dosya, neyin
denendiğini, kanıtı ve güvenilir yolu belgeler. (Alperen: "renkle olmaz, paslara göre
kıyasla" — araştırma onu doğruladı; ama pas/etkileşim sinyali de bu klipte zayıf.)

## Denenen yöntemler ve sonuç (14 CORE oyuncu üzerinde)

| yöntem | sonuç | neden |
|---|---|---|
| jersey hue-kmeans (HSV) | 12-7 / conf 0.30 | yelek sarı-YEŞİL → hue çim hue'suna yakın |
| sarılık (R+G)/2−B + gap-split | 10-4 / 2-12 | çim-yansıması + gece → koyu-şort pozitif okuyor |
| parlaklık V gap-split | 13-1 | dağılım sürekli, bimodal değil |
| doygunluk S | 5-9 | ayırt etmiyor |
| normalize-sarılık | 13-1 | aynı süreklilik |
| **seed-propagate** (uç-seed, çim-maskeli desc.) | **4-10** | desc. iki takımı 7-7 ayırmıyor (yapı yok) |
| **velocity-korelasyon** (ball-chase çıkarılmış) | 7-7 ama **stabilite %57** | spektral-denge artefaktı; ilk/ikinci yarı ≈ rastgele |

## Görsel doğrulama (en güçlü kanıt)
Atamayı gerçek kareye çizdim (scratchpad/team_check.png, team_seed.png): sahada
**çoğunlukla sarı-yeşil yelekli + birkaç koyu** oyuncu görünüyor — temiz 7/7 yok.

## En olası KÖK SEBEP (operatör teyidi gerek)
Yaygın amatör düzen: **bir takım tek-tip SARI yelek (~7), diğer takım kendi KARIŞIK
kıyafetleri (yeleksiz)**. O zaman "diğer takım"ın ortak renk imzası YOKTUR →
renk ayrımı ilkesel olarak imkânsız. Gözlem (10 parlak/yelek-veya-açık + 4 koyu) buna uyar.

## Güvenilir yollar (öncelik)
1. **Manuel-seed (ÖNERİLEN, ucuz):** kullanıcı her takımdan 1-2 oyuncu işaretler →
   relative appearance-benzerliğiyle yay. AMA "karışık takım" senaryosunda bu da
   sınırlı; en garantisi 14 oyuncuyu bir kez elle etiketlemek (tek seferlik, kesin).
2. **Yelek-tespiti tek-yönlü:** eğer bir takım tek-tip yelekse, SADECE yelek-takımı
   güvenle bul, gerisi "diğer". Dengeyi (≈7) prior olarak kullan, zorlama yok.
3. **Top-tespiti → gerçek pas-ağı:** "paslara göre"nin doğru hali; ama bu çözünürlükte
   top zor (ileri faz).
4. **Daha net klip:** iki ayrık bib rengi / gündüz → renk çalışır.

## DÜRÜSTLÜK
Sahte 7-7 ASLA shippetme. velocity-7-7 stabil olmadığı için (≈%57) takım çıktısı
"low-confidence / doğrulanmamış" diye işaretlenir veya manuel-seed beklenir. Mevcut
`detect/team_split.py` dengesizliği gizlemiyor (balance_note); bu doğru davranış.
