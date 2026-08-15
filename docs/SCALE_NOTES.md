# Ölçek (metre) kilidini DÜRÜSTÇE açma — kaynaklar, hata, karar ağacı

Tek sabit kameradan üst-bakış homografisi **şekli** (aspect + perspektif) verir;
**mutlak ölçeği** vermez. Homografi `findHomography(img_pts, world_pts)` ile kurulur
ve `world_pts`'in birimi neyse çıktı o birimdedir. Cankaya'da saha ölçüsü
**yayınlanmamış** ve tesis **kapalı** (uydu tabandan ölçemez) → mutlak metre bir
**varsayımdan** gelmek zorunda. Bu dosya o varsayımı ve hatasını belgeler.

> Değişmez kural: `scale_anchor` doğrulanmadıkça birim `relative_m`'dir; metre yalnızca
> **etiketli** verilir (`m (approx ±N%)`). Etiketsiz çıplak `m`/`km/h` ve standart-snap
> üzerinden `sprint` sayısı **yasaktır** (bkz. `stats/topdown_stats.py` honesty bloğu).

## Ölçek kaynakları ve beklenen hata

| Kaynak | Hata (mesafe) | Notlar |
|---|---|---|
| **Yayınlı/ölçülü** (operatör metre verir veya kale ağzı 3 m çapraz-kontrol) | ~%1-2, **kesin** | En iyi. `scale_anchor.verified=True` → çıplak `m`, sprint serbest. |
| **Uydu/Maps — açık saha** | ~%3-5 | Touchline'lar uydudan izlenebilir; hata = görüntü çözünürlüğü + georeferans + ayak-noktası düzlem varsayımı. |
| **Uydu/Maps — kapalı saha** | ~%5-10 **+ bias** | Çatı tabanı gizler; çatı ayak-izi ≠ saha; eğik görüntü paralaksı. Cankaya buradadır → uydu GÜVENİLMEZ. |
| **Oyuncu-boyu metroloji** (bu repo, ÖNERİLEN) | ~±13% (band, veri-sürümlü) | **Sahadan/kaleden BAĞIMSIZ.** Sahadaki ~22k insan ~1.75m ölçü çubuğu. `pitch/height_scale.py`. Aşağıda. |
| **Standart-boyut snap** (bu repo, v3) | ~%5-10 (band ±10%) | Saha "standart" VARSAYILIR (Alperen: ölçüler oynak → güvenilmez); fit-aspect en yakın ayrık boyutu seçer. Geri-plan/çapraz-kontrol. |

## Oyuncu-boyu metroloji (ÖNERİLEN, `pitch/height_scale.py`) — neden daha doğru

Saha ve kale ölçüleri **oynak** (Alperen) → 25×45 snap ve 3m-kale çapası ikisi de varsayım.
Ama sahadaki **on-larca insan EVRENSEL ~1.75 m** ölçü çubuğu. Tek-görüntü metroloji:

1. QA-geçmiş zemin homografisi `H` natural-camera (kare piksel, ana-nokta merkez) varsayımıyla
   **ayrıştırılır** → odak `f`, poz `R,t`, **kamera merkezi C**. (Zhang; `decompose_ground_homography`.)
2. Temiz **dik** tespitler (conf>0.6, alt-kırpık değil, dik en-boy) için tam kamera projeksiyonu
   `P=K[R|t]` ile her oyuncunun **boyu rel_m'de çözülür** (foot→ground; head ray ile dikey kesişim).
3. **Robust medyan-boy := 1.75 m** → ölçek `c` (m/rel_m). Mutlak metre böyle gelir; hiçbir saha/kale
   ölçüsü varsayılmaz.

**Çankaya cam2 sonucu (22k tespit):** medyan-boy 1.83 rel_m, **CV 0.085** (gerçek insan-boyu
dağılımı gibi sıkı → yöntem doğru), `c=0.957`, **saha ≈ 32.5×17.2 m**, **kamera 3.28 m** (halısaha
köşe-kamerası için fiziksel ✓). Yani **relative_m birimleri zaten ≈ gerçek metre** (±13%);
katalog-snap'in 46×24'ü ~%40 fazla tahminmiş.

**Band ±~13% nereden:** (a) ortalama-boy priori [1.70–1.80] (~±3%); (b) **tek-görüntü odak
belirsizliği** — `|r1|=|r2|` norm-eşitliği undistort kusuru yüzünden hiçbir `f`'te tam sağlanmaz
(kalıcı anizotropi), `r1·r2=0` ise temiz minimumlu ama `f` yine ±%15 oynak → alan 31×16 ↔ 40×21.
Tutarlılık için undistort+poz+projeksiyon **aynı f** (= calib K'sı, H bununla kuruldu).

**Sanity kapıları (fail → metrik İDDİA EDİLMEZ, relative_m kalır):** kamera yüksekliği 2–8 m,
boy CV < 0.20, ≥200 temiz tespit. `--scale-height` ile pipeline'a girer; `scale_anchor.kind='player_height'`,
`approximate=true verified=false` → rapor **'m (approx ±13%)'**.

## Standart-boyut snap (v3) — ne yapar, hatası nedir

Kanonik üreteç + okuyucu: **`pitch/scale_snap.py`** (`make_v3()` üreten,
`scale_state()` okuyan). `calib/make_standard_v3.py` artık buna delege eden
ince bir kabuktur. Tüketiciler (`topdown_stats`, `topdown_viz`) yalnız
`scale_state()`'i çağırır — JSON'u hangi şema yazmış olursa olsun (generator
şeması ya da eski JSON aliasları) aynı şekilde sınıflandırılır.

`pitch/scale_snap.make_v3()`:
1. QA-geçmiş calib'ten **fitlenmiş aspect** = L/W okunur (Cankaya: 34/18 = **1.889**).
   Bu aspect 4 köşe tıklamasından türetilmiş **gerçek geometri** — katalogdan güvenilir.
2. Aspect, format kataloğundan en yakın ayrık boyutu seçer
   (7v7: `20x40 / 25x45 / 30x50`; aspect 1.889 → **25x45**, çünkü 1.80'e |0.089| < 2.0'a |0.111|).
3. **İzotropik** (geometrik-ortalama) ölçek `s = √((L_kat/L₀)·(W_kat/W₀))` ile dünya
   noktaları yeniden ölçeklenir → **fitlenmiş aspect KORUNUR**, hata iki eksene eşit
   bölünür. Cankaya: s = **1.3558**, asserted dims = **46.1 × 24.4 m** (ikisi de
   25x45'in ±10% bandında: L∈[40.5,49.5]✓, W∈[22.5,27.5]✓).
4. H **aynı** undistorted görüntü noktaları + ölçekli dünya noktalarıyla yeniden fitlenir.
   İzotropik dünya-ölçeklemesi H tarafından **tam absorbe** edilir → reprojection (piksel)
   QA değişmez: **v2 median 9.875 px → v3 median 9.875 px** (doğrulandı).

### Neden izotropik (default), anizotropik değil
Fitlenmiş aspect 1.889, katalog 25x45'in aspect'i 1.80. dims'i **tam** 45x25 yapmak
(anizotropik) aspect'i 1.889→1.80'e **zorlar** = ~%5 **uydurma** anizotropik bozulma
(X ve Y'de farklı ölçek hatası). İzotropik snap bunu yapmaz: veriden gelen aspect'e
dokunmaz, yalnızca katalogdan **mutlak ölçeği** alır. `--anisotropic` ile açılabilir
ama default değildir (honesty). Anizotropik bias kalan kalemdir; izotropik tek
çarpan (~1.356) tüm mesafeleri **düzgün** ölçekler.

### Kalan hata kalemleri (v3'te ±10% bandın içinde)
- **Boyut belirsizliği**: 7v7 sahası 38-50 m × 25-30 m aralığında; 45×25 yalnızca medyan tahmin.
- **Ayak-noktası düzlem varsayımı**: oyuncu boyu → ayak projeksiyonu ~%2-5 (uzak-uçte artar).
- **Uzak üçte-bir piksel-fakirliği**: köşe-montaj uzak yarıyı sıkıştırır (per-zone QA işaretler).

## Karar ağacı — hangi ölçek kaynağı?

```
Operatör/tesis sahayı metre verdi mi?  (veya kale ağzı 3 m net görünüyor mu?)
├─ EVET → ölç/çapraz-kontrol et → scale_anchor.verified=True → çıplak 'm', sprint serbest
└─ HAYIR
   └─ Saha AÇIK mı (uydu tabanı görüyor)?
      ├─ EVET → Maps'ten touchline ölç (~%3-5) → scale_anchor (kind='satellite') → 'm (approx ±5%)'
      └─ HAYIR (kapalı; Cankaya) → uydu GÜVENİLMEZ (%5-10+bias)
         └─ Standart-boyut snap (pitch/scale_snap.py make_v3) → 'm (approx ±10%)', sprint YOK
            (default v2/relative_m korunur; v3 yan-yana sunulur)
```

## Cankaya cam2 — gerçek sonuç (doğrulandı)

`calib/cankaya_cam2_v3.json` (default v2 DEĞİŞMEDEN durur), `pitch/scale_snap.py`
ile **yeniden üretildi** → JSON, üreteç ve bu doküman **tek** yöntem+ölçekte uyumlu
(izotropik geomean; eski lsq 1.33784 JSON'u terk edildi):
- catalog **25x45**, asserted **46.1×24.4 m**, scale_factor **1.35582**, QA median **9.875 px** (gate ✓).
- Örnek oyuncu mesafesi (clip2400 player-keyed, foot→pitch re-projeksiyon, w_conf-ağırlıklı, gap-gate):

| player_id | n_obs | v2 `relative_m` | v3 `m (approx ±10%)` | oran |
|---|---|---|---|---|
| 0 | 2679 | 78.10 | 105.89 | 1.356 |
| 1 | 2624 | 43.31 | 58.72 | 1.356 |
| 2 | 2603 | 57.69 | 77.78 | 1.348 |
| 3 | 2484 | 34.15 | 46.30 | 1.356 |
| 4 | 2405 | 31.74 | 42.83 | 1.349 |

İzotropik snap → tüm oyuncular **aynı** çarpanla (1.356) ölçeklenir; eksen-bağımlı
bozulma yok (oran ufak sapmalar veri-uyarlamalı gap-cap'in birkaç segmenti farklı
kesmesinden). **Ham (ağırlıksız) mesafe oranı tam 1.35582** (std ~8e-4); ağırlıklı
mesafe de aynı çarpanı izler çünkü konum-güven ağırlığı (`w_conf`) **kanonik
relative çerçevede** hesaplanır — sabit bir ölçek-varsayımı tespit kalitesi güvenini
değiştirmez. v3 mesafeleri `m (approx ±10%)` etiketiyle sunulur, asla çıplak `m`.

### En ucuz "verified"e terfi yolu
Tek bir **kale ağzı 3 m** çapraz-kontrolü (operatör kale direkleri arasını ölçer
ya da bilinen kale genişliği) snap'i doğrular: `scale_anchor.verified=True`
(`approximate=False`) → birim çıplak **`m`** (~%1-2), sprint sayımı açılır. ±10%
snap bandı, ölçülen tek bir mesafeyle ~%1-2'ye iner. Buna kadar varsayılan
**relative_m** kanonik kalır; v3 yalnız opt-in `m (approx ±10%)` sunar.

## Bilinen entegrasyon tuzağı (KRİTİK)

`*_stitched.parquet` / `*_player.parquet` içine `pitch_x/pitch_y` **v2 ile pişirilir**.
`topdown_stats.prepare()` disk `pitch_x` NaN değilse onu **yeniden kullanır** — yani v3
calib geçilse bile baked v2 koordinatları kullanır (sessizce yanlış ölçek). v3 ile
çalıştırmak için `prepare()`/`generate_topdown_report()` `foot_x/foot_y`'den **yeniden
projekte** etmeli (bkz. `force_reproject`). Doğrulanmış bug: v2/v3 prepare çıktıları
disk-reuse yüzünden aynı span (33.19×21.51) verdi; foot'tan re-projeksiyon doğru
1.356 çarpanını verdi.
