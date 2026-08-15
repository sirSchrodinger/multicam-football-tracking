# Top konumu — oyuncu-yerleşiminden çıkarım: derin deneme + dürüst sonuç

Alperen: "top konumu belirlemede saha içi yerleşimi kullanabilirsin." Denendi; **bu klipte
güvenilir DEĞİL.** Bu dosya kanıtı ve neden top-tespiti gerektiğini belgeler.

## Denenen proxy'ler (14 core, continuous_state pozisyon+hız)
| proxy | frame-frame hız (medyan / p90) | sonuç |
|---|---|---|
| hız-ağırlıklı oyuncu merkezi | 38 / 90 m/s | çok gürültülü (laggy değil ama zıplıyor) |
| hız-yakınsama (oyuncular topa koşar → ray-kesişimi) | 110 / 322 m/s | aşırı gürültü, fiziksel-dışı |

Gerçek top: pas ~15-25 m/s, ama trajektori PÜRÜZSÜZ (ani sıçrama yok). Proxy'lerin frame-frame
sıçraması bunun 2-15 katı → top/aksiyon-merkezi sinyali **off-ball koşu gürültüsünün altında**.
(Herkes topa koşmaz: boşa koşu, markaj, geri çekilme → yakınsama bozulur.)

## Sonuç
Oyuncu-yerleşimi tek başına topu vermez. Seçenekler:
1. **Top-tespiti** (doğru yol): RF-DETR'de top sınıfı var ama bu çözünürlük/gece'de top ~birkaç
   piksel, çoğu karede görünmez. Yine de tespit + interpolasyon, proxy'den iyi.
2. **Çok-ağır zaman-yumuşatma** (~1-2s): kararlı ama artık "top" değil, "oyun-yoğunluk bölgesi"
   (territory pressure) — düşük-bilgi, oyuncu-merkezine yakın. Ayrı/dürüst etiketle sunulabilir.
3. **Ani ivme sinyali**: topla etkileşen oyuncu ani hızlanır/yavaşlar; ileri faz, yine gürültülü.

## DÜRÜSTLÜK
Sahte ball-proxy SHIPLENMEZ (takım-renk gibi). Continuous_state pozisyon haritası SAĞLAM ve
kendi başına değerli; top ayrı bir katman, top-tespiti olmadan "yaklaşık ama mantıklı" eşiğini
geçmiyor. Belgelendi, uydurulmadı.
