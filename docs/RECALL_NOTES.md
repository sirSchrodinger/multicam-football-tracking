# RECALL_NOTES — amatör gece halısaha tespit recall'ı (ÖLÇÜLDÜ, 2026-06-28)

Projenin #1 dürüstlük açığı kapandı. `detect/recall_qc.py` docstring'i diyordu ki:
*"hiçbir recall % iddiası DONDURULMUŞ insan-GT olmadan yapılmaz."* O GT yoktu —
artık var, ve recall **ölçüldü** (eyeball değil, çok-sayıcı kör insan-GT).

## Yöntem
- **Dondurulmuş doğrulama seti:** `raw/cankaya_cam2.mp4` (59 dk, 1080p) içinden 8
  kare (t = 1500–3450 s), tüm bölgeleri + zor far-küme durumlarını kapsayacak
  şekilde seçildi (`recall_val/frames/`). Vestler ancak t≈2400 sonrası çıkıyor
  (dinamik-vest içgörüsü doğrulandı).
- **Tespit:** `eval/recall_eval.py detect` — base RF-DETR (thr 0.30, MIN_H 25,
  NMS 0.6, `export_tracks` ile birebir) + far-band tiling recovery (ust-bant crop
  → 2× upscale → thr 0.40 re-detect → base ile foot-dedup). `recall_val/detections.json`.
- **Kör insan-GT:** 8 kare × 3 bağımsız sayıcı (skeptic / thorough / balanced
  lensleri), her biri önce ham görüntüden gerçek insanları KÖR sayar (kutuları
  görmeden), sonra base/tile/uncovered kapsamasını işaretler. 24 ajan
  (ultracode workflow). **API düşüşünde 1 sayıcı öldü (f44769:thorough) →
  tek-ajan backfill ile tamamlandı; her kare 3 sayıcı garanti edildi** (yarım
  GT üstüne sayı kurulmadı). Uzlaştırma: sayıcı-başına medyan
  (`eval/recall_score.py`).
- **Bölge:** görüntü-uzayı `foot_y` — far (<360, küçük/düşük-kontrast), rest
  (mid+near, büyük). Bkz. `docs/report/` (saha bölgeleri raporu).

## Sonuçlar (8 kare, 3 sayıcı, medyan-uzlaştırılmış)

| Bölge | GT insan | base recall | +far-band tiling |
|------|---------:|------------:|-----------------:|
| **FAR** (küçük/karanlık) | 92 | **68.5–76%** | **79–88%** (+11 puan) |
| **REST** (büyük/yakın)   | 15 | **%100** | — |
| **TOPLAM** | 107 | **72.9–79.6%** | **82.2–89.8%** |

- **Precision ~%100:** 24 etiketin tamamında `n_false_pos = 0` — dedektör oyuncu
  UYDURMUYOR; sadece kaçırıyor (recall problemi, precision değil).
- **Band** GT belirsizliğinden: düşük-kontrast (ambiguous) GT'leri paydaya
  katınca alt-sınır, çıkarınca üst-sınır.
- **Per-frame** (base → +tiling): f37307 %46→54, f52230 %62→69, f74615 %62→85,
  f82076 %77→92, f59692 %85→100, f67153 %86→93, f85807 %86→86, f44769 %77→77.
  Şekil: `docs/report/figures/recall.pdf`.

## Bulgular (dürüst)
1. **Eski "~%85" eyeball iyimsermiş.** Tek-kare göz-tahminiydi; titiz çok-kare GT
   base recall'ı **~%73**'e çekiyor. Bu doğru sayı.
2. **Tüm kayıp far-üçte-bir'de.** Near/mid (büyük oyuncu) **%100**. Kaçanlar
   küçük + düşük-kontrast (koyu giysi, karanlık file arka-planı), özellikle
   vestsiz erken karelerde (f37307 %46). Bu, kalibrasyon/zone notlarındaki
   "far = recall'ın zayıf noktası" hipotezini sayısal doğrular.
3. **Far-band tiling DOĞRULANDI.** Zaten yazılı ama GT'ye karşı hiç sınanmamış
   `FarBandRecovery` toplam recall'ı **+9–10 puan** (far'da +11) artırıyor, FP
   getirmeden (precision %100 kalıyor). Kazanç tam da base'in zayıf olduğu
   karanlık karelerde yoğunlaşıyor (f74615 +23, f59692 +15). **→ pipeline'a
   wire edilmeli** (`export_tracks --recover-far`, entegrasyon notları
   recall_qc.py'de hazır).

## Sınırlar
- **Pilot N=8 kare** (tek maç, tek kamera). Sayı yön-gösterici; ±birkaç puan
  oynar. GPU'da daha geniş set + farklı tesis/ışık ile genişletilmeli.
- **far/rest sınırı bulanık:** sayıcılar orta-boy oyuncuları bazen far bazen
  rest sayıyor (skeptic vs thorough). Bu yüzden **TOPLAM recall birincil**
  (sınır-sağlam); far/rest ayrı sayıları ±1-2 oynar. Toplam-kapsama sayıcılar
  arası neredeyse birebir aynıydı (yüksek güven).
- GT insan-yargısı (multimodal); far-küme yoğunken ±1 insan belirsizlik var,
  band buna izin veriyor.

## Sıradaki
1. **Tiling'i pipeline'a wire et** (recall_qc entegrasyon bloğu) + tüm maçta
   recall/FP'yi yeniden ölç.
2. **Karanlık-küme açığı:** CLAHE (recall_qc'de opsiyon var, ölçülmemiş) +
   pseudo-label self-training (far-third tiling tespitlerini eğitim sinyali yap).
3. Val seti GPU'da genişlet (farklı ışık/tesis); recall'ı tesis-başına raporla.

Reproduce: `eval/recall_eval.py {frames,detect}` → workflow GT →
`eval/recall_score.py` → `eval/recall_figure.py`. GT: `recall_val/gt_labels.json`.
