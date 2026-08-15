# Halısaha Tespit/Tracking/Re-ID Yol Haritası (SOTA, web-grounded + doğrulanmış)

29 Haz 2026 — 14-ajan research workflow (web-grounded) + adversaryal lisans/varlık doğrulaması.
Hepsi **ticari-kullanılabilir (Apache/MIT)** lisansla doğrulandı. Kurulum: sabit kamera, gece,
yapay çim, yelekli takım + karışık, ~12-16 oyuncu, 1920×1080@25fps, 4GB GPU.

## Öncelik sırası (kâr-marjı yüksek → düşük)

1. **Ölçüm harness'i — sn-trackeval / TrackEval (MIT).** HOTA = **DetA** (far-band recall) +
   **AssA** (swap-süzlük) + IDF1. Mevcut 3-kör-sayıcı GT'yi MOT JSON'a çevir, her değişikliği
   ölç. *Neden önce:* göz kararıyla yaptığım deneyler kopya-bug'la şişmişti (bkz dedup); ölçüm
   olmadan hiçbir iyileştirme doğrulanamaz.

2. **Takım-rengi HARD-GATE + tracker yükselt (BoT-SORT/OC-SORT, roboflow/trackers Apache, GMC kapalı).**
   - Takım-rengi gate: gövde-merkez crop → illumination-equalize (white-balance/CLAHE) → tek-yönlü
     sınıf (sarı-yelek vs değil) → association maliyetinde **çapraz-takım eşleşme = SONSUZ**.
     Alperen'in "yeşil-sarı çarpışınca yanlış kişiye gitmesin" kuralının appearance'tan BAĞIMSIZ
     mekanik garantisi. ~0 GPU, retrain yok.
   - BoT-SORT ReID kanalı + OC-SORT observation-centric re-update: gece-swap + jitter düşer; sabit
     kamerada GMC kapat. **DİKKAT: boxmot ve Ultralytics AGPL → onlardan ÇEKME; roboflow/trackers Apache.**

3. **Far-band küçük-nesne: Supervision InferenceSlicer (MIT) + NMM, SADECE uzak-1/3 ROI.**
   Naif/dedup-bug'lı tiling'i olgun slice+merge ile değiştir; dilim-sınırı çift-tespiti (teleport
   katkısı) sıfır-retrain kapanır. 4GB için tam-kare dilimleme YAPMA, sadece far ROI.

4. **Top: WASB-SBDT (MIT) ayrı heatmap top-kafası + Supervision BallTracker yörünge filtresi.**
   "Top insan sanıldı → teleport" bug'ını çözer (Alperen gözlemi; aspect-ratio FP'leri ölçüldü:
   ~457-671 kare). Ayrı kafa player-detektöründen compute çalmaz, 1.5M param 4GB-ucuz.
   *Hızlı ara-filtre:* h/w < 1.3 olan kutuları insan-değil say (kesin kazanç, hemen).

5. **RF-DETR fine-tune (gece/turf/yelek) — projected-label bootstrap + gece-aug.**
   Mevcut `self_training/pseudo_label.py` + `drift_guard.py` + `calib/export_line_labels.py` +
   `recall_val/` kullan. ~300-800 ELLE etiket SADECE zor bölgeye (far-band + çaprazlaşma + karanlık
   köşe). Albumentations (MIT): gamma-karartma + shot-noise + RandomSunFlare/Shadow; renk-jitter HAFİF
   (yelek/çim takım-rengini bozmasın). Kalıcı far-band recall artışı.

6. **Plato olursa detektör değiştir: D-FINE-S (Apache, ICLR'25, FDR=en iyi küçük-nesne lokalizasyon)
   → DEIMv2-Nano (HGNetv2 backbone, TEMİZ Apache).** Switching-cost düşük (aynı DETR+tracker hattı).

## Lisans mayınları (B2B — doğrulandı)
- ❌ **YOLOv11/v12/YOLO26 (Ultralytics): AGPL-3.0** → B2B'de kaynak-açma zorunlu veya ücretli Enterprise.
- ❌ **boxmot: AGPL.** ❌ **DEIMv2 S/M/L/X: DINOv3 backbone** (redistribution kısıtı; sadece Nano/Pico temiz).
- ❌ **Deep-EIoU: lisanssız** (re-implement et). ❌ **GOIS: README ticari yasak** (MIT rozetine rağmen).
- ✅ Temiz: RF-DETR / D-FINE / DEIMv2-Nano (Apache), supervision / torchreid-OSNet / WASB-SBDT /
  TrackEval (MIT), roboflow/trackers (Apache).

## Alperen'in isteklerine eşleme
- "tüm oyuncuları algıla (recall)" → #3 slicing + #5 fine-tune (asıl kaldıraç; tracker recall yaratmaz).
- "yeşil-sarı swap olmasın / özelliğe göre ID" → #2 takım-rengi hard-gate + OSNet.
- "olmayacak git-gel / jitter yok" → #2 OC-SORT + (mevcut) speed-clamp + smoothing.
- "top insan sanılmasın" → #4 WASB + aspect filtre.
- "N=12/14/16 maça göre" → kadro configurable (roster-tracker'da N parametresi).
- "SOTA / fine-tune / yüksek kalite / ticari" → hepsi Apache/MIT, doğrulandı.

## Riskler
- Far-band recall DETEKTÖR-bağımlı (tracker kaçanı yaratmaz) → asıl iş #3+#5.
- Takım-rengi gate floodlight/karanlıkta kırılgan → illumination-equalize + gövde-merkez crop ZORUNLU;
  Fener-forması sarı-edge manuel kural.
- 4GB: tam-kare SAHI+ağır ReID+SR 25fps tutmaz → far-ROI slice, OSNet-x0.25 her N. kare, fp16/TensorRT batch=1.
- Yayınlanmış AP/HOTA broadcast/gündüz → gece-halısaha sayıları KENDİ GT'mizde ölçülene kadar tahmini.
