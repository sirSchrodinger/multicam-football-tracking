# selfimprove/ — kendi-kendini geliştiren KONTROL-DÖNGÜSÜ (mimari)

Halısaha tespit/takip hattının kalitesini SÜREKLİ ölçen, en zayıf halkayı bulan,
düzeltici üreten, **insan-doğrulamalı held-out'ta gate'leyen** ve her cycle'ı
append-only bir deftere yazan kapalı-döngü.

```
   MEASURE ──▶ DIAGNOSE ──▶ ( RAY A: recovery_search ─┐
   (proxy +    (kök-neden    RAY B: mine→stage→gate ──┼─▶ GATE ──▶ LEDGER ──▶ (tekrar)
    held-out)   3-yol)       gt-request: label_queue ─┘   (truth)   (append-only)
```

Birincil hedef (Alperen): **far/karanlık band'de KAÇAN oyuncu = recall.** İkincil:
identity/calibration (insan-onaylı, loop otomatik değiştirmez).

## Dürüstlük sözleşmesi (pazarlık edilemez)

İki sinyal fiziksel olarak ayrı, iki ayrı gate:

- **PROXY (route):** `eval/match_quality.assess_match` -> `axes{detection,identity,
  motion,calibration}` + `limiting_factor`. Ucuz, GT'siz. SADECE *nereye bakacağını*
  söyler. `est_recall=box/14` ASLA recall iddiası ya da promosyon gerekçesi olamaz.
- **TRUTH (gate):** insan-GT. İki biçim:
  - Track A: `eval/recall_score.reconcile` 3-kör-sayıcı sayı-GT (`recall_val/
    score_summary.json`) üzerinde `config_gate` (frozen dev/test).
  - Track B: `self_training/drift_guard.evaluate_held_out` LOSO (>=2 saha) +
    `gate_or_rollback` (per-saha çökme kapısı).
  "Geliştirdi" YALNIZ buradan ledger'a yazılır.

## İki ray

### RAY A — yerel, EĞİTİMSİZ (RunPod beklemez, bugün gerçek delta)
`detcache` + `recovery_search` + `gate.config_gate` + `ledger.promote_config`.

FarBandRecovery POST-PROC config'ini (önce `recover_far` bayrağı) frozen insan-GT'ye
karşı arar. Config DEV(5 kare)'te seçilir, TEST(3 kare)'te gate edilir (anti-overfit) +
k-fold çapraz-delta + `gt_n` her satıra. KEEP koşulu: `far_recall↑ AND rest_recall
düşmez AND FP kötüleşmez`. Promote reversible (`state.promoted_config`); production
(`export_tracks.py --recover-far`) bunu okur. Kazanç **saha-spesifik** etiketlenir.

- count-GT yolu (BUGÜN gate'lenir): `recover_far` ON/OFF etkisi sayı-GT'den doğrudan
  (`n_base_covered` vs `+n_tile_covered`). RF-DETR fine-tune YOK, GPU YOK, uydurma YOK.
- box-GT yolu (KİLİTLİ): `tile_thresh/up/clahe` gibi ince knob'lar kutu-konumlu GT
  ister; `detcache.apply` kutuyu üretir ama sayı-GT ayırt edemez -> nokta-GT gelene
  dek `needs_point_gt` (sahte knob-delta YASAK).

`detcache.build` = 1× GPU forward (thr=0.15, far-band crop × up × clahe) -> ham kutu
parquet; `apply` = saf-CPU filtre/NMS/dedup (yüzlerce config ucuz). GPU paralel-cleanup
ile yarışır -> cache 1× kurulur, sweep CPU'da.

### RAY B — RunPod fine-tune (GATED, fırsatçı)
`mine.far_band_corpus` -> `improve.stage_runpod_job` -> (RunPod 4090) -> `gate.decide`
(LOSO >=2 saha). Track A tavana vurunca + LOSO hazırsa detektör tavanını yükseltir.
**2. saha blind-GT gelene dek KİLİTLİ**: `decide -> blocked`, promosyon YASAK
(sahte "fine-tune ile geliştirdik" üretmez). Yerelde GPU EĞİTİM YOK (1650Ti 4GB).

### gt-request — kilidi kıran adım
`mine.label_queue`: 2. saha (`KıbrısDorukHalıS`) track'lerinden EN BELİRSİZ kareleri
(deficit + far-yoğun + tek-parite + düşük-conf) öne koyar -> sınırlı insan bütçesi en
çok bilgi taşıyan karelere harcanır -> 3-kör-sayıcı blind-GT topla -> `gt_venues`'a
ekle -> Track B gate açılır.

## Router tablosu (diagnose 3-yol; context = loop state)

| limiting_factor | truth durumu | -> action | ray |
|---|---|---|---|
| detection (far<0.85) | recover_far promote değil + frozen-GT var | recovery_search | A |
| detection (far<0.85) | Track A tüketildi + LOSO hazır (>=2 saha) | mine.far_band_corpus -> stage | B |
| detection (far<0.85) | gt_venues<2 | gt_request (label_queue) | - |
| identity (teleport/frag) | - | stitch_tune (loop dışı; ayrı held-out) | - |
| motion->calib (off-field) | - | recalibrate_flag (insan-onaylı) | - |

`diagnose(scorecard, context=None)`: context YOKsa (tek-arg, geriye-dönük uyum) recall
-> Track B çerçevesi. context VARsa 3-yol ayrışır.

## Ledger şeması (append-only JSONL + state.json)

`_state/ledger.jsonl` — her cycle BİR satır (rollback'ler de KALIR; negatif=bilgi).
`_state/state.json` — canlı durum: `promoted_weights` (Track B), `promoted_config`
(Track A, reversible), `baseline_metrics` (LOSO), `gt_venues`, `cycle_seq`.

Kurallar: (a) her cycle bir satır. (b) `gt_n`+`venue` her recovery satırına ->
güven kalibre, abartılmaz. (c) `reversible:true` -> her KEEP tek satırla geri alınır.
(d) "ilerleme" = held-out TRUTH'un MUTLAK değeri (delta zinciri değil). (e) box-count
recall şişirme YAPISAL olarak imkânsız (gate yalnız count-GT/LOSO okur).

## Komutlar

```
python -m selfimprove.cli status                  # durum
python -m selfimprove.cli run --mode measure      # ölç + teşhis (model değiştirmez)
python -m selfimprove.cli run --mode improve-local# RAY A (eğitimsiz, frozen-GT gate)
python -m selfimprove.cli run --mode gt-request   # 2. saha etiket kuyruğu
python -m selfimprove.cli run --mode stage        # RAY B iş-spec (RunPod, GATED)
python -m selfimprove.cli run --mode gate --candidate <w> --candidate-metrics <json>
python -m selfimprove.loop <mode>                 # eşdeğer düşük-seviye giriş
pytest selfimprove/test_loop.py -q                # dürüstlük invariantları
```

## Bilinen kör-noktalar (dürüst)

- **N=8, tek-saha (Çankaya).** Track A held-out 3 kare -> istatistiksel zayıf;
  k-fold + gt-request ile büyür. Her satıra `gt_n` yazılır. En büyük dürüstlük açığı.
- **Pseudo-label kaçanı öğretemez (Track B).** corpus base'in *bulduğu* track'lerden;
  kaçırdığı far oyuncu girmez. Mitigasyon: build far-band tiling-kurtarılan kutuları
  zorunlu katar; held-out'ta görülmezse mine gevşetme YASAK.
- **CLAHE iki-uçlu.** Gece-karanlıkta FP getirebilir (default OFF, ÖLÇÜLMEMİŞ);
  config_gate precision-guard (fp_delta) yakalamalı, box-GT gelince gate'lenir.
- **`eval/recall_eval.py score` ÖLÜ KOD** (`recall_val/gt.json` okur, dosya YOK).
  Kanonik = `recall_score.py` -> `score_summary.json` (measure + recovery_search doğru
  olanı okur). `cmd_frames`/`cmd_detect` sağlam; Track A bunların ürettiği frozen
  kareleri/detections.json'u kullanır.
