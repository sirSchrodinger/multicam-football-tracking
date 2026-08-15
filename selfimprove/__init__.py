"""selfimprove/ — QUALITY-FEEDBACK CONTROL-LOOP (kendi-kendini geliştiren modül).

Kapalı-döngü kontrol:  MEASURE -> DIAGNOSE -> MINE/IMPROVE -> GATE -> LEDGER -> tekrar.

Dürüstlük sözleşmesi (pazarlık edilemez):
  - ROUTE (yönlendirme) UCUZ proxy'lerden yapılır (match_quality: calib/identity/
    motion/detection-proxy). Bunlar her cycle, her sahada koşar.
  - GATE (model promosyonu) SADECE insan-doğrulamalı held-out GT'den okunur
    (recall_score 3-kör-sayıcı + drift_guard leave-one-stadium-out). Box-count
    proxy ASLA recall iddiası ya da promosyon gerekçesi olamaz.
  - "kendini geliştirdi" iddiası ANCAK drift_guard held-out'ta ölçülen iyileşme +
    regresyon-yok kanıtı LEDGER'a yazıldığında doğrudur. Aksi her şey "staged".

İki ray (graft sonrası):
  RAY A (yerel, EĞİTİMSİZ): detcache + recovery_search -> FarBandRecovery config'ini
    frozen insan-GT dev/test'te ara + config_gate -> promote_config. RunPod beklemez;
    bugün gerçek recall deltası (recover_far PROMOTE, far +%10.9 held-out-kanıtlı).
  RAY B (RunPod, GATED): mine.far_band_corpus -> improve.stage_runpod_job -> gate.decide
    (LOSO >=2 saha blind-GT). 2. saha GT gelene dek KİLİTLİ (mine.label_queue ile aç).

Modüller:
  ledger          — append-only JSONL + state.json (promoted weights/CONFIG + baseline)
  measure         — birleşik ScoreCard (proxy eksenler + held-out recall)
  diagnose        — kök-neden + 3-yol recall routing (recovery_search/mine+improve/gt_request)
  mine            — düzeltici korpus (Track B) + label_queue (2. saha GT-talebi)
  detcache        — 1x GPU forward -> saf-CPU config sweep cache (Track A makinesi)
  recovery_search — eğitimsiz config araması; dev/test frozen-GT (Track A)
  improve         — RunPod fine-tune iş-spec'i (GATED; yerelde EĞİTİM YOK)
  gate            — drift_guard sarmalayıcı (Track B LOSO) + config_gate (Track A frozen-GT)
  loop            — orkestratör: measure->diagnose->{improve-local|stage|gt-request|gate}->ledger
"""
from __future__ import annotations

__all__ = ["ledger", "measure", "diagnose", "mine", "detcache", "recovery_search",
           "improve", "gate", "loop"]
