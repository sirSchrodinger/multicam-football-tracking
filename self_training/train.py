"""train.py — lisans-temiz fine-tune girisi (STUB; DEFERRED).

Gercek egitim KIRALANAN 4090'da (RunPod/Vast.ai) calisir, lokal GTX 1650 Ti'de DEGIL.
Bu dosya egitim cagrisini bilincli olarak STUB birakir: cekirdek imza + recete +
lisans/EMA kurallari burada, agir bagimliliklar (torch/transformers) calistirma
aninda lazy yuklenir. Boylece paket GPU'suz import edilebilir.

LISANS (pazarlik edilemez):
  - Sadece BSD/MIT backbone: SegFormer-b0 (NVIDIA, non-commercial-clause'a DIKKAT;
    tercih MIT/Apache govde) ya da TVCalib-MIT iskeleti.
  - ASLA PnLCalib (GPL-2.0), ASLA Ultralytics (AGPL-3.0) bir kapali urunde.
  - SoccerNet-winner repo lisansi DOGRULANMADAN kullanilmaz.
"""
from __future__ import annotations

from typing import Any


def finetune_detector(corpus: Any, val: Any,
                      base_weights: str = "models/weights/checkpoint_best_regular.pth",
                      backbone: str = "rf-detr-nano",
                      ema: bool = True,
                      max_steps: int = 4000) -> str:
    """RF-DETR dedektorunu pseudo-label korpusunda ince-ayarla (STUB).

    corpus: pseudo_label.to_coco/to_yolo ciktilarinin (image+label) yolu/manifesti.
    val:    insan-dogrulamali leave-one-stadium-out seti (drift_guard.evaluate_held_out).
    Donen:  yeni agirlik dosyasinin yolu.

    Recete (README.md detayli):
      1) corpus'u Mean-Teacher ile egit (ema=True): teacher pseudo-label uretir,
         student gradyan alir, teacher = EMA(student) (drift_guard.ema_update mantigi).
      2) Her epoch sonu drift_guard.evaluate_held_out + gate_or_rollback.
      3) Gate KEEP derse agirligi yaz; aksi halde rollback.
    """
    raise NotImplementedError(
        "DEFERRED: dedektor fine-tune kiralanan 4090'da calisir. "
        "Once K>=~5 manuel-kalibre saha + donmus held-out val seti birikmeli. "
        "Recete: self_training/README.md")


def finetune_segnet(corpus: Any, val: Any,
                    backbone: str = "segformer_b0_BSD",
                    ema: bool = True) -> str:
    """Lisans-temiz cizgi-segmentasyon agini ince-ayarla (STUB; interface spec imzasi).

    corpus: pseudo_label.line_seg_labels_from_H / line_seg_corpus_from_calib ciktilari
            (frame .jpg + *_lineseg.png yogun GT maske ciftleri).
    val:    held-out sahalarda pitch-reprojection val seti.
    Donen:  yeni seg-agi agirlik yolu.

    "Bu pikselde cizgi var mi?" sorusu sahalar arasi "bu hangi keypoint?" sorusundan
    cok daha iyi transfer eder (D omurgasinin segmentasyon savi). Ama bu adim
    KLASIK-CV auto_calib yeterince saha biriktirmeden ASLA baslatilmaz.
    """
    raise NotImplementedError(
        "DEFERRED: seg-agi fine-tune kiralanan 4090'da, BSD/MIT backbone ile. "
        "PnLCalib(GPL)/Ultralytics(AGPL) YASAK. Recete: self_training/README.md")
