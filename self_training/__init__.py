"""self_training — DEFERRED self-improvement loop for the halısaha CV stack.

Bu paket SADECE su iki onkosul saglandiktan sonra calistirilir:
  1) Elle (manuel-H ile) kalibre edilmis K >= ~5 ayri saha (camera_id) birikmis olmali.
  2) Donmus (frozen) bir leave-one-stadium-out validasyon seti kurulmus olmali.

Asla derin bir ag ile baslama. Once klasik-CV (pitch/auto_calib.py) + manuel kalibrasyon
yeterince saha biriktirsin; sonra bu paket pseudo-label madenciligi yapar.

Akis (recete README.md icinde):
    harvest  ->  pseudolabel  ->  drift_guard  ->  (rented 4090) train  ->  drift_guard
    (ham video)  (pseudo_label)   (regresyon kapisi)         (stub)        (kabul/rollback)

LISANS KURALI (pazarlik edilemez): primary path yalniz OpenCV(BSD)+scipy(BSD)+numpy.
Ogrenilmis bir segmentasyon agi eklenirse BSD/MIT backbone (SegFormer-b0 / TVCalib-MIT)
kullan. ASLA PnLCalib (GPL-2.0) veya Ultralytics (AGPL-3.0) ship etme.
"""

from . import pseudo_label  # canonical
from . import drift_guard
from . import train

# Interface-spec dosya adi `pseudo_label.py`; kodlama gorevi `pseudolabel.py` istedi.
# Ikisi de import edilebilsin diye `pseudolabel` ince bir re-export shim'idir.
from . import pseudolabel  # noqa: F401  (alias)

__all__ = ["pseudo_label", "pseudolabel", "drift_guard", "train"]
