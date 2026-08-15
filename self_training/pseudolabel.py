"""pseudolabel.py — ince re-export shim.

Kodlama gorevi dosya adini `pseudolabel.py` istedi; interface spec ise
`pseudo_label.py` diyor. Tek bir gercek implementasyon (pseudo_label.py) olsun
diye bu dosya yalniz onu yeniden disa aktarir. Ikisi de import edilebilir:

    from self_training.pseudo_label import detection_pseudo_labels  # spec adi
    from self_training.pseudolabel  import detection_pseudo_labels  # gorev adi
"""
from .pseudo_label import (  # noqa: F401
    detection_pseudo_labels,
    line_seg_labels_from_H,
    line_seg_corpus_from_calib,
    reconstruct_bbox_xyxy,
    to_coco,
    to_yolo,
    last_summary,
    PERSON_CLASS_ID,
)

__all__ = [
    "detection_pseudo_labels",
    "line_seg_labels_from_H",
    "line_seg_corpus_from_calib",
    "reconstruct_bbox_xyxy",
    "to_coco",
    "to_yolo",
    "last_summary",
    "PERSON_CLASS_ID",
]
