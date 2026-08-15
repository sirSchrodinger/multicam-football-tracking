"""improve.py — fine-tune iş-spec'i (GATED arayüz; YERELDE EĞİTİM YOK).

KISIT (Alperen): GPU eğitim yerelde YAPILMAZ (1650Ti 4GB yetersiz). improve adımı
bu yüzden bir RunPod 4090 İŞ-SPEC'i emit eder: korpus bundle + base ağırlık + epoch
+ çalıştırma komutu. Gerçek train self_training/runpod_train.py içinde, kiralanan
pod'da koşar; sonuç ağırlık geri indirilince gate (drift_guard) devreye girer.

Bu modül torch import ETMEZ; sadece manifest + komut yazar (CPU). LEDGER'a 'staged'
olarak girer -> hiçbir iyileşme iddia edilmez ta ki gate KEEP diyene kadar.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

JOBS_DIR = ROOT / "selfimprove/_state/jobs"
BUNDLE = ROOT / "self_training/data/coco_bundle.tar.gz"


def stage_runpod_job(corpus_manifest: dict, *, epochs: int = 20,
                     base_weights: str = "checkpoint_best_regular.pth",
                     reason: str = "") -> dict:
    """RunPod fine-tune iş-spec'i yaz (dosyaya) + deploy komutunu döndür.

    corpus_manifest: mine.far_band_corpus çıktısı.
    Döndürür: {job_id, spec_path, deploy_cmd, status:'staged'}.
    """
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    job_id = time.strftime("ft_%Y%m%d_%H%M%S")
    spec = {
        "job_id": job_id,
        "kind": "rfdetr_finetune",
        "base_weights": base_weights,
        "epochs": epochs,
        "corpus": {
            "bundle": str(BUNDLE),
            "n_images": corpus_manifest.get("n_images"),
            "n_boxes": corpus_manifest.get("n_boxes"),
            "train_venues": corpus_manifest.get("train_venues"),
            "val_venues": corpus_manifest.get("val_venues"),
        },
        "reason": reason,
        "recipe": "self_training/runpod_train.py (Mean-Teacher EMA; BSD/MIT backbone)",
        "deploy": "bash self_training/deploy_runpod.sh <pod_ip> <ssh_port>",
        "download": "scp root@<ip> -P <port> /workspace/ft_out/checkpoint_best_regular.pth "
                    "models/weights/checkpoint_ft.pth",
        "after_download": "selfimprove.gate ile held-out LOSO değerlendir + gate_or_rollback",
        "status": "staged",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    spec_path = JOBS_DIR / f"{job_id}.json"
    spec_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False))
    return {"job_id": job_id, "spec_path": str(spec_path),
            "deploy_cmd": spec["deploy"], "status": "staged",
            "bundle_present": BUNDLE.exists()}
