"""RF-DETR fine-tune (halısaha gece/amatör recall) — RunPod 4090.

Yerel pipeline ile birebir mimari: RFDETRLargeDeprecated, Soccernet checkpoint.
train() imzası sürümler arası oynadığı için desteklenen kwargs'ı introspect edip
sadece onları geçiyoruz (kör hardcode yok). num_classes uyumu: model 4 sınıfla
(Soccernet) kuruluyor; dataset tek 'person' → rfdetr uyumsuz class-head'i
yeniden başlatır (fine-tune normu).

Kullanım: python runpod_train.py [dataset_dir=dataset] [out=ft_out] [epochs=20]
"""
import inspect, sys
from pathlib import Path

DATASET = sys.argv[1] if len(sys.argv) > 1 else "dataset"
OUT     = sys.argv[2] if len(sys.argv) > 2 else "ft_out"
EPOCHS  = int(sys.argv[3]) if len(sys.argv) > 3 else 20
WEIGHTS = "/workspace/checkpoint_best_regular.pth"

import torch
print(f"torch {torch.__version__} cuda={torch.cuda.is_available()} "
      f"dev={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}", flush=True)

from rfdetr import RFDETRLargeDeprecated

# Modeli Soccernet checkpoint'iyle kur (yerel export ile birebir: num_classes=4).
model = RFDETRLargeDeprecated(pretrain_weights=WEIGHTS, num_classes=4)

# rfdetr train(self, **kwargs) -> pydantic TrainConfig ile doğrular; signature SÜZME YANLIŞ
# (params={'kwargs'} görünür -> hepsi düşer -> dataset_dir missing). Doğrudan geç + extra
# alan reddinde kademeli kırp.
want = dict(dataset_dir=DATASET, epochs=EPOCHS, batch_size=4, grad_accum_steps=4,
            lr=1e-4, lr_encoder=1.5e-4, output_dir=OUT, num_workers=4, early_stopping=False)
tiers = [want,
         dict(dataset_dir=DATASET, epochs=EPOCHS, batch_size=4, grad_accum_steps=4, lr=1e-4, output_dir=OUT),
         dict(dataset_dir=DATASET, epochs=EPOCHS, output_dir=OUT)]
last = None
for kw in tiers:
    try:
        print("train kwargs denemesi:", kw, flush=True)
        model.train(**kw); last = None; break
    except TypeError as e:                       # beklenmeyen kwarg -> daha minimal tier
        last = e; print("TypeError, minimal tier'a düşülüyor:", e, flush=True); continue
    except Exception as e:
        # pydantic extra-field reddi de minimal'e düşsün; başka hata YENIDEN fırlat
        msg = str(e).lower()
        if "unexpected" in msg or "extra" in msg or "permitted" in msg or "validation" in msg:
            last = e; print("config reddi, minimal tier'a düşülüyor:", e, flush=True); continue
        raise
if last is not None:
    raise last
print(f"\nBİTTİ → checkpoint dizini: {Path(OUT).resolve()}", flush=True)
print("indir: scp root@<ip> -P <port> /workspace/%s/checkpoint_best*.pth ." % OUT, flush=True)
