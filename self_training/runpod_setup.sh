#!/usr/bin/env bash
# RunPod 4090 — halısaha RF-DETR fine-tune ortam kurulumu (deterministik kısım).
# Lokalden /workspace'e coco_bundle.tar.gz scp edildikten SONRA pod'da çalıştır.
# Checkpoint'i (1.57GB) lokalden yüklemek yerine HF'den datacenter hızında çeker.
set -euo pipefail
cd /workspace

echo "===== [1/5] GPU kontrol ====="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

echo "===== [2/5] python deps (yerel sürümlerle birebir) ====="
pip install -q --upgrade pip
# torch genelde RunPod PyTorch image'ında hazır; rfdetr stack'i sabitliyoruz.
pip install -q "rfdetr[train,loggers]==1.7.1" "supervision==0.28.0" "transformers==5.10.2" "huggingface_hub" || {
  echo "transformers 5.10.2 çakışırsa rfdetr'ın istediğine bırak:"; pip install -q "rfdetr[train,loggers]==1.7.1" "supervision==0.28.0" "huggingface_hub"; }

echo "===== [3/5] Soccernet checkpoint (HF: julianzu9612/RFDETR-Soccernet) ====="
if [ ! -f /workspace/checkpoint_best_regular.pth ]; then
  python - <<'PY'
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download(repo_id="julianzu9612/RFDETR-Soccernet",
                    filename="weights/checkpoint_best_regular.pth")
shutil.copy(p, "/workspace/checkpoint_best_regular.pth")
print("checkpoint ->", "/workspace/checkpoint_best_regular.pth")
PY
else
  echo "checkpoint zaten var (scp ile gelmiş), atlanıyor."
fi
ls -lh /workspace/checkpoint_best_regular.pth

echo "===== [4/5] dataset bundle aç + rfdetr yapısına reorg ====="
[ -f /workspace/coco_bundle.tar.gz ] && tar xzf /workspace/coco_bundle.tar.gz -C /workspace
# reorg: coco/{train,val}.json + images/ -> dataset/{train,valid}/_annotations.coco.json
python - <<'PY'
import json, shutil
from pathlib import Path
src=Path("/workspace/coco"); dst=Path("/workspace/dataset")
for split,j in [("train","train.json"),("valid","val.json")]:
    coco=json.load(open(src/j)); d=dst/split; d.mkdir(parents=True,exist_ok=True)
    n=0
    for im in coco["images"]:
        s=src/"images"/im["file_name"]
        if s.exists(): shutil.copy(s, d/im["file_name"]); n+=1
    json.dump(coco, open(d/"_annotations.coco.json","w"))
    print(f"{split}: {n} kare -> {d}  (cats={[c['name'] for c in coco['categories']]})")
PY

echo "===== [5/5] hazır ====="
echo "Eğitim:  cd /workspace && python runpod_train.py dataset ft_out 20 2>&1 | tee /workspace/train.log"
