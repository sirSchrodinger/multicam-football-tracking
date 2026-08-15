#!/usr/bin/env bash
# 2-KAMERA GT fine-tune setup — TEMIZ VENV (base-env distutils/torch çakışmalarından kaçın).
# Ders: base env'de `pip install rfdetr` distutils-uninstall'da ölüyor; --ignore-installed
# torch'u cu130'a yükseltip CUDA'yı kırıyor. Çözüm: izole venv + torch cu124 PIN.
set -uo pipefail   # -e YOK: adımlar bağımsız
cd /workspace
echo "===== [1/4] dataset (pip'ten BAĞIMSIZ, önce) ====="
mkdir -p /workspace/dataset && tar xzf /workspace/coco_bundle_2cam.tar.gz -C /workspace/dataset
echo "===== [2/4] Soccernet checkpoint (curl, HF) ====="
[ -f /workspace/checkpoint_best_regular.pth ] || \
  curl -sL "https://huggingface.co/julianzu9612/RFDETR-Soccernet/resolve/main/weights/checkpoint_best_regular.pth" \
       -o /workspace/checkpoint_best_regular.pth
ls -lh /workspace/checkpoint_best_regular.pth
echo "===== [3/4] temiz venv + torch cu124 PIN + rfdetr ====="
python -m venv /workspace/venv
/workspace/venv/bin/pip install -q --upgrade pip
/workspace/venv/bin/pip install -q torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
/workspace/venv/bin/pip install -q "rfdetr[train,loggers]==1.7.1" "supervision==0.28.0" huggingface_hub
echo "===== [4/4] doğrula ====="
/workspace/venv/bin/python -c "import torch,rfdetr,json;print('OK torch',torch.__version__,'cuda',torch.cuda.is_available());print('train kutu',len(json.load(open('/workspace/dataset/train/_annotations.coco.json'))['annotations']))"
echo "hazır. Eğitim: /workspace/venv/bin/python runpod_train.py dataset ft_out 24"
