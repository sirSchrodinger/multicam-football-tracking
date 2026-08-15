#!/usr/bin/env bash
# Halısaha RF-DETR fine-tune'u RunPod pod'una tek komutla deploy + başlat.
# Kullanım:  bash self_training/deploy_runpod.sh <pod_ip> <ssh_port>
#   (RunPod console -> pod -> Connect -> "SSH over exposed TCP":
#    ssh root@<ip> -p <port> -i ~/.ssh/id_ed25519  satırındaki ip ve port)
set -euo pipefail
IP="${1:?pod_ip gerekli}"; PORT="${2:-22}"
KEY="$HOME/.ssh/id_ed25519"
SSH="ssh -i $KEY -p $PORT -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 root@$IP"
SCP="scp -i $KEY -P $PORT -o StrictHostKeyChecking=accept-new"
cd "$HOME/halisaha-stats"

echo "===== [0] bağlantı + GPU testi ====="
$SSH 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader' || { echo "BAĞLANAMADI / GPU yok"; exit 1; }

echo "===== [1] dataset bundle (781MB) + scriptler upload ====="
$SCP self_training/data/coco_bundle.tar.gz root@"$IP":/workspace/
$SCP self_training/runpod_setup.sh self_training/runpod_train.py root@"$IP":/workspace/

echo "===== [2] ortam kurulumu (deps + HF checkpoint + reorg) ====="
$SSH 'cd /workspace && bash runpod_setup.sh'

echo "===== [3] eğitim DETACHED başlat (SSH kopsa da sürer) ====="
$SSH 'cd /workspace && nohup python runpod_train.py dataset ft_out 20 > /workspace/train.log 2>&1 & echo "PID $!"; sleep 4; echo "--- ilk loglar ---"; head -20 /workspace/train.log'

echo ""
echo "İZLE:    $SSH 'tail -f /workspace/train.log'"
echo "İNDİR:   $SCP root@$IP:/workspace/ft_out/checkpoint_best_regular.pth models/weights/checkpoint_ft.pth"
