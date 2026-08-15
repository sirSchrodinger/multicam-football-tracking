#!/usr/bin/env python3
"""Otomatik-calib keypoint detektörü eğitimi — sentetik çizgi-maskesi on-the-fly.

Girdi: çizgi-maskesi (görünümden bağımsız, sim-to-real köprüsü). Çıktı: 15 keypoint
ısı-haritası. Çıkarımda gerçek kareden turf-maskeli beyaz-çizgi maskesi -> bu model
-> keypoint -> homografi -> otomatik flatten + 2D (manuel tık YOK).

Kullanım: python train.py [steps=4000] [out=calib_ckpt]
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import torch
import torch.nn.functional as F
from calib_train.synth_gen import sample, KP_NAMES
from calib_train.model import CalibUNet

H, Wd = 288, 512
NKP = len(KP_NAMES)
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
OUT = sys.argv[2] if len(sys.argv) > 2 else "calib_ckpt"
BATCH = int(os.environ.get("BATCH",16))
SIGMA = 4.0
os.makedirs(OUT, exist_ok=True)


def make_heatmaps(kp_list):
    """kp_list: batch of {name:(u,v,vis)} -> (B,NKP,H,W) gaussian targets."""
    yy, xx = np.mgrid[0:H, 0:Wd]
    hm = np.zeros((len(kp_list), NKP, H, Wd), np.float32)
    for b, KP in enumerate(kp_list):
        for i, name in enumerate(KP_NAMES):
            u, v, vis = KP[name]
            if vis:
                hm[b, i] = np.exp(-((xx - u) ** 2 + (yy - v) ** 2) / (2 * SIGMA ** 2))
    return torch.from_numpy(hm)


def gen_batch(seed0):
    masks, kps = [], []
    for j in range(BATCH):
        m, KP, _ = sample(seed0 + j, H, Wd)
        masks.append((m.astype(np.float32) / 255.0)[None])
        kps.append(KP)
    x = torch.from_numpy(np.stack(masks))
    y = make_heatmaps(kps)
    return x, y


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev} torch={torch.__version__}", flush=True)
    net = CalibUNet(NKP).to(dev)
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
    net.train(); t0 = time.time(); seed = 0
    for step in range(STEPS):
        x, y = gen_batch(seed); seed += BATCH
        x, y = x.to(dev), y.to(dev)
        pred = net(x)
        # KEYPOINT-AĞIRLIKLI MSE: ısı-haritasının %99'u arka plan -> düz MSE "her yere 0"
        # diyerek hile yapıyordu (loss düşük ama tepe~0.03). Pozitif bölge 80x ağırlık.
        w = 1.0 + 80.0 * y
        loss = (w * (pred - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 100 == 0:
            print(f"step {step}/{STEPS} loss {loss.item():.5f} "
                  f"({(time.time()-t0)/max(1,step+1)*1000:.0f}ms/it)", flush=True)
        if step % 1000 == 0 and step > 0:
            torch.save(net.state_dict(), f"{OUT}/calib_unet.pth")
    torch.save(net.state_dict(), f"{OUT}/calib_unet.pth")
    print(f"BİTTİ -> {OUT}/calib_unet.pth  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
