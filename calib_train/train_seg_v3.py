#!/usr/bin/env python3
"""train_seg_v3 — seg modelini AYDINLATMA-DAYANIKLI yap (mavi-cast/glare/karanlık).
Fark: AĞIR renk-augmentasyonu (hue ±40, sat, gamma, güçlü brightness) TÜM örneklere
(synth+real). Böylece model 'yeşil'e değil YAPIYA bakar -> Cengiz(mavi)/Rize(karanlık)/
Ayazma(glare) genelleşir. Çıktı: seg_ckpt/seg_unet_v3.pth. Env: STEPS(4000) BATCH(6) NREAL(3)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from calib_train.make_seg_data import synth_sample, real_sample, real_ok, ORDER, J

TW, TH = 512, 288
STEPS = int(os.environ.get("STEPS", "4000")); BATCH = int(os.environ.get("BATCH", "6"))
NR = int(os.environ.get("NREAL", "3"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seg_ckpt"); os.makedirs(OUT, exist_ok=True)
VAL = {"AvanosHaliSaha.jpg", "KaynarcaAdaHaliSah.jpg", "KucukcekmeceIdmanY.jpg"}
reals = [n for n in ORDER if n in J and real_ok(n)]
train_reals = [n for n in reals if n not in VAL]; val_reals = [n for n in reals if n in VAL]
print(f"gerçek: {len(train_reals)} train + {len(val_reals)} val | AĞIR-renk-aug", flush=True)


def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))


class UNet(nn.Module):
    def __init__(s, nin=3, no=2, b=24):
        super().__init__(); s.d1 = cbr(nin, b); s.d2 = cbr(b, b*2); s.d3 = cbr(b*2, b*4); s.d4 = cbr(b*4, b*8)
        s.p = nn.MaxPool2d(2); s.u3 = nn.ConvTranspose2d(b*8, b*4, 2, 2); s.c3 = cbr(b*8, b*4)
        s.u2 = nn.ConvTranspose2d(b*4, b*2, 2, 2); s.c2 = cbr(b*4, b*2)
        s.u1 = nn.ConvTranspose2d(b*2, b, 2, 2); s.c1 = cbr(b*2, b); s.h = nn.Conv2d(b, no, 1)
    def forward(s, x):
        c1 = s.d1(x); c2 = s.d2(s.p(c1)); c3 = s.d3(s.p(c2)); c4 = s.d4(s.p(c3))
        x = s.c3(torch.cat([s.u3(c4), c3], 1)); x = s.c2(torch.cat([s.u2(x), c2], 1)); x = s.c1(torch.cat([s.u1(x), c1], 1))
        return s.h(x)


def color_aug(img):
    """AĞIR renk-aug: hue±40 (mavi-cast), sat, gamma, güçlü brightness, blur, kanal-karıştır."""
    img = img.astype(np.float32) * np.random.uniform(0.55, 1.4) + np.random.uniform(-30, 30)
    img = np.clip(img, 0, 255)
    g = np.random.uniform(0.6, 1.7); img = 255.0 * np.clip(img/255.0, 0, 1)**g
    img = np.clip(img, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + np.random.randint(-40, 41)) % 180        # mavi-cast dahil
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.5, 1.4), 0, 255)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if np.random.rand() < 0.25:
        img = cv2.GaussianBlur(img, (0, 0), np.random.uniform(0.6, 1.8))
    return img


def aug(img, m):
    if np.random.rand() < 0.5:
        img = img[:, ::-1].copy(); m = m[:, :, ::-1].copy()
    return color_aug(img), m


def to_t(img, m):
    x = torch.from_numpy(cv2.resize(img, (TW, TH)).transpose(2, 0, 1).astype(np.float32)/255.)
    y = torch.from_numpy(np.stack([cv2.resize(m[c], (TW, TH)) for c in range(2)]).astype(np.float32)/255.)
    return x, y


def get_batch(step):
    xs, ys = [], []
    for i in range(BATCH-NR):
        img, m = synth_sample(step*BATCH+i); img = color_aug(img)   # synth'e de renk-aug
        x, y = to_t(img, m); xs.append(x); ys.append(y)
    for _ in range(NR):
        n = train_reals[np.random.randint(len(train_reals))]; img, m = real_sample(n); img, m = aug(img, m)
        x, y = to_t(img, m); xs.append(x); ys.append(y)
    return torch.stack(xs), torch.stack(ys)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)
    net = UNet().to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS); t0 = time.time(); best = 1e9
    for step in range(STEPS):
        x, y = get_batch(step); x, y = x.to(dev), y.to(dev)
        pred = net(x); w = 1.0+30.0*y
        loss = (w*F.binary_cross_entropy_with_logits(pred, y, reduction="none")).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if step % 100 == 0:
            print(f"step {step}/{STEPS} loss {loss.item():.4f} {(time.time()-t0)/max(1,step+1)*1000:.0f}ms/it", flush=True)
        if step % 500 == 0 and step > 0:
            net.eval(); iou = 0
            with torch.no_grad():
                for n in val_reals:
                    img, m = real_sample(n); x1, y1 = to_t(img, m)
                    p = (torch.sigmoid(net(x1[None].to(dev)))[0].cpu().numpy() > 0.4)
                    gt = y1.numpy() > 0.4
                    inter = (p & gt).sum(); uni = (p | gt).sum()+1e-6; iou += inter/uni
            iou /= len(val_reals)
            tag = ""
            if -iou < best:
                best = -iou; torch.save(net.state_dict(), f"{OUT}/seg_unet_v3.pth"); tag = " *KAYDET*"
            print(f"  [val] IoU={iou:.3f}{tag}", flush=True); net.train()
    torch.save(net.state_dict(), f"{OUT}/seg_unet_v3_last.pth")
    print(f"BİTTİ {time.time()-t0:.0f}s best_IoU={-best:.3f}", flush=True)


if __name__ == "__main__":
    main()
