#!/usr/bin/env python3
"""train_kp2 — kp modelini GENELLEŞTİRMEK için ağır augmentasyon + daha çok val.
Fark (train_kp.py'den): geometrik aug (affine, gerçek örneğe), güçlü renk (gamma/hue/
blur), cutout (oyuncu-occlusion), daha büyük model, daha çok val sahası, gerçek-ağır
batch. Hedef: held-out gerçek sahada 97px'i düşür + yeni-saha genelleşmesi.
Env: BASE(40) STEPS(5000) BATCH(6) NREAL(4) NVAL(8)
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F, json
from calib_train.kp_data import real_kp, synth_kp, TW, TH
from calib_train.make_seg_data2 import ORDER, real_ok
from calib_train.model_kp import KeypointUNet, dsnt, NK, KP_NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(HERE, "calib_solved.json")))
BASE = int(os.environ.get("BASE", "40")); STEPS = int(os.environ.get("STEPS", "5000"))
BATCH = int(os.environ.get("BATCH", "6")); NR = int(os.environ.get("NREAL", "4"))
NVAL = int(os.environ.get("NVAL", "8"))
OUT = os.path.join(HERE, "kp_ckpt2"); os.makedirs(OUT, exist_ok=True)

reals = [n for n in ORDER if n in C and not C[n].get("bad") and real_ok(n)]
rng0 = np.random.RandomState(0); rng0.shuffle(reals)
val_reals = reals[:NVAL]; train_reals = reals[NVAL:]
print(f"gerçek {len(train_reals)} train + {len(val_reals)} val | {NK} kp | BASE={BASE} aug=AĞIR", flush=True)


def aug_color(img):
    img = img.astype(np.float32) * np.random.uniform(0.68, 1.28) + np.random.uniform(-20, 20)
    img = np.clip(img, 0, 255)
    g = np.random.uniform(0.72, 1.45); img = 255.0 * np.clip(img / 255.0, 0, 1) ** g
    img = np.clip(img, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + np.random.randint(-13, 14)) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * np.random.uniform(0.7, 1.3), 0, 255)
    img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    if np.random.rand() < 0.3:
        img = cv2.GaussianBlur(img, (0, 0), np.random.uniform(0.6, 1.8))
    return img


def cutout(img, nmax=3):
    h, w = img.shape[:2]
    for _ in range(np.random.randint(0, nmax + 1)):
        cw, ch = np.random.randint(20, 90), np.random.randint(30, 130)
        x, y = np.random.randint(0, max(1, w - cw)), np.random.randint(0, max(1, h - ch))
        img[y:y + ch, x:x + cw] = np.random.randint(0, 90)
    return img


def aug_geo(img, hm, v):
    h, w = img.shape[:2]
    ang = np.random.uniform(-9, 9); sc = np.random.uniform(0.82, 1.2)
    tx = np.random.uniform(-0.07, 0.07) * w; ty = np.random.uniform(-0.07, 0.07) * h
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc); M[0, 2] += tx; M[1, 2] += ty
    img2 = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    hm2 = np.stack([cv2.warpAffine(hm[k], M, (w, h)) for k in range(hm.shape[0])])
    v2 = v.copy()
    for k in range(len(v)):
        if v[k] and hm2[k].max() < 0.3:
            v2[k] = 0
    return img2, hm2, v2


def to_t(img, hm, v):
    return (torch.from_numpy(img.transpose(2, 0, 1).astype(np.float32) / 255.),
            torch.from_numpy(hm.astype(np.float32)), torch.from_numpy(v.astype(np.float32)))


def get_batch(step):
    xs, ys, vs = [], [], []
    for i in range(BATCH - NR):
        img, hm, v = synth_kp(step * BATCH + i)
        img = aug_color(img)
        xs_, y_, v_ = to_t(img, hm, v); xs.append(xs_); ys.append(y_); vs.append(v_)
    for _ in range(NR):
        n = train_reals[np.random.randint(len(train_reals))]; r = real_kp(n)
        if r is None:
            r = synth_kp(step * 7 + 13)
        else:
            img, hm, v = aug_geo(r[0].copy(), r[1].copy(), r[2].copy())
            img = cutout(aug_color(img))
            r = (img, hm, v)
        xs_, y_, v_ = to_t(*r); xs.append(xs_); ys.append(y_); vs.append(v_)
    return torch.stack(xs), torch.stack(ys), torch.stack(vs)


def gt_xy(y):
    B, K, H, W = y.shape; idx = y.reshape(B, K, -1).argmax(-1)
    return torch.stack([(idx % W).float() / (W - 1), (idx // W).float() / (H - 1)], -1)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"; print("device", dev, flush=True)
    net = KeypointUNet(b=BASE).to(dev); opt = torch.optim.Adam(net.parameters(), 1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS); t0 = time.time(); best = 1e9
    for step in range(STEPS):
        x, y, v = get_batch(step); x, y, v = x.to(dev), y.to(dev), v.to(dev)
        hm, sig = net(x); ph = torch.sigmoid(hm); w = 1.0 + 30.0 * y
        hm_loss = (w * (ph - y) ** 2).mean()
        kp, _ = dsnt(hm); g = gt_xy(y); vm = v.unsqueeze(-1)
        coord_loss = (vm * (kp - g) ** 2).sum() / (v.sum().clamp(min=1) * 2)
        conf_loss = F.binary_cross_entropy(sig, v)
        loss = hm_loss + 0.15 * coord_loss + 0.1 * conf_loss
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        if step % 100 == 0:
            print(f"step {step}/{STEPS} loss {loss.item():.4f} crd {coord_loss.item():.4f} "
                  f"{(time.time()-t0)/max(1,step+1)*1000:.0f}ms/it", flush=True)
        if step % 500 == 0 and step > 0:
            net.eval(); errs = []
            with torch.no_grad():
                for n in val_reals:
                    r = real_kp(n)
                    if r is None:
                        continue
                    x1, y1, v1 = to_t(*r); hm1, _ = net(x1[None].to(dev)); kp1, _ = dsnt(hm1)
                    g1 = gt_xy(y1[None].to(dev))[0]; p1 = kp1[0].cpu().numpy(); gg = g1.cpu().numpy(); vv = v1.numpy()
                    for i in range(NK):
                        if vv[i] > 0:
                            dx = (p1[i, 0] - gg[i, 0]) * TW; dy = (p1[i, 1] - gg[i, 1]) * TH
                            errs.append((dx * dx + dy * dy) ** 0.5)
            e = float(np.mean(errs)) if errs else 9e9
            tag = ""
            if e < best:
                best = e; torch.save(net.state_dict(), os.path.join(OUT, "kp_unet2.pth")); tag = " *KAYDET*"
            print(f"  [val] ort piksel hata={e:.1f}px (best {best:.1f}) n={len(val_reals)}{tag}", flush=True)
            net.train()
    torch.save(net.state_dict(), os.path.join(OUT, "kp_unet2_last.pth"))
    print(f"BİTTİ {time.time()-t0:.0f}s best={best:.1f}px (baseline 97px)", flush=True)


if __name__ == "__main__":
    main()
