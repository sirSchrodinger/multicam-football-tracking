"""ÇOK-SAHA warp-fit DENETİMİ (cherry-pick YOK): her saha için 4-panel
[ orijinal | seg-maske | undistort(fisheye-out) | 2D-warp+TEMPLATE ].
Panel-4 warp'a metrik template (dikdörtgen+orta-çizgi+yuvarlak) bindirir → boyalı
beyaz çizgiler template'e OTURUYOR mu = zero-setup calib DÜRÜST fit testi.
Kullanım: python warp_audit.py            (tüm seg_auto_solved.json sahaları)
Çıktı: cand_warp/<venue>.jpg + cand_warp/_grid_*.jpg
"""
import sys, os, json, numpy as np, cv2, torch, torch.nn as nn
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from top_down_view import undimg, topdown, und_norm
L, S, M = 34.0, 20, 3.0
TW, TH = 512, 288

def cbr(i, o):
    return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True),
                         nn.Conv2d(o, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))
class UNet(nn.Module):
    def __init__(s, nin=3, no=2, b=24):
        super().__init__(); s.d1=cbr(nin,b);s.d2=cbr(b,b*2);s.d3=cbr(b*2,b*4);s.d4=cbr(b*4,b*8);s.p=nn.MaxPool2d(2)
        s.u3=nn.ConvTranspose2d(b*8,b*4,2,2);s.c3=cbr(b*8,b*4);s.u2=nn.ConvTranspose2d(b*4,b*2,2,2);s.c2=cbr(b*4,b*2)
        s.u1=nn.ConvTranspose2d(b*2,b,2,2);s.c1=cbr(b*2,b);s.h=nn.Conv2d(b,no,1)
    def forward(s, x):
        c1=s.d1(x);c2=s.d2(s.p(c1));c3=s.d3(s.p(c2));c4=s.d4(s.p(c3))
        x=s.c3(torch.cat([s.u3(c4),c3],1));x=s.c2(torch.cat([s.u2(x),c2],1));x=s.c1(torch.cat([s.u1(x),c1],1));return s.h(x)
dev = "cuda" if torch.cuda.is_available() else "cpu"
net = UNet().to(dev)
net.load_state_dict(torch.load(os.path.join(HERE, "seg_ckpt", os.environ.get("SEG_CK", "seg_unet_v3.pth")), map_location=dev))
net.eval()

def seg_masks(img):
    h, w = img.shape[:2]
    x = torch.from_numpy(cv2.resize(img, (TW, TH)).transpose(2, 0, 1).astype(np.float32)/255.)[None].to(dev)
    with torch.no_grad():
        p = torch.sigmoid(net(x))[0].cpu().numpy()
    return cv2.resize(p[0], (w, h)), cv2.resize(p[1], (w, h))

def label(im, txt, col=(0, 230, 255)):
    cv2.rectangle(im, (0, 0), (im.shape[1], 30), (25, 25, 25), -1)
    cv2.putText(im, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2); return im

def panels(path, rec):
    img = cv2.imread(path)
    if img is None:
        return None
    h, w = img.shape[:2]; cx, cy, s = w/2, h/2, w/2
    k1, k2 = float(rec["k1"]), float(rec["k2"]); Wp = float(rec.get("Wp", 18)); fit = float(rec.get("fit", -1))
    # param sınıra pinli mi (marjinal calib işareti)
    pin = (abs(k1-0.10) < 0.005 or abs(k1-0.22) < 0.005 or abs(Wp-17.3) < 0.05 or abs(Wp-19.0) < 0.05)
    P1 = label(img.copy(), os.path.basename(path).split('.')[0][:20])
    pb, pinm = seg_masks(img)
    seg = img.copy(); seg[pb > 0.4] = (0, 0, 255); seg[pinm > 0.4] = (0, 200, 0)
    P2 = label(seg, "seg (kirmizi=sinir yesil=ic)")
    und = undimg(img, k1, k2, cx, cy, s)
    P3 = label(und, f"undistort k1={k1:.2f} k2={k2:.2f}")
    top = topdown(img, rec, None)
    fitcol = (0, 220, 0) if (fit >= 0 and fit < 0.20 and not pin) else (0, 165, 255) if not pin else (0, 100, 255)
    P4 = label(top, f"2D-WARP+template fit={fit:.2f}{' PIN' if pin else ''}", fitcol)
    # eşit yükseklik
    Hh = 300
    def rz(im): return cv2.resize(im, (int(im.shape[1]*Hh/im.shape[0]), Hh))
    row = [rz(P1), rz(P2), rz(P3), rz(P4)]
    sep = np.full((Hh, 4, 3), 40, np.uint8)
    return np.hstack([row[0], sep, row[1], sep, row[2], sep, row[3]]), pin, fit

if __name__ == "__main__":
    solved = json.load(open(os.path.join(HERE, "seg_auto_solved.json")))
    os.makedirs(os.path.join(HERE, "cand_warp"), exist_ok=True)
    # path'i bul (label_frames veya unseen)
    def find(key):
        for d in ("label_frames", "unseen"):
            p = os.path.join(HERE, d, key)
            if os.path.exists(p):
                return p
        return None
    rows = []; meta = []
    for key in sorted(solved):
        p = find(key)
        if p is None:
            continue
        r = panels(p, solved[key])
        if r is None:
            continue
        strip, pin, fit = r
        cv2.imwrite(os.path.join(HERE, "cand_warp", key), strip)
        rows.append(strip); meta.append((key, pin, fit))
        print(f"{key[:22]:<22} fit={fit:.3f} {'PIN(marjinal)' if pin else ''}", flush=True)
    # grid'ler (12'şerli)
    wmax = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], wmax-r.shape[1], 3), 18, np.uint8)]) if r.shape[1] < wmax else r for r in rows]
    for gi in range(0, len(rows), 12):
        grid = np.vstack([np.vstack([r, np.full((5, wmax, 3), 60, np.uint8)]) for r in rows[gi:gi+12]])
        cv2.imwrite(os.path.join(HERE, "cand_warp", f"_grid_{gi//12}.jpg"), grid)
    npin = sum(m[1] for m in meta)
    print(f"\n{len(rows)} saha · {npin} param-PIN (marjinal-calib şüphesi) · grid: cand_warp/_grid_*.jpg", flush=True)
    json.dump([{"venue": m[0], "pin": bool(m[1]), "fit": m[2]} for m in meta],
              open(os.path.join(HERE, "cand_warp", "_meta.json"), "w"), indent=2)
