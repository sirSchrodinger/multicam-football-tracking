"""ZERO-SETUP dürüst test: seg-auto calib (SARI) vs manuel calib (YEŞİL) ORİJİNAL karede.
Sarı, yeşil kadar boyaya oturuyorsa zero-setup ÇALIŞIYOR. Held-out sahalar işaretli.
cand/verify_segauto.jpg"""
import os, json, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
SEG = json.load(open(os.path.join(HERE, "seg_auto_solved.json")))
MAN = json.load(open(os.path.join(HERE, "calib_solved.json")))
LF = os.path.join(HERE, "label_frames"); L = 34.0
HELDOUT = {"AvanosHaliSaha.jpg", "KaynarcaAdaHaliSah.jpg", "KucukcekmeceIdmanY.jpg"}  # seg2 held-out

def m2px(Xm, Ym, k1, k2, H, cx, cy, s):
    Hi = np.linalg.inv(np.array(H, float)); Xm = np.asarray(Xm, float); Ym = np.asarray(Ym, float)
    den = Hi[2, 0]*Xm+Hi[2, 1]*Ym+Hi[2, 2]
    un = (Hi[0, 0]*Xm+Hi[0, 1]*Ym+Hi[0, 2])/den; vn = (Hi[1, 0]*Xm+Hi[1, 1]*Ym+Hi[1, 2])/den
    ru = np.sqrt(un*un+vn*vn); rd_g = np.linspace(0, 2.6, 5000); ru_g = rd_g*(1+k1*rd_g*rd_g+k2*rd_g**4)
    rd = np.interp(ru, ru_g, rd_g); sc = np.divide(rd, ru, out=np.ones_like(ru), where=ru > 1e-9)
    return cx+un*sc*s, cy+vn*sc*s

def draw(img, rec, col, cx, cy, s, th=3):
    k1, k2 = float(rec["k1"]), float(rec["k2"]); H = rec["H"]
    H = np.array(json.loads(H) if isinstance(H, str) else H, float)
    W = float(rec.get("Wp", 18)) or 18.0; R = float(rec.get("R", 2.0)) or 2.0
    def dense(a, b, n=40): return [(a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t) for t in np.linspace(0, 1, n)]
    for seg in [((0, 0), (L, 0)), ((L, 0), (L, W)), ((L, W), (0, W)), ((0, W), (0, 0)), ((L/2, 0), (L/2, W))]:
        pts = dense(*seg); xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        px, py = m2px(np.array(xs), np.array(ys), k1, k2, H, cx, cy, s)
        for i in range(len(px)-1):
            cv2.line(img, (int(px[i]), int(py[i])), (int(px[i+1]), int(py[i+1])), col, th, cv2.LINE_AA)

tiles = []
for v in SEG:
    fp = os.path.join(LF, v)
    if not os.path.exists(fp):
        continue
    img = cv2.imread(fp); h, w = img.shape[:2]; cx, cy, s = w/2, h/2, w/2
    if v in MAN and not MAN[v].get("bad") and "H" in MAN[v]:
        draw(img, MAN[v], (0, 230, 0), cx, cy, s, 3)       # YEŞİL = manuel
    draw(img, SEG[v], (0, 220, 255), cx, cy, s, 2)          # SARI = seg-auto
    tag = " [HELD-OUT]" if v in HELDOUT else ""
    cv2.putText(img, v[:16]+tag, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(img, "SARI=seg-auto(zero-setup)  YESIL=manuel", (16, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    tiles.append(cv2.resize(img, (860, 484)))
grid = np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles)-len(tiles) % 2, 2)])
cv2.imwrite(os.path.join(HERE, "cand", "verify_segauto.jpg"), grid)
print("yazıldı cand/verify_segauto.jpg", len(tiles), "saha")
