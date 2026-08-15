"""calib_solved kalibrasyonlar GERÇEKTEN doğru mu? Template'i ORİJİNAL kareye reprojekte
et (fisheye dahil). Sarı çizgi gerçek boyanın üstünde mi? Sayı değil GÖZ. cand/verify_onpaint.jpg"""
import sys, os, json, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__))
C = json.load(open(os.path.join(HERE, "calib_solved.json")))
LF = os.path.join(HERE, "label_frames")
L, Wc = 34.0, 18.0

def m2px(Xm, Ym, k1, k2, H, cx, cy, s):
    Hi = np.linalg.inv(H)
    Xm = np.asarray(Xm, float); Ym = np.asarray(Ym, float)
    den = Hi[2, 0]*Xm + Hi[2, 1]*Ym + Hi[2, 2]
    un = (Hi[0, 0]*Xm+Hi[0, 1]*Ym+Hi[0, 2])/den
    vn = (Hi[1, 0]*Xm+Hi[1, 1]*Ym+Hi[1, 2])/den
    ru = np.sqrt(un*un+vn*vn)
    rd_g = np.linspace(0, 2.6, 5000); ru_g = rd_g*(1+k1*rd_g*rd_g+k2*rd_g**4)
    rd = np.interp(ru, ru_g, rd_g); sc = np.divide(rd, ru, out=np.ones_like(ru), where=ru > 1e-9)
    return cx+un*sc*s, cy+vn*sc*s

def polyline_m(pts_m, k1, k2, H, cx, cy, s, img, col, th=3):
    xs = [p[0] for p in pts_m]; ys = [p[1] for p in pts_m]
    px, py = m2px(np.array(xs), np.array(ys), k1, k2, H, cx, cy, s)
    for i in range(len(px)-1):
        cv2.line(img, (int(px[i]), int(py[i])), (int(px[i+1]), int(py[i+1])), col, th, cv2.LINE_AA)

def dense(a, b, n=40): return [(a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t) for t in np.linspace(0, 1, n)]

venues = sys.argv[1:] or ["AydinogluHaliSaha.jpg", "OzluceTimsahaHaliS.jpg", "NigdeOlimpiyatHali.jpg",
                          "KibrisDorukHaliSah.jpg", "GardenParkHaliSaha.jpg", "MamakBirlikHaliSah.jpg"]
tiles = []
for v in venues:
    rec = C.get(v)
    fp = os.path.join(LF, v)
    if rec is None or not os.path.exists(fp) or "H" not in rec:
        continue
    img = cv2.imread(fp); h, w = img.shape[:2]; cx, cy, s = w/2, h/2, w/2
    k1, k2 = float(rec["k1"]), float(rec["k2"]); H = np.array(json.loads(rec["H"]) if isinstance(rec["H"], str) else rec["H"], float)
    R = float(rec.get("R_use", 3.0)) or 3.0
    W = float(rec.get("Wp", Wc)) or Wc
    # boundary (fisheye için yoğun) + halfway + circle
    for seg in [((0, 0), (L, 0)), ((L, 0), (L, W)), ((L, W), (0, W)), ((0, W), (0, 0)), ((L/2, 0), (L/2, W))]:
        polyline_m(dense(*seg), k1, k2, H, cx, cy, s, img, (0, 220, 255), 3)
    circ = [(L/2+R*np.cos(a), W/2+R*np.sin(a)) for a in np.linspace(0, 2*np.pi, 60)]
    polyline_m(circ+[circ[0]], k1, k2, H, cx, cy, s, img, (0, 220, 255), 3)
    cv2.putText(img, f"{v[:18]} R={R:.1f} W={W:.1f} fit={float(rec.get('fit',0)):.2f}m",
                (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(img, "SARI cizgi gercek boyanin ustunde mi?", (16, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (200, 200, 200), 2)
    tiles.append(cv2.resize(img, (960, 540)))
grid = np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, len(tiles)-len(tiles) % 2, 2)])
cv2.imwrite(os.path.join(HERE, "cand", "verify_onpaint.jpg"), grid)
print("yazıldı cand/verify_onpaint.jpg", len(tiles), "saha")
