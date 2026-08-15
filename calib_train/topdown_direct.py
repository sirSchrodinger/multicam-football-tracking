"""DOĞRU top-down render: her canvas(metrik) pikseli -> m2px ile KAYNAK(distorted) piksel -> TEK remap.
Eski topdown() iki-aşamalı (tüm-kare undistort + homografi) → periferi patlıyor (streak). Bu, verify_onpaint'in
DOĞRULANMIŞ m2px projeksiyonunu kullanır → warp calib kadar doğru, streak yok.
"""
import numpy as np, cv2
L_DEF, S, M = 34.0, 20, 3.0

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

def _rec(rec):
    import json
    k1, k2 = float(rec["k1"]), float(rec["k2"])
    H = np.array(json.loads(rec["H"]) if isinstance(rec["H"], str) else rec["H"], float)
    Wp = float(rec.get("Wp", 18)) or 18.0
    L = float(rec.get("L", L_DEF)) or L_DEF
    R = float(rec.get("R_use", 2.0)) or 2.0
    return k1, k2, H, Wp, L, R

def topdown_direct(img, rec, draw_template=True):
    h, w = img.shape[:2]; cx, cy, s = w/2, h/2, w/2
    k1, k2, H, Wp, L, R = _rec(rec)
    LS, WS = int((L+2*M)*S), int((Wp+2*M)*S)
    ox, oy = np.meshgrid(np.arange(LS, dtype=np.float32), np.arange(WS, dtype=np.float32))
    Xm = ox/S - M
    Ym = (WS - oy)/S - M
    px, py = m2px(Xm.ravel(), Ym.ravel(), k1, k2, H, cx, cy, s)
    map_x = px.reshape(WS, LS).astype(np.float32)
    map_y = py.reshape(WS, LS).astype(np.float32)
    # kaynak-dışı (pitch dışı) pikselleri maskele
    valid = (map_x >= 0) & (map_x < w) & (map_y >= 0) & (map_y < h)
    top = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=(18, 22, 20))
    top[~valid] = (18, 22, 20)
    if draw_template:
        def MX(X, Y): return (int((M+X)*S), int(WS-(M+Y)*S))
        cv2.rectangle(top, MX(0, Wp), MX(L, 0), (0, 220, 255), 2)
        cv2.line(top, MX(L/2, 0), MX(L/2, Wp), (0, 220, 255), 1)
        cv2.circle(top, MX(L/2, Wp/2), int(R*S), (0, 220, 255), 1)
    return top

def feet_to_metric(feet, rec):
    """ham ayak-piksel -> metrik (X,Y). m2px'in TERSİ: distort->undist-norm->H."""
    k1, k2, H, Wp, L, R = _rec(rec)
    feet = np.asarray(feet, float)
    # cx,cy,s image-bağımlı; çağıran versin — burada sadece H uygula (und_norm dışarıda)
    raise NotImplementedError

if __name__ == "__main__":
    import sys, os, json
    HERE = os.path.dirname(os.path.abspath(__file__))
    which = sys.argv[1] if len(sys.argv) > 1 else "manual"   # manual|auto
    src = "calib_solved.json" if which == "manual" else "seg_auto_solved.json"
    C = json.load(open(os.path.join(HERE, src)))
    from top_down_view import topdown as topdown_old
    venues = sys.argv[2:] or ["ErzurumHaliSahalar.jpg", "KibrisDorukHaliSah.jpg",
                              "AydinogluHaliSaha.jpg", "MamakBirlikHaliSah.jpg",
                              "GardenParkHaliSaha.jpg", "KaynarcaAdaHaliSah.jpg"]
    def find(k):
        for d in ("label_frames", "unseen"):
            p = os.path.join(HERE, d, k)
            if os.path.exists(p):
                return p
        return None
    rows = []
    for v in venues:
        p = find(v)
        if p is None or v not in C:
            print("atla", v); continue
        img = cv2.imread(p)
        old = topdown_old(img, C[v], None)
        new = topdown_direct(img, C[v])
        Hh = 300
        def rz(im, txt, col):
            im = cv2.resize(im, (int(im.shape[1]*Hh/im.shape[0]), Hh))
            cv2.rectangle(im, (0, 0), (im.shape[1], 26), (25, 25, 25), -1)
            cv2.putText(im, txt, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2); return im
        o = rz(cv2.imread(p), v.split('.')[0][:16], (0, 230, 255))
        a = rz(old, "ESKI render (streak)", (0, 140, 255))
        b = rz(new, "YENI render (m2px direct)", (0, 255, 0))
        sep = np.full((Hh, 4, 3), 50, np.uint8)
        rows.append(np.hstack([o, sep, a, sep, b]))
        print("ok", v, flush=True)
    wmax = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], wmax-r.shape[1], 3), 18, np.uint8)]) if r.shape[1] < wmax else r for r in rows]
    grid = np.vstack([np.vstack([r, np.full((5, wmax, 3), 60, np.uint8)]) for r in rows])
    out = os.path.join(HERE, "cand_warp", f"_RENDER_FIX_{which}.jpg")
    cv2.imwrite(out, grid); print("yazıldı", out, grid.shape, flush=True)
