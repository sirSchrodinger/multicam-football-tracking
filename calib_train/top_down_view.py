"""GERÇEK yukarıdan-2D: seg-auto calib ile sahayı KUŞ-BAKIŞI warp + oyuncu noktaları.
Dürüst test — saha yukarıdan düzgün dikdörtgen mi, oyuncular doğru yerde mi. Süsleme YOK.
Kullanım: python top_down_view.py <solved.json> <frame_dir> <out.jpg> [venue1 venue2 ...]"""
import sys, os, json, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HERE = os.path.dirname(os.path.abspath(__file__))
L, S, M = 34.0, 20, 3.0   # kanonik uzunluk, px/m, margin

def undimg(img, k1, k2, cx, cy, s, z=1.5):
    h, w = img.shape[:2]; rd = np.linspace(0, 2.4, 2200); rru = rd*(1+k1*rd*rd+k2*rd**4)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32); uo = (xs-cx)/s*z; vo = (ys-cy)/s*z; ro = np.sqrt(uo*uo+vo*vo)
    rdv = np.interp(ro, rru, rd); sc = np.divide(rdv, ro, out=np.ones_like(ro), where=ro > 1e-6)
    return cv2.remap(img, (cx+uo*sc*s).astype(np.float32), (cy+vo*sc*s).astype(np.float32), cv2.INTER_LINEAR)

def und_norm(P, k1, k2, cx, cy, s):
    u = (P[:, 0]-cx)/s; v = (P[:, 1]-cy)/s; r2 = u*u+v*v; f = 1+k1*r2+k2*r2*r2
    return np.stack([u*f, v*f], 1)

def topdown(img, rec, det=None):
    h, w = img.shape[:2]; cx, cy, s = w/2, h/2, w/2
    k1, k2 = float(rec["k1"]), float(rec["k2"]); H = np.array(json.loads(rec["H"]) if isinstance(rec["H"], str) else rec["H"], float)
    Wp = float(rec.get("Wp", 18)) or 18.0
    uimg = undimg(img, k1, k2, cx, cy, s)
    LS, WS = int((L+2*M)*S), int((Wp+2*M)*S)
    # undistorted-pixel -> metrik: H undist-norm->metrik. warp için köşe eşlemesi kur.
    def MX(X, Y): return [(M+X)*S, WS-(M+Y)*S]
    # dört metrik köşeyi undistorted-pixel'e geri getir: H^-1 (metrik->undist-norm) -> undist-pixel
    Hi = np.linalg.inv(H)
    corners_m = np.array([[0, 0], [0, Wp], [L, 0], [L, Wp]], float)
    den = Hi[2, 0]*corners_m[:, 0]+Hi[2, 1]*corners_m[:, 1]+Hi[2, 2]
    un = (Hi[0, 0]*corners_m[:, 0]+Hi[0, 1]*corners_m[:, 1]+Hi[0, 2])/den
    vn = (Hi[1, 0]*corners_m[:, 0]+Hi[1, 1]*corners_m[:, 1]+Hi[1, 2])/den
    src = np.stack([cx+un/1.5*s, cy+vn/1.5*s], 1)   # undistorted-pixel (z=1.5 undimg ile tutarlı)
    dst = np.array([MX(0, 0), MX(0, Wp), MX(L, 0), MX(L, Wp)], float)
    Hm, _ = cv2.findHomography(src, dst)
    top = cv2.warpPerspective(uimg, Hm, (LS, WS), borderValue=(18, 22, 20))
    # saha çizgileri
    cv2.rectangle(top, tuple(map(int, MX(0, Wp))), tuple(map(int, MX(L, 0))), (0, 220, 255), 2)
    cv2.line(top, tuple(map(int, MX(L/2, 0))), tuple(map(int, MX(L/2, Wp))), (0, 220, 255), 1)
    cv2.circle(top, tuple(map(int, MX(L/2, Wp/2))), int(2*S), (0, 220, 255), 1)
    # oyuncu noktaları (verilirse): ham ayak -> undist-norm -> metrik -> top
    if det is not None and len(det["feet"]):
        feet = np.asarray(det["feet"], float)
        m = und_norm(feet, k1, k2, cx, cy, s)
        z = H[2, 0]*m[:, 0]+H[2, 1]*m[:, 1]+H[2, 2]
        mx = (H[0, 0]*m[:, 0]+H[0, 1]*m[:, 1]+H[0, 2])/z; my = (H[1, 0]*m[:, 0]+H[1, 1]*m[:, 1]+H[1, 2])/z
        for X, Y in zip(mx, my):
            if -M < X < L+M and -M < Y < Wp+M:
                px, py = MX(X, Y); cv2.circle(top, (int(px), int(py)), 8, (60, 60, 235), -1)
                cv2.circle(top, (int(px), int(py)), 8, (255, 255, 255), 1)
    return top

if __name__ == "__main__":
    solved = json.load(open(sys.argv[1])); FD = sys.argv[2]; OUT = sys.argv[3]
    venues = sys.argv[4:] or list(solved.keys())
    det_model = None
    if os.environ.get("NO_DET") != "1":
        try:
            from viz.detect_frame import FrameDetector
            det_model = FrameDetector()
            print("detektör hazır", flush=True)
        except Exception as e:
            print("detektör yok:", e, flush=True)
    tiles = []
    for v in venues:
        key = v if v in solved else v+".jpg"
        fp = os.path.join(FD, key)
        if key not in solved or not os.path.exists(fp):
            continue
        img = cv2.imread(fp)
        det = det_model.detect(img) if det_model else None
        top = topdown(img, solved[key], det)
        if det is not None:   # orijinale de ayak-işareti (top-down noktalarıyla eşleşme kontrolü)
            for fx, fy in det["feet"]:
                cv2.circle(img, (int(fx), int(fy)), 6, (60, 60, 235), -1)
                cv2.circle(img, (int(fx), int(fy)), 6, (255, 255, 255), 1)
        # orijinal + top yan yana
        Hh = 460
        o = cv2.resize(img, (int(img.shape[1]*Hh/img.shape[0]), Hh))
        t = cv2.resize(top, (int(top.shape[1]*Hh/top.shape[0]), Hh))
        cv2.putText(o, key.split('.')[0][:16], (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2)
        cv2.putText(t, "YUKARIDAN 2D", (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 230, 255), 2)
        tiles.append(np.hstack([o, np.full((Hh, 5, 3), 40, np.uint8), t]))
        print(f"  {key}: {len(det['feet']) if det else 0} oyuncu", flush=True)
    # tek sütun dikey (her satır bir saha)
    Wm = max(t.shape[1] for t in tiles)
    tiles = [np.hstack([t, np.full((t.shape[0], Wm-t.shape[1], 3), 18, np.uint8)]) if t.shape[1] < Wm else t for t in tiles]
    grid = np.vstack(tiles)
    cv2.imwrite(OUT, grid); print("yazıldı", OUT, grid.shape, flush=True)
