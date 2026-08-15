"""joint_calib2: İÇ-ÖZELLİK constraint EKLENMİŞ (Alperen'in ÇEMBER + BOX etiketleri artık KULLANILIYOR).
Eski joint_calib SADECE 4 sınır + orta-çizgi kullanıyordu; çember/box ignore → orta gevşek (touchline-skew).
YENİ: çember-noktaları (L/2,W/2)'den R-eşit-uzak (roundness+center constraint) + W ve R serbest (per-venue).
Test: birkaç sahada eski-vs-yeni warp + çember-oturma. Kullanım: python joint_calib2.py [venue...]"""
import json, os, sys, numpy as np, cv2
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__)); LF = os.path.join(HERE, "label_frames")
J = json.load(open(os.path.join(HERE, "calib_lines.json")))
L = 34.0  # kanonik uzunluk (mutlak-ölçek oyuncu-boyuyla sonra); W,R serbest

def und_norm(pts, k1, k2, cx, cy, s):
    u = (pts[:, 0]-cx)/s; vv = (pts[:, 1]-cy)/s; r2 = u*u+vv*vv; f = 1+k1*r2+k2*r2*r2
    return np.stack([u*f, vv*f], 1)
def applyH(uv, H):
    z = H[2, 0]*uv[:, 0]+H[2, 1]*uv[:, 1]+H[2, 2]
    return np.stack([(H[0, 0]*uv[:, 0]+H[0, 1]*uv[:, 1]+H[0, 2])/z, (H[1, 0]*uv[:, 0]+H[1, 1]*uv[:, 1]+H[1, 2])/z], 1)
def fitL(P): c = P.mean(0); _, _, vt = np.linalg.svd(P-c); nv = np.array([-vt[0, 1], vt[0, 0]]); return np.array([nv[0], nv[1], -nv.dot(c)])
def inter(a, b):
    a1, b1, c1 = a; a2, b2, c2 = b; D = a1*b2-a2*b1
    return None if abs(D) < 1e-9 else np.array([(b1*c2-b2*c1)/D, (a2*c1-a1*c2)/D])

def solve(name, use_circle=True):
    ln = J[name]["lines"]; img = cv2.imread(os.path.join(LF, name)); h, w = img.shape[:2]; cx, cy, s = w/2, h/2, w/2
    has_c = bool(ln.get("center") and len(ln["center"]) >= 4)
    has_circ = bool(ln.get("circle") and len(ln["circle"]) >= 6) and use_circle
    # sınır + orta-çizgi constraint'leri (W paramına bağlı)
    bcons = []
    for lid, ax, tg in [("goalN", 0, 0.0), ("goalF", 0, L), ("touchN", 1, 0.0), ("touchF", 1, None), ("center", 0, L/2)]:
        if lid == "center" and not has_c: continue
        p = ln.get(lid)
        if p and len(p) >= 2: bcons.append((np.asarray(p, float), ax, tg))
    circ = np.asarray(ln["circle"], float) if has_circ else None
    # init: k=0 köşeler -> H (W0=18)
    W0 = 18.0
    fl = {l: fitL(und_norm(np.asarray(ln[l], float), 0, 0, cx, cy, s)) for l in ("goalN", "goalF", "touchN", "touchF")}
    src = np.array([inter(fl["goalN"], fl["touchN"]), inter(fl["goalN"], fl["touchF"]),
                    inter(fl["goalF"], fl["touchN"]), inter(fl["goalF"], fl["touchF"])], float)
    H0, _ = cv2.findHomography(src, np.array([[0, 0], [0, W0], [L, 0], [L, W0]], float))
    p0 = [0.0, 0.0]+list((H0/H0[2, 2]).ravel()[:8])+[W0, 3.0]   # +W +R

    def resid(p, Wp, R):
        k1, k2 = p[0], p[1]; H = np.array([[p[2], p[3], p[4]], [p[5], p[6], p[7]], [p[8], p[9], 1.0]])
        out = []
        for pts, ax, tg in bcons:
            m = applyH(und_norm(pts, k1, k2, cx, cy, s), H)
            t = Wp if tg is None else tg
            out.append((m[:, ax]-t))
        if circ is not None:
            mc = applyH(und_norm(circ, k1, k2, cx, cy, s), H)
            d = np.hypot(mc[:, 0]-L/2, mc[:, 1]-Wp/2)
            out.append((d-R)*2.0)   # çember-roundness+center (ağırlık 2)
        return np.concatenate(out)
    def full(p): return resid(p[:10], p[10], p[11])
    base = np.sqrt((full(p0)**2).mean())
    sol = least_squares(full, p0, method="lm", max_nfev=20000)
    fit = np.sqrt((full(sol.x)**2).mean())
    x = sol.x; k1, k2 = x[0], x[1]; H = np.array([[x[2], x[3], x[4]], [x[5], x[6], x[7]], [x[8], x[9], 1.0]]); Wp, R = x[10], x[11]
    return dict(name=name, k1=k1, k2=k2, H=H.tolist(), Wp=float(Wp), R=float(R), fit=float(fit), base=float(base), has_circ=has_circ)

def warp_panel(rec, tag):
    name = rec["name"]; img = cv2.imread(os.path.join(LF, name)); h, w = img.shape[:2]; cx, cy, s = w/2, h/2, w/2
    k1, k2 = rec["k1"], rec["k2"]; H = np.array(rec["H"]); Wp, R = rec["Wp"], rec["R"]
    S = 22; LS, WS = int((L+6)*S), int((Wp+6)*S); M = 3
    ox, oy = np.meshgrid(np.arange(LS, dtype=np.float32), np.arange(WS, dtype=np.float32))
    Xm = (ox/S-M).ravel(); Ym = ((WS-oy)/S-M).ravel()
    Hi = np.linalg.inv(H); den = Hi[2, 0]*Xm+Hi[2, 1]*Ym+Hi[2, 2]
    un = (Hi[0, 0]*Xm+Hi[0, 1]*Ym+Hi[0, 2])/den; vn = (Hi[1, 0]*Xm+Hi[1, 1]*Ym+Hi[1, 2])/den
    ru = np.hypot(un, vn); rd_g = np.linspace(0, 2.6, 4000); ru_g = rd_g*(1+k1*rd_g*rd_g+k2*rd_g**4)
    rd = np.interp(ru, ru_g, rd_g); sc = np.divide(rd, ru, out=np.ones_like(ru), where=ru > 1e-9)
    mx = (cx+un*sc*s).reshape(WS, LS).astype(np.float32); my = (cy+vn*sc*s).reshape(WS, LS).astype(np.float32)
    top = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderValue=(18, 22, 20))
    def MX(X, Y): return (int((M+X)*S), int(WS-(M+Y)*S))
    cv2.rectangle(top, MX(0, Wp), MX(L, 0), (0, 220, 255), 2); cv2.line(top, MX(L/2, 0), MX(L/2, Wp), (0, 220, 255), 1)
    cv2.circle(top, MX(L/2, Wp/2), int(R*S), (0, 220, 255), 2)
    cv2.putText(top, f"{tag} {name[:12]} Wp={Wp:.1f} R={R:.2f} fit={rec['fit']:.2f}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2)
    return top

if __name__ == "__main__":
    venues = sys.argv[1:] or ["AydinogluHaliSaha.jpg", "GardenParkHaliSaha.jpg", "OzluceTimsahaHaliS.jpg",
                              "MamakBirlikHaliSah.jpg", "NigdeOlimpiyatHali.jpg", "5MevsimHaliSaha.jpg"]
    rows = []
    for v in venues:
        if v not in J: print("yok", v); continue
        r_no = solve(v, use_circle=False); r_yes = solve(v, use_circle=True)
        print(f"{v[:18]:18s} çember={'VAR' if r_yes['has_circ'] else 'yok'} | "
              f"ESKİ(çembersiz) fit={r_no['fit']:.2f} R={r_no['R']:.2f} Wp={r_no['Wp']:.1f} | "
              f"YENİ(çemberli) fit={r_yes['fit']:.2f} R={r_yes['R']:.2f} Wp={r_yes['Wp']:.1f}", flush=True)
        a = warp_panel(r_no, "ESKI"); b = warp_panel(r_yes, "YENI(cember)")
        Hh = 300
        def rz(im): return cv2.resize(im, (int(im.shape[1]*Hh/im.shape[0]), Hh))
        rows.append(np.hstack([rz(a), np.full((Hh, 4, 3), 60, np.uint8), rz(b)]))
    wmax = max(r.shape[1] for r in rows)
    rows = [np.hstack([r, np.full((r.shape[0], wmax-r.shape[1], 3), 18, np.uint8)]) if r.shape[1] < wmax else r for r in rows]
    cv2.imwrite(os.path.join(HERE, "cand_warp", "_CIRCLE_CONSTRAINT.jpg"), np.vstack(rows))
    print("yazıldı cand_warp/_CIRCLE_CONSTRAINT.jpg", flush=True)
