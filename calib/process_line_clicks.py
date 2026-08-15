#!/usr/bin/env python3
"""line_clicker.html çıktısını (line_clicks.json) işle → PENTAGON kalibrasyon.

Akış (Alperen 3-nokta teşhisini çözer):
 1) (ops) PLUMB-LINE: dünyada düz çizgilere k1,k2 fit → lens OVALLİĞİ düzelir.
 2) line_calib.build_homography_from_lines: occluded YAKIN KÖŞE = çizgi kesişimi
    (tıklanmaz), HALFWAY ile orta-saha kısıtı, dürüstlük-kapılı.
 3) overlay render + QA → calib/cankaya_cam2_v4.json.

Kullanım: venv/bin/python calib/process_line_clicks.py [line_clicks.json]
  (yol verilmezse repo + ~/Downloads aranır)
"""
import cv2, numpy as np, json, sys, os
sys.path.insert(0, ".")
from pitch.template import PitchTemplate
from pitch import line_calib
from scipy.optimize import least_squares

def find_clicks(argv):
    if len(argv) > 1 and os.path.exists(argv[1]): return argv[1]
    for p in ["line_clicks.json", "calib/line_clicks.json",
              os.path.expanduser("~/Downloads/line_clicks.json")]:
        if os.path.exists(p): return p
    raise SystemExit("line_clicks.json bulunamadı (clicker'dan indir, repo'ya koy)")

CL = json.load(open(find_clicks(sys.argv)))
v2 = json.load(open("calib/cankaya_cam2_v2.json"))
K = np.array(v2["K"], float); D = np.array(v2["dist"], float)
L, W = v2["template"]["dims_m"]
lc = {n: np.array(p, float) for n, p in CL["line_clicks"].items()}
posts = CL.get("goal_posts", [])
print("çizgiler:", {n: len(p) for n, p in lc.items()})

# ---- 1) PLUMB-LINE: ovallik (k1,k2) — sadece iyileştirirse uygula ----
def line_curv(dist):
    """her adlı çizgi: undistort(raw, K, dist) sonrası TLS residual toplamı (px)."""
    tot = 0.0
    for n, und in lc.items():
        raw = line_calib._redistort(und, K, D)                  # mevcut D ile raw'a
        u = cv2.undistortPoints(raw.reshape(-1,1,2), K, dist, P=K).reshape(-1,2)
        if len(u) < 3: continue
        m = u.mean(0); c = np.cov((u-m).T); w,_ = np.linalg.eigh(c)
        tot += np.sqrt(max(w[0], 0))                            # küçük eigen = çizgiye dik yayılım
    return tot
base = line_curv(D)
def per_line_curv(dist):
    """her çizginin TLS-dik rezidüel RMS'i (px) — yakın kale çizgisi uzak-ucu izlemek için."""
    r = {}
    for n, und in lc.items():
        raw = line_calib._redistort(und, K, D)
        u = cv2.undistortPoints(raw.reshape(-1,1,2), K, dist, P=K).reshape(-1,2)
        if len(u) < 3: r[n] = float("nan"); continue
        m = u.mean(0); vx,vy = np.linalg.svd((u-m))[2][0]
        d = (u-m) @ np.array([vy,-vx]); r[n] = float(np.sqrt(np.mean(d**2)))
    return r
def resid(k):
    dd = np.array([k[0], k[1], 0, 0, 0], float)
    out = []
    for n, und in lc.items():
        raw = line_calib._redistort(und, K, D)
        u = cv2.undistortPoints(raw.reshape(-1,1,2), K, dd, P=K).reshape(-1,2)
        if len(u) < 3: continue
        m = u.mean(0); vx,vy = np.linalg.svd((u-m))[2][0]       # ana yön
        out += list((u-m) @ np.array([vy,-vx]))                 # dik mesafe
    return out
# NOT (29 Haz): Alperen "yakın çizgi uzak-ucunda lens kırılması" dedi -> K3 (3.derece radyal)
# DENENDİ: çizgi-rezidüeli iyileşti (endline_x0 5.5->0.5px, QA 0.05->0.04m) AMA seyrek çizgi
# kısıtını AŞIRI-FİT etti -> alpha=1 tam-kare görüntü KÜRESEL bozuldu (periferi swirl, saha küçük
# balon). Fiziksel lens değil, optimizer suistimali. REDDEDİLDİ; fiziksel-sağlam k1,k2 korunuyor.
# Yakın-çizgi uzak-uç ~5px rezidüel = tek-kare seyrek-tık kalibrasyonun doğal sınırı (birkaç cm).
try:
    sol = least_squares(resid, [D[0], 0.0], method="lm", max_nfev=4000)
    Dref = np.array([sol.x[0], sol.x[1], 0, 0, 0], float)
    impr = base - line_curv(Dref)
    use_ref = impr > 0.5 and abs(sol.x[0]) < 0.6                # anlamlı + makul
    pl1 = per_line_curv(Dref)
    print(f"plumb-line: k1 {D[0]:.3f}->{sol.x[0]:.3f}, k2 {sol.x[1]:.4f}; "
          f"çizgi-eğrilik {base:.1f}->{line_curv(Dref):.1f}px  uygula={use_ref} "
          f"(endline_x0 dik-rez {pl1.get('endline_x0',float('nan')):.2f}px; k3 overfit-reddedildi)")
except Exception as e:
    use_ref = False; Dref = D; print("plumb-line atlandı:", e)
Duse = Dref if use_ref else D

# çizgi tıklamaları mevcut-D undistorted'tı; D değiştiyse RE-undistort
def reund(und):
    raw = line_calib._redistort(und, K, D)
    return cv2.undistortPoints(raw.reshape(-1,1,2), K, Duse, P=K).reshape(-1,2)
lc2 = {n: (reund(p) if use_ref else p) for n, p in lc.items()}
post_lms = [(reund(np.array([q["img"]]))[0] if use_ref else np.array(q["img"]),
            np.array(q["world"])) for q in posts]

# ---- 2) PENTAGON homografi (occluded köşe = kesişim) ----
tmpl = PitchTemplate.from_dict(v2["template"])
homo, corners, qa = line_calib.build_homography_from_lines(
    "cankaya_cam2", tmpl, lc2, K=K, dist=Duse, point_lms=post_lms, ransac_px=5.0)
print("\nKÖŞELER (occluded dahil, hesaplanan):")
for cid, c in corners.items():
    p = c["img_und"]
    print(f"  {cid}: world={tuple(c['world'])} img={None if p is None else p.round(0)} "
          f"ill={c['ill']} extrap={c['extrap']:.2f} sin={c['sin_angle']:.2f}")
print("line QA (metre):", {k: round(v,2) for k,v in qa.items() if isinstance(v,(int,float))})

# ---- 3) kaydet + overlay ----
homo.save("calib/cankaya_cam2_v4.json")
print("\nyazildi calib/cankaya_cam2_v4.json")

# CEPHE-1 FIX: overlay'i homografi ile AYNI distorsiyon-uzayinda render et.
# Eski hata: undist_clean.png D(v2) ile undistort'tu ama homografi Duse ile
# kuruldu; use_ref=True iken uzaylar uyusmadigi icin cizgiler ~3-7px kayik
# GORUNUYORDU (homografi ~1px DOGRU oldugu halde). Cozum: RAW kareyi K,Duse ile
# alpha=1 (getOptimalNewCameraMatrix) undistort et -> kirpilmaz tam-FOV + dogru-uzay.
# pitch_to_pixel undistorted-K uzayinda doner; M=newK@inv(K) ile display'e tasi.
cap = cv2.VideoCapture("raw/cankaya_cam2.mp4"); cap.set(cv2.CAP_PROP_POS_FRAMES, 67153)
ok, raw = cap.read(); cap.release()
if not ok:                                                  # video yoksa eski statik kareye dus
    raw = cv2.imread("calib/cankaya_cam2_frame_raw.png")
h, w = raw.shape[:2]
newK, _ = cv2.getOptimalNewCameraMatrix(K, Duse, (w, h), 1.0)
m1, m2 = cv2.initUndistortRectifyMap(K, Duse, None, newK, (w, h), cv2.CV_16SC2)
disp = cv2.remap(raw, m1, m2, cv2.INTER_LINEAR); M = newK @ np.linalg.inv(K)
def p2i_arr(world):                                         # pitch(m) -> display px
    und = homo.pitch_to_pixel(np.asarray(world, float))
    q = (M @ np.column_stack([und, np.ones(len(und))]).T).T; return q[:, :2] / q[:, 2:3]
def seg(p0,p1,n=160):
    ts = np.linspace(0,1,n)[:,None]
    return p2i_arr(np.asarray(p0)[None]*(1-ts) + np.asarray(p1)[None]*ts)
for s in tmpl.line_segments_m or [[(0,0),(L,0)],[(L,0),(L,W)],[(L,W),(0,W)],[(0,W),(0,0)],[(L/2,0),(L/2,W)]]:
    pts = seg(s[0], s[1])
    for i in range(1,len(pts)):
        cv2.line(disp, tuple(np.round(pts[i-1]).astype(int)), tuple(np.round(pts[i]).astype(int)), (0,255,255), 2)
th = np.linspace(0, 2*np.pi, 160)
cc = np.column_stack([L/2 + tmpl.center_circle_r_m*np.cos(th), W/2 + tmpl.center_circle_r_m*np.sin(th)])
pts = p2i_arr(cc)
for i in range(1,len(pts)):
    cv2.line(disp, tuple(np.round(pts[i-1]).astype(int)), tuple(np.round(pts[i]).astype(int)), (255,200,0), 2)
for gx in (0.0, L):
    pts = p2i_arr([[gx, W/2-1.5],[gx, W/2+1.5]])
    cv2.line(disp, tuple(np.round(pts[0]).astype(int)), tuple(np.round(pts[1]).astype(int)), (0,0,255), 3)
cv2.imwrite("scratchpad/calib_v4_overlay.png", disp)
cv2.imwrite("scratchpad/calib_v4_overlay_near.png", disp[300:1080, 0:1100])
print("overlay (dogru-uzay, alpha=1 kirpilmaz): scratchpad/calib_v4_overlay.png (+_near)")
