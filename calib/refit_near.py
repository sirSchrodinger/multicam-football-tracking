#!/usr/bin/env python3
"""Yakın-taraf kayıklığı düzeltme: v2 homografisini yakın ceza-sahası kısıtıyla
yeniden çöz. Kök sorun: 8 landmark'ın hepsi iki kale çizgisinde (orta+yakın-iç
ekstrapole), yakın kale-ağzı noktaları (L0,L1) 0.45m residual'la near-bölgeyi
çarpıtıyor. Çözüm: 4 güvenilir köşe + tespit edilen yakın ceza-sahası (4 köşe,
ölçüleri de optimize) ile H'yi yeniden fit et, gürültülü kale-ağzı noktalarını at.
"""
import cv2, numpy as np, json
from scipy.optimize import least_squares

d = json.load(open("calib/cankaya_cam2_v2.json"))
K = np.array(d["K"], float); D = np.array(d["dist"], float)
ip = np.array(d["_img_pts_und"], float); wp = np.array(d["_world_pts"], float)
L, W = d["template"]["dims_m"]                       # 34, 18
Hi2p0 = np.array(d["H_img2pitch"], float)

# --- 1. yakın ceza-sahası kenarlarını beyaz-maskeden çıkar ---
white = cv2.imread("scratchpad/white_mask.png", 0)
def fit_line_from_segs(angle_lo, angle_hi, xr, yr, min_len=80):
    """verilen açı+bölge bandındaki beyaz pikselleri topla, TLS doğru fit -> (a,b,c) ax+by+c=0."""
    m = np.zeros_like(white); m[yr[0]:yr[1], xr[0]:xr[1]] = 255
    sub = cv2.bitwise_and(white, m)
    lines = cv2.HoughLinesP(sub, 1, np.pi/180, 60, minLineLength=min_len, maxLineGap=40)
    pts = []
    for l in (lines if lines is not None else []):
        x1, y1, x2, y2 = l[0]; ang = np.degrees(np.arctan2(y2-y1, x2-x1)) % 180
        if angle_lo <= ang <= angle_hi:
            pts += [(x1, y1), (x2, y2)]
    pts = np.array(pts, float)
    if len(pts) < 2: return None, pts
    vx, vy, x0, y0 = cv2.fitLine(pts.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    # normal (a,b,c): hat yönü (vx,vy) -> normal (vy,-vx)
    a, b = vy, -vx; c = -(a*x0 + b*y0)
    return (a, b, c), pts

bot, bot_pts = fit_line_from_segs(150, 168, (350, 1150), (300, 830))   # box alt-uzun kenar (158°)
top, top_pts = fit_line_from_segs(8, 26, (350, 1150), (300, 830))      # box üst-uzun kenar (17°)
# uzun kenarları ÇİZGİ-kısıtı olarak kullanacağız (far köşeye gerek yok). nokta bulutlarını süz:
def clean_pts(pts, line, max_d=6.0):
    if len(pts) == 0: return pts
    a, b, c = line; dd = np.abs(a*pts[:, 0] + b*pts[:, 1] + c) / np.hypot(a, b)
    return pts[dd < max_d]
bot_pts = clean_pts(bot_pts, bot); top_pts = clean_pts(top_pts, top)
print(f"box alt-kenar nokta={len(bot_pts)}  üst-kenar nokta={len(top_pts)}")

# --- 2. güvenilir köşeler (4) ; gürültülü kale-ağzı (L0..L3, |Y-9|=1.5) AT ---
corner_mask = np.array([min(w[0], L-w[0]) < 1 and (w[1] in (0.0, 18.0)) for w in wp])
img_c = ip[corner_mask]; wld_c = wp[corner_mask]
print(f"kullanılan köşe: {len(img_c)} (kale-ağzı L0-3 atıldı)")

# --- 3. joint optimize: H (img->pitch, 8 dof) + bhw(box yarı-en) ---
#   box uzun kenarları = dünyada Y=9±bhw SABİT çizgileri (X serbest). Bu, yakın
#   perspektif yakınsamasını (kayıklığın kaynağı) doğrudan pinler.
def project(H, pts):
    q = H @ np.c_[pts, np.ones(len(pts))].T
    return (q[:2]/q[2]).T
cy = W/2.0
def resid(params):
    H = np.append(params[:8], 1.0).reshape(3, 3); bhw = params[8]
    r = []
    r += list(((project(H, img_c) - wld_c)).ravel())               # 4 köşe (point)
    # box alt-kenar -> Y = cy-bhw ; üst-kenar -> Y = cy+bhw  (sadece Y kısıtı)
    if len(bot_pts): r += list(project(H, bot_pts)[:, 1] - (cy - bhw))
    if len(top_pts): r += list(project(H, top_pts)[:, 1] - (cy + bhw))
    return r
x0 = np.append(Hi2p0.ravel()[:8], [6.0])                    # init: v2 H + box yarı-en guess 6m
sol = least_squares(resid, x0, method="lm", max_nfev=30000)
H = np.append(sol.x[:8], 1.0).reshape(3, 3); bhw = sol.x[8]; bd = float("nan")
print(f"\noptimize bitti. box yarı-en={bhw:.2f}m (en={2*bhw:.1f}m)  cost={2*sol.cost:.3f}")

# --- 4. residual karşılaştır (yeni H vs v2) ---
def zone_res(H):
    out = {}
    for w, p in zip(wp, ip):
        q = H @ np.array([p[0], p[1], 1.0]); q = q[:2]/q[2]
        e = np.hypot(q[0]-w[0], q[1]-w[1])
        z = "YAKIN" if w[0] < 11 else ("UZAK" if w[0] > 23 else "ORTA")
        out.setdefault(z, []).append(e)
    return {k: (np.mean(v), np.max(v)) for k, v in out.items()}
print("\nresidual (TÜM landmark, bilgi) — v2 vs yeni:")
for z in ("YAKIN", "UZAK"):
    a = zone_res(Hi2p0).get(z); b = zone_res(H).get(z)
    print(f"  {z}: v2 ort={a[0]:.2f}/max={a[1]:.2f}  ->  yeni ort={b[0]:.2f}/max={b[1]:.2f}")
# box köşe residual (yeni H ile)
bw = world_box(bd, bhw); bp = project(H, box_img)
print(f"  BOX köşe residual (yeni): {np.hypot(*(bp-bw).T).round(2)}")

# --- 5. yeni calib kaydet ---
out = dict(d)
out["H_img2pitch"] = H.tolist()
out["H_pitch2img"] = np.linalg.inv(H).tolist()
out["template"]["has_penalty_box"] = True
out["template"]["center_circle_r_m"] = 3.0
out["template"]["penalty_box_m"] = {"depth": float(bd), "half_width": float(bhw)}
out["calib_method"] = "manual_corners+auto_nearbox_refit"
out["calib_note"] = "yakın-taraf kayıklığı düzeltme: 4 köşe + auto yakın-ceza-sahası; gürültülü kale-ağzı atıldı"
json.dump(out, open("calib/cankaya_cam2_v4.json", "w"), indent=1)
print("\nyazildi calib/cankaya_cam2_v4.json")
