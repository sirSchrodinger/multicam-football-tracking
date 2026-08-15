#!/usr/bin/env python3
"""calib/build_v5.py — CEPHE 1: v5 kalibrasyon (çok-kare zenginleştirme + pentagon).

Alperen teşhisi (v4 "yakın ama oturmuyor"): touchline_y0 SADECE 2 nokta + tek kare
→ near-touchline / bl köşesi yüksek ekstrapolasyon (v4 corner_cond bl extrap=1.97).
Ek olarak v4 overlay BUG: v4-uzay reprojeksiyonu v2-undistort arka plana çiziliyordu.

v5 yaklaşımı (düşük-risk, yüksek-değer):
 1) Alperen'in line_clicks.json çizgilerini KORU (v2-undistort uzayda → RAW'a çevir).
 2) Aynı adlı çizgileri ÇOK-KAREDE statik-beyaz maskeden ZENGİNLEŞTİR: oyuncular
    hareketli, çizgiler sabit → kare-üstü kalıcılık (temporal persistence) ile
    statik beyaz çıkarılır; her çizgi Alperen prior'ı etrafında perpendiküler
    bantta toplanır (yanlış-eşleme yok). touchline_y0 2 → çok nokta, tüm boyunca.
 3) Plumb-line ile distorsiyon (k1,k2[,k3]) zenginleştirilmiş noktalardan yeniden
    çöz; SADECE düz-çizgi residual'ı düşerse uygula (gate).
 4) Pentagon homografi (occluded yakın köşe = çizgi kesişimi), goal_posts ayna-kırıcı.
 5) calib/cankaya_cam2_v5.json + DÜZGÜN overlay (alpha=1 full-frame, getOptimalNewCameraMatrix).

Tüm tıklama/zenginleştirme RAW piksel uzayında saklanır (distorsiyon-bağımsız çapa);
homografi son D ile undistort edilmiş uzayda kurulur (pixel_to_pitch ile aynı, P=K).
"""
import cv2, numpy as np, json, sys, os
sys.path.insert(0, ".")
from pitch.template import PitchTemplate
from pitch import line_calib
from pitch.homography import PitchHomography
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "scratchpad", "wf2_calib")
os.makedirs(OUT, exist_ok=True)

K = np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1]], float)
D2 = np.array([-0.13, 0, 0, 0, 0], float)            # v2 undistort (Alperen clicks bu uzayda)
D4 = np.array([-0.24097812604157173, 0.04860354809260233, 0, 0, 0], float)  # v4 (gather D)
L, W = 34.0, 18.0
VIDEO = os.path.join(ROOT, "raw", "cankaya_cam2.mp4")

# çok-kare örnekleme: çizgiler tüm maç boyunca sabit → geniş pencere, oyuncu çeşitliliği
SAMPLE_FRAMES = [60000, 62000, 64000, 66000, 66905, 67153, 67451, 68000, 70000, 72000]


def redistort(und, K, D):
    return line_calib._redistort(np.asarray(und, float).reshape(-1, 2), K, D)


def undist_pts(raw, K, D):
    return cv2.undistortPoints(np.asarray(raw, float).reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)


def load_clicks_raw():
    """Alperen line_clicks.json (v2-undist) → RAW piksel (distorsiyon-bağımsız çapa)."""
    CL = json.load(open(os.path.join(ROOT, "calib", "line_clicks.json")))
    lines_raw = {n: redistort(np.array(p, float), K, D2)
                 for n, p in CL["line_clicks"].items()}
    posts_raw = [(redistort(np.array([q["img"]], float), K, D2)[0], np.array(q["world"], float))
                 for q in CL["goal_posts"]]
    return lines_raw, posts_raw


def grab_frames(frames):
    cap = cv2.VideoCapture(VIDEO)
    out = {}
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, im = cap.read()
        if ok:
            out[f] = im
    cap.release()
    return out


def _field_mask(undist_frames):
    """Çim-tabanlı saha maskesi (dilate): beyaz çizgiler çimle çevrili → dilate
    çizgiyi kapsar; overexpose duvar/ışık + tribün YEŞİL DEĞİL → elenir.
    Kare-üstü median ile gürültü azalt."""
    gsum = None
    for u in undist_frames:
        hsv = cv2.cvtColor(u, cv2.COLOR_BGR2HSV)
        Hh, Ss, Vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        green = ((Hh > 30) & (Hh < 95) & (Ss > 45) & (Vv > 35)).astype(np.float32)
        gsum = green if gsum is None else gsum + green
    gfrac = gsum / len(undist_frames)
    fieldm = (gfrac > 0.4).astype(np.uint8) * 255
    # en büyük bağlı bileşen (saha) + büyük dilate (çizgi + kenar payı)
    num, lab, stats, _ = cv2.connectedComponentsWithStats(fieldm, 8)
    if num > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        fieldm = (lab == big).astype(np.uint8) * 255
    fieldm = cv2.dilate(fieldm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    fieldm = cv2.morphologyEx(fieldm, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61)))
    return fieldm > 0


def static_white_map(undist_frames):
    """Statik beyaz-çizgi haritası: yüksek parlaklık + düşük doygunluk, kare-üstü
    kalıcılık (oyuncu/şort beyazı hareketli → elenir), saha-maskesi (duvar/ışık
    elenir). Döner: float [0,1] (HxW)."""
    votes = None
    n = 0
    for u in undist_frames:
        hsv = cv2.cvtColor(u, cv2.COLOR_BGR2HSV)
        Hh, Ss, Vv = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        # beyaz çizgi: parlak + az doygun. Çim parlak-yeşil (yüksek S) elenir.
        wm = ((Vv > 150) & (Ss < 70)).astype(np.float32)
        votes = wm if votes is None else votes + wm
        n += 1
    frac = votes / max(n, 1)
    fieldm = _field_mask(undist_frames)
    frac = frac * fieldm.astype(np.float32)
    return frac  # 1.0 = her karede beyaz (statik çizgi), saha içi


def enrich_line(prior_und, white_static, band_px=18.0, persist=0.5,
                extend_frac=0.06, step=4.0):
    """Alperen prior çizgisi (undist) etrafında statik-beyaz noktaları topla.

    prior_und: (N>=2,2) undist nokta. Çizgiyi TLS fit et, prior aralığı boyunca
    (+%extend) örnekle; her örnekte perpendiküler ±band içinde statik-beyaz
    (>persist) piksel ağırlık-merkezini ekle. Döner: (M,2) undist zenginleştirme.
    """
    p = np.asarray(prior_und, float)
    fit = line_calib.fit_line_tls(p)
    a, b, c = fit["l"]; t = fit["t"]; c0 = fit["c0"]
    smin, smax = fit["smin"], fit["smax"]
    span = smax - smin
    smin -= extend_frac * span; smax += extend_frac * span
    nrm = np.array([a, b])
    Hh, Ww = white_static.shape[:2]
    pts = []
    s = smin
    while s <= smax:
        base = c0 + s * t                         # prior üzerindeki nokta
        # perpendiküler tarama: en yakın statik-beyaz kümesi ağırlık-merkezi
        offs = np.arange(-band_px, band_px + 1, 1.0)
        cand = base[None, :] + offs[:, None] * nrm[None, :]
        xi = np.round(cand[:, 0]).astype(int); yi = np.round(cand[:, 1]).astype(int)
        ok = (xi >= 0) & (xi < Ww) & (yi >= 0) & (yi < Hh)
        if ok.any():
            wv = np.zeros(len(offs))
            wv[ok] = white_static[yi[ok], xi[ok]]
            wv[wv < persist] = 0.0
            if wv.sum() > 0:
                ctr = (offs * wv).sum() / wv.sum()
                pts.append(base + ctr * nrm)
        s += step
    return np.array(pts, float) if pts else np.empty((0, 2))


def robust_prune(prior_und, enriched, thr=8.0):
    """Prior + enrich → RANSAC doğru (prior ağırlıklı), thr-px dışı enrich at.
    Döner: (inlier_enriched, fit). Parlak-leke/çember kirliliğini eler."""
    if len(enriched) == 0:
        return enriched, line_calib.fit_line_tls(prior_und)
    allp = np.vstack([np.repeat(prior_und, 3, axis=0), enriched])  # prior 3x ağırlık
    best_in, best_fit = None, None
    rng = np.random.default_rng(0)
    n = len(allp)
    for _ in range(200):
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue
        p0, p1 = allp[i], allp[j]
        d = p1 - p0
        nn = np.hypot(*d)
        if nn < 1e-6:
            continue
        nrm = np.array([-d[1], d[0]]) / nn
        res = np.abs((allp - p0) @ nrm)
        inl = res < thr
        if best_in is None or inl.sum() > best_in.sum():
            best_in, = (inl,)
    fit = line_calib.fit_line_tls(allp[best_in])
    a, b, c = fit["l"]
    rese = np.abs(enriched @ np.array([a, b]) + c)
    return enriched[rese < thr], fit


COLORS = {"endline_xL": (255, 0, 0), "touchline_y0": (0, 0, 255),
          "touchline_yW": (0, 255, 0), "endline_x0": (255, 0, 255),
          "halfway": (0, 165, 255)}
# adlı çizgi → dünya örnekleme (reproject prior için, full uzunluk)
_LINE_WORLD = {
    "endline_x0":  lambda: np.column_stack([np.zeros(60), np.linspace(0, W, 60)]),
    "endline_xL":  lambda: np.column_stack([np.full(60, L), np.linspace(0, W, 60)]),
    "touchline_y0": lambda: np.column_stack([np.linspace(0, L, 120), np.zeros(120)]),
    "touchline_yW": lambda: np.column_stack([np.linspace(0, L, 120), np.full(120, W)]),
    "halfway":     lambda: np.column_stack([np.full(60, L / 2), np.linspace(0, W, 60)]),
}


def build_homography(line_und, posts_raw, Duse, tmpl):
    """Adlı-çizgi undist nokta-setlerinden pentagon homografi.

    line_und: {name: (N,2) undist}. point_lms = goal_posts(4) + halfway∩touchline(2).
    halfway kesişimleri orta-saha kısıtı ekler (pentagon yalnız köşeyi kısıtlar).
    """
    posts_und = [(undist_pts(p[0], K, Duse)[0], p[1]) for p in posts_raw]
    homo, corners, qa = line_calib.build_homography_from_lines(
        "cankaya_cam2", tmpl, line_und, K=K, dist=Duse,
        point_lms=posts_und, ransac_px=5.0)
    return homo, corners, qa


def enrich_round(priors_und, wstat):
    out = {}
    for n, pri in priors_und.items():
        e = enrich_line(pri, wstat)
        e2, _ = robust_prune(pri, e)
        # prior'ı her zaman dahil et (güvenilir çapa, 3x ağırlık)
        out[n] = np.vstack([np.repeat(pri, 3, axis=0), e2]) if len(e2) else np.repeat(pri, 3, axis=0)
    return out


_CONST = {"endline_x0": ("X", 0.0), "endline_xL": ("X", L),
          "touchline_y0": ("Y", 0.0), "touchline_yW": ("Y", W),
          "halfway": ("X", L / 2)}


def eval_on_alperen(homo, Duse, lines_raw, posts_raw):
    """ADİL GT: homografiyi Alperen'in EL-TIKLADIĞI çizgilere (RAW çapa) karşı ölç.
    Her çizgi için sabit-eksen metre residual'ı (endline→X, touchline→Y, halfway→X).
    Döner: (median, p95, per_line{med,max,n}, post_err_list[m])."""
    per = {}
    allr = []
    for n, raw in lines_raw.items():
        und = undist_pts(raw, K, Duse)
        XY = homo.pixel_to_pitch(und, already_undistorted=True)
        axis, val = _CONST[n]
        col = 0 if axis == "X" else 1
        res = np.abs(XY[:, col] - val)
        per[n] = (float(np.median(res)), float(np.max(res)), int(len(res)))
        allr += list(res)
    pres = []
    for raw, world in posts_raw:
        und = undist_pts(raw, K, Duse)
        XY = homo.pixel_to_pitch(und, already_undistorted=True)[0]
        pres.append(float(np.hypot(XY[0] - world[0], XY[1] - world[1])))
    allr = np.array(allr)
    return float(np.median(allr)), float(np.percentile(allr, 95)), per, pres


def line_straightness(line_pts):
    """Her çizgi için TLS perpendiküler RMS (px) — düz-çizgi (plumb) ölçütü."""
    res = {}
    for n, p in line_pts.items():
        if len(p) >= 2:
            f = line_calib.fit_line_tls(p)
            a, b, c = f["l"]
            res[n] = float(np.sqrt(np.mean((p @ np.array([a, b]) + c) ** 2)))
    return res


def plumb_refine(enriched_raw, D_init):
    """Düz-çizgi (plumb) ile distorsiyon refine: raw noktaları undistort edip her
    çizginin perpendiküler residual'ını minimize et. Döner: (D_ref, before, after)."""
    names = [n for n in enriched_raw if len(enriched_raw[n]) >= 3]

    def resid(k):
        Dk = np.array([k[0], k[1], 0, 0, k[2]], float)
        out = []
        for n in names:
            u = undist_pts(enriched_raw[n], K, Dk)
            f = line_calib.fit_line_tls(u)
            a, b, c = f["l"]
            out += list(u @ np.array([a, b]) + c)
        return out

    def total(k):
        r = np.array(resid(k))
        return float(np.sqrt(np.mean(r ** 2)))

    k0 = [D_init[0], D_init[1], D_init[4] if len(D_init) > 4 else 0.0]
    before = total(k0)
    sol = least_squares(resid, k0, method="lm", max_nfev=6000)
    after = total(sol.x)
    D_ref = np.array([sol.x[0], sol.x[1], 0, 0, sol.x[2]], float)
    return D_ref, before, after


def make_overlay(raw_frame, homo, Duse, tmpl, path, near_path=None):
    """alpha=1 full-frame overlay (getOptimalNewCameraMatrix, görüntü KIRPILMAZ).

    Arka plan: remap(raw, alpha=1). Çizgiler: pitch_to_pixel (same-K undist) →
    newK@K^-1 ile alpha=1 uzaya taşı. Yeşil=template, kırmızı=kale ağzı."""
    h, w = raw_frame.shape[:2]
    newK, _ = cv2.getOptimalNewCameraMatrix(K, Duse, (w, h), 1.0)
    map1, map2 = cv2.initUndistortRectifyMap(K, Duse, None, newK, (w, h), cv2.CV_16SC2)
    disp = cv2.remap(raw_frame, map1, map2, cv2.INTER_LINEAR)
    M = newK @ np.linalg.inv(K)  # same-K-undist px → alpha=1 newK px

    def to_disp(und):
        q = (M @ np.column_stack([und, np.ones(len(und))]).T).T
        return q[:, :2] / q[:, 2:3]

    def p2d(world_pts):
        und = homo.pitch_to_pixel(np.asarray(world_pts, float))
        return to_disp(und)

    def seg(p0, p1, n=120):
        ts = np.linspace(0, 1, n)[:, None]
        ws = np.asarray(p0)[None] * (1 - ts) + np.asarray(p1)[None] * ts
        return p2d(ws)

    for s in tmpl.line_segments_m:
        pts = seg(s[0], s[1])
        for i in range(1, len(pts)):
            cv2.line(disp, tuple(np.round(pts[i - 1]).astype(int)),
                     tuple(np.round(pts[i]).astype(int)), (0, 255, 255), 2)
    # orta yuvarlak
    if tmpl.center_circle_r_m:
        th = np.linspace(0, 2 * np.pi, 90)
        cc = np.column_stack([L / 2 + tmpl.center_circle_r_m * np.cos(th),
                              W / 2 + tmpl.center_circle_r_m * np.sin(th)])
        pts = p2d(cc)
        for i in range(1, len(pts)):
            cv2.line(disp, tuple(np.round(pts[i - 1]).astype(int)),
                     tuple(np.round(pts[i]).astype(int)), (255, 200, 0), 2)
    # kale ağzı (3m): X=0 ve X=L, Y∈[7.5,10.5]
    for gx in (0.0, L):
        pts = p2d([[gx, 7.5], [gx, 10.5]])
        cv2.line(disp, tuple(np.round(pts[0]).astype(int)),
                 tuple(np.round(pts[1]).astype(int)), (0, 0, 255), 3)
    cv2.imwrite(path, disp)
    if near_path is not None:
        cv2.imwrite(near_path, disp[max(0, h // 3):, :w // 2])
    return disp


def main():
    lines_raw, posts_raw = load_clicks_raw()
    print("Alperen çizgileri (RAW):", {n: len(p) for n, p in lines_raw.items()})
    frames = grab_frames(SAMPLE_FRAMES)
    print("okunan kare:", sorted(frames))
    raw_list = list(frames.values())
    bg_raw = frames.get(67153, raw_list[0])

    tmpl = PitchTemplate.from_dict(json.load(open(os.path.join(ROOT, "calib", "cankaya_cam2_v2.json")))["template"])
    Duse = D4.copy()

    # statik beyaz (gather D=v4)
    und_frames = [cv2.undistort(im, K, Duse) for im in raw_list]
    wstat = static_white_map(und_frames)
    cv2.imwrite(os.path.join(OUT, "static_white.png"), (wstat * 255).astype(np.uint8))

    # ITER0 ONLY: Alperen prior'larının DAR bandında enrich → homografi.
    # (Çok-tur reproject-reenrich DRIFT yapıyordu: halfway→orta-yuvarlak,
    #  touchline→ceza-sahası; homografi kendi kaymış kanıtına overfit oluyordu.
    #  ÖLÇÜLDÜ: çok-tur v5 Alperen-tıklamada 0.50m, ITER0 ~0.05m. Bu yüzden tek-tur.)
    priors_und = {n: undist_pts(raw, K, Duse) for n, raw in lines_raw.items()}
    line_und = enrich_round(priors_und, wstat)
    homo, corners, qa = build_homography(line_und, posts_raw, Duse, tmpl)
    print(f"\nITER0 line-QA (self-evidence) median={qa['median_m']:.4f} p95={qa['p95_m']:.4f} "
          + " ".join(f"{n}={v['median_m']:.3f}" for n, v in qa['per_line'].items()))

    # enriched (ITER0) RAW olarak sakla
    enriched_raw = {n: redistort(line_und[n], K, Duse) for n in line_und}
    np.savez(os.path.join(OUT, "enriched_raw.npz"),
             **{n: enriched_raw[n] for n in enriched_raw})

    # ---- ADİL KARŞILAŞTIRMA: v4 vs v5, Alperen EL-TIKLAMA çizgilerine karşı ----
    v4 = PitchHomography.load(os.path.join(ROOT, "calib", "cankaya_cam2_v4.json"))
    m4, p4, per4, pe4 = eval_on_alperen(v4, np.array(v4.dist, float), lines_raw, posts_raw)
    m5, p5, per5, pe5 = eval_on_alperen(homo, Duse, lines_raw, posts_raw)
    print("\n=== ADİL: Alperen el-tıklama çizgilerine karşı (metre) ===")
    print(f"  v4  median={m4:.4f} p95={p4:.4f}  post_med={np.median(pe4):.3f}")
    print(f"  v5  median={m5:.4f} p95={p5:.4f}  post_med={np.median(pe5):.3f}")
    print("  per-line (med / max / n):")
    for n in _CONST:
        a = per4[n]; b = per5[n]
        print(f"    {n:14s} v4 {a[0]:.4f}/{a[1]:.4f}  v5 {b[0]:.4f}/{b[1]:.4f}  (n={b[2]})")
    alperen_eval = dict(v4=dict(median=m4, p95=p4, per_line=per4, post_err=pe4),
                        v5=dict(median=m5, p95=p5, per_line=per5, post_err=pe5))

    # PLUMB distorsiyon refine (gate'li)
    straight0 = line_straightness(line_und)
    s0 = float(np.mean(list(straight0.values())))
    D_ref, b, a = plumb_refine(enriched_raw, Duse)
    print(f"\nplumb: çizgi-eğrilik RMS {b:.3f}→{a:.3f}px  k1 {Duse[0]:.3f}→{D_ref[0]:.3f} "
          f"k2 {Duse[1]:.4f}→{D_ref[1]:.4f} k3 {D_ref[4]:.4f}")
    used_refine = False
    if a < b - 0.05 and abs(D_ref[0]) < 0.6 and abs(D_ref[4]) < 0.5:
        # refined D ile homografiyi yeniden kur, QA karşılaştır
        line_und_r = {n: undist_pts(enriched_raw[n], K, D_ref) for n in enriched_raw}
        homo_r, corners_r, qa_r = build_homography(line_und_r, posts_raw, D_ref, tmpl)
        print(f"  refined homografi line-QA median={qa_r['median_m']:.4f} p95={qa_r['p95_m']:.4f}")
        if qa_r['median_m'] <= qa['median_m'] * 1.05 and qa_r['p95_m'] <= qa['p95_m'] * 1.1:
            homo, corners, qa, Duse, line_und = homo_r, corners_r, qa_r, D_ref, line_und_r
            used_refine = True
            print("  → refined D UYGULANDI")
        else:
            print("  → refined D reddedildi (homografi QA iyileşmedi)")
    else:
        print("  → plumb refine gate FAIL (kazanım yetersiz)")

    # ---- kaydet ----
    homo.scale_anchor = ["goal_width", 3.0]
    homo.status = "manual"
    homo.calib_method = "manual_line_multiframe"
    # DİKKAT: bu script çok-kare ENRICHMENT DENEYİ. ÖLÇÜLDÜ: enrichment v4'ten
    # KÖTÜ (Alperen el-tıklamada 0.05→0.37m). Bu yüzden deney çıktısı scratch'e
    # yazılır; GERÇEK v5 = finalize_v5.py (v4 geometri + r=3 + overlay-fix).
    out_json = os.path.join(OUT, "cankaya_cam2_v5_ENRICH_EXPERIMENT.json")
    homo.save(out_json)
    print(f"\nyazıldı {out_json}  (D refine kullanıldı={used_refine}) [DENEY — calib/'e KOYULMAZ]")
    print("KÖŞELER:")
    for cid, c in corners.items():
        p = c["img_und"]
        print(f"  {cid}: world={tuple(c['world'])} ill={c['ill']} extrap={c['extrap']:.2f} sin={c['sin_angle']:.2f}")

    # ---- overlay (alpha=1 full-frame), çok-kare ----
    make_overlay(bg_raw, homo, Duse, tmpl,
                 os.path.join(OUT, "v5_overlay.png"),
                 os.path.join(OUT, "v5_overlay_near.png"))
    for fno in (66905, 67451, 70000):
        if fno in frames:
            make_overlay(frames[fno], homo, Duse, tmpl,
                         os.path.join(OUT, f"v5_overlay_f{fno}.png"))
    print("overlay yazıldı: scratchpad/wf2_calib/v5_overlay*.png")

    # enrich viz (final)
    bg_und = cv2.undistort(bg_raw, K, Duse)
    viz = bg_und.copy()
    for n in line_und:
        col = COLORS.get(n, (0, 0, 255))
        for x, y in line_und[n]:
            cv2.circle(viz, (int(round(x)), int(round(y))), 2, col, -1)
    cv2.imwrite(os.path.join(OUT, "enrich_viz.png"), viz)

    # QA özeti json
    summ = dict(version="v5", D=Duse.tolist(), used_distortion_refine=used_refine,
                n_points_per_line={n: int(len(v)) for n, v in line_und.items()},
                line_qa_self_m=qa, plumb_before_px=b, plumb_after_px=a,
                fair_eval_on_alperen_clicks=alperen_eval)
    json.dump(summ, open(os.path.join(OUT, "v5_qa.json"), "w"), indent=2)
    print("QA özeti: scratchpad/wf2_calib/v5_qa.json")


if __name__ == "__main__":
    main()
