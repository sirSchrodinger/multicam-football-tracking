#!/usr/bin/env python3
"""dimension_sim.py — insan-boyu -> saha-olcusu yontemini SIMULASYONLA dogrula.

Soru: "sahanin gercek olcusunu, sahadaki ~1.75m insanlari olcu-cubugu kullanarak
ne kadar guvenilir kestirebiliyoruz?" Bu script GERCEK pitch.height_scale kodunu
(toy kopya degil) bilinen-dogru sentetik sahnelere uygular:

  1) IDEAL  : kusursuz kalibrasyon + sifir gurultu -> yontem sahayi ~%0 hatayla
              geri buluyor mu? (matematiksel dogruluk kaniti, sistematik bias yok)
  2) MONTE CARLO: gercekci boy-degisimi (sigma~7cm) + tespit-kutusu piksel gurultusu
              -> kestirim DAGILIMI; ortalama=dogru mu (unbiased), yayilim ne?
  3) DUYARLILIK: bandin NEREDEN geldigi —
        (a) ODAK belirsizligi (tek-goruntu f pinlenemez): f' = alpha*f, alpha in [0.8,1.3]
        (b) varsayilan ORTALAMA-boy priori [1.70,1.80] m
        ikisinin zarfi = Cankaya'da raporlanan +-~13% band.

Cikti: docs/report/figures/{sim_scene,sim_montecarlo,sim_sensitivity}.{pdf,png}
       + docs/report/sim_results.json  (rapora gomulen GERCEK sayilar)

Lisans: numpy + cv2 + matplotlib. GPU yok.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import cv2
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pitch.height_scale import (  # noqa: E402
    player_height_scale, estimate_heights_relm, _pose_at_f, estimate_focal,
)

FIGDIR = ROOT / "docs/report/figures"
OUTJSON = ROOT / "docs/report/sim_results.json"

# paper_style (havalı figür toolkit)
sys.path.insert(0, str(ROOT.parent / "Templates/article-twocolumn/figures"))
import paper_style as ps  # noqa: E402


# --------------------------------------------------------------------------- #
# Sentetik sahne (test_height_scale._synthetic_scene'in genisletilmis hali)
# --------------------------------------------------------------------------- #
def synthetic_scene(L=40.0, W=20.0, f=1100.0, cam_h=4.0, n=2500, seed=0,
                    height_mu=1.75, height_sigma=0.0, pix_noise=0.0):
    """Bilinen kamera+saha+insanlar -> (calib_dict, df, gt, proj_fn, corners_img).

    height_sigma : gercek boy populasyon std'si (m). pix_noise: foot/head piksel std.
    """
    cx, cy = 960.0, 540.0
    C = np.array([-3.0, W / 2, cam_h]); target = np.array([L / 2, W / 2, 0.0]); up = np.array([0, 0, 1.0])
    fwd = (target - C) / np.linalg.norm(target - C)
    right = np.cross(fwd, up); right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.vstack([right, down, fwd]); t = -R @ C
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    P = K @ np.column_stack([R, t])

    def proj(Xw):
        h = (P @ np.column_stack([Xw, np.ones(len(Xw))]).T).T
        return h[:, :2] / h[:, 2:3]

    corners = np.array([[0, 0, 0], [L, 0, 0], [L, W, 0], [0, W, 0.0]])
    corners_img = proj(corners)
    H, _ = cv2.findHomography(corners_img.astype(np.float64),
                              corners[:, :2].astype(np.float64), 0)
    rng = np.random.default_rng(seed)
    X = rng.uniform(2, L - 2, n); Y = rng.uniform(2, W - 2, n)
    Z = height_mu + rng.normal(0, height_sigma, n)
    foot = proj(np.column_stack([X, Y, np.zeros(n)]))
    head = proj(np.column_stack([X, Y, Z]))
    if pix_noise > 0:
        foot = foot + rng.normal(0, pix_noise, foot.shape)
        head = head + rng.normal(0, pix_noise, head.shape)
    bh = foot[:, 1] - head[:, 1]
    df = pd.DataFrame(dict(tid=np.arange(n), frame=np.arange(n), t_sec=np.arange(n) / 25.0,
                           foot_x=foot[:, 0], foot_y=foot[:, 1], box_h=bh, box_w=bh / 2.5,
                           conf=np.full(n, 0.9), bottom_cropped=np.zeros(n, bool),
                           pitch_x=np.nan, pitch_y=np.nan, in_pitch=np.ones(n, bool)))
    calib = dict(H_img2pitch=H.tolist(), K=K.tolist(), dist=[0, 0, 0, 0, 0.0],
                 pitch_dims_m=dict(L=L, W=W))
    gt = dict(L=L, W=W, cam_h=cam_h, f=f, C=C.tolist())
    return calib, df, gt, proj, corners_img


def recover(calib, df, assumed_height_m=1.75):
    """GERCEK player_height_scale'i temp dosyalarla calistir -> sonuc dict."""
    with tempfile.TemporaryDirectory() as td:
        cp = Path(td) / "c.json"; tp = Path(td) / "t.parquet"
        cp.write_text(json.dumps(calib)); df.to_parquet(tp)
        return player_height_scale(str(cp), str(tp), assumed_height_m=assumed_height_m)


def recover_with_focal(calib, df, f_assumed, assumed_height_m=1.75):
    """Bilinen H'yi YANLIS bir f varsayimiyla coz (tek-goruntu odak belirsizligi).

    player_height_scale icindeki _med_c_at mantigini f_assumed ile tekrarlar:
    poz(R,t) ve projeksiyon hepsi f_assumed ile -> implied L. Band'in odak bileseni.
    """
    H = np.array(calib["H_img2pitch"]); dist = np.array(calib["dist"])
    cx, cy = 960.0, 540.0
    Hwi = np.linalg.inv(H)
    Kb = np.array([[f_assumed, 0, cx], [0, f_assumed, cy], [0, 0, 1.0]])
    Rb, tb, Cb, _ = _pose_at_f(Hwi, f_assumed, cx, cy)
    Z = estimate_heights_relm(df, H, Kb, dist, Rb, tb)
    if not Z.size:
        return None
    med = float(np.median(Z)); c = assumed_height_m / med
    L_rel = float(calib["pitch_dims_m"]["L"]); W_rel = float(calib["pitch_dims_m"]["W"])
    return dict(scale=c, L=L_rel * c, W=W_rel * c, cam_h=float(Cb[2] * c), median_relm=med)


# --------------------------------------------------------------------------- #
# 1) IDEAL exact recovery
# --------------------------------------------------------------------------- #
def run_ideal():
    calib, df, gt, _, _ = synthetic_scene(L=40.0, W=20.0, f=1100.0, cam_h=4.0,
                                          n=2500, height_sigma=0.0, pix_noise=0.0)
    r = recover(calib, df)
    out = dict(truth=gt, recovered=dict(
        L=r["implied_field_m"]["L"], W=r["implied_field_m"]["W"],
        cam_h=r["camera_height_m"], f=r["focal_px"], cv=r["height_cv"]))
    out["err_pct"] = dict(
        L=100 * abs(r["implied_field_m"]["L"] - gt["L"]) / gt["L"],
        W=100 * abs(r["implied_field_m"]["W"] - gt["W"]) / gt["W"],
        cam_h=100 * abs(r["camera_height_m"] - gt["cam_h"]) / gt["cam_h"],
        f=100 * abs(r["focal_px"] - gt["f"]) / gt["f"])
    return out


# --------------------------------------------------------------------------- #
# 2) MONTE CARLO (gercekci gurultu, dogru f)
# --------------------------------------------------------------------------- #
def run_montecarlo(K=300, L=40.0, W=20.0, height_sigma=0.07, pix_noise=1.5):
    Ls, Ws, chs, cvs = [], [], [], []
    for s in range(K):
        calib, df, gt, _, _ = synthetic_scene(L=L, W=W, f=1100.0, cam_h=4.0, n=1200,
                                              seed=s + 1, height_sigma=height_sigma,
                                              pix_noise=pix_noise)
        r = recover(calib, df)
        if not r["implied_field_m"]:
            continue
        Ls.append(r["implied_field_m"]["L"]); Ws.append(r["implied_field_m"]["W"])
        chs.append(r["camera_height_m"]); cvs.append(r["height_cv"])
    Ls = np.array(Ls); Ws = np.array(Ws); chs = np.array(chs); cvs = np.array(cvs)
    return dict(L=Ls.tolist(), W=Ws.tolist(), cam_h=chs.tolist(), cv=cvs.tolist(),
                truth=dict(L=L, W=W),
                stats=dict(L_mean=float(Ls.mean()), L_std=float(Ls.std()),
                           L_bias_pct=float(100 * (Ls.mean() - L) / L),
                           L_spread_pct=float(100 * Ls.std() / L),
                           cv_mean=float(cvs.mean()),
                           W_mean=float(Ws.mean()), W_std=float(Ws.std())))


# --------------------------------------------------------------------------- #
# 3) DUYARLILIK: odak + boy priori -> band
# --------------------------------------------------------------------------- #
def _focal_sweep(calib, df, alphas, f_true):
    out = []
    for a in alphas:
        rr = recover_with_focal(calib, df, f_true * a)
        out.append(rr["L"] if rr else np.nan)
    return np.array(out)


def run_sensitivity(L=40.0, W=20.0):
    """Band'in NEREDEN geldigi. Iki bagimsiz kaynak:
      - SENTETIK (kusursuz homografi): odak belirsizligi neredeyse ETKISIZ ->
        band bir YONTEM kusuru DEGIL.
      - GERCEK Cankaya homografisi (fisheye-undistort artigi): tek-goruntu odak
        PINLENEMEZ (f_calib=1000 vs f_ortho~1187) -> L 28.5-37.1m, +-13% (asil band).
      - Boy-priori (1.70-1.80m): +-2.9% (her iki durumda).
    """
    alphas = np.linspace(0.8, 1.3, 26)
    # sentetik kusursuz
    calib_s, df_s, _, _, _ = synthetic_scene(L=L, W=W, f=1100.0, cam_h=4.0, n=2500,
                                             height_sigma=0.07, pix_noise=1.5, seed=7)
    Lf_syn = _focal_sweep(calib_s, df_s, alphas, 1100.0)
    L0_syn = float(recover(calib_s, df_s)["implied_field_m"]["L"])

    # GERCEK Cankaya
    real_calib = json.loads((ROOT / "calib/cankaya_cam2_v2.json").read_text())
    real_df = pd.read_parquet(ROOT / "raw/tracks_cankaya_cam2_clip2400.parquet")
    f_real = float(np.array(real_calib["K"])[0, 0])
    f_ortho = float(estimate_focal(np.array(real_calib["H_img2pitch"])))
    Lf_real = _focal_sweep(real_calib, real_df, alphas, f_real)
    base_real = recover_with_focal(real_calib, real_df, f_real)
    L0_real = base_real["L"]
    # boy-priori (gercek L0 uzerinden)
    heights = np.linspace(1.70, 1.80, 11)
    Lh = L0_real * heights / 1.75

    def band(arr, L0):
        a = arr[np.isfinite(arr)]
        return float(100 * 0.5 * (a.max() - a.min()) / L0)
    real_focal_band = band(Lf_real, L0_real)
    height_band = band(Lh, L0_real)
    return dict(
        alphas=alphas.tolist(),
        L_focal_syn=Lf_syn.tolist(), L0_syn=L0_syn,
        L_focal_real=Lf_real.tolist(), L0_real=float(L0_real),
        f_real=f_real, f_ortho=f_ortho,
        heights=heights.tolist(), L_height=Lh.tolist(),
        truth_L=L,
        syn_focal_band_pct=band(Lf_syn, L0_syn),
        real_focal_band_pct=real_focal_band,
        height_band_pct=height_band,
        onsite_band_pct=2.0)


# =========================================================================== #
# FIGURES
# =========================================================================== #
def fig_scene():
    """Sentetik sahneyi gorsellestir: (sol) kamera goruntusu — oyuncular kutu olarak;
    (sag) kus-bakisi saha — oyuncu noktalari + kamera. Ileri-model somut."""
    calib, df, gt, proj, corners_img = synthetic_scene(
        L=40.0, W=20.0, n=60, seed=3, height_sigma=0.07, pix_noise=0.0)
    import matplotlib.pyplot as plt
    ps.use()
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    # --- sol: kamera goruntusu (foot->head kutular) ---
    sub = df.iloc[:40]
    for _, r in sub.iterrows():
        fx, fy, bh = r.foot_x, r.foot_y, r.box_h
        bw = bh / 2.8
        axL.add_patch(plt.Rectangle((fx - bw / 2, fy - bh), bw, bh,
                      fill=True, fc=ps.C["sky"], ec=ps.C["blue"], lw=0.6, alpha=0.7))
    cim = np.vstack([corners_img, corners_img[0]])
    axL.plot(cim[:, 0], cim[:, 1], color=ps.C["red"], lw=1.4)
    axL.set_xlim(0, 1920); axL.set_ylim(1080, 0)
    axL.set_aspect("equal"); axL.set_title("(a) sentetik kamera görüntüsü")
    axL.set_xlabel("piksel x"); axL.set_ylabel("piksel y")
    axL.grid(False)
    # --- sag: kus-bakisi ---
    axR.add_patch(plt.Rectangle((0, 0), gt["L"], gt["W"], fill=False, ec=ps.C["ink"], lw=1.4))
    axR.plot([gt["L"] / 2, gt["L"] / 2], [0, gt["W"]], color=ps.C["slate"], lw=0.8)
    # oyuncu ground konumlari (H ile geri-projekte)
    H = np.array(calib["H_img2pitch"])
    foot = df[["foot_x", "foot_y"]].to_numpy(float)
    G = cv2.perspectiveTransform(foot.reshape(-1, 1, 2), H).reshape(-1, 2)
    axR.scatter(G[:, 0], G[:, 1], s=14, color=ps.C["blue"], alpha=0.8, ec="white", lw=0.4)
    axR.scatter([gt["C"][0]], [gt["C"][1]], marker="^", s=70, color=ps.C["red"],
                ec="white", zorder=5, label="kamera")
    axR.set_xlim(-5, gt["L"] + 2); axR.set_ylim(-2, gt["W"] + 2)
    axR.set_aspect("equal"); axR.set_title("(b) kuş bakışı (gerçek 40×20 m)")
    axR.set_xlabel("X (m)"); axR.set_ylabel("Y (m)"); axR.legend(loc="upper right")
    fig.tight_layout()
    ps.save(fig, str(FIGDIR / "sim_scene"))
    plt.close(fig)


def fig_montecarlo(mc):
    import matplotlib.pyplot as plt
    ps.use()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.7))
    L = np.array(mc["L"]); tL = mc["truth"]["L"]
    a1.hist(L, bins=20, color=ps.C["blue"], alpha=0.85, ec="white")
    a1.axvline(tL, color=ps.C["red"], lw=1.8, label=f"gerçek L = {tL:.0f} m")
    a1.axvline(L.mean(), color=ps.C["ink"], lw=1.2, ls="--",
               label=f"ortalama = {L.mean():.2f} m\n(bias +{100*(L.mean()-tL)/tL:.2f}%)")
    # eksen GERCEK MC yayilimina yakinlas: yontem-ici gurultu cok kucuk (band kalibrasyon, fig.3)
    lo, hi = L.mean() - 6 * L.std() - 0.05, L.mean() + 6 * L.std() + 0.05
    a1.set_xlim(lo, hi)
    a1.set_xlabel("kestirilen saha boyu L (m)"); a1.set_ylabel("frekans")
    a1.set_title(f"(a) Monte Carlo — yöntem-içi gürültü (±{100*L.std()/tL:.2f}%)")
    a1.legend(loc="upper right", fontsize=7.5)
    # cam_h dagilimi
    ch = np.array(mc["cam_h"])
    a2.hist(ch, bins=24, color=ps.C["teal"], alpha=0.8, ec="white")
    a2.axvline(4.0, color=ps.C["red"], lw=1.8, label="gerçek = 4.0 m")
    a2.set_xlabel("kestirilen kamera yüksekliği (m)"); a2.set_ylabel("frekans")
    a2.set_title("(b) kamera yüksekliği"); a2.legend(loc="upper right")
    fig.tight_layout()
    ps.save(fig, str(FIGDIR / "sim_montecarlo"))
    plt.close(fig)


def fig_sensitivity(se):
    """Band nereden geliyor: (a) GERCEK Cankaya odak belirsizligi (±13%) vs sentetik
    kusursuz (düz çizgi → band yöntem değil kalibrasyon kaynaklı); (b) boy-priori + fix."""
    import matplotlib.pyplot as plt
    ps.use()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    al = np.array(se["alphas"])
    Lr = np.array(se["L_focal_real"]); L0r = se["L0_real"]
    Ls = np.array(se["L_focal_syn"]); L0s = se["L0_syn"]
    # gercek: oranli eksende (L/L0) ki iki egri ayni grafikte karsilastirilsin
    a1.plot(al, Lr / L0r, color=ps.C["red"], lw=2.2, label=f"gerçek Çankaya (±{se['real_focal_band_pct']:.0f}%)")
    a1.plot(al, Ls / L0s, color=ps.C["blue"], lw=2.0, ls="--",
            label=f"sentetik kusursuz (±{se['syn_focal_band_pct']:.1f}%)")
    a1.axhline(1.0, color=ps.C["slate"], lw=0.8)
    a1.axvline(1.0, color=ps.C["slate"], lw=0.6, ls=":")
    a1.annotate("$f_{ortho}$", (se["f_ortho"] / se["f_real"], (Lr / L0r)[np.argmin(np.abs(al - se["f_ortho"] / se["f_real"]))]),
                textcoords="offset points", xytext=(4, -10), color=ps.C["red"], fontsize=8.5)
    a1.set_xlabel(r"varsayılan odak $f'/f_{\rm calib}$ (tek görüntüden pinlenemez)")
    a1.set_ylabel(r"kestirilen $L\,/\,L_0$")
    a1.set_title("(a) odak belirsizliği → asıl band")
    a1.legend(loc="upper left", fontsize=8)
    # (b) boy-priori + on-site fix
    h = np.array(se["heights"]); Lh = np.array(se["L_height"])
    a2.plot(h, Lh, color=ps.C["teal"], lw=2.2)
    a2.axhline(L0r, color=ps.C["slate"], ls=":", lw=0.8)
    a2.fill_between([1.695, 1.805], L0r * (1 - se["onsite_band_pct"] / 100),
                    L0r * (1 + se["onsite_band_pct"] / 100), color=ps.C["amber"],
                    alpha=0.18, label=f"on-site ölçüm sonrası (±{se['onsite_band_pct']:.0f}%)")
    a2.set_xlabel("varsayılan ortalama boy (m)"); a2.set_ylabel("kestirilen L (m)")
    a2.set_title(f"(b) boy-priori (±{se['height_band_pct']:.0f}%) + on-site fix")
    a2.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    ps.save(fig, str(FIGDIR / "sim_sensitivity"))
    plt.close(fig)


def main():
    FIGDIR.mkdir(parents=True, exist_ok=True)
    print("1) IDEAL exact recovery ...")
    ideal = run_ideal()
    print(json.dumps(ideal["err_pct"], indent=2))
    print("2) MONTE CARLO ...")
    mc = run_montecarlo(K=300)
    print(json.dumps(mc["stats"], indent=2))
    print("3) SENSITIVITY ...")
    se = run_sensitivity()
    print(f"  REAL focal band ±{se['real_focal_band_pct']:.1f}%  (f_calib={se['f_real']:.0f} "
          f"vs f_ortho={se['f_ortho']:.0f})")
    print(f"  SYN  focal band ±{se['syn_focal_band_pct']:.2f}%  (kusursuz homografi → band yöntem değil)")
    print(f"  height-priori band ±{se['height_band_pct']:.1f}%   on-site fix → ±{se['onsite_band_pct']:.0f}%")
    print("figures ...")
    fig_scene(); fig_montecarlo(mc); fig_sensitivity(se)
    OUTJSON.write_text(json.dumps(dict(ideal=ideal, montecarlo=mc, sensitivity=se),
                                  indent=2))
    print("yazildi:", OUTJSON)


if __name__ == "__main__":
    main()
