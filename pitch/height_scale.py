#!/usr/bin/env python3
"""Oyuncu-boyu metrik olcek — sahadan/kaleden BAGIMSIZ mutlak olcek.

NEDEN: Halisaha saha olculeri ve kale boyutlari OYNAK (Alperen) -> 25x45 snap veya
3m-kale capasi guvenilmez. Ama sahadaki ON-LARCA insan EVRENSEL ~1.75m olcu cubugu.

YONTEM (tek-goruntu metroloji, dogru sürum):
  1. QA-gecmis zemin homografisi H (undist img -> world rel_m, dogru ASPECT) natural-camera
     (kare piksel, ana-nokta merkez) varsayimiyla AYRISTIRILIR -> odak f, poz R,t, kamera merkezi C.
     (Zhang: |r1|=|r2|, r1.r2=0 -> f; lambda=1/|K^-1 h1|.)
  2. Temiz DIK tespitler (yuksek conf, alt-kirpik degil, dik en-boy) icin TAM kamera projeksiyonu
     P=K[R|t] ile her oyuncunun boyu rel_m'de COZULUR (foot->ground G; head ray ile G+Z*z_hat kesisimi).
  3. Robust medyan-boy := varsayilan 1.75 m -> olcek c (m / rel_m). Mutlak metre boyle gelir;
     hicbir saha/kale olcusu varsayilmaz.

DURUSTLUK: c bir VARSAYIMDAN gelir (populasyon ortalama boyu ~1.75 m) -> 'm (approx +-N%)'.
Band = boy-priori araligi (1.70-1.80) + model belirsizligi. Sanity kapilari: kamera yuksekligi
2-8 m, boy CV < 0.2, f makul; aksi halde metrik IDDIA EDILMEZ (relative_m kalir).

Lisans: numpy + cv2 (BSD/MIT). GPU yok.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import cv2

ASSUMED_HEIGHT_M = 1.75          # populasyon ortalama (karma yetiskin)
HEIGHT_PRIOR_M = (1.70, 1.80)    # band icin makul ortalama-boy araligi
CAM_HEIGHT_OK_M = (2.0, 8.0)     # halisaha kamera montaj yuksekligi sanity
CV_MAX = 0.20                    # boy dagilimi cok genisse model bozuk


def _pose_at_f(Hwi, f, cx, cy):
    """Sabit f icin poz: (R, t, C, lam). Hwi = inv(H) (world->img)."""
    K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    Ki = np.linalg.inv(K)
    lam = 1.0 / np.linalg.norm(Ki @ Hwi[:, 0])
    r1 = lam * (Ki @ Hwi[:, 0]); r2 = lam * (Ki @ Hwi[:, 1]); t = lam * (Ki @ Hwi[:, 2])
    r3 = np.cross(r1, r2)
    R = np.column_stack([r1, r2, r3])     # R'yi SVD ile ZORLAMA: t ile tutarli kalmali
    C = -R.T @ t
    if C[2] < 0:                          # kamera zeminin ustunde olmali -> isaret duzelt
        r1, r2, t = -r1, -r2, -t
        r3 = np.cross(r1, r2); R = np.column_stack([r1, r2, r3]); C = -R.T @ t
    return R, t, C, lam


def estimate_focal(H_img2pitch, cx=960.0, cy=540.0):
    """Zemin homografisinden odak f (px) — ORTOGONALLIK kriteri (Zhang, dogru sürüm).

    Norm-esitligi |r1|=|r2| undistort kusuru yuzunden hicbir f'te tam saglanmaz
    (kalici anizotropi -> tek basina f'i pinlemez); r1.r2=0 ise temiz minimumlu.
    """
    Hwi = np.linalg.inv(np.asarray(H_img2pitch, float))
    h1, h2 = Hwi[:, 0], Hwi[:, 1]

    def resid(f):
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]); Ki = np.linalg.inv(K)
        w = Ki.T @ Ki
        s = 0.5 * (h1 @ w @ h1 + h2 @ w @ h2) + 1e-12
        return abs(h1 @ w @ h2) / s

    fs = np.linspace(300.0, 4000.0, 3701)
    f = float(fs[int(np.argmin([resid(x) for x in fs]))])
    fs2 = np.linspace(max(300.0, f - 20), f + 20, 401)
    return float(fs2[int(np.argmin([resid(x) for x in fs2]))])


def decompose_ground_homography(H_img2pitch, cx=960.0, cy=540.0, f=None):
    """Zemin homografisini ayristir -> (f, R, t, C, scale). H: undist img -> world(rel_m).

    f verilmezse estimate_focal (ortogonallik) ile bulunur; verilirse o f kullanilir
    (band/duyarlilik analizi icin). Doner: f, R, t, C(kamera merkezi rel_m), lam.
    """
    Hwi = np.linalg.inv(np.asarray(H_img2pitch, float))
    if f is None:
        f = estimate_focal(H_img2pitch, cx, cy)
    R, t, C, lam = _pose_at_f(Hwi, f, cx, cy)
    return f, R, t, C, lam


def _clean_upright(df):
    """Temiz dik tespit maskesi: yuksek conf, alt-kirpik degil, dik en-boy."""
    w = df["box_w"].to_numpy(float)
    h = df["box_h"].to_numpy(float)
    conf = df["conf"].to_numpy(float)
    crop = df["bottom_cropped"].to_numpy(bool) if "bottom_cropped" in df else np.zeros(len(df), bool)
    ar = h / np.clip(w, 1.0, None)
    return (conf > 0.6) & (~crop) & (h > 50) & (ar > 1.7) & (ar < 5.0)


def estimate_heights_relm(df, H_img2pitch, K, dist, R, t):
    """Temiz tespitler icin oyuncu boyu (rel_m). Vektorize TAM-kamera Z cozumu.

    foot/head HAM piksel -> undistort -> foot ground G (via H); head ray ile dikey
    G+Z*z_hat kesisimi (P=K[R|t], a=P[:,2] sabit). Doner: Z dizisi (rel_m, >0, sonlu).
    """
    H = np.asarray(H_img2pitch, float)
    K = np.asarray(K, float); dist = np.asarray(dist, float)
    m = _clean_upright(df)
    foot = df.loc[m, ["foot_x", "foot_y"]].to_numpy(float)
    hpx = df.loc[m, "box_h"].to_numpy(float)
    head = np.column_stack([foot[:, 0], foot[:, 1] - hpx])

    def undist(Pn):
        return cv2.undistortPoints(Pn.reshape(-1, 1, 2).astype(np.float64), K, dist,
                                   P=K).reshape(-1, 2)
    fu, tu = undist(foot), undist(head)
    G = cv2.perspectiveTransform(fu.reshape(-1, 1, 2), H).reshape(-1, 2)  # ground rel_m

    P = K @ np.column_stack([R, t])           # 3x4
    a = P[:, 2]                                 # Z katsayisi (sabit)
    X, Y = G[:, 0], G[:, 1]
    b = (P[:, 0][None, :] * X[:, None] + P[:, 1][None, :] * Y[:, None]
         + P[:, 3][None, :])                   # (N,3) sabit kisim
    u, v = tu[:, 0], tu[:, 1]
    A0 = a[0] - u * a[2]; A1 = a[1] - v * a[2]            # (N,)
    R0 = u * b[:, 2] - b[:, 0]; R1 = v * b[:, 2] - b[:, 1]
    num = A0 * R0 + A1 * R1
    den = A0 * A0 + A1 * A1 + 1e-12
    Z = num / den
    Z = Z[np.isfinite(Z) & (Z > 0)]
    return Z


def player_height_scale(calib_path, tracks_path, assumed_height_m=ASSUMED_HEIGHT_M,
                        height_prior_m=HEIGHT_PRIOR_M, cx=960.0, cy=540.0):
    """Ana giris: oyuncu-boyundan mutlak olcek. Doner: dururst dict.

    scale_factor c (m/rel_m), implied_field_LW, camera_height_m, height stats, band_pct,
    sanity (ok/reasons). Hicbir saha/kale olcusu varsayilmaz.
    """
    import pandas as pd
    d = json.loads(Path(calib_path).read_text())
    H = np.array(d["H_img2pitch"]); K = np.array(d["K"]); dist = np.array(d["dist"])
    Ldims = d.get("pitch_dims_m", {}) or {}
    L_rel = float(Ldims.get("L", 34.0)); W_rel = float(Ldims.get("W", 18.0))

    df = pd.read_parquet(tracks_path)
    Hwi = np.linalg.inv(H)
    # TUTARLILIK: undistort + poz + projeksiyon AYNI f ile olmali. H, calib'in K'siyla
    # (fx) undistort edilmis uzayda kuruldu -> f = K[0,0]. (estimate_focal ortogonallik
    # optimumu farkli verebilir ama o zaman undistort-f ile H tutarsiz kalir.)
    f = float(K[0, 0])
    f_ortho = estimate_focal(H, cx, cy)   # teshis: ortogonallik-optimal f (band ipucu)
    R, t, C, lam = _pose_at_f(Hwi, f, cx, cy)

    def _med_c_at(fb):
        """Sabit fb ile (undistort+poz+P hepsi fb) medyan-boy ve c. TUTARLI."""
        Kb = np.array([[fb, 0, cx], [0, fb, cy], [0, 0, 1.0]])
        Rb, tb, _, _ = _pose_at_f(Hwi, fb, cx, cy)
        Zb = estimate_heights_relm(df, H, Kb, dist, Rb, tb)
        if not Zb.size:
            return None, None
        mb = float(np.median(Zb))
        return mb, (assumed_height_m / mb if mb > 0 else None)

    med, c = _med_c_at(f)
    Z = estimate_heights_relm(df, H, K, dist, R, t)   # rapor istatistikleri icin (f tutarli)
    n = int(Z.size)
    mad = float(np.median(np.abs(Z - med))) if n else float("nan")
    cv = float(np.std(Z) / med) if n and med and med > 0 else float("nan")
    cam_h_m = float(C[2] * c) if c else float("nan")

    # BAND: (a) ortalama-boy priori [1.70,1.80]; (b) ODAK belirsizligi f in [0.8f,1.3f]
    #       (ortogonallik-optimal f bu araliktadir; norm-anizotropisi cozulmedigi icin gercek).
    cand_c = [c]
    for fb in (0.80 * f, 1.30 * f, f_ortho):
        _, cb = _med_c_at(fb)
        if cb:
            cand_c.append(cb)
    c_f_lo, c_f_hi = min(cand_c), max(cand_c)
    c_h_lo, c_h_hi = height_prior_m[0] / med, height_prior_m[1] / med
    c_all_lo = min(c_f_lo, c_h_lo); c_all_hi = max(c_f_hi, c_h_hi)
    band_pct = float(round(100.0 * 0.5 * (c_all_hi - c_all_lo) / c, 1)) if c else float("nan")

    reasons = []
    if not (n >= 200):
        reasons.append(f"yetersiz temiz tespit ({n} < 200)")
    if not np.isfinite(cv) or cv > CV_MAX:
        reasons.append(f"boy CV {cv:.2f} > {CV_MAX} (model/undistort suphesi)")
    if not (CAM_HEIGHT_OK_M[0] <= cam_h_m <= CAM_HEIGHT_OK_M[1]):
        reasons.append(f"kamera yuksekligi {cam_h_m:.1f}m mantiksiz {CAM_HEIGHT_OK_M}")
    ok = (len(reasons) == 0)

    return dict(
        ok=ok, reasons=reasons,
        scale_factor=round(c, 4) if med > 0 else None,
        focal_px=round(f, 1),
        camera_height_m=round(cam_h_m, 2),
        median_height_relm=round(med, 4), height_cv=round(cv, 3),
        height_mad_relm=round(mad, 4), n_detections=n,
        assumed_height_m=assumed_height_m, band_pct=band_pct,
        implied_field_m=dict(L=round(L_rel * c, 1), W=round(W_rel * c, 1)) if med > 0 else None,
        rel_dims=dict(L=L_rel, W=W_rel),
        method="player_height_singleview_metrology",
        provenance=("sahadan/kaleden BAGIMSIZ; zemin homografisi ayristirma (f) + "
                    "oyuncu boyu (~1.75m evrensel) tek-goruntu metroloji"),
    )


def make_v_height(calib_path, out_path, tracks_path, assumed_height_m=ASSUMED_HEIGHT_M):
    """QA-gecmis calib'i oyuncu-boyu olcegiyle zenginlestir -> yeni calib JSON.

    scale_anchor kind='player_height'; reconciled pitch_dims_m = implied_field. Sanity
    fail ise scale_anchor YAZILMAZ (relative_m kalir, dururst). Doner: meta dict.
    """
    res = player_height_scale(calib_path, tracks_path, assumed_height_m)
    d = json.loads(Path(calib_path).read_text())
    if res["ok"]:
        c = res["scale_factor"]
        d["scale_anchor"] = dict(
            kind="player_height", schema="player_height/v1",
            method=res["method"], scale_factor=c,
            assumed_height_m=assumed_height_m, median_height_relm=res["median_height_relm"],
            height_cv=res["height_cv"], n_detections=res["n_detections"],
            camera_height_m=res["camera_height_m"], focal_px=res["focal_px"],
            asserted_dims_LW=[res["implied_field_m"]["L"], res["implied_field_m"]["W"]],
            band_pct=res["band_pct"], approximate=True, verified=False, reconciled=True,
            label=f"approximate oyuncu-boyu olcek, +-{res['band_pct']:.0f}%",
            provenance=res["provenance"])
        # implied field dims'i kanonik yap (rel scale ile)
        d["pitch_dims_m"] = dict(L=res["implied_field_m"]["L"], W=res["implied_field_m"]["W"])
    else:
        d["scale_anchor"] = None
        d["_height_scale_rejected"] = res["reasons"]
    Path(out_path).write_text(json.dumps(d, indent=2))
    return res


def scale_from_known_distance(calib_path, out_path, p1_px, p2_px, meters):
    """KESIN olcek: sahada bilinen TEK bir mesafe (iki HAM piksel + gercek metre).

    En dogru/ucuz yol (Alperen on-site): bir cizgi/kale-direkleri arasi serit-metre olcumu.
    p1_px,p2_px HAM piksel; H undistorted bekler -> once undistort. c = meters / rel_m_mesafe.
    scale_anchor verified=True, band ~+-2% (olcum+ayak-noktasi hatasi) -> rapor cipak 'm', sprint serbest.
    """
    d = json.loads(Path(calib_path).read_text())
    H = np.array(d["H_img2pitch"]); K = np.array(d["K"]); dist = np.array(d["dist"])
    P = np.array([p1_px, p2_px], float).reshape(-1, 1, 2)
    Pu = cv2.undistortPoints(P, K, dist, P=K)
    G = cv2.perspectiveTransform(Pu, H).reshape(-1, 2)
    rel = float(np.hypot(*(G[0] - G[1])))
    c = float(meters) / rel if rel > 0 else float("nan")
    L_rel = float((d.get("pitch_dims_m") or {}).get("L", 34.0))
    W_rel = float((d.get("pitch_dims_m") or {}).get("W", 18.0))
    d["scale_anchor"] = dict(
        kind="known_distance", schema="known_distance/v1",
        scale_factor=round(c, 5), measured_m=float(meters), rel_distance=round(rel, 4),
        asserted_dims_LW=[round(L_rel * c, 2), round(W_rel * c, 2)],
        band_pct=2.0, approximate=True, verified=True, reconciled=True,
        label="measured (tek mesafe, +-~2%)",
        provenance="on-site bilinen mesafe olcumu (serit-metre/satranc); en guvenilir ucuz yol")
    d["pitch_dims_m"] = dict(L=round(L_rel * c, 2), W=round(W_rel * c, 2))
    Path(out_path).write_text(json.dumps(d, indent=2))
    return dict(scale_factor=c, rel_distance=rel, measured_m=meters,
                implied_field_m=d["pitch_dims_m"], verified=True)


if __name__ == "__main__":
    import sys
    cp = sys.argv[1] if len(sys.argv) > 1 else "calib/cankaya_cam2_v2.json"
    tp = sys.argv[2] if len(sys.argv) > 2 else "raw/tracks_cankaya_cam2_clip2400.parquet"
    r = player_height_scale(cp, tp)
    print(json.dumps(r, indent=2, ensure_ascii=False))
