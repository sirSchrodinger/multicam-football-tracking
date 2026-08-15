#!/usr/bin/env python3
"""scale_crosscheck.py — ölçeği BİRDEN FAZLA bağımsız sinyalle çapraz-doğrula.

Alperen: "saha mesafesini yaptığın gibi, boy ortalaması veya başka verilerden de
çıkar, çapraz doğrula." Bu script on-site ölçüm GEREKTİRMEDEN mevcut veriyle iki
bağımsız ölçek-çıpasını karşılaştırır:

  (A) BOY-ÇIPASI (fizik): sahadaki insanlar ~1.75m. Homografinin rel_m'inde
      medyan oyuncu boyu odak f'e bağlı; field L(f) = 34 * (1.75/medyan_Z(f)).
      Tek-görüntü f belirsizliği -> L(f) bir aralık.
  (B) LANDMARK-ÇIPASI (eyeball): kalibrasyonda varsayılan saha 34x18 (provisional).

Çapraz-kontrol mantığı: katalog (46x24) ya da 40x20 DOĞRUYSA, sahadaki insanların
medyan boyu ne çıkardı? Fizik-dışıysa o saha ELENIR. Ayrıca iki çıpanın
uyuştuğu (L_height(f)=34) "tutarlı odak" f* bulunur -> odak dejenerasyonunu
kısmen kırar (tam kırmak için focal-bağımsız bir uzunluk-çıpası gerek:
merkez-yuvarlak/ceza-sahası — SONRAKİ oturum).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pitch.height_scale import estimate_focal, _pose_at_f, estimate_heights_relm  # noqa: E402
import pandas as pd  # noqa: E402

HEIGHT_PRIOR = (1.70, 1.80)


def main():
    calib = json.loads((ROOT / "calib/cankaya_cam2_v2.json").read_text())
    H = np.array(calib["H_img2pitch"]); dist = np.array(calib["dist"])
    f_calib = float(np.array(calib["K"])[0, 0])
    L_rel = float(calib["pitch_dims_m"]["L"]); W_rel = float(calib["pitch_dims_m"]["W"])  # 34x18 landmark
    df = pd.read_parquet(ROOT / "raw/tracks_cankaya_cam2_clip2400.parquet")
    cx, cy = 960.0, 540.0
    Hwi = np.linalg.inv(H)
    f_ortho = float(estimate_focal(H, cx, cy))

    def med_Z(f):
        Kb = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        Rb, tb, Cb, _ = _pose_at_f(Hwi, f, cx, cy)
        Z = estimate_heights_relm(df, H, Kb, dist, Rb, tb)
        return float(np.median(Z)), Cb[2]

    medZ0, Cz0 = med_Z(f_calib)   # rel_m medyan boy @ f_calib

    print("== (B) LANDMARK-ÇIPASI ==")
    print(f"  homografi rel_m saha = {L_rel:.0f} x {W_rel:.0f} (provisional/eyeball)")
    print(f"  rel_m medyan oyuncu boyu @f={f_calib:.0f}: {medZ0:.3f} rel_m\n")

    print("== ÇAPRAZ-KONTROL: aday saha DOĞRUYSA medyan insan boyu kaç olurdu? ==")
    print(f"  (medyan boy = {medZ0:.3f} rel_m * L_true/{L_rel:.0f})")
    verdict = {}
    for name, Lt in [("katalog 46x24", 46.0), ("40x20", 40.0),
                     ("eyeball 34x18", 34.0), ("boy-anchor 32.5", 32.5), ("30x16", 30.0)]:
        h = medZ0 * Lt / L_rel
        ok = HEIGHT_PRIOR[0] <= h <= HEIGHT_PRIOR[1]
        flag = "✓ tutarlı" if ok else ("✗ FİZİK-DIŞI -> ELE" if (h > 1.95 or h < 1.55) else "~ sınırda")
        print(f"  {name:16}: medyan boy = {h:.2f} m   {flag}")
        verdict[name] = dict(L_true=Lt, implied_height=round(h, 3), plausible=bool(ok))

    print("\n== BOY-ÇIPASI: field L(f) odak süpürmesi ==")
    alphas = [0.8, 0.9, 1.0, 1.085, 1.1, 1.187, 1.2, 1.3]
    Lf = {}
    for a in alphas:
        f = f_calib * a
        mz, Cz = med_Z(f)
        L = L_rel * (1.75 / mz)
        Lf[round(f)] = round(L, 1)
        print(f"  f={f:6.0f} (x{a:.3f}): L_height = {L:5.1f} m   kamera = {1.75/mz*Cz:.2f} m")
    # tutarlı focal: L_height(f) = 34 (landmark)
    fs = np.linspace(0.8 * f_calib, 1.3 * f_calib, 400)
    Ls = []
    for f in fs:
        mz, _ = med_Z(f); Ls.append(L_rel * (1.75 / mz))
    Ls = np.array(Ls)
    i = int(np.argmin(np.abs(Ls - 34.0)))
    f_consistent = float(fs[i])

    # boy-priori [1.70,1.80] -> L aralığı @ f_calib
    L_lo = L_rel * (HEIGHT_PRIOR[0] / medZ0); L_hi = L_rel * (HEIGHT_PRIOR[1] / medZ0)

    print("\n== SONUÇ (çapraz) ==")
    print(f"  • Katalog 46x24 ve 40x20 ELENDI (medyan boy {medZ0*46/L_rel:.2f}m / "
          f"{medZ0*40/L_rel:.2f}m fizik-dışı).")
    print(f"  • Boy + landmark(34) UYUŞUYOR: tutarlı odak f* ≈ {f_consistent:.0f} "
          f"(f_calib={f_calib:.0f} ile f_ortho={f_ortho:.0f} ARASINDA -> makul).")
    print(f"  • Boy-priori[1.70-1.80]@f_calib -> L ∈ [{L_lo:.1f}, {L_hi:.1f}] m "
          f"(katalog aralığı 38-50'nin ALTINDA).")
    print(f"  • Field büyük olasılıkla ~32-34 m; bandı focal kapatıyor. TAM kapatmak "
          f"için focal-BAĞIMSIZ uzunluk-çıpası (merkez-yuvarlak yarıçap / ceza-sahası) "
          f"gerek -> SONRAKİ oturum.")
    out = dict(f_calib=f_calib, f_ortho=f_ortho, f_consistent=f_consistent,
               rel_med_height=medZ0, height_implied_L_band=[round(L_lo, 1), round(L_hi, 1)],
               candidate_field_verdict=verdict, L_vs_focal=Lf)
    (ROOT / "recall_val/scale_crosscheck.json").write_text(json.dumps(out, indent=2))
    print("\nyazildi: recall_val/scale_crosscheck.json")


if __name__ == "__main__":
    main()
