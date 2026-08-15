#!/usr/bin/env python3
"""temporal_mine — izlemeden (tracking) gozetimsiz ZOR-ORNEK madenciligi.

Dayanak: Jin, Liang & Fowlkes, "Unsupervised Hard Example Mining from Videos
for Improved Object Detection" (arXiv:1808.04285, ECCV 2018). Temel fikir:
KISA bir zaman penceresinde nesnelerin gorunumu/konumu surekli (smooth)
degisir, ama bir tek-kare dedektoru tutarsizdir. Tracker'in sagladigi
zamansal tutarliligi "bedava etiket" gibi kullanip dedektorun iki tip hatasini
otomatik kazariz:

  (1) ZAMANSAL FALSE-NEGATIVE (flicker/kacirma): bir iz (track) t-k ve t+k
      karelerinde VAR ama t'de YOK. Gercek bir oyuncu o aradaki karelerde de
      sahada olmali; tracker'in saglam-koprusu kayip kareyi soyler. Bu kayip
      (kare, ara-konum) ciftleri dedektorun KACIRDIGI zor pozitiflerdir
      (uzak/karanlik/occluded oyuncular). Ara konum, kusatan iki tespit
      arasinda DOGRUSAL interpolasyonla kestirilir -> retrain icin hard-FN
      hedefi.

  (2) ORPHAN (yetim) FALSE-POSITIVE: yalnizca cok az ardisik karede beliren,
      hicbir zamansal komsusu olmayan kisa "blip" iz. Tracker onu daha uzun
      bir izle baglayamamissa, buyuk olasilikla bir SAHTE tespittir (golge,
      reklam panosu, seyirci, yari-vucut). Bunlar dedektor icin hard-negative
      (hard-FP) adaylaridir.

API:
  mine_false_negatives(tracks, max_gap_frames, ...) -> [(tid, frame, x, y), ...]
  mine_orphan_fp(tracks, min_len, ...)              -> [tid, ...]

GIRDI: pandas DataFrame; repo izleme-semasiyla uyumlu sutunlar
  tid, frame, (pitch_x, pitch_y) [veya x_col/y_col ile override].
Metrik saha: X=uzunluk, Y=genislik (homography.pixel_to_pitch ciktisi).

DURUSTLUK / SINIRLAR:
  * FN interpolasyonu DOGRUSAL'dir: kusatan iki tespit arasinda sabit-hiz
    varsayar. Gercek yorunge egriyse ara-konum yaklasimdir (kisa boslukta
    hata kucuk; max_gap_frames bu yuzden siniri tutar). Cok uzun bosluklar
    (oyuncunun gercekten cikip-girdigi) mayinlanmaz -> bunlar interp degil,
    yeniden-giris; max_gap_frames ustu atlanir.
  * Orphan-FP olcutu iz-uzunlugu (< min_len) temellidir. Re-ID parcalanmasi
    (bir gercek oyuncunun cok kisa tracklet'lere bolunmesi) da kisa izler
    uretir; boyle bir tracklet hatali yetim sayilabilir. Yani bu modul
    ADAY uretir, kesin-etiket degil; nihai etiket icin (stitch sonrasi) izler
    veya elle dogrulama onerilir.
  * Bu modul yalnizca numpy/pandas kullanir; GPU/dis-model gerektirmez.
    (Kazinan ornekleri kullanan retrain dongusu bu modulun KAPSAMI DISINDA;
    burasi madencilik mantigidir.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Yardimci: bir iz icin (kare, x, y) dizisini temizle/sirala
# ---------------------------------------------------------------------------
def _track_series(g: pd.DataFrame, frame_col: str, x_col: str, y_col: str):
    """Tek bir tid icin kareye gore sirali, kare-tekil (kare, x, y) dizileri.

    Ayni karede birden fazla tespit varsa (dedup oncesi olabilir) konumlar
    ortalanir; boylece bosluk/komsuluk mantigi kare-tekil calisir.
    """
    g2 = (g.groupby(frame_col, as_index=False)[[x_col, y_col]]
            .mean()
            .sort_values(frame_col))
    frames = g2[frame_col].to_numpy()
    xs = g2[x_col].to_numpy(np.float64)
    ys = g2[y_col].to_numpy(np.float64)
    return frames, xs, ys


# ---------------------------------------------------------------------------
# (1) Zamansal FALSE-NEGATIVE madenciligi
# ---------------------------------------------------------------------------
def mine_false_negatives(tracks: pd.DataFrame,
                         max_gap_frames: int,
                         tid_col: str = "tid",
                         frame_col: str = "frame",
                         x_col: str = "pitch_x",
                         y_col: str = "pitch_y"):
    """Izlerin ic-bosluklarini hard-FN hedefi olarak kaz.

    Bir iz f0 ve f1 karelerinde VAR ama aradaki kareler YOK ise (bosluk),
    ve bosluk uzunlugu (f1-f0-1) <= max_gap_frames ise, her kayip kare icin
    f0..f1 arasinda DOGRUSAL interpolasyonla bir (tid, frame, x, y) hedefi
    uretilir.

    Parametreler
    ------------
    tracks : DataFrame  (tid, frame, x_col, y_col sutunlari)
    max_gap_frames : int
        Doldurulabilir en buyuk bosluk (kayip kare sayisi). Bunun ustundeki
        bosluklar gercek cikis/yeniden-giris kabul edilir, MAYINLANMAZ.
    x_col, y_col : konum sutunlari (varsayilan pitch_x/pitch_y, metrik saha).

    Doner
    -----
    list[tuple]  (tid, frame, x, y), (tid, frame)'e gore sirali.
        Kusatan tespitlerden biri NaN konumluysa o bosluk atlanir.
    """
    if max_gap_frames is None or max_gap_frames < 1:
        return []
    need = {tid_col, frame_col, x_col, y_col}
    miss = need - set(tracks.columns)
    if miss:
        raise ValueError(f"mine_false_negatives: eksik sutun(lar): {sorted(miss)}")

    out = []
    for tid, g in tracks.groupby(tid_col, sort=True):
        frames, xs, ys = _track_series(g, frame_col, x_col, y_col)
        if frames.size < 2:
            continue  # tek-tespit izde ic-bosluk olamaz
        for i in range(frames.size - 1):
            f0 = int(frames[i]); f1 = int(frames[i + 1])
            gap = f1 - f0 - 1
            if gap < 1 or gap > max_gap_frames:
                continue  # bitisik kareler ya da cok uzun bosluk -> atla
            x0, y0 = xs[i], ys[i]
            x1, y1 = xs[i + 1], ys[i + 1]
            if not (np.isfinite(x0) and np.isfinite(y0)
                    and np.isfinite(x1) and np.isfinite(y1)):
                continue  # kusatan konum NaN -> interpolasyon yapilamaz
            span = float(f1 - f0)
            for t in range(f0 + 1, f1):
                a = (t - f0) / span  # 0<a<1 dogrusal kesir
                x = x0 + a * (x1 - x0)
                y = y0 + a * (y1 - y0)
                out.append((tid, int(t), float(x), float(y)))
    return out


# ---------------------------------------------------------------------------
# (2) Orphan (yetim) FALSE-POSITIVE madenciligi
# ---------------------------------------------------------------------------
def mine_orphan_fp(tracks: pd.DataFrame,
                   min_len: int,
                   tid_col: str = "tid",
                   frame_col: str = "frame"):
    """Cok kisa (zamansal komsusuz) izleri hard-FP adayi olarak isaretle.

    Bir iz min_len'den AZ tekil karede goruluyorsa (kisa, izole blip),
    tracker onu surdurulebilir bir yorungeye baglayamamis demektir -> sahte
    tespit (hard-negative) adayi.

    Parametreler
    ------------
    tracks : DataFrame  (tid, frame sutunlari)
    min_len : int
        Esik. nunique(frame) < min_len olan izler yetim sayilir.

    Doner
    -----
    list  yetim-FP aday tid'leri (tid'e gore sirali).
    """
    if min_len is None or min_len < 1:
        return []
    need = {tid_col, frame_col}
    miss = need - set(tracks.columns)
    if miss:
        raise ValueError(f"mine_orphan_fp: eksik sutun(lar): {sorted(miss)}")

    lengths = tracks.groupby(tid_col)[frame_col].nunique()
    orphans = lengths.index[lengths.to_numpy() < int(min_len)]
    return list(orphans)
