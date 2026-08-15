#!/usr/bin/env python3
"""calib/click_landmarks.py — saha ÇİZGİ tıklayıcı (line_calib girdisi).

Neden çizgi, köşe değil: yakın-sol köşe occluded + uzak çizgiler piksel-fakiri.
line_calib 5 adlı çizgiyi fit edip kesiştirerek 4 köşeyi (occluded dahil) kurtarır.
Bu araç o 5 çizgi + (ölçek için) 4 kale-direği tabanını toplar.

TÜM tıklamalar UNDISTORTED piksel uzayında (calib/undist_clean.png üzerinde).

KULLANIM (kendi oturumunda, `!` ile):
    ! venv/bin/python calib/click_landmarks.py

KONTROL:
    sol-tık  : nokta ekle (aktif aşamaya)
    u        : son noktayı geri al
    n / Enter: sonraki aşama
    b        : önceki aşama
    s        : ANINDA kaydet (JSON)
    q        : kaydet + çık
    fare-tekerlek / toolbar: yakınlaş-kaydır (hassas tık için ZOOM yap)
       not: pan/zoom modundayken tık nokta EKLEMEZ; tıklamadan önce modu kapat.

Aşamalar (sıra önemli; her çizgi için noktaları ÇİZGİ BOYUNCA yay):
"""
import sys, json
from pathlib import Path
import numpy as np, cv2
import matplotlib
import matplotlib.pyplot as plt

# varsayılan kısayolları kapat (s=kaydet-dialog, ok=geçmiş, q=çık çakışmasın)
for _k in ("keymap.save", "keymap.back", "keymap.forward", "keymap.quit",
           "keymap.xscale", "keymap.yscale"):  # p=pan, o=zoom KALSIN (çakışmaz)
    matplotlib.rcParams[_k] = []

OUT = "calib/clicks_cankaya_cam2.json"
IMG = "calib/undist_clean.png"

# (anahtar, insan-tarif, min nokta, renk)
STAGES = [
    ("touchline_y0", "YAKIN kenar çizgisi (kameraya en yakın; yeşil↔kırmızı-tartan sınırı, "
                     "sol-alttan sağa). Çizgi boyunca >=3 nokta yay.", 3, "#ff3b30"),
    ("touchline_yW", "UZAK kenar çizgisi (sahanın ÜST kenarı, uzak boylu çizgi). >=3 nokta.", 3, "#ff9500"),
    ("endline_x0",   "YAKIN kale çizgisi (yakın kalenin direkleri arasından geçen dip çizgi). >=2 nokta.", 2, "#34c759"),
    ("endline_xL",   "UZAK kale çizgisi (uzak kale dip çizgisi, sağ-üst). >=2 nokta.", 2, "#00c7be"),
    ("halfway",      "ORTA çizgi (orta yuvarlağın içinden geçen saha-ortası çizgi). >=2 nokta.", 2, "#5856d6"),
    ("goal_near_L",  "[ÖLÇEK] YAKIN kale SOL direk TABANI (zemine değdiği nokta). 1 tık.", 1, "#af52de"),
    ("goal_near_R",  "[ÖLÇEK] YAKIN kale SAĞ direk TABANI. 1 tık.", 1, "#ff2d55"),
    ("goal_far_L",   "[ÖLÇEK] UZAK kale SOL direk TABANI. 1 tık.", 1, "#ffcc00"),
    ("goal_far_R",   "[ÖLÇEK] UZAK kale SAĞ direk TABANI. 1 tık.", 1, "#a2845e"),
]
LINE_KEYS = {"touchline_y0", "touchline_yW", "endline_x0", "endline_xL", "halfway"}


class Clicker:
    def __init__(self, img):
        self.img = img
        self.idx = 0
        self.data = {k: [] for k, *_ in STAGES}
        self.fig, self.ax = plt.subplots(figsize=(17, 10))
        self.ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self._grid()
        self.artists = {k: [] for k, *_ in STAGES}
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self._title()

    def _grid(self):
        h, w = self.img.shape[:2]
        for x in range(0, w, 100):
            self.ax.axvline(x, color="cyan", lw=0.3, alpha=0.35)
        for y in range(0, h, 100):
            self.ax.axhline(y, color="cyan", lw=0.3, alpha=0.35)
        self.ax.set_xlim(0, w); self.ax.set_ylim(h, 0)

    def _title(self):
        key, desc, need, col = STAGES[self.idx]
        n = len(self.data[key])
        ok = "✓" if n >= need else f"{n}/{need}"
        self.ax.set_title(f"[{self.idx+1}/{len(STAGES)}] {key} ({ok})  —  {desc}\n"
                          f"sol-tık=ekle  u=geri  n/Enter=sonraki  b=önceki  s=kaydet  q=çık",
                          fontsize=10, color=col)
        self.fig.canvas.draw_idle()

    def _toolbar_busy(self):
        tb = getattr(self.fig.canvas, "toolbar", None)
        return bool(getattr(tb, "mode", "")) if tb else False

    def on_click(self, ev):
        if ev.inaxes != self.ax or ev.button != 1 or self._toolbar_busy():
            return
        if ev.xdata is None:
            return
        key, desc, need, col = STAGES[self.idx]
        self.data[key].append([float(ev.xdata), float(ev.ydata)])
        a = self.ax.plot(ev.xdata, ev.ydata, "o", ms=7, mfc=col, mec="white", mew=1.2)[0]
        t = self.ax.annotate(str(len(self.data[key])), (ev.xdata, ev.ydata),
                             color="white", fontsize=8, xytext=(4, 4),
                             textcoords="offset points")
        self.artists[key].append((a, t))
        self._title()

    def on_key(self, ev):
        key, *_ = STAGES[self.idx]
        if ev.key in ("n", "enter", "right"):
            self.idx = min(self.idx + 1, len(STAGES) - 1); self._title()
        elif ev.key in ("b", "left"):
            self.idx = max(self.idx - 1, 0); self._title()
        elif ev.key == "u" and self.data[key]:
            self.data[key].pop()
            a, t = self.artists[key].pop(); a.remove(); t.remove()
            self._title()
        elif ev.key == "s":
            self.save()
        elif ev.key == "q":
            self.save(); plt.close(self.fig)

    def save(self):
        lines = {k: v for k, v in self.data.items() if k in LINE_KEYS and v}
        points = {k: v[0] for k, v in self.data.items() if k not in LINE_KEYS and v}
        Path(OUT).write_text(json.dumps(
            {"image": IMG, "space": "undistorted", "lines": lines, "points": points},
            indent=2))
        done = [f"{k}:{len(v)}" for k, v in self.data.items() if v]
        print("kaydedildi ->", OUT, "|", "  ".join(done))


if __name__ == "__main__":
    img = cv2.imread(IMG)
    if img is None:
        sys.exit(f"okunamadi: {IMG} (önce `venv/bin/python calib/recalib.py undist_grid`)")
    print(__doc__)
    for i, (k, d, n, _) in enumerate(STAGES, 1):
        print(f"  {i}. {k:14s} (>= {n}) — {d}")
    print("\nPencere açılıyor… ZOOM yaparak hassas tıkla. Bitince 'q'.")
    c = Clicker(img)
    plt.show()
    c.save()
    print("bitti.")
