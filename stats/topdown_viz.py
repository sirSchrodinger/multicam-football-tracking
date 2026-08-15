#!/usr/bin/env python3
"""Durust top-down GORSEL urun katmani (render-only): tek cok-panel rapor figuru.

Bu modul stats.topdown_stats ciktilarini matplotlib figurlerine cevirir.
Hicbir YENI projeksiyon/istatistik mantigi YOKTUR: tek projeksiyon kaynagi
PitchHomography, tek durustluk kaynagi generate_topdown_report'un report dict'i.
topdown_stats.py / homography.py / line_calib.py'ye SIFIR duzenleme yapilir.

ANA CIKTI -> render_report_figure(): GridSpec(3,2) tek figur
  [A] tum-oyuncu esit-alan doluluk heatmap'i  (gs[0,0])
  [B] kus-bakisi tracklet yorunge overlay      (gs[0,1])  -- Design #3
  [C] per-tracklet mini-heatmap kucuk-coklu    (gs[1,:])
  [D] durustluk/QC footer                       (gs[2,:])
Panel cizicileri ayrica tekil PNG'ler icin de yeniden kullanilabilir.

DURUSTLUK SOZLESMESI (kilitli, her panelde tutulur):
  - NaN kapsama = GRI (cmap.set_bad); 0 = gozlendi-bos (en dusuk renk). asla
    sifiri gri yapma. invariant: np.ma.masked_invalid + set_bad HER ZAMAN cift.
  - birim relative_m; ASLA 'm'/'km/h'; ivme ASLA cizilmez.
  - per-tid == FRAGMENT -> basliklar 'tracklet' der; footer stitcher unwired der.
  - takim katmani YOK; tek tum-oyuncu heatmap.
  - 3 m kale agzi NOMINAL tohum (scale_anchor=null) -> 'provisional'.
  - fail-closed: reddedilen/eksik calib -> RuntimeError; bozuk H'den
    esit-alan gorunumlu PNG asla uretilmez.

Lisans: matplotlib (PSF/BSD-uyumlu) + numpy/scipy. GPU yok.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless; PNG render
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from stats.topdown_stats import (  # noqa: E402
    accept_calib_qa, prepare, generate_topdown_report,
    heatmap_grid_equal_area, coverage_mask_pitch,
    MIN_TRACK_FRAMES, MAX_GAP_S,
)

GOAL_HALF_M = 1.5            # 3 m kale agzi -> +/-1.5 m merkez etrafinda
UNKNOWN_GREY = "#9aa0a6"     # kapsama-disi (NaN) hucre rengi = 'bilinmiyor'
PITCH_LINE = "#e8e8e8"
GOAL_COLOR = "yellow"        # 3 m kale agzi (nominal tohum)

# NaN=gri invariant: set_bad daima masked_invalid ile eslesir.
_OCC_CMAP = plt.get_cmap("inferno").copy()
_OCC_CMAP.set_bad(UNKNOWN_GREY)
_MINI_CMAP = plt.get_cmap("magma").copy()
_MINI_CMAP.set_bad(UNKNOWN_GREY)
_TRAJ_CMAP = plt.get_cmap("viridis")


# ===========================================================================
# saha cizimi (metre uzayinda; perspektif yok)
# ===========================================================================
def _draw_pitch(ax, homo, dims, line_c=PITCH_LINE, lw=1.4, goal_lw=4.0):
    """Template cizgileri + orta daire + 3 m SARI kale agzi (metre uzayi)."""
    L, W = dims
    for (p0, p1) in homo._template_segments_m():
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=line_c, lw=lw, zorder=3)
    r = getattr(getattr(homo, "template", None), "center_circle_r_m", None)
    if r:
        th = np.linspace(0, 2 * np.pi, 100)
        ax.plot(L / 2 + r * np.cos(th), W / 2 + r * np.sin(th),
                color=line_c, lw=lw, zorder=3)
    yc = W / 2.0
    for gx in (0.0, L):  # x=0 ve x=L kale agizlari
        ax.plot([gx, gx], [yc - GOAL_HALF_M, yc + GOAL_HALF_M],
                color=GOAL_COLOR, lw=goal_lw, solid_capstyle="butt", zorder=4)
    ax.set_xlim(-1.0, L + 1.0)
    ax.set_ylim(-1.0, W + 1.0)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])


# ===========================================================================
# per-fragment presence grid (mini-heatmap; KENDI tepesine normalize -> SEKIL)
# ===========================================================================
def _presence_grid(px, py, use, xedges, yedges, fps, cover, smooth=True):
    """Bir tracklet-altkumesi icin esit-alan presence grid; KENDI nanmax'ina
    normalize. heatmap_grid_equal_area ile AYNI binleme + maske-bilincli
    smoothing; ama mini-heatmap bir oran degil SEKIL gosterir (panel-arasi
    BUYUKLUK kiyaslanamaz). cover False hucre NaN (gri). Bos altkume -> tum-NaN.
    """
    x = px[use]; y = py[use]
    Hcnt, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])  # (nx,ny)
    pres = (Hcnt.T) / max(fps, 1e-9)                          # (ny,nx)
    if smooth and pres.any():
        try:
            from scipy.ndimage import gaussian_filter
            bin_m = float(xedges[1] - xedges[0])
            sigma = max(0.5, 1.0 / bin_m)
            w = cover.astype(np.float64)
            num = gaussian_filter(pres * w, sigma)
            den = gaussian_filter(w, sigma)
            with np.errstate(divide="ignore", invalid="ignore"):
                pres = np.where(den > 1e-9, num / den, pres)
        except Exception:
            pass
    g = np.where(cover, pres, np.nan)
    pk = np.nanmax(g) if np.isfinite(g).any() else 0.0
    if pk > 0:
        g = g / pk
    g[~cover] = np.nan
    return g


def _select_top_tracklets(prep, n):
    """En uzun n tracklet (n_obs); MIN_TRACK_FRAMES alti elenir. -> [(i,tid),...]."""
    order = np.argsort(prep.counts)[::-1]
    out = [(int(i), int(prep.uniq[i])) for i in order
           if prep.counts[i] >= MIN_TRACK_FRAMES]
    return out[:n]


# ===========================================================================
# PANEL A — tum-oyuncu esit-alan doluluk heatmap'i
# ===========================================================================
def panel_heatmap(ax, grid, edges, homo, dims, unit, cbar=True, fig=None,
                  title="Tum-oyuncu yogunluk (esit-alan)"):
    """[A] tum-oyuncu esit-alan heatmap. NaN=gri, 0=gozlendi-bos, kale sari.

    masked_invalid + set_bad cift; vmin=0, vmax=p98(finite) -> yogun bolge
    diger her seyi soldurmaz.
    """
    L, W = dims
    finite = grid[np.isfinite(grid)]
    vmax = float(np.percentile(finite, 98)) if finite.size else 1.0
    if not (vmax > 0):
        vmax = 1.0
    im = ax.imshow(np.ma.masked_invalid(grid), origin="lower",
                   extent=[0.0, L, 0.0, W], cmap=_OCC_CMAP, aspect="equal",
                   interpolation="nearest", vmin=0.0, vmax=vmax, zorder=1)
    _draw_pitch(ax, homo, dims)
    ax.set_title(title, fontsize=11, pad=6)
    if cbar and fig is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label(
            f"gorunur-zaman normalize doluluk  [{unit}]  (color@p98)",
            fontsize=8)
        cb.ax.tick_params(labelsize=7)
    ax.text(0.5, -0.02, "gri = bilinmiyor (0 degil)",
            transform=ax.transAxes, ha="center", va="top", fontsize=7,
            color="0.4")
    return im


# ===========================================================================
# PANEL B — kus-bakisi yorunge overlay (Design #3: vektorize LineCollection)
# ===========================================================================
def panel_trajectories(ax, prep, homo, dims, fig=None,
                       title="Kus-bakisi yorunge (tum tracklet; renk=zaman, "
                             "saydam=dusuk-guven)"):
    """[B] TUM tracklet'ler tek LineCollection. Renk=klip zamani; alpha+kalinlik
    = guven (uzak-uc/piksel-fakiri segmentler soldurulur). Segmentler bosluk/
    teleport/kadraj-disinda KIRILIR -> uydurma duz cizgi yok (#2,#5).
    Yon Panel A ile AYNI (origin lower / Y-yukari).
    """
    _draw_pitch(ax, homo, dims)

    P = np.column_stack([prep.px, prep.py])
    good = (prep.in_pitch & prep.in_frame
            & np.isfinite(prep.px) & np.isfinite(prep.py))
    dt = np.diff(prep.t_sec)
    same = prep.codes[1:] == prep.codes[:-1]
    keep = same & (dt > 0) & (dt <= MAX_GAP_S) & good[:-1] & good[1:]

    t_min = float(np.nanmin(prep.t_sec)) if prep.t_sec.size else 0.0
    t_max = float(np.nanmax(prep.t_sec)) if prep.t_sec.size else 1.0
    norm = Normalize(t_min, t_max if t_max > t_min else t_min + 1.0)

    if keep.any():
        seg = np.stack([P[:-1][keep], P[1:][keep]], axis=1)   # (M,2,2)
        t_seg = prep.t_sec[1:][keep]
        wcol = (0.5 * (prep.w_conf[:-1] + prep.w_conf[1:]))[keep]
        rgba = _TRAJ_CMAP(norm(t_seg))
        rgba[:, 3] = np.clip(wcol, 0.15, 0.9)                 # guven -> alpha
        lw = 0.6 + 2.0 * np.clip(wcol, 0.0, 1.0)              # guven -> kalinlik
        lc = LineCollection(seg, colors=rgba, linewidths=lw, zorder=5,
                            capstyle="round")
        ax.add_collection(lc)

    ax.set_title(title, fontsize=11, pad=6)
    if fig is not None:
        sm = ScalarMappable(norm=norm, cmap=_TRAJ_CMAP)
        sm.set_array([])
        cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label("klip zamani (s)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    ax.text(0.5, -0.02, "saydam/ince = dusuk-guven (uzak-uc, piksel-fakiri)",
            transform=ax.transAxes, ha="center", va="top", fontsize=7,
            color="0.4")


# ===========================================================================
# PANEL C — per-tracklet mini-heatmap kucuk-coklu
# ===========================================================================
def panel_tracklet_grid(fig, subspec, prep, homo, dims, top, cover, edges,
                        report=None, ncols=6):
    """[C] per-tracklet mini-heatmap grid. Kendi tepesine normalize (SEKIL).
    Baslik 't{tid} n={n}' + report'tan far_third_frac>0.4 ise ' far!'.
    Baslik 'tracklet' der, ASLA 'player'.
    """
    xe, ye = edges
    players = (report or {}).get("players", {}) if report else {}
    n = len(top)
    nrows = int(np.ceil(n / ncols)) if n else 1
    gs = subspec.subgridspec(nrows, ncols, wspace=0.10, hspace=0.30)
    for k, (i, tid) in enumerate(top):
        ax = fig.add_subplot(gs[k // ncols, k % ncols])
        s, c = int(prep.starts[i]), int(prep.counts[i])
        use = np.zeros(prep.px.size, bool); use[s:s + c] = True
        use &= prep.in_pitch & prep.in_frame
        g = _presence_grid(prep.px, prep.py, use, xe, ye, prep.fps, cover)
        # aspect='auto' -> mini panels FILL their cell (gs[1,:] bandindaki olu
        # bant kalkar). Tum mini'ler ayni sekilde gerildigi icin SEKIL-kiyasi
        # tutarli kalir; gercek esit-alan geometrisi Panel A'da korunur.
        ax.imshow(np.ma.masked_invalid(g), origin="lower",
                  extent=[xe[0], xe[-1], ye[0], ye[-1]], cmap=_MINI_CMAP,
                  aspect="auto", interpolation="nearest")
        for (p0, p1) in homo._template_segments_m():
            ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=PITCH_LINE,
                    lw=0.5, alpha=0.6)
        ax.set_xlim(xe[0] - 1, xe[-1] + 1); ax.set_ylim(ye[0] - 1, ye[-1] + 1)
        ax.set_xticks([]); ax.set_yticks([])
        rec = players.get(int(tid))  # KEYS ARE INTS
        far = (rec is not None and float(rec.get("far_third_frac", 0.0)) > 0.4)
        ax.set_title(f"t{tid}  n={c}{'  far!' if far else ''}",
                     fontsize=7, pad=2)
    for k in range(n, nrows * ncols):  # bos hucreleri gizle
        fig.add_subplot(gs[k // ncols, k % ncols]).axis("off")


# ===========================================================================
# PANEL D — durustluk/QC footer (metin TAMAMEN report dict'ten)
# ===========================================================================
def _footer_text(report) -> str:
    co = report["coordinate"]; ho = report["honesty"]; g = ho["calib_gate"]
    pq = report["parity_qc"]
    hm = report.get("heatmap") or {}
    unknown = hm.get("unknown_frac")
    med_px = (ho.get("per_zone") or {}).get("median_px")
    dims = co["pitch_dims_m"]
    unk_s = f"{unknown:.3f}" if isinstance(unknown, (int, float)) else str(unknown)
    medpx_s = f"{med_px:.2f}" if isinstance(med_px, (int, float)) else str(med_px)
    q = co.get("scale_quality", "relative")
    birim = (
        f"BIRIM: {co.get('unit_label', co['unit'])}"
        + (" (metre DEGIL)" if q == "relative" else "")
        + f" | saha L={dims['L']:.0f} W={dims['W']:.0f}"
        + (" PROVISIONAL" if q == "relative"
           else f" ASSERTED ±{int(co.get('band_pct') or 10)}%"))
    return "\n".join([
        birim,
        f"OLCEK: {ho['scale_uncertainty']}",
        f"KIMLIK: {ho['identity']}",
        f"IVME: {ho['acceleration']} | TAKIM: not assigned (non-goal) | "
        f"POSSESSION: {ho['possession']}",
        f"KAPSAMA: unknown_frac={unk_s} (gri=bilinmiyor, 0=gozlendi-bos) | "
        f"per-zone median_px={medpx_s}",
        f"PARITE 7v7=14: medyan {pq['median_count']:.0f}, "
        f"%{100*pq['frac_at_expected']:.1f} tam-14, tek %{100*pq['odd_frame_frac']:.1f}; "
        "eksik = uzak-uctaki kacan (yoklugu gercek sanma)",
        f"CALIB GATE: ok={g['ok']}",
    ])


# ===========================================================================
# BIRLESIK COK-PANEL RAPOR FIGURU  (ana cikti)
# ===========================================================================
def render_report_figure(tracks_path, calib_path, out_path, top_n=12,
                         bin_m=1.0):
    """GERCEK veriden tek-figur cok-panel durust rapor PNG'si uretir.

    load_tracks (dogru fps 24.87) -> calib gate (FAIL-CLOSED: reddedilirse
    RuntimeError, perspektif-yanli PNG yok) -> generate_topdown_report BIR KEZ
    (report_topdown.json yazar + tek durustluk kaynagi) -> prepare BIR KEZ
    (geometri) -> 4 panel. Doner: yazilan PNG yolu (str).
    """
    import stats_report as SR
    from pitch.homography import PitchHomography

    df = SR.load_tracks(tracks_path)
    meta = (getattr(df, "attrs", {}) or {}).get("meta", {}) or {}
    fps = float(meta.get("fps") or 25.0)

    # --- calib gate: fail-closed (rapor figuru icin sessiz pixel fallback YOK)
    try:
        homo = PitchHomography.load(calib_path)
    except Exception as e:
        raise RuntimeError(
            f"calib gate reddi [load failed: {e}]; perspektif-yanli figur "
            "uretilmez") from e
    ok, reasons = accept_calib_qa(homo._qa)
    if not ok:
        raise RuntimeError(
            f"calib gate reddi {reasons}; perspektif-yanli figur uretilmez")
    qa_med = round(float((homo._qa or {}).get("median_px", float("nan"))), 2)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # --- tek durustluk kaynagi + report_topdown.json (out_path dizinine)
    report = generate_topdown_report(tracks_path, calib_path,
                                     out_dir=str(out.parent))

    # --- geometri (tek prepare)
    prep = prepare(df, homo, fps)
    grid, edges = heatmap_grid_equal_area(prep, homo, prep.frame_wh, bin_m=bin_m)
    cover = coverage_mask_pitch(homo, prep.frame_wh, edges[0], edges[1])
    top = _select_top_tracklets(prep, top_n)

    # --- layout: GridSpec(3,2); mini-grid satir-1'in HER IKI kolonu (bos bant yok)
    fig = plt.figure(figsize=(16, 16))
    # birim etiketi: relative_m -> '(metre DEGIL)'; approx-m -> 'm (approx +-N%)'; m -> kesin
    _ulab = getattr(prep, "unit_label", None) or prep.unit
    _suffix = " (metre DEGIL)" if getattr(prep, "scale_quality", "relative") == "relative" else ""
    fig.suptitle(
        f"Top-down rapor — {Path(tracks_path).stem}  |  birim={_ulab}{_suffix}  "
        f"|  calib median={qa_med}px",
        fontsize=13, y=0.995)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.05, 1.15, 0.20],
                          hspace=0.18, wspace=0.12,
                          left=0.04, right=0.97, top=0.95, bottom=0.03)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    panel_heatmap(ax_a, grid, edges, homo, prep.pitch_dims, prep.unit,
                  cbar=True, fig=fig)
    panel_trajectories(ax_b, prep, homo, prep.pitch_dims, fig=fig)
    panel_tracklet_grid(fig, gs[1, :], prep, homo, prep.pitch_dims, top,
                        cover, edges, report=report, ncols=6)

    ax_d = fig.add_subplot(gs[2, :]); ax_d.axis("off")
    ax_d.text(0.005, 0.96, "DURUSTLUK / QC", fontsize=10, fontweight="bold",
              va="top", transform=ax_d.transAxes)
    ax_d.text(0.005, 0.74, _footer_text(report), fontsize=8.2, va="top",
              family="monospace", transform=ax_d.transAxes, color="0.15",
              linespacing=1.5)

    fig.savefig(out, dpi=130)
    plt.close(fig)
    return str(out)


# ===========================================================================
# optional debug-only pixel fallback (render_report_figure ASLA cagirmaz)
# ===========================================================================
def render_pixel_fallback(df, reasons, png_path):
    """SADECE debug: calib reddedildiginde piksel-uzayi sacilim. Bu rapor
    figuru DEGILDIR (perspektif-yanli) ve render_report_figure tarafindan ASLA
    cagrilmaz; manuel teshis icindir. Baslik bunu acikca soyler.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    fx = df["foot_x"].to_numpy(float); fy = df["foot_y"].to_numpy(float)
    ax.scatter(fx, fy, s=2, alpha=0.3, color="0.3")
    ax.invert_yaxis()
    ax.set_title("DEBUG pixel-space (calib REDDEDILDI; rapor DEGIL)\n"
                 + "; ".join(map(str, reasons)), fontsize=8)
    p = Path(png_path); p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=110); plt.close(fig)
    return str(p)


def _main(argv):
    tracks = argv[0] if argv else "raw/tracks_cankaya_cam2_clip2400.parquet"
    calib = argv[1] if len(argv) > 1 else "calib/cankaya_cam2_v2.json"
    out = argv[2] if len(argv) > 2 else "stats_out/report_figure_cankaya_cam2.png"
    p = render_report_figure(tracks, calib, out)
    print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
