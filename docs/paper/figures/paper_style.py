"""Publication-quality matplotlib style for the paper figures.

Serif (STIX, Times-like) typography that matches the LaTeX body font, a refined
muted palette, clean spines, subtle grid, and vector-PDF output. Drop-in:

    import paper_style as ps
    ps.use()
    ...
    ps.save(fig, "/abs/path/myfig")   # writes myfig.pdf (vector) + myfig.png

Designed so figures stop looking like default matplotlib and read like a
typeset journal paper.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- palette: Okabe-Ito (colorblind-safe categorical, the field standard) ----
# Distinguishable under deuteranopia/protanopia/tritanopia. Keep the semantic
# names so plot scripts read naturally; ink/slate are neutral non-data colors.
C = {
    "ink":    "#1a1a2e",   # near-black for text / axes
    "blue":   "#0072B2",   # Okabe-Ito blue       (primary series)
    "red":    "#D55E00",   # Okabe-Ito vermillion (contrast / energy)
    "teal":   "#009E73",   # Okabe-Ito green
    "amber":  "#E69F00",   # Okabe-Ito orange
    "purple": "#CC79A7",   # Okabe-Ito reddish purple
    "sky":    "#56B4E9",   # Okabe-Ito sky blue
    "slate":  "#5c6b73",   # neutral gray for context / muted lines
}
CYCLE = [C["blue"], C["red"], C["teal"], C["amber"], C["purple"], C["sky"]]


def sequential(n, lo=0.12, hi=0.88, cmap="viridis"):
    """n colors from a perceptually-uniform map for ORDERED series (e.g. energy).
    Truncated to avoid the too-dark/too-bright ends. CVD-safe. Use this instead
    of the categorical CYCLE whenever the series have a natural ordering."""
    import numpy as np
    from matplotlib import colormaps
    cm = colormaps[cmap]
    return [cm(t) for t in np.linspace(lo, hi, n)]


def label_line(ax, x, y, text, color, dx=4, dy=0, **kw):
    """Direct label at the end of a curve, in the curve's color (beats a legend)."""
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                color=color, va="center", fontsize=kw.pop("fontsize", 9), **kw)


def _ensure_fonts():
    """Bundled STIX/Times fonts exist on disk but a stale cache can hide them."""
    import matplotlib.font_manager as fm
    names = {f.name for f in fm.fontManager.ttflist}
    if not ({"STIXGeneral", "Nimbus Roman", "Liberation Serif"} & names):
        try:
            fm._load_fontmanager(try_read_cache=False)
        except Exception:
            pass


def use():
    _ensure_fonts()
    mpl.rcParams.update({
        # typography: serif, Times-like, matches the paper body
        "font.family": "serif",
        # Nimbus Roman / Liberation Serif are Times clones -> match the paper body.
        "font.serif": ["Nimbus Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",   # Times-like math, bundled with matplotlib
        "axes.unicode_minus": True,
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        # color
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
        "text.color": C["ink"],
        "axes.labelcolor": C["ink"],
        "axes.edgecolor": C["ink"],
        "xtick.color": C["ink"],
        "ytick.color": C["ink"],
        # spines: drop top/right, thin the rest
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        # grid: subtle, behind data
        "axes.grid": True,
        "grid.color": "#b8bcc4",
        "grid.linewidth": 0.5,
        "grid.alpha": 0.35,
        "axes.axisbelow": True,
        # ticks
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        # lines / markers
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "lines.markeredgewidth": 0.8,
        "lines.solid_capstyle": "round",
        # legend
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": "#cfd3da",
        "legend.fancybox": True,
        "legend.borderpad": 0.6,
        # figure / export
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "figure.dpi": 120,
        "savefig.dpi": 300,
        # embed real TrueType fonts (Type-3 defaults get rejected by publishers)
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axes(ax):
    """Per-axes finishing touches that rcParams can't set globally."""
    ax.tick_params(which="both", top=False, right=False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["ink"])
    return ax


def save(fig, path_no_ext):
    """Vector PDF (primary, used by LaTeX) + PNG preview."""
    fig.savefig(path_no_ext + ".pdf")
    fig.savefig(path_no_ext + ".png", dpi=200)
    print("saved", path_no_ext + ".{pdf,png}")
