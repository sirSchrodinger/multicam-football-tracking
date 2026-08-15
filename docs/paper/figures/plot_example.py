"""Example: a publication-quality figure using the shared paper style.

Run:  python3 plot_example.py   ->  writes example.pdf (vector) + example.png
LaTeX picks up example.pdf via \\includegraphics in sections/Intro.tex.

The point of paper_style: serif (Times-like) fonts that match the body, a muted
palette, clean spines, subtle grid, vector output -- so figures don't look like
default matplotlib.
"""
import numpy as np
import matplotlib.pyplot as plt
import paper_style as ps

ps.use()
C = ps.C

fig, ax = plt.subplots(figsize=(5.2, 3.4))
ps.style_axes(ax)

x = np.linspace(0, 2 * np.pi, 400)
ax.fill_between(x, np.sin(x), 0, where=(np.sin(x) >= 0),
                color=C["blue"], alpha=0.10, lw=0)
ax.plot(x, np.sin(x), color=C["blue"], lw=2.3, label=r"$\sin x$")
ax.plot(x, 0.6 * np.cos(x), color=C["red"], lw=2.0, ls=(0, (6, 3)),
        label=r"$\frac{3}{5}\cos x$")  # NB: mathtext supports \frac, NOT \tfrac/\dfrac
ax.axhline(0, color=C["ink"], lw=0.6)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$f(x)$")
ax.set_title("Example publication-quality figure")
ax.legend(loc="upper right")
ax.set_xlim(0, 2 * np.pi)

fig.tight_layout()
ps.save(fig, "example")   # -> example.pdf + example.png in this folder
plt.close(fig)
