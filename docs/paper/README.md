# Default article template (two-column, ICML 2026 / PMLR style)

The layout from arXiv:2602.14486, set up as a reusable default.

## Layout
- Two-column body.
- **Abstract**: full page width, single column, at the top (it lives inside the
  `\twocolumn[...]` block in `main.tex`).
- **Figures / tables**: `figure` / `table` span one column; `figure*` / `table*`
  span both columns (full page width). See `sections/Intro.tex` for examples.

## Figures (publication-quality, not "default matplotlib")
`figures/paper_style.py` is a shared style: serif **Times-like** fonts (Nimbus Roman +
STIX math) that match the body, a muted palette, clean spines, subtle grid, and **vector
PDF** output. Use it in every plot script so figures don't look like default matplotlib:

```python
import paper_style as ps
ps.use()                       # apply the style
C = ps.C                       # palette: C["blue"], C["red"], C["teal"], ...
fig, ax = plt.subplots()
ps.style_axes(ax)              # drop top/right spines etc.
ax.plot(x, y, color=C["blue"])
ps.save(fig, "figures/myfig")  # writes myfig.pdf (vector, for LaTeX) + myfig.png
```

Palette is **Okabe-Ito** (colorblind-safe): `C["blue"|"red"|"teal"|"amber"|"purple"|"sky"]`
+ neutral `C["ink"|"slate"]`. For **ordered** series use `ps.sequential(n)`
(perceptually-uniform viridis), not the categorical cycle. Direct-label a curve with
`ps.label_line(ax, x, y, text, color)`. See **`figures/FIGURE_GUIDE.md`** for the full
12-rule checklist (data-ink, color, vector output, the explainer aesthetic) with sources.

`figures/plot_example.py` is a worked example -> `example.pdf`. Gotchas:
- mathtext in labels supports `\frac` only — **not** `\tfrac` / `\dfrac`, and `\frac`
  needs braces (`\frac{1}{2}`, not `\frac12`).
- Plain (non-`$...$`) label text must not contain `\ ` etc. — those render literally.
- First run on a new machine may need a font-cache rebuild; `ps.use()` does it
  automatically if the Times/STIX fonts aren't registered.

## Build
```
pdflatex main
bibtex main
pdflatex main
pdflatex main
```
(Or `latexmk -pdf main`.)

## New paper
Copy this whole folder, then edit `main.tex` (title, authors, abstract) and the
files in `sections/`. The `template/` and `figures/` folders travel with it.

## Style file options
In `main.tex`: `\usepackage[preprint]{template/icml2026}`
- `preprint` — **default for a standalone paper**: shows authors, NO conference footer.
- `accepted` — shows authors AND prints *"Proceedings of the 43rd ICML ... PMLR 306 ...
  Copyright"*. Use ONLY if the paper is genuinely accepted to ICML — otherwise it
  falsely claims acceptance (and is wrong in both the footer and the PDF metadata).
- (no option) — blind review (hides authors).

The line `\hypersetup{pdfsubject={Preprint}}` right after the package overrides the
style file's hard-coded `pdfsubject = "Proceedings of ... ICML 2026"`, which it sets
regardless of the option. Keep it for a standalone paper.

## Gotchas
- **Math in the abstract / title block**: the whole `\twocolumn[{ ... }]` argument is
  wrapped in `{ }` so a `]` inside (e.g. `\left[...\right]`) doesn't close the optional
  argument early. Without the braces, math in the abstract breaks the build.
- **Wide display equations**: a single column is ~3.25in, so long `=`-chains overflow
  silently (the class sets `\sloppy`, so they may not even warn). Break them:
  - single equation → `\begin{equation}\begin{aligned} ... \end{aligned}\end{equation}`
    (keeps ONE number);
  - `align` row → break with `\notag\\` so numbering is preserved.
  - Detect overflow reliably with `LC_ALL=C grep -a "Overfull \\hbox" main.log`.
