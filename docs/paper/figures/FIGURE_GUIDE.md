# How to make figures that look designed, not "default matplotlib"

The gap between a default plot and a Nature/Distill figure is a handful of
deliberate defaults — most are already baked into `paper_style.py`. This is the
reasoning + the checklist so you can extend it well.

## The 12 highest-impact rules (priority order)

1. **Size to final column width, use real pt fonts.** Don't draw big and shrink.
   Set `figsize` in inches at the printed size (single col ≈ 3.5 in / 89 mm,
   double ≈ 7.2 in / 183 mm). `paper_style` uses small fonts (8–10 pt) for this.
2. **Despine.** Remove top + right spines, axes background, heavy borders. *(done in `ps.use`)*
3. **Color with intent:**
   - categorical → **Okabe-Ito** (colorblind-safe). `ps.C` / `ps.CYCLE` are Okabe-Ito.
   - ordered/continuous → **perceptually-uniform** map (`ps.sequential(n)` = truncated viridis; or `viridis`/`cividis`/`batlow`).
   - **never jet/rainbow** — non-uniform lightness injects false structure and fails CVD.
4. **Embed fonts + vector output:** `pdf.fonttype=42`, `ps.fonttype=42`, save **PDF**. *(done; `ps.save` writes vector PDF + PNG preview)*
5. **Direct-label series** at the line end in the line's color (`ps.label_line`) — beats a legend lookup. Use a legend only when labels would collide.
6. **Gray the context, highlight the key series** in one saturated color. The result line should be the most prominent ink.
7. **One light grid** behind data, or none. *(done: subtle gray, `axisbelow`)*
8. **Data lines are the heaviest ink** (~1.5–2.3 pt); spines/grid thin (0.5–0.9 pt). *(done)*
9. **Confidence bands, not error bars,** for continuous x: `fill_between(x, lo, hi, alpha=0.12–0.2, lw=0)` in the line's hue.
10. **Few round-number ticks,** outward, short. Use `MaxNLocator` if cramped.
11. **Panel labels** `(a)(b)(c)` consistent; multi-panel shares axis scales for honest comparison.
12. **Redundant coding:** color *and* linestyle/marker/label, so the figure survives grayscale + colorblindness. `rasterized=True` on dense scatter to keep the PDF small.

## The "explainer" feel (3Blue1Brown / Distill)
- Annotate directly on the data (arrow + short callout at the interesting point).
- Restrained palette + lots of gray; color = emphasis, not decoration.
- Generous whitespace; don't fill the frame.
- Same color = same meaning across every panel and figure (one mental legend).

## Gotchas (matplotlib specifics)
- mathtext supports `\frac` only — **not** `\tfrac`/`\dfrac`; braces required (`\frac{1}{2}`).
- Non-`$...$` label text renders `\ ` etc. literally.
- **Verify scientific content**, don't just check it renders: assert a known invariant
  in the script (e.g. a turning point sits on the energy shell). See
  `~/pinney-ermakov-paper/figures/plot_phase.py` for the pattern.

Sources: Nature figure specs; Crameri et al. 2020 (*Nat. Commun.*, misuse of colour);
Okabe-Ito; Claus Wilke, *Fundamentals of Data Visualization*; Tufte (data-ink); Distill.
