# Halısaha Single-Frame Calibration — Full State Briefing (2026-07-04)

## What the product is
Turn ONE frame from a **fixed, corner-mounted, strongly fisheye amateur five-a-side (halısaha) camera** into a top-down 2D map of player positions, and from it: distance covered, speed, occupancy heatmaps, coarse team split. Target: `sosyalhalisaha` B2B analytics. Camera never moves → **one calibration serves the whole match**, so heavy/2-pass per-match compute is acceptable.

## Pipeline (repo: ~/halisaha-stats, code under calib_train/)
1. **Foot detection** — person detector → ground-contact point.
2. **Line-role segmentation** — a compact UNet (`seg2_hr2.pth`, BASE=32, 1024×576) segments painted markings into 7 semantic ROLES: 0 goalN(near goal) 1 goalF(far goal) 2 touchN(near touch) 3 touchF(far touch) 4 center 5 box 6 circle. Trained mostly on procedural synthetic pitches (`make_seg_data2.synth_sample`) + ~50 hand-labeled real venues. CLAHE-free inference.
3. **Joint calibration** — shared radial fisheye (k1≈0.167,k2≈0.240) + per-frame homography, nonlinear least squares, residual in METRIC space. `auto_clean2d.calibrate_frame`: keep-best chain baseline → multihyp → VP-recovery (rejects if inlier-consistency IC<0.50, deliberately tuned to reject fakes). Wired keep-best refinements: center-circle constraint (`center_refine`), robust-boundary goal-post outlier rejection (`robust_boundary`), per-venue box/circle marks (`measure_marks`).
4. **Projection** — feet → metric → 2D radar / bird's-eye warp.

## Measured state (fix-on, agent-judged over 121 venues, product/boundary bar)
- USABLE **72/121 (60%)**, MARGINAL 13, WRONG **36 (30%)**; usable+marginal 85 (70%).
- Raw solve-rate: **hr2 = 88/121 (73%)**.
- WRONG=36 breakdown: **33 never solve (kalibrasyon YOK)** + **3 solve-but-skewed** (idx0,35,59 — far edge).
  - Of the 33 no-solve: **13 are dark** (mean luminance <60/255, several near-black e.g. idx74=1.3 — information-limited, no seg recovers them) + **20 are BRIGHT** (luminance 62–142) yet fail.
  - Of the 20 bright no-solve: **15 have all 4 boundary channels active but the seg output is diffuse BLOBS / role-confusion on dim-field or structural clutter** (dome trusses/nets/ads mistaken for lines) → solver correctly rejects (VP IC 0.11–0.35). **5 have a genuinely weak role (gF or tN).**
- Fixes CONFIRMED working: only 3/108 solvable venues are solve-but-skewed → the geometry refinements do their job; **the bottleneck is SOLVE-RATE and warp SKEW, not the mid-field geometry fixes.**

## The TOP-RIGHT / FAR-CORNER SKEW (Alperen's main concern)
On many SOLVING venues the warp's top-right = the pitch FAR corner (X=L far-goal × Y=W far-touch) is skewed/sheared. Root: the far third is resolution-starved (few camera pixels), at max fisheye periphery, and the two far lines meet at a grazing angle → **geometric leverage**: a small far-line pixel error → meters of top-down error → whole warp shears (even good near corners fall into the void, e.g. idx0). **Better far-line DETECTION did NOT fix it** (idx12/34 warps unchanged before/after far-training). So the skew is a **GEOMETRY/solver problem, not a detection problem.**

## What we TRIED and the honest outcome
- **center-circle constraint / robust-boundary / per-venue marks** → WORK (mid-field geometry, goal-post rejection). KEEP.
- **dark-augmentation training** (aggressive gamma+extreme-dark+poisson noise, weighted-BCE) → FAILED. Val role-IoU rose (0.245→0.374) but solved 0/38 — degenerate blobs (touch channels saturate, goal channels dead). Lesson: recall-weighted loss + heavy aug buys pixel coverage, kills line crispness. val-IoU ≠ product metric.
- **pseudo-label fine-tune (hr3)** → plateau (self-referential: solver grading its own labels).
- **agent corner-recovery** → net regression (2 better/14 worse).
- **2 automatic quality proxies** (far-line goodness, Youden separation cut) → did NOT match expert verdicts.
- **inference CLAHE on dark** → no help (near-black frames have ~0 line signal).
- **clutter + far-weight training (seg2_hr_far)** → recovers a DIFFERENT 6 hard venues but LOSES 8 hr2 solved (86/121 alone, net −2 vs hr2). **Complementary, not dominant.**
- **ENSEMBLE hr2+far** (run both, keep whichever solves) → 94/121 (78%) solve-rate. **HONESTY CORRECTION: this is a NON-DEPLOYED upper bound** — `cov_ensemble.py` is imported by nothing in the product; nothing ships it. **DEPLOYED today = hr2-only: ~73% solve / 60% usable over 121.** The 78% is achievable-if-wired, not shipping. (`seg2_hr_far2` = 12000-step retrain in progress.)
- **METRIC HYGIENE:** do not merge the three different numbers into one headline — strict-bar (all 7 roles, ~2.5%), agent product-bar (72/121 USABLE = 60%, 85 usable+marginal = 70%), and raw solve-rate (hr2 73% / ensemble-upper-bound 78%) are DIFFERENT measurements. Deployed product bar = **60% usable, 73% solve, hr2-only.**

## Structural constraints & assets
- **Single oblique fixed camera** → far half physically resolution-starved. The structural remedy for the far half is a **2nd opposite-corner camera + fusion**, but there's a re-ID / ID-fragmentation blocker (~75 id fragments/player) and we have NO 2nd feed for the 121 venues.
- **Data**: 121 venues are SINGLE JPEGs (no per-venue video except Cankaya) → no temporal aggregation / brighter-frame selection possible for the set. ~50 hand-labeled real venues. Alperen does NOT want to hand-label all venues.
- **Absolute scale** ±13% single-frame (focal×size ambiguity); ±2–4% with a one-time on-site center-circle measurement.
- **Product gaps** beyond calibration: ball NOT tracked (needs own detector); team split = one-sided vest classifier (bib team clean, rest lumped); stats pipeline exists (player_state_continuous.parquet, distance/speed).
- **Resources**: tokens + GPU + RunPod FREE (RTX 3090 ~$0.22/hr, ~$0.30/run). Iron-guard pattern auto-terminates pods.
- **Evaluation**: agent judges are the only reliable quality signal (proxies failed); good at coarse usable/wrong, weak at fine geometry ranking. There is a LaTeX paper (docs/paper) documenting the two-bar methodology + honest negatives.

## The candidate levers (rank these)
1. **Solver-side far-corner/skew fix** — down-weight unreliable far lines, anchor on center-circle + reliable near corner, better/higher-order or per-venue fisheye, skew-detection + alt-solve. Targets the skew (Alperen's concern) + the 3 bad-geom + marginal warp quality.
2. **Multi-model ensemble** (hr2+far+far2, keep-best) — coverage (already 78%; more models = diminishing?).
3. **More/better seg training** — data realism, longer runs (diminishing? dark/clutter attacked already).
4. **2-camera fusion** — structural far-half fix (big; needs 2nd feed + solves re-ID).
5. **Product completeness** — ball detection, real 2-way team classifier, ship stats. Is 78% coverage "good enough" to pivot from calibration to product features?
6. **Absolute-scale anchoring** — on-site center-circle measurement workflow (±13%→±2-4%).
7. **Real-label expansion** — targeted (NOT all venues; Alperen refused blanket labeling).

## The question for this workflow
Given ALL of the above, what is the highest-leverage way to advance the product NOW? Rank the levers by (impact × tractability ÷ effort), be honest about diminishing returns and what we're possibly fooling ourselves about, and decide THE single most valuable concrete next action to execute.
