# Multi-camera player tracking for amateur football

Research code for turning two phone-camera feeds of an amateur 5-a-side match into
per-player statistics: where each player was, how far they ran, who had the ball.

This repository contains **code only**. No match footage, no frames and no player data are
included, because the recordings show identifiable people who did not agree to be published.

## What is in here

| Area | What it does |
|---|---|
| `calib/`, `calib_train/`, `pitch/` | Pitch calibration from line detection; image ↔ pitch homography |
| `detect/` | Player and ball detection, plus the evaluation harness |
| `fusion/` | Cross-camera association and re-identification |
| `stats/`, `analysis/` | Distance, speed and possession from stitched tracks |
| `eval/`, `tests/` | Ground-truth comparison, IDF1/MOTA scoring, regression tests |
| `learned/`, `self_training/`, `selfimprove/` | Learned calibration and self-labelling experiments |
| `docs/` | Design notes and the audit logs written while debugging |

## Where the accuracy actually came from

IDF1 reached **0.835** against hand-labelled ground truth. Almost none of that came from a
bigger model.

Two diagnoses did most of the work:

1. **The calibration was being fitted against a physically unsatisfiable constraint.** The solver
   dutifully returned a "best" answer for a geometry that could not exist, and every downstream
   metric inherited the error. Replacing it with a schematic 2D formulation fixed a whole class of
   failures at once.

2. **Part of my own ground truth was wrong.** Some of the tracking errors I was chasing were
   annotation errors. Fixing the reference changed what the numbers meant.

The same lesson shows up in `detect/`: an early version of the detector scored perfectly on an
offline test set that, on inspection, could not have failed it — no negative sample in the set
could even reach the firing threshold. A test that cannot fail is not measuring anything. The
evaluation harness here is written so that it can produce errors.

## Status

Active research code, not a product. Interfaces move, some experiments are dead ends and are kept
because the notes in `docs/` explain why.

## Author

Alperen Uğur Erden — physics undergraduate, Bilkent University
[sirschrodinger.com](https://sirschrodinger.com)
