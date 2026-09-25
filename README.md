# Player tracking for amateur football

Research code for turning match video from fixed venue cameras at an amateur football pitch
into per-player tracks and statistics: where each player was, how far they ran, who had the ball.

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

IDF1 reached **0.835** on a 2-minute window with 18 identities, against ground truth built by
LLM vision agents reviewing tracklet cards. The score is optimistic, and over a full 32-minute
match the method still fragments into 698 identities. Almost none of the 0.835 came from a
bigger model.

Two diagnoses did most of the work:

1. **The calibration was being fitted against a physically unsatisfiable constraint.** The solver
   dutifully returned a "best" answer for a geometry that could not exist, and every downstream
   metric inherited the error. Replacing it with a schematic 2D formulation fixed a whole class of
   failures at once.

2. **Part of the ground truth was wrong.** Some of the tracking errors I was chasing were
   annotation errors. Fixing the reference changed what the numbers meant.

The same lesson shows up in `detect/`: an early version of the detector scored perfectly on an
offline test set that, on inspection, could not have failed it: no negative sample in the set
could even reach the firing threshold. A test that cannot fail is not measuring anything. The
evaluation harness here is written so that it can produce errors.

## Status

Research code, paused since August 2026. Some experiments are dead ends and are kept because the
notes in `docs/` explain why.

## Author

Alperen Uğur Erden, physics undergraduate, Bilkent University
[cv.sirschrodinger.com](https://cv.sirschrodinger.com)
