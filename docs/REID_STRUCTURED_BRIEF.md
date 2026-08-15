# Halısaha Structured Re-ID + Multi-Camera-Training — Design Brief (2026-07-05)

## Goal
Turn fixed single-camera amateur 7v7 football video into CLEAN per-player tracks (14 players = 7v7) with trustworthy per-player stats (distance, speed, heatmap, role). Then use two-camera data (same pitch) to TRAIN the single-camera to predict better. Owner (Alperen) explicitly wants: keeper-antagonism, defense-parallelism, positional roles (right-back vs left-back), 7v7 constraint — used as structural priors to disambiguate similar-jersey players that appearance alone cannot.

## What is SOLVED
- **Calibration**: single-frame human-clicker (`calib/make_clicker_html.py` → `solve_points_joint.py`) gives 2.6px calib; auto-line-seg calibration is unreliable (every geometric proxy Goodhart-games). So per-venue calibration is a solved onboarding step. Pitch coords are metric (X∈[0,L≈34], Y∈[0,W≈18]); near goal X=0, far goal X=L; near touch Y=0, far touch Y=W; camera at a near corner.

## What is BROKEN (the target)
Per-player re-ID/tracking on the full match. Rigorously measured on Cankaya (raw/cankaya_cam2.mp4, ~720s @ 24.86fps, known roster = 14 players / 7v7):
- `stats_out/cankaya_cam2_fullgame/state14.parquet` = 14 "players" but **9/14 are Frankenstein** (jersey-HSV inconsistent → multiple real players merged). Verified by appearance.
- **Keeper-swap** (owner spotted): track P11 jumps far-goal(x=28, white jersey H89/S17) → near-goal(x=0, yellow vest H36/S185) = definitely different players wrongly stitched. No single track behaves like a keeper (max goal-proximity only 41%).
- **Distance ~60% interpolated**: state14 is continuous only because gaps were filled; observed-only distance ≈419m/player, interpolation-included ≈1053m; truth ~500-700m. Far-goal region observation only 30-45% (rest guessed).
- **Appearance re-ID has a WALL**: greedy joint (appearance+motion+time) → 26 clusters (target 14), over-split. Most players wear LOW-saturation whitish jerseys (H60-145, S8-73) → appearance CANNOT separate them. Only vivid-colored (yellow vest S>150, orange S>100) separate cleanly. One team is uniform-ish, the other MIXED colors.
- Team-split by color alone: 4v10 (should be 7v7) — imperfect.

## AVAILABLE DATA / TOOLS
- `tracks14.parquet`: per-detection observed foot positions (foot_x,foot_y, box_h,box_w, conf, pitch_x, pitch_y, in_pitch, frame, t_sec) + a (wrong) player_id + tid. ~135k observed detections.
- `state14.parquet`: 14 continuous (interpolated) trajectories (player_id, frame, x, y, status, conf).
- `appearance_segs.json`: 401 appearance-consistent segments (pid, f0, f1, Hmed, Smed, n) from splitting the Frankenstein tracks at HSV change-points.
- `tid_appearance.json`: per-track jersey HSV signatures.
- Detector: RF-DETR (models/weights/checkpoint_best_regular.pth) + ByteTrack (motion-only, fragments at occlusions). GPU: local 4GB (RF-DETR fits ~3fps) or RunPod (RTX 3090 ~$0.22/hr). Full-match re-detect ~7.5h local / ~1h RunPod.
- Aggregate occupancy HEATMAPS already work and are re-ID-independent (shippable now).

## OWNER'S STRUCTURAL-PRIOR INSIGHTS (the core idea to develop)
1. **Keeper antagonism**: exactly 2 keepers, at OPPOSITE goals, stay near their goal, low movement. Strong disambiguator + fixes the keeper-swap.
2. **Positional roles / defense parallelism**: players have stable roles — a right-back stays right, left-back stays left; defenders move in a parallel line. A player's POSITIONAL signature (mean side, depth, spread) distinguishes same-jersey players that appearance can't.
3. **Formation**: at any moment a team's 7 players occupy a spread formation; assign fragments to formation slots/roles.
4. **7v7 hard constraint**: exactly 7 per team, 14 total. Use as a global constraint.

## TWO-CAMERA-FOR-TRAINING INSIGHT (owner)
Two cameras on the SAME pitch (opposite corners) give ground-truth positions (triangulate a player seen in both). Use this GT to TRAIN the single-camera model to predict correct positions even in the far/occluded region — "more data → single-camera predicts better." This is a TRAINING-DATA strategy, NOT inference fusion. Caveats (owner): "two-camera venues" are sometimes 1-camera-on-2-different-pitches (not 2-on-1); and 2 cameras aren't time-synced (won't start same second) → sync must be solved.

## THE DESIGN QUESTION
Design the concrete approach to produce CLEAN 7v7 per-player tracks on single-camera by combining: appearance (where separable) + motion continuity + time-exclusivity + STRUCTURAL PRIORS (keeper-antagonism, positional roles, formation, 7v7). Specify the exact algorithm (global data-association formulation, cost terms, constraints, solver) and the validation (roster=14, 2 keepers at opposite goals, formation plausibility, realistic distance). Separately, design the 2-camera-for-training data pipeline (sync solution, GT generation, what single-camera signal it improves, architecture). Be honest about the residual single-camera ceiling even WITH structural priors (unavoidable crossings/occlusions of same-role same-jersey players).
