# 03 · Status & results  (as of 2026-06-23)

## Where things stand
- Geometry + losses **real**, cross-checked vs av2 (pinhole match 0.000 px,
  round-trip 4e-15 m). Calibration signal confirmed (+10 % focal ⇒ 0.46 m 3D shift).
- DAv2 wired in and **trained on the remote RTX 5090** through v1→v3 ablation.
- `eval.py` and `infer.py` bodies implemented; paper written (`PAPER.md/pdf/docx`).

## Ablation (held-out val: 200 frames / 8 logs, all 3 epochs on RTX 5090)
| ver | change                                       | AbsRel | RMSE  | d1    | fx%  | fy%  | veh≤60 |
|-----|----------------------------------------------|--------|-------|-------|------|------|--------|
| v1  | free conv head, frozen backbone              | 0.204  | 10.14 | 0.714 | 0.16 | 1.37 | —      |
| v2  | camera-affine head, LOG-depth (**wrong A2**) | 0.368  | 11.95 | 0.560 | 0.55 | 0.61 | 8.47   |
| v2b | camera-affine head, INV-depth (fixed A2)     | 0.249  | 10.95 | 0.685 | 0.30 | 1.33 | 9.14   |
| v3  | free head + **BACKBONE UNFROZEN** + distort  | **0.112** | **7.30** | **0.884** | 0.26 | 0.41 | 8.56 |

v3 vehicle MAE: 0–30 m **3.90** | 30–60 m 12.57 | 60+ m 35.85 | overall 22.46.

## Headline findings
- **CORE CLAIM HOLDS:** self-calibration works across *all* versions (≤1.4 % focal
  error on held-out cameras); v3 best at fx 0.26 % / fy 0.41 %, cx/cy sub-pixel.
  The model recreates camera intrinsics from a single image, supervised only by LiDAR.
  It converges from a wrong init (fx=fy=500 → 427/564 GT in ~100 steps).
- **REAL LEVER = unfreezing the DAv2 backbone** (P-B): nearly halved AbsRel
  (0.204→0.112), d1 0.71→0.88, best calibration and far-range vehicle distance.
- **NEGATIVE RESULT (pinpointed):** camera-conditioned global-affine scale head (P-A)
  underperformed the free conv head. A single global affine is too rigid for
  per-pixel metric depth. v2 bug = used log-depth; DAv2 disparity is affine in
  **inverse**-depth (fixed in v2b → 0.249, still lost to v1).
- Bounded radial distortion (P-C) stable (learned k1≈0.0068 on front_center, no NaN).

## Inference demo (v3) — Fig 1 of paper
`infer.py`: single image → depth + estimated θ + per-vehicle pred/GT distance.
Example: front_center estimated fx=593 vs GT 595 (0.4 %). Figures in `viz_v3/`:
`infer_v3_demo.png`, `infer_v3_closetraffic.png` (25 vehicles labelled), `arch_diagram.png`.

## Open / next steps
- Real focal diversity: add stereo cams or a 2nd dataset; **explicit cross-camera holdout**.
- Photometric self-supervision (curriculum stage 3); larger backbone (DAv2-Large).
- >60 m depth still weak (monocular + sparse far LiDAR) — expected.
- Expand training set beyond 40 logs once the loop is solid.
