# 06 · UniDepth architecture analysis + improvement plan (incl. our method)

Grounded in the installed source (`unidepth/` in the remote baselines venv), not memory.

## Architecture (UniDepthV2)
- **Backbone:** DINOv2 ViT (`models/backbones/metadinov2`), cls + patch tokens.
- **Decoder** (`models/unidepthv2/decoder.py`): two heads off the shared features.
  - **CameraHead** (`decoder.py:48`): attention over cls tokens with `num_params=4`
    learnable latent tokens → `fill_intrinsics` → `[fx,fy,cx,cy]`
    (`fx=exp·0.7·diag`, `cx=sigmoid·W`). **Pinhole, 4-DoF, full stop.**
  - **DepthHead** (`decoder.py:117`): turns a camera into a per-pixel **ray** field,
    embeds rays (polar/azimuth → Fourier features), and *prompts* the depth features
    with cross-attention (`prompt_camera`). Predicts a **radius** (Euclidean range)
    along each ray; `pts_3d = rays · radius`. **The depth path is camera-agnostic —
    it only ever sees `rays`.**
- **Camera zoo** (`utils/camera.py`): `Pinhole`, `EUCM`(6 params, closed-form),
  `OPENCV`(rational radial in θ, Newton unproject, ≤10 iters), `Fisheye624`(KB-style,
  10 params, Newton), `MEI`(mirror ξ, Newton ≤20), `Spherical`. All differentiable
  with `project`/`unproject`/`get_rays`.
- **`infer(camera=...)`** (`unidepthv2.py:241`): if a camera is **given**, it is used
  (any model in the zoo); **if not, the CameraHead predicts a pinhole** and rays come
  from it.

## Diagnosis — *why it fails under strong distortion* (our experiments)
The depth decoder is **not** the bottleneck: it consumes whatever `rays` it is given.
The bottleneck is that **zero-shot self-calibration predicts only a pinhole**, so on a
distorted camera the predicted peripheral rays are wrong → depth degrades (our result:
AbsRel 0.09→0.28 on KITTI, catastrophic 0.10→0.78 on nuScenes at strong distortion).
The capability to *represent* distortion exists (EUCM/OPENCV/Fisheye624) but is never
*predicted*. **Fix the camera head and you fix the failure.**

(Confirming experiment, cheap, recommended: feed UniDepth the GT-undistorted image, or
a provided EUCM/OPENCV camera matching the distortion — depth should recover. That
isolates the head as the sole cause.)

## Suggestions, ranked by impact/effort

**S1 — Predict EUCM instead of pinhole (highest impact, lowest effort).** ★ core
Change `CameraHead.num_params 4→6` and extend `fill_intrinsics` to emit
`(α=sigmoid, β=softplus)` alongside `fx,fy,cx,cy`; swap `Pinhole(...)`→`EUCM(...)` in
the predicted-camera path. Why EUCM specifically: (a) **+2 params** only; (b)
**closed-form** differentiable unproject (no Newton — unlike OPENCV/Fisheye624 — so
it's stable and cheap inside a prediction head); (c) **continuously degrades to
pinhole** (α→0), so it cannot hurt pinhole performance. Everything downstream
(`get_rays`→ray embedding→depth) already supports it unchanged.
*Heuristic (stability):* init α≈0 (pinhole) and anneal an α/β regularizer from
strong→weak — a distortion **curriculum**. This mirrors our own A4→widened-bound
finding: start near-pinhole, relax the bound as evidence demands.

**S2 — Our method as a test-time plug-in: LiDAR few-shot self-calibration.** ★ ours
At deployment on a new camera with a handful of LiDAR frames, **freeze all of
UniDepth and optimize only the predicted camera params** (6 EUCM numbers, or a residual
on them) against our LiDAR 3-D + reprojection losses (`losses.loss_3d`, `loss_reproj`).
This is exactly our `adapt_latent`, retargeted onto UniDepth's CameraHead output. Our
experiments show ~20 frames recover distortion and depth — UniDepth's pinhole head
cannot, so this directly repairs the failure **without any retraining**. Pairs
perfectly with S1 (adapt the EUCM α,β few-shot).

**S3 — Let the dense ray field carry a distortion residual.** ★ ours-adjacent
UniDepth already predicts a dense `rays` field *and* parametric intrinsics, then
reconciles them. Loosen the coupling: predict a **smooth residual ray field** over the
parametric (EUCM) rays, TV-regularized, supervised by LiDAR reprojection when
available. This absorbs whatever the parametric model misses (tangential/decentering,
true >180° fisheye) — a non-parametric safety net. Our LiDAR anchor is the supervision.

**S4 — Distortion augmentation + camera 3-D loss in pretraining.** ★ ours (data)
Train the (EUCM) CameraHead on distortion it will face: use our `synth_distort.py`
(Brown–Conrady) and `synth_fisheye.py` (Kannala–Brandt) as **augmentation**, with a
LiDAR/GT-camera 3-D loss on the predicted camera. The pinhole prior comes from
pinhole-heavy training data; distortion augmentation removes it.
*Heuristic:* anneal max distortion strength upward over training (curriculum).

**S5 — Periphery-aware, uncertainty-weighted calibration.** ★ heuristic
Distortion is observable at the **periphery**, but driving LiDAR is **central** (our
central-LiDAR bias *understates* the failure). Importance-weight the calibration loss
by ray radius (peripheral returns count more), and add/weight an uncertainty head so
depth is downweighted where camera confidence is low.

**S6 — Model-family gate (only if >180° needed).** ★ optional
A tiny classifier over cls tokens picks `{EUCM, MEI, Fisheye624}`; EUCM handles the
vast majority, MEI/Fisheye624 for catadioptric / >180°. Lowest priority — EUCM alone
is the 90% solution.

## The combined proposal (the paper-shaped contribution)
**UniDepth-EUCM + LiDAR few-shot self-calibration:** swap the pinhole head for an EUCM
head (S1), trained with distortion augmentation (S4); at deployment, adapt the 6 camera
params on ~20 LiDAR frames (S2) with a residual ray-field safety net (S3). This keeps
UniDepth's strong zero-shot depth on normal cameras *and* closes the distortion gap we
demonstrated — a concrete, defensible improvement over the current pinhole-only
self-calibration, with our LiDAR-anchored adaptation as the enabling mechanism.

## PROTOTYPE VALIDATION (S1+S2 on UniDepth, real code; src/calib_depth/unidepth_probe.py)
Tested on synthetic KB-fisheye KITTI (UniDepth-V2-L). Since `infer(camera=...)` lets
UniDepth *consume* any camera and its zoo includes Fisheye624 (= our KB model), we did
NOT retrain — we (a) gave it the GT camera, and (b) fit the camera from LiDAR (S2).

| KB k1 | zero-shot (pinhole) | +GT camera (S1) | +LiDAR few-shot fit (S1+S2) |
|-------|---------------------|-----------------|------------------------------|
| 2.5   | 0.112 / 0.845       | 0.087 / 0.927   | **0.087 / 0.927**            |
| 4.0   | 0.158 / 0.695       | 0.106 / 0.866   | **0.106 / 0.866**            |
(AbsRel / d1, 15 held-out frames; calib on 20 disjoint frames.)

CONCLUSIONS (validated):
  - S1 confirmed: UniDepth's depth decoder IS distortion-capable — giving it a
    distortion camera recovers depth (0.158->0.106 at k1=4.0). The pinhole-only
    self-calibration head is the *sole* bottleneck.
  - S2 confirmed: our 20-frame LiDAR self-calibration recovers the distortion
    EXACTLY (k1=4.000, reproj 0.00 px) and MATCHES the GT-camera upper bound when fed
    to UniDepth (+33% AbsRel / +0.17 d1 over zero-shot at k1=4.0). No retraining.

HONEST NOTES / heuristics that were NECESSARY (scientific):
  - Fit k1 ONLY with base K fixed: central LiDAR under-constrains higher-order k2 AND
    the principal point; a free (k1,k2,cx,cy) fit reprojects centrally (3px) but
    corrupts PERIPHERAL rays -> UniDepth depth collapses (0.38). Low-order + fixed-base
    is the well-posed choice (Occam). For unknown base K, need peripheral constraints.
  - EUCM (the recommended head) could NOT fit the extreme synthetic k1=4 (its single
    alpha,beta family is for real moderate fisheye) -> used Fisheye624 (exact family).
    For realistic lens distortion EUCM is the right minimal head; for extreme/true
    fisheye, Fisheye624/MEI.
  - Residual gap to undistorted UniDepth (~0.076): the backbone still sees the warped
    RGB, so fixing only the camera/rays is partial; S4 (train with distortion aug)
    would close it. BUG fixed: infer() mutates the camera in place (resize/crop), so
    the provided camera must be rebuilt per frame.

## S1 prototype attempt: zero-shot EUCM distortion head (src/calib_depth/eucm_head.py)
Built a head that predicts EUCM distortion (alpha,beta) from a single image (base K
fixed), trained with Brown-Conrady augmentation (k1 in [-0.5,0]) and a ray-matching
loss vs the GT ray field (KITTI, frozen small CalibEncoder).

RESULT — honest negative: the head **collapses to a near-constant** (alpha≈0.48,
beta≈0.65) independent of the input distortion; ray error tracks |k1| exactly (0.1deg
at k1=-0.04 -> ~7deg at k1=-0.5), the signature of a constant prediction. Numerically
stabilised (mask EUCM FOV, clamp, grad-clip, skip non-finite) but the *information*
problem remains: a lightweight encoder cannot read per-image distortion from one image.

INTERPRETATION (valuable): this empirically explains WHY UniDepth predicts only a
pinhole — zero-shot per-image distortion estimation is hard. It reinforces that the
robust, validated fix is **S2 (LiDAR few-shot self-calibration)**, which recovers the
distortion exactly and lifts UniDepth to the oracle, rather than a learned zero-shot
head. A faithful S1 would need UniDepth's strong DINOv2 features (not a small conv) +
large-scale, multi-camera distortion augmentation, and remains uncertain — a clean
negative-result / future-work contribution.

## Intervention points (file:line)
- `decoder.py:60` `self.num_params = 4` → 6
- `decoder.py:84` `out_pinhole` (output_dim=1 per token) — add α,β tokens
- `decoder.py:85-99` `fill_intrinsics` — emit α=sigmoid, β=softplus
- `unidepthv2.py:~272/362` predicted-camera construction `Pinhole(...)` → `EUCM(...)`
- few-shot loop: reuse our `adapt_latent.py` losses on `out["intrinsics"]`/camera params
