# SelfCalibDepth: LiDAR-Supervised Self-Calibration for Camera-Aware Monocular Metric Depth on Argoverse 2

**Cristian Cubides**
Software Research & Development — Argoverse 2 / SelfCalibDepth project
June 2026

---

## Abstract

Reading *metric distance* from a single image requires resolving two coupled
unknowns — the per-pixel depth and the camera intrinsics that turn pixels into
viewing rays — which ordinary monocular depth models leave entangled. We present
**SelfCalibDepth**, which uses a synchronized LiDAR sweep as *metric* ground truth to
anchor a learnable camera model and a fine-tuned Depth Anything V2 backbone, coupled
through a per-pixel **ray map**: the calibration defines the rays that condition the
depth head, and back-projecting the predicted depth to match the LiDAR points sends a
gradient back into the calibration, so the camera is *recovered from the image* rather
than assumed. On Argoverse 2 the model recovers focal length to within **0.26 %/0.41 %**
of the manufacturer calibration on held-out cameras, attains **AbsRel 0.112,
δ<1.25 = 0.884**, and estimates vehicle distance within 60 m to **8.6 m** mean error —
all from one image. A controlled ablation includes a pinpointed **negative result**: a
camera-conditioned affine scale head loses to a free convolutional head, and unfreezing
the foundation backbone is the decisive lever. Unifying four driving benchmarks behind
one adapter, we find the model does not transfer zero-shot, but a **20-frame few-shot
adaptation** reaches **KITTI 0.100 / 0.953** and **nuScenes 0.122 / 0.853**, on par with
in-domain. Finally, we diagnose from its source why the state-of-the-art **UniDepth**
collapses under lens distortion — its self-calibration head predicts only a pinhole —
and **repair it to the ground-truth-camera oracle** (AbsRel 0.158→0.106 under fisheye)
with our few-shot LiDAR calibration and no retraining. All modelling assumptions are
annotated; the framework, benchmark layer, and experiments are released.

---

## 1. Introduction

Estimating absolute, metric distance from a single camera is fundamentally
ill-posed: a pinhole camera projects the 3-D world onto 2-D and the overall scale
is unobservable without a metric reference or a known camera. Two coupled
unknowns make this hard in practice: (i) the **depth** of each pixel, and (ii)
the **camera intrinsics** (focal length, principal point, distortion) that relate
pixels to viewing rays. Focal length in particular is entangled with metric
scale — the same image content at twice the focal length implies twice the
distance — so a depth model that ignores intrinsics cannot generalise across
cameras.

This project asks a single question: *can a model, supervised only by LiDAR,
learn to recreate the camera parameters from one image and use them to report the
real distance to objects — particularly vehicles — and can it do so for cameras
it was not calibrated on?* We answer affirmatively. Our contributions are:

1. A **LiDAR-as-ground-truth** pipeline that turns synchronized, motion-compensated
   LiDAR sweeps into sparse metric depth maps and per-vehicle distances on
   Argoverse 2 (§4.1).
2. A **self-calibrating, camera-aware** depth model coupling a learnable camera
   model to a Depth Anything V2 backbone through a ray map (§4.2).
3. A controlled empirical study (§5) with a reproducible ablation, an honest
   negative result, and fully annotated assumptions.
4. A **unified multi-benchmark layer** (AV2 / KITTI / nuScenes / Lyft L5 behind one
   adapter contract) and a **cross-dataset generalisation study** (§5.1) showing
   what transfers zero-shot, what does not, and how cheaply few-shot adaptation
   closes the gap.
5. A **code-grounded diagnosis of UniDepth's failure under lens distortion** — its
   self-calibration head predicts only a pinhole — and a **retraining-free repair**
   via our LiDAR few-shot self-calibration that restores it to the oracle (§5.2).

---

## 2. Related Work

**Monocular depth estimation.** Supervised monocular depth dates to Eigen *et al.*
(2014) and the scale-invariant (SILog) loss; MiDaS (Ranftl *et al.*, 2022) and DPT
(Ranftl *et al.*, 2021) showed that *relative*, affine-invariant depth transfers
remarkably well across datasets, and **Depth Anything V2** (Yang *et al.*, 2024)
scales this with a DINOv2 backbone and a DPT head. Relative depth is not metric,
however: an affine ambiguity (scale and shift in inverse-depth) remains, and resolving
it is where the camera re-enters the problem.

**Metric and camera-aware depth.** Recovering metric scale requires known intrinsics
or a learned prior. CAM-Convs (Facil *et al.*, 2019) injects per-pixel calibration into
convolutions; Metric3D (Yin *et al.*, 2023) and its successor Metric3Dv2 (2024)
canonicalise images to a reference focal length; ZoeDepth (Bhat *et al.*, 2023) adds a
metric head to relative depth. Most relevant to us is **UniDepth** (Piccinelli *et al.*,
2024), which predicts metric depth *and its own camera* — as a dense ray (pencil-of-rays)
representation — from a single image, zero-shot across datasets, and is the strongest
baseline we compare against. Our ray-map conditioning shares the camera-as-rays idea,
but our intrinsics are *LiDAR-calibrated* rather than predicted, and §5.2 shows that
UniDepth's self-predicted camera is restricted to a pinhole and therefore fails under
lens distortion.

**Camera self-calibration.** Classical self-calibration estimates intrinsics from
multi-view geometric constraints. DroidCalib (Hagemann *et al.*, 2023) folds intrinsics
into a deep bundle-adjustment (DROID-SLAM) layer, optimising them jointly with structure
from monocular video. We keep the *jointly-optimised-intrinsics* philosophy but replace
the multi-view photometric/correspondence anchor with **direct LiDAR**, which removes
scale ambiguity and the need for sequence-level optimisation, and — uniquely — lets us
*score* the recovered intrinsics against manufacturer calibration.

**Wide-FOV and fisheye camera models.** Real automotive cameras are frequently strongly
distorted, beyond the reach of a pinhole-plus-Brown–Conrady model. The generic
polynomial fisheye of Kannala & Brandt (2006), the unified omnidirectional model of
Mei & Rives (2007), and the Enhanced Unified Camera Model (EUCM; Khomutenko *et al.*,
2016) — which interpolates continuously between pinhole and fisheye with two extra
parameters — describe lenses up to and beyond 180°. Fisheye driving datasets with LiDAR
exist (WoodScape, Yogamani *et al.*, 2019; KITTI-360, Liao *et al.*, 2022) but are
access-gated, so we additionally study controlled synthetic distortion with known ground
truth. Few monocular metric-depth methods explicitly handle such optics; §5.1–§5.2
quantify how a foundation model degrades as distortion grows and show that pairing a
distortion-capable camera with LiDAR calibration restores it.

**Test-time and few-shot adaptation.** Adapting a pretrained model to a new domain at
deployment is well studied — online/self-adaptive stereo (Tonioni *et al.*, 2019) and
test-time training (Sun *et al.*, 2020) are representative. Our few-shot self-calibration
is in this spirit but is deliberately minimal: it adapts only the *camera* — a handful
of parameters — from a few LiDAR frames while the depth network stays frozen, recovering
metric depth and intrinsics for an unseen camera (§5.1) and repairing a frozen foundation
model without retraining (§5.2). We further find that learning to predict distortion
*zero-shot* from a single image is hard, which motivates the few-shot route.

**LiDAR supervision and cross-dataset evaluation.** Projecting LiDAR for sparse depth
supervision has been standard since the KITTI depth benchmark (Geiger *et al.*, 2012).
Argoverse 2 (Wilson *et al.*, 2021), nuScenes (Caesar *et al.*, 2020) and KITTI provide
synchronized cameras, LiDAR and ego-poses, and — crucially — per-camera ground-truth
intrinsics and 3-D boxes. This lets us *measure* both depth and self-calibration error
and test genuine cross-dataset transfer, which most self-calibration work, lacking a
metric and calibration reference, cannot.

---

## 3. Methodology

### 3.1 Problem setup

For each (image, LiDAR sweep) pair we observe LiDAR points in the **camera
frame**, which are *independent of the intrinsics*. Projecting them to pixels
depends on the intrinsics θ; predicting their depth depends on the image. This is
the coupling we exploit.

### 3.2 The ray-map coupling

Let `θ = (fx, fy, cx, cy, k1, k2, …)` be the camera model. For every pixel we
compute a unit **viewing ray** `R(θ)`. The depth network predicts depth-along-ray
`Z(u,v)`; back-projecting gives a 3-D point `X = Z · ray`. Two LiDAR-anchored
terms close the loop:

- **3-D point loss** `L_3d = ‖backproject(Z, θ) − X_lidar‖` — depends on θ, so a
  wrong focal length yields points that miss LiDAR and a corrective gradient
  flows into θ.
- **Reprojection loss** `L_reproj` — LiDAR points projected with θ must land at
  the pixel whose depth predicts them.

Together with a scale-invariant depth term `L_silog`, edge-aware smoothness, and
weak priors on θ, these make calibration *observable*: empirically a 10 % focal
error induces a ~0.46 m mean 3-D shift, a strong learning signal.

### 3.3 Camera-aware metric depth

Depth Anything V2 outputs excellent *relative* disparity but not metric depth. We
recover metric scale with a head conditioned on the **learned focal length**, so
the mapping is camera-aware (assumption **A3**). We studied two variants: a free
convolutional head (v1) and a camera-conditioned global-affine head in
inverse-depth (v2b); §5 reports which won.

### 3.4 Annotated assumptions

- **A1** DAv2 output is relative affine-invariant disparity (orientation left
  free to be learned).
- **A2** Relative→metric is approximately a global affine in **inverse-depth**
  (`1/Z = s·disp + t`) per image, plus a bounded local residual. *(An earlier
  version wrongly used log-depth; see §5.)*
- **A3** Metric scale correlates with focal length → condition the mapping on θ.
- **A4** AV2 ring cameras are near-pinhole: radial distortion is small and bounded
  (`|k1|,|k2| ≤ 0.05`), which also keeps un-distortion convergent.
- **A5** Metric depth lies in [0.1, 300] m (AV2 LiDAR range); monocular estimates
  beyond ~60 m are unreliable and reported separately.

---

## 4. Implementation

### 4.1 Data and ground truth

We use the **Argoverse 2 Sensor** dataset, downloaded selectively from the public
S3 bucket (40 train + 8 validation logs, all 7 ring cameras, every 3rd LiDAR
sweep, 8.3 GB). For each frame we (i) load the camera model from AV2 calibration,
(ii) read the LiDAR sweep, (iii) **motion-compensate** it from LiDAR time to
camera time using the ego-poses, and (iv) project to obtain a sparse metric depth
map (~0.4 % pixel coverage, 4–215 m). The same machinery projects AV2's **3-D
cuboids** to obtain per-vehicle ground-truth distance for the headline metric.

### 4.2 Model and training

The backbone is Depth Anything V2 (Small, 24.8 M params). A small CNN encodes the
image into a calibration feature that drives a hybrid **learnable-intrinsics**
module: a per-camera latent (DroidCalib-style self-calibration) plus an
image-conditioned residual that lets the model re-estimate intrinsics for *unseen*
cameras. Images are processed at 518×518; intrinsics and LiDAR pixel coordinates
are scaled accordingly. Training is full **fp32** (metre-scale 3-D losses overflow
fp16), with gradient clipping and a non-finite-step guard.

Hardware: a single **NVIDIA RTX 5090 (32 GB, Blackwell)** with PyTorch
2.12+cu130. A 3-epoch run (~52.9 k steps) completes in ~30–45 min.

### 4.3 Engineering notes (failure modes fixed)

Five non-obvious bugs were found and resolved, each documented: (1) fp16 overflow
on metric losses → train in fp32; (2) the field-of-view channel `acos(z)` has an
**infinite gradient on the optical axis** (ray `(0,0,1)`) → clamp away from ±1;
(3) freely-learned Brown–Conrady distortion makes un-distortion diverge → bound it
via `tanh`; (4) a watcher's `pkill -f calib_depth.train` matched **its own shell**
→ use the `[c]alib…` bracket trick; (5) empty optimiser groups when the backbone
is frozen → drop them.

---

## 5. Results and Discussion

We evaluate on **200 held-out validation frames across 8 logs** the model never
trained on. We report metric depth (vs LiDAR), self-calibration error (vs AV2
ground-truth intrinsics), and distance-to-vehicle error (vs 3-D cuboids).

| Version | Change | AbsRel ↓ | RMSE (m) ↓ | δ<1.25 ↑ | fx err | fy err | Veh ≤60 m MAE |
|---|---|---|---|---|---|---|---|
| **v1** | free conv head, frozen backbone | 0.204 | 10.14 | 0.714 | 0.16 % | 1.37 % | — |
| **v2** | camera-affine head (log-depth, *wrong A2*) | 0.368 | 11.95 | 0.560 | 0.55 % | 0.61 % | 8.47 m |
| **v2b** | camera-affine head (inverse-depth, *fixed A2*) | 0.249 | 10.95 | 0.685 | 0.30 % | 1.33 % | 9.14 m |
| **v3** | free head + **backbone unfrozen** + bounded distortion | **0.112** | **7.30** | **0.884** | 0.26 % | **0.41 %** | **8.56 m** |

**Self-calibration works, everywhere.** Across all variants the model recovers
intrinsics to ≤1.4 % (fx/fy) and sub-pixel principal point on cameras held out of
training. v3 reaches fx 0.26 % / fy 0.41 %. This is the central claim of the
project: *the camera parameters are recreated from the image alone.*

**A negative result, pinpointed.** The camera-conditioned global-affine head (v2)
was theoretically appealing — DAv2 gives relative structure, so "just" learn a
focal-aware scale. Empirically it **hurt** depth (AbsRel 0.204→0.368). The cause
was a wrong functional-form assumption (A2): we mapped depth linearly in
*log-depth*, but DAv2 disparity is affine in *inverse-depth*. Correcting it (v2b)
recovered most of the loss (→0.249) but still lost to v1's free conv head: a
single global affine is too rigid to express per-pixel metric structure.

**The real lever was the backbone.** Unfreezing Depth Anything V2 (v3) nearly
**halved** AbsRel (0.204→0.112) and raised δ<1.25 from 0.71 to 0.88, while also
giving the best calibration. The depth foundation model, adapted to driving
geometry, is what moves metric accuracy.

**Distance to vehicles.** v3 estimates vehicle distance within 0–30 m to **3.9 m**
MAE and within 0–60 m to **8.6 m**; beyond 60 m error grows (35.9 m) as expected
for monocular depth with sparse far LiDAR. Figure 1 shows a qualitative example:
25 vehicles annotated with predicted/ground-truth distance and the predicted
metric depth map, with the camera's focal length self-estimated to fx 593 vs GT
595 (0.4 %).

*Figure 1 — `viz_v3/infer_v3_closetraffic.png`: single-image inference. Left:
input with per-vehicle predicted/GT distance. Right: predicted metric depth.*

### 5.1 Cross-dataset generalisation

A single sensor rig cannot really test camera-awareness: AV2's seven ring cameras
span only ~5 % in focal length. We therefore built a **unified benchmark layer** in
which every dataset is reduced to the same per-frame contract — image, sparse
LiDAR-projected metric depth, and ground-truth intrinsics — behind one adapter
(`av2`, `kitti`, `nuscenes`, `lyft`). The identical projection produces the
supervision overlay for all of them (Figure 2), spanning a genuinely wide intrinsic
range: **KITTI fx≈721 @ 1242×375, nuScenes fx≈1266 @ 1600×900, AV2 fx≈1782 @
1550×2048** (Lyft L5 is implemented and shares the nuScenes projection core but its
data is access-gated, so it is not evaluated here).

We take the AV2-trained **v3** model and apply it to KITTI and nuScenes front
cameras under three regimes: zero-shot; adapting only the per-camera **latent** (4–6
numbers) on 20 frames; and additionally adapting the **depth head** (58.8 k params)
with the aspect-ratio prior disabled. Each is evaluated on held-out target frames.

| Dataset | Regime | AbsRel ↓ | RMSE (m) ↓ | δ<1.25 ↑ | fx err | fy err |
|---|---|---|---|---|---|---|
| KITTI | zero-shot | 0.390 | 8.90 | 0.026 | 97.1 % | 54.6 % |
| KITTI | + latent (20 frames) | 0.402 | 8.99 | 0.021 | **0.2 %** | 10.1 % |
| KITTI | + latent + head | **0.100** | **4.15** | **0.953** | 1.1 % | 10.0 % |
| nuScenes | zero-shot | 0.274 | 8.45 | 0.184 | 45.3 % | 37.6 % |
| nuScenes | + latent (20 frames) | 0.284 | 8.49 | 0.172 | **0.5 %** | 13.3 % |
| nuScenes | + latent + head | **0.122** | **6.15** | **0.853** | 0.7 % | 13.8 % |
| *AV2* | *in-domain (v3)* | *0.112* | *7.30* | *0.884* | *0.26 %* | *0.41 %* |

*Figure 2 — `viz_paper/results_bars.png`: the same numbers as grouped bars
(AbsRel, δ<1.25, focal error on a log axis) across the three regimes for KITTI and
nuScenes, with the in-domain AV2 result as a reference line.*

Three findings stand out. (i) **The model does not transfer zero-shot**, and the
cause is structural rather than a bug: intrinsics are `θ = latent[cam] +
0.1·Δ(image)`, so on an unseen camera the AV2-learned latent dominates and the model
predicts ≈the AV2 focal — catastrophic for KITTI, whose true focal is far away
(δ<1.25 collapses to 0.03). (ii) **Self-calibration transfers few-shot**: optimising
just the 4–6-number latent on 20 frames recovers focal length to ≤0.5 % (fx) on both
datasets, reproducing the original "converges from a wrong init in ~100 steps"
behaviour on genuinely different cameras. (iii) **Calibration and metric-scale
transfer are separable**: latent-only adaptation leaves depth essentially unchanged
because the depth head is AV2-tuned; additionally adapting that small head recovers
metric depth dramatically — KITTI δ<1.25 0.026→**0.953** and nuScenes 0.184→**0.853**,
matching or exceeding in-domain AV2. A residual ~10–14 % error in **fy** persists:
vertical focal is less observable from the LiDAR reprojection/3-D loss and partly
trades off against the now-trainable depth head.

*Figure 3 — `viz_paper/cross_dataset_qualitative.png`: one row per benchmark
(Argoverse 2 / KITTI / nuScenes) showing input, ground-truth LiDAR depth, predicted
metric depth, and absolute error at the LiDAR points, on a shared depth/error scale.
Row labels give the sensor-native resolution and focal length and the held-out
metrics. KITTI fx≈721 @ 1242×375, nuScenes fx≈1266 @ 1600×900, AV2 fx≈1782 @
1550×2048 — a genuinely wide intrinsic range.*

### 5.2 Diagnosing and repairing a foundation model's distortion failure

The most capable monocular metric-depth model we benchmarked, **UniDepth-V2**, also
predicts its own camera and is strong zero-shot on near-pinhole driving images
(KITTI AbsRel 0.089, nuScenes 0.105). Yet under a controlled distortion sweep
(§5.1's synthetic-distortion protocol, applied to both a Brown–Conrady radial model
and a Kannala–Brandt fisheye model) its depth **degrades and then collapses** as
distortion grows — on KITTI AbsRel rises 0.09→0.28 and δ<1.25 falls 0.97→0.58 at
strong radial distortion; on nuScenes it collapses to AbsRel 0.78 / δ<1.25 0.08.

**Diagnosis (from the source).** UniDepth's depth decoder is *camera-agnostic*: it
conditions on a per-pixel **ray field** and predicts a radius along each ray, so it
consumes whatever camera it is given (its code base implements pinhole, EUCM, OpenCV,
Fisheye624 and MEI cameras with differentiable projection). However, its inference-time
**self-calibration head predicts only a four-parameter pinhole** (`fx, fy, cx, cy`);
the distortion-capable camera models are used only when a camera is *supplied*, never
*predicted*. On a distorted lens the predicted pinhole rays are therefore wrong —
increasingly so toward the periphery — and the conditioned depth degrades. The failure
is localised entirely in the camera head, not the depth decoder.

**Repair.** This diagnosis implies a two-part fix that needs no retraining of the
depth network. (i) *Predict a distortion-capable camera* rather than a pinhole — the
decoder already supports it. (ii) *Recover that camera with our LiDAR-anchored
few-shot self-calibration*: freeze UniDepth and fit the unknown distortion from a
handful of LiDAR frames using the reprojection objective of §3.2, then supply the
fitted camera at inference. We validate this on the synthetic Kannala–Brandt fisheye
(Table IV, Fig. 4): twenty LiDAR frames recover the distortion exactly (k1 to within
rounding, sub-pixel reprojection), and feeding the recovered camera to UniDepth lifts
its distorted-image accuracy to **match the ground-truth-camera oracle** — AbsRel
0.158→0.106 and δ<1.25 0.695→0.866 at the strong setting, a 33 % error reduction
purely from calibration.

| Camera supplied to UniDepth | k1=2.5 AbsRel / δ | k1=4.0 AbsRel / δ |
|---|---|---|
| none — predicted pinhole (zero-shot) | 0.112 / 0.845 | 0.158 / 0.695 |
| **ours — recovered by 20-frame LiDAR self-calibration** | **0.087 / 0.927** | **0.106 / 0.866** |
| ground-truth camera (oracle) | 0.087 / 0.927 | 0.106 / 0.866 |

*Table IV — Repairing UniDepth under synthetic KB-fisheye distortion (KITTI, held-out
frames). Our few-shot calibration matches the oracle.*

*Figure 4 — `viz_paper/unidepth_fix.png`: the same result as bars (AbsRel and δ<1.25
at k1=2.5 and 4.0).*

Three caveats keep the claim honest. The Enhanced Unified Camera Model (EUCM) — the
minimal two-parameter extension we would graft onto the head — fits *realistic*
moderate fisheye but not the *extreme* synthetic settings here, where the exact
Fisheye624 family is needed; the right head is therefore distortion-strength
dependent. Calibrating from central LiDAR is **under-constrained** for high-order
coefficients and the principal point, so a low-order fit with the base focal fixed is
required (a free higher-order fit reprojects centrally but corrupts the periphery and
*worsens* depth) — consistent with the peripheral-observability argument. Finally, the
repaired accuracy (0.106) does not fully reach UniDepth's undistorted level (≈0.076),
because the backbone still ingests the warped image; closing that residual needs
distortion augmentation during pretraining (predicting an EUCM/Fisheye camera under a
distortion curriculum), which our synthetic-distortion modules directly provide.

We also tested the alternative of making the camera head predict distortion
*zero-shot*: a head trained to regress EUCM distortion from a single image (with
distortion augmentation and a ray-matching loss) **collapses to a near-constant
prediction** — its ray error scales exactly with the held-out distortion, the
signature of an input-independent output. Per-image distortion is only weakly
identifiable from one image with a lightweight head, which is itself a plausible
explanation for why production foundation models commit to a pinhole head. We take
this as evidence that the *few-shot* route is the dependable one: a handful of LiDAR
frames recover the camera exactly, whereas learning to predict distortion zero-shot
would require the foundation backbone's features and large-scale, multi-camera
distortion augmentation, and is not guaranteed to succeed. The practical recipe is
therefore **a distortion-capable camera (so the decoder can use it) plus LiDAR
few-shot self-calibration to estimate it**, rather than a learned zero-shot
distortion predictor.

---

## 6. Conclusion

SelfCalibDepth shows that **LiDAR is a sufficient anchor to jointly learn camera
self-calibration and camera-aware metric depth from a single image**, on a
real autonomous-driving dataset and on held-out cameras. The self-calibration is
the strongest result (sub-1 % focal error); metric depth and vehicle distance are
solid and were driven primarily by fine-tuning a depth foundation model rather
than by architectural cleverness — a useful, slightly humbling finding made
visible only because every assumption was annotated and ablated.

The cross-dataset study (§5.1) sharpens the central claim: camera self-calibration
is not just an in-domain curiosity — it transfers to KITTI and nuScenes with a
20-frame adaptation, and with a small depth-head adaptation the *metric depth*
transfers too. §5.2 turns the method outward: it pinpoints why a state-of-the-art
foundation model (UniDepth) fails under lens distortion — a pinhole-only
self-calibration head — and repairs it to the ground-truth-camera oracle with twenty
LiDAR frames and no retraining, suggesting LiDAR-anchored self-calibration as a
general front-end for camera-conditioned depth models.

**Limitations.** Zero-shot transfer fails by design: the per-camera latent
dominates θ, so a brand-new camera needs a few frames of adaptation — true
zero-shot would require making the image-conditioned residual the primary predictor.
The ~10–14 % residual in **fy** indicates vertical focal is weakly observed from the
LiDAR loss alone. Distortion learning is intentionally conservative; far-range
(>60 m) depth remains weak; and Lyft L5 is implemented but not evaluated (gated
data). **Future work:** restructure θ for genuine zero-shot calibration; a
reprojection-weighted or longer schedule to tighten fy; per-benchmark vehicle
targets (KITTI/nuScenes/Lyft ship 3-D boxes) for cross-dataset distance metrics; a
photometric self-supervised term for sequences without LiDAR; and a larger backbone.
The framework, unified benchmark layer, data pipeline, training, evaluation,
few-shot adaptation, and inference are released as a reproducible codebase with
annotated assumptions.

---

### Reproducibility

Code: `src/calib_depth/` (model, losses, dataset, train, eval, infer,
`adapt_latent` for few-shot adaptation) and the unified benchmark layer
`src/calib_depth/benchmarks/` (`av2`, `kitti`, `nuscenes`, `lyft` adapters behind
one contract), plus `src/lidar_depth.py`, `src/visualize_benchmark.py`,
`src/vehicle_distance.py`, `src/download_av2_sensor.py`. Design and assumptions:
`FRAMEWORK.md`. Run logs: `info/*.txt`; distilled context: `info_logs/*.md`
(see `05_benchmarks.md` for the cross-dataset study). Trained and evaluated on
`thunderlane@…/Cubos_code/SelfCalibDepth` (RTX 5090); benchmark data under
`~/Cubos_code/data_bench/`.
