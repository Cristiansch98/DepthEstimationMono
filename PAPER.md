# SelfCalibDepth: LiDAR-Supervised Self-Calibration for Camera-Aware Monocular Metric Depth on Argoverse 2

**Cristian Cubides**
Software Research & Development — Argoverse 2 / SelfCalibDepth project
June 2026

---

## Abstract

We present **SelfCalibDepth**, a framework that learns to read *metric distance*
from a single monocular image while simultaneously **recovering the camera's
intrinsic parameters** from that same image. The key idea is that a synchronized
LiDAR sweep, projected into the camera, provides dense-enough *metric* ground
truth to anchor both a depth network and a learnable camera model. The two are
coupled through a per-pixel **ray map**: the calibration defines the viewing
rays, the rays condition a fine-tuned Depth Anything V2 backbone, the network
predicts depth-along-ray, and back-projecting with the learned intrinsics and
comparing to LiDAR sends a gradient back into the calibration. On the Argoverse 2
Sensor dataset our best model recovers focal length to within **0.26 % (fx) /
0.41 % (fy)** of the manufacturer calibration on held-out cameras, attains
**AbsRel 0.112 / δ<1.25 = 0.884** metric depth, and estimates **distance to
vehicles within 0–60 m to a mean absolute error of 8.6 m** (3.9 m within 30 m) —
all from a single image. We document a controlled ablation (v1→v3), including a
**negative result**: a camera-conditioned global-affine scale head, although
theoretically attractive, underperformed a free convolutional head, and the
decisive improvement came from unfreezing the foundation backbone. Finally, to
test generalisation beyond a single sensor rig, we unify four driving benchmarks
(Argoverse 2, KITTI, nuScenes, Lyft L5) behind one adapter interface and show
that, although the AV2-trained model does **not** transfer zero-shot, a **20-frame
few-shot adaptation** of a tiny per-camera latent plus a 58.8 k-parameter depth
head recovers focal length to ~1 % and reaches **KITTI AbsRel 0.100 / δ<1.25 =
0.953** and **nuScenes 0.122 / 0.853** — on par with in-domain AV2. We annotate
every modelling assumption (A1–A5) so the results are reproducible and the
failure mode is attributable.

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

---

## 2. Literature Review

**Monocular depth estimation.** Supervised monocular depth dates to Eigen *et al.*
(2014) and the scale-invariant (SILog) loss; subsequent work (MiDaS, DPT) showed
that *relative* (affine-invariant) depth transfers remarkably well across
datasets. **Depth Anything V2** (2024) scales this with a DINOv2 backbone and a
DPT head, producing high-quality relative disparity. Relative depth, however, is
not metric: an affine ambiguity (scale and shift in inverse-depth) remains.

**Metric and camera-aware depth.** Recovering metric scale requires either known
intrinsics or a learned prior. **CamConvs** (Facil *et al.*, 2019) injects
per-pixel calibration into convolutions; **Metric3D** (2023) and **ZoeDepth**
(2023) recover metric scale by conditioning on, or canonicalising to, the camera
intrinsics. Our ray-map conditioning is in this lineage, but the intrinsics it
consumes are *learned*, not given.

**Camera self-calibration.** Classical self-calibration estimates intrinsics from
multi-view geometry. **DroidCalib** (2023) folds intrinsics into a deep
bundle-adjustment (DROID-SLAM) layer, optimising them jointly with structure from
monocular video. We adopt the same *jointly-optimised-intrinsics* philosophy but
replace the multi-view photometric/correspondence anchor with **direct LiDAR**,
which removes scale ambiguity and the need for sequence-level optimisation.

**Autonomous-driving depth + LiDAR.** Projecting LiDAR into the image for sparse
supervision is standard since the KITTI depth benchmark. **Argoverse 2**
(Wilson *et al.*, 2021) provides synchronized ring cameras, dual LiDAR, accurate
ego-poses, per-camera ground-truth intrinsics, and 3-D cuboid annotations —
making it uniquely suited to *measuring* both depth and self-calibration error,
which most self-calibration work cannot.

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
transfers too.

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
