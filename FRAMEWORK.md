# SelfCalibDepth — self-calibrating, camera-aware metric depth from LiDAR

A framework that learns **metric distance from a single camera image** by:

1. taking **LiDAR as ground-truth depth** (synchronized + motion-compensated),
2. **self-calibrating** the camera (`f, c, distortion`) DroidCalib-style, and
3. **fine-tuning Depth Anything V2** with the calibration fed in, so depth is
   *camera-aware* and generalizes across focal lengths / cameras.

Per the scoping decisions: full self-calibration · foundation-model backbone ·
camera-aware generalization.

## The core idea — everything couples through a ray map

```
            image  ──────────────▶ [ DAv2 encoder ] ──┐
                                                       ├─▶ [ DPT decoder ] ─▶ depth-along-ray  t(u,v)
  self-calib θ=(f,c,k) ─▶ [ ray map R(θ) ] ───────────┘                         │
       ▲                       (3×H×W bearings)                                  │ back-project with θ
       │                                                                         ▼
       └────────────── gradient ◀── 3D point loss vs LiDAR ◀──────────  X(u,v) = t · ray(u,v)
```

- The **calibration** `θ` turns pixels into per-pixel viewing **rays** `R(θ)`.
- `R(θ)` **conditions the decoder** (camera-aware) — same image under a different
  focal length yields different metric depth.
- The network predicts depth **along the ray**; back-projecting with `θ` gives a
  3D point cloud that is compared to the **metric LiDAR** points.
- A wrong focal length ⇒ back-projected points don't match LiDAR ⇒ gradient flows
  back into `θ`. LiDAR is the geometric anchor that replaces multi-view bundle
  adjustment. This is the DroidCalib insight, re-grounded on LiDAR.

## Ground truth (already working — `src/lidar_depth.py`)

Project a synchronized LiDAR sweep into a ring camera with ego-motion
compensation → a **sparse metric depth map** (~0.4 % pixel coverage, 4–200 m).
LiDAR points in the **camera frame** are intrinsics-independent, so they anchor
both the metric scale of depth *and* the calibration.

## Components (package `src/calib_depth/`)

| Module            | Role                                                                 | Status |
|-------------------|----------------------------------------------------------------------|--------|
| `camera_model.py` | Learnable pinhole+Brown–Conrady model; `θ → ray map`; back-project   | real geometry |
| `ray_map.py`      | Pixel grid → undistorted bearings; camera-aware conditioning tensor  | real geometry |
| `dataset.py`      | AV2 Sensor → (image, sparse LiDAR depth, LiDAR-in-cam, poses, GT θ)  | wraps `lidar_depth` |
| `model.py`        | Depth Anything V2 + ray-map conditioning + calibration head          | scaffold |
| `losses.py`       | SILog + 3D-point + reprojection + photometric + weak priors          | scaffold |
| `train.py`        | param-grouped optim, checkpointing, logging                          | scaffold |
| `eval.py`         | depth metrics (AbsRel, RMSE, δ<1.25) + calibration error vs AV2 GT   | scaffold |
| `infer.py`        | single image → metric depth + estimated θ + point-cloud export       | scaffold |

## Calibration model `θ`

Pinhole + radial/tangential distortion: `θ = (fx, fy, cx, cy, k1, k2, k3, p1, p2)`.
Two heads, summed (hybrid, enables generalization):

- **per-camera latent** `θ_cam` — a free parameter vector optimized over the
  dataset (the DroidCalib-style self-calibration);
- **image-conditioned residual** `Δθ(image)` — a tiny CNN head so the model can
  adapt to *unseen* cameras at inference from the image alone.

Weak priors only (full self-calibration): `cx,cy` near image centre, distortion
small, `f` in a broad plausible band — for optimization stability, not strong
supervision.

## Losses

| Term                | Definition                                                          | Drives |
|---------------------|---------------------------------------------------------------------|--------|
| `L_silog`           | scale-invariant log error of `t(u,v)` at LiDAR pixels               | depth  |
| `L_3d`              | ‖ back-project(t,θ) − LiDAR_cam ‖ at LiDAR pixels                    | depth + **θ** |
| `L_reproj`          | LiDAR_cam projected with θ should hit the pixel its depth predicts  | **θ**  |
| `L_photo`           | SSIM+L1 warp of adjacent frames via (t, θ, ego-pose)                | depth + θ (self-sup) |
| `L_smooth`          | edge-aware gradient smoothness                                      | depth  |
| `L_prior`           | weak Gaussian priors on `c`, distortion, `log f`                    | θ stability |

## Training

- Backbone: low LR (fine-tune); decoder + calibration: higher LR (separate groups).
- **Checkpoint** every N steps (weights + optimizer + `θ`), resume-safe.
- Log start/step/milestone/val metrics; AMP on the RTX 5090 (32 GB).
- Curriculum: (1) freeze θ at GT, fine-tune depth → (2) unfreeze θ latent →
  (3) enable image-conditioned `Δθ` + photometric self-supervision.

## Evaluation

- **Depth**: AbsRel, SqRel, RMSE, RMSE-log, δ<1.25^{1,2,3} vs LiDAR on AV2 val.
- **Calibration**: |f̂ − f_GT|/f_GT, principal-point px error, distortion error —
  AV2 ships per-camera GT intrinsics, so calibration is *measurable*.
- **Generalization**: train on a subset of the 7 ring cameras (varied FOV/focal),
  test on held-out cameras → does camera-aware conditioning transfer?

## Vehicle distance — the headline objective

The end goal is "real distance to vehicles from a single image." AV2 ships 3D
cuboid annotations (`annotations.feather`, per log) for every vehicle, so we get
**exact per-object metric distance** for free (`src/vehicle_distance.py`,
already working: projects cuboids → image, reads off range, e.g. 9–219 m).

Used two ways:
- **`L_vehicle` (optional object term)**: at vehicle cuboid pixels, supervise the
  predicted depth toward the cuboid-centre range — sharpens the metric scale
  exactly where it matters and is robust where LiDAR is sparse on car surfaces.
- **Headline metric**: *distance-to-vehicle error* — `|d̂ − d_GT|` and ALE per
  range bucket (0–30 / 30–60 / 60 m+), reported alongside the dense depth
  metrics. This is the number that answers the user's question directly.

At inference the per-vehicle distance is just the predicted metric depth sampled
at a detected/queried vehicle's pixels (back-projected with the estimated `θ`).

## v2 improvements (pinpoints) + assumptions

Each pinpoint is an isolated change with a rationale and the assumption it rests on.

- **P-A · Camera-conditioned scale mapping** (`ScaleMappingHead`). v1 regressed
  metric depth from normalized relative depth with a conv — weak. DAv2's *relative*
  structure is already excellent; only the metric **scale** is missing, and scale
  is focal-dependent. So we map relative→metric with a **global log-depth affine**
  whose two endpoints are predicted from the image feature **and `(log fx, log fy)`**
  (camera-aware), plus a bounded local residual.
  *Assumptions:* **A1** DAv2 output is relative affine-invariant disparity
  (orientation left free); **A2** relative→metric ≈ one global affine in log-depth
  per image + a ±0.2 log-space (~±22 %) local residual; **A3** metric scale
  correlates with focal length (hence the conditioning).
- **P-C · Bounded radial distortion** re-enabled → true *full* self-calibration.
  `k1,k2 = 0.05·tanh(raw)`. *Assumption:* **A4** AV2 ring cams are near-pinhole, so
  small bounded radial terms suffice and keep `undistort` convergent (tangential
  and k3 stay 0).
- **P-B · Optional backbone unfreeze** (`--unfreeze-backbone`, low LR). The single
  biggest remaining depth lever; off by default to isolate P-A/P-C.
- **P-D · Range-capped headline** for vehicle distance (≤60 m). *Assumption:* **A5**
  monocular metric depth past ~60 m is unreliable and LiDAR is sparse there, so the
  honest headline number caps range; full per-bucket numbers still reported.
- **Stability assumptions carried from v1:** fp32 (metre-scale 3D losses overflow
  fp16); `acos` clamped off ±1 (infinite gradient on the optical axis); depth
  clamped to [0.1, 300] m (A5, AV2 LiDAR range).

## Multi-benchmark generalization (`src/calib_depth/benchmarks/`)

The whole framework needs only one signal — image + sparse LiDAR-projected metric
depth + GT intrinsics — so that signal is made **pluggable**. A `BenchmarkAdapter`
turns any dataset into the same `Frame(image, uv, depth, intrinsics)`; everything
downstream (resize/scale, the LiDAR-in-cam anchor, `θ_gt`, model, losses, train,
eval) is benchmark-agnostic. Adapters ship for **Argoverse 2** (reference, wraps
`lidar_depth.sparse_depth_map`), **KITTI** (raw: `P_rect·R_rect·Tr_velo→cam`),
**nuScenes** and **Lyft L5** (shared `sensor→ego→global→ego→cam` motion-compensated
projection; SDKs imported lazily). One CLI renders the identical LiDAR-depth overlay
for all four (`src/visualize_benchmark.py --benchmark {av2,kitti,nuscenes,lyft}`),
and `train`/`eval` take `--benchmark`. This enables a true **cross-dataset holdout**
(train AV2 → test KITTI fx≈721 / nuScenes fx≈1266) — the strongest test of the
camera-aware self-calibration, which the single-rig AV2 setup couldn't provide.
See `info_logs/05_benchmarks.md`.

## Why Argoverse 2 fits

- 7 ring + 2 stereo cameras with **different intrinsics** on one platform →
  built-in focal-length diversity to test camera-aware generalization.
- Dense 2×VLP-32 LiDAR, accurate ego-poses, and **GT calibration** to score
  the self-calibration against.

## Roadmap / milestones

1. ✅ Ground-truth depth pipeline (LiDAR→camera, motion-compensated).
2. ⬜ `Dataset` + ray-map conditioning; sanity-check back-projection ≈ LiDAR.
3. ⬜ Fine-tune DAv2 with θ fixed at GT (metric-depth baseline).
4. ⬜ Unfreeze θ latent; verify f̂ converges toward f_GT from a wrong init.
5. ⬜ Add image-conditioned `Δθ` + photometric loss; test cross-camera transfer.
6. ⬜ Inference + point-cloud export; compare reconstructed cloud vs raw LiDAR.

GPU-heavy steps (3–6) run on the remote RTX 5090 under `~/Cubos_code/`.
