# 02 · SelfCalibDepth architecture

## Core idea — everything couples through a ray map
```
   image ─────────────▶ [ DAv2 encoder ] ──┐
                                            ├─▶ [ DPT decoder ] ─▶ depth-along-ray t(u,v)
 θ=(f,c,k) ─▶ [ ray map R(θ) ] ────────────┘                         │ back-project with θ
     ▲                                                                ▼
     └────────── gradient ◀── 3D-point loss vs LiDAR ◀──── X(u,v) = t · ray(u,v)
```
- Calibration `θ` turns pixels into per-pixel viewing **rays** `R(θ)`.
- `R(θ)` **conditions the decoder** → camera-aware (same image, different focal ⇒ different metric depth).
- Network predicts depth **along the ray**; back-project with `θ` → 3D cloud compared to **metric LiDAR**.
- Wrong focal ⇒ points miss LiDAR ⇒ gradient flows into `θ`. LiDAR replaces multi-view
  bundle adjustment as the geometric anchor (the DroidCalib insight, re-grounded on LiDAR).

## Ground-truth depth (proven) — `src/lidar_depth.py`
Project a synchronized LiDAR sweep into a ring camera with ego-motion compensation
→ sparse metric depth map (~0.4 % pixel coverage, 4–200 m). Points in the **camera
frame** are intrinsics-independent → anchor both metric scale *and* calibration.

## Package `src/calib_depth/` (status)
| Module            | Role                                                            | Status        |
|-------------------|----------------------------------------------------------------|---------------|
| `camera_model.py` | Differentiable pinhole+Brown–Conrady; project/back_project/rays| REAL geometry |
| `ray_map.py`      | Pixel grid → bearings; camera-aware conditioning tensor        | REAL geometry |
| `dataset.py`      | AV2 Sensor → (image, uv, gt_depth, lidar_cam, poses, θ_GT)     | wraps lidar_depth |
| `model.py`        | DAv2 + ray-map conditioning + hybrid LearnableIntrinsics       | implemented (v1–v3) |
| `losses.py`       | SILog + 3D-point + reproj + photometric + smooth + weak priors | REAL          |
| `train.py`        | param-grouped optim, checkpointing, logging                    | runs on 5090  |
| `eval.py`         | depth metrics + calib error vs AV2 GT + vehicle-distance ALE   | implemented   |
| `infer.py`        | image → metric depth + estimated θ + per-vehicle distance      | implemented   |

## Calibration model θ
`θ = (fx, fy, cx, cy, k1, k2, k3, p1, p2)`, two heads summed (hybrid → generalization):
- **per-camera latent** `θ_cam` — free parameter vector (DroidCalib-style self-calib);
- **image-conditioned residual** `Δθ(image)` — tiny CNN head → adapt to unseen cameras.
- Weak Gaussian priors only (`c` near centre, distortion small, `log f` broad band).

## Losses
| Term       | Definition                                              | Drives             |
|------------|---------------------------------------------------------|--------------------|
| `L_silog`  | scale-invariant log error of `t(u,v)` at LiDAR pixels   | depth              |
| `L_3d`     | ‖back-project(t,θ) − LiDAR_cam‖ at LiDAR pixels         | depth + **θ**      |
| `L_reproj` | LiDAR_cam projected with θ hits its predicted pixel     | **θ**              |
| `L_photo`  | SSIM+L1 warp of adjacent frames via (t, θ, ego-pose)    | depth + θ self-sup |
| `L_smooth` | edge-aware gradient smoothness                          | depth              |
| `L_prior`  | weak Gaussian priors on `c`, distortion, `log f`        | θ stability        |
| `L_vehicle`| (optional) predicted depth → cuboid-centre range        | metric scale       |

## Key annotated assumptions (A1–A5)
- **A1** DAv2 output is relative affine-invariant disparity (**in inverse-depth**, not log — see v2 bug).
- **A2** relative→metric ≈ one global affine `1/Z = s·disp + t` + small local residual.
- **A3** metric scale correlates with focal length → condition the scale head on `log f`.
- **A4** AV2 ring cams are near-pinhole → only small bounded radial `k=0.05·tanh(raw)` (tangential/k3 = 0).
- **A5** monocular metric depth past ~60 m is unreliable + LiDAR sparse there → cap headline range ≤ 60 m.

## Stability rules (hard-won, keep them)
- Train in **fp32** (metre-scale 3D losses overflow fp16 → NaN); grad-clip; skip non-finite steps.
- Clamp `acos` input to `[-1+1e-4, 1-1e-4]` (FOV channel has ∞ gradient on the optical axis).
- Clamp depth to `[0.1, 300] m`. Drop empty AdamW param groups (frozen backbone).
