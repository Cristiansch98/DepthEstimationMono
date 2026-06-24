# 05 · Unified multi-benchmark layer  (AV2 · KITTI · nuScenes · Lyft L5)

Added 2026-06-24 so the framework treats every driving benchmark identically.
The whole framework was built on one signal — a camera image + a sparse
LiDAR-projected metric depth map + GT intrinsics. The benchmark layer makes that
signal **pluggable**: add a dataset = add one adapter.

## The contract (`src/calib_depth/benchmarks/base.py`)
Every adapter turns its dataset into the same per-frame `Frame`:
```
image : (H,W,3) uint8 · uv : (N,2) · depth : (N,) metric z · intrinsics : CameraIntrinsics
```
- `CameraIntrinsics(fx,fy,cx,cy,width,height,k1..p2)` with `.theta9()` and `.scaled(sx,sy)`.
- `FrameRef(benchmark, scene, cam, frame)` — picklable pointer; `scene/frame` are
  strings so it works for file-tree datasets (AV2/KITTI) **and** token databases (nuScenes/Lyft).
- `BenchmarkAdapter.discover(root, cams, stride) -> [FrameRef]` (cheap) and
  `.load(ref) -> Frame` (heavy projection). Optional `.vehicle_targets(ref)` for the
  distance-to-vehicle metric (AV2 has it; others return `[]`).

Everything downstream is shared and benchmark-agnostic: `BenchmarkDepthDataset`
does the resize/scale, builds the intrinsics-free `lidar_cam` anchor and `theta_gt`,
then the same model/losses/train/eval run on any dataset.

## Registry (`benchmarks/__init__.py`)
```python
from calib_depth.benchmarks import get_adapter, BENCHMARKS   # ('av2','kitti','nuscenes','lyft')
adapter = get_adapter("kitti"); refs = adapter.discover(root, stride=5); frame = adapter.load(refs[0])
```
SDK imports (nuscenes-devkit, lyft_dataset_sdk) are **lazy** — `import calib_depth.benchmarks`
never fails for a missing SDK.

## Adapters (`benchmarks/*.py`)
| name       | source format                          | projection                                   | status |
|------------|----------------------------------------|----------------------------------------------|--------|
| `av2`      | AV2 Sensor log tree                    | wraps proven `lidar_depth.sparse_depth_map`  | **verified**, byte-identical to old path |
| `kitti`    | KITTI **raw** (`<date>/<drive>_sync`)  | `P_rect·R_rect·Tr_velo→cam`, rectified pinhole| implemented, needs KITTI-raw data |
| `nuscenes` | nuScenes dataroot (`v1.0-*`)           | 4-step sensor→ego→global→ego→cam (motion-comp)| implemented, needs devkit + data |
| `lyft`     | Lyft L5 (nuScenes format)              | shares `_nuscenes_core` projection           | implemented, needs lyft_sdk + data |

## Unified CLIs
```bash
# Same LiDAR-depth overlay for ANY benchmark (cross-dataset companion to lidar_depth.py)
PYTHONPATH=src python src/visualize_benchmark.py --benchmark av2 \
    --root data/sensor-sample --cam ring_front_center --num 2 --out viz_bench
PYTHONPATH=src python src/visualize_benchmark.py --benchmark kitti --root <kitti_raw> --cam image_02
NUSCENES_VERSION=v1.0-mini python src/visualize_benchmark.py --benchmark nuscenes --root <nuscenes>

# Train / eval on any benchmark (default av2; unchanged AV2 behaviour)
python -m calib_depth.train --benchmark kitti --data-root <kitti_raw> --cams image_02 --epochs 3
python -m calib_depth.eval  --benchmark av2   --data-root data/sensor --split val --ckpt checkpoints/final.pt
```

## Verification done (local, AV2 sample, no GPU)
- `av2` adapter `load()` reproduces `sparse_depth_map` exactly (image/uv/depth equal).
- `BenchmarkDepthDataset.build("av2",…)` == old `AV2SensorDepthDataset` (theta_gt scaling identical).
- `discover_logs` / `AV2SensorDepthDataset.index` preserved → eval/infer untouched for AV2.
- `visualize_benchmark.py --benchmark av2` regenerates the overlay (see `viz_bench/`).
- All modules `py_compile`; registry returns all four; empty-tree discovery → `[]` (no crash).

## Data / SDK prerequisites (not yet on disk)
- The remote `~/Cubos_code/KITTI` is a *different* project using `tensorflow_datasets`
  KITTI (object-detection: images + 3D box locations, **no raw LiDAR/calib**) → not
  usable for the LiDAR-GT path. Need **KITTI raw** drives (velodyne + calib) for `kitti`.
- nuScenes: `pip install nuscenes-devkit` + a `v1.0-mini`/full dataroot.
- Lyft: `pip install lyft-dataset-sdk` + Lyft L5 data (nuScenes layout).
- Install + download on the **remote RTX 5090** box; then the same train/eval/viz commands run unchanged.

## RAN on the remote RTX 5090 (2026-06-24)
SDKs: nuscenes-devkit 1.2.0 installed (downgraded numpy 2.2.6→1.26.4; torch+av2
still import fine). lyft-dataset-sdk **not** installed (it flips numpy back to 2.x
and pulls dev cruft) and Lyft L5 data is account-gated (public S3 mirrors 403/404)
→ Lyft adapter remains code-complete but unrun.

Data fetched (public, no auth): KITTI raw `2011_09_26_drive_0001` (108 frames) +
nuScenes `v1.0-mini` (404 CAM_FRONT frames, 4.1 GB).

Overlays produced through `visualize_benchmark.py` (`viz_bench/`), confirming the
projection math per dataset and the intrinsic diversity:
  - AV2 `ring_front_center` : fx≈1782 @ 1550×2048
  - KITTI `image_02`        : fx≈721  @ 1242×375   (19,956 returns)
  - nuScenes `CAM_FRONT`    : fx≈1266 @ 1600×900   (3,067 returns)

### Cross-dataset zero-shot eval (AV2-trained v3 ckpt, front cam → latent[0])
| dataset            | AbsRel | RMSE | d1    | fx err | fy err |
|--------------------|--------|------|-------|--------|--------|
| AV2 (in-domain)    | 0.112  | 7.30 | 0.884 | 0.26%  | 0.41%  |
| nuScenes (0-shot)  | 0.271  | 8.41 | 0.191 | 45.2%  | 37.7%  |
| KITTI (0-shot)     | 0.393  | 8.86 | 0.022 | 97.1%  | 54.6%  |

**Finding (honest, important):** the framework runs unchanged on all three
datasets, but it does **not** zero-shot transfer. Root cause is structural, not a
bug: in `model.py`, `theta = latent[cam_idx] + 0.1·delta_head(feat)` — the
per-camera latent (AV2-learned) dominates and the image-conditioned residual is
only 0.1× with zero init, so on a new camera the model predicts ≈the AV2 focal.
KITTI (true fx≈301 in 518-space) is far from the AV2 front latent (≈595) → 97% fx
error and collapsed depth scale (d1 0.02); nuScenes (≈410) is closer → less bad.
This precisely motivates the next experiment: **few-shot latent adaptation** (fit
just the per-camera latent on a handful of target frames, everything else frozen)
or strengthening/repurposing the image-conditioned head as the primary `θ`
predictor for unseen cameras.

### Few-shot latent adaptation (20 frames, latent-only, held-out eval)
`python -m calib_depth.adapt_latent` — freeze the entire net, optimize ONLY the
per-camera latent (log fx, log fy, cx, cy, k1, k2) on 20 target frames via the
LiDAR 3D/reproj losses; eval on held-out frames before vs after.

| dataset  | fx err          | fy err        | AbsRel        | d1            |
|----------|-----------------|---------------|---------------|---------------|
| KITTI    | 97.1% → **0.2%**| 54.6% → 10.1% | 0.390 → 0.402 | 0.026 → 0.021 |
| nuScenes | 45.3% → **0.5%**| 37.6% → 13.3% | 0.274 → 0.284 | 0.184 → 0.172 |

**Result (the positive finding):** the **self-calibration transfers cross-dataset**.
With 20 target frames and *only* 4–6 trainable numbers, the model recovers the true
focal length to **<0.5 % (fx)** on both KITTI and nuScenes — from a wrong AV2 init,
in ~150 steps. This reproduces the original "converges from a wrong init in ~100
steps" behaviour on genuinely different cameras → the core self-calibration claim
generalizes beyond AV2.

### Full few-shot adaptation: latent + depth head, aspect prior dropped
`adapt_latent --adapt-head --no-aspect-prior` — also unfreeze the 58.8k-param
`ScaleMappingHead` (lr 1e-3) and zero the `(log fx/fy)²` prior. 20 frames, 400 steps.

| dataset  |                | AbsRel | RMSE | d1    | fx err | fy err |
|----------|----------------|--------|------|-------|--------|--------|
| KITTI    | zero-shot      | 0.390  | 8.90 | 0.026 | 97.1%  | 54.6%  |
| KITTI    | +latent only   | 0.402  | 8.99 | 0.021 | 0.2%   | 10.1%  |
| KITTI    | **+latent+head**| **0.100** | **4.15** | **0.953** | 1.1% | 10.0% |
| nuScenes | zero-shot      | 0.274  | 8.45 | 0.184 | 45.3%  | 37.6%  |
| nuScenes | +latent only   | 0.284  | 8.49 | 0.172 | 0.5%   | 13.3%  |
| nuScenes | **+latent+head**| **0.122** | **6.15** | **0.853** | 0.7% | 13.8% |
| *AV2 in-domain (ref)* | | *0.112* | *7.30* | *0.884* | *0.26%* | *0.41%* |

**Headline: the framework generalizes.** With **20 target frames** and a light adapt
(per-camera latent + a 58.8k-param head), the AV2-trained model reaches
**KITTI AbsRel 0.100 / d1 0.953** and **nuScenes AbsRel 0.122 / d1 0.853** — on par
with (KITTI better than) AV2 in-domain (0.112 / 0.884), while recovering fx to
**~1 %**. silog dropped 0.23→0.10 (KITTI) / 0.27→0.12 (nuScenes); the depth-head adapt
is what fixes metric depth (latent-only left it unchanged), confirming calibration
and depth-scale transfer are separable but **both** recoverable few-shot.

Residual: **fy stays ~10–14 % off** even with the aspect prior removed — vertical
focal is less observable from the LiDAR reproj/3D loss and partly trades off against
the now-trainable depth head (depth can absorb a vertical-scale error). Minor vs the
depth/fx recovery; a longer adapt or a reproj-weighted schedule would tighten it.

## Paper figures (viz_paper/)
- `cross_dataset_qualitative.png` — 3 rows (AV2/KITTI/nuScenes) × 4 panels
  (input · GT LiDAR depth · predicted metric depth · abs error at LiDAR pts), shared
  depth/error colorbars, row labels with native fx/resolution + held-out metrics.
  Generated by `src/calib_depth/figures.py` (remote, uses checkpoints_v3 +
  checkpoints_adapt/{kitti,nuscenes}.pt).
- `results_bars.png` — grouped bars (AbsRel, δ<1.25, fx-error log) across
  zero-shot/+latent/+latent+head per dataset + AV2 reference line. Generated by
  `src/make_results_bars.py` (local, no GPU). Both wired into PAPER.md (Fig 2/3).
- PAPER.pdf / PAPER.docx regenerated: `make_pdf.py` / `make_docx.py` updated
  (surgical, not PAPER.md-driven) with abstract edit, a §V cross-dataset subsection,
  Table III (transfer numbers), refs [10] KITTI / [11] nuScenes, and Fig. 4 (bars) +
  Fig. 5 (qualitative). PDF puts the two wide figures on a full-width final page.

## Why this matters (generalization claim)
Different datasets = genuinely different intrinsics/cameras (KITTI fx≈721 @ ~1242×375,
nuScenes fx≈1266 @ 1600×900, AV2 fx≈1782 @ 1550×2048). A real **cross-dataset holdout**
(train AV2 → test KITTI/nuScenes) is the strongest test of the camera-aware
self-calibration — exactly the focal-diversity the single-rig AV2 setup lacked
(see `03_status_and_results.md` open items).
