# info_logs/ — fast-context entry point

Read these **first**, before the full `FRAMEWORK.md` / `PAPER.md` / source, to get
the whole project state in ~1/10th the tokens. The detailed, dated run journals
live in `info/*.txt` (per the repo's mandatory-`info/` convention); the files here
are a distilled, always-current index that points back into them.

## Reading order
1. `01_project_overview.md` — what this repo is, the two deliverables, the core idea.
2. `02_architecture.md`     — SelfCalibDepth design: ray-map coupling, modules, losses, θ model.
3. `03_status_and_results.md` — current state, v1→v3 ablation numbers, headline metrics, next steps.
4. `04_environment_and_data.md` — Python/venv, remote RTX 5090, dataset on disk, key constants, gotchas.
5. `05_benchmarks.md`         — unified multi-benchmark layer (AV2·KITTI·nuScenes·Lyft): the adapter contract, registry, CLIs, data prerequisites; cross-dataset + few-shot + distortion-robustness results.
6. `06_unidepth_analysis.md`  — UniDepth architecture (code-grounded), why it fails under distortion (pinhole-only camera head), and ranked improvements incl. our LiDAR few-shot method.

## Source of truth map
| Topic                        | Authoritative file                          |
|------------------------------|---------------------------------------------|
| Full framework design        | `FRAMEWORK.md`                              |
| Paper (4 pp.)                | `PAPER.md` / `PAPER.pdf` / `PAPER.docx`    |
| Data-format reference        | `FORMAT.md`                                |
| Visualizer usage             | `README.md`                                |
| Dated run journals           | `info/2026-*.txt`                          |
| Depth/calib package          | `src/calib_depth/`                         |

> Keep these files current: after a significant run, append the result to the
> relevant dated journal in `info/` **and** update `03_status_and_results.md` here.
