"""Bird's-Eye-View car localization + tracking from a monocular metric-depth model.

Pipeline (best available depth model = UniDepth-V2-L, but model-agnostic):
  1. run the depth model on a camera image -> metric depth map (camera-z);
  2. for each ground-truth car (nuScenes 3-D boxes give us cars + an evaluation GT),
     project the box, sample predicted depth over its footprint, and back-project to
     a camera-frame point -> its BEV position (lateral x, forward z);
  3. compare to the GT box centre -> BEV localization metrics;
  4. render: camera image with car boxes + a top-down BEV (ego at origin, camera FOV,
     range rings, GT vs predicted car positions);
  5. (tracking) across a scene's frames, associate cars by instance token and draw
     GT vs predicted BEV trajectories in the global frame.

Run in the baselines venv (has nuscenes-devkit + unidepth):
    NUSCENES_VERSION=v1.0-mini XFORMERS_DISABLED=1 PYTHONPATH=src \
      <baselines-venv>/bin/python -m calib_depth.bev_tracker --root <nuscenes> --out viz_bev
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("bev")

CAR_KEYS = ("vehicle.car", "vehicle.truck", "vehicle.bus", "vehicle.construction",
            "vehicle.trailer", "vehicle.emergency")
MAX_RANGE = 40.0


def is_car(name: str) -> bool:
    return name.startswith(CAR_KEYS)


def load_unidepth(dev):
    from unidepth.models import UniDepthV2
    net = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(dev).eval()

    @torch.no_grad()
    def predict(img_uint8):
        h, w = img_uint8.shape[:2]
        rgb = torch.from_numpy(img_uint8).permute(2, 0, 1).to(dev)
        out = net.infer(rgb)
        d = out["depth"].squeeze()
        if d.shape != (h, w):
            d = torch.nn.functional.interpolate(d[None, None], size=(h, w),
                                                mode="bilinear", align_corners=False)[0, 0]
        return d.cpu().numpy()
    return predict


def sample_depth(dmap, u, v, box2d):
    """Median predicted depth over the car's projected 2-D footprint (robust)."""
    h, w = dmap.shape
    x0, y0, x1, y1 = box2d
    x0, x1 = np.clip([x0, x1], 0, w - 1).astype(int)
    y0, y1 = np.clip([y0, y1], 0, h - 1).astype(int)
    patch = dmap[y0:y1 + 1, x0:x1 + 1]
    patch = patch[np.isfinite(patch) & (patch > 0)]
    if patch.size >= 8:
        return float(np.median(patch))
    uu, vv = int(np.clip(u, 0, w - 1)), int(np.clip(v, 0, h - 1))
    return float(dmap[vv, uu])


def estimate_cars(predict, nusc, sample, dev, cam="CAM_FRONT"):
    """Return per-car (gt_xz, pred_xz, dist_gt, uv, box2d, instance) for cars in view."""
    import cv2
    from nuscenes.utils.geometry_utils import BoxVisibility, view_points

    from pyquaternion import Quaternion

    cam_tok = sample["data"][cam]
    path, boxes, K = nusc.get_sample_data(cam_tok, box_vis_level=BoxVisibility.ANY)
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    dmap = predict(img)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    H, W = img.shape[:2]
    # camera -> ego -> global transforms (for tracking in the global frame)
    sd = nusc.get("sample_data", cam_tok)
    cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
    ep = nusc.get("ego_pose", sd["ego_pose_token"])
    R_ce, t_ce = Quaternion(cs["rotation"]).rotation_matrix, np.array(cs["translation"])
    R_eg, t_eg = Quaternion(ep["rotation"]).rotation_matrix, np.array(ep["translation"])

    def to_global(p_cam):
        return R_eg @ (R_ce @ p_cam + t_ce) + t_eg

    rows = []
    for b in boxes:
        if not is_car(b.name):
            continue
        c = b.center                       # camera frame (x right, y down, z forward)
        if c[2] <= 1.0 or np.linalg.norm([c[0], c[2]]) > MAX_RANGE:
            continue
        u, v = fx * c[0] / c[2] + cx, fy * c[1] / c[2] + cy
        corners = view_points(b.corners(), K, normalize=True)[:2]   # (2,8)
        box2d = [corners[0].min(), corners[1].min(), corners[0].max(), corners[1].max()]
        z_pred = sample_depth(dmap, u, v, box2d)
        x_pred, y_pred = (u - cx) / fx * z_pred, (v - cy) / fy * z_pred
        ann = nusc.get("sample_annotation", b.token) if b.token else {}
        rows.append(dict(gt_xz=np.array([c[0], c[2]]), pred_xz=np.array([x_pred, z_pred]),
                         dist_gt=float(np.linalg.norm([c[0], c[2]])), uv=(u, v),
                         box2d=box2d, name=b.name, instance=ann.get("instance_token"),
                         gt_global=np.array(ann["translation"][:2]) if ann else None,
                         pred_global=to_global(np.array([x_pred, y_pred, z_pred]))[:2],
                         pred_g3=to_global(np.array([x_pred, y_pred, z_pred])),
                         gt_g3=np.array(ann["translation"]) if ann else None,
                         ego_global=t_eg[:2]))
    tf = (R_ce, t_ce, R_eg, t_eg)
    return img, dmap, rows, (fx, fy, cx, cy, H, W), tf


def global_to_cam(p_g3, tf):
    """Inverse of to_global: global 3-D point -> camera-frame (x, y, z)."""
    R_ce, t_ce, R_eg, t_eg = tf
    return R_ce.T @ (R_eg.T @ (np.asarray(p_g3) - t_eg) - t_ce)


def kalman_cv(times, meas, sigma_a=1.5, r_std=3.0, gate=9.21):
    """Constant-velocity Kalman smoother with innovation gating, on 2-D measurements.

    State [x, y, vx, vy]. Process noise from acceleration std ``sigma_a`` (m/s^2,
    small => smooth). A measurement whose Mahalanobis distance exceeds ``gate``
    (chi-square, 2 dof, 99% = 9.21) is treated as a false/jittery detection and
    REJECTED — the filter predicts through it, so the track cannot jump. Returns the
    filtered positions and the number of rejected (gated) measurements."""
    n = len(meas)
    x = np.array([meas[0, 0], meas[0, 1], 0.0, 0.0])
    P = np.diag([r_std**2, r_std**2, 100.0, 100.0])
    R = np.eye(2) * r_std**2
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0.0]])
    out = np.zeros((n, 2))
    rejected = 0
    for t in range(n):
        if t > 0:
            dt = float(max(1e-2, times[t] - times[t - 1]))
            F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.0]])
            q = sigma_a**2
            Q = q * np.array([[dt**4 / 4, 0, dt**3 / 2, 0], [0, dt**4 / 4, 0, dt**3 / 2],
                              [dt**3 / 2, 0, dt**2, 0], [0, dt**3 / 2, 0, dt**2]])
            x = F @ x
            P = F @ P @ F.T + Q
        z = meas[t]
        if np.all(np.isfinite(z)):
            S = H @ P @ H.T + R
            y = z - H @ x
            d2 = float(y @ np.linalg.solve(S, y))
            if d2 <= gate:                      # accept measurement
                K = P @ H.T @ np.linalg.inv(S)
                x = x + K @ y
                P = (np.eye(4) - K @ H) @ P
            else:                                # gated out: predict-only (jump protection)
                rejected += 1
        out[t] = x[:2]
    return out, rejected


def metrics(rows):
    if not rows:
        return {}
    err = np.array([np.linalg.norm(r["pred_xz"] - r["gt_xz"]) for r in rows])
    lon = np.array([abs(r["pred_xz"][1] - r["gt_xz"][1]) for r in rows])
    lat = np.array([abs(r["pred_xz"][0] - r["gt_xz"][0]) for r in rows])
    d = np.array([r["dist_gt"] for r in rows])
    out = {"n": len(rows), "BEV_err_mean": err.mean(), "BEV_err_med": np.median(err),
           "lon_mean": lon.mean(), "lat_mean": lat.mean()}
    for lo, hi in [(0, 20), (20, 40)]:
        m = (d >= lo) & (d < hi)
        out[f"BEV_err_{lo}-{hi}"] = float(err[m].mean()) if m.any() else float("nan")
    return out


def render(img, rows, K, out_path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fx, fy, cx, cy, H, W = K
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(20, 8), dpi=120,
                                 gridspec_kw={"width_ratios": [1.4, 1]})
    a1.imshow(img); a1.set_title("camera + GT car boxes (pred/GT range, m)", fontweight="bold")
    for r in rows:
        x0, y0, x1, y1 = r["box2d"]
        a1.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#1B9E9E", lw=1.6))
        a1.text(r["uv"][0], y0 - 4, f"{np.linalg.norm(r['pred_xz']):.0f}/{r['dist_gt']:.0f}",
                color="white", fontsize=7, ha="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="#1B9E9E", ec="none", alpha=0.8))
    a1.set_xlim(0, W); a1.set_ylim(H, 0); a1.axis("off")

    # BEV: forward z = up, lateral x = right, ego/camera at origin
    a2.set_title("Bird's-Eye-View: car positions (● GT, ✕ predicted)", fontweight="bold")
    for rng in [r for r in (10, 20, 30, 40, 60, 80) if r <= MAX_RANGE]:
        a2.add_patch(plt.Circle((0, 0), rng, fill=False, ls=":", ec="#bbb", lw=0.8))
        a2.text(0, rng, f"{rng} m", color="#888", fontsize=7, ha="center", va="bottom")
    hf = np.degrees(np.arctan2(W / 2, fx))      # half horizontal FOV
    for s in (-1, 1):
        a2.plot([0, s * MAX_RANGE * np.sin(np.radians(hf))],
                [0, MAX_RANGE * np.cos(np.radians(hf))], color="#ddd", lw=0.8)
    a2.scatter([0], [0], c="k", marker="^", s=120, label="ego/camera", zorder=5)
    for r in rows:
        gx, gz = r["gt_xz"]; px, pz = r["pred_xz"]
        a2.plot([gx, px], [gz, pz], color="#E8743B", lw=0.8, alpha=0.7, zorder=3)
        a2.scatter([gx], [gz], c="#1B9E9E", s=55, zorder=4)
        a2.scatter([px], [pz], c="#E8743B", marker="x", s=55, zorder=4)
    a2.scatter([], [], c="#1B9E9E", s=55, label="GT")
    a2.scatter([], [], c="#E8743B", marker="x", s=55, label="predicted (UniDepth)")
    a2.set_xlabel("lateral x [m]"); a2.set_ylabel("forward z [m]")
    a2.set_xlim(-MAX_RANGE * 0.7, MAX_RANGE * 0.7); a2.set_ylim(0, MAX_RANGE)
    a2.set_aspect("equal"); a2.grid(alpha=0.2); a2.legend(loc="upper right", fontsize=8)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.savefig(out_path, bbox_inches="tight", facecolor="white"); plt.close(fig)
    log.info("saved BEV figure -> %s", out_path)


def render_tracks(tracks, ego_xy, out_path, title):
    """Global-frame BEV: per-car GT vs predicted trajectories over a scene."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 11), dpi=120)
    ego = np.array(ego_xy)
    ax.plot(ego[:, 0], ego[:, 1], "-", color="k", lw=2.2, label="ego path", zorder=5)
    ax.scatter(ego[0, 0], ego[0, 1], c="k", marker="^", s=110, zorder=6)
    cmap = plt.get_cmap("tab20")
    kept = 0
    for i, (inst, tr) in enumerate(sorted(tracks.items(),
                                           key=lambda kv: -len(kv[1]))):
        if len(tr) < 3:                       # only cars seen across >=3 frames
            continue
        gt = np.array([p["gt"] for p in tr]); pr = np.array([p["pred"] for p in tr])
        c = cmap(kept % 20)
        ax.plot(gt[:, 0], gt[:, 1], "-o", color=c, ms=3, lw=1.6, zorder=3)
        ax.plot(pr[:, 0], pr[:, 1], "--x", color=c, ms=4, lw=1.2, alpha=0.9, zorder=3)
        kept += 1
    ax.plot([], [], "-o", color="#555", label="GT track")
    ax.plot([], [], "--x", color="#555", label="predicted track (UniDepth)")
    ax.set_aspect("equal"); ax.grid(alpha=0.2)
    ax.set_xlabel("global x [m]"); ax.set_ylabel("global y [m]")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(title, fontweight="bold")
    fig.savefig(out_path, bbox_inches="tight", facecolor="white"); plt.close(fig)
    log.info("saved tracking figure (%d car tracks) -> %s", kept, out_path)


def _accel_rms(p):
    """Time-consistency proxy: RMS of the per-frame 2nd difference (acceleration) of a
    position track. Smooth motion -> small; jitter/jumps -> large."""
    if len(p) < 3:
        return 0.0
    a = p[2:] - 2 * p[1:-1] + p[:-2]
    return float(np.sqrt((a ** 2).sum(axis=1).mean()))


def _jumps(p, thresh):
    """Count frame-to-frame displacements above a plausibility threshold (m)."""
    if len(p) < 2:
        return 0
    return int((np.linalg.norm(np.diff(p, axis=0), axis=1) > thresh).sum())


def render_sequence(predict, nusc, dev, cam, scene_idx, out_dir, fps=6):
    """Collect a scene, Kalman-smooth each car track (jump-protected), measure
    time-consistency (raw vs smoothed), and render a raw-vs-smoothed BEV GIF."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image

    scene = nusc.scene[scene_idx]
    toks, t = [], scene["first_sample_token"]
    while t:
        toks.append(t); t = nusc.get("sample", t)["next"]

    # 1. collect per-frame data + per-instance global-position series
    frames, series = [], {}
    for fi, tk in enumerate(toks):
        s = nusc.get("sample", tk)
        img, _, rows, K, tf = estimate_cars(predict, nusc, s, dev, cam)
        frames.append(dict(img=img, rows=rows, K=K, tf=tf, time=s["timestamp"] / 1e6))
        for r in rows:
            if r["instance"] and r["pred_g3"] is not None and r["gt_g3"] is not None:
                series.setdefault(r["instance"], []).append((fi, r["pred_g3"], r["gt_g3"]))

    # 2. Kalman-smooth each track (in the global frame) + accumulate metrics
    smoothed = {}                       # (instance, frame) -> smoothed global 3-D
    raw_acc, sm_acc, raw_err, sm_err = [], [], [], []
    raw_jumps = sm_jumps = gated = 0
    JUMP_M = 12.0                       # >24 m/s between keyframes is implausible for a car
    for inst, seq in series.items():
        if len(seq) < 3:
            for fi, pg, _gg in seq:
                smoothed[(inst, fi)] = pg
            continue
        times = np.array([frames[e[0]]["time"] for e in seq])
        meas = np.array([e[1][:2] for e in seq])
        gtg = np.array([e[2][:2] for e in seq])
        sm, rej = kalman_cv(times, meas)
        gated += rej
        raw_acc.append(_accel_rms(meas)); sm_acc.append(_accel_rms(sm))
        raw_err.append(np.linalg.norm(meas - gtg, axis=1).mean())
        sm_err.append(np.linalg.norm(sm - gtg, axis=1).mean())
        raw_jumps += _jumps(meas, JUMP_M); sm_jumps += _jumps(sm, JUMP_M)
        for k, (fi, pg, _gg) in enumerate(seq):
            smoothed[(inst, fi)] = np.array([sm[k, 0], sm[k, 1], pg[2]])

    print("\n==========  TIME-CONSISTENCY (BEV tracks, scene %s)  ==========" % scene["name"])
    print(f"  tracks (>=3 frames): {len(raw_acc)} | gated jump-measurements: {gated}")
    print(f"  track smoothness (RMS step accel, lower=smoother):  raw {np.mean(raw_acc):.2f} m"
          f"  ->  smoothed {np.mean(sm_acc):.2f} m  ({100*(1-np.mean(sm_acc)/max(np.mean(raw_acc),1e-6)):.0f}% less jitter)")
    print(f"  implausible jumps (>{JUMP_M:.0f} m/frame):  raw {raw_jumps}  ->  smoothed {sm_jumps}")
    print(f"  accuracy vs GT (global BEV):  raw {np.mean(raw_err):.2f} m  ->  smoothed {np.mean(sm_err):.2f} m")
    print("=" * 64)

    # 3. render GIF: GT, raw prediction (faint), smoothed prediction (bold)
    rings = [r for r in (10, 20, 30, 40) if r <= MAX_RANGE]
    hf_lat, imgs = MAX_RANGE * 0.7, []
    inst_at = {}                        # frame -> {instance: row}
    for fi, fr in enumerate(frames):
        inst_at[fi] = {r["instance"]: r for r in fr["rows"] if r["instance"]}
    for fi, fr in enumerate(frames):
        fx, fy, cx, cy, H, W = fr["K"]
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.4), dpi=72,
                                     gridspec_kw={"width_ratios": [1.5, 1]})
        a1.imshow(fr["img"]); a1.axis("off")
        a1.set_title(f"{cam}   frame {fi + 1}/{len(frames)}", fontsize=10, fontweight="bold")
        for r in fr["rows"]:
            x0, y0, x1, y1 = r["box2d"]
            a1.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#1B9E9E", lw=1.4))
        for rng in rings:
            a2.add_patch(plt.Circle((0, 0), rng, fill=False, ls=":", ec="#ccc", lw=0.7))
            a2.text(0, rng, f"{rng}", color="#999", fontsize=6, ha="center", va="bottom")
        hf = np.degrees(np.arctan2(W / 2, fx))
        for sgn in (-1, 1):
            a2.plot([0, sgn * MAX_RANGE * np.sin(np.radians(hf))],
                    [0, MAX_RANGE * np.cos(np.radians(hf))], color="#e3e3e3", lw=0.7)
        a2.scatter([0], [0], c="k", marker="^", s=80, zorder=6)
        for r in fr["rows"]:
            gx, gz = r["gt_xz"]; px, pz = r["pred_xz"]
            a2.scatter([gx], [gz], c="#1B9E9E", s=42, zorder=4)            # GT
            a2.scatter([px], [pz], c="#E8743B", marker="x", s=34, alpha=0.35, zorder=3)  # raw
            sg = smoothed.get((r["instance"], fi))
            if sg is not None:
                pc = global_to_cam(sg, fr["tf"])
                a2.scatter([pc[0]], [pc[2]], c="#D11", marker="D", s=30, zorder=5)        # smoothed
                a2.plot([gx, pc[0]], [gz, pc[2]], color="#D11", lw=0.6, alpha=0.5, zorder=2)
        a2.scatter([], [], c="#1B9E9E", s=42, label="GT")
        a2.scatter([], [], c="#E8743B", marker="x", s=34, label="raw prediction")
        a2.scatter([], [], c="#D11", marker="D", s=30, label="smoothed (Kalman)")
        a2.set_xlim(-hf_lat, hf_lat); a2.set_ylim(0, MAX_RANGE); a2.set_aspect("equal")
        a2.grid(alpha=0.2); a2.legend(loc="upper right", fontsize=7)
        a2.set_xlabel("lateral x [m]"); a2.set_ylabel("forward z [m]")
        a2.set_title(f"BEV (cap {MAX_RANGE:.0f} m) — {len(fr['rows'])} cars", fontsize=10, fontweight="bold")
        fig.suptitle(f"Time-consistent BEV car tracking (UniDepth-V2-L + Kalman)  ·  "
                     f"nuScenes scene {scene['name']}", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.canvas.draw()
        imgs.append(Image.frombytes("RGBA", fig.canvas.get_width_height(),
                                    fig.canvas.buffer_rgba().tobytes()).convert("RGB"))
        plt.close(fig)
    out = out_dir / f"bev_seq_{scene['name']}.gif"
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=int(1000 / fps), loop=0)
    log.info("saved time-consistent BEV GIF (%d frames) -> %s", len(imgs), out)


def bev_consistency(predict, nusc, dev, cam, n_scenes):
    """Video-level metrics over a set of scenes (the 'rolling videos').

    Spatial: BEV-delta@tau = fraction of car observations localized within tau metres
    in BEV (the position analogue of depth's d1; tau in {1,2,4} m = nuScenes detection
    centre-distance thresholds). Temporal: track jitter (RMS step-accel) + jump rate,
    reported raw vs Kalman-smoothed."""
    taus = [1.0, 2.0, 4.0]
    raw_e, sm_e, raw_acc, sm_acc = [], [], [], []
    gated = raw_jumps = sm_jumps = 0
    for si in range(n_scenes):
        scene = nusc.scene[si]
        toks, t = [], scene["first_sample_token"]
        while t:
            toks.append(t); t = nusc.get("sample", t)["next"]
        times, series = [], {}
        for fi, tk in enumerate(toks):
            s = nusc.get("sample", tk)
            _, _, rows, _, _ = estimate_cars(predict, nusc, s, dev, cam)
            times.append(s["timestamp"] / 1e6)
            for r in rows:
                if r["instance"] and r["pred_g3"] is not None and r["gt_g3"] is not None:
                    series.setdefault(r["instance"], []).append((fi, r["pred_g3"], r["gt_g3"]))
        for seq in series.values():
            meas = np.array([e[1][:2] for e in seq]); gtg = np.array([e[2][:2] for e in seq])
            if len(seq) >= 3:
                tt = np.array([times[e[0]] for e in seq])
                sm, rej = kalman_cv(tt, meas); gated += rej
                raw_acc.append(_accel_rms(meas)); sm_acc.append(_accel_rms(sm))
                raw_jumps += _jumps(meas, 12.0); sm_jumps += _jumps(sm, 12.0)
            else:
                sm = meas
            raw_e.append(np.linalg.norm(meas - gtg, axis=1))
            sm_e.append(np.linalg.norm(sm - gtg, axis=1))
    raw_e = np.concatenate(raw_e); sm_e = np.concatenate(sm_e)

    print("\n========  ROLLING-VIDEO BEV METRICS (%d scenes, %d car-obs, cap %.0f m)  ========"
          % (n_scenes, len(raw_e), MAX_RANGE))
    print("  BEV-delta@tau  (fraction of cars within tau m of GT in BEV; higher better):")
    for tau in taus:
        print(f"     tau={tau:.0f} m :  raw {(raw_e <= tau).mean():.3f}   smoothed {(sm_e <= tau).mean():.3f}")
    print(f"  temporal jitter (RMS step-accel, lower better): raw {np.mean(raw_acc):.2f} m -> "
          f"smoothed {np.mean(sm_acc):.2f} m  ({100*(1-np.mean(sm_acc)/max(np.mean(raw_acc),1e-6)):.0f}% less)")
    print(f"  implausible jumps (>12 m/frame): raw {raw_jumps} -> smoothed {sm_jumps} "
          f"(gated {gated} measurements)")
    print(f"  mean BEV error: raw {raw_e.mean():.2f} m -> smoothed {sm_e.mean():.2f} m")
    print("=" * 72)
    return {"taus": taus, "raw_e": raw_e, "sm_e": sm_e,
            "raw_acc": float(np.mean(raw_acc)), "sm_acc": float(np.mean(sm_acc))}


def run_tracking(predict, nusc, dev, cam, out_dir, scene_idx=0):
    """Walk one scene's samples, associate cars by instance token -> BEV trajectories."""
    scene = nusc.scene[scene_idx]
    tracks, ego_xy, tok = {}, [], scene["first_sample_token"]
    nframes = 0
    while tok:
        s = nusc.get("sample", tok)
        _, _, rows, _, _ = estimate_cars(predict, nusc, s, dev, cam)
        nframes += 1
        for r in rows:
            if r["instance"] is None or r["gt_global"] is None:
                continue
            tracks.setdefault(r["instance"], []).append(
                {"gt": r["gt_global"], "pred": r["pred_global"]})
            ego_xy.append(r["ego_global"])
        tok = s["next"]
    log.info("scene '%s': %d frames, %d car instances", scene["name"], nframes, len(tracks))
    # dedup ego path (one per frame)
    ego = np.array(ego_xy)[:: max(1, len(ego_xy) // max(1, nframes))][:nframes] if ego_xy else np.zeros((1, 2))
    render_tracks(tracks, ego, out_dir / f"bev_tracks_{scene['name']}.png",
                  f"BEV car tracking from UniDepth-V2-L  ·  scene {scene['name']}  ·  "
                  f"{sum(len(v) >= 3 for v in tracks.values())} tracked cars")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--cam", default="CAM_FRONT")
    ap.add_argument("--frames", type=int, default=80, help="samples to score for metrics")
    ap.add_argument("--track", action="store_true", help="also render per-car BEV trajectories")
    ap.add_argument("--animate", action="store_true", help="render an animated BEV GIF of the scene")
    ap.add_argument("--consistency", type=int, default=0,
                    help="compute rolling-video BEV metrics over the first N scenes")
    ap.add_argument("--scene-idx", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("viz_bev"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=os.environ.get("NUSCENES_VERSION", "v1.0-mini"),
                    dataroot=str(args.root), verbose=False)
    predict = load_unidepth(dev)

    if args.consistency:
        bev_consistency(predict, nusc, dev, args.cam, args.consistency)
        return

    samples = nusc.sample[: args.frames]
    log.info("BEV car localization | %s | %d samples", args.cam, len(samples))
    all_rows, best = [], (None, -1)
    for i, s in enumerate(samples):
        img, dmap, rows, K, _tf = estimate_cars(predict, nusc, s, dev, args.cam)
        all_rows += rows
        if len(rows) > best[1]:                       # keep the busiest frame for the figure
            best = ((img, rows, K, s["token"]), len(rows))
        if (i + 1) % 20 == 0:
            log.info("  %d/%d samples, %d cars so far", i + 1, len(samples), len(all_rows))

    m = metrics(all_rows)
    print("\n==========  BEV CAR LOCALIZATION (UniDepth-V2-L, nuScenes %s)  ==========" % args.cam)
    print(f"  cars evaluated: {m['n']}")
    print(f"  BEV position error:  mean {m['BEV_err_mean']:.2f} m | median {m['BEV_err_med']:.2f} m")
    print(f"  longitudinal (z):    mean {m['lon_mean']:.2f} m   lateral (x): mean {m['lat_mean']:.2f} m")
    print(f"  by range:  0-20 m {m['BEV_err_0-20']:.2f} | 20-40 m {m['BEV_err_20-40']:.2f}  "
          f"(capped at {MAX_RANGE:.0f} m)")
    print("=" * 72)

    img, rows, K, tok = best[0]
    render(img, rows, K, args.out / f"bev_{args.cam}_{tok[:10]}.png",
           f"BEV car localization from UniDepth-V2-L  ·  nuScenes {args.cam}  ·  {len(rows)} cars")

    if args.track:
        run_tracking(predict, nusc, dev, args.cam, args.out, args.scene_idx)
    if args.animate:
        render_sequence(predict, nusc, dev, args.cam, args.scene_idx, args.out)


if __name__ == "__main__":
    main()
