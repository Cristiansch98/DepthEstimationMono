"""Monocular video -> depth heatmap + detection overlay + BEV car tracking (GIFs).

Fully monocular, no GT/LiDAR/calibration: UniDepth-V2-L gives metric depth AND the
camera intrinsics; torchvision Faster R-CNN (COCO) detects vehicles; cars are
back-projected with the predicted camera to a Bird's-Eye-View and tracked with a
per-track constant-velocity Kalman filter (online, with gating) for time consistency.

Outputs three GIFs: <out>/heatmap.gif, <out>/direct.gif, <out>/bev.gif.

    PYTHONPATH=src <venv>/bin/python -m calib_depth.video_demo \
        --frames data/realvideo/frames --out viz_video --fps 6
"""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("video-demo")
VEHICLE = {"car", "truck", "bus"}
HEAT_MAX, BEV_MAX = 60.0, 50.0


def load_unidepth(dev):
    from unidepth.models import UniDepthV2
    net = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14").to(dev).eval()

    @torch.no_grad()
    def predict(img):
        h, w = img.shape[:2]
        out = net.infer(torch.from_numpy(img).permute(2, 0, 1).to(dev))
        d = out["depth"].squeeze().float()
        if d.shape != (h, w):
            d = torch.nn.functional.interpolate(d[None, None], (h, w), mode="bilinear",
                                                align_corners=False)[0, 0]
        K = out["intrinsics"].squeeze().float().cpu().numpy()
        return d.cpu().numpy(), K
    return predict


def load_detector(dev):
    from torchvision.models.detection import (fasterrcnn_resnet50_fpn_v2,
                                              FasterRCNN_ResNet50_FPN_V2_Weights)
    w = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
    net = fasterrcnn_resnet50_fpn_v2(weights=w).eval().to(dev)
    cats = w.meta["categories"]

    @torch.no_grad()
    def detect(img, thr=0.55):
        x = torch.from_numpy(img).permute(2, 0, 1).float().div(255).to(dev)
        o = net([x])[0]
        out = []
        for b, l, s in zip(o["boxes"].cpu().numpy(), o["labels"].cpu().numpy(),
                           o["scores"].cpu().numpy()):
            if s >= thr and cats[l] in VEHICLE:
                out.append(b)
        return out
    return detect


class KF:
    """Online constant-velocity Kalman (2-D), with innovation gating."""
    def __init__(self, xz, sigma_a=2.0, r=2.5):
        self.x = np.array([xz[0], xz[1], 0.0, 0.0]); self.P = np.diag([r, r, 50, 50.0])
        self.sa, self.r = sigma_a, r

    def predict(self, dt):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.0]])
        q = self.sa ** 2
        Q = q * np.array([[dt**4/4, 0, dt**3/2, 0], [0, dt**4/4, 0, dt**3/2],
                          [dt**3/2, 0, dt**2, 0], [0, dt**3/2, 0, dt**2]])
        self.x = F @ self.x; self.P = F @ self.P @ F.T + Q

    def update(self, z, gate=9.21):
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0.0]]); R = np.eye(2) * self.r
        S = H @ self.P @ H.T + R; y = z - H @ self.x
        if float(y @ np.linalg.solve(S, y)) > gate:
            return False
        Kk = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + Kk @ y; self.P = (np.eye(4) - Kk @ H) @ self.P
        return True

    @property
    def pos(self):
        return self.x[:2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("viz_video"))
    ap.add_argument("--fps", type=int, default=6)
    ap.add_argument("--gif-w", type=int, default=640)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    files = sorted(glob.glob(str(args.frames / "*.jpg")) + glob.glob(str(args.frames / "*.png")))
    log.info("frames: %d", len(files))
    predict, detect = load_unidepth(dev), load_detector(dev)

    heat_imgs, direct_imgs, bev_imgs = [], [], []
    tracks, next_id, dt = {}, 0, 1.0 / args.fps
    rings = [10, 20, 30, 40, 50]

    for fi, fp in enumerate(files):
        img = np.array(Image.open(fp).convert("RGB"))
        H, W = img.shape[:2]
        depth, K = predict(img)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

        # ---- heatmap (per-frame RELATIVE normalization; UniDepth's absolute metric
        #      scale is unreliable on an OOD dashcam, but its depth structure is good) ----
        lo, hi = np.percentile(depth, [2, 98])
        dn = np.clip((depth - lo) / (hi - lo + 1e-6), 0, 1)
        heat = cv2.cvtColor(cv2.applyColorMap((dn * 255).astype(np.uint8), cv2.COLORMAP_TURBO),
                            cv2.COLOR_BGR2RGB)
        cv2.putText(heat, "relative depth  near -> far  (UniDepth-V2)", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        heat_imgs.append(heat)

        # ---- ground-plane metric-scale recovery (assume camera height ~1.5 m): the
        #      road strip below the horizon / above the hood should sit at Y=h_cam ----
        H_CAM = 1.5
        v0, v1, c0, c1 = int(0.60 * H), int(0.82 * H), int(0.35 * W), int(0.65 * W)
        sub = depth[v0:v1, c0:c1]
        vv = (np.arange(v0, v1)[:, None] - cy) / fy
        ys = (vv * sub)[(sub > 0) & (vv > 0)]
        scale = float(H_CAM / np.median(ys)) if ys.size and np.median(ys) > 0 else 1.0
        depth_m = depth * scale

        # ---- detection + back-projection to BEV ----
        dets = []
        for b in detect(img):
            x1, y1, x2, y2 = b
            uc, vc = (x1 + x2) / 2, min(y2, H - 1)            # bottom-centre = ground contact
            patch = depth_m[int(max(0, y2 - 8)):int(min(H, y2)), int(max(0, uc - 6)):int(min(W, uc + 6))]
            z = float(np.median(patch[patch > 0])) if (patch > 0).any() else float(depth_m[int(vc), int(uc)])
            if not (0 < z < BEV_MAX):
                continue
            X = (uc - cx) / fx * z
            dets.append((np.array([X, z]), b))

        # ---- online tracking (greedy NN + Kalman) ----
        for t in tracks.values():
            t["kf"].predict(dt); t["missed"] += 1
        used = set()
        for xz, b in dets:
            best, bd = None, 6.0
            for tid, t in tracks.items():
                if tid in used:
                    continue
                d = np.linalg.norm(t["kf"].pos - xz)
                if d < bd:
                    best, bd = tid, d
            if best is None:
                tracks[next_id] = {"kf": KF(xz), "missed": 0, "trail": [xz.copy()], "box": b}
                used.add(next_id); next_id += 1
            else:
                tracks[best]["kf"].update(xz); tracks[best]["missed"] = 0
                tracks[best]["trail"].append(tracks[best]["kf"].pos.copy())
                tracks[best]["box"] = b; used.add(best)
        tracks = {k: t for k, t in tracks.items() if t["missed"] <= 3}

        # ---- direct frame (boxes + track id) ----
        direct = img.copy()
        for tid in used:
            if tid not in tracks:
                continue
            b = tracks[tid]["box"]; x1, y1, x2, y2 = [int(v) for v in b]
            dist = float(np.linalg.norm(tracks[tid]["kf"].pos))
            cv2.rectangle(direct, (x1, y1), (x2, y2), (27, 158, 158), 2)
            cv2.putText(direct, f"#{tid} {dist:.0f}m", (x1, max(12, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (27, 158, 158), 2)
        direct_imgs.append(direct)

        # ---- BEV frame ----
        fig, ax = plt.subplots(figsize=(5.2, 5.2), dpi=80)
        for r in rings:
            ax.add_patch(plt.Circle((0, 0), r, fill=False, ls=":", ec="#ccc", lw=0.7))
            ax.text(0, r, f"{r}", color="#999", fontsize=6, ha="center")
        hf = np.degrees(np.arctan2(W / 2, fx))
        for sgn in (-1, 1):
            ax.plot([0, sgn * BEV_MAX * np.sin(np.radians(hf))],
                    [0, BEV_MAX * np.cos(np.radians(hf))], color="#e3e3e3", lw=0.7)
        ax.scatter([0], [0], c="k", marker="^", s=90, zorder=6)
        for tid in used:
            if tid not in tracks:
                continue
            p = tracks[tid]["kf"].pos; tr = np.array(tracks[tid]["trail"][-8:])
            ax.plot(tr[:, 0], tr[:, 1], "-", color="#E8743B", lw=1.0, alpha=0.6, zorder=3)
            ax.scatter([p[0]], [p[1]], c="#D11", marker="D", s=34, zorder=5)
            ax.text(p[0] + 1, p[1], f"#{tid}  {np.linalg.norm(p):.0f} m", color="#D11", fontsize=6.5)
        ax.set_xlim(-BEV_MAX * 0.7, BEV_MAX * 0.7); ax.set_ylim(0, BEV_MAX); ax.set_aspect("equal")
        ax.grid(alpha=0.2); ax.set_xlabel("lateral x [m]"); ax.set_ylabel("forward z [m]")
        ax.set_title(f"BEV car tracking (monocular)  ·  frame {fi+1}/{len(files)}", fontsize=10)
        fig.tight_layout(); fig.canvas.draw()
        bev_imgs.append(np.asarray(Image.frombytes("RGBA", fig.canvas.get_width_height(),
                                                    fig.canvas.buffer_rgba().tobytes()).convert("RGB")))
        plt.close(fig)
        if (fi + 1) % 20 == 0:
            log.info("  processed %d/%d", fi + 1, len(files))

    def save_mp4(arrs, name, width=None):
        w = (width or args.gif_w); w -= w % 2
        frames = []
        for a in arrs:
            h2 = int(round(w * a.shape[0] / a.shape[1])); h2 -= h2 % 2
            frames.append(cv2.cvtColor(cv2.resize(a, (w, h2)), cv2.COLOR_RGB2BGR))
        Hh, Ww = frames[0].shape[:2]
        out = str(args.out / name)
        vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (Ww, Hh))
        if not vw.isOpened():
            raise RuntimeError("cv2.VideoWriter could not open (mp4v codec missing)")
        for f in frames:
            vw.write(f)
        vw.release()
        log.info("saved %s (%d frames, %dx%d, %d fps)", out, len(frames), Ww, Hh, args.fps)

    save_mp4(heat_imgs, "heatmap.mp4")
    save_mp4(direct_imgs, "direct.mp4")
    save_mp4(bev_imgs, "bev.mp4")

    # ---- combined side-by-side panel: camera+detections | depth | BEV ----
    def panelize(d, h, b, ph=420):
        def rh(a):
            w = int(round(ph * a.shape[1] / a.shape[0])); w -= w % 2
            return cv2.resize(a, (w, ph)).astype(np.uint8).copy()
        out = []
        for im, txt in [(rh(d), "camera + detections"), (rh(h), "depth (relative)"),
                        (rh(b), "BEV  (id + distance)")]:
            cv2.rectangle(im, (0, 0), (im.shape[1], 24), (0, 0, 0), -1)
            cv2.putText(im, txt, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)
            out.append(im)
        return np.hstack(out)

    combined = [panelize(direct_imgs[i], heat_imgs[i], bev_imgs[i]) for i in range(len(direct_imgs))]
    save_mp4(combined, "combined.mp4", width=1440)


if __name__ == "__main__":
    main()
