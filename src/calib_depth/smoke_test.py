"""Quick GPU sanity check: forward+backward on real samples, verify theta grads."""
import time
from pathlib import Path

import torch

from calib_depth.camera_model import Intrinsics
from calib_depth.dataset import AV2SensorDepthDataset, discover_logs
from calib_depth.losses import total_loss
from calib_depth.model import CameraAwareDepthNet

CAMS = ["ring_front_center", "ring_front_left", "ring_front_right", "ring_side_left",
        "ring_side_right", "ring_rear_left", "ring_rear_right"]
W = {"silog": 1.0, "l3d": 0.5, "reproj": 0.05, "smooth": 0.1, "prior": 0.01}


def main():
    logs = discover_logs(Path("data/sensor/train"))
    ds = AV2SensorDepthDataset(logs, CAMS, target_hw=(518, 518))
    print("logs:", len(logs), "| frames:", len(ds))

    dev = "cuda"
    model = CameraAwareDepthNet(7, Intrinsics(500, 500, 259, 259), freeze_backbone=True).to(dev)
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable params (M): %.2f" % (ntr / 1e6))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    for step in range(6):
        s = ds[step * 50]
        img = s["image"][None].to(dev)
        cam_idx = torch.tensor([s["cam_idx"]], device=dev)
        lb = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in s.items()}
        lb["image"] = img
        t0 = time.time()
        opt.zero_grad()
        depth, theta = model(img, cam_idx)
        loss, parts = total_loss(depth, theta[0], lb, W)
        loss.backward()
        gfin = bool(torch.isfinite(model.calib.latent.grad).all())
        opt.step()
        cam = CAMS[s["cam_idx"]]
        fxp, fxg = theta[0, 0].item(), lb["theta_gt"][0].item()
        print("step%d cam=%-16s depth=%s loss=%.3f fx=%.0f/%.0f theta_grad_finite=%s %.2fs"
              % (step, cam, tuple(depth.shape), loss.item(), fxp, fxg, gfin, time.time() - t0))
    print("peak GPU mem (GB): %.2f" % (torch.cuda.max_memory_allocated() / 1e9))
    print("SMOKE_OK")


if __name__ == "__main__":
    main()
