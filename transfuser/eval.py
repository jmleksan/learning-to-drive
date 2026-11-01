"""
Open-loop eval of TransFuser-lite checkpoints on held-out runs.

Prints one row per checkpoint, so ablations (fusion vs camera-only vs
lidar-only, aux vs no aux) line up in one table.

  ADE / FDE      mean / final waypoint error (m)
  turn ADE       frames where the expert's 2 s waypoint is >1.5 m sideways
  stop ADE       frames where the expert moves <1 m in 2 s
  stop acc       controller brakes (from predicted waypoints) iff expert stops
  obey true/mir  on frames just before a turn, the 2 s waypoint's sideways
                 position in meters (+ = expert's turn side), given the true
                 and the mirrored targets. A model that follows the route goes
                 negative when mirrored, one that follows the scene stays
                 positive. Open-loop ADE cannot show this.
  img / bev mIoU aux semantics; red / brake acc aux affordances

Examples:
  python -m transfuser.eval --data data/run_C --models models/transfuser.pt
  python -m transfuser.eval --data data/run_C --models models/tf_*.pt --viz 20
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from transfuser.controller import WaypointController
from transfuser.data import (
    BEV_CELL_M,
    BEV_CLASSES,
    BEV_FORWARD_M,
    BEV_LATERAL_M,
    IGNORE,
    IMAGE_MEAN,
    IMAGE_STD,
    IMG_CLASSES,
    TransfuserDataset,
)
from transfuser.model import load_checkpoint
from transfuser.train import run_model, to_device
from utils.device import resolve_device

TURN_LATERAL_M = 1.5
STOP_TRAVEL_M = 1.0
# Pre-turn frames: the expert is straight for the next 0.5 s, turning within 2 s.
PRETURN_NOW_M = 0.2
PRETURN_LATER_M = 1.0


def _confusion(pred: torch.Tensor, label: torch.Tensor, n: int) -> np.ndarray:
    keep = label != IGNORE
    idx = label[keep] * n + pred[keep]
    return torch.bincount(idx, minlength=n * n).reshape(n, n).cpu().numpy()


def _miou(conf: np.ndarray) -> float:
    inter = np.diag(conf)
    union = conf.sum(0) + conf.sum(1) - inter
    valid = union > 0
    return float((inter[valid] / union[valid]).mean()) if valid.any() else float("nan")


def _bev_px(xy: np.ndarray, scale: int) -> list[tuple[float, float]]:
    """Ego (x fwd, y right) meters → pixel in the BEV image."""
    return [
        ((y + BEV_LATERAL_M) / BEV_CELL_M * scale, (BEV_FORWARD_M - x) / BEV_CELL_M * scale)
        for x, y in xy
    ]


def render(batch: dict, out: dict, i: int, path: Path) -> None:
    """Image strip on top, lidar BEV below with target (blue), expert (green), net (red)."""
    img = batch["image"][i].cpu().numpy().transpose(1, 2, 0) * IMAGE_STD + IMAGE_MEAN
    strip = Image.fromarray((img.clip(0, 1) * 255).astype(np.uint8))
    lidar = batch["lidar"][i].cpu().numpy()
    scale = 3
    # Above-ground hits magenta, ground hits green; boosted so single hits show.
    bev = np.stack([lidar[1], lidar[0] * 0.6, lidar[1]], axis=-1) * 3.0
    bev_img = Image.fromarray((bev.clip(0, 1) * 255).astype(np.uint8)).resize(
        (bev.shape[1] * scale, bev.shape[0] * scale), Image.NEAREST
    )
    draw = ImageDraw.Draw(bev_img)
    for xy, color in (
        (batch["waypoints"][i].cpu().numpy(), (0, 220, 0)),
        (out["waypoints"][i].float().cpu().numpy(), (255, 60, 60)),
    ):
        pts = [_bev_px(np.zeros((1, 2)), scale)[0]] + _bev_px(xy, scale)
        draw.line(pts, fill=color, width=2)
        for px, py in pts[1:]:
            draw.ellipse([px - 3, py - 3, px + 3, py + 3], outline=color)
    for xy in batch["target"][i].cpu().numpy().reshape(-1, 2):
        tx, ty = _bev_px(xy[None], scale)[0]
        tx = min(max(tx, 4), bev_img.width - 5)
        ty = min(max(ty, 4), bev_img.height - 5)
        draw.rectangle([tx - 4, ty - 4, tx + 4, ty + 4], outline=(80, 140, 255), width=2)
    canvas = Image.new("RGB", (max(strip.width, bev_img.width), strip.height + bev_img.height))
    canvas.paste(strip, (0, 0))
    canvas.paste(bev_img, ((canvas.width - bev_img.width) // 2, strip.height))
    canvas.save(path)


@torch.no_grad()
def evaluate(
    model_path: Path,
    loader: DataLoader,
    device: torch.device,
    *,
    viz_dir: Path | None = None,
    viz_count: int = 0,
) -> dict:
    model, ckpt = load_checkpoint(model_path, device)
    err_all, turn, stop = [], [], []
    obey_true, obey_mirror = [], []
    stop_hits = 0
    img_conf = np.zeros((len(IMG_CLASSES),) * 2, dtype=np.int64)
    bev_conf = np.zeros((len(BEV_CLASSES),) * 2, dtype=np.int64)
    aff_correct = np.zeros(2)
    n = 0
    viz_every = max(1, len(loader.dataset) // max(1, viz_count)) if viz_count else 0
    seen = 0
    for batch in loader:
        batch = to_device(batch, device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out = run_model(model, batch)
        pred = out["waypoints"].float()
        gt = batch["waypoints"]

        side = torch.sign(gt[:, -1, 1])
        preturn = (gt[:, 0, 1].abs() < PRETURN_NOW_M) & (gt[:, -1, 1].abs() > PRETURN_LATER_M)
        if preturn.any():
            mirrored = dict(batch)
            mirrored["target"] = batch["target"] * torch.tensor([1.0, -1.0, 1.0, -1.0], device=device)
            lidar = batch["lidar"].clone()
            lidar[:, 2:] = torch.flip(lidar[:, 2:], dims=[-1])
            mirrored["lidar"] = lidar
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                mir = run_model(model, mirrored)["waypoints"].float()
            obey_true += (pred[:, -1, 1] * side)[preturn].cpu().tolist()
            obey_mirror += (mir[:, -1, 1] * side)[preturn].cpu().tolist()
        err = torch.linalg.norm(pred - gt, dim=-1).cpu().numpy()
        gt_np = gt.cpu().numpy()
        pred_np = pred.cpu().numpy()
        speed = batch["speed"][:, 0].cpu().numpy()
        for b in range(len(err)):
            err_all.append(err[b])
            if abs(gt_np[b, -1, 1]) > TURN_LATERAL_M:
                turn.append(err[b])
            gt_stop = np.linalg.norm(gt_np[b, -1]) < STOP_TRAVEL_M
            if gt_stop:
                stop.append(err[b])
            _, _, _, dbg = WaypointController().step(pred_np[b], float(speed[b]))
            stop_hits += int(dbg["braking"] == gt_stop)
        if "img_seg" in out and "img_seg" in batch:
            img_conf += _confusion(out["img_seg"].argmax(1), batch["img_seg"], len(IMG_CLASSES))
        if "bev_seg" in out:
            bev_conf += _confusion(out["bev_seg"].argmax(1), batch["bev_seg"], len(BEV_CLASSES))
        if "affordance" in out:
            hit = (out["affordance"].float() > 0) == (batch["affordance"] > 0.5)
            aff_correct += hit.float().sum(0).cpu().numpy()
        n += len(err)

        if viz_dir is not None and viz_every:
            for b in range(len(err)):
                if (seen + b) % viz_every == 0:
                    render(batch, out, b, viz_dir / f"{model_path.stem}_{seen + b:06d}.png")
        seen += len(err)

    err_all = np.array(err_all)
    result = {
        "model": str(model_path),
        "modality": ckpt["modality"],
        "aux": ckpt["aux"],
        "backbone": ckpt["backbone"],
        "pretrained": ckpt["pretrained"],
        "frames": n,
        "ade_m": float(err_all.mean()),
        "fde_m": float(err_all[:, -1].mean()),
        "turn_ade_m": float(np.mean(turn)) if turn else float("nan"),
        "turn_frames": len(turn),
        "stop_ade_m": float(np.mean(stop)) if stop else float("nan"),
        "stop_frames": len(stop),
        "stop_acc": stop_hits / max(1, n),
        "obey_true_m": float(np.mean(obey_true)) if obey_true else float("nan"),
        "obey_mirror_m": float(np.mean(obey_mirror)) if obey_mirror else float("nan"),
        "preturn_frames": len(obey_true),
    }
    if img_conf.any():
        result["img_miou"] = _miou(img_conf)
    if bev_conf.any():
        result["bev_miou"] = _miou(bev_conf)
    if "affordance" in out:
        result["red_light_acc"] = float(aff_correct[0] / n)
        result["brake_acc"] = float(aff_correct[1] / n)
    return result


def _fmt(v) -> str:
    if isinstance(v, float):
        return "   -  " if np.isnan(v) else f"{v:6.3f}"
    return f"{v!s:>6}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, nargs="+", required=True, help="Held-out runs")
    parser.add_argument("--models", type=Path, nargs="+", required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--viz", type=int, default=0, help="Save N BEV/waypoint PNGs per model")
    parser.add_argument("--out", type=Path, default=Path("data/transfuser_eval"))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    ds = TransfuserDataset(args.data, stride=args.stride)
    loader = DataLoader(ds, batch_size=args.batch_size, num_workers=args.workers, shuffle=False)
    print(f"{len(ds)} frames from {len(args.data)} runs")
    args.out.mkdir(parents=True, exist_ok=True)

    results = [
        evaluate(m, loader, device, viz_dir=args.out if args.viz else None, viz_count=args.viz)
        for m in args.models
    ]
    cols = [
        "modality", "aux", "backbone", "pretrained", "ade_m", "fde_m", "turn_ade_m", "stop_ade_m", "stop_acc",
        "obey_true_m", "obey_mirror_m",
        "img_miou", "bev_miou", "red_light_acc", "brake_acc",
    ]
    print("\n" + f"{'model':28s} " + " ".join(f"{c[:10]:>10s}" for c in cols))
    for r in results:
        print(f"{Path(r['model']).name:28s} " + " ".join(f"{_fmt(r.get(c, float('nan'))):>10s}" for c in cols))
    if results:
        print(
            f"(turn frames {results[0]['turn_frames']}, stop frames {results[0]['stop_frames']}, "
            f"pre-turn frames {results[0]['preturn_frames']})"
        )
    path = args.out / "open_loop.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {path}" + (f" and {args.viz} PNGs per model" if args.viz else ""))


if __name__ == "__main__":
    main()
