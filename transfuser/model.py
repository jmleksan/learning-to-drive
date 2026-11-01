"""TransFuser-lite: camera + lidar BEV → 4 waypoints.

Same layout as TransFuser (Chitta et al., 2022), scaled down for a 6 GB GPU:

  image strip ─► ResNet stage 1 ─► stage 2 ─► stage 3 ─► stage 4 ─┐
                        ▲ │            ▲ │         ▲ │        ▲ │   ├─► pool + sum ─► MLP ─► GRU ─► waypoints
  lidar BEV   ─► ResNet stage 1 ─► stage 2 ─► stage 3 ─► stage 4 ─┘          (target points fed every step)
                        └─ transformer fusion after each stage ─┘

Fusion pools each feature map to a small token grid, runs self-attention
over image + lidar tokens (+ a speed embedding), upsamples the result back
and adds it to both branches. So the image can attend to where lidar sees
an obstacle and vice versa.

Aux heads (switch off with ``aux=False``): image semantics from the camera
branch, BEV semantics from the lidar branch, and two affordances (red light
for ego, expert braking) from the fused vector.

Backbones: ``resnet18`` (default, as in TransFuser) uses torchvision
ResNet-18 for both branches, the camera one from ImageNet weights when
``pretrained``; the lidar one always starts random (2-channel input, and
ImageNet features don't transfer to a BEV histogram). The lidar branch also
reads the goal channels drawn in transfuser/data.py. ``small`` is a
narrower ResNet-10-ish trained from scratch.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from transfuser.data import BEV_CLASSES, IMG_CLASSES, LIDAR_CHANNELS, NUM_TARGETS, NUM_WAYPOINTS

BACKBONE_CHANNELS = {
    "resnet18": (64, 128, 256, 512),
    "small": (32, 64, 128, 256),
}
BACKBONES = tuple(BACKBONE_CHANNELS)
IMAGE_ANCHORS = (4, 16)  # token grid for the 120×480 strip
LIDAR_ANCHORS = (8, 8)
MODALITIES = ("both", "camera", "lidar")


class BasicBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.skip = (
            nn.Sequential(nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))
            if stride != 1 or c_in != c_out
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.skip(x))


class Encoder(nn.Module):
    """ResNet-10-ish: stem /4, then 4 stages at /4, /8, /16, /32."""

    def __init__(self, in_channels: int, channels: tuple[int, ...]) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 7, 2, 3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        stages = []
        c_prev = channels[0]
        for i, c in enumerate(channels):
            stride = 1 if i == 0 else 2
            stages.append(nn.Sequential(BasicBlock(c_prev, c, stride), BasicBlock(c, c, 1)))
            c_prev = c
        self.stages = nn.ModuleList(stages)


class ResNet18Encoder(nn.Module):
    """torchvision ResNet-18 split into the same stem + 4 stages as Encoder."""

    def __init__(self, in_channels: int, *, pretrained: bool) -> None:
        super().__init__()
        import torchvision

        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = torchvision.models.resnet18(weights=weights)
        if in_channels != 3:
            net.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.stages = nn.ModuleList([net.layer1, net.layer2, net.layer3, net.layer4])


def build_encoder(backbone: str, in_channels: int, *, pretrained: bool) -> nn.Module:
    if backbone == "resnet18":
        return ResNet18Encoder(in_channels, pretrained=pretrained)
    return Encoder(in_channels, BACKBONE_CHANNELS[backbone])


class Fusion(nn.Module):
    """Self-attention over pooled tokens of every enabled modality."""

    def __init__(self, channels: int, anchors: list[tuple[int, int]], layers: int = 1) -> None:
        super().__init__()
        self.anchors = anchors
        n_tokens = sum(h * w for h, w in anchors)
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, channels))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.speed = nn.Linear(1, channels)
        layer = nn.TransformerEncoderLayer(
            channels,
            nhead=4,
            dim_feedforward=4 * channels,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, feats: list[torch.Tensor], speed: torch.Tensor) -> list[torch.Tensor]:
        tokens = [
            F.adaptive_avg_pool2d(f, a).flatten(2).transpose(1, 2)
            for f, a in zip(feats, self.anchors)
        ]
        x = torch.cat(tokens, dim=1) + self.pos + self.speed(speed).unsqueeze(1)
        x = self.transformer(x)
        out = []
        start = 0
        for f, (h, w) in zip(feats, self.anchors):
            t = x[:, start : start + h * w].transpose(1, 2).reshape(f.shape[0], -1, h, w)
            start += h * w
            out.append(f + F.interpolate(t, size=f.shape[-2:], mode="bilinear", align_corners=False))
        return out


class SegDecoder(nn.Module):
    """Small FPN: stage 4 → stage 1 resolution (/4), then optional upsample."""

    def __init__(self, channels: tuple[int, ...], n_classes: int, upsample: int = 1) -> None:
        super().__init__()
        width = 64
        self.lateral = nn.ModuleList(nn.Conv2d(c, width, 1) for c in channels)
        self.smooth = nn.Sequential(
            nn.Conv2d(width, width, 3, 1, 1), nn.BatchNorm2d(width), nn.ReLU(inplace=True)
        )
        self.head = nn.Conv2d(width, n_classes, 1)
        self.upsample = upsample

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        x = self.lateral[-1](feats[-1])
        for i in range(len(feats) - 2, -1, -1):
            lat = self.lateral[i](feats[i])
            x = lat + F.interpolate(x, size=lat.shape[-2:], mode="bilinear", align_corners=False)
        x = self.head(self.smooth(x))
        if self.upsample > 1:
            x = F.interpolate(x, scale_factor=self.upsample, mode="bilinear", align_corners=False)
        return x


class TransFuserLite(nn.Module):
    def __init__(
        self,
        *,
        modality: str = "both",
        aux: bool = True,
        backbone: str = "resnet18",
        pretrained: bool = False,
    ) -> None:
        """``pretrained`` only matters at construction; loading a checkpoint overwrites it."""
        super().__init__()
        if modality not in MODALITIES:
            raise ValueError(f"modality must be one of {MODALITIES}")
        if backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {BACKBONES}")
        self.modality = modality
        self.aux = aux
        self.backbone = backbone
        self.pretrained = pretrained and backbone == "resnet18"
        self.use_camera = modality in ("both", "camera")
        self.use_lidar = modality in ("both", "lidar")
        channels = BACKBONE_CHANNELS[backbone]

        anchors = []
        if self.use_camera:
            self.image_encoder = build_encoder(backbone, 3, pretrained=self.pretrained)
            anchors.append(IMAGE_ANCHORS)
        if self.use_lidar:
            self.lidar_encoder = build_encoder(backbone, LIDAR_CHANNELS, pretrained=False)
            anchors.append(LIDAR_ANCHORS)
        self.fusions = nn.ModuleList(Fusion(c, anchors) for c in channels)

        self.join = nn.Sequential(
            nn.Linear(channels[-1], 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )
        # Inputs per step: previous waypoint + target points (all /10 m).
        self.gru = nn.GRUCell(2 + 2 * NUM_TARGETS, 64)
        self.delta = nn.Linear(64, 2)

        if aux:
            if self.use_camera:
                self.img_seg = SegDecoder(channels, len(IMG_CLASSES))
            if self.use_lidar:
                # Stage 1 of the 128² BEV is 32²; labels are 64².
                self.bev_seg = SegDecoder(channels, len(BEV_CLASSES), upsample=2)
            self.affordance = nn.Linear(64, 2)

    def forward(
        self,
        image: torch.Tensor | None,
        lidar: torch.Tensor | None,
        speed: torch.Tensor,
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        encoders = []
        feats = []
        if self.use_camera:
            encoders.append(self.image_encoder)
            feats.append(self.image_encoder.stem(image))
        if self.use_lidar:
            encoders.append(self.lidar_encoder)
            feats.append(self.lidar_encoder.stem(lidar))

        per_stage: list[list[torch.Tensor]] = [[] for _ in encoders]
        for s, fusion in enumerate(self.fusions):
            feats = [enc.stages[s](f) for enc, f in zip(encoders, feats)]
            feats = fusion(feats, speed)
            for k, f in enumerate(feats):
                per_stage[k].append(f)

        pooled = sum(F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats)
        z = self.join(pooled)

        h = z
        wp = torch.zeros(z.shape[0], 2, device=z.device, dtype=z.dtype)
        tp = target / 10.0
        waypoints = []
        for _ in range(NUM_WAYPOINTS):
            h = self.gru(torch.cat([wp / 10.0, tp], dim=1), h)
            wp = wp + self.delta(h)
            waypoints.append(wp)
        out = {"waypoints": torch.stack(waypoints, dim=1)}

        if self.aux:
            k = 0
            if self.use_camera:
                out["img_seg"] = self.img_seg(per_stage[k])
                k += 1
            if self.use_lidar:
                out["bev_seg"] = self.bev_seg(per_stage[k])
            out["affordance"] = self.affordance(z)
        return out


def save_checkpoint(path: Path, model: TransFuserLite, **extra) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename, so a crash mid-save can't corrupt the last good file.
    tmp = path.with_suffix(".tmp")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "modality": model.modality,
            "aux": model.aux,
            "backbone": model.backbone,
            "pretrained": model.pretrained,
            **extra,
        },
        tmp,
    )
    tmp.replace(path)


def load_checkpoint(path: Path, device: torch.device) -> tuple[TransFuserLite, dict]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    # Checkpoints from before the backbone option are all "small".
    ckpt.setdefault("backbone", "small")
    ckpt.setdefault("pretrained", False)
    model = TransFuserLite(
        modality=ckpt["modality"], aux=ckpt["aux"], backbone=ckpt["backbone"]
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
