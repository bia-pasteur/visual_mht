from __future__ import annotations

import dataclasses
from typing import List, Union
import warnings

import torch
from torch import nn

import torch_resnet


BackBone = Union[torch_resnet.ResNet, torch_resnet.PreActResNet]


@dataclasses.dataclass
class ModelConfig:
    """Model configuration

    backbone: A valid klass name in torch_resnet
    width: The width of the resnet
    in_planes: Number of input planes
    out_planes: Number of features to represent the spots
    drop_rate: Dropout rate
    """

    backbone: str
    width: int = 1
    in_planes: int = 2
    out_planes: int = 50
    head_layers: int = 1
    drop_rate: float = 0.0

    def build(self) -> Encoder:
        klass = getattr(torch_resnet, self.backbone)
        backbone = klass(
            shortcut=torch_resnet.ProjectionShortcut,
            in_planes=self.in_planes,
            width=self.width,
            drop_rate=self.drop_rate,
            small_images=True,
            zero_init_residual=True,
        )

        return Encoder(backbone, self.out_planes, self.head_layers, self.drop_rate)


class Encoder(nn.Module):
    """Encoder model which use a BackBone model with a MLP head

    Attributes:
        backbone (torch_resnet.ResNet | torch_resnet.PreActResNet): The backbone resnet to extract features
        head (nn.Module): MLP head that will project the features onto the final output space.

    """

    def __init__(self, backbone: BackBone, out_planes: int, head_layers=0, drop_rate=0.0) -> None:
        super().__init__()
        self.backbone = backbone
        self.out_planes = out_planes

        # MLP Head
        dim = self.backbone.out_planes
        if head_layers == 0 and out_planes != dim:
            warnings.warn(f"Without any head layers, unable to project features from {dim} to {out_planes}")

        head: List[torch.nn.Module] = []

        for _ in range(head_layers - 1):
            head.append(nn.Dropout(drop_rate))
            head.append(torch.nn.Linear(dim, dim // 2))
            head.append(torch.nn.BatchNorm1d(dim // 2))
            head.append(torch.nn.ReLU())
            dim = dim // 2

        if head_layers > 0:
            head.append(nn.Dropout(drop_rate))
            head.append(torch.nn.Linear(dim, out_planes))

        self.head = torch.nn.Sequential(*head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))
