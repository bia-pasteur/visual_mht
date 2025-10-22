from __future__ import annotations

import dataclasses
from typing import Tuple

import torch.nn
import torch.optim

# Supported optimizers
OPTIMIZERS = {klass.__name__: klass for klass in [torch.optim.Adam, torch.optim.AdamW, torch.optim.SGD]}


@dataclasses.dataclass
class AdamConfig:
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8
    weight_decay: float = 0.0
    amsgrad: bool = False


@dataclasses.dataclass
class AdamWConfig:
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1.0e-8
    weight_decay: float = 0.01
    amsgrad: bool = False


@dataclasses.dataclass
class SGDConfig:
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = 0.0
    nesterov: bool = False


@dataclasses.dataclass
class OptimizerConfig:
    name: str
    lr: float
    Adam: AdamConfig = dataclasses.field(default_factory=AdamConfig)
    AdamW: AdamWConfig = dataclasses.field(default_factory=AdamWConfig)
    SGD: SGDConfig = dataclasses.field(default_factory=SGDConfig)

    def build(self, model: torch.nn.Module) -> torch.optim.Optimizer:
        if self.name not in OPTIMIZERS:
            raise ValueError(f"Unknown optimizer {self.name}. Expected: {OPTIMIZERS.keys()}")

        kwargs = dataclasses.asdict(getattr(self, self.name))
        kwargs["lr"] = self.lr

        constructor = OPTIMIZERS[self.name]

        if kwargs.get("weight_decay", 0) > 0:
            no_decay = ["bias", "bn"]  # No decay for biases and BN layers

            grouped_parameters = [
                {
                    "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": kwargs.pop("weight_decay"),
                },
                {
                    "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            return constructor(grouped_parameters, **kwargs)

        return constructor(model.parameters(), **kwargs)
