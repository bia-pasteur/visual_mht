"""Code and config to run MHT"""

import dataclasses
from typing import Optional

import torch

import byotrack

from ..linkers import bskt


@dataclasses.dataclass
class BSKTConfig(bskt.KalmanMHTLinkerParameters):
    """Configuration for BSKT algorithm"""

    def build(self, _optflow: Optional[byotrack.OpticalFlow], _features=None) -> bskt.KalmanMHTLinker:
        link = bskt.KalmanMHTLinker(self)
        link.debug = False
        link.device = torch.device("cpu")
        return link
