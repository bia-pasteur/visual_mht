"""Code and config to run Visual MHT"""

import dataclasses
from typing import Optional

import torch

import byotrack

from ..linkers import visual_mht


@dataclasses.dataclass
class VisualMHTConfig(visual_mht.VisualMHTLinkerParameters):
    """Configuration for VisualMHT algorithm"""

    def build(
        self, _optflow: Optional[byotrack.OpticalFlow], features: Optional[byotrack.FeaturesExtractor]
    ) -> visual_mht.VisualMHTLinker:
        link = visual_mht.VisualMHTLinker(self, features_extractor=features)
        link.debug = False
        link.device = torch.device("cpu")
        return link
