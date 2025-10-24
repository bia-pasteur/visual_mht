"""Code and config to run Visual NN"""

import dataclasses
from typing import Optional

import byotrack

from ..linkers import visual_nn


@dataclasses.dataclass
class VisualNNConfig(visual_nn.VisualNNParameters):
    """Configuration for VisualNN algorithm"""

    def build(
        self, optflow: Optional[byotrack.OpticalFlow], features: Optional[byotrack.FeaturesExtractor]
    ) -> visual_nn.VisualNNLinker:
        return visual_nn.VisualNNLinker(self, optflow, features)
