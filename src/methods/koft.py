"""Code and config to run koft"""

import dataclasses

import byotrack
from byotrack.implementation.linker.frame_by_frame import koft


@dataclasses.dataclass
class KOFTConfig(koft.KOFTLinkerParameters):
    """Configuration for KOFT algorithm"""

    def build(self, optflow: byotrack.OpticalFlow, _features=None) -> koft.KOFTLinker:
        return koft.KOFTLinker(self, optflow)
