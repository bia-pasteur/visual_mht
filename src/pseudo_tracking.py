import dataclasses
from typing import Collection, Sequence, Union

import numpy as np

import byotrack
from byotrack.implementation.linker.frame_by_frame.nearest_neighbor import (
    NearestNeighborLinker,
    NearestNeighborParameters,
)


@dataclasses.dataclass
class PseudoTrackerConfig:
    """Code for pseudo tracking, with very few identity switches, but potentially high fragmentation"""

    association_threshold: float = 3.0  # Let's be very restrictive
    n_valid: int = 2  # Only keep tracks of length >= 2
    n_gap: int = 0  # Don't try to close gaps (too dangerous)

    def build(self) -> byotrack.Linker:
        return NearestNeighborLinker(
            NearestNeighborParameters(
                self.association_threshold,
                n_valid=self.n_valid,
                n_gap=self.n_gap,
                association_method="sparse_opt_smooth",
            )
        )

    def run(
        self, video: Union[np.ndarray, Sequence[np.ndarray]], detections_sequence: Sequence[byotrack.Detections]
    ) -> Collection[byotrack.Track]:
        return self.build().run(video, detections_sequence)
