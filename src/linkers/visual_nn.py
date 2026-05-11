import dataclasses
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

import byotrack
from byotrack.implementation.linker.frame_by_frame import nearest_neighbor


@dataclasses.dataclass
class VisualNNParameters(nearest_neighbor.NearestNeighborParameters):
    """Parameters of VisualNNLinker

    Attributes:
        association_threshold (float): This is the main hyperparameter, it defines the threshold on the distance used
            not to link tracks with detections. It prevents to link with false positive detections.
        position_gate (float): Threshold on the positional distance (in pixels)
            Default: inf  (No gate)
        features_gate (float): Threshold on the features distance (in features distance)
            Default: inf  (No gate)
        alpha (float): Merge the two distances with a proportion lambda
            Default: 1.0 (Only features based)
        n_valid (int): Number of frames with a correct association required to validate the track at its creation.
            Default: 3
        n_gap (int): Number of frames with no association before the track termination.
            Default: 3
        association_method (AssociationMethod): The frame-by-frame association to use. See `AssociationMethod`.
            It can be provided as a string. (Choice: GREEDY, OPT_HARD, OPT_SMOOTH)
            Default: OPT_SMOOTH
        ema (float): Optional exponential moving average to reduce detection noise. Detection positions are smoothed
            using this EMA. Should be smaller than 1. It use: x_{t+1} = ema x_{t} + (1 - ema) det(t)
            As motion is not modeled, EMA may introduce lag that will hinder tracking. It is more effective with
            optical flow to compensate motions, in this case, a typical value is 0.5, to average the previous position
            with the current measured one. For more advanced modelisation, see `KalmanLinker`.
            Default: 0.0 (No EMA)
        features_ema (float): Optional exponential moving average on the features to reduce features extraction noise.
            Default: 0.0 (No EMA)
        fill_gap (bool): Fill the gap of missed detections using a forward optical flow
            propagation (Only when optical flow is provided). We advise to rather use a
            ForwardBackward interpolation using the same optical flow: it will produce
            smoother interpolations.
            Default: False

    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        association_threshold: float,
        *,
        position_gate=float("inf"),
        features_gate=float("inf"),
        alpha=1.0,
        n_valid=3,
        n_gap=3,
        association_method: Union[
            str, nearest_neighbor.AssociationMethod
        ] = nearest_neighbor.AssociationMethod.SPARSE_OPT_SMOOTH,
        ema=0.0,
        features_ema=0.0,
        fill_gap=False,
    ):
        super().__init__(
            association_threshold,
            n_valid=n_valid,
            n_gap=n_gap,
            association_method=association_method,
            ema=ema,
            fill_gap=fill_gap,
        )
        self.position_gate = position_gate
        self.features_gate = features_gate
        self.alpha = alpha
        self.features_ema = features_ema

    position_gate: float = float("inf")
    features_gate: float = float("inf")
    alpha: float = 1.0
    features_ema: float = 0.0


class VisualNNLinker(nearest_neighbor.NearestNeighborLinker):
    """Frame by frame linker by associating with the closest detections

    Motion is not modeled, but if an optical flow method is provided, it
    will be used to compensate motion online. Matching is done using a merged distance between positions
    and features.

    See `NearestNeighborLinker` for the other attributes.

    Attributes:
        specs (FeaturesNNParameters): Parameters specifications of the algorithm.
            See `FeaturesNNParameters`.
        active_features (Optional[torch.Tensor]): The features of actives tracks.
            Shape: (N, D), dtype: float32

    """

    progress_bar_description = "Nearest Neighbor linking"

    def __init__(
        self,
        specs: nearest_neighbor.NearestNeighborParameters,
        optflow: Optional[byotrack.OpticalFlow] = None,
        features_extractor: Optional[byotrack.FeaturesExtractor] = None,
        save_all=False,
    ) -> None:
        super().__init__(specs, optflow, features_extractor, save_all)
        self.specs: VisualNNParameters
        self.active_features = torch.zeros(0, 1)
        self.all_masses: List[torch.Tensor] = []

    def reset(self, dim=2) -> None:
        super().reset(dim)
        self.active_features = torch.zeros(0, 1)

    def cost(self, _: np.ndarray, detections: byotrack.Detections) -> Tuple[torch.Tensor, float]:
        if self.active_positions is None:
            self.active_positions = torch.empty((0, detections.position.shape[1]))

        dist = torch.cdist(self.active_positions, detections.position)
        unfeasible = dist > self.specs.position_gate

        if "features" in detections.data:
            device = detections.data["features"].device
            if self.active_features.shape[0] == 0:
                self.active_features = torch.empty((0, detections.data["features"].shape[1]), device=device)

            feat = torch.cdist(self.active_features, detections.data["features"]).cpu()
            unfeasible |= feat > self.specs.features_gate

            if self.specs.alpha == 1.0:
                dist = feat
            elif self.specs.alpha != 0.0:
                feat *= self.specs.alpha
                dist *= 1 - self.specs.alpha
                dist += feat

        dist[unfeasible] = torch.inf

        return dist, self.specs.association_threshold

    def post_association(self, _: np.ndarray, detections: byotrack.Detections, active_mask: torch.Tensor):
        super().post_association(_, detections, active_mask)

        if "features" in detections.data:  # Handle features EMA and concatenation, like for positions
            device = detections.data["features"].device
            if self.active_features.shape[0] == 0:
                self.active_features = torch.empty((0, detections.data["features"].shape[1]), device=device)

            # EMA
            self.active_features[self._links[:, 0]] -= (1.0 - self.specs.features_ema) * (
                self.active_features[self._links[:, 0]] - detections.data["features"][self._links[:, 1]]
            )

            # Merge with newly created tracks
            self.active_features = torch.cat(
                (self.active_features[active_mask], detections.data["features"][self._unmatched_detections])
            )
