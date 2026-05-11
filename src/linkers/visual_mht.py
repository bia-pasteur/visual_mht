import dataclasses
from typing import Optional, Union

import torch

import byotrack

from .bskt import KalmanMHTLinker, KalmanMHTLinkerParameters, Cost, Solver, TrackBuilding


def _hypothesize(
    hypotheses_features: torch.Tensor,
    features: torch.Tensor,
    links: torch.Tensor,
) -> torch.Tensor:
    """Generate features hypotheses"""
    n_link, n_h, n_det = len(links), len(hypotheses_features), len(features)
    depth = hypotheses_features.shape[1]
    dim = features.shape[-1]

    new_features = torch.zeros((n_link + n_h + n_det, depth, dim), dtype=features.dtype, device=features.device)

    # Simply propagate linked and non-linked hypothesis
    new_features[:n_link] = hypotheses_features[links[:, 0]]
    new_features[n_link : n_link + n_h] = hypotheses_features

    # And fill starting hypotheses
    new_features[n_link + n_h :, -1] = features

    return new_features


@dataclasses.dataclass
class VisualMHTLinkerParameters(KalmanMHTLinkerParameters):
    """Parameters of the VisualMHT Linker.

    Attributes:
        association_threshold (float): Threshold on the linking cost. Linking hypotheses are not made
            above this threshold. This allows to reduce the exponential growth of the hypotheses tree. By default (-1),
            it is set to the theoretical value dr / [(1-dr)(1-fnr)] * lambda_b/V that corresponds to the limit
            above which it is preferable to recreate a track instead of linking. It can be reduced to its own value.
            With a `cost` EUCLIDEAN or EUCLIDEAN_SQ, it corresponds to the highest acceptable normalized Euclidean
            distance (||(x_1 - x_2)/ std||_2).
            Otherwise, it is the lowest acceptable probability (See `Cost`).
            Default: -1.0
        detection_std (Union[float, torch.Tensor]): Expected measurement noise on the detection process.
            The detection process is modeled with a Gaussian noise with this given std. (You can provide a different
            noise for each dimension). See `torch_kf.ckf.constant_kalman_filter`.
            Default: 3.0 pixels
        process_std (Union[float, torch.Tensor]): Expected process noise. See `torch_kf.ckf.constant_kalman_filter`, the
            process is modeled as constant order-th derivative motion. This quantify how much the supposely "constant"
            order-th derivative can change between two consecutive frames. A common rule of thumb is to use
            3 * process_std ~= max_t(| x^(order)(t) - x^(order)(t+1)|). It can be provided for each dimension).
            Default: 1.5 pixels / frame^order
        kalman_order (int): Order of the Kalman filter to use.
            0 for brownian motions, 1 for directed brownian motions, 2 for accelerated brownian motions, etc...
            Default: 1
        fpr (float): False positive rate of the detection process. Increasing this parameter will make the linker
            more careful when building tracks, choosing to track only the most plausible ones.
            Given n_det detections, we expect fpr * n_det false alarms and (1 - fpr) * n_det true objects.
            We have by definition fpr = 1 - precision: It can be estimated from few detection annotations.
            This allows an easy estimation of lambda_f = fpr * n_det.
            Default: 0.1
        fnr (float): False negative rate of the detection process. Increasing this parameter, will increase
            its robustness to missing detections, which may hinder tracking if missing detections are not common.
            Given n expected targets, we expect to detect only (1 - fnr) n of them. By definition, fnr = 1 - recall:
            It can be estimated from few detection annotations.
            Default: 0.1
        birth_rate (float): Rate of birth of the observed targets. Increase to start more tracks inside
            the sequence (after t > 0). Given n(t) targets at time t, we expect lambda_b = br * n(t) to be new targets.
            Given the death rate dr, we have the following expectation: (1 - dr) n(t - 1) = (1 - br) n(t).
            Having br > dr will bias targets to be more numerous at the end than at the beginning.
            Default: 1e-2
        death_rate (float): Rate of death of the observed targets. Increase to terminate tracks quicklier.
            Given n(t) targets at time t, we expect dr * n(t) to disappear on this frame.
            Given the birth rate br, we have the following expectation: (1 - dr) n(t) = (1 - br) n(t + 1).
            Having dr > br will bias targets to be less numerous at the end than at the beginning.
            Default: 1e-2
        tree_depth (int): Depth d of the hypotheses tree. It consists of the frame t on which tracks are fixed
            and d - 1 (= \\tau_{depth}) following frames with detections and their associated hypotheses.
            The minimal value is therefore d = 2. Increasing it, will usually increase performances, but will require
            exponential computations. This implementation supports up to d=6 for around 1000 targets.
            Default: 4
        step (int): Step size after an optimizaton. Given confirmed tracks on frame t, we solve a MAP of tree_depth
            frames (t, t+1, ..., t + tree_depth -1), then the tracks are updated to frame t + step. For large
            tree_depth, the optimization is the bottleneck. Using step > 1 reduces computational time, but comes
            with frames solved with smaller depths than tree_depth.
            Default: 1
        n_det (float): Expected constant n_det. If provided, it is used to estimate lambda_b and lambda_f.
            Otherwise, the observed n_det on each frame is used instead.
            Default: -1.0
        volume (float): Number of pixels considered in the images. If provided, it is used to defined
            p(z | None) = 1 / V. Otherwise the true volume of the images is used. Useful, if targets are clustered in
            a small ROI inside the full volume.
            Default: -1.0
        n_0 (float): Expected number of target before the starting point of the video. If not given, it will be
            estimated from the first observed n_det.
            Default: -1.0
        solver (Solver): Solver of the MAP problem. See `Solver`. For d=2, you can use and optimal and faster
            LAP solver. Otherwise, mixed integer linear programming is used with SCIPY (HIGHS) or GUROBI.
            Currently, GUROBI is much slower because of the creation time of the problem (Gurobi python api
            is pretty slow...)
            Default: SCIPY
        track_building (TrackBuilding): Method to build tracks. See `collect`. With DETECTION,
            the position of the detection is used, when a detection is missed, NaN is used. FILTERED
            will the use the Kalman filtered position on each frame. And SMOOTHED implements the RTS
            optimal Kalman smoothing in `collect`.
            Default: DETECTION
        cost (Cost): Linking cost method, see `Cost`. The detection position is distributed either following the Kalman
            filter likelihood model (MODEL), or a LAPLACE/GAUSSIAN distribution with a constant standard
            deviation `std`. EUCLIDEAN (resp. EUCLIDEAN_SQ), refers to LAPLACE (resp. GAUSSIAN) distribution with
            dist-based thresholding (see `association_threshold`).
            Default: EUCLIDEAN
        std (Union[float, torch.Tensor]): Standard deviation of the Laplace or Gaussian distribution.
            Typically characterize the precision of the motion_model.
            Shape: (1, ) or (dim, ), dtype: float
            Default: 1.0
        model_cutoff (float): With Model `Cost`, in particular for standard Kalman filters, the likelihood threshold
            is very permissive for missed-tracks (as the covariance increases rapidly). If given this enforces
            that any association respect the additional constraints: |.|_2 / std < model_cutoff.
            It is equivalent to `association_threshold` for other Costs, and therefore, will not be used.
            Default: -1.0 (Disabled)
        feature_cutoff (float): Cutoff distance in the feature space.
            Default: -1.0 (Disabled)
        feature_ema (float): Exponential moving average used for smoothing features temporally.
            Default: 0.0 (No EMA)
        feature_std (float): Expected standard deviation in the feature space. TODO: Compute on annotated track(s)?
            Default: 1.0
        drop_trailing (bool): Drop the last missing detections of tracks that are still more probable to be active
            than finished on the last frame.
            Default: True
        false_tracks (bool): If True, we allow the creation of false positive tracks, that can be linked even
            if the track seems to best false. This enable track birth if birth_rate << fpr with small depths
            (as otherwise, all births are considered as false positives with small depth)
            Default: True
        initial_std_factor (float): The uncertainties on initial velocities/accelerations are set
            to initial_std_factor * process_std. See `KalmanLinker.build_initial_covariance`.
            Having a small factor will prevent handling correctly starting tracks with large initial velocity
            on their first frames. But large values will lead to large uncertainty on the first prediction, making
            it hard to associate to a detection with MAHALANOBIS or LIKELIHOOD methods.
            Typical values lies between 3.0 to 10.0.
            Default: 10.0

    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        association_threshold: float = -1.0,
        detection_std: Union[float, torch.Tensor] = 3.0,
        process_std: Union[float, torch.Tensor] = 1.5,
        kalman_order: int = 1,
        fpr: float = 0.1,
        fnr: float = 0.1,
        birth_rate: float = 1e-2,
        death_rate: float = 1e-2,
        tree_depth: int = 4,
        step: int = 1,
        n_det: float = -1.0,
        volume: float = -1.0,
        n_0: float = -1.0,
        solver: Union[str, Solver] = Solver.SCIPY,
        track_building: Union[str, TrackBuilding] = TrackBuilding.DETECTION,
        cost: Union[str, Cost] = Cost.EUCLIDEAN,
        std: Union[float, torch.Tensor] = 1.0,
        model_cutoff: float = -1.0,
        features_cutoff: float = -1.0,
        features_ema: float = 0.0,
        features_std: float = 1.0,
        drop_trailing: bool = True,
        false_tracks: bool = True,
        initial_std_factor: float = 10.0,
    ):
        super().__init__(
            association_threshold=association_threshold,
            detection_std=detection_std,
            process_std=process_std,
            kalman_order=kalman_order,
            fpr=fpr,
            fnr=fnr,
            birth_rate=birth_rate,
            death_rate=death_rate,
            tree_depth=tree_depth,
            step=step,
            n_det=n_det,
            volume=volume,
            n_0=n_0,
            solver=solver,
            track_building=track_building,
            cost=cost,
            std=std,
            model_cutoff=model_cutoff,
            drop_trailing=drop_trailing,
            false_tracks=false_tracks,
            initial_std_factor=initial_std_factor,
        )

        self.features_cutoff = features_cutoff
        self.features_ema = features_ema
        self.features_std = features_std

    features_cutoff: float = 3.0
    features_ema: float = 0.0
    features_std: float = 1.0


class VisualMHTLinker(KalmanMHTLinker):
    """Visual MHT Linking.

    It extends the KalmanMHTLinker with visual features, used for hypotheses pruning and associations costs.
    """

    progress_bar_description = "Visual MHT linking"

    def __init__(
        self,
        specs: VisualMHTLinkerParameters,
        optflow: Optional[byotrack.OpticalFlow] = None,
        features_extractor: Optional[byotrack.FeaturesExtractor] = None,
        save_all=False,
    ) -> None:
        super().__init__(specs, optflow, features_extractor, save_all)
        assert self.features_extractor is not None
        self.features_extractor: byotrack.FeaturesExtractor
        self.specs: VisualMHTLinkerParameters

        self.hypotheses_features = torch.zeros((0, 1, 0), dtype=torch.float32)

    def reset(self, dim=2):
        super().reset(dim)

        self.hypotheses_features = torch.zeros((0, 1, 0), dtype=torch.float32)

    def linking_cost(self, frame, detections):
        indices, costs = super().linking_cost(frame, detections)

        if indices.shape[0] == 0:
            return indices, costs

        if self.specs.features_std <= 0.0 and self.specs.features_cutoff <= 0.0:
            return indices, costs

        feat_costs = (
            (
                self.hypotheses_features[:, -1][indices[:, 0]]
                - detections.data["features"].to(self.device)[indices[:, 1]]
            )
            .pow(2)
            .sum(dim=-1)
        )

        if self.specs.features_cutoff > 0.0:
            valid = feat_costs < self.specs.features_cutoff**2
            indices = indices[valid]
            costs = costs[valid]
            feat_costs = feat_costs[valid]

        if self.specs.features_std > 0.0:
            dim = self.hypotheses_features.shape[-1]

            # Uniform cost for false positive/birth
            uniform_cost = dim * (
                torch.log(torch.tensor(self.specs.features_std)) + 0.5 * torch.log(2 * torch.tensor(torch.pi))
            ) + torch.log(torch.tensor(2 * self._n_0_est))

            # Gaussian cost on the features
            normalization_cost = dim * (
                torch.log(torch.tensor(self.specs.features_std)) + 0.5 * torch.log(2 * torch.tensor(torch.pi))
            )

            feat_costs = 0.5 / self.specs.features_std**2 * feat_costs
            feat_costs += normalization_cost - uniform_cost

            return indices, costs + feat_costs

        return indices, costs

    def max_likelihood(self):
        if self.specs.features_std <= 0:
            return super().max_likelihood()

        dim = self.hypotheses_features.shape[-1]
        uniform_cost = dim * (
            torch.log(torch.tensor(self.specs.features_std)) + 0.5 * torch.log(2 * torch.tensor(torch.pi))
        ) + torch.log(torch.tensor(2 * self._n_0_est))

        # Gaussian cost on the features
        normalization_cost = dim * (
            torch.log(torch.tensor(self.specs.features_std)) + 0.5 * torch.log(2 * torch.tensor(torch.pi))
        )

        return normalization_cost - uniform_cost + super().max_likelihood()

    def motion_model(self) -> None:
        super().motion_model()
        # Simply propagate hypotheses features
        next_features = self.hypotheses_features[:, -1]

        self.hypotheses_features = torch.cat((self.hypotheses_features, next_features[:, None]), dim=1)

    def hypothesize_states(self, links: torch.Tensor):
        super().hypothesize_states(links)

        features = self.last_detections[-1].data["features"].to(self.device)

        if self.hypotheses_features.shape[-1] == 0:  # Set features_dim
            assert self.frame_id == 0 and self.hypotheses_features.shape[0] == 0

            self.hypotheses_features = torch.zeros(
                (0, self.hypotheses_features.shape[1], features.shape[-1]), dtype=self.dtype, device=self.device
            )

        self.hypotheses_features = _hypothesize(self.hypotheses_features, features, links)

    def update_states(self, kept):
        super().update_states(kept)

        # Update states and positions for detected particles
        n_non_start = len(kept) - len(self.last_detections[-1])
        updated = kept[:n_non_start] & (self.hypotheses_indices[:n_non_start, -1] != -1)

        self.hypotheses_features[:n_non_start, -1][updated] -= (1.0 - self.specs.features_ema) * (
            self.hypotheses_features[:n_non_start, -1][updated]
            - self.last_detections[-1]
            .data["features"]
            .to(self.device)[self.hypotheses_indices[:n_non_start, -1][updated]]
        )

    def filter_states(self, kept):
        super().filter_states(kept)

        self.hypotheses_features = self.hypotheses_features[:, 1:][kept]
