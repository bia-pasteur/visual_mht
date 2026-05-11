import dataclasses
from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch_kf
import torch_kf.ckf

import byotrack


from .base import (
    _fast_euclidean_sq_cdist,
    Cost,
    Solver,
    TrackBuilding,
    TrackHandler,
    MHTLinker,
    MHTLinkerParameters,
)


def _hypothesize(
    states: torch_kf.GaussianState,
    links: torch.Tensor,
    *,
    positions: torch.Tensor,
    initial_covariance: torch.Tensor,
    linked_only: bool = False,
) -> torch_kf.GaussianState:
    """Build new hypotheses internal states. Called in `hypothesize`."""
    n_h, n_link, n_det = len(states.mean), len(links), len(positions)
    depth = states.mean.shape[1]
    device = states.mean.device
    x_dim = states.mean.shape[2]
    z_dim = positions.shape[-1]

    new_states = torch_kf.GaussianState(
        torch.zeros((n_link + n_h + n_det, depth, x_dim, 1), dtype=states.mean.dtype, device=device),
        torch.zeros((n_link + n_h + n_det, depth, x_dim, x_dim), dtype=states.covariance.dtype, device=device),
    )

    # Propagate linked hypotheses
    new_states.mean[:n_link] = states.mean[links[:, 0]]
    new_states.covariance[:n_link] = states.covariance[links[:, 0]]

    if linked_only:
        return new_states

    # Then non-linked hypotheses
    new_states.mean[n_link : n_link + n_h] = states.mean
    new_states.covariance[n_link : n_link + n_h] = states.covariance

    # And fill starting hypotheses (already filled with 0)
    new_states.mean[n_link + n_h :, -1, :z_dim, 0] = positions
    new_states.covariance[n_link + n_h :, -1] = initial_covariance

    return new_states


def fast_likelihood_cost(
    projections: torch_kf.GaussianState,
    positions: torch.Tensor,
    association_threshold: float,
    hypotheses_states: torch.Tensor,
    cutoff: Union[float, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute log likelihood cost between projected hypotheses states and given detections positions

    It returns the sparse cost matrix between hypotheses and detections.

    Args:
        projections (torch_kf.GaussianState): Hypotheses projected states
            Shape: mean=(n_h, dim, 1), covariance=(n_h, dim, dim), precision=(n_h, dim, dim)
            dtype: float
        positions (torch.Tensor): Detections positions
            Shape: (n_det, dim), dtype: float
        association_threshold (float): Do not consider association below this probability threshold.
        hypotheses_states (torch.Tensor): State of each hypotheses (0: Active, 1: Finished, 2: Invalid).
            We do not compute linking cost for non-active states.
            Shape: (n_h,), dtype: int
        cutoff (Union[float, torch.Tensor]): Additional cutoff based on Euclidean distance
            such that |(h_pos - pos) / cutoff|_2 < 1.0 for any association.
            Shape: (dim, ) or (1, ), dtype: float

    Returns:
        torch.Tensor: Indices (i, j) of potential links between active hypothesis i and detection j
            Shape: (n_link, 2), dtype: int32
        torch.Tensor: The linking cost for each potential link
            Shape: (n_link,), dtype: float

    """
    # Filter out pruned hypotheses
    valid = hypotheses_states == 0
    mapping = torch.arange(len(hypotheses_states), device=positions.device)[valid]
    projections = projections[valid]

    if len(projections.mean) == 0:  # Not a single valid hypothesis. Let's return empty links
        indices = torch.zeros((0, 2), dtype=torch.int32, device=positions.device)
        costs = torch.zeros((0,), dtype=positions.dtype, device=positions.device)

        return indices, costs

    stds = projections.covariance.diagonal(dim1=1, dim2=2).sqrt()
    ratios = stds / stds[:, -2:-1]

    # Fast if diagonal cov and shared ratio (Usually the case)
    fast = torch.allclose(projections.covariance.abs().sqrt().sum(dim=(1, 2)), stds.sum(dim=1))
    fast &= torch.allclose(ratios, ratios[:1])

    if fast:
        # Let's go through a much faster MAHA_SQ computations exploiting the structure of the covariance
        dist = _fast_euclidean_sq_cdist(projections.mean[..., 0] / ratios[:1], positions / ratios[:1])
        exp_factor = 2 * stds[:, -2] ** 2  # prob = C exp(-dist/exp_factor)
        # In that case, MAHA_SQ = dist / stds[:, -2:-1]**2
    else:
        dist = projections[:, None].mahalanobis_squared(positions[None, ..., None])
        exp_factor = torch.tensor(2.0, dtype=positions.dtype, device=positions.device)

    # prob = C * exp(-dist/exp_factor) > thresh
    # -log(C) + dist / exp_factor < -log(thresh)
    # dist < (-log(thresh) + log(C)) * exp_factor
    normalization_costs = 0.5 * torch.log(torch.det(projections.covariance))
    normalization_costs += 0.5 * projections.covariance.shape[-1] * torch.log(2 * torch.tensor(torch.pi))
    association_thresholds = -torch.log(torch.tensor(association_threshold)) - normalization_costs
    association_thresholds *= exp_factor

    indices = torch.nonzero(dist < association_thresholds[:, None]).to(torch.int32)
    costs = dist[indices[:, 0], indices[:, 1]]

    cutoff = torch.broadcast_to(
        torch.as_tensor(cutoff, dtype=positions.dtype, device=positions.device), (positions.shape[-1],)
    )
    if (cutoff > 0.0).all():  # Additional cutoff based on Euclidean distance
        valid = ((projections.mean[indices[:, 0], :, 0] - positions[indices[:, 1]]) / cutoff).pow_(2).sum(dim=-1) < 1.0
        indices = indices[valid]
        costs = costs[valid]

    if fast:
        costs /= exp_factor[indices[:, 0]]
    else:
        costs /= exp_factor

    costs += normalization_costs[indices[:, 0]]

    # Remap indices to take into account filtered hypotheses
    indices[:, 0] = mapping[indices[:, 0]]

    return indices, costs


@dataclasses.dataclass
class KalmanMHTLinkerParameters(MHTLinkerParameters):  # pylint: disable=too-many-instance-attributes
    """Parameters of the KalmanMHTLinker

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
        drop_trailing: bool = True,
        false_tracks: bool = True,
        initial_std_factor: float = 10.0,
    ):
        super().__init__(
            association_threshold=association_threshold,
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
            drop_trailing=drop_trailing,
            false_tracks=false_tracks,
        )

        self.detection_std = detection_std
        self.process_std = process_std
        self.kalman_order = kalman_order
        self.initial_std_factor = initial_std_factor
        self.model_cutoff = model_cutoff

        if self.model_cutoff > 0.0 and self.cost is not Cost.MODEL:
            warnings.warn(f"The model_cutoff ({model_cutoff}) given will be ignored for non MODEL Costs ({cost}).")

    detection_std: Union[float, torch.Tensor] = 3.0
    process_std: Union[float, torch.Tensor] = 1.5
    kalman_order: int = 1
    initial_std_factor: float = 10.0
    model_cutoff: float = -1.0


class KalmanMHTLinker(MHTLinker):
    """MHT linker using Kalman filters

    Motion is modeled with a Kalman filter of a specified order (See `torch_kf.ckf`)
    If optical flow is provided, it is used online to warp the predicted state positions
    of the kalman filter. This will work, but it is sub-optimal: consider using `KOFTLinker`
    that exploits in a finer way optical flow inside Kalman filters.

    Note:
        This implementation requires torch-kf. (pip install torch-kf)

    See `MHTLinker` for the other attributes.

    Attributes:
        specs (KalmanMHTLinkerParameters): Parameters specifications of the algorithm.
            See `KalmanMHTLinkerParameters`.
        kalman_filter (torch_kf.KalmanFilter): The Kalman filter that models tracks data.
        kf_states (torch_kf.GaussianState): The Kalman filter estimation for each hypothesis.
            Shape: mean=(n_h, depth, dim * (ord + 1), 1), covariance=(n_h, depth, dim * (ord + 1), dim * (ord + 1))
            dtype: float32
        projections (torch_kf.GaussianState): The Kalman filter projection for each hypothesis.
            Shape: mean=(n_h, dim, 1), covariance=(n_h, dim, dim), precision=(n_h, dim, dim)
            dtype: float32  # XXX: Not used any longer
        all_states (List[torch_kf.GaussianState]): The Kalman filter estimation for each TrackHandler at each seen
            frame. States are only registered when save_all=True or if you build tracks from RTS smoothing.
            Shape: mean=(n, dim * (ord + 1), 1), covariance=(n, dim * (ord + 1), dim * (ord + 1))
            dtype: float32

    """

    progress_bar_description = "Kalman filter MHT linking"

    def __init__(
        self,
        specs: KalmanMHTLinkerParameters,
        optflow: Optional[byotrack.OpticalFlow] = None,
        features_extractor: Optional[byotrack.FeaturesExtractor] = None,
        save_all=False,
    ) -> None:
        super().__init__(specs, optflow, features_extractor, save_all)

        self.specs: KalmanMHTLinkerParameters
        self.kalman_filter = (
            torch_kf.ckf.constant_kalman_filter(
                0.0,  # Initialized with dummy values as we do not know the dim
                0.0,
                dim=2,
                order=self.specs.kalman_order,
            )
            .to(self.dtype)
            .to(self.device)
        )

        self.kf_states = torch_kf.GaussianState(
            torch.empty((0, 1, self.kalman_filter.state_dim, 1), dtype=self.dtype, device=self.device),
            torch.empty(
                (0, 1, self.kalman_filter.state_dim, self.kalman_filter.state_dim), dtype=self.dtype, device=self.device
            ),
        )
        # self.projections = self.kalman_filter.project(self.states[:, -1:])

        self.all_states: List[torch_kf.GaussianState] = []

    def reset(self, dim=2) -> None:
        super().reset(dim)

        self.kalman_filter = (
            torch_kf.ckf.constant_kalman_filter(
                self.specs.detection_std,
                self.specs.process_std,
                dim=dim,
                order=self.specs.kalman_order,
            )
            .to(self.dtype)
            .to(self.device)
        )

        self.kf_states = torch_kf.GaussianState(
            torch.empty((0, 1, self.kalman_filter.state_dim, 1), dtype=self.dtype, device=self.device),
            torch.empty(
                (0, 1, self.kalman_filter.state_dim, self.kalman_filter.state_dim), dtype=self.dtype, device=self.device
            ),
        )
        # self.projections = self.kalman_filter.project(self.states[:, -1:])
        self.all_states = []

    def collect(self) -> List[byotrack.Track]:  # pylint: disable=too-many-locals
        if self.specs.track_building != TrackBuilding.SMOOTHED:
            return super().collect()

        # We need to solve the association up to the end
        self.associate()

        # First let's gather tracks states data
        handlers: List[TrackHandler] = []
        states_l: List[torch_kf.GaussianState] = []
        det_ids: List[torch.Tensor] = []

        # From terminatated tracks
        for handler in self.inactive_tracks:
            if handler.track_state is TrackHandler.TrackState.INVALID:
                continue  # Ignore non-valid tracks

            states_l.append(
                torch_kf.GaussianState(
                    torch.cat(
                        [
                            states_[track_id : track_id + 1].mean
                            for track_id, states_ in zip(
                                handler.track_ids[: len(handler)],
                                self.all_states[handler.start :],
                            )
                        ]
                    ),
                    torch.cat(
                        [
                            states_[track_id : track_id + 1].covariance
                            for track_id, states_ in zip(
                                handler.track_ids[: len(handler)],
                                self.all_states[handler.start :],
                            )
                        ]
                    ),
                )
            )
            handlers.append(handler)
            det_ids.append(torch.tensor(handler.detection_ids[: len(handler)], dtype=torch.int32))

        # For active tracks, we rely on the selected hypotheses
        hypotheses = torch.arange(len(self.hypotheses_indices), device=self.device)[self.selected]
        for hypothesis in hypotheses:
            state = int(
                self.hypotheses_costs[hypothesis].argmin().item()
            )  # Can be different from hypotheses_states[-1]
            if state == TrackHandler.TrackState.INVALID:
                continue  # More likely to be False Positive, let's ignore it

            i = int(self.hypotheses_indices[hypothesis, 0].item())

            if i != -1:
                handler = self.active_tracks[i]
                det_ids.append(
                    torch.cat(
                        [
                            torch.tensor(handler.detection_ids, dtype=torch.int32),
                            self.hypotheses_indices[hypothesis, 1:].cpu(),
                        ]
                    )
                )

                states_l.append(
                    torch_kf.GaussianState(
                        torch.cat(
                            [
                                states_[track_id : track_id + 1].mean
                                for track_id, states_ in zip(
                                    handler.track_ids[: len(handler)],
                                    self.all_states[handler.start :],
                                )
                            ]
                            + [self.kf_states[hypothesis, 1:].mean.cpu()]
                        ),
                        torch.cat(
                            [
                                states_[track_id : track_id + 1].covariance
                                for track_id, states_ in zip(
                                    handler.track_ids[: len(handler)],
                                    self.all_states[handler.start :],
                                )
                            ]
                            + [self.kf_states[hypothesis, 1:].covariance.cpu()]
                        ),
                    )
                )

            else:
                # We have to create the track
                first_element = int(torch.nonzero(self.hypotheses_indices[hypothesis] + 1)[0].item())
                start = self.frame_id - self.hypotheses_indices.shape[1] + 1 + first_element
                identifier = self._next_identifier
                self._next_identifier += 1
                handler = TrackHandler(start, identifier, self.debug)

                det_ids.append(self.hypotheses_indices[hypothesis, first_element:].cpu().clone())
                states_l.append(self.kf_states[hypothesis, first_element:].to(self.device).clone())

            # Remove the missing trailing points
            if self.specs.drop_trailing or state == TrackHandler.TrackState.FINISHED:
                n_miss = len(det_ids[-1]) - int(torch.nonzero(det_ids[-1] + 1)[-1].item()) - 1
                if n_miss > 0:
                    states_l[-1] = states_l[-1][:-n_miss]
                    det_ids[-1] = det_ids[-1][:-n_miss]

            handlers.append(handler)

        # Build a global state to be smoothed
        states = torch_kf.GaussianState(
            torch.full((self.frame_id + 1, len(handlers), self.kalman_filter.state_dim, 1), torch.nan),
            torch.zeros((self.frame_id + 1, len(handlers), self.kalman_filter.state_dim, self.kalman_filter.state_dim)),
        )
        is_defined = torch.full((self.frame_id + 1, len(handlers)), False)

        for i, handler in enumerate(handlers):
            states[handler.start : handler.start + len(det_ids[i]), i] = states_l[i]
            is_defined[handler.start : handler.start + len(det_ids[i]), i] = True

        kalman_filter = self.kalman_filter.to(torch.device("cpu"))
        # Iterate backward to update all states (Update done for active t where t+1 is defined)
        for t in range(self.frame_id + 1 - 2, -1, -1):
            mask = is_defined[t + 1] & is_defined[t]
            cov_at_process = states.covariance[t, mask] @ kalman_filter.process_matrix.mT
            predicted_covariance = kalman_filter.process_matrix @ cov_at_process + kalman_filter.process_noise

            kalman_gain = cov_at_process @ predicted_covariance.inverse().mT
            states.mean[t, mask] += kalman_gain @ (
                states.mean[t + 1, mask] - kalman_filter.process_matrix @ states.mean[t, mask]
            )
            states.covariance[t, mask] += (
                kalman_gain @ (states.covariance[t + 1, mask] - predicted_covariance) @ kalman_gain.mT
            )

        dim = self.hypotheses_positions.shape[-1]  # For KOFT, using kf.measure_dim would not work
        tracks = []
        for i, handler in enumerate(handlers):
            tracks.append(
                byotrack.Track(
                    handler.start,
                    states.mean[handler.start : handler.start + len(det_ids[i]), i, :dim, 0],
                    handler.identifier,
                    det_ids[i],
                )
            )

        return tracks

    def motion_model(self) -> None:
        # Use KF to predict the next states
        predictions = self.kalman_filter.predict(self.kf_states[:, -1:])
        positions = predictions.mean[:, -1, : self.kalman_filter.measure_dim, 0]

        # Add optical flow motion to the position
        if self.optflow and self.optflow.flow_map is not None:
            positions[:] = torch.tensor(
                self.optflow.optflow.transform(self.optflow.flow_map, positions.cpu().numpy()), device=self.device
            )

        # Project states for association
        # self.projections = self.kalman_filter.project(predictions)

        # Update states & positions (Add depth dimension only if we neeed to register states)
        if self.save_all or self.specs.track_building == TrackBuilding.SMOOTHED:
            self.kf_states = torch_kf.GaussianState(
                torch.cat((self.kf_states.mean, predictions.mean), dim=1),
                torch.cat((self.kf_states.covariance, predictions.covariance), dim=1),
            )
        else:
            self.kf_states = predictions

        self.hypotheses_positions = torch.cat((self.hypotheses_positions, positions[:, None]), dim=1)

    def max_likelihood(self) -> torch.Tensor:
        """Returns the normalization constant (upper bound) of the likelihood distribution"""
        if self.specs.cost != Cost.MODEL:
            return super().max_likelihood()

        # For Kalman filter, the normalization constant depend on the hypothesis.
        # This is for unassociated track on frame t to be reassociated on frame t+1
        # The current state is at time t update, so we need to predict and project to compute the likelihood
        projections = self.kalman_filter.project(
            self.kalman_filter.predict(self.kf_states[:, -1]), precompute_precision=False
        )

        normalization_costs = 0.5 * torch.log(torch.det(projections.covariance))
        normalization_costs += 0.5 * projections.covariance.shape[-1] * torch.log(2 * torch.tensor(torch.pi))

        return normalization_costs

    def linking_cost(self, frame: np.ndarray, detections: byotrack.Detections) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.specs.cost != Cost.MODEL:
            return super().linking_cost(frame, detections)

        return fast_likelihood_cost(
            self.kalman_filter.project(self.kf_states[:, -1], precompute_precision=False),
            detections.position.to(self.dtype).to(self.device),
            self.specs.threshold(len(detections), self._volume, detections.dim),
            self.hypotheses_states[:, -1],
            self.specs.std * self.specs.model_cutoff,
        )

    def hypothesize_states(self, links: torch.Tensor):
        self.kf_states = _hypothesize(
            self.kf_states,
            links,
            positions=self.last_detections[-1].position.to(self.device),
            initial_covariance=self.build_initial_covariance(self.last_detections[-1].dim),
            linked_only=False,
        )

        # And projections using linked_only=True ?

    def update_states(self, kept: torch.Tensor):
        # Update states and positions for detected particles
        n_non_start = len(kept) - len(self.last_detections[-1])
        updated = kept[:n_non_start] & (self.hypotheses_indices[:n_non_start, -1] != -1)

        measures = self.last_detections[-1].position.to(self.device)[self.hypotheses_indices[:n_non_start, -1][updated]]
        self.kf_states[:n_non_start, -1][updated] = self.kalman_filter.update(
            self.kf_states[:n_non_start, -1][updated],
            measures[..., None],
            # projection=self.projections[:n_non_start, -1][updated],
        )
        self.hypotheses_positions[:n_non_start, -1][updated] = self.kf_states[:n_non_start, -1][updated].mean[
            ..., : measures.shape[1], 0
        ]

    def record_active_positions(self, active_hypotheses: torch.Tensor):
        super().record_active_positions(active_hypotheses)

        if self.save_all or self.specs.track_building == TrackBuilding.SMOOTHED:
            self.all_states.append(self.kf_states[:, 1][active_hypotheses].to(torch.device("cpu")))

    def filter_states(self, kept: torch.Tensor):
        super().filter_states(kept)

        if self.save_all or self.specs.track_building == TrackBuilding.SMOOTHED:
            self.kf_states = self.kf_states[:, 1:][kept]
        else:
            self.kf_states = self.kf_states[kept]

    def build_initial_covariance(self, dim: int) -> torch.Tensor:
        """Build the diagonal initial covariance matrix

        The position is initially unknown, leading to a belief (given by the first detection) set
        to the position of the first detection, with detection_std uncertainty.

        The velocity (and higher order derivatives) are assumed to be 0.0 with a relatively high uncertainty:
        initial_std_factor * process_std.

        Note that having a large initial_std_factor (>10) may decrease performances, as the first prediction
        will be impacted and largely uncertain, leading to low probabilies for every associations. In KOFT,
        as the velocity is measured before the first prediction, the initial_std_factor can be increased to
        reduce this bias toward a nul initial velocity. We found that initial_std_factor=0.0 is a good
        trade off in practice.

        Args:
            dim (int): Dimension of images

        Returns:
            torch.Tensor: Covariance matrix of the initial state
                Shape: (dim * order, dim * order)

        """
        process_std = torch.broadcast_to(torch.as_tensor(self.specs.process_std, dtype=self.dtype), (dim,)).clone()

        # In the case of Brownian motion, the intitial covariance if fully rewritten in SKT (no impact)
        # But in KOFT, Brownian motion models velocity with an initial velocity centered on 0 and with
        # an uncertainty given by the process_std, therefore we don't use initial_std_factor
        if self.specs.kalman_order > 0:
            process_std *= self.specs.initial_std_factor

        # Process std is squared and set for each order of the process (pos, vel, acc, ...)
        covariance = torch.diag(torch.cat([process_std**2] * (self.kalman_filter.state_dim // dim)))

        # Then, it is overwritten for the position, to be set to measurement_std
        measurement_std = torch.broadcast_to(torch.as_tensor(self.specs.detection_std, dtype=self.dtype), (dim,))
        torch.diagonal(covariance)[:dim] = measurement_std**2

        return covariance.to(self.device)
