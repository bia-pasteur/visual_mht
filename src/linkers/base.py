# pylint: disable=too-many-lines

import dataclasses
from typing import List, Optional, Tuple, Union
import warnings

import enum

import numpy as np
import scipy.special  # type: ignore
import torch

import byotrack
from byotrack.implementation.linker.frame_by_frame.base import OnlineFlowExtractor

# TODO: Support spawning ?


def _fast_euclidean_sq_cdist(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes very quickly the euclidean sq distance between two set of points"""
    # Cf torch.cdist without sqrt
    dim = x.shape[-1]
    x_pad = torch.zeros((x.shape[0], dim + 2), dtype=x.dtype, device=x.device)
    y_pad = torch.zeros((y.shape[0], dim + 2), dtype=y.dtype, device=y.device)
    x_pad[:, :dim] = x
    x_pad[:, dim] = 1
    x_pad[:, dim + 1] = (x * x).sum(dim=-1)
    y_pad[:, :dim] = -2 * y
    y_pad[:, dim] = (y * y).sum(dim=-1)
    y_pad[:, dim + 1] = 1

    dist = x_pad @ y_pad.mT
    dist.clip_(0.0)
    return dist


class TrackBuilding(enum.Enum):
    """How to build the final tracks

    * DETECTION
        Build tracks from detections without filtering nor filling gaps
    * FILTERED
        Build tracks from the filtering task. It directly uses `all_positions`.
    * SMOOTHED
        Build tracks from a smoothing task. Available for Kalman filters implementation
        (We use RTS)

    """

    DETECTION = "detection"
    FILTERED = "filtered"
    SMOOTHED = "smoothed"


class Cost(enum.Enum):
    """The Cost modeling for the next position of the track

    It also provides helpers to solve the cost computations efficiently.

    * MODEL
        User defined model to compute P(z_t = z | (z_k)_k<t). See `KalmanLinker`. The cost is the log likelihood of
        this probability. It automatically triggers probabilist thresholding, where `association_threshold`
        should correspond to the lowest probability acceptable.
    * LAPLACE/EUCLIDEAN
        P(z | z_pred) = 1 / C_L exp(-sqrt(dim+1) * ||z - z_pred||_2 / std)
        Uses a generalized Laplace distribution: the cost is an affine function of the Euclidean distance.
        LAPLACE triggers probabilist thresholding. `association_threshold` should correspond
        to the lowest probability acceptable.
        EUCLIDEAN triggers the dist-based thresholding. `association_threshold` corresponds
        to the highest normalized Euclidean distance acceptable. It is equivalent to a
        Laplace distribution with a suited probability threshold.
    * GAUSSIAN/EUCLIDEAN_SQ
        P(z | z_pred) = 1 / C_G exp(-||z - z_pred||_2^2 / std^2)
        Uses a Gaussian distribution: the cost is an affine function of the squared Euclidean distance.
        GAUSSIAN triggers probabilist thresholding. `association_threshold` should correspond
        to the lowest probability acceptable.
        EUCLIDEAN_SQ triggers the dist-based thresholding. `association_threshold` corresponds
        to the highest normalized Euclidean distance acceptable. It is equivalent to a
        Gaussian distribution with a suited probability threshold.

    """

    MODEL = "model"
    EUCLIDEAN = "euclidean"
    EUCLIDEAN_SQ = "euclidean_sq"
    LAPLACE = "laplace"
    GAUSSIAN = "gaussian"

    @staticmethod
    def gaussian_normalization(std: torch.Tensor) -> float:
        """Normalization constant of the Gaussian distribution (1 / C_g)

        Args:
            std (torch.Tensor): Standard deviation along each axis. (Covariances are fixed to 0.0)
                Shape: (dim,), dtype: float

        Returns:
            float: The Gaussian normalization factor (1 / (2 pi)^(d/2) / det(Sigma)^(1/2)
        """
        return 1 / (2 * torch.pi) ** (len(std) / 2) / std.prod().item()

    @staticmethod
    def laplace_normalization(std: torch.Tensor) -> float:
        """Normalization constant of the Laplace distribution (1 / C_l)

        Args:
            std (torch.Tensor): Standard deviation along each axis. (Covariances are fixed to 0.0)
                Shape: (dim,), dtype: float

        Returns:
            float: The Laplace normalization factor

        """
        dim = len(std)
        alpha = std / torch.sqrt(torch.tensor(dim) + 1)
        return (
            float(scipy.special.gamma(dim / 2) / scipy.special.gamma(dim))
            / 2
            / torch.pi ** (dim / 2)
            / alpha.prod().item()
        )

    def cost(
        self,
        hypotheses_positions: torch.Tensor,
        positions: torch.Tensor,
        association_threshold: float,
        hypotheses_states: torch.Tensor,
        *,
        std: Union[float, torch.Tensor] = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute linking cost between hypotheses and given detections positions

        It handles all Cost except MODEL which should be implemented inside the specific linkers.

        It returns the sparse cost matrix between hypotheses and detections.

        Args:
            hypotheses_positions (torch.Tensor): Hypotheses positions
                Shape: (n_h, dim), dtype: float
            positions (torch.Tensor): Detections positions
                Shape: (n_det, dim), dtype: float
            association_threshold (float): Do not consider association above this threshold.
                EUCLIDEAN[_SQ] triggers dist-based thresholding and `association_threshold` should
                correspond to the maximum value of the normalized Euclidean distance (||(x_1 - x_2)/ std||_2)
                to consider. Otherwise, it expresses the minimal probability to accept.
            hypotheses_states (torch.Tensor): State of each hypotheses (0: Active, 1: Finished, 2: Invalid).
                We do not compute linking cost for non-active states.
                Shape: (n_h,), dtype: int
            # anisotropy (Tuple[float, float, float]): Anisotropy in (D, H, W) axes ( <=> (Z, Y, X))
            #     Default: (1., 1., 1.)
            std (Union[float, torch.Tensor]): Standard deviation of the Laplace or Gaussian distribution
                Shape: (1, ) or (dim, ), dtype: float
                Default: 1.0

        Returns:
            torch.Tensor: Indices (i, j) of potential links between active hypothesis i and detection j
                Shape: (n_link, 2), dtype: int32
            torch.Tensor: The linking cost for each potential link
                Shape: (n_link,), dtype: float

        """
        if self == Cost.MODEL:
            raise NotImplementedError("MODEL cost should be implemented by the linker itself.")

        dim = positions.shape[1]
        std = torch.broadcast_to(torch.as_tensor(std, dtype=positions.dtype, device=positions.device), (dim,)).clone()

        # Filter out pruned hypotheses
        valid = hypotheses_states == 0
        mapping = torch.arange(len(hypotheses_states), device=positions.device)[valid]
        hypotheses_positions = hypotheses_positions[valid]

        if self in (Cost.LAPLACE, Cost.EUCLIDEAN):
            normalization_cost = -float(np.log(self.laplace_normalization(std)))
            alpha = float(np.sqrt(dim + 1))
        else:
            normalization_cost = -float(np.log(self.gaussian_normalization(std)))
            alpha = float(np.sqrt(0.5))

        # Let's compute compute d^2 = a^2 |.|_2^2 / sigma^2
        # with a=sqrt(1/2) for gaussians and a=sqrt(dim+1) for laplace
        hypotheses_positions = hypotheses_positions * (alpha / std)
        positions = positions * (alpha / std)
        dist = _fast_euclidean_sq_cdist(hypotheses_positions, positions)

        if self == Cost.LAPLACE:  # Convert prob thresh into d^2 thresh: -np.log(C_l) + d < -log(thresh)
            association_threshold = -float(np.log(association_threshold)) - normalization_cost
            association_threshold **= 2
        elif self == Cost.GAUSSIAN:  # Convert prob thresh into d^2 thresh: -np.log(C_g) + d^2 < -log(thresh)
            association_threshold = -float(np.log(association_threshold)) - normalization_cost
        else:  # Convert |.|_2 / sigma thresh into d^2 thresh
            association_threshold = association_threshold**2 * alpha**2

        indices = torch.nonzero(dist < association_threshold).to(torch.int32)
        costs = dist[indices[:, 0], indices[:, 1]]
        indices[:, 0] = mapping[indices[:, 0]]

        # convert from |.|^2 to prob
        if self in (Cost.EUCLIDEAN, Cost.LAPLACE):
            costs.sqrt_()

        costs += normalization_cost
        return indices, costs


class Solver(enum.Enum):
    """Milp solvers

    * SCIPY
    * GUROBI
    * LAP
        It will reconvert the problem into a LAP one and use a sparse optimal solver.
        Only valid for tree_depth = 2

    """

    SCIPY = "scipy"
    GUROBI = "gurobi"
    LAP = "lap"

    def solve(self, cost: torch.Tensor, indices: torch.Tensor, sizes: List[int]) -> torch.Tensor:
        """Solve tracks-to-detections association

        Args:
            cost (torch.Tensor): Cost for each hypothesis
                Shape: (n_h,), dtype: float
            indices (float): Indices mapping of each hypothesis to the index of the corresponding track/detection
                Shape: (n_h, tree_depth), dtype: int

        Returns:
            torch.Tensor: Selected hypotheses
                Shape: (n_h), dtype: bool
        """

        if self == Solver.SCIPY:
            from . import scipy_solver  # pylint: disable=import-outside-toplevel

            return torch.tensor(
                scipy_solver.solve_map(cost.cpu().numpy(), indices.cpu().numpy(), sizes), device=cost.device
            )

        if self == Solver.GUROBI:
            from . import gurobi_solver  # pylint: disable=import-outside-toplevel

            return torch.tensor(
                gurobi_solver.solve_map(cost.cpu().numpy(), indices.cpu().numpy(), sizes), device=cost.device
            )

        from . import lap_solver  # pylint: disable=import-outside-toplevel

        return torch.tensor(lap_solver.solve_map(cost.cpu().numpy(), indices.cpu().numpy(), sizes), device=cost.device)


class TrackHandler:  # pylint: disable=too-many-instance-attributes
    """Store track data during the tracking procedure.

    It accumulates the track data at each new validated association.

    A TrackHandler is created for each validated track creation in the hypotheses process. Then
    it is updated with the following associated detections. A track can be either false positive (INVALID),
    where each associated detection is considered as independant FP of the detection process,
    or a true track, where all detections are true detections which can then be FINISHED (no future association)
    or still ACTIVE.

    In this implementation, all the work is handled in hypotheses tree, this class simply stored the data to be
    collected in `collect`.

    Attributes:
        start (int): Starting frame of the track
        identifier (int): Identifier of the track handler (and of the track)
        track_state (TrackState): Current state of the handler
        n_miss (int): Number of frames since the last association
        n_det (int): Number of true detections associated
        detection_ids (List[int]): Identifiers of the associated detection (-1 if None)
        track_ids (List[int]): Index of the track at each frame in the `linker.active_tracks` list.
            It allows the linker to store data as tensor and be able to rebuild tracks at the end.
            See `collect`

    """

    class TrackState(enum.IntEnum):
        """TrackState of a TrackHandler

        * ACTIVE
            The track is still active (Not yet finished, neither classified FP)
        * FINISHED
            The track is valid but finished
        * INVALID
            The track has been classified as invalid (False positive)

        """

        ACTIVE = 0
        FINISHED = 1
        INVALID = 2

    def __init__(self, start: int, identifier: int, debug=False) -> None:
        self.debug = debug
        self.start = start
        self.identifier = identifier
        self.track_state: int = TrackHandler.TrackState.ACTIVE
        self.n_miss = 0
        self.n_det = 0
        self.detection_ids: List[int] = []
        self.track_ids: List[int] = []

    def __len__(self) -> int:
        if self.track_state == TrackHandler.TrackState.INVALID:
            return self.n_det

        if self.track_state == TrackHandler.TrackState.FINISHED:
            return len(self.detection_ids) - self.n_miss

        return len(self.detection_ids)

    def is_active(self) -> bool:
        return self.track_state == 0

    def update(self, frame_id: int, detection_id: int, state: int) -> None:  # Could register cost for debug
        """Update track handler. It stores the detection_id and update the track state.

        It should be called for each time frame and each active track.

        Args:
            frame_id (int): The current frame. This is given for safety checks
                to ensure that the Linker and TrackHandler agree.
            detection_id (int): Detection id in the Detections object.
                -1 if not associated to a particular detection.
            state (int): Current track state

        """
        if self.debug:
            assert self.is_active()
            assert len(self.track_ids) == len(
                self.detection_ids
            ), "The linker should call `update` then `register_track_id` at each linking step"
            assert (
                self.start + len(self.detection_ids) == frame_id
            ), "The linker should update each active track on each time frame."

        self.detection_ids.append(detection_id)

        if detection_id == -1:  # Not associated
            self.n_miss += 1
        else:
            self.n_miss = 0
            self.n_det += 1

        self.track_state = state

    def register_track_id(self, track_id: int) -> None:
        """For still active tracks, it registers the track id after the update step.

        Args:
            track_id (int): The index of the track in `linker.active_tracks` at this time frame.
        """
        self.track_ids.append(track_id)


def _hypothesize(  # pylint: disable=too-many-arguments,too-many-locals
    hypotheses_indices: torch.Tensor,
    hypotheses_positions: torch.Tensor,
    hypotheses_costs: torch.Tensor,
    hypotheses_states: torch.Tensor,
    *,
    links: torch.Tensor,
    costs: torch.Tensor,
    positions: torch.Tensor,
    max_likelihood: torch.Tensor,
    fnr: float,
    death_rate: float,
    lambda_b: float,
    lambda_f: float,
    volume: float,
    false_tracks: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate new hypothesis"""
    n_h, n_link, n_det, depth = len(hypotheses_costs), len(links), len(positions), hypotheses_states.shape[1]
    dim = positions.shape[1]
    device = hypotheses_indices.device

    # Allocate new arrays for hypotheses
    new_indices = torch.empty((n_link + n_h + n_det, depth + 1), dtype=hypotheses_indices.dtype, device=device)
    new_states = torch.zeros((n_link + n_h + n_det, depth + 1), dtype=hypotheses_states.dtype, device=device)
    new_costs = torch.empty((n_link + n_h + n_det, 3), dtype=hypotheses_costs.dtype, device=device)
    new_positions = torch.zeros((n_link + n_h + n_det, depth + 1, dim), dtype=hypotheses_positions.dtype, device=device)

    # Convert prob to cost
    false_cost = -float(np.log(lambda_f / volume))  # lambda_f / V
    birth_cost = -float(np.log(lambda_b / volume))  # lambda_b / V
    death_cost = -float(np.log(death_rate))
    non_death_cost = -float(np.log(1 - death_rate))
    miss_cost = -float(np.log(fnr) + non_death_cost)
    link_cost = -float(np.log(1 - fnr) + non_death_cost)

    # Add link_cost to both likelihood costs
    max_likelihood += link_cost
    costs += link_cost

    # 1. Handle links
    # Propagate indices and positions
    new_indices[:n_link, :depth] = hypotheses_indices[links[:, 0]]
    new_indices[:n_link, depth] = links[:, 1]
    new_positions[:n_link] = hypotheses_positions[links[:, 0]]

    # Always active when associated, but states is already 0
    # new_states[:n_link, :depth] = hypotheses_states[links[:, 0]]
    # new_states[:n_link, depth] = 0

    # Compute costs
    new_costs[:n_link, 0] = hypotheses_costs[links[:, 0], 0]
    new_costs[:n_link, 0] += costs  # Already contains likelihood + prior
    new_costs[:n_link, 1] = new_costs[:n_link, 0]  # Active + termination
    new_costs[:n_link, 1] += death_cost
    new_costs[:n_link, 2] = hypotheses_costs[links[:, 0], 2]  # n * false_cost
    new_costs[:n_link, 2] += false_cost if false_tracks else torch.inf

    # 2. Handle non-linked hypotheses
    # Propagate indices with -1, positions & states
    new_indices[n_link : n_link + n_h, :depth] = hypotheses_indices
    new_indices[n_link : n_link + n_h, depth] = -1
    new_positions[n_link : n_link + n_h] = hypotheses_positions
    new_states[n_link : n_link + n_h, :depth] = hypotheses_states
    new_states[n_link : n_link + n_h, depth] = hypotheses_states[:, -1]  # propagate last state, unless future pruning

    # Compute costs
    new_costs[n_link : n_link + n_h, 0] = hypotheses_costs[:, 0]
    new_costs[n_link : n_link + n_h, 0] += miss_cost
    new_costs[n_link : n_link + n_h, 1] = hypotheses_costs[:, 1]
    new_costs[n_link : n_link + n_h, 2] = hypotheses_costs[:, 2]

    # 3. Create birth hypotheses from detections
    # Set indices (-1, ..., -1, k)
    new_indices[n_link + n_h :, :depth] = -1
    new_indices[n_link + n_h :, depth] = torch.arange(n_det, dtype=new_indices.dtype, device=device)

    # Set positions to (0, ..., 0, pos)
    new_positions[n_link + n_h :, -1] = positions  # Other depths are already at 0.0

    # State is already 0
    # Set initial cost
    new_costs[n_link + n_h :, 0] = birth_cost
    new_costs[n_link + n_h :, 1] = birth_cost + death_cost
    new_costs[n_link + n_h :, 2] = false_cost

    # 4. Hypotheses pruning for non-associated hypotheses
    # Let's reevaluate the state (we would filter already non-active targets)
    # Finish/invalidate hypothesis if a perfect association (p(z) = max_likelihood) is still beaten by a track creation
    hypotheses_costs[:] = new_costs[n_link : n_link + n_h]  # Reuse hypotheses costs that will be deleted anyway
    hypotheses_costs[:, 0] += max_likelihood  # Best future link
    hypotheses_costs[:, 1] += birth_cost  # Terminates and create
    hypotheses_costs[:, 2] += birth_cost  # Invalidate and create
    new_states[n_link : n_link + n_h, -1] = hypotheses_costs.argmin(dim=-1)  # If best future link is beaten, we prune

    return new_indices, new_positions, new_costs, new_states


@dataclasses.dataclass
class MHTLinkerParameters:  # pylint: disable=too-many-instance-attributes
    """Parameters of the MHTLinker

    Attributes:
        association_threshold (float): Threshold on the linking cost. Linking hypotheses are not made
            above this threshold. This allows to reduce the exponential growth of the hypotheses tree. By default (-1),
            it is set to the theoretical value dr / [(1-dr)(1-fnr)] * lambda_b/V that corresponds to the limit
            above which it is preferable to recreate a track instead of linking. It can be reduced to its own value.
            With a `cost` EUCLIDEAN or EUCLIDEAN_SQ, it corresponds to the highest acceptable normalized Euclidean
            distance (||(x_1 - x_2)/ std||_2).
            Otherwise, it is the lowest acceptable probability (See `Cost`).
            Default: -1.0
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
            will the use predicted (and updated) position on each frame (filtering distribution).
            SMOOTHED is not implemented here, any child class can implement its own smoothing post tracking.
            Default: DETECTION
        cost (Cost): Linking cost method, see `Cost`. The detection position is distributed either following a
            specific model (MODEL) implemented in a child class, or a LAPLACE/GAUSSIAN distribution with a standard
            deviation `std`. EUCLIDEAN (resp. EUCLIDEAN_SQ), refers to LAPLACE (resp. GAUSSIAN) distribution with
            dist-based thresholding (see `association_threshold`).
            Default: EUCLIDEAN
        std (Union[float, torch.Tensor]): Standard deviation of the Laplace or Gaussian distribution.
            Typically characterize the precision of the motion_model.
            Shape: (1, ) or (dim, ), dtype: float
            Default: 1.0
        drop_trailing (bool): Drop the last missing detections of tracks that are still more probable to be active
            than finished on the last frame.
            Default: True
        false_tracks (bool): If True, we allow the creation of false positive tracks, that can be linked even
            if the track seems to best false. This enable track birth if birth_rate << fpr with small depths
            (as otherwise, all births are considered as false positives with small depth)
            Default: True

    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        *,
        association_threshold: float = -1.0,
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
        drop_trailing: bool = True,
        false_tracks: bool = True,
    ):
        self.association_threshold = association_threshold

        assert 0.0 <= fpr < 1.0
        assert 0.0 <= fnr < 1.0
        assert 0.0 < self.birth_rate < 1.0
        assert 0.0 < self.death_rate < 1.0
        self.fpr = fpr
        self.fnr = fnr
        self.birth_rate = birth_rate
        self.death_rate = death_rate

        assert tree_depth >= 2, "tree_depth >= 2, as it requires at least frame t and t+1 to solve the association"
        assert step >= 0
        self.tree_depth = tree_depth
        self.step = step

        self.n_det = n_det
        self.volume = volume
        self.n_0 = n_0

        self.solver = solver if isinstance(solver, Solver) else Solver[solver.upper()]
        self.drop_trailing = drop_trailing
        self.false_tracks = false_tracks
        self.track_building = (
            track_building if isinstance(track_building, TrackBuilding) else TrackBuilding[track_building.upper()]
        )
        self.cost = cost if isinstance(cost, Cost) else Cost[cost.upper()]
        self.std = std

    association_threshold: float = -1.0
    fpr: float = 0.1
    fnr: float = 0.1
    birth_rate: float = 1e-2
    death_rate: float = 1e-2
    tree_depth: int = 4
    step: int = 1
    n_det: float = -1.0
    volume: float = -1.0
    n_0: float = -1.0
    solver: Solver = Solver.SCIPY
    track_building: TrackBuilding = TrackBuilding.DETECTION
    cost: Cost = Cost.EUCLIDEAN
    std: Union[float, torch.Tensor] = 1.0
    drop_trailing: bool = True
    false_tracks: bool = True

    @property
    def _det_factor(self) -> float:
        """E(n(t)) = _det_factor * E(n_det(t))"""
        return (1 - self.fpr) / ((1 - self.fnr) * (1 - self.birth_rate) + self.birth_rate)

    def lambda_b(self, n_det: int) -> float:
        """Expected number of births on frame t

        hat{lambda_b} = br * (1-fpr)/((1-fnr)(1-br) + br) * n_det(t)

        Args:
            n_det (int): Number of detections observed on frame t
                Used if a default constant one is not provided in self.n_det

        Returns:
            float: Single frame estimation of lambda_b
        """
        return self.birth_rate * self._det_factor * (n_det if self.n_det < 0.0 else self.n_det)

    def lambda_f(self, n_det: int) -> float:
        """Expected number of false positives on frame t

        hat{lambda_f} = fpr * n_det(t)

        Args:
            n_det (int): Number of detections observed on frame t
                Used if a default constant one is not provided in self.n_det

        Returns:
            float: Single frame estimation of lambda_f
        """
        return self.fpr * (n_det if self.n_det < 0.0 else self.n_det)

    def n_0_est(self, n_det: int, frame_id: int) -> float:
        """Expected number of object before the first frame

        hat{n_0} = (1-br)/(1-dr)^t * (1-fpr)/((1-fnr)(1-br) + br) * n_det(t)

        Args:
            n_det (int): Number of detections observed on frame t
            frame_id (int): Frame id (t-1)

        Returns:
            float: Single frame estimation of n_0
        """
        if self.n_0 >= 0.0:
            return self.n_0

        return ((1 - self.birth_rate) / (1 - self.death_rate)) ** (frame_id + 1) * self._det_factor * n_det

    def lambda_b_fixed(self, n_0: float, n_det: int, frame_id: int) -> float:
        """Fixed lambda_b that accounts for the unknown initial number of existing objects n_0

        hat{lambda_b}* = hat{lambda_b} + (1 - dr)^t fnr^(t-1) (1-fnr) hat{n_0}

        Args:
            n_0 (float): Estimation of n_0 the number of initial existing objects before the first frame
            n_det (int): Number of detections observed on frame t
                Used if a default constant one is not provided in self.n_det
            frame_id (int): Frame id (t-1)

        Returns:
            float: Single frame estimation of lambda_b, with correction for existing objects n_0
        """
        return (
            self.lambda_b(n_det) + (1 - self.death_rate) ** (frame_id + 1) * self.fnr**frame_id * (1 - self.fnr) * n_0
        )

    def threshold(self, n_det: int, volume: float, dim: Optional[int] = None) -> float:
        """Compute the association threshold if not given.

        It defaults to max(dr * lambda_b / V  / (1-dr) / (1-fnr), fnr * lambda_f / V / (1 - fnr)).
        That corresponds to the alternative costs of terminating and starting a new track,
        or classifying false the detection and missed the track, rather than the linking.

        Works for MODEL, GAUSSIAN, and LAPLACE, as it returns a probability threshold.
        For EUCLIDEAN/EUCLIDEAN_SQ, it is converted to a euclidean threshold, but it
        requires to know the dimension of the problem.

        Args:
            n_det (int): Number of detections observed on frame t
                Used if a default constant one is not provided in self.n_det
            volume (float): Number of pixels considered in the images.
                Used if a default constant one is not provided in self.volume
            dim (int): Required to extend self.std in EUCLIDEAN[_SQ]
                Default: None

        Returns:
            float: Association threshold to use
                (prob threshold or euclidean threshold depending on the cost method)
        """
        # Consider the following scenario with 3 detections, one per frame
        # (1) o => o => o
        # Then the first link can be broken, and still having a single track at the end, with
        # two possibilities:
        # (2) o => x => x (death)
        #     x => o => o (birth)
        # And:
        # (3) o => x => o (miss)
        #     x => o => x (false)
        #
        # (2) is better than (1) as long as p_link (1-dr)(1-fnr) < (1-dr)fnr * lambda_f / V
        # (3) is better than (1) as long as p_link (1-dr)(1-fnr) < dr lambda_b / V
        # This gives a rather good approximation of the association threshold

        if self.association_threshold >= 0.0:
            return self.association_threshold

        if self.volume > 0.0:  # Use given volume (should already be the case)
            volume = self.volume

        association_threshold = max(
            self.death_rate / ((1 - self.death_rate) * (1 - self.fnr)) * self.lambda_b(n_det) / volume,
            self.fnr / (1 - self.fnr) * self.lambda_f(n_det) / volume,
        )

        assert association_threshold > 0.0

        if self.cost not in (Cost.EUCLIDEAN, Cost.EUCLIDEAN_SQ):
            return association_threshold

        if dim is None:
            raise ValueError("Cannot compute the default cost for EUCLIDEAN[_SQ] without `dim`")

        std = torch.broadcast_to(torch.as_tensor(self.std, dtype=torch.float32), (dim,))

        # In EUCLIDEAN[_SQ], threshold is expressed in normalized euclidean distance: d = |.|_2 / sigma
        if self.cost == Cost.EUCLIDEAN:  # 1 / C_l exp(-alpha * d)
            # 1 / C_l * e(-d) > thresh
            # -log(1/C_l) + d < -log(thresh)
            # d < (-log(thresh) + log(1/C_l)) / alpha
            association_threshold = -float(np.log(association_threshold))
            association_threshold += float(np.log(Cost.laplace_normalization(std)))
            association_threshold /= float(np.sqrt(dim + 1))
        elif self.cost == Cost.EUCLIDEAN_SQ:  # 1 / C_g exp(-0.5 d^2)
            # 1 / C_g * e(-0.5 d^2) > thresh
            # -log(1/C_g) + 0.5 d^2 < -log(thresh)
            # d < sqrt((-log(thresh) + log(1/C_g)) * 2)
            association_threshold = -float(np.log(association_threshold))
            association_threshold += float(np.log(Cost.gaussian_normalization(std)))
            association_threshold *= 2
            association_threshold = float(np.sqrt(association_threshold))

        return association_threshold


class MHTLinker(byotrack.OnlineLinker):  # pylint: disable=too-many-instance-attributes
    """Links detections online using Multiple Hypotheses Tracking.

    It poses the MHT problem as a track-oriented MHT that is solved through a Multidimensionnal
    Assignement Problem (MAP). At time t, we have a pool of n_t active tracks. The linker
    maintains a hypotheses tree for each track, stating the future association of the track and
    the cost of each branch. MAP is used to find the optimal links inside the trees that use
    each detections only once.

    It decomposes the update step in 6 parts:

    1. Optional optical flow computations (Handled by this class with the `optflow` given)
    2. Motion modeling to predict hypotheses positions (`motion_model`)
    3. Features extraction (handled by this class with the `features_extractor` given)
    4. Hypotheses formulation and association cost computations. (`hypothesize`)
    5. Solving the MAP to select hypotheses (`associate`)
    6. If tree_depth is reached, filtering and update hypotheses (`filter_hypotheses`)

    # TODO: Describe a bit more precisely each step

    Attributes:
        specs (MHTLinkerParameters): Parameters specifications of the algorithm.
            See `MHTLinkerParameters`.
        optflow (Optional[OnlineFlowExtractor]): Optional wrapper around the given optional OpticalFlow that
            will extract flow maps of the video online. (The underlying OpticalFlow object is accessible in
            self.optflow.optflow)
            Default: None
        features_extractor (Optional[FeaturesExtractor]): Optional features extractor that will extract
            features for the detections, which could be useful for tracking.
            Default: None
        save_all (bool): Save metadata useless for the final building of tracks
            but that could be useful for analysis. For instance, it will keep invalid tracks.
            Or the computed features inside the Detections objects.
        debug (bool): Triggers debugging: log more info and run test to ensure the implementation is correct
            Default: False
        frame_id (int): Current frame id of the linking process
        inactive_tracks (List[TrackHandler]): Terminated tracks (FINISHED ones, and INVALID ones if `save_all` is True)
        active_tracks (List[TrackHandler]): Current track handlers with a lag of `tree_depth` - 1 frames.
        all_positions (List[torch.Tensor]): Positions of the active tracks at each seen frames.
            Using the valid track handlers `track_ids`, it allows the reconstruction of tracks.
        selected (torch.Tensor): Selected hypotheses from the last `associate` call.
            Shape: (n_h,), dtype: bool
        last_detections (List[byotrack.Detections]): Record the last `tree_depth` detections
        hypotheses_indices (torch.Tensor): (i_t, j_{t+1}, ..., j_{t+d - 1}) where i_t is
            an identifier for an active track at time t, and j_{t+k} is an identifier for a
            detection on frame t + k. This characterize a linking hypothesis.
            Shape: (n_h, depth), dtype: int32
        hypotheses_costs (torch.Tensor): Associated cost for each hypotheses computed recursively.
            See `hypothesize` and `_update_hypotheses_cost` for details. It holds a cost for the 3
            possible states of hypothesis.
            Shape: (n_h, 3), dtype: float32
        hypotheses_states (torch.Tensor): State of each hypothesis. All hypothesis are ACTIVE (0)
            at first. Then when they are non-linked, they can become FINISHED (1) or INVALID (2)
            if no future association can be made to change the state.
            Shape: (n_h, depth), dtype: int32
        hypotheses_positions (torch.Tensor): Record the filtered position of each hypotheses.
            `motion_model` moves the position forward in time (adding a depth).
            `update_state` can optionnally update the registered position.
            Shape: (n_h, depth, dim), dtype: float32

    """

    progress_bar_description = "MHT linking"

    def __init__(
        self,
        specs: MHTLinkerParameters,
        optflow: Optional[byotrack.OpticalFlow] = None,
        features_extractor: Optional[byotrack.FeaturesExtractor] = None,
        save_all=False,
    ) -> None:
        super().__init__()
        self.debug = False
        self.specs = specs
        self.optflow = OnlineFlowExtractor(optflow) if optflow else None
        self.features_extractor = features_extractor
        self.save_all = save_all
        self.frame_id = -1

        self.dtype = torch.float32
        self.device = torch.device("cpu")

        # Frozen data with a lag of tree_depth frame
        self.inactive_tracks: List[TrackHandler] = []
        self.active_tracks: List[TrackHandler] = []
        self.all_positions: List[torch.Tensor] = []

        # Hypotheses informations
        self.selected = torch.full((0,), False)
        self.last_detections = [byotrack.Detections(data={"position": torch.empty((0, 2))})]
        self.hypotheses_indices = torch.zeros((0, 1), dtype=torch.int32, device=self.device)
        self.hypotheses_costs = torch.zeros((0, 3), dtype=self.dtype, device=self.device)
        self.hypotheses_states = torch.zeros((0, 1), dtype=torch.int32, device=self.device)
        self.hypotheses_positions = torch.zeros((0, 1, 2), dtype=self.dtype, device=self.device)

        self._next_identifier = 0
        self._n_0_est = 1.0
        self._volume = 1.0
        self._last_solve = self.frame_id

    def reset(self, dim=2) -> None:
        super().reset(dim)
        if self.optflow:
            self.optflow.reset()
        self.frame_id = -1
        self.inactive_tracks = []
        self.active_tracks = []
        self.all_positions = []

        self.last_detections = [byotrack.Detections(data={"position": torch.empty((0, dim))})]
        self.hypotheses_indices = torch.zeros((0, 1), dtype=torch.int32, device=self.device)
        self.hypotheses_costs = torch.zeros((0, 3), dtype=self.dtype, device=self.device)
        self.hypotheses_states = torch.zeros((0, 1), dtype=torch.int32, device=self.device)
        self.hypotheses_positions = torch.zeros((0, 1, dim), dtype=self.dtype, device=self.device)
        self.selected = torch.full((0,), False)

        self._next_identifier = 0
        self._n_0_est = 1.0
        self._volume = 1.0
        self._last_solve = self.frame_id

    def collect(self) -> List[byotrack.Track]:  # pylint: disable=too-many-branches,too-many-locals
        if self.specs.track_building == TrackBuilding.SMOOTHED:
            warnings.warn("Unable to build smoothed tracks. Will build filtered ones")

        # We need to solve the association up to the end
        self.associate()

        tracks = []
        points: torch.Tensor
        for handler in self.inactive_tracks:
            if handler.track_state is TrackHandler.TrackState.INVALID:
                continue  # Ignore non-valid tracks

            points = torch.cat(
                [
                    positions[track_id : track_id + 1]
                    for track_id, positions in zip(
                        handler.track_ids[: len(handler)],
                        self.all_positions[handler.start :],
                    )
                ]
            )

            det_ids = torch.tensor(handler.detection_ids[: len(handler)], dtype=torch.int32)

            tracks.append(byotrack.Track(handler.start, points, handler.identifier, det_ids))

        # For active tracks, we rely on the selected hypotheses
        hypotheses = torch.arange(len(self.hypotheses_indices), device=self.device)[self.selected]
        for hypothesis in hypotheses:
            state = int(self.hypotheses_costs[hypothesis].argmin().item())  # Different from hypotheses_states[-1]
            if state == TrackHandler.TrackState.INVALID:
                continue  # More likely to be False Positive, let's ignore it

            i = int(self.hypotheses_indices[hypothesis, 0].item())

            if i != -1:
                handler = self.active_tracks[i]
                start = handler.start
                identifier = handler.identifier
                det_ids = torch.cat(
                    [
                        torch.tensor(handler.detection_ids, dtype=torch.int32),
                        self.hypotheses_indices[hypothesis, 1:].cpu(),
                    ]
                )

                points = torch.cat(
                    [
                        positions[track_id : track_id + 1]
                        for track_id, positions in zip(
                            self.active_tracks[i].track_ids, self.all_positions[handler.start :]
                        )
                    ]
                )

                if self.specs.track_building == TrackBuilding.DETECTION:
                    points_ = []
                    for detections, j in zip(self.last_detections[1:], self.hypotheses_indices[hypothesis, 1:].cpu()):
                        points_.append(
                            detections.position[j : j + 1] if j >= 0 else torch.full((1, detections.dim), torch.nan)
                        )

                    points = torch.cat([points] + points_)
                else:
                    points = torch.cat([points, self.hypotheses_positions[hypothesis, 1:].cpu()])
            else:
                # We have to create the track
                first_element = int(torch.nonzero(self.hypotheses_indices[hypothesis] + 1)[0].item())
                start = self.frame_id - self.hypotheses_indices.shape[1] + 1 + first_element
                identifier = self._next_identifier
                self._next_identifier += 1
                det_ids = self.hypotheses_indices[hypothesis, first_element:].cpu().clone()

                if self.specs.track_building == TrackBuilding.DETECTION:
                    points_ = []
                    for detections, j in zip(self.last_detections[first_element:], det_ids):
                        points_.append(
                            detections.position[j : j + 1] if j >= 0 else torch.full((1, detections.dim), torch.nan)
                        )

                    points = torch.cat(points_)
                else:
                    points = self.hypotheses_positions[hypothesis, first_element:].cpu().clone()

            # Remove the missing trailing points
            if self.specs.drop_trailing or state == TrackHandler.TrackState.FINISHED:
                n_miss = len(det_ids) - int(torch.nonzero(det_ids + 1)[-1].item()) - 1
                if n_miss > 0:
                    points = points[:-n_miss]
                    det_ids = det_ids[:-n_miss]

            tracks.append(byotrack.Track(start, points, identifier, det_ids))

        return tracks

    def motion_model(self) -> None:
        """Modelisation of tracks motion (to model going forward of one frame in time)

        It should update any track state/model used in the tracker and add a new time point to `hypotheses_positions`.
        It is called after the optical flow computation (if any) and before computing the linking cost and
        generating new hypotheses.

        By default, we do not model anything and we only extend the former positions.

        * hypotheses_positions: (n_h, depth, dim) => (n_h, depth + 1, dim)

        """
        next_positions = self.hypotheses_positions[:, -1]  # Shape: n_h, dim

        if self.optflow and self.optflow.flow_map is not None:
            next_positions = torch.tensor(
                self.optflow.optflow.transform(self.optflow.flow_map, next_positions.cpu().numpy()),
                dtype=self.dtype,
                device=self.device,
            )

        # Goes from depth to depth + 1, by building a constant predictions (or using optical flow)
        self.hypotheses_positions = torch.cat((self.hypotheses_positions, next_positions[:, None]), dim=1)

    def linking_cost(
        self, frame: np.ndarray, detections: byotrack.Detections  # pylint: disable=unused-argument
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the association cost between active hypotheses and detections

        Args:
            frame (np.ndarray): The current frame of the video
                Shape: (H, W, C), dtype: float
            detections (byotrack.Detections): Detections for the given frame

        Returns:
            torch.Tensor: Indices (i, j) of potential links between active hypothesis i and detection j
                Shape: (n_link, 2), dtype: int32
            torch.Tensor: The linking cost for each potential link
                Shape: (n_link,), dtype: float

        """
        return self.specs.cost.cost(
            self.hypotheses_positions[:, -1],
            detections.position.to(self.dtype).to(self.device),
            self.specs.threshold(len(detections), self._volume, detections.dim),
            self.hypotheses_states[:, -1],
            std=self.specs.std,
        )

    def hypothesize_states(self, links: torch.Tensor) -> None:
        """Optional set up of the hypotheses internal states in `hypothesize`

        It is called in `hypothesize` after selecting n_link linking hypotheses.

        See `hypothesize`.

        Args:
            torch.Tensor: Indices (i, j) of potential links between active hypothesis i and detection j
                Shape: (n_link, 2), dtype: int32

        """

    def update_states(self, kept: torch.Tensor) -> None:
        """Update of internal states and positions

        Called after association but before filtering hypotheses and recording positions.
        It is given the filtering mask `kept`, to prevent update useless (to be filtered) states.

        See `filter_hypotheses`. Also called when hypotheses are not filtered (with kept.all() = True)

        Args:
            kept (torch.Tensor): Boolean mask of the kept hypotheses to update.
                Shape: (n_link + n_h + n_det), dtype: bool

        """
        # XXX: Can't we drop `kept``, and do the update after? (Need to either find a good init for kf,
        #      or find a marker for birthing tracks)
        # By default, simply set the hypotheses positions to detection position if associated.
        # Update hypotheses positions with detections
        # Optionally using an EMA to reduce detections noise
        n_non_start = len(kept) - len(self.last_detections[-1])
        updated = kept[:n_non_start] & (self.hypotheses_indices[:n_non_start, -1] != -1)

        self.hypotheses_positions[:n_non_start, -1][updated] = self.last_detections[-1].position.to(self.device)[
            self.hypotheses_indices[:n_non_start, -1][updated]
        ]

    def record_active_positions(self, active_hypotheses: torch.Tensor) -> None:
        """Record the positions of active tracks in `all_positions`

        Called after association but before filtering hypotheses. It is given the selected
        hypothesis index for each active track.

        See `filter_hypotheses`.

        Args:
            active_hypotheses (torch.Tensor): Selected hypotheses index for each active tracks
                Shape: (n), dtype: int32

        """
        if self.specs.track_building == TrackBuilding.DETECTION:
            positions = torch.cat(
                (self.last_detections[1].position, torch.full((1, self.last_detections[1].dim), torch.nan))
            )
            self.all_positions.append(positions[self.hypotheses_indices[:, 1][active_hypotheses].cpu()])
            return

        self.all_positions.append(self.hypotheses_positions[:, 1][active_hypotheses].cpu())

    def filter_states(self, kept: torch.Tensor) -> None:
        """Optional filter of internal states.

        Depth (if any) should be reduced of one, and only `kept` hypotheses are kept.

        Args:
            kept (torch.Tensor): Boolean mask of the kept hypotheses.
                Shape: (n_h), dtype: bool

        """

    def update_detections(self, detections: byotrack.Detections) -> byotrack.Detections:
        """Optional modification of the currrent detections based on the current state

        This is called by `update` after the motion modeling but before the cost/association.

        By default, it does not change anything.

        Args:
            detections (byotrack.Detections): Detections at the current frame

        Returns:
            byotrack.Detections: The (optionally modified) detections to use at this current frame

        """
        return detections

    def max_likelihood(self) -> torch.Tensor:
        """Returns the normalization constant (upper bound) of the likelihood distribution"""
        if self.specs.cost == Cost.MODEL:
            raise NotImplementedError("MODEL cost should be implemented by the linker itself.")

        dim = self.last_detections[-1].dim
        std = torch.broadcast_to(torch.as_tensor(self.specs.std, dtype=self.dtype, device=self.device), (dim,))

        if self.specs.cost in (Cost.LAPLACE, Cost.EUCLIDEAN):
            normalization_cost = -float(np.log(Cost.laplace_normalization(std)))
        else:
            normalization_cost = -float(np.log(Cost.gaussian_normalization(std)))

        return torch.full_like(self.hypotheses_costs[:, 0], normalization_cost)

    def hypothesize(self, frame: np.ndarray, detections: byotrack.Detections):
        """Produce new hypotheses for the given n_det detections

        Given n_h hypotheses at depth d, it will produce n_link + n_h + n_det extended hypotheses on a depth d+1.

        First, it computes a sparse linking matrix between hypotheses and detections (See `linking_cost`). It yields
        n_link linking hypotheses. Then for each hypothesis, non-link extension are added. Finally, n_det track birth
        hypotheses are created (one for each detection).

        * hypotheses_indices:            (n_h, depth) => (n_link + n_h + n_det, depth + 1)
        * hypotheses_costs:                  (n_h, 3) => (n_link + n_h + n_det, 3)
        * hypotheses_states:             (n_h, depth) => (n_link + n_h + n_det, depth + 1)
        * hypotheses_positions: (n_h, depth + 1, dim) => (n_link + n_h + n_det, depth + 1, dim)
        * last_detections:                      depth => depth + 1

        Args:
            frame (np.ndarray): Current frame
            detections (byotrack.Detections): Detections on the frame

        """
        n_h, depth = self.hypotheses_indices.shape

        # Sanity checks
        if self.debug:
            assert len(self.last_detections) == depth
            assert depth < self.specs.tree_depth

        # Register detections
        self.last_detections.append(detections)

        links, costs = self.linking_cost(frame, detections)
        if self.debug:
            assert (self.hypotheses_states[:, -1][links[:, 0]] == 0).all(), "Do not link with non-active hypotheses"
            assert self.hypotheses_positions.shape == (n_h, depth + 1, detections.dim)

        self.hypotheses_indices, self.hypotheses_positions, self.hypotheses_costs, self.hypotheses_states = (
            _hypothesize(
                self.hypotheses_indices,
                self.hypotheses_positions,
                self.hypotheses_costs,
                self.hypotheses_states,
                links=links,
                costs=costs,
                positions=detections.position.to(self.device),
                max_likelihood=self.max_likelihood(),
                fnr=self.specs.fnr,
                death_rate=self.specs.death_rate,
                lambda_b=self.specs.lambda_b_fixed(self._n_0_est, len(detections), self.frame_id),
                lambda_f=self.specs.lambda_f(len(detections)),
                volume=self._volume,
                false_tracks=self.specs.false_tracks,
            )
        )

        # Optionnally set up internal states (such as KF)
        self.hypothesize_states(links)

        if self.debug:
            n_h = n_h + len(links) + len(detections)
            assert self.hypotheses_positions.shape == (n_h, depth + 1, detections.dim)
            assert self.hypotheses_indices.shape == (n_h, depth + 1)
            assert self.hypotheses_states.shape == (n_h, depth + 1)
            assert self.hypotheses_costs.shape == (n_h, 3)
            # Check states are correctly set ?

    def filter_hypotheses(self):
        """Post association handling of hypotheses.

        `associate` has solved MAP for the reference frame t using frames up to t + tree_depth - 1.
        This method moves the reference frame to t + 1, reducing the depth of all the hypotheses of one
        and keeping only those compatible with the selected ones.

        It will also store the selected links from t to t+1 inside the TrackHandlers.

        Precisely it goes through the following steps:

        1. Check which are the hypotheses selected and kept, and how it will impact `active_tracks`
        2. Maps track id linked to a det id (i, j) into a new active track id i2
        3. Update `active_tracks` (first continuation, then creation) (See `_filter_handlers`)
        4. Update internal states of kept tracks (See `update_states`)
        5. Record TrackHandlers' positions in `all_positions` (See `record_active_positions`)
        6. Filter the hypotheses (Optionally, see `filter_states`)

        In terms of shape, with n_kept = n_h(t + 1) we have

        * hypotheses_indices:        (n_link + n_h + n_det, depth + 1) => (n_kept, depth)
        * hypotheses_costs:                    (n_link + n_h + n_det,) => (n_kept,)
        * hypotheses_states:         (n_link + n_h + n_det, depth + 1) => (n_kept, depth)
        * hypotheses_positions: (n_link + n_h + n_det, depth + 1, dim) => (n_kept, depth, dim)
        * last_detections:                                   depth + 1 => depth

        """

        # Select corresponding indices, states and hypotheses index
        selected_indices = self.hypotheses_indices[self.selected, :2]
        selected_states = self.hypotheses_states[self.selected, 1]
        selected_hypotheses = torch.arange(len(self.hypotheses_indices), dtype=torch.int32, device=self.device)[
            self.selected
        ]

        # Check which hypothesis correspond to creation, continuation, termination and false tracks
        empty_track = selected_indices[:, 0] == -1
        empty_det = selected_indices[:, 1] == -1
        creation = empty_track & ~empty_det
        continuation = ~empty_track & (selected_states == 0)
        termination = ~empty_track & (selected_states == 1)
        invalid = ~empty_track & (selected_states == 2)

        # Build a mapping from (i, j) to the new track_id (-2 if not kept)
        mapping = torch.full(
            (len(self.active_tracks) + 1, self.last_detections[1].length + 1), -2, dtype=torch.int32, device=self.device
        )
        mapping[-1, -1] = -1  # -1 for tracks to be created
        mapping[selected_indices[continuation, 0], selected_indices[continuation, 1]] = torch.arange(
            continuation.sum().item(), dtype=torch.int32, device=self.device
        )  # First continuation
        mapping[selected_indices[creation, 0], selected_indices[creation, 1]] = continuation.sum() + torch.arange(
            creation.sum().item(), dtype=torch.int32, device=self.device
        )  # Then creation

        # Map hypotheses indices. Hypotheses to remove are mapped onto -2
        new_indices = mapping[self.hypotheses_indices[:, 0], self.hypotheses_indices[:, 1]]
        kept = new_indices != -2

        # Update active/inactive tracks and build the active track hypotheses index tensor
        self._filter_handlers(selected_indices, creation, continuation, termination, invalid)
        active_hypotheses = torch.cat((selected_hypotheses[continuation], selected_hypotheses[creation]))

        # Update internal states & register positions & filter hyp states
        self.update_states(kept)
        self.record_active_positions(active_hypotheses)
        self.filter_states(kept)

        # Filter hypotheses & detections
        self.last_detections = self.last_detections[1:]
        self.selected = self.selected[kept]
        self.hypotheses_indices[:, 1] = new_indices
        self.hypotheses_indices = self.hypotheses_indices[kept, 1:]
        self.hypotheses_costs = self.hypotheses_costs[kept]
        self.hypotheses_states = self.hypotheses_states[kept, 1:]
        self.hypotheses_positions = self.hypotheses_positions[kept, 1:]

    def _filter_handlers(
        self,
        selected_indices: torch.Tensor,
        creation: torch.Tensor,
        continuation: torch.Tensor,
        termination: torch.Tensor,
        invalid: torch.Tensor,
    ):
        """Update handlers, filter inactive ones and create new ones

        Args:
            selected_indices (torch.Tensor): Links (i,j) between active tracks and detections.
                Shape: (n, 2), dtype: int32
            creation (torch.Tensor): Indicates which links correspond to track handler creation.
                Shape: (n,), dtype: bool
            continuation (torch.Tensor): Indicates which links correspond to track continuation.
                Shape: (n,), dtype: bool
            termination (torch.Tensor): Indicates which links correspond to track termination.
                Shape: (n,), dtype: bool
            invalid (torch.Tensor): Indicates which links correspond to a termination into an false positive track.
                Shape: (n,), dtype: bool
        """
        # Build indices tensors onto cpu
        creation_indices = selected_indices[creation].cpu()
        continuation_indices = selected_indices[continuation].cpu()
        termination_indices = selected_indices[termination].cpu()

        frame_id = self.frame_id - len(self.last_detections) + 2

        active_tracks = []
        i: int
        j: int
        for i, j in continuation_indices.tolist():
            self.active_tracks[i].update(frame_id, j, TrackHandler.TrackState.ACTIVE)
            active_tracks.append(self.active_tracks[i])

        for i, j in creation_indices.tolist():
            track = TrackHandler(frame_id, self._next_identifier, self.debug)
            self._next_identifier += 1
            track.update(frame_id, j, TrackHandler.TrackState.ACTIVE)
            active_tracks.append(track)

        for i, j in termination_indices.tolist():
            self.active_tracks[i].update(frame_id, j, TrackHandler.TrackState.FINISHED)
            self.inactive_tracks.append(self.active_tracks[i])

        if self.save_all:
            invalid_indices = selected_indices[invalid].cpu()
            for i, j in invalid_indices.tolist():
                self.active_tracks[i].update(frame_id, j, TrackHandler.TrackState.INVALID)
                self.inactive_tracks.append(self.active_tracks[i])

        self.active_tracks = active_tracks

    def associate(self):
        """Select hypotheses by solving a MAP

        It updates the `selected` boolean tensor.
        """
        if self._last_solve == self.frame_id:  # Nothing to do: Already solved
            return

        self._last_solve = self.frame_id

        # The hypothesis cost is minimum cost over the three states:
        cost = self.hypotheses_costs.min(dim=1).values
        if self.specs.false_tracks:  # Add bias toward track costs (to sort false positives)
            cost += self.hypotheses_costs[:, 0] * 0.001

        self.selected = self.specs.solver.solve(
            cost,
            self.hypotheses_indices,
            [len(self.active_tracks)] + [len(detections) for detections in self.last_detections[1:]],
        )

        if self.debug:
            for depth in range(self.hypotheses_indices.shape[1]):
                _, count = torch.unique(self.hypotheses_indices[self.selected, depth], return_counts=True)
                assert (count > 1).sum() == 1, "1-to-1 association not respected"

    def update(self, frame: np.ndarray, detections: byotrack.Detections) -> None:
        if self.frame_id == -1:
            # Let's reset again just in case with the right dim
            self.reset(detections.dim)
            self._n_0_est = self.specs.n_0_est(len(detections), 0)
            # Set to the initial volume
            self._volume = float(np.prod(frame.shape[:-1])) if self.specs.volume <= 0.0 else self.specs.volume

            if self.debug:
                print("=================Estimations=================")
                print("N0: ", self._n_0_est)
                print("V:", self._volume)
                print("Eta:", self.specs.threshold(len(detections), self._volume, detections.dim))
                print(
                    "Lambda_b",
                    self.specs.lambda_b(len(detections)),
                    self.specs.lambda_b_fixed(self._n_0_est, len(detections), 0),
                )
                print("Lambda_f", self.specs.lambda_f(len(detections)))

        self.frame_id += 1

        if self.debug:
            print(f"========================{self.frame_id}==========================")
            self._log()

        # Compute the flow map if optflow given
        if self.optflow is not None:
            self.optflow.update(frame)

        self.motion_model()
        detections = self.update_detections(detections)
        self._log()

        # Compute features if the extractor is given and register inside the detections
        # Do not recompute the features if some are already registered
        remove_feats = False
        if self.features_extractor is not None:
            if "features" in detections.data:
                warnings.warn("Some features are already computed. They will be used.")
            else:
                remove_feats = True
                self.features_extractor.register(frame, detections)

        self.hypothesize(frame, detections)
        self._log()

        if self.hypotheses_indices.shape[1] == self.specs.tree_depth:
            self.associate()
            self._log()

            for _ in range(min(self.specs.tree_depth - 1, self.specs.step)):
                # Reduces hypotheses tree and save into self.[in]active_tracks
                self.filter_hypotheses()

                assert len(self.all_positions[-1]) == len(self.active_tracks)

                # Register track_id
                for i, track in enumerate(self.active_tracks):
                    track.register_track_id(i)
        else:
            # Keep all hypotheses but update states
            self.update_states(torch.full((len(self.hypotheses_indices),), True, device=self.device))

        self._log()

        # Remove the computed features if save_all is False
        if not self.save_all and remove_feats:
            detections.data.pop("features")

    def _log(self):
        if self.debug:
            print(
                self.hypotheses_indices.shape,
                self.hypotheses_costs.shape,
                self.hypotheses_states.shape,
                self.hypotheses_positions.shape,
                len(self.last_detections),
                len(self.last_detections[-1]),
                self.specs.association_threshold,
                (self.hypotheses_states[:, -1] != 0).sum(),
            )
