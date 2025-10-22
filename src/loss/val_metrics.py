from typing import cast, Any, Iterable, List, Tuple
from deep_trainer.pytorch.metric import Metric, Prerequisite

import numpy as np
import torch

import pylapy


def make_increasing(points: Iterable[float]) -> List[float]:
    """Used to make monotone recalls and precision"""
    increasing = []
    max_point = 0.0
    for point in points:
        if point > max_point:
            max_point = point

        increasing.append(max_point)

    return increasing


def compute_ap(recalls: List[float], precisions: List[float]) -> float:
    """Compute average precision with AP = \\sum_k [R_{k+1} - R_k] * P_{k+1}

    It will underestimate the true area under the curve as we use P_{k+1} (P_k will overestimate)
    in the sum. (This is diffferent than what used Reme and al, for tracklet stitching)

    It also handle the fact that in our case recall is not monotone nor goes up to 1.

    Args:
        recalls (List[float]): Recall list for each level of cost limit
        precision (List[float]) Precision list for each level of cost limit

    Returns:
        float: Average precision metrics (precision is set to 0 for each point without recall)
    """
    # Let's put the first point as 1 of precision for 0 of recall by default (when predicting no links)
    recalls = [0.0] + make_increasing(recalls)
    precisions = [1.0] + list(reversed(make_increasing(reversed(precisions))))

    average_precision = 0.0
    for i, precision in enumerate(precisions[1:]):
        average_precision += precision * (recalls[i + 1] - recalls[i])

    return average_precision


class OutputsSaver(Prerequisite):
    """Save the outputs of the network

    It should not be used with large datasets
    """

    def __init__(self) -> None:
        super().__init__()
        self._outputs: List[torch.Tensor] = []
        self.outputs = torch.tensor([])

    def update(self, _: Any, outputs: torch.Tensor) -> None:
        self._outputs.append(outputs.detach().cpu())

    def aggregate(self) -> None:
        self.outputs = torch.cat(self._outputs)

    def reset(self) -> None:
        self._outputs = []
        self.outputs = torch.tensor([])


class ValidationTargetsSaver(Prerequisite):
    """Save the validation targets (pair_ids, frame_ids, patch_ids)

    Assume that batches are (inputs, targets)

    It should not be used with large datasets
    """

    def __init__(self) -> None:
        super().__init__()
        self._video_ids: List[torch.Tensor] = []
        self._pair_ids: List[torch.Tensor] = []
        self._frame_ids: List[torch.Tensor] = []
        self._patch_ids: List[torch.Tensor] = []
        self.video_ids = torch.tensor([])
        self.pair_ids = torch.tensor([])
        self.frame_ids = torch.tensor([])
        self.patch_ids = torch.tensor([])
        self.pairs = torch.tensor([])

    def update(
        self, batch: Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], outputs: Any
    ) -> None:
        _, (video_ids, pair_ids, frame_ids, patch_ids) = batch
        self._video_ids.append(video_ids.detach().cpu())
        self._pair_ids.append(pair_ids.detach().cpu())
        self._frame_ids.append(frame_ids.detach().cpu())
        self._patch_ids.append(patch_ids.detach().cpu())

    def aggregate(self) -> None:
        self.video_ids = torch.cat(self._video_ids)
        self.pair_ids = torch.cat(self._pair_ids)
        self.frame_ids = torch.cat(self._frame_ids)
        self.patch_ids = torch.cat(self._patch_ids)
        self.pairs = cast(torch.Tensor, torch.unique(self.pair_ids))

    def reset(self) -> None:
        self._video_ids = []
        self._pair_ids = []
        self._frame_ids = []
        self._patch_ids = []
        self.video_ids = torch.tensor([])
        self.pair_ids = torch.tensor([])
        self.frame_ids = torch.tensor([])
        self.patch_ids = torch.tensor([])
        self.pairs = torch.tensor([])


class PatchDist(Prerequisite):
    """Compute the distance between each sample in the validation data

    It should not be used with large datasets.
    """

    def __init__(
        self,
        outputs: OutputsSaver,
        targets: ValidationTargetsSaver,
        use_wavelet=True,
        across_video=True,
        across_pairs=True,
        miss_prop=0.0,  # Allow for miss detections?
    ) -> None:
        super().__init__()
        self.prerequisites.add(outputs)
        self.prerequisites.add(targets)
        self.outputs = outputs
        self.targets = targets
        self.use_wavelet = use_wavelet
        self.across_video = across_video
        self.across_pairs = across_pairs
        self.miss_prop = miss_prop
        self.gt_mask = torch.full((0,), True)
        self.detected_mask = torch.full((0,), True)
        self.left_mask = torch.full((0,), False)
        self.dist = torch.zeros((0, 0))
        self.ground_truth = torch.zeros((0, 0))

    def update(self, batch: Any, outputs: Any) -> None:
        pass

    def aggregate(self) -> None:
        features = self.outputs.outputs
        video_ids = self.targets.video_ids
        pair_ids = self.targets.pair_ids
        frame_ids = self.targets.frame_ids
        patch_ids = self.targets.patch_ids

        self.gt_mask = torch.full((len(features),), True)
        if not self.use_wavelet:
            self.gt_mask = self.targets.patch_ids != -1
            features = features[self.gt_mask]
            video_ids = video_ids[self.gt_mask]
            pair_ids = pair_ids[self.gt_mask]
            frame_ids = frame_ids[self.gt_mask]
            patch_ids = patch_ids[self.gt_mask]

        self.detected_mask = torch.full((len(features),), True)
        if self.miss_prop > 0.0:
            self.detected_mask = torch.rand(len(features)) >= self.miss_prop
            features = features[self.detected_mask]
            video_ids = video_ids[self.detected_mask]
            pair_ids = pair_ids[self.detected_mask]
            patch_ids = patch_ids[self.detected_mask]
            frame_ids = frame_ids[self.detected_mask]

        # Split the samples into left and right for each pairs
        self.left_mask = torch.full((len(features),), False)

        for pair_id in self.targets.pairs:
            frame_0 = frame_ids[pair_ids == pair_id][0]
            self.left_mask[(pair_ids == pair_id) & (frame_ids == frame_0)] = True

        self.dist = torch.cdist(
            features[self.left_mask],  # Shape: (N_1, D)
            features[~self.left_mask],  # Shape: (N_2, D)
        )

        # Compute useful booleans masks
        same_video = video_ids[self.left_mask][:, None] == video_ids[~self.left_mask][None]
        same_pair = pair_ids[self.left_mask][:, None] == pair_ids[~self.left_mask][None]
        same_patch = patch_ids[self.left_mask][:, None] == patch_ids[~self.left_mask][None]

        # Some links are unknown because it comes from the imperfect wavelet detections (patch_ids == -1)
        half_unknown = ((patch_ids[self.left_mask][:, None] != -1) & (patch_ids[~self.left_mask][None] == -1)) | (
            (patch_ids[self.left_mask][:, None] == -1) & (patch_ids[~self.left_mask][None] != -1)
        )
        full_unknown = (patch_ids[self.left_mask][:, None] == -1) & (patch_ids[~self.left_mask][None] == -1)

        # Ground truth links are the det thats shares the same pair and patch identifier
        # The rest of the links are either 0 or unknown. A link should not be made (ground_truth[i, j] = 0) if
        # i and j are from different videos or if they are from the same pair but one from the added wavelet detection
        # and the other from the ground truth detections. Currently, in the dataset there is no overlap between two
        # annotated pair in the same video. Therefore we also set at 0 if i and j are ground truths detections from
        # different pairs but from the same video. (Disabled if across_pair is False)
        # Finally, there are some really hard unknown links between ground truth det and wavelet dets of two different
        # pairs in the same video: The annotated neurons of the first pair are likely to be in the wavelet detections
        # of the other pair. Therefore we do not allow the linking between those two (dist = inf)
        self.ground_truth = 1.0 * (same_pair & same_patch)
        self.ground_truth[same_video & (full_unknown | half_unknown & ~same_pair)] = torch.nan

        # Set dist to inf for the potential true links unknown
        self.dist[same_video & ~same_pair & half_unknown] = torch.inf

        if not self.across_pairs:  # Disable links between pairs in the same video
            self.dist[~same_pair & same_video] = torch.inf
            self.ground_truth[~same_pair & same_video] = torch.nan  # It means those links are unknown

        if not self.across_video:  # Disable links between videos
            self.dist[~same_video] = torch.inf

    def reset(self) -> None:
        self.gt_mask = torch.full((0,), True)
        self.detected_mask = torch.full((0,), True)
        self.left_mask = torch.full((0,), False)
        self.dist = torch.zeros((0, 0))
        self.ground_truth = torch.zeros((0, 0))


class LinkSolver(Prerequisite):
    """Compute the links at different dist thresholds"""

    EVALUATE_EVERY = 10

    def __init__(self, dist: PatchDist) -> None:
        super().__init__()
        self.prerequisites.add(dist)
        self.dist = dist
        self.solver = pylapy.LapSolver("lap")
        self.cost_limits = [k / 10 for k in range(11)] + [float("inf")]
        self.links: List[torch.Tensor] = []
        self.f1: List[float] = []
        self.recall: List[float] = []
        self.precision: List[float] = []

        self.last_call = 0

    def update(self, batch: Any, outputs: Any) -> None:
        pass

    def aggregate(self) -> None:
        self.last_call += 1
        if self.last_call < self.EVALUATE_EVERY:
            return

        self.last_call = 0

        max_value = self.dist.dist[self.dist.dist != torch.inf].median()

        for cost_limit in self.cost_limits:
            if self.dist.across_video:
                links = torch.tensor(self.solver.solve(self.dist.dist.numpy(), cost_limit * max_value).astype(np.int32))
            else:
                links_ = []
                n = torch.tensor([0, 0], dtype=torch.int32)

                if self.dist.across_pairs:  # Solve by video
                    ids = self.dist.targets.video_ids[self.dist.gt_mask][self.dist.detected_mask]
                else:  # Solve by pair
                    ids = self.dist.targets.pair_ids[self.dist.gt_mask][self.dist.detected_mask]

                for id_ in torch.unique(ids):
                    dist = self.dist.dist[ids[self.dist.left_mask] == id_][:, ids[~self.dist.left_mask] == id_]
                    links = torch.tensor(self.solver.solve(dist.numpy(), cost_limit * max_value).astype(np.int32))
                    links_.append(links + n)
                    n += torch.tensor(dist.shape)

                links = torch.cat(links_)

            links_valid = self.dist.ground_truth[links[:, 0], links[:, 1]]
            links_valid = links_valid[~torch.isnan(links_valid)]  # Ignore unknown links (If use_wavelet=True)

            recall = links_valid.sum().item() / (self.dist.ground_truth == 1).sum().item()
            precision = links_valid.mean().item() if links_valid.numel() else 1.0
            f1 = 2 * precision * recall / (precision + recall + 1e-8)

            self.links.append(links)
            self.f1.append(f1)
            self.recall.append(recall)
            self.precision.append(precision)

    def reset(self) -> None:
        self.links = []
        self.f1 = []
        self.recall = []
        self.precision = []


class LinkAP(Metric):
    """Average precision of the links retrieval task"""

    def __init__(self, links: LinkSolver):
        super().__init__(train=False, minimize=False)
        self.prerequisites.add(links)
        self.links = links

    def update(self, batch: Any, outputs: Any) -> None:
        pass

    def aggregate(self) -> float:
        if not self.links.links:
            return 0.0

        return compute_ap(self.links.recall, self.links.precision)


class LinkBestF1(Metric):
    """Best F1 found for the links retrieval task"""

    def __init__(self, links: LinkSolver):
        super().__init__(train=False, minimize=False)
        self.prerequisites.add(links)
        self.links = links

    def update(self, batch: Any, outputs: Any) -> None:
        pass

    def aggregate(self) -> float:
        if not self.links.links:
            return 0.0

        return max(self.links.f1)


class LinkBestF1Recall(Metric):
    """Recall for the best F1 score in the links retrieval task"""

    def __init__(self, links: LinkSolver):
        super().__init__(train=False, minimize=False)
        self.prerequisites.add(links)
        self.links = links

    def update(self, batch: Any, outputs: Any) -> None:
        pass

    def aggregate(self) -> float:
        if not self.links.links:
            return 0.0

        return self.links.recall[np.argmax(self.links.f1)]


class LinkBestF1Precision(Metric):
    """Precision for the best F1 score in the links retrieval task"""

    def __init__(self, links: LinkSolver):
        super().__init__(train=False, minimize=False)
        self.prerequisites.add(links)
        self.links = links

    def update(self, batch: Any, outputs: Any) -> None:
        pass

    def aggregate(self) -> float:
        if not self.links.links:
            return 0.0

        return self.links.precision[np.argmax(self.links.f1)]


class HardestTripletLoss(Metric):
    """Triplet loss averaged on each hardest triplet in the dataset"""

    def __init__(self, dist: PatchDist, margin=1.0):
        super().__init__(train=False)
        self.prerequisites.add(dist)
        self.dist = dist
        self.margin = margin

    def update(self, batch: Any, outputs: Any) -> None:
        pass

    def aggregate(self) -> float:
        dist = self.dist.dist.clone()
        dist[self.dist.ground_truth == 1.0] = torch.inf
        indices = torch.tensor(np.indices(dist.shape))[:, self.dist.ground_truth == 1.0]

        cum_loss = torch.tensor(0.0)
        n = 0
        prop = 0
        for i, j in indices.T:
            mini_i = dist[:, j].min()
            mini_j = dist[i, :].min()

            loss = torch.relu(self.dist.dist[i, j] - mini_i + self.margin)
            if loss > 0.0:
                prop += 1
            cum_loss += loss

            loss = torch.relu(self.dist.dist[i, j] - mini_j + self.margin)
            if loss > 0.0:
                prop += 1
            cum_loss += loss
            n += 2

        return cum_loss.item() / n  # prop / n
