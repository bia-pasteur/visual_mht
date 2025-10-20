"""Load the saved patched into a usable dataset"""

from __future__ import annotations

import dataclasses
import os
import pathlib
import random
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import torch.utils.data

from . import augmentations


# Has to do some tricks for python < 3.10
# to support inheritance with default values in dataclasses
@dataclasses.dataclass
class _DatasetConfig:
    root: pathlib.Path
    val_root: pathlib.Path
    video_ids: List[str]
    delta_t: int = 0


@dataclasses.dataclass
class DatasetConfig(augmentations.AugmentationConfig, _DatasetConfig):
    def build_patch_dataset(self, train=True) -> PatchDataset:
        return PatchDataset(self.root, self.video_ids, self.train_common() if train else self.test())

    def build_triplet_dataset(self) -> TripletPatchDataset:
        return TripletPatchDataset(self.build_patch_dataset(train=True), self.train_specific(), self.delta_t)

    def build_siamese_dataset(self) -> SiameseDataset:
        return SiameseDataset(self.build_patch_dataset(train=True), self.train_specific(), self.delta_t)

    def build_val_dataset(self) -> ValPatchDataset:
        return ValPatchDataset(self.val_root, self.test())


T = TypeVar("T")


def sorted_alphanumeric(data: Iterable[T]) -> List[T]:
    """Sorts alphanumeriacally an iterable of strings-like objects

    "1" < "2" < "10" < "foo1" < "foo2" < "foo3"
    """

    def convert(text: str):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return tuple(convert(c) for c in re.split("([0-9]+)", str(key)))

    return sorted(data, key=alphanum_key)


def identity(inpt):
    return inpt


class PatchDataset(torch.utils.data.Dataset):
    """Dataset of patches, load patches and apply the given transformation

    It preloads all the patches in RAM.

    Attributes:
        root (pathlib.Path): Path to the root folder of the dataset
        video_ids (List[str]): Select specific videos inside the root folder.
            By default, all the videos found inside root are used.
        transforms (Callable): A callable to transform a patch when accessed
        samples (torch.Tensor): The concatenated patches.
            Shape: (N, T, H, W, C), dtype: uint8
        specs (List[Tuple[int, int]]): Video_id and frame_id for all patches.
        samples_range (Dict[int, Dict[int, Tuple[int, int]]]): Idx range for each (video, frame)
    """

    def __init__(
        self,
        root: Union[str, os.PathLike],
        video_ids: Optional[List[str]] = None,
        transforms: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        super().__init__()
        self.root = pathlib.Path(root)
        self.transforms = transforms if transforms is not None else identity
        self.video_ids = video_ids if video_ids else []
        if not self.video_ids:
            for video_path in sorted_alphanumeric(self.root.iterdir()):
                if video_path.is_dir():
                    self.video_ids.append(video_path.stem)

        # Load samples and specs
        self.samples_range: Dict[int, Dict[int, Tuple[int, int]]] = {
            video_id: {} for video_id in range(len(self.video_ids))
        }
        self.specs: List[Tuple[int, int]] = []
        samples = []

        num_samples = 0
        for i, video_id in enumerate(self.video_ids):
            for frame_path in sorted_alphanumeric((self.root / video_id).glob("patches_*.pt")):
                frame_id = int(frame_path.stem.split("_")[-1])
                patches = torch.load(frame_path)

                self.specs.extend((i, frame_id) for _ in range(len(patches)))
                samples.append(patches)
                self.samples_range[i][frame_id] = (num_samples, num_samples + len(patches))

                num_samples += len(patches)

        # Store in a large tensor
        self.samples = torch.cat(samples)
        self.invalid = (self.samples == 0).all(dim=(2, 3, 4))

    def __getitem__(self, idx):
        return self.transforms(self.samples[idx,])

    def __len__(self) -> int:
        return len(self.specs)


class TripletPatchDataset(torch.utils.data.Dataset):
    """Triplet dataset of patch

    Randomly yields triplets of patches.

    Attributes:
        dataset (PatchDataset): The underlying data
        transforms (Callable): Random transforms to apply to the patches
            They will be apply to the sample to create two views (anchor and positive)
            and also to a negative sample.
        delta_t (int): Use temporal augmentation up to this maximum time offset*
            Default: 0 (Disable temporal augmentation)

    """

    def __init__(self, dataset: PatchDataset, transforms: Callable, delta_t=0):
        super().__init__()
        self.dataset = dataset
        self.transforms = transforms
        self.delta_t = min(delta_t, dataset.samples.shape[1] // 2)

    def __getitem__(self, idx: int) -> tuple:
        patch = self.dataset.samples[idx]
        temporal_choices = [self.dataset.samples.shape[1] // 2 + k for k in range(-self.delta_t, self.delta_t + 1)]
        # Filter invalid temporal augmentation (for untracked object for instance)
        temporal_choices = [t for t in temporal_choices if ~self.dataset.invalid[idx, t]]

        # Let's choose potential frame among those available (with replacement)
        temporal_choices = random.choices(temporal_choices, k=2)
        patch = patch[temporal_choices]

        # Apply PatchDataset transforms to both view and shared (if flip both flip)
        patch = self.dataset.transforms(patch)

        # Apply random transform to each patch
        anchor = self.transforms(patch[0])
        positive = self.transforms(patch[1])

        # Now let's sample a negative from another patch
        # Either same frame same video, or different video and any frame
        video_id, frame_id = self.dataset.specs[idx]
        negative_video = random.randint(0, len(self.dataset.video_ids) - 1)
        if negative_video != video_id:  # Choose a random frame_id if different video
            frame_id = random.choice(list(self.dataset.samples_range[negative_video].keys()))

        start, stop = self.dataset.samples_range[negative_video][frame_id]
        neg_idx = random.randint(start, stop - 1)
        while neg_idx == idx:  # Let's just loop until we find someone different
            neg_idx = random.randint(start, stop - 1)

        # Temporal augmentation
        temporal_choices = [self.dataset.samples.shape[1] // 2 + k for k in range(-self.delta_t, self.delta_t + 1)]

        # Filter invalid temporal augmentation (for untracked object for instance)
        temporal_choices = [t for t in temporal_choices if ~self.dataset.invalid[neg_idx, t]]

        negative = self.dataset.samples[neg_idx, random.choice(temporal_choices)]

        # Apply negative transforms
        negative = self.transforms(self.dataset.transforms(negative))

        return anchor, positive, negative

    def __len__(self):
        return len(self.dataset)


class SiameseDataset(TripletPatchDataset):
    """Yield two augmented views for each sample"""

    def __getitem__(self, idx: int) -> tuple:
        patch = self.dataset.samples[idx]
        temporal_choices = [self.dataset.samples.shape[1] // 2 + k for k in range(-self.delta_t, self.delta_t + 1)]
        # Filter invalid temporal augmentation (for untracked object for instance)
        temporal_choices = [t for t in temporal_choices if ~self.dataset.invalid[idx, t]]

        # Let's choose potential frame among those available (with replacement)
        temporal_choices = random.choices(temporal_choices, k=2)
        patch = patch[temporal_choices]

        # Apply PatchDataset transforms to both view and shared (if flip both flip)
        patch = self.dataset.transforms(patch)

        # Apply random transform to each patch
        return self.transforms(patch[0]), self.transforms(patch[1])


class ValPatchDataset(torch.utils.data.Dataset):
    """Annotated dataset of pairs of patches.

    For a given pair of frames (f_i, f_j), a spatial ROI is fully (and manually) instance-segmented into background
    and neurons. Let N be the number of neurons found inside the ROI. For each neuron k, a patch P_ik (resp. P_jk) is
    extracted on frame f_i (resp. f_j). Also, on each frame, we complete the ground-truth segmentation
    with Wavelet-based detection outside of the ROI. For these additional found neurons a patch is extracted on their
    respective frame (and we do not know if any of these neurons can be linked with another one from the other frame).

    For any patch in the dataset, the labels are the pair_id, the neurons_id in the pair and a boolean stating if the
    neuron_id in the pair is shared.
    """

    def __init__(
        self,
        root: Union[str, os.PathLike],
        transforms: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        super().__init__()
        self.root = pathlib.Path(root)
        self.transforms = transforms if transforms is not None else identity

        self.video_ids = []
        self.specs: List[Tuple[int, int, int, int]] = []
        samples = []
        self.annotated = []

        video_id = 0
        pair_id = 0
        for video_path in sorted_alphanumeric(self.root.iterdir()):
            if not video_path.is_dir():
                continue

            self.video_ids.append(video_path.stem)

            for pair_path in sorted_alphanumeric(video_path.iterdir()):
                if not video_path.is_dir():
                    continue

                if not (pair_path / "annotated.txt").exists():
                    continue

                self.annotated.append(int((pair_path / "annotated.txt").read_text()))

                for frame_path in sorted_alphanumeric(pair_path.glob("patches_*.pt")):
                    frame_id = int(frame_path.stem.split("_")[-1])
                    patches = torch.load(frame_path)

                    self.specs.extend((video_id, pair_id, frame_id, k) for k in range(len(patches)))
                    samples.append(patches)

                pair_id += 1

            video_id += 1

        self.samples = torch.cat(samples)

    def __getitem__(self, idx):
        video_id, pair_id, frame_id, patch_id = self.specs[idx]

        return (
            self.transforms(self.samples[idx]),
            video_id,
            pair_id,
            frame_id,
            patch_id if patch_id < self.annotated[pair_id] else -1,
        )

    def __len__(self) -> int:
        return len(self.specs)
