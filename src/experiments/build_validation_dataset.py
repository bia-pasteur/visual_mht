"""Build a similar validation dataset for Sinetra"""

import dataclasses
import enum
import os
import pathlib
from typing import Dict, List, Tuple, Sequence

import dacite
import numpy as np
import torch
import yaml

import byotrack
import pylapy

from ..data.video import VideoConfig
from ..data import sinetra_data
from ..data import trasein_annotations
from ..detect import DetectionConfig
from .build_dataset import extract_at_frame_id


@dataclasses.dataclass
class ExperimentConfig:
    """Config of the experiments"""

    video: VideoConfig
    detection: DetectionConfig
    pairs: List[trasein_annotations.PairConfig]
    roi_size: int
    patch_size: int


def relabel_detections(detections_sequence: Sequence[byotrack.Detections]) -> Sequence[byotrack.Detections]:
    """Relabel GT detections to match indices of the first det"""
    reference = detections_sequence[0]
    relabeled_sequence = [reference]

    for detections in detections_sequence[1:]:
        # Links from detections to reference
        links = pylapy.LapSolver().solve(torch.cdist(detections.position, reference.position).numpy())
        mapping = torch.zeros(len(links) + 1, dtype=torch.int32)
        mapping[1:] = torch.tensor(links[:, 1].astype(np.int32)) + 1

        # Apply mapping
        relabeled_sequence.append(byotrack.Detections({"segmentation": mapping[detections.segmentation]}))

    return relabeled_sequence


def merge_detections(
    detections: byotrack.Detections, detections_2: byotrack.Detections, loc: Tuple[int, int], roi_size: int
) -> byotrack.Detections:
    # Add some margin to the roi mask
    roi_mask = torch.full(detections.shape, False, dtype=torch.bool)
    roi_mask[loc[0] - 5 : loc[0] + 5 + roi_size, loc[1] - 5 : loc[1] + 5 + roi_size] = True

    # Mask seg_2 and offset by N
    seg_2 = detections_2.segmentation.clone()
    seg_2[roi_mask] = 0
    seg_2[seg_2 != 0] += len(detections) + 10  # With offset just in case

    # Overwrite seg_2 with seg (should be in mask)
    seg = detections.segmentation
    seg_2[seg != 0] = seg[seg != 0]

    return byotrack.Detections({"segmentation": seg_2})  # Will relabel_consecutive, while keeping the order


def main_sinetra(name: str, cfg_data: Dict):
    """Use SINETRA ground-truth tracks to build a similar validation dataset."""
    print("Running:", name)
    print(yaml.dump(cfg_data))

    cfg = dacite.from_dict(ExperimentConfig, cfg_data, dacite.Config(cast=[pathlib.Path, tuple, enum.Enum]))

    # Open video
    video = cfg.video.open()

    # Load GT in sinetra
    gt_positions = sinetra_data.load_ground_truth(cfg.video.path.parent)["mu"]

    # Store in root
    root = pathlib.Path(os.environ.get("EXPYRUN_CWD", ".")) / "data" / name
    root.mkdir(parents=True, exist_ok=False)

    # Process each pair
    for i, pair in enumerate(cfg.pairs):
        print(f"Processing pair: {pair.frames}")

        (root / f"{i}").mkdir()

        for frame_id in pair.frames:
            patches = extract_at_frame_id(
                frame_id, torch.from_numpy(video[frame_id])[None], gt_positions[frame_id, :, None], cfg.patch_size
            )
            torch.save(patches[:, 0], root / f"{i}" / f"patches_{frame_id}.pt")

        # Store the number of dets in patches that we know should match
        (root / f"{i}" / "annotated.txt").write_text(f"{gt_positions.shape[1]}")


def main_trasein(name: str, cfg_data: Dict):
    """Use our segmentation annotation to build a validation dataset."""
    print("Running:", name)
    print(yaml.dump(cfg_data))

    cfg = dacite.from_dict(ExperimentConfig, cfg_data, dacite.Config(cast=[pathlib.Path, tuple, enum.Enum]))

    # Open video
    video = cfg.video.open()
    shape = (video[0].shape[0], video[0].shape[1])  # H, W

    # Detector
    detector = cfg.detection.build()

    # Store in root
    root = pathlib.Path(os.environ.get("EXPYRUN_CWD", ".")) / "data" / name
    root.mkdir(parents=True, exist_ok=False)

    # Process each pair
    for i, pair in enumerate(cfg.pairs):
        print(f"Processing pair: {pair.frames}")

        # Load annotation
        detections_pair = trasein_annotations.load_trasein_annotations(cfg.video.video_id, pair, shape)
        num_det = detections_pair[0].length

        if not all(len(det) == num_det for det in detections_pair):
            print(f"Number of annotated neurons differs ({num_det} != {len(detections_pair[1])})), skipping...")
            continue

        # Relabel to match detection indices
        detections_pair = relabel_detections(detections_pair)

        # Run imperfect detection process for outside of the ROI
        noisy_detections_pair = detector.run([video[frame] for frame in pair.frames])
        detections_pair = [
            merge_detections(det_1, det_2, pair.roi_loc, cfg.roi_size)
            for det_1, det_2 in zip(detections_pair, noisy_detections_pair)
        ]

        print(f"Only {num_det} associations have been labeled out of ~{detections_pair[0].length}")

        (root / f"{i}").mkdir()

        for frame_id, detections in zip(pair.frames, detections_pair):
            patches = extract_at_frame_id(
                frame_id, torch.from_numpy(video[frame_id])[None], detections.position[:, None], cfg.patch_size
            )
            torch.save(patches[:, 0], root / f"{i}" / f"patches_{frame_id}.pt")

        # Store the number of dets in patches that we know should match
        (root / f"{i}" / "annotated.txt").write_text(f"{num_det}")
