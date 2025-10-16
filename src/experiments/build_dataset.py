"""Extract ROI from detections

It will extract the ROI for each detections (every frame_gap=K frames). In addition, for each extracted ROI at frame t, NN tracking is used to compensate motion and extract ROIs at frames [t - M, t + M].

To allow random augmentations in the training phase, ROIs are extracting with twice the size.
"""

import dataclasses
import enum
import os
import pathlib
from typing import Collection, Dict, Sequence, Tuple

import dacite
import torch
import tqdm.auto as tqdm
import yaml

import byotrack

from ..data.video import VideoConfig
from ..detect import DetectionConfig
from ..pseudo_tracking import PseudoTrackerConfig


@dataclasses.dataclass
class ExperimentConfig:
    video: VideoConfig
    detection: DetectionConfig
    pseudo_tracker: PseudoTrackerConfig
    patch_size: int
    frame_gap: int
    frame_interval: int


def extract_at_frame_id(
    frame_id: int,
    frames: torch.Tensor,
    positions: torch.Tensor,
    patch_size: int,
) -> torch.Tensor:
    """Extract patches from the given frames at the given positions.

    It will extract patches of shape (N, 2M + 1, C, 2w + 1, 2w + 1) for each detection of frame_id.
    Where 2 M + 1 is the size of the frame interval considered and 2w + 1 the size of the patches.

    Args:
        frame_id (int): The current frame id in the video
        frames (torch.Tensor): Concatenated frames in the time interval around frame_id
            Shape: (2M + 1, H, W, C)
        positions (torch.Tensor): Estimated positions of the N detected particles on frame_id
            at each time.
            Shape: (N, 2M + 1, 2)
        patch_size (int): Half size of the patch (w)
    """
    # Allocate (N, 2M+1, C, 2w+1, 2w+1) patches
    patches = torch.zeros(
        (positions.shape[0], frames.shape[0], frames.shape[-1], 2 * patch_size + 1, 2 * patch_size + 1),
        dtype=torch.uint8,
    )

    frame_id -= len(frames) // 2

    for det_id in range(positions.shape[0]):
        patches_ = torch.zeros(
            (frames.shape[0], 2 * patch_size + 1, 2 * patch_size + 1, frames.shape[-1]), dtype=frames.dtype
        )
        for t in range(positions.shape[1]):
            point = positions[det_id, t].round() - patch_size
            if torch.isnan(point).any():  # Ignore NaN, let's keep 0 in the patch
                continue

            i = int(point[0])
            j = int(point[1])

            # Complexe slice:
            # if i < 0 then we will only copy starting from -i in the patch
            # if i + patch_size > H then we do not copy till the end
            i_patch_slice = slice(max(0, -i), frames.shape[1] - i)
            j_patch_slice = slice(max(0, -j), frames.shape[2] - j)

            if patches_[t, i_patch_slice, j_patch_slice, 0].shape != (2 * patch_size + 1, 2 * patch_size + 1):
                tqdm.tqdm.write(
                    f"Partial patch at frame {frame_id + t} for detection {det_id} ({positions[t][det_id].numpy()})"
                )

            patches_[t, i_patch_slice, j_patch_slice] = frames[
                t, max(0, i) : i + 2 * patch_size + 1, max(0, j) : j + 2 * patch_size + 1
            ]

        # Save as uint8 (much lighter)
        patches_ *= 255
        patches_.round_()
        patches[det_id] = patches_.to(torch.uint8).permute(0, 3, 1, 2)

    return patches


def print_stats(detections_sequence: Sequence[byotrack.Detections], tracks: Collection[byotrack.Track]):
    n_dets = sum(len(detections) for detections in detections_sequence)
    mean_dets = n_dets / len(detections_sequence)
    n_tracks = len(tracks)
    n_tracked_dets = sum(len(track) for track in tracks)
    mean_tracks = n_tracked_dets / len(tracks)
    fraction = n_tracked_dets / n_dets

    print(f"Number of detections: {n_dets} (~ {mean_dets:.1f}x{len(detections_sequence)})")
    print(f"Pseudo tracker convers {fraction * 100:.1f}% detections: {n_tracked_dets} (~ {mean_tracks:.1f}x{n_tracks})")


def build_detection_mapping(tracks: Collection[byotrack.Track]) -> Dict[Tuple[int, int], int]:
    mapping = {}

    for track in tracks:
        for frame_offset, detection_id in enumerate(track.detection_ids):
            if detection_id == -1:
                continue
            mapping[(detection_id, track.start + frame_offset)] = track.identifier

    return mapping


def main(name: str, cfg_data: dict) -> None:
    print("Running:", name)
    print(yaml.dump(cfg_data))
    cfg = dacite.from_dict(ExperimentConfig, cfg_data, dacite.Config(cast=[pathlib.Path, tuple, enum.Enum]))

    # Open the video
    video = cfg.video.open()

    # Build detector and run detections
    detector = cfg.detection.build()
    detections_sequence = detector.run(video)

    # Run a our pseudo tracking algorithm
    pseudo_tracks = cfg.pseudo_tracker.run(video, detections_sequence)
    track_from_id = {track.identifier: track for track in pseudo_tracks}
    print_stats(detections_sequence, pseudo_tracks)

    # Build mapping from det_id, t to track_id
    det_to_track = build_detection_mapping(pseudo_tracks)

    root = pathlib.Path(os.environ.get("EXPYRUN_CWD", ".")) / "data" / name
    root.mkdir(parents=True, exist_ok=False)

    # Let's extract at each frame every frame_gap on an interval of 2 frame_interval + 1
    for frame_id in tqdm.trange(cfg.frame_gap, len(video) - cfg.frame_interval, cfg.frame_gap):
        frames = torch.zeros(2 * cfg.frame_interval + 1, *video[0].shape)
        for i in range(-cfg.frame_interval, cfg.frame_interval + 1):
            frames[i + cfg.frame_interval] = torch.from_numpy(video[frame_id + i])

        detections = detections_sequence[frame_id]

        positions = torch.full(
            (detections.length, 2 * cfg.frame_interval + 1, detections.position.shape[-1]), torch.nan
        )
        # Set position in the middle
        positions[:, cfg.frame_interval] = detections.position

        # Find positions for the given detections on previous/future frames
        # (only available if a detection has been pseudo tracked)
        for det_id in range(detections.length):
            if (det_id, frame_id) not in det_to_track:  # Untracked
                continue

            track = track_from_id[det_to_track[(det_id, frame_id)]]
            for i in range(-cfg.frame_interval, cfg.frame_interval + 1):
                positions[det_id, i + cfg.frame_interval] = track[frame_id + i]

        patches = extract_at_frame_id(frame_id, frames, positions, cfg.patch_size)

        # Save patches, as well as positions and frames for debug purposes
        torch.save(patches, root / f"patches_{frame_id}.pt")
        torch.save(positions, root / f"positions_{frame_id}.pt")
        torch.save(torch.round(frames * 255).to(torch.uint8), root / f"frames_{frame_id}.pt")
