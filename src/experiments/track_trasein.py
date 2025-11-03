import dataclasses
import enum
import pathlib
import random
from typing import List

import dacite
from PIL import Image
import scipy.ndimage  # type: ignore
import torch
import tqdm.auto as tqdm
import yaml  # type: ignore

import byotrack
import byotrack.visualize

from .. import detect, stardist, manual_annotator
from ..data.video import VideoConfig
from ..linkers import features
from ..methods.koft import KOFTConfig
from ..methods.bskt import BSKTConfig
from ..methods.emht import EMHTConfig, icy_emht
from ..methods.trackmate import TrackMateConfig
from ..methods.visual_mht import VisualMHTConfig
from ..methods.visual_nn import VisualNNConfig
from ..optical_flow import OpticalFlowConfig
from ..utils import enforce_all_seeds


@dataclasses.dataclass
class ExperimentConfig:
    tracking_methods: List[str]
    video: VideoConfig
    detection: detect.DetectionConfig
    model: features.FeaturesConfig
    optflow: OpticalFlowConfig
    trackmate: TrackMateConfig
    emht: EMHTConfig
    koft: KOFTConfig
    bskt: BSKTConfig
    visual_mht: VisualMHTConfig
    visual_nn: VisualNNConfig


STARDIST = True


def keep_long(tracks, min_length):
    return [track for track in tracks if len(track) > min_length]


def project(video, tracks, valid):
    scale = 1
    colors = [(200, 0, 0) if valid[i] else (0, 200, 0) for i in range(len(tracks))]

    visu = byotrack.visualize.temporal_projection(
        byotrack.Track.tensorize(tracks) * scale,
        colors=colors,
        background=scipy.ndimage.zoom(video[0], (scale, scale, 1), order=1),
    )

    return visu


def main(name: str, cfg_data: dict) -> None:
    print("Running:", name)
    print(yaml.dump(cfg_data))
    cfg = dacite.from_dict(
        ExperimentConfig,
        cfg_data,
        dacite.Config(
            cast=[pathlib.Path, tuple, enum.Enum], type_hooks={icy_emht.Motion: lambda s: icy_emht.Motion[s.upper()]}
        ),
    )

    # Seed
    enforce_all_seeds(cfg.video.seed)  # Nothing is random except the sampling, but anyway let's enforce also here

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU is", torch.cuda.get_device_name(device))
    else:
        device = torch.device("cpu")
        print("Only CPU available")

    # Load data
    video = cfg.video.open()[130:230]  # Let's use 100 frames of the 1st contraction

    # Detections
    # Let's not use the cfg.detection and fall back on stardist for trase-in
    if STARDIST:
        detections_sequence = stardist.run_trasein_stardist(video)
    else:
        detector = cfg.detection.build()
        detections_sequence = detector.run(video)

    # Prepare features and optical flow
    optflow = cfg.optflow.build()
    extractor = cfg.model.build_extractor(device)

    linker: byotrack.Linker
    for method in tqdm.tqdm(cfg.tracking_methods):
        linker = getattr(cfg, method).build(optflow, extractor)
        linker.device = device  # Run on device if possible

        tracks = linker.run(video, detections_sequence)
        long_tracks = keep_long(tracks, 0.9 * len(video))
        print(f"{method} produced {len(tracks)} tracks with {len(long_tracks)} long trajectories (> 90% tracked)")

        enforce_all_seeds(cfg.video.seed)

        annotator = manual_annotator.InteractiveTrackValidator(video, random.sample(long_tracks, 100))
        annotator.run()

        print(f"Found {(~annotator.is_valid).sum()} invalid tracks")
        Image.fromarray(project(video, annotator.tracks, annotator.is_valid)).save("method.png")
