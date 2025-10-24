import dataclasses
import enum
import pathlib
import time
from typing import List

import dacite
import torch
import tqdm  # type: ignore
import yaml  # type: ignore

import byotrack
from byotrack.implementation.refiner.smoother import RTSSmoother

from .. import detect
from ..data import sinetra_data
from ..data.video import VideoConfig
from ..linkers import features
from ..methods.koft import KOFTConfig
from ..methods.bskt import BSKTConfig
from ..methods.emht import EMHTConfig, icy_emht
from ..methods.trackmate import TrackMateConfig
from ..methods.visual_mht import VisualMHTConfig
from ..methods.visual_nn import VisualNNConfig
from ..metrics import detection as detection_metrics, tracking as tracking_metrics
from ..optical_flow import OpticalFlowConfig
from ..utils import enforce_all_seeds, kill_java_in_our_pgrp_pkill


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


def main_sinetra(name: str, cfg_data: dict) -> None:
    print("Running:", name)
    print(yaml.dump(cfg_data))
    cfg = dacite.from_dict(
        ExperimentConfig,
        cfg_data,
        dacite.Config(
            cast=[pathlib.Path, tuple, enum.Enum], type_hooks={icy_emht.Motion: lambda s: icy_emht.Motion[s.upper()]}
        ),
    )

    enforce_all_seeds(cfg.video.seed)

    # Load data
    video = cfg.video.open()

    # Load ground truth
    gt_tracks = sinetra_data.load_tracks(cfg.video.path.parent)
    ground_truth = byotrack.Track.tensorize(gt_tracks)

    # Detections
    detector = cfg.detection.build()
    detections_sequence = detector.run(video)

    # Prepare features and optical flow
    optflow = cfg.optflow.build()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    extractor = cfg.model.build_extractor(device)

    # Evaluate detections step performances
    tp = 0.0
    n_pred = 0.0
    n_true = 0.0
    for i, detections in enumerate(detections_sequence):
        det_metrics = detection_metrics.DetectionMetric(2.0).compute_at(detections, ground_truth[i])
        tp += det_metrics["tp"]
        n_pred += det_metrics["n_pred"]
        n_true += det_metrics["n_true"]

    print("=======Detection======")
    print("Recall", tp / n_true if n_true else 1.0)
    print("Precision", tp / n_pred if n_pred else 1.0)
    print("f1", 2 * tp / (n_true + n_pred) if n_pred + n_true else 1.0)

    linker: byotrack.Linker
    # Add the same smoothing to all competing methods (with same parameters as our linkers)
    refiner = RTSSmoother(detection_std=1.0, process_std=2.0, kalman_order=1, initial_std_factor=3.0)
    metrics = {}
    for method in tqdm.tqdm(cfg.tracking_methods):
        linker = getattr(cfg, method).build(optflow, extractor)
        linker.device = device  # Run on device if possible

        t = time.time()
        try:
            tracks = linker.run(video, detections_sequence)
            tracks = refiner.run(video, tracks)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            kill_java_in_our_pgrp_pkill()  # Kill Java just in case it survives (ugly, needs to be fixed in ByoTrack)
            tqdm.tqdm.write(str(exc))
            tracks = []  # Tracking failed (For instance: timeout in EMHT)

        t = time.time() - t

        tqdm.tqdm.write("\n")

        # FPS are not so much fair for external methods (eMHT, TrackMate).
        # In practice, we manually measure the time from when the tracking in Java truly starts
        # and up to when it truly ends, removing the python wraping time.
        tqdm.tqdm.write(f"Built {len(tracks)} tracks in {t} seconds ({len(video) / t} fps)")

        if len(tracks) == 0 or len(tracks) > len(gt_tracks) * 20:
            tqdm.tqdm.write(f"{method} failed (too few or too many tracks). Continuing...\n")
            continue

        hota = tracking_metrics.compute_tracking_metrics(tracks, gt_tracks)

        # Hota @ 2 (-8 => Thresholds is 2)
        metrics[method] = {key: value[-8].item() for key, value in hota.items()}
        metrics[method]["time"] = t
        byotrack.Track.save(tracks, f"{method}_tracks.pt")

        tqdm.tqdm.write(f"{method} => HOTA@2.0: {metrics[method]['HOTA']}\n")

    with open("metrics.yml", "w", encoding="utf-8") as file:
        file.write(yaml.dump(metrics))


# trasein is run and inspected manually
