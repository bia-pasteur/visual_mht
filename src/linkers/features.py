from __future__ import annotations

import dataclasses
import pathlib
from typing import Optional

import numba
import numpy as np
import scipy.ndimage  # type: ignore
import torch

import byotrack

from .. import encoder


@dataclasses.dataclass
class FeaturesConfig(encoder.ModelConfig):
    """Configuration for features extraction with our self-supervised models"""

    patch_size: int = 32  # Could use increased or reduced patch size at test time.
    weights: Optional[pathlib.Path] = None
    extractor: str = "deep"

    def build_extractor(self, device: torch.device) -> Optional[byotrack.FeaturesExtractor]:
        if self.extractor.lower() == "none":
            return None

        if self.extractor.lower() == "identity":
            return IdentityFeatures(self.patch_size)

        if self.extractor.lower() != "deep":
            raise ValueError(f"Unknown extractor: {self.extractor}.")

        model = self.build()
        if self.weights is None:
            raise ValueError("Cannot build DeepFeatures without model weights")

        if not self.weights.exists():
            raise ValueError(f"Weights ({self.weights}) do not exist. You have to train a model before tracking.")

        # Load weights
        model.load_state_dict(torch.load(self.weights, map_location="cpu", weights_only=True)["model"])

        return DeepFeatures(model, self.patch_size, device)


class IdentityFeatures(byotrack.FeaturesExtractor):
    def __init__(self, patch_size: int) -> None:
        self.patch_size = patch_size

    def __call__(self, frame: np.ndarray, detections: byotrack.Detections) -> torch.Tensor:
        patches = torch.zeros((len(detections), self.patch_size, self.patch_size, frame.shape[-1]), dtype=torch.float32)

        for k, point in enumerate(detections.position):
            point = (point - self.patch_size / 2).round()
            i = int(point[0])
            j = int(point[1])

            # Complexe slice:
            # if i < 0 then we will only copy starting from -i in the patch
            # if i + patch_size > H then we do not copy till the end
            i_patch_slice = slice(max(0, -i), frame.shape[0] - i)
            j_patch_slice = slice(max(0, -j), frame.shape[1] - j)

            # if patches[t, i_patch_slice, j_patch_slice, 0].shape != (2 * patch_size + 1, 2 * patch_size + 1):
            #     tqdm.tqdm.write(
            #         f"Partial patch at frame {frame_id + t} for detection {det_id} ({positions[t][det_id].numpy()})"
            #     )

            patches[k, i_patch_slice, j_patch_slice] = torch.tensor(
                frame[max(0, i) : i + self.patch_size, max(0, j) : j + self.patch_size]
            ).to(torch.float32)

        return patches.reshape(len(patches), -1)


class IdentityFeatures3D(byotrack.FeaturesExtractor):
    def __init__(self, patch_size: int, downscale=1) -> None:
        self.patch_size = patch_size
        self.downscale = downscale

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Downscale a frame before running the optical flow algorithm

        It blurs then downscales the frame. It also converts to float32.

        Args:
            frame (np.ndarray): The frame to pre-preprocess
                Shape: ([D, ]H, W, C), dtype: float

        Returns:
            np.ndarray: The blurred and downscaled frame
                Shape: ([D', ]H', W', C), dtype: float32

        """
        downscale = np.ones(frame.ndim, np.float32)
        downscale[:-1] = self.downscale  # Do not downscale channels

        blur = np.zeros(frame.ndim, np.float32)  # Do not blur along channel dimension
        blur[:-1] = 2 * downscale[:-1] / 6

        # Blur
        frame = scipy.ndimage.gaussian_filter(frame.astype(np.float32), blur, mode="reflect")

        # Downscale (Linear interpolation)
        return scipy.ndimage.zoom(frame, 1 / downscale, order=1, mode="mirror", grid_mode=True)

    def __call__(self, frame: np.ndarray, detections: byotrack.Detections) -> torch.Tensor:
        frame = self.preprocess(frame)

        patches = torch.zeros(
            (len(detections), self.patch_size, self.patch_size, self.patch_size, frame.shape[-1]), dtype=torch.float32
        )

        for k, point in enumerate(detections.position / self.downscale):
            point = (point - self.patch_size / 2).round()
            z = int(point[0])
            i = int(point[1])
            j = int(point[2])

            # Complexe slice:
            # if i < 0 then we will only copy starting from -i in the patch
            # if i + patch_size > H then we do not copy till the end
            z_patch_slice = slice(max(0, -z), frame.shape[0] - z)
            i_patch_slice = slice(max(0, -i), frame.shape[1] - i)
            j_patch_slice = slice(max(0, -j), frame.shape[2] - j)

            # if patches[t, i_patch_slice, j_patch_slice, 0].shape != (2 * patch_size + 1, 2 * patch_size + 1):
            #     tqdm.tqdm.write(
            #         f"Partial patch at frame {frame_id + t} for detection {det_id} ({positions[t][det_id].numpy()})"
            #     )

            patches[k, z_patch_slice, i_patch_slice, j_patch_slice] = torch.tensor(
                frame[max(0, z) : z + self.patch_size, max(0, i) : i + self.patch_size, max(0, j) : j + self.patch_size]
            ).to(torch.float32)

        return patches.reshape(len(patches), -1)


class DeepFeatures(byotrack.FeaturesExtractor):
    def __init__(self, model: encoder.Encoder, patch_size: int, device: torch.device) -> None:
        self.model = model.to(device)
        self.patch_size = patch_size
        self.device = device

    def __call__(self, frame: np.ndarray, detections: byotrack.Detections) -> torch.Tensor:
        patches = torch.zeros((len(detections), self.patch_size, self.patch_size, frame.shape[-1]), dtype=torch.float32)

        for k, point in enumerate(detections.position):
            point = (point - self.patch_size / 2).round()
            i = int(point[0])
            j = int(point[1])

            # Complexe slice:
            # if i < 0 then we will only copy starting from -i in the patch
            # if i + patch_size > H then we do not copy till the end
            i_patch_slice = slice(max(0, -i), frame.shape[0] - i)
            j_patch_slice = slice(max(0, -j), frame.shape[1] - j)

            # if patches[t, i_patch_slice, j_patch_slice, 0].shape != (2 * patch_size + 1, 2 * patch_size + 1):
            #     tqdm.tqdm.write(
            #         f"Partial patch at frame {frame_id + t} for detection {det_id} ({positions[t][det_id].numpy()})"
            #     )

            patches[k, i_patch_slice, j_patch_slice] = torch.tensor(
                frame[max(0, i) : i + self.patch_size, max(0, j) : j + self.patch_size]
            ).to(torch.float32)

        with torch.no_grad():
            return self.model(patches.permute(0, 3, 1, 2).to(self.device))  # .cpu()


class DeepFeatures3D(byotrack.FeaturesExtractor):
    def __init__(self, model: encoder.Encoder, patch_size: int, device: torch.device) -> None:
        self.model = model.to(device)
        self.patch_size = patch_size
        self.device = device

    def __call__(self, frame: np.ndarray, detections: byotrack.Detections) -> torch.Tensor:
        patches = torch.zeros(
            (len(detections) * 3, self.patch_size, self.patch_size, frame.shape[-1]), dtype=torch.float32
        )

        for k, point in enumerate(detections.position):
            point = (point - self.patch_size / 2).round()
            z = int(point[0])
            i = int(point[1])
            j = int(point[2])

            # Complexe slice:
            # if i < 0 then we will only copy starting from -i in the patch
            # if i + patch_size > H then we do not copy till the end
            z_patch_slice = slice(max(0, -z), frame.shape[0] - z)
            i_patch_slice = slice(max(0, -i), frame.shape[1] - i)
            j_patch_slice = slice(max(0, -j), frame.shape[2] - j)

            # if patches[t, i_patch_slice, j_patch_slice, 0].shape != (2 * patch_size + 1, 2 * patch_size + 1):
            #     tqdm.tqdm.write(
            #         f"Partial patch at frame {frame_id + t} for detection {det_id} ({positions[t][det_id].numpy()})"
            #     )

            patches[3 * k, i_patch_slice, j_patch_slice] = torch.tensor(
                frame[z, max(0, i) : i + self.patch_size, max(0, j) : j + self.patch_size]
            ).to(torch.float32)
            patches[3 * k + 1, z_patch_slice, j_patch_slice] = torch.tensor(
                frame[max(0, z) : z + self.patch_size, i, max(0, j) : j + self.patch_size]
            ).to(torch.float32)
            patches[3 * k + 2, z_patch_slice, i_patch_slice] = torch.tensor(
                frame[max(0, z) : z + self.patch_size, max(0, i) : i + self.patch_size, j]
            ).to(torch.float32)

        with torch.no_grad():
            return (
                self.model(patches.permute(0, 3, 1, 2).to(self.device)).reshape(
                    len(detections), 3 * self.model.out_planes
                )
                # .cpu()
            )


@numba.njit(cache=byotrack.NUMBA_CACHE)
def _fpt_moments(segmentation: np.ndarray, intensity: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Extract the cumulated intensity of each detection and their normalized second order moment

    Corresponds to m0 and m2 in Features Point Tracking.

    Args:
        segmentation (np.ndarray): Segmentation of targets
            Shape: ([D, ]H, W), dtype: int
        intensity (np.ndarray): Video frame
            Shape: ([D, ]H, W), dtype: float
        positions (np.ndarray): Precomputed position of each instance
            Shape: (n, dim), dtype: float

    Returns:
        np.ndarray: Intensity and second order moment for each target
            Shape: (n, 2), dtype: float

    """
    n = segmentation.max()

    moments = np.zeros((n, 2), dtype=intensity.dtype)

    for index in np.ndindex(*segmentation.shape):
        instance = segmentation[index] - 1
        if instance != -1:
            moments[instance, 0] += intensity[index]
            offset = np.array(index) - positions[instance]
            offset **= 2
            moments[instance, 1] = offset.sum() * intensity[index]

    moments[:, 1] /= moments[:, 0] + (moments[:, 0] == 0)
    return moments


class FPT(byotrack.FeaturesExtractor):
    """Feature Point Tracking features

    Extract the 0-order and 2-order intensity moments
    """

    def __call__(self, frame, detections):
        moments = _fpt_moments(detections.segmentation.numpy(), frame.sum(axis=-1), detections.position.numpy())

        return torch.tensor(moments, dtype=torch.float32)
