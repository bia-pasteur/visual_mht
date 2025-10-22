"""From our previous project that supported channels"""

import dataclasses
from typing import Sequence, Tuple

import cv2
from kornia import augmentation, filters
import numpy as np
import torch.nn
from torchvision.transforms import v2  # type: ignore


class BlurShotNoise(torch.nn.Module):
    """Apply a Gaussian blur and Poisson Shot Noise

    Attributes:
        sigma (Tuple[float, float]): Min and max values for the Gaussian blur sigma.
            It will be uniformly sampled in this interval.
        max_psnr (List[float]): The PSNR of the Poisson Shot Noise for the brightest pixels
            (It can be estimated as E(I) / STD(I) on a bright spot).

    """

    def __init__(self, sigma: Tuple[float, float], max_psnr: Sequence[float], prob: float) -> None:
        super().__init__()
        kernel_size = 2 * int(3 * max(sigma)) + 1

        self.blur = augmentation.RandomGaussianBlur(kernel_size, sigma, p=prob, separable=False)
        self.max_psnr = torch.tensor(max_psnr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Blur then apply a new Poisson Shot Noise

        Args:
            x (torch.Tensor): 2d input with channels
                Shape: (C, H, W) or (B, C, H, W)
        """
        n_channel = x.shape[-3]
        assert n_channel == len(self.max_psnr)

        blurred: torch.Tensor = self.blur(x)
        maximum = blurred.max(dim=-1).values.max(dim=-1).values[..., None, None]  # Max over images
        maximum[maximum == 0] = 1

        selected = self.blur._params["batch_prob"] == 1.0  # pylint: disable=protected-access

        ratio = self.max_psnr[:, None, None].to(maximum.device) ** 2 / maximum

        blurred[selected] = (
            torch.distributions.Poisson(ratio[selected] * blurred[selected]).sample((1,))[0] / ratio[selected]
        )

        # Renormalize and send to the right dtype
        if torch.is_floating_point(x):
            blurred = blurred.clip(0, 1).to(x.dtype)
        else:
            blurred = blurred.clip(0, 255).to(x.dtype)

        if x.ndim == 3:
            return blurred[0]  # Remove the additional batch dimension added by blur

        return blurred


class MotionBlur(torch.nn.Module):
    """Let's do it manually as kornia do not support batch for MotionBlur

    (Kornia reuse the same motion amplitude across the batch...)
    """

    def __init__(self, max_displacement: int, prob: float):
        super().__init__()
        self.max_displacement = max_displacement
        self.prob = prob

    def generate_kernels(self, n: int, device: torch.device) -> torch.Tensor:
        kernel_size = self.max_displacement
        offset = kernel_size // 2  # Middle position

        # Sample a random angle and length
        angles = np.random.rand(n) * 360
        half_length = np.random.rand(n) * offset
        points = np.stack((np.cos(angles) * half_length, np.sin(angles) * half_length), axis=-1)

        kernels = np.zeros((n, kernel_size, kernel_size), dtype=np.uint8)

        # Draw lines on kernels
        source = (offset - points).round().astype(np.int32)
        target = (offset + points).round().astype(np.int32)
        for i in range(n):
            cv2.line(kernels[i], source[i], target[i], 1)

        kernels_pt = torch.tensor(kernels, device=device)
        kernels_pt = kernels_pt / kernels_pt.sum(dim=(1, 2), keepdims=True)

        return kernels_pt

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the motion blur to x.

        It samples the amplitude of motion uniformly
        """
        if self.prob == 0.0:
            return x

        has_batch = True
        if x.ndim == 3:
            has_batch = False
            x = x[None]

        apply = torch.rand(x.shape[0]) < self.prob

        y = x.clone()
        n = int(apply.sum().item())

        if n != 0:
            kernels = self.generate_kernels(n, device=x.device)
            y[apply] = filters.filter2d(x[apply], kernels)

        if not has_batch:
            y = y[0]

        return y


@dataclasses.dataclass
class ElasticConfig:
    prob: float = 0.0
    alpha: float = 20.0
    sigma: float = 5.0

    def build(
        self,
        patch_size: int,
    ) -> augmentation.RandomElasticTransform:
        patch_size = 2 * patch_size + 1  # Extracted patches are twice bigger than the patch_size
        kernel_size = 2 * int(4 * self.sigma) + 1
        return augmentation.RandomElasticTransform(
            (kernel_size, kernel_size),
            (self.sigma, self.sigma),
            (self.alpha / patch_size, self.alpha / patch_size),
            p=self.prob,
            keepdim=True,
        )


@dataclasses.dataclass
class AffineConfig:
    prob: float = 0.0
    max_rotate: float = 5.0  # 5°
    max_translate: float = 2.0  # 2 pixels

    def build(self, patch_size: int) -> augmentation.RandomAffine:
        patch_size = 2 * patch_size + 1  # Extracted patches are twice bigger than the patch_size
        return augmentation.RandomAffine(
            self.max_rotate,
            (self.max_translate / patch_size, self.max_translate / patch_size),
            p=self.prob,
            keepdim=True,
        )


@dataclasses.dataclass
class MotionBlurConfig:
    prob: float = 0.0
    max_displacement: int = 9  # Max displacement

    def build(self) -> MotionBlur:
        return MotionBlur(self.max_displacement, self.prob)


@dataclasses.dataclass
class EraseConfig:
    prob: float = 0.0
    scale: Tuple[float, float] = (0.005, 0.05)

    def build(self) -> augmentation.RandomErasing:
        return augmentation.RandomErasing(self.scale, p=self.prob, keepdim=True)


@dataclasses.dataclass
class BlurShotNoiseConfig:
    prob: float = 0.0
    sigma: Tuple[float, float] = (0.5, 1.0)
    psnr: Tuple[float] = (20.0,)

    def build(self) -> BlurShotNoise:
        return BlurShotNoise(self.sigma, self.psnr, self.prob)


@dataclasses.dataclass
class ColorJitterConfig:
    prob: float = 0.0
    brightness: float = 0.05  # Seems to give similar augmentation that the previous torchvision [0.85, 1.20]
    contrast: float = 0.05

    def build(self) -> augmentation.ColorJiggle:
        return augmentation.ColorJiggle(self.brightness, self.contrast, p=self.prob, keepdim=True)


class ConvertFloat32(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_floating_point(x):
            return x.to(torch.float32)

        return x / 255


@dataclasses.dataclass
class AugmentationConfig:
    patch_size: int = 32
    elastic: ElasticConfig = dataclasses.field(default_factory=ElasticConfig)
    affine: AffineConfig = dataclasses.field(default_factory=AffineConfig)
    erase: EraseConfig = dataclasses.field(default_factory=EraseConfig)
    motion_blur: MotionBlurConfig = dataclasses.field(default_factory=MotionBlurConfig)
    blur_shot_noise: BlurShotNoiseConfig = dataclasses.field(default_factory=BlurShotNoiseConfig)
    jitter: ColorJitterConfig = dataclasses.field(default_factory=ColorJitterConfig)

    def train_common(self) -> augmentation.ImageSequential:
        """Build the common augmentations

        It will allow any axial/central symmetry
        """
        return v2.Compose(  # torchvision is quite faster if not batch processed
            [
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(180),
            ]
        )
        # return augmentation.ImageSequential(
        #     ConvertFloat32(),
        #     augmentation.RandomHorizontalFlip(same_on_batch=True, keepdim=True),
        #     augmentation.RandomRotation(180.0, same_on_batch=True, keepdim=True),
        # )

    def train_specific(self) -> augmentation.ImageSequential:
        """Build the specific annotations

        It will apply in order any of the following optionnal augmentations:

        - elastic: Spatial elastic random deformation of the image
        - affine: Affine random deformation of the image (rotation and translation only)
        - motion_blur: Random motion blurring
        - blur_shot_noise: Random blurring followed by Poisson Shot Noise
        - jitter: Random color jittering
        - erase: Random erasing of a rectangle area. It is applied after all the deformations
            so that it does not give any informations on them. Also it is applied before cropping
            to increase the proportion of patch at the border of the the selected patch. Note that
            the center cropping may not select the erased area.
        - croping: Center croping to remove the augmentations artefacts on the border

        """
        return augmentation.ImageSequential(
            ConvertFloat32(),
            self.elastic.build(self.patch_size),
            self.affine.build(self.patch_size),
            self.motion_blur.build(),
            self.blur_shot_noise.build(),
            self.jitter.build(),
            self.erase.build(),
            augmentation.CenterCrop(self.patch_size, keepdim=True),
        )

    def test(self) -> augmentation.ImageSequential:
        return augmentation.ImageSequential(
            ConvertFloat32(),
            augmentation.CenterCrop(self.patch_size, keepdim=True),
        )
