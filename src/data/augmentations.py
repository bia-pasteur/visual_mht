"""From our previous project that supported channels"""

import dataclasses
from typing import Sequence, Tuple

import torch.nn
from torchvision.transforms import v2  # type: ignore


class ChannelSelector(torch.nn.Module):
    """Select given channels"""

    def __init__(self, channels: Sequence[int]) -> None:
        super().__init__()
        self.channels = list(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Select the channels

        Args:
            x (torch.Tensor): 2d input with channels
                Shape: (..., C, H, W)
        """
        return x[..., self.channels, :, :]


class ColorJitter(torch.nn.Module):
    """Wraps color jitter from torchvision to make it work with 2 channels"""

    def __init__(self, color_jitter: v2.ColorJitter) -> None:
        super().__init__()
        self.color_jitter = color_jitter

    def forward(self, x: torch.Tensor):
        if x.shape[-3] == 2:  # color jitter already supports 1 or 3 channels
            x = torch.cat((x, torch.zeros((*x.shape[:-3], 1, *x.shape[-2:]), dtype=x.dtype)), dim=-3)
            x = self.color_jitter(x)
            return x[..., :2, :, :]

        return self.color_jitter(x)


class BlurShotNoise(torch.nn.Module):
    """Apply a Gaussian blur and Poisson Shot Noise

    Attributes:
        sigma (Tuple[float, float]): Min and max values for the Gaussian blur sigma.
            It will be uniformly sampled in this interval.
        max_psnr (List[float]): The PSNR of the Poisson Shot Noise for the brightest pixels
            (It can be estimated as E(I) / STD(I) on a bright spot).

    """

    def __init__(self, sigma: Tuple[float, float], max_psnr: Sequence[float]) -> None:
        super().__init__()
        kernel_size = int(4 * max(sigma))
        if kernel_size % 2 == 0:
            kernel_size += 1

        self.blur = v2.GaussianBlur(kernel_size, sigma)
        self.max_psnr = torch.tensor(max_psnr)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Blur then apply a new Poisson Shot Noise

        Args:
            x (torch.Tensor): 2d input with channels
                Shape: (..., C, H, W)
        """
        n_channel = x.shape[-3]
        assert n_channel == len(self.max_psnr)

        blurred: torch.Tensor = self.blur(x)
        maximum = blurred.transpose(-1, -3).reshape(-1, n_channel).max(dim=0).values
        maximum[maximum == 0] = 1

        ratio = self.max_psnr**2 / maximum
        psn = torch.distributions.Poisson(ratio[:, None, None] * blurred).sample((1,))[0] / ratio[:, None, None]

        # Renormalize and send to the right dtype
        if torch.is_floating_point(x):
            psn = psn.clip(0, 1).to(x.dtype)
        else:
            psn = psn.clip(0, 255).to(x.dtype)

        return psn


@dataclasses.dataclass
class ElasticConfig:
    prob: float = 0.0
    alpha: float = 20.0
    sigma: float = 5.0

    def build(self, interpolation: v2.InterpolationMode = v2.InterpolationMode.BILINEAR) -> v2.RandomApply:
        return v2.RandomApply([v2.ElasticTransform(self.alpha, self.sigma, interpolation=interpolation)], p=self.prob)


@dataclasses.dataclass
class AffineConfig:
    prob: float = 0.0
    max_rotate: float = 5.0  # 5°
    max_translate: float = 2.0  # 2 pixels

    def build(
        self, patch_size: int, interpolation: v2.InterpolationMode = v2.InterpolationMode.BILINEAR
    ) -> v2.RandomApply:
        patch_size = 2 * patch_size + 1  # Extracted patches are twice bigger than the patch_size
        return v2.RandomApply(
            [
                v2.RandomAffine(
                    self.max_rotate,
                    (self.max_translate / patch_size, self.max_translate / patch_size),
                    interpolation=interpolation,
                )
            ],
            p=self.prob,
        )


@dataclasses.dataclass
class EraseConfig:
    prob: float = 0.0
    scale: Tuple[float, float] = (0.005, 0.05)

    def build(self) -> v2.RandomErasing:
        return v2.RandomErasing(self.prob, self.scale, inplace=True)


@dataclasses.dataclass
class BlurShotNoiseConfig:
    prob: float = 0.0
    sigma: Tuple[float, float] = (0.5, 1.0)
    psnr: Tuple[float, float, float] = (1.0, 5.0, 20.0)

    def build(self, channels: Tuple[bool, bool, bool]) -> v2.RandomApply:
        return v2.RandomApply(
            [BlurShotNoise(self.sigma, [psnr for keep, psnr in zip(channels, self.psnr) if keep])], self.prob
        )


@dataclasses.dataclass
class ColorJitterConfig:
    prob: float = 0.0
    brightness: Tuple[float, float] = (0.8, 1.25)
    contrast: Tuple[float, float] = (0.8, 1.25)
    saturation: Tuple[float, float] = (0.8, 1.25)

    def build(self) -> v2.RandomApply:
        return v2.RandomApply([ColorJitter(v2.ColorJitter(self.brightness, self.contrast, self.saturation))], self.prob)


@dataclasses.dataclass
class AugmentationConfig:
    patch_size: int = 32
    # channels: Tuple[bool, bool, bool] = (False, True, True)
    interpolation: v2.InterpolationMode = v2.InterpolationMode.BILINEAR
    elastic: ElasticConfig = ElasticConfig()
    affine: AffineConfig = AffineConfig()
    erase: EraseConfig = EraseConfig()
    blur_shot_noise: BlurShotNoiseConfig = BlurShotNoiseConfig()
    jitter: ColorJitterConfig = ColorJitterConfig()

    def train_common(self) -> v2.Compose:
        """Build the common augmentations

        It will allow any axial/central symmetry
        """
        return v2.Compose(
            [
                # ChannelSelector([i for i in range(len(self.channels)) if self.channels[i]]),
                v2.RandomHorizontalFlip(),
                v2.RandomRotation(180, interpolation=self.interpolation),
            ]
        )

    def train_specific(self) -> v2.Compose:
        """Build the specific annotations

        It will apply in order any of the following optionnal augmentations:

        - elastic: Spatial elastic random deformation of the image
        - affine: Affine random deformation of the image (rotation and translation only)
        - blur_shot_noise: Random blurring followed by Poisson Shot Noise
        - jitter: Random color jittering
        - erase: Random erasing of a rectangle area. It is applied after all the deformations
            so that it does not give any informations on them. Also it is applied before cropping
            to increase the proportion of patch at the border of the the selected patch. Note that
            the center cropping may not select the erased area.
        - croping: Center croping to remove the augmentations artefacts on the border

        """
        return v2.Compose(
            [
                self.elastic.build(self.interpolation),
                self.affine.build(self.patch_size, self.interpolation),
                self.blur_shot_noise.build(self.channels),
                self.jitter.build(),
                self.erase.build(),
                v2.CenterCrop(self.patch_size),
                v2.ToDtype(torch.float32),
            ]
        )

    def test(self) -> v2.Compose:
        return v2.Compose(
            [
                # ChannelSelector([i for i in range(len(self.channels)) if self.channels[i]]),
                # v2.CenterCrop(self.patch_size),  # In the validation set, the patch are already at the right size
                v2.ToDtype(torch.float32),
            ]
        )
