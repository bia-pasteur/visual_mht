import dataclasses
import enum

import byotrack
from byotrack.implementation.detector.wavelet import WaveletDetector


class DetectionMethod(enum.Enum):
    """Implemented detections method"""

    WAVELET = "wavelet"


@dataclasses.dataclass
class WaveletConfig:
    """Wavelet detection configuration"""

    k: float = 2.5
    scale: int = 1
    min_area: float = 7.5


@dataclasses.dataclass
class DetectionConfig:
    """Detection configuration"""

    detector: DetectionMethod
    wavelet: WaveletConfig

    def build(self) -> byotrack.Detector:
        if self.detector == DetectionMethod.WAVELET:
            return WaveletDetector(self.wavelet.scale, self.wavelet.k, min_area=self.wavelet.min_area, batch_size=5)

        raise ValueError("Unsupported Detector")
