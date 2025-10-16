from __future__ import annotations

import dataclasses
import pathlib

import byotrack


@dataclasses.dataclass
class VideoConfig:
    """For loading any video"""

    path: pathlib.Path
    config: byotrack.VideoTransformConfig

    def open(self) -> byotrack.Video:
        video = byotrack.Video(self.path)
        video.set_transform(self.config)
        return video
