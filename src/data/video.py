from __future__ import annotations

import dataclasses
import pathlib

import byotrack


@dataclasses.dataclass
class VideoConfig:
    """For loading any video"""

    path: pathlib.Path
    config: byotrack.VideoTransformConfig
    video_id: str = ""
    scenario: str = ""
    seed: int = 0

    def open(self) -> byotrack.Video:
        video = byotrack.Video(self.path)
        video.set_transform(self.config)
        return video
