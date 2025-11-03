import os
import pathlib
from typing import Sequence, Union

import numpy as np

import byotrack
from byotrack.implementation.detector import stardist


# TODO: In the paper, we used another stardist model (also trained for trase-in, just not the one published...)
#       With only 100 validated tracks, KOFT # errors can change from 1 to 7 and visualMHT from 0 to 3
#       We need to extend the validation to more tracks and may be more videos to increase perf confidence.
#       With this, you will not perfectly reproduce the paper, though the general idea should be here (1 VMHT vs 4 KOFT)


def run_trasein_stardist(video: Union[np.ndarray, Sequence[np.ndarray]]) -> Sequence[byotrack.Detections]:
    detector = stardist.StarDistDetector.from_trained(pathlib.Path(os.environ.get("EXPYRUN_CWD", ".")) / "stardist")
    detector.prob_threshold = 0.05
    return detector.run(video)
