import dataclasses
import os
import pathlib
from typing import Dict, List, Tuple, Sequence
from xml.etree import ElementTree as ET

import skimage
import numpy as np
import torch
import tqdm

import byotrack


@dataclasses.dataclass
class PairConfig:
    """Config for annotated pairs"""

    frames: List[int]
    roi_loc: Tuple[int, int]
    annotator: str = "Unknown"

    def trasein_annotation_file(self, prefix: str) -> pathlib.Path:
        folder = pathlib.Path(os.environ.get("EXPYRUN_CWD", ".")) / "data" / "trasein_annotations"
        # XXX: Could handle triplet instead of pairs in the future?
        return folder / f"tdt_{prefix}_{self.frames[0]}_{self.frames[1]}_{self.annotator.lower()}.xml"


# From the annotated dataset
def parse_annotations(file) -> Dict[int, Dict[int, List[Tuple[int, int]]]]:
    """Parse an annotation file following CVAT format.

    It expect "neuron" objects (that are tracks containing a polygon annotation)

    This function outputs the polygons for each annotated frame & neuron.

    Returns:
        Dict[int, Dict[int, List[Tuple[int, int]]]]: A dict containing for each frame ids
            the neuron ids and their polygons.

    """
    # create element tree object
    tree = ET.parse(file)

    # get root element
    root = tree.getroot()

    neurons: Dict[int, Dict[int, List[Tuple[int, int]]]] = {}

    for track in root.findall("./track"):  # Why the "./track", couldn't we use "track" ?
        label = track.attrib["label"]
        if label != "neuron":
            if label != "roi":
                tqdm.tqdm.write(f"Unrecognized label: {label}")
            continue

        for poly in track:
            if poly.tag != "polygon":
                raise ValueError(f"We only support polygon annotations. Found: {poly.tag}")

            # We parse every poly in the track and select only those that are both the keyframe
            # and inside. (This will work both case that seems to appear in the file of Anja)
            if poly.attrib["outside"] != "0" or poly.attrib["keyframe"] != "1":
                continue

            frame = int(poly.attrib["frame"])
            if frame not in neurons:
                neurons[frame] = {}

            points = [
                (
                    round(float(coord.split(",")[1])),
                    round(float(coord.split(",")[0])),
                )
                for coord in poly.attrib["points"].split(";")
            ]

            neuron_id = int(track.attrib["id"])
            assert neuron_id > 0, "The id should be strictly positive"
            assert neuron_id not in neurons[frame], "Id is not unique in the frame"

            neurons[frame][neuron_id] = points

    return neurons


# From the annotated dataset
def convert_to_segmentation(
    frame_shape: Tuple[int, int],
    neurons: Dict[int, List[Tuple[int, int]]],
) -> np.ndarray:
    """Convert all the polygons of the frames into a segmentation mask

    Pixels inside a neuron polygon are set to its neuron_id (> 0)
    """
    # Convert neurons to seg
    segmentation = np.full((frame_shape), 0, dtype=np.int32)

    for neuron_id, poly_ in neurons.items():
        poly = np.array(poly_)
        rows, cols = skimage.draw.polygon(poly[:, 0], poly[:, 1], segmentation.shape)
        former_points = segmentation[rows, cols]
        if not (np.logical_or(former_points == 0, former_points == -1)).all():
            overriden_labels = np.unique(former_points)
            overriden_labels = overriden_labels[overriden_labels != 0]
            overriden_labels = overriden_labels[overriden_labels != -1]
            overriden_prop = (former_points[..., None] == overriden_labels).mean(axis=0)
            if overriden_prop.sum() > 0.3:
                overriden_str = " and ".join(
                    f"{100 * prop:.1f}% is {label_id}" for prop, label_id in zip(overriden_prop, overriden_labels)
                )
                tqdm.tqdm.write(f"Overriding some neurons with {neuron_id}: {overriden_str}")
                # start = poly.min(axis=0)
                # stop = poly.max(axis=0)
                # tqdm.tqdm.write(
                #     str(segmentation[start[0] : stop[0] + 1, start[1] : stop[1] + 1])
                # )
        segmentation[rows, cols] = neuron_id

    return segmentation


def load_trasein_annotations(video_id: str, pair: PairConfig, shape: Tuple[int, int]) -> Sequence[byotrack.Detections]:
    all_neurons = parse_annotations(pair.trasein_annotation_file(video_id))

    segs = [convert_to_segmentation(shape, all_neurons[frame_id]) for frame_id in pair.frames]

    dets = [byotrack.Detections({"segmentation": torch.from_numpy(seg)}) for seg in segs]

    return dets
