from typing import Collection, List, Sequence, Union

import numpy as np
import torch

import byotrack
from byotrack.visualize import _convert_to_uint8
import cv2


class InteractiveTrackValidator:
    """Interactive visualization for annotating erronenous tracks.

    By default all the given tracks are supposed correct.
    Double click on a track to accept or reject it: Green tracks are accepted, red are rejected.

    Keys:
        * w/x: Move backward/forward in the video
        * t: Switch tracks display mode (Not displayed, Both, Valid, Invalid)

    Keys:
        * `space`: Pause/Unpause the video
        * w/x: Move backward/forward in the video (when paused)
        * t: Switch tracks display mode (Not displayed, Both, Valid, Invalid)
        * v: Switch on/off the display of the video

    Keys can be modified in the dict `keys` (PAUSE, RIGHT, LEFT, TRA, VID).


    Attributes:
        video (Sequence[np.ndarray]): Video to display tracks with
        valid_tracks (List[byotrack.Track]): Current list of valid tracks
        invalid_tracks (List[byotrack.Track]): Current list of invalid tracks

    """

    keys = {
        "QUIT": "q",
        "PAUSE": " ",
        "RIGHT": "x",
        "LEFT": "w",
        "TRA": "t",
        "VID": "v",
    }

    window_name = "Manual track labeling"
    scale = 1
    valid_color = (100, 255, 100)
    invalid_color = (255, 100, 100)

    def __init__(
        self,
        video: Union[Sequence[np.ndarray], np.ndarray],
        tracks: Collection[byotrack.Track],
    ) -> None:
        assert len(video) != 0 and len(tracks) != 0

        self.video = video
        self.tracks = list(tracks)

        self.is_valid = np.full(len(tracks), True)
        self._tracks_tensor = byotrack.Track.tensorize(self.tracks)

        self.frame_shape = video[0][..., 0].shape
        self.n_frames = max(len(video), len(self._tracks_tensor))

        self.scale = 1
        self.interpolation = cv2.INTER_NEAREST

        self._frame_id = 0
        self._video_frame = self.video[self._frame_id] if len(self.video) != 0 else np.zeros((*self.frame_shape, 0))
        self._display_video = 1
        self._display_tracks = 1
        self._running = False

    @property
    def valid_tracks(self) -> List[byotrack.Track]:
        return [track for i, track in enumerate(self.tracks) if self.is_valid[i]]

    @property
    def invalid_tracks(self) -> List[byotrack.Track]:
        return [track for i, track in enumerate(self.tracks) if not self.is_valid[i]]

    def run(self, frame_id=0, fps=20) -> None:
        """Run the visualization

        Args:
            frame_id (int): Starting frame_id
            fps (int): Frame rate
        """
        try:
            self._frame_id = frame_id
            self._run(fps)
        finally:
            cv2.destroyWindow(self.window_name)

    def _run(self, fps=20) -> None:  # pylint: disable=too-many-branches,too-many-statements
        self._video_frame = self.video[self._frame_id]
        while True:
            frame = np.zeros((*self.frame_shape[-2:], 3), dtype=np.uint8)

            if self._display_video:
                _frame = self._video_frame
                frame[:] = _convert_to_uint8(_frame)  # We only support grayscale (C=1) and RGB (C=3)

            if self.scale != 1:
                frame = cv2.resize(  # type: ignore
                    frame, None, fx=self.scale, fy=self.scale, interpolation=self.interpolation
                )

            frame = self.draw_tracks(frame)

            # Display the resulting frame
            cv2.imshow(self.window_name, np.flip(frame, 2))
            cv2.setMouseCallback(self.window_name, self._mouse_callback)
            title = f"Frame {self._frame_id} / {self.n_frames}"
            cv2.setWindowTitle(self.window_name, title)

            # Handle user actions
            key = cv2.waitKey(1000 // fps) & 0xFF
            if self.handle_actions(key):
                break

    def draw_tracks(self, frame: np.ndarray) -> np.ndarray:
        """Draw the tracks on the frame

        It will draw the tracks for `frame_id` and `stack_id`.

        Args:
            frame (np.ndarray): The 2D frame to draw on (will not be modified)
                Shape: (H, W), dtype: np.uint8

        Returns:
            np.ndarray: The resulting frame with the drawn tracks
        """
        frame = frame.copy()  # Do not draw inplace
        if self._display_tracks in (1, 2):
            for track in self.valid_tracks:
                point = track[self._frame_id] * self.scale
                if torch.isnan(point).any():
                    continue

                i, j = point.round().to(torch.int).tolist()

                cv2.circle(frame, (j, i), 5, self.valid_color)
                cv2.putText(
                    frame, str(track.identifier % 100), (j + 4, i - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.valid_color
                )

        if self._display_tracks in (1, 3):
            for track in self.invalid_tracks:
                point = track[self._frame_id] * self.scale
                if torch.isnan(point).any():
                    continue

                i, j = point.round().to(torch.int).tolist()

                cv2.circle(frame, (j, i), 5, self.invalid_color)
                cv2.putText(
                    frame,
                    str(track.identifier % 100),
                    (j + 4, i - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    self.invalid_color,
                )

        return frame

    def handle_actions(self, key: int) -> bool:  # pylint: disable=too-many-branches
        """Handle inputs from user

        Return True to quit

        Args:
            key (int): Key input from user

        Returns:
            bool: True to quit visualization
        """
        if key == ord(self.keys["QUIT"]):
            return True

        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            return True

        if key == ord(self.keys["PAUSE"]):
            self._running = not self._running

        old_frame_id = self._frame_id

        if not self._running and key == ord(self.keys["LEFT"]):  # Prev
            self._frame_id = (self._frame_id - 1) % self.n_frames

        if not self._running and key == ord(self.keys["RIGHT"]):  # Next
            self._frame_id = (self._frame_id + 1) % self.n_frames

        if self._running:  # Stop running if we reach the last frame
            self._frame_id += 1
            if self._frame_id >= self.n_frames:
                self._running = False
                self._frame_id = self.n_frames - 1

        if self._frame_id != old_frame_id and len(self.video) != 0:
            self._video_frame = self.video[self._frame_id]  # Read video only once when we change change frame_id

        if key == ord(self.keys["TRA"]):
            self._display_tracks = (self._display_tracks + 1) % 4

        if key == ord(self.keys["VID"]):
            self._display_video = 1 - self._display_video

        return False

    def _mouse_callback(self, event: int, x: int, y: int, _flags: int, _) -> None:
        """Handle mouse clicks

        Switch the selected track to accepted or rejected.

        Args:
            event (int): Opencv event type
            x (int), y (int): position of the click
            _flags (int): Opencv modifiers
            _ (Any): Additional data given by opencv
        """
        if event != cv2.EVENT_LBUTTONDBLCLK:
            return

        # Find the closest track if any
        dists = (self._tracks_tensor[self._frame_id] - torch.tensor([[y, x]])).pow(2).sum(dim=1).sqrt()
        dists[torch.isnan(dists)] = torch.inf
        dists[dists > 5] = torch.inf
        target_id = int(dists.argmin())

        if dists[target_id] == torch.inf:  # No track close enough
            return

        # Switch track state
        self.is_valid[target_id] = 1 - self.is_valid[target_id]
        print(f"Manual switch of track {self.tracks[target_id].identifier}")
