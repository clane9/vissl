# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Dict
import math

import torch
from pytorchvideo.data.video import Video


def get_mean_video(crop_size, fps=30.0):
    """
    Helper function that returns a gray pytorchvideo Video of the specified size
    and frame rate.

    Args:
        crop_size (int): used to generate (crop_size x crop_size x 3) frames.
        fps (float): frames per second

    Returns:
        vid: torchvideo Video
    """
    return GrayVideo(crop_size, crop_size, fps)


class GrayVideo(Video):
    """
    A dummy Video class for generating all gray clips.

    Args:
        width (int): frame width
        heigh (int): frame height
        fps (float): frames per second
    """
    def __init__(self,
        width: int,
        height: int,
        fps: float
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = math.inf

    def get_clip(
        self, start_sec: float, end_sec: float
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Generates gray frames for the duration between the start and end times.

        Args:
            start_sec (float): the clip start time in seconds
            end_sec (float): the clip end time in seconds

        Returns:
            clip_data: A dictionary with keys "video" and "audio".
                "video": A tensor of the clip's RGB frames with shape:
                    (channel, time, height, width). The frames are of type
                    torch.float32 and in the range [0 - 255]. Or None if no
                    frames found.
                "audio": None
        """
        start_sec = max(0.0, start_sec)
        start_idx = math.ceil(start_sec * self.fps)
        end_idx = math.ceil(end_sec * self.fps)
        num_frames = end_idx - start_idx
        if num_frames <= 0:
            return {"video": None, "audio": None}

        video = torch.full((3, num_frames, self.height, self.width), 128.0,
            dtype=torch.float32)
        return {"video": video, "audio": None}
