# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from fractions import Fraction
from typing import Any, Dict

import torch
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from vissl.utils.misc import is_pytorchvideo_available


if is_pytorchvideo_available():
    from pytorchvideo.data.video import Video
else:
    Video = Any


@register_transform("VideoPtvRandomFrame")
class VideoPtvRandomFrame(ClassyTransform):
    """
    Sample a random frame from a pytorchvideo Video and return a PIL image.
    """

    def __init__(self, fps: float):
        """
        Args:
            fps (float): video frames per second
        """
        self.fps = Fraction.from_float(fps)

    def __call__(self, vid: Video) -> Image.Image:
        duration = vid.duration
        num_frames = math.ceil(duration * self.fps)
        frame_idx = int(torch.randint(0, num_frames, (1,)).item())
        start_sec = frame_idx / self.fps
        end_sec = (frame_idx + 1) / self.fps
        clip = vid.get_clip(start_sec, end_sec)
        # tensor of shape CTHW with values in [0, 255] of type float32
        frames = clip["video"]
        if frames is None or frames.size(1) == 0:
            raise RuntimeError("No frames present in the video clip")

        img = to_pil_image(frames[:, 0, ...].to(torch.uint8))
        return img

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "VideoPtvRandomFrame":
        """
        Instantiates VideoPtvRandomFrame from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            VideoPtvRandomFrame instance.
        """
        fps = config.get("fps", 30)
        logging.info(f"VideoPtvRandomFrame | Using fps: {fps}")
        return cls(fps=fps)
