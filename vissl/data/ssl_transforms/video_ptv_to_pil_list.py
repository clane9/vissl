# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, List

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


@register_transform("VideoPtvToPilList")
class VideoPtvToPilList(ClassyTransform):
    """
    Extract all frames from a video and return a list of PIL images.
    """

    def __call__(self, vid: Video) -> List[Image.Image]:
        clip = vid.get_clip(0.0, vid.duration + 1.0)
        # tensor of shape CTHW with values in [0, 255] of type float32
        frames = clip["video"]
        if frames is None or frames.size(1) == 0:
            raise RuntimeError("No frames present in the video clip")

        images = []
        for ii in range(frames.size(1)):
            img = to_pil_image(frames[:, ii, ...].to(torch.uint8))
            images.append(img)
        return images
