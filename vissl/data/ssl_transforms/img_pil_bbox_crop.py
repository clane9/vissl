# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from PIL import Image


@register_transform("ImgPilBboxCrop")
class ImgPilBboxCrop(ClassyTransform):
    """
    Custom center cropping transform used on openimages data. For tall images,
    crops from top down. For wide images, crops to center. Otherwise, leaves as is.
    """

    def __call__(self, img: Image.Image):
        width, height = img.size
        bbox_size = min(img.size)
        ratio = max(img.size) / min(img.size)
        if ratio >= 1.2:
            if width < height:  # bigger height
                bbox = (0, 0, bbox_size, bbox_size)
            else:  # bigger width.
                bbox = (
                    int((width - bbox_size) / 2),
                    0,
                    int((width - bbox_size) / 2) + bbox_size,
                    height,
                )
            img = img.crop(bbox)
        return img
