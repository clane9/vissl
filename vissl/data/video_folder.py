# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path
from typing import Any, Callable, List, Optional, Tuple

from torchvision.datasets.folder import (
    IMG_EXTENSIONS, DatasetFolder, has_file_allowed_extension
)
from pytorchvideo.data.video import VideoPathHandler, Video


VIDEO_EXTENSIONS = ('avi', 'mp4')


def video_loader(path: str, decode_audio=False, decoder="pyav", fps=30) -> Video:
    handler = VideoPathHandler()
    vid = handler.video_from_path(
        path, decode_audio=decode_audio, decoder=decoder, fps=fps)
    return vid


class VideoFolder(DatasetFolder):
    """A generic video data loader similar to :class:`~torchvision.datasets.ImageFolder`.

    Able to load videos represented as encoded files: ::

        root/dog/dog001.avi
        root/dog/dog002.avi

        root/cat/cat001.avi
        root/cat/cat002.avi

    or directories containing individual frames: ::

        root/dog/dog001/xxx.png
        root/dog/dog001/xxy.png
        root/dog/dog001/xxz.png

        root/cat/cat001/aaa.png
        root/cat/cat001/aab.png
        root/cat/cat001/aac.png

    This class inherits from :class:`~torchvision.datasets.DatasetFolder` so
    the same methods can be overridden to customize the dataset.

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in a
            pytorchvideo Video and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load a video given its path.
        is_valid_file (callable, optional): A function that takes path of a video file
            and check if the file is a valid file (used to check of corrupt files)

     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (video path, class_index) tuples

    NOTE: Why not use pytorchvideo LabeledVideoDataset? Bc iterable dataset.
    Also, `LabeledVideoPaths.from_directory` doesn't support frame directories,
    which I want.
    """
    def __init__(
            self,
            root: str,
            loader: Callable[[str], Any] = video_loader,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        super(VideoFolder, self).__init__(root, loader, IMG_EXTENSIONS + VIDEO_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)

        self.samples = self.merge_samples(self.samples)
        self.targets = [s[1] for s in self.samples]
        self._has_encoded_videos = has_file_allowed_extension(self.samples[0][0], VIDEO_EXTENSIONS)

    @staticmethod
    def merge_samples(samples: List[Tuple[str, int]]
    ) -> List[Tuple[str, int]]:
        """Merge samples of individual images into frame directories.

        Args:
            samples: list of (path, class_index) tuples. Paths may point to
                encoded video files or frame images.

        Returns:
            samples: list of (path, class_index) tuples. Paths point to encoded
                video files or directories containing frame images.

        Raises:
            ValueError: In case samples contains both encoded videos and frame images.
        """
        image_count = sum((has_file_allowed_extension(s[0], IMG_EXTENSIONS) for s in samples))
        if 0 < image_count < len(samples):
            raise ValueError(
                "Video folder shouldn't contain both videos and frame directories."
            )
        if image_count == 0:
            return samples

        # TODO: Better to keep directory name or list of frames? List of frames
        # might be faster for loading, since you avoid listing the directory.
        # But a little more work.
        merged_samples = []
        frame_dirs = set()
        for fname, target in samples:
            dirname = os.path.dirname(fname)
            if dirname not in frame_dirs:
                frame_dirs.add(dirname)
                merged_samples.append((dirname, target))
        return merged_samples
