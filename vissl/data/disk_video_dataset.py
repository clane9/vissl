# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from iopath.common.file_io import g_pathmgr
from vissl.data.data_helper import QueueDataset
from vissl.utils.io import load_file
from vissl.utils.misc import is_pytorchvideo_available


if is_pytorchvideo_available():
    from pytorchvideo.data.video import VideoPathHandler
    from vissl.data.video_folder import VideoFolder
    from vissl.data.video_helper import get_mean_video


class DiskVideoDataset(QueueDataset):
    """
    Base Dataset class for loading videos from Disk.
    Can load a predefined list of videos, or all videos inside a folder.
    Videos can be either encoded video files or frame directories.

    Inherits from QueueDataset class in VISSL to provide better
    handling of the invalid videos by replacing them with the
    valid and seen videos.

    Args:
        cfg (AttrDict): configuration defined by user
        data_source (string): data source either of "disk_video_filelist" or "disk_video_folder"
        path (string): can be either of the following
            1. A .npy file containing a list of video filepaths or frame directories.
               In this case `data_source = "disk_video_filelist"`
            2. A folder such that folder/split contains video filepaths or frame directories.
               In this case `data_source = "disk_video_folder"`
        split (string): specify split for the dataset.
                        Usually train/val/test.
                        Used to read videos if reading from a folder `path` and retrieve
                        settings for that split from the config path.
        dataset_name (string): name of dataset. For information only.

    NOTE: This dataset class only returns videos (not labels or other metdata).
    To load labels you must specify them in `LABEL_SOURCES` (See `ssl_dataset.py`).
    LABEL_SOURCES follows a similar convention as the dataset and can either be a filelist
    or a torchvision ImageFolder compatible folder -
    1. Store labels in a numpy file
    2. Store images in a nested directory structure so that torchvision ImageFolder
       dataset can infer the labels.

    TODO: Would it be better to add video cases to disk_dataset rather than a
    new dataset class? Maybe they'll diverge in the future. Maybe it's nice to
    separate the video stuff from the more important image stuff.
    """

    def __init__(self, cfg, data_source, path, split, dataset_name):
        assert (
            is_pytorchvideo_available()
        ), "pytorchvideo must be available to use DiskVideoDataset"
        super(DiskVideoDataset, self).__init__(
            queue_size=cfg["DATA"][split]["BATCHSIZE_PER_REPLICA"]
        )
        assert data_source in [
            "disk_video_filelist",
            "disk_video_folder",
        ], "data_source must be either disk_video_filelist or disk_video_folder"
        if data_source == "disk_video_filelist":
            assert g_pathmgr.isfile(path), f"File {path} does not exist"
        elif data_source == "disk_video_folder":
            assert g_pathmgr.isdir(path), f"Directory {path} does not exist"
        self.cfg = cfg
        self.split = split
        self.dataset_name = dataset_name
        self.data_source = data_source
        self._path = path
        self.video_dataset = []
        self.is_initialized = False
        self._video_path_handler = VideoPathHandler()
        self._load_data(path)
        self._num_samples = len(self.video_dataset)
        self._remove_prefix = cfg["DATA"][self.split]["REMOVE_IMG_PATH_PREFIX"]
        self._new_prefix = cfg["DATA"][self.split]["NEW_IMG_PATH_PREFIX"]
        if self.data_source in ["disk_video_filelist"]:
            # Set dataset to null so that workers dont need to pickle this file.
            # This saves memory when disk_filelist is large, especially when memory mapping.
            self.video_dataset = []
        # whether to use QueueDataset class to handle invalid images or not
        self.enable_queue_dataset = cfg["DATA"][self.split]["ENABLE_QUEUE_DATASET"]
        # TODO: Possibly add support for replacing invalid videos from queue.
        if self.enable_queue_dataset:
            logging.warning(
                "ENABLE_QUEUE_DATASET not supported for DiskVideoDataset; disabling."
            )
            self.enable_queue_dataset = False

    def _load_data(self, path):
        if self.data_source == "disk_video_filelist":
            if self.cfg["DATA"][self.split].MMAP_MODE:
                self.video_dataset = load_file(path, mmap_mode="r")
            else:
                self.video_dataset = load_file(path)
        elif self.data_source == "disk_video_folder":
            self.video_dataset = VideoFolder(path)
            logging.info(f"Loaded {len(self.video_dataset)} samples from folder {path}")

            # mark as initialized.
            # Creating VideoFolder dataset can be expensive because of repeated os.listdir calls
            # Avoid creating it over and over again.
            self.is_initialized = True
        # alias for consistent access
        self.image_dataset = self.video_dataset

    def num_samples(self):
        """
        Size of the dataset
        """
        return self._num_samples

    def get_image_paths(self):
        """
        Get paths of all images in the datasets. See load_data()
        """
        self._load_data(self._path)
        if self.data_source == "disk_video_folder":
            assert isinstance(self.video_dataset, VideoFolder)
            return [sample[0] for sample in self.video_dataset.samples]
        else:
            return self.video_dataset

    @staticmethod
    def _replace_img_path_prefix(img_path: str, replace_prefix: str, new_prefix: str):
        if img_path.startswith(replace_prefix):
            return img_path.replace(replace_prefix, new_prefix)
        return img_path

    def __len__(self):
        """
        Size of the dataset
        """
        return self.num_samples()

    def __getitem__(self, idx):
        """
        - We do delayed loading of data to reduce the memory size due to pickling of
          dataset across dataloader workers.
        - Loads the data if not already loaded.
        - Sets and initializes the queue if not already initialized
        - Depending on the data source (folder or filelist), get the image.
          If using the QueueDataset and image is valid, save the image in queue if
          not full. Otherwise return a valid seen image from the queue if queue is
          not empty.
        """
        if not self.is_initialized:
            self._load_data(self._path)
            self.is_initialized = True
        if not self.queue_init and self.enable_queue_dataset:
            self._init_queues()
        is_success = True
        try:
            if self.data_source == "disk_video_filelist":
                video_path = self.video_dataset[idx]
                video_path = self._replace_img_path_prefix(
                    video_path,
                    replace_prefix=self._remove_prefix,
                    new_prefix=self._new_prefix,
                )
                # TODO: these defaults appear in a few places
                video = self._video_path_handler.video_from_path(
                    video_path, decode_audio=False, decoder="pyav", fps=30
                )
            elif self.data_source == "disk_video_folder":
                video_path = self.video_dataset.samples[idx][0]
                video = self.video_dataset[idx][0]
            if is_success and self.enable_queue_dataset:
                self.on_sucess(video)
        except Exception as e:
            logging.warning(f"Couldn't load: {video_path}. Exception: \n{e}")
            is_success = False
            # if we have queue dataset class enabled, we try to use it to get
            # the seen valid images
            if self.enable_queue_dataset:
                video, is_success = self.on_failure()
                if video is None:
                    video = get_mean_video(
                        self.cfg["DATA"][self.split].DEFAULT_GRAY_IMG_SIZE, fps=30
                    )
            else:
                video = get_mean_video(
                    self.cfg["DATA"][self.split].DEFAULT_GRAY_IMG_SIZE, fps=30
                )
        return video, is_success
