# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

from iopath.common.file_io import g_pathmgr
from torchvision.datasets import DatasetFolder, ImageFolder
from vissl.data.data_helper import QueueDataset, get_mean_image, image_loader
from vissl.utils.io import load_file
from vissl.utils.misc import is_pytorchvideo_available


if is_pytorchvideo_available():
    from vissl.data.video_folder import VideoFolder
    from vissl.data.video_helper import get_mean_video, video_loader


class DiskImageDataset(QueueDataset):
    """
    Base Dataset class for loading images or videos from Disk.
    Can load a predefined list of images or all images inside
    a folder. And also a json file containing the image path
    and the RoI annotation for the image.

    Inherits from QueueDataset class in VISSL to provide better
    handling of the invalid images by replacing them with the
    valid and seen images.

    Args:
        cfg (AttrDict): configuration defined by user
        data_source (string): data source either of "disk_filelist",
            "disk_folder", "disk_roi_annotations", "disk_video_filelist", or
            "disk_video_folder"
        path (string): can be either of the following
            1. A .npy file containing a list of filepaths.
               In this case `data_source in ["disk_filelist", "disk_video_filelist"]`
            2. A folder such that folder/split contains images.
               In this case `data_source in ["disk_folder", "disk_video_folder"]`
            3. A .json file containing list of dictionary where
               each dictionary has "img_path" and "bbox" entry.
               In this case `data_source = "disk_roi_annotations"`
        split (string): specify split for the dataset.
                        Usually train/val/test.
                        Used to read images if reading from a folder `path` and retrieve
                        settings for that split from the config path.
        dataset_name (string): name of dataset. For information only.

    NOTE: This dataset class only returns images (not labels or other metdata).
    To load labels you must specify them in `LABEL_SOURCES` (See `ssl_dataset.py`).
    LABEL_SOURCES follows a similar convention as the dataset and can either be a filelist
    or a torchvision ImageFolder compatible folder -
    1. Store labels in a numpy file
    2. Store images in a nested directory structure so that torchvision ImageFolder
       dataset can infer the labels.
    """

    def __init__(self, cfg, data_source, path, split, dataset_name):
        super(DiskImageDataset, self).__init__(
            queue_size=cfg["DATA"][split]["BATCHSIZE_PER_REPLICA"]
        )
        assert data_source in [
            "disk_filelist",
            "disk_folder",
            "disk_roi_annotations",
            "disk_video_filelist",
            "disk_video_folder",
        ], (
            "data_source must be either disk_filelist, disk_folder, "
            "disk_roi_annotations, disk_video_filelist, or disk_video_folder"
        )
        if data_source in ["disk_filelist", "disk_video_filelist"]:
            assert g_pathmgr.isfile(path), f"File {path} does not exist"
        elif data_source in ["disk_folder", "disk_video_folder"]:
            assert g_pathmgr.isdir(path), f"Directory {path} does not exist"
        elif data_source == "disk_roi_annotations":
            assert g_pathmgr.isfile(path), f"File {path} does not exist"
            assert path.endswith("json"), "Annotations must be in json format"
        if data_source in ["disk_video_filelist", "disk_video_folder"]:
            assert (
                is_pytorchvideo_available()
            ), "pytorchvideo required for video data sources"
        self.cfg = cfg
        self.split = split
        self.dataset_name = dataset_name
        self.data_source = data_source
        self._path = path
        self.image_dataset = []
        self.image_roi_bbox = []
        self.is_initialized = False
        self._load_data(path)
        self._num_samples = len(self.image_dataset)
        self._remove_prefix = cfg["DATA"][self.split]["REMOVE_IMG_PATH_PREFIX"]
        self._new_prefix = cfg["DATA"][self.split]["NEW_IMG_PATH_PREFIX"]
        if self.data_source in [
            "disk_filelist",
            "disk_video_filelist",
            "disk_roi_annotations",
        ]:
            # Set dataset to null so that workers dont need to pickle this file.
            # This saves memory when disk_filelist is large, especially when memory mapping.
            self.image_dataset = []
            self.image_roi_bbox = []
        if data_source in ["disk_video_filelist", "disk_video_folder"]:
            self._load_image = video_loader
            self._get_mean_image = get_mean_video
        else:
            self._load_image = image_loader
            self._get_mean_image = get_mean_image
        # whether to use QueueDataset class to handle invalid images or not
        self.enable_queue_dataset = cfg["DATA"][self.split]["ENABLE_QUEUE_DATASET"]
        # TODO: Possibly add support for replacing invalid videos from queue.
        if data_source in ["disk_video_filelist", "disk_video_folder"]:
            logging.warning(
                "ENABLE_QUEUE_DATASET not supported for video data sources; disabling."
            )
            self.enable_queue_dataset = False

    def _load_data(self, path):
        if self.data_source in ["disk_filelist", "disk_video_filelist"]:
            if self.cfg["DATA"][self.split].MMAP_MODE:
                self.image_dataset = load_file(path, mmap_mode="r")
            else:
                self.image_dataset = load_file(path)
        elif self.data_source in ["disk_folder", "disk_video_folder"]:
            if self.data_source == "disk_folder":
                self.image_dataset = ImageFolder(path)
            else:
                self.image_dataset = VideoFolder(path)
            logging.info(f"Loaded {len(self.image_dataset)} samples from folder {path}")

            # mark as initialized.
            # Creating ImageFolder dataset can be expensive because of repeated os.listdir calls
            # Avoid creating it over and over again.
            self.is_initialized = True
        elif self.data_source == "disk_roi_annotations":
            # we load the annotations and then parse the image paths and the image roi
            self.image_dataset, self.image_roi_bbox = [], []
            json_annotations = load_file(path)
            self.image_dataset = [item["path"] for item in json_annotations]
            self.image_roi_bbox = [item["bbox"] for item in json_annotations]

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
        if self.data_source in ["disk_folder", "disk_video_folder"]:
            assert isinstance(self.image_dataset, DatasetFolder)
            return [sample[0] for sample in self.image_dataset.samples]
        else:
            return self.image_dataset

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
            if self.data_source in ["disk_filelist", "disk_video_filelist"]:
                image_path = self.image_dataset[idx]
                image_path = self._replace_img_path_prefix(
                    image_path,
                    replace_prefix=self._remove_prefix,
                    new_prefix=self._new_prefix,
                )
                img = self._load_image(image_path)
            elif self.data_source in ["disk_folder", "disk_video_folder"]:
                image_path = self.image_dataset.samples[idx][0]
                img = self.image_dataset[idx][0]
            elif self.data_source == "disk_roi_annotations":
                image_path = self.image_dataset[idx]
                img = self._load_image(image_path)
                bbox = [float(item) for item in self.image_roi_bbox[idx]]
                img = img.crop(bbox)
            if is_success and self.enable_queue_dataset:
                self.on_sucess(img)
        except Exception as e:
            logging.warning(f"Couldn't load: {image_path}. Exception: \n{e}")
            is_success = False
            img = None
            # if we have queue dataset class enabled, we try to use it to get
            # the seen valid images
            if self.enable_queue_dataset:
                img, is_success = self.on_failure()
            if img is None:
                img = self._get_mean_image(
                    self.cfg["DATA"][self.split].DEFAULT_GRAY_IMG_SIZE
                )
        return img, is_success
