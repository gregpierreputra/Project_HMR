from pathlib import Path
from typing import List, Optional

import pytorch_lightning as pl
from webdataset.compat import WebDataset
from torch.utils.data import DataLoader

from hmr.datasets.motion_capture_dataset import MotionCaptureDataset
from hmr.datasets.webdataset import load_tars_as_webdataset

BASE_DATA_PATH = Path("/opt/ml/data/MDN")

IMAGE_DATASETS = {
    "COCO-TRAIN-2014-VITPOSE-REPLICATE-PRUNED12": {
        "urls": str(
            BASE_DATA_PATH
            / "hmr2_training_data/dataset_tars/coco-train-2014-vitpose-pruned/{000000..000044}.tar"
        ),
        "epoch_size": 45_000,
    },
    "COCO-TRAIN-2014-WMASK-PRUNED": {
        "urls": str(
            BASE_DATA_PATH
            / "hmr2_training_data/dataset_tars/coco-train-2014-pruned/{000000..000017}.tar"
        ),
        "epoch_size": 18_000,
    },
    "COCO-VAL": {
        "urls": str(
            BASE_DATA_PATH
            / "hmr2_training_data/dataset_tars/coco-val/{000000..000000}.tar"
        ),
        "epoch_size": None,
    },
}

MOCAP_DATASET_FILE = str(BASE_DATA_PATH / "cmu_mocap.npz")

DEFAULT_AUG_PARAMS = {
    "SCALE_FACTOR": 0.3,
    "ROT_FACTOR": 30,
    "TRANS_FACTOR": 0.02,
    "COLOR_SCALE": 0.2,
    "ROT_AUG_RATE": 0.6,
    "DO_FLIP": True,
    "FLIP_AUG_RATE": 0.5,
    "EXTREME_CROP_AUG_RATE": 0.10,
    "EXTREME_CROP_AUG_LEVEL": 1,
}


class HMRDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dataset_name: str,
        val_dataset_name: str,
        mocap_datafile_path: str,
        amass_poses_hist100_path: str,
        batch_size: int = 10,
        num_workers: int = 2,
        prefetch_factor: int = 2,
        mocap_num_train_samples: int = 10,
    ) -> None:
        super().__init__()
        self.train_dataset_name = train_dataset_name
        self.val_dataset_name = val_dataset_name
        self.mocap_datafile_path = mocap_datafile_path
        self.amass_poses_hist100_path = amass_poses_hist100_path

        # Initialize dataset variables
        self.training_dataset = None
        self.validation_dataset = None
        self.test_dataset = None
        self.mocap_dataset = None

        # Initialize general class variables
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.mocap_batch_size = mocap_num_train_samples * batch_size

        self._dataset_initialized = False

    def __str__(self):
        if self._dataset_initialized:
            return "Successfully loaded all of the data!"

    def setup(self, *args, **kwargs) -> None:
        """
        Load the necessary training data using the WebLoader

        Training dataset will utilise the Ava Midframes training set
        Validation dataset will utilise the COCO validation set
        """
        if self.training_dataset is None:
            train_dataset_conf = IMAGE_DATASETS[self.train_dataset_name]
            self.training_dataset: WebDataset = load_tars_as_webdataset(
                urls=train_dataset_conf["urls"],
                train=True,
                amass_poses_hist100_path=self.amass_poses_hist100_path,
                epoch_size=train_dataset_conf["epoch_size"] // self.num_workers,
                shuffle_size=4000,
                **DEFAULT_AUG_PARAMS,
            )

            self.mocap_dataset = MotionCaptureDataset(self.mocap_datafile_path)

            val_dataset_conf = IMAGE_DATASETS[self.val_dataset_name]
            self.validation_dataset: WebDataset = load_tars_as_webdataset(
                urls=val_dataset_conf["urls"],
                train=False,
                amass_poses_hist100_path=self.amass_poses_hist100_path,
                epoch_size=None,
                shuffle_size=0,
                resampled=False,
            )

    def train_dataloader(self) -> DataLoader:
        """
        Setup the training DataLoader for both images and motion capture

        Returns:
            training_dataloaders (dict) : Dictionary containing the image and motion capture dataloaders. Keys are appropriately 'img' and 'mocap'
        """
        train_dataloader = DataLoader(
            self.training_dataset,
            self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
        )
        motion_capture_dataloader = DataLoader(
            self.mocap_dataset,
            self.mocap_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=1,
        )

        training_dataloaders = {
            "img": train_dataloader,
            "mocap": motion_capture_dataloader,
        }

        return training_dataloaders

    def val_dataloader(self) -> DataLoader:
        """
        Setup the validation DataLoader

        Returns:
            validation_dataloader (torch.utils.data.DataLoader): Validation dataset as a dataloader
        """
        validation_dataloader = DataLoader(
            self.validation_dataset,
            self.batch_size,
            drop_last=False,
            num_workers=1,  # there might be dataset that only has one shard/tarfile, so adding more workers will throw an error
        )

        return validation_dataloader
